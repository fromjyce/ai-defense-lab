"""Evolutionary attacker: mutates transactions against the detector's own score.

Fitness = 1 - detector_score (higher is more evasive). Mutation and
crossover touch only ATTACKER_CONTROLLABLE_FIELDS (see
generate/synth/schema.py) — the attacker inherits every other field
(payer_id, timestamp, issuer_country, account age, velocity, ...) from a
seed transaction and cannot rewrite the issuer's own ground truth about
that account. Candidates that fail generate/attacker/validity are repaired
by re-mutating, falling back to the (guaranteed-valid) parent if repair
doesn't converge within a few attempts.

This is the v1 default per the team brief: scripted evolutionary/genetic
search, not RL. Fitness/selection/mutation are fully transparent, which
matters for explaining the evasion curve to judges.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config.settings import AttackerConfig, GeneratorConfig
from defend.transaction.model import FraudDetector
from generate.attacker.validity import is_valid
from generate.synth.schema import (
    ALLOWED_AUTH_METHODS_BY_CHANNEL,
    ATTACKER_CONTROLLABLE_FIELDS,
    COLUMNS,
)

_MAX_REPAIR_ATTEMPTS = 5


@dataclass
class GenerationStats:
    generation: int
    mean_evasion_score: float
    max_evasion_score: float
    mean_detector_score: float
    attack_success_rate: float
    n_valid_candidates: int


@dataclass
class AttackResult:
    generation_log: list[GenerationStats]
    final_population: pd.DataFrame
    evaders: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=COLUMNS))


def _randomize_controllable(candidate: dict, gen_cfg: GeneratorConfig, rng: np.random.Generator) -> dict:
    for f in ATTACKER_CONTROLLABLE_FIELDS:
        _mutate_field(candidate, f, gen_cfg, rng)
    return candidate


def _mutate_field(candidate: dict, f: str, gen_cfg: GeneratorConfig, rng: np.random.Generator) -> None:
    if f == "amount":
        perturb = rng.uniform(0.5, 2.0)
        candidate["amount"] = max(round(float(candidate["amount"]) * perturb, 2), 0.01)
    elif f == "currency":
        candidate["currency"] = rng.choice(list(gen_cfg.currencies.keys()))
    elif f == "mcc":
        candidate["mcc"] = int(rng.choice(gen_cfg.mcc_codes))
    elif f == "channel":
        new_channel = rng.choice(list(gen_cfg.channels.keys()))
        candidate["channel"] = new_channel
        candidate["auth_method"] = rng.choice(ALLOWED_AUTH_METHODS_BY_CHANNEL[new_channel])
    elif f == "auth_method":
        candidate["auth_method"] = rng.choice(ALLOWED_AUTH_METHODS_BY_CHANNEL[candidate["channel"]])
    elif f == "is_new_payee":
        candidate["is_new_payee"] = 1 - int(candidate["is_new_payee"])
    elif f == "device_id":
        candidate["device_id"] = f"dev_attacker_{int(rng.integers(0, 1_000_000)):07d}"
    elif f == "ip_country":
        candidate["ip_country"] = rng.choice(gen_cfg.countries)
    elif f == "merchant_country":
        candidate["merchant_country"] = rng.choice(gen_cfg.countries)
    else:
        raise ValueError(f"unknown attacker-controllable field: {f}")


def _mutate(candidate: dict, gen_cfg: GeneratorConfig, mutation_rate: float, rng: np.random.Generator) -> dict:
    child = dict(candidate)
    for f in ATTACKER_CONTROLLABLE_FIELDS:
        if rng.random() < mutation_rate:
            _mutate_field(child, f, gen_cfg, rng)
    return child


def _crossover(parent1: dict, parent2: dict, rng: np.random.Generator) -> tuple[dict, dict]:
    child1, child2 = dict(parent1), dict(parent2)
    for f in ATTACKER_CONTROLLABLE_FIELDS:
        if rng.random() < 0.5:
            child1[f], child2[f] = parent2[f], parent1[f]
    for child in (child1, child2):
        if child["auth_method"] not in ALLOWED_AUTH_METHODS_BY_CHANNEL[child["channel"]]:
            child["auth_method"] = rng.choice(ALLOWED_AUTH_METHODS_BY_CHANNEL[child["channel"]])
    return child1, child2


def _repair(candidate: dict, parent_fallback: dict, gen_cfg: GeneratorConfig, rng: np.random.Generator) -> dict:
    repaired = candidate
    for _ in range(_MAX_REPAIR_ATTEMPTS):
        if is_valid(repaired, gen_cfg):
            return repaired
        repaired = _mutate(repaired, gen_cfg, mutation_rate=1.0, rng=rng)
    return dict(parent_fallback) if is_valid(parent_fallback, gen_cfg) else repaired


def _tournament_select(fitness: np.ndarray, tournament_size: int, rng: np.random.Generator) -> int:
    contenders = rng.choice(len(fitness), size=min(tournament_size, len(fitness)), replace=False)
    return int(contenders[np.argmax(fitness[contenders])])


def _to_frame(population: list[dict], generation: int) -> pd.DataFrame:
    df = pd.DataFrame(population)
    df["txn_id"] = [f"atk_gen{generation:03d}_{i:05d}" for i in range(len(df))]
    return df[list(COLUMNS)]


class EvolutionaryAttacker:
    def __init__(self, cfg: AttackerConfig, gen_cfg: GeneratorConfig, seed: int) -> None:
        self.cfg = cfg
        self.gen_cfg = gen_cfg
        self.rng = np.random.default_rng(seed)

    def run(self, seed_pool: pd.DataFrame, detector: FraudDetector) -> AttackResult:
        cfg, gen_cfg, rng = self.cfg, self.gen_cfg, self.rng

        population: list[dict] = []
        for _ in range(cfg.population_size):
            base = seed_pool.iloc[int(rng.integers(0, len(seed_pool)))].to_dict()
            candidate = _randomize_controllable(dict(base), gen_cfg, rng)
            candidate["label"] = 1
            population.append(candidate)

        generation_log: list[GenerationStats] = []
        evader_frames: list[pd.DataFrame] = []

        for gen in range(cfg.n_generations):
            df = _to_frame(population, gen)
            detector_scores = detector.score(df)
            fitness = 1.0 - detector_scores

            attack_success_rate = float(np.mean(detector_scores < cfg.deployed_threshold))
            n_valid = sum(is_valid(c, gen_cfg) for c in population)
            generation_log.append(
                GenerationStats(
                    generation=gen,
                    mean_evasion_score=float(fitness.mean()),
                    max_evasion_score=float(fitness.max()),
                    mean_detector_score=float(detector_scores.mean()),
                    attack_success_rate=attack_success_rate,
                    n_valid_candidates=n_valid,
                )
            )

            evader_mask = detector_scores < cfg.deployed_threshold
            if evader_mask.any():
                evader_frames.append(df.loc[evader_mask])

            if gen == cfg.n_generations - 1:
                break  # no need to evolve past the last scored generation

            elite_count = min(cfg.elitism_count, cfg.population_size)
            elite_idx = np.argsort(-fitness)[:elite_count]
            new_population: list[dict] = [population[i] for i in elite_idx]

            while len(new_population) < cfg.population_size:
                p1_idx = _tournament_select(fitness, cfg.tournament_size, rng)
                p2_idx = _tournament_select(fitness, cfg.tournament_size, rng)
                parent1, parent2 = population[p1_idx], population[p2_idx]

                if rng.random() < cfg.crossover_rate:
                    child1, child2 = _crossover(parent1, parent2, rng)
                else:
                    child1, child2 = dict(parent1), dict(parent2)

                child1 = _mutate(child1, gen_cfg, cfg.mutation_rate, rng)
                child2 = _mutate(child2, gen_cfg, cfg.mutation_rate, rng)
                child1 = _repair(child1, parent1, gen_cfg, rng)
                child2 = _repair(child2, parent2, gen_cfg, rng)

                new_population.append(child1)
                if len(new_population) < cfg.population_size:
                    new_population.append(child2)

            population = new_population

        evaders = (
            pd.concat(evader_frames, ignore_index=True).drop_duplicates(subset=list(COLUMNS))
            if evader_frames
            else pd.DataFrame(columns=COLUMNS)
        )

        return AttackResult(generation_log=generation_log, final_population=df, evaders=evaders)
