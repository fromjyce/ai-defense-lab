"""Closed-loop orchestration: attack -> mine evasions -> retrain -> re-evaluate.

Each loop generation:
  1. The evolutionary attacker evolves a fresh population against the
     current detector.
  2. The most evasive candidates (score below the deployed threshold) are
     mined and appended to the training pool.
  3. A new detector is trained from scratch on the enlarged pool.
  4. The new detector is scored against a clean, never-mined holdout set.

Step 4 is the guard against the loop's obvious failure mode: a detector
that "wins" by degrading into a blanket blocker would show attack success
rate collapsing while clean-set PR-AUC also collapses. We report both so
that failure mode is visible, not hidden by the headline attack-success
number.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import Settings, load_config
from defend.eval.metrics import evaluate
from defend.transaction.model import FraudDetector
from generate.attacker.evolutionary import EvolutionaryAttacker
from generate.synth.schema import COLUMNS

SEED_POOL_SIZE = 2000


@dataclass
class LoopGenerationRecord:
    loop_generation: int
    attack_success_rate_initial: float
    attack_success_rate_final: float
    mean_evasion_score_final: float
    max_evasion_score_final: float
    n_evaders_found: int
    n_evaders_mined: int
    train_pool_size: int
    clean_pr_auc: float
    clean_roc_auc: float
    clean_recall_at_fpr: dict[str, float]


def _reconstruct_train_pool(full_df: pd.DataFrame, holdout_df: pd.DataFrame) -> pd.DataFrame:
    holdout_ids = set(holdout_df["txn_id"])
    return full_df[~full_df["txn_id"].isin(holdout_ids)].reset_index(drop=True)


def run_loop(
    full_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    initial_detector: FraudDetector,
    settings: Settings,
) -> tuple[list[LoopGenerationRecord], FraudDetector]:
    rng = np.random.default_rng(settings.seed)
    train_pool = _reconstruct_train_pool(full_df, holdout_df)
    detector = initial_detector
    records: list[LoopGenerationRecord] = []

    for loop_gen in range(settings.loop.n_generations):
        seed_pool = train_pool.sample(
            n=min(SEED_POOL_SIZE, len(train_pool)),
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
        attacker = EvolutionaryAttacker(settings.attacker, settings.generator, seed=settings.seed + loop_gen)
        result = attacker.run(seed_pool, detector)

        evaders = result.evaders.sort_values("detector_score") if len(result.evaders) else result.evaders
        mined = evaders.head(settings.loop.n_evasions_to_mine_per_generation)

        if len(mined) > 0:
            train_pool = pd.concat([train_pool, mined[list(COLUMNS)]], ignore_index=True)

        new_detector = FraudDetector(settings.detector, settings.seed)
        new_detector.fit(train_pool)

        clean_metrics = evaluate(new_detector, holdout_df, settings.attacker.deployed_threshold, settings.metrics)

        first_gen, last_gen = result.generation_log[0], result.generation_log[-1]
        records.append(
            LoopGenerationRecord(
                loop_generation=loop_gen,
                attack_success_rate_initial=first_gen.attack_success_rate,
                attack_success_rate_final=last_gen.attack_success_rate,
                mean_evasion_score_final=last_gen.mean_evasion_score,
                max_evasion_score_final=last_gen.max_evasion_score,
                n_evaders_found=len(result.evaders),
                n_evaders_mined=len(mined),
                train_pool_size=len(train_pool),
                clean_pr_auc=clean_metrics.pr_auc,
                clean_roc_auc=clean_metrics.roc_auc,
                clean_recall_at_fpr=clean_metrics.recall_at_fpr,
            )
        )

        detector = new_detector

    return records, detector


def write_results(records: list[LoopGenerationRecord], results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "evasion_curve.json", "w") as f:
        json.dump([asdict(r) for r in records], f, indent=2)

    rows = []
    for r in records:
        row = asdict(r)
        row["clean_recall_at_fpr"] = json.dumps(row["clean_recall_at_fpr"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(results_dir / "loop_log.csv", index=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the closed-loop attacker/detector co-evolution.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    settings: Settings = load_config(args.config)

    full_df = pd.read_csv(args.data, parse_dates=["timestamp"])
    holdout_df = pd.read_csv(args.holdout, parse_dates=["timestamp"])
    initial_detector = FraudDetector.load(args.model)

    records, final_detector = run_loop(full_df, holdout_df, initial_detector, settings)
    write_results(records, args.results_dir)

    final_model_path = args.model.with_name(args.model.stem + "_retrained" + args.model.suffix)
    final_detector.save(final_model_path)

    print(f"loop complete: {len(records)} generations")
    for r in records:
        print(
            f"  gen {r.loop_generation}: attack success {r.attack_success_rate_initial:.2f} -> "
            f"{r.attack_success_rate_final:.2f} evolved | mined {r.n_evaders_mined}/{r.n_evaders_found} evaders | "
            f"clean PR-AUC {r.clean_pr_auc:.4f}"
        )
    print(f"wrote {args.results_dir / 'evasion_curve.json'} and {args.results_dir / 'loop_log.csv'}")
    print(f"final retrained detector saved to {final_model_path}")


if __name__ == "__main__":
    main()
