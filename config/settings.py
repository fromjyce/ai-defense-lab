from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"


class PathsConfig(BaseModel):
    model_config = {"protected_namespaces": ()}

    data_dir: str = "data"
    results_dir: str = "results"
    model_dir: str = "data/models"


class GeneratorConfig(BaseModel):
    n_rows: int = 20000
    fraud_prevalence: float = 0.005
    start_date: str = "2026-01-01"
    end_date: str = "2026-06-30"
    n_payers: int = 5000
    n_payees: int = 3000
    devices_per_payer: int = 2
    currencies: dict[str, float] = {"INR": 0.7, "USD": 0.2, "EUR": 0.1}
    mcc_codes: list[int] = [5411, 5812, 5732, 6011, 5999, 4111, 5941, 7995]
    channels: dict[str, float] = {
        "card_present": 0.25,
        "ecom": 0.35,
        "upi_p2p": 0.20,
        "upi_p2m": 0.15,
        "mandate": 0.05,
    }
    countries: list[str] = ["IN", "US", "GB", "AE", "SG", "CN", "NG", "RU"]
    domestic_country: str = "IN"
    fraud_amount_multiplier: float = 3.0
    fraud_new_payee_rate: float = 0.6
    legit_new_payee_rate: float = 0.05
    fraud_country_mismatch_rate: float = 0.4
    legit_country_mismatch_rate: float = 0.02
    fraud_velocity_multiplier: float = 4.0


class MultimodalConfig(BaseModel):
    """Beta-distribution params for the synthetic session-risk placeholder
    features (see generate/synth/multimodal.py) — NOT real audio/video
    features. alpha/beta pairs control the mean/skew of each score in
    [0, 1]; fraud rows are drawn from the "fraud" pair, legit rows from the
    "legit" pair, mirroring the fraud/legit split already used by
    GeneratorConfig.
    """

    fraud_liveness_alpha: float = 2.0
    fraud_liveness_beta: float = 5.0
    legit_liveness_alpha: float = 6.0
    legit_liveness_beta: float = 1.5
    fraud_similarity_alpha: float = 2.0
    fraud_similarity_beta: float = 4.0
    legit_similarity_alpha: float = 7.0
    legit_similarity_beta: float = 1.5


class LightGBMConfig(BaseModel):
    n_estimators: int = 200
    learning_rate: float = 0.05
    max_depth: int = -1
    num_leaves: int = 31
    min_child_samples: int = 20
    class_weight: str = "balanced"


class DetectorConfig(BaseModel):
    lightgbm: LightGBMConfig = LightGBMConfig()
    calibration_method: str = "isotonic"
    calibration_cv: int = 3
    test_size: float = 0.2


class AttackerConfig(BaseModel):
    population_size: int = 60
    n_generations: int = 15
    mutation_rate: float = 0.3
    crossover_rate: float = 0.5
    elitism_count: int = 4
    tournament_size: int = 3
    deployed_threshold: float = 0.5


class LoopConfig(BaseModel):
    n_generations: int = 5
    evasion_score_threshold: float = 0.5
    n_evasions_to_mine_per_generation: int = 30
    clean_holdout_fraction: float = 0.2


class MetricsConfig(BaseModel):
    fpr_targets: list[float] = [0.001, 0.01]
    precision_at_k: list[int] = [50, 100]
    latency_n_iterations: int = 200


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ADL_", extra="forbid")

    seed: int = 42
    paths: PathsConfig = PathsConfig()
    generator: GeneratorConfig = GeneratorConfig()
    multimodal: MultimodalConfig = MultimodalConfig()
    detector: DetectorConfig = DetectorConfig()
    attacker: AttackerConfig = AttackerConfig()
    loop: LoopConfig = LoopConfig()
    metrics: MetricsConfig = MetricsConfig()


def load_config(path: Optional[Path | str] = None) -> Settings:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return Settings(**raw)


def seed_everything(seed: int) -> None:
    """Thread one seed through every source of randomness in use.

    LightGBM draws its randomness from numpy/Python `random` plus its own
    `seed`/`random_state` params (set per-call at model construction time).
    torch is seeded here too, defensively, for when the multimodal streams
    stop being stubs — but it is not a hard dependency of this repo yet, so
    the import is optional.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
