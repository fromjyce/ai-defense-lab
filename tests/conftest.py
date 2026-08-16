"""Shared fixtures. Sizes are kept small so the smoke suite runs in seconds,
not to reflect realistic dataset scale."""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from config.settings import (
    AttackerConfig,
    DetectorConfig,
    GeneratorConfig,
    LightGBMConfig,
    LoopConfig,
    MetricsConfig,
    Settings,
)
from defend.transaction.model import FraudDetector
from generate.synth.generator import generate_synthetic


@pytest.fixture
def fast_settings() -> Settings:
    return Settings(
        seed=42,
        generator=GeneratorConfig(n_rows=3000, fraud_prevalence=0.02, n_payers=500, n_payees=300),
        detector=DetectorConfig(lightgbm=LightGBMConfig(n_estimators=50), calibration_cv=3, test_size=0.2),
        attacker=AttackerConfig(population_size=20, n_generations=3, elitism_count=2, tournament_size=3),
        loop=LoopConfig(n_generations=2, n_evasions_to_mine_per_generation=10),
        metrics=MetricsConfig(latency_n_iterations=20),
    )


@pytest.fixture
def small_df(fast_settings: Settings) -> pd.DataFrame:
    return generate_synthetic(fast_settings.generator, fast_settings.seed)


@pytest.fixture
def train_holdout_split(fast_settings: Settings, small_df: pd.DataFrame):
    return train_test_split(
        small_df,
        test_size=fast_settings.detector.test_size,
        stratify=small_df["label"],
        random_state=fast_settings.seed,
    )


@pytest.fixture
def trained_detector(fast_settings: Settings, train_holdout_split) -> FraudDetector:
    train_df, _ = train_holdout_split
    detector = FraudDetector(fast_settings.detector, fast_settings.seed)
    detector.fit(train_df.reset_index(drop=True))
    return detector
