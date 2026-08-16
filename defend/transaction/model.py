"""Transaction-risk detector: sklearn Pipeline + LightGBM + isotonic calibration.

FraudDetector.score(df) is the interface the attacker, the loop, and the
eval harness all call against — it must accept a raw transaction DataFrame
(same schema as generate/synth/schema.COLUMNS, minus label) and return
calibrated fraud probabilities.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config.settings import DetectorConfig, Settings, load_config

CATEGORICAL_FEATURES = ["currency", "mcc", "channel", "auth_method"]
NUMERIC_FEATURES = [
    "amount",
    "is_new_payee",
    "payer_account_age_days",
    "velocity_1h",
    "velocity_24h",
    "hour_of_day",
    "day_of_week",
    "ip_merchant_mismatch",
    "issuer_merchant_mismatch",
    "issuer_ip_mismatch",
]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Derives model features from raw transaction fields.

    Drops high-cardinality identifiers (txn_id, payer_id, payee_id,
    device_id) and raw country codes in favor of mismatch flags, and
    expands the timestamp into hour-of-day / day-of-week.
    """

    def fit(self, X: pd.DataFrame, y=None):  # noqa: ARG002
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        timestamp = pd.to_datetime(X["timestamp"])
        out["hour_of_day"] = timestamp.dt.hour
        out["day_of_week"] = timestamp.dt.dayofweek
        out["ip_merchant_mismatch"] = (X["ip_country"] != X["merchant_country"]).astype(int)
        out["issuer_merchant_mismatch"] = (X["issuer_country"] != X["merchant_country"]).astype(int)
        out["issuer_ip_mismatch"] = (X["issuer_country"] != X["ip_country"]).astype(int)
        for col in ["amount", "currency", "mcc", "channel", "auth_method", "is_new_payee",
                    "payer_account_age_days", "velocity_1h", "velocity_24h"]:
            out[col] = X[col]
        return out[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


def _build_pipeline(cfg: DetectorConfig, seed: int) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )
    lgbm_params = cfg.lightgbm.model_dump()
    base_clf = LGBMClassifier(**lgbm_params, random_state=seed, verbosity=-1)
    calibrated_clf = CalibratedClassifierCV(
        estimator=base_clf, method=cfg.calibration_method, cv=cfg.calibration_cv
    )
    return Pipeline(
        [
            ("features", FeatureEngineer()),
            ("preprocess", preprocess),
            ("clf", calibrated_clf),
        ]
    )


class FraudDetector:
    def __init__(self, cfg: DetectorConfig, seed: int) -> None:
        self.cfg = cfg
        self.seed = seed
        self.pipeline: Pipeline = _build_pipeline(cfg, seed)

    def fit(self, df: pd.DataFrame) -> "FraudDetector":
        X = df.drop(columns=["label"])
        y = df["label"]
        self.pipeline.fit(X, y)
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        X = df.drop(columns=["label"]) if "label" in df.columns else df
        return self.pipeline.predict_proba(X)[:, 1]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "FraudDetector":
        return joblib.load(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the LightGBM transaction-risk detector.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--holdout-out", type=Path, required=True)
    return parser.parse_args()


# Invoked via `python -m defend.transaction` (see __main__.py), not this
# module directly — running this file itself as __main__ would make joblib
# pickle FeatureEngineer under the "__main__" module path, breaking load()
# from any other entrypoint.
def main() -> None:
    args = _parse_args()
    settings: Settings = load_config(args.config)

    df = pd.read_csv(args.data, parse_dates=["timestamp"])
    train_df, holdout_df = train_test_split(
        df,
        test_size=settings.detector.test_size,
        stratify=df["label"],
        random_state=settings.seed,
    )

    detector = FraudDetector(settings.detector, settings.seed)
    detector.fit(train_df.reset_index(drop=True))
    detector.save(args.model_out)

    args.holdout_out.parent.mkdir(parents=True, exist_ok=True)
    holdout_df.reset_index(drop=True).to_csv(args.holdout_out, index=False)

    scores = detector.score(holdout_df.reset_index(drop=True))
    print(
        f"trained on {len(train_df)} rows ({train_df['label'].sum()} fraud); "
        f"holdout {len(holdout_df)} rows ({holdout_df['label'].sum()} fraud); "
        f"holdout mean score {scores.mean():.4f}; model saved to {args.model_out}"
    )


