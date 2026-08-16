"""Evaluation harness: detection-quality metrics + single-row latency.

PR-AUC is the primary metric (per the team brief: ROC-AUC is flattering
under the ~0.5% fraud prevalence used here). recall@fixed-FPR and
precision@k answer "is this usable by a real fraud desk" — an analyst can
only review a fixed number of alerts.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from config.settings import MetricsConfig, Settings, load_config
from defend.transaction.model import FraudDetector


@dataclass
class EvalMetrics:
    n_samples: int
    n_positive: int
    pr_auc: float
    roc_auc: float
    f1: float
    threshold: float
    brier_score: float
    recall_at_fpr: dict[str, float] = field(default_factory=dict)
    precision_at_k: dict[str, float] = field(default_factory=dict)
    latency_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def recall_at_fpr(y_true: np.ndarray, scores: np.ndarray, fpr_target: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    eligible = np.where(fpr <= fpr_target)[0]
    if len(eligible) == 0:
        return 0.0
    return float(tpr[eligible.max()])


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    k = min(k, len(scores))
    if k == 0:
        return 0.0
    top_k_idx = np.argsort(-scores)[:k]
    return float(np.mean(y_true[top_k_idx]))


def measure_latency(detector: FraudDetector, sample_row: pd.DataFrame, n_iterations: int) -> dict[str, float]:
    """p50/p95/p99 latency in ms for scoring a single transaction row."""
    assert len(sample_row) == 1, "measure_latency expects exactly one row"
    detector.score(sample_row)  # warm-up, excluded from measurement
    times_ms = np.empty(n_iterations)
    for i in range(n_iterations):
        start = time.perf_counter()
        detector.score(sample_row)
        times_ms[i] = (time.perf_counter() - start) * 1000
    return {
        "p50_ms": float(np.percentile(times_ms, 50)),
        "p95_ms": float(np.percentile(times_ms, 95)),
        "p99_ms": float(np.percentile(times_ms, 99)),
    }


def compute_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    cfg: MetricsConfig,
) -> EvalMetrics:
    y_pred = (scores >= threshold).astype(int)
    return EvalMetrics(
        n_samples=len(y_true),
        n_positive=int(y_true.sum()),
        pr_auc=float(average_precision_score(y_true, scores)),
        roc_auc=float(roc_auc_score(y_true, scores)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        threshold=threshold,
        brier_score=float(brier_score_loss(y_true, scores)),
        recall_at_fpr={str(t): recall_at_fpr(y_true, scores, t) for t in cfg.fpr_targets},
        precision_at_k={str(k): precision_at_k(y_true, scores, k) for k in cfg.precision_at_k},
    )


def evaluate(detector: FraudDetector, holdout_df: pd.DataFrame, threshold: float, cfg: MetricsConfig) -> EvalMetrics:
    y_true = holdout_df["label"].to_numpy()
    scores = detector.score(holdout_df)
    metrics = compute_metrics(y_true, scores, threshold, cfg)
    metrics.latency_ms = measure_latency(detector, holdout_df.iloc[[0]], cfg.latency_n_iterations)
    return metrics


def write_json(metrics: EvalMetrics, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics.to_dict(), f, indent=2)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the transaction detector on a held-out set.")
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    settings: Settings = load_config(args.config)

    detector = FraudDetector.load(args.model)
    holdout_df = pd.read_csv(args.holdout, parse_dates=["timestamp"])

    metrics = evaluate(detector, holdout_df, settings.attacker.deployed_threshold, settings.metrics)
    write_json(metrics, args.out)
    print(f"PR-AUC {metrics.pr_auc:.4f}  ROC-AUC {metrics.roc_auc:.4f}  F1 {metrics.f1:.4f}  "
          f"p50 {metrics.latency_ms['p50_ms']:.3f}ms  wrote {args.out}")


if __name__ == "__main__":
    main()
