import numpy as np

from defend.eval.metrics import compute_metrics, precision_at_k, recall_at_fpr
from config.settings import MetricsConfig


def test_compute_metrics_in_range() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.binomial(1, 0.05, size=1000)
    scores = np.clip(y_true * 0.6 + rng.uniform(0, 0.4, size=1000), 0, 1)

    cfg = MetricsConfig(fpr_targets=[0.01, 0.1], precision_at_k=[10, 50])
    metrics = compute_metrics(y_true, scores, threshold=0.5, cfg=cfg)

    assert 0.0 <= metrics.pr_auc <= 1.0
    assert 0.0 <= metrics.roc_auc <= 1.0
    assert 0.0 <= metrics.f1 <= 1.0
    assert 0.0 <= metrics.brier_score <= 1.0
    assert metrics.n_samples == 1000
    assert metrics.n_positive == int(y_true.sum())
    for v in metrics.recall_at_fpr.values():
        assert 0.0 <= v <= 1.0
    for v in metrics.precision_at_k.values():
        assert 0.0 <= v <= 1.0


def test_recall_at_fpr_perfect_separation() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])
    assert recall_at_fpr(y_true, scores, fpr_target=0.0) == 1.0


def test_precision_at_k_caps_at_population_size() -> None:
    y_true = np.array([1, 0, 1])
    scores = np.array([0.9, 0.1, 0.8])
    assert precision_at_k(y_true, scores, k=100) == precision_at_k(y_true, scores, k=3)
