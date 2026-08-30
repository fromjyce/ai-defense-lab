import numpy as np
import pandas as pd

from defend.eval.fidelity import (
    compare_synthetic_runs,
    correlation_similarity,
    dcr_privacy_check,
    enforce_common_dtypes,
    marginal_similarity,
)


def test_enforce_common_dtypes_casts_to_float() -> None:
    a = pd.DataFrame({"x": [1, 2, 3]})  # int64
    b = pd.DataFrame({"x": [1.0, 2.0, 3.0]})  # float64
    a2, b2 = enforce_common_dtypes(a, b, ["x"])
    assert str(a2["x"].dtype) == "float64"
    assert str(b2["x"].dtype) == "float64"


def test_marginal_similarity_high_for_same_distribution() -> None:
    rng = np.random.default_rng(0)
    a = pd.DataFrame({"x": rng.normal(0, 1, size=2000)})
    b = pd.DataFrame({"x": rng.normal(0, 1, size=2000)})
    result = marginal_similarity(a, b, ["x"])
    assert result["x"] > 0.9


def test_marginal_similarity_low_for_different_distribution() -> None:
    rng = np.random.default_rng(0)
    a = pd.DataFrame({"x": rng.normal(0, 1, size=2000)})
    b = pd.DataFrame({"x": rng.normal(20, 1, size=2000)})
    result = marginal_similarity(a, b, ["x"])
    assert result["x"] < 0.2


def test_correlation_similarity_identical_frames_is_one() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=500), "y": rng.normal(size=500)})
    df["y"] = df["x"] * 2 + rng.normal(scale=0.1, size=500)
    assert correlation_similarity(df, df, ["x", "y"]) > 0.99


def test_dcr_privacy_check_returns_finite_positive_ratio() -> None:
    rng = np.random.default_rng(0)
    candidate = pd.DataFrame({"x": rng.normal(size=200), "y": rng.normal(size=200)})
    reference = pd.DataFrame({"x": rng.normal(size=200), "y": rng.normal(size=200)})
    ratio = dcr_privacy_check(candidate, reference, ["x", "y"], sample_size=200)
    assert ratio > 0
    assert np.isfinite(ratio)


def test_compare_synthetic_runs_produces_in_range_report(fast_settings) -> None:
    report = compare_synthetic_runs(fast_settings.generator, fast_settings.seed, fast_settings.seed + 1)
    assert 0.0 <= report.marginal_similarity_mean <= 1.0
    assert 0.0 <= report.correlation_similarity <= 1.0
    assert report.dcr_median_ratio > 0
    assert report.n_candidate_rows == fast_settings.generator.n_rows
