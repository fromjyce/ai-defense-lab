from pathlib import Path

import numpy as np

from defend.transaction.model import FraudDetector


def test_detector_scores_are_valid_probabilities(trained_detector: FraudDetector, train_holdout_split) -> None:
    _, holdout_df = train_holdout_split
    scores = trained_detector.score(holdout_df)
    assert len(scores) == len(holdout_df)
    assert np.all((scores >= 0) & (scores <= 1))


def test_detector_discriminates_fraud_from_legit(trained_detector: FraudDetector, train_holdout_split) -> None:
    _, holdout_df = train_holdout_split
    scores = trained_detector.score(holdout_df)
    fraud_mean = scores[holdout_df["label"].to_numpy() == 1].mean()
    legit_mean = scores[holdout_df["label"].to_numpy() == 0].mean()
    assert fraud_mean > legit_mean


def test_detector_save_load_roundtrip(trained_detector: FraudDetector, train_holdout_split, tmp_path: Path) -> None:
    _, holdout_df = train_holdout_split
    scores_before = trained_detector.score(holdout_df)

    model_path = tmp_path / "detector.joblib"
    trained_detector.save(model_path)
    reloaded = FraudDetector.load(model_path)
    scores_after = reloaded.score(holdout_df)

    np.testing.assert_array_equal(scores_before, scores_after)
