import numpy as np
import pandas as pd

from defend.multimodal.fusion import LateFusionScorer
from defend.multimodal.scorer import SyntheticFeatureScorer
from generate.synth.multimodal import generate_multimodal_features


def test_multimodal_features_in_unit_range(fast_settings, small_df: pd.DataFrame) -> None:
    features = generate_multimodal_features(small_df, fast_settings.multimodal, fast_settings.seed)
    assert (features["voice_liveness_score"].between(0, 1)).all()
    assert (features["face_similarity_score"].between(0, 1)).all()
    assert list(features["txn_id"]) == list(small_df["txn_id"])


def test_synthetic_feature_scorer_discriminates(fast_settings, small_df: pd.DataFrame) -> None:
    features = generate_multimodal_features(small_df, fast_settings.multimodal, fast_settings.seed)
    labels = small_df["label"].to_numpy()

    scorer = SyntheticFeatureScorer(fast_settings.seed).fit(features, labels)
    scores = scorer.score(features)

    assert len(scores) == len(features)
    assert np.all((scores >= 0) & (scores <= 1))
    assert scores[labels == 1].mean() > scores[labels == 0].mean()


def test_late_fusion_scorer_combines_streams(fast_settings, small_df: pd.DataFrame, trained_detector) -> None:
    features = generate_multimodal_features(small_df, fast_settings.multimodal, fast_settings.seed)
    labels = small_df["label"].to_numpy()

    transaction_scores = trained_detector.score(small_df)
    mm_scorer = SyntheticFeatureScorer(fast_settings.seed).fit(features, labels)
    multimodal_scores = mm_scorer.score(features)

    fusion = LateFusionScorer(fast_settings.seed).fit(transaction_scores, multimodal_scores, labels)
    fused = fusion.score(transaction_scores, multimodal_scores)

    assert len(fused) == len(small_df)
    assert np.all((fused >= 0) & (fused <= 1))
