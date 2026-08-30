"""Real (not stub) multimodal scorer, fit on the synthetic session-risk
placeholder features from generate/synth/multimodal.py.

This is the "feature-level stub upgrade": a genuine fit/score classifier so
the late-fusion path (see defend/multimodal/fusion.py) is real and testable
today. Swapping in an actual ASVspoof/FaceForensics++-backed scorer later
means implementing MultimodalScorer against real evidence — no caller
changes required.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from defend.multimodal.interface import MultimodalScorer

_FEATURES = ["voice_liveness_score", "face_similarity_score"]


class SyntheticFeatureScorer(MultimodalScorer):
    """Logistic regression over the synthetic voice/face placeholder scores."""

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self._model = LogisticRegression(random_state=seed)
        self._fitted = False

    def fit(self, multimodal_df: pd.DataFrame, labels: np.ndarray) -> "SyntheticFeatureScorer":
        self._model.fit(multimodal_df[_FEATURES], labels)
        self._fitted = True
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("SyntheticFeatureScorer must be fit before scoring")
        return self._model.predict_proba(df[_FEATURES])[:, 1]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "SyntheticFeatureScorer":
        return joblib.load(path)
