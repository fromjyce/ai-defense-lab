"""Late-fusion meta-classifier combining the transaction-risk stream with
the multimodal (session-risk) stream into a single score, per the team
brief's recommended architecture: two independent per-stream scorers fused
by a small meta-classifier, not one joint model.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression


class LateFusionScorer:
    def __init__(self, seed: int) -> None:
        self.seed = seed
        self._model = LogisticRegression(random_state=seed)
        self._fitted = False

    def fit(self, transaction_scores: np.ndarray, multimodal_scores: np.ndarray, labels: np.ndarray) -> "LateFusionScorer":
        features = np.column_stack([transaction_scores, multimodal_scores])
        self._model.fit(features, labels)
        self._fitted = True
        return self

    def score(self, transaction_scores: np.ndarray, multimodal_scores: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("LateFusionScorer must be fit before scoring")
        features = np.column_stack([transaction_scores, multimodal_scores])
        return self._model.predict_proba(features)[:, 1]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "LateFusionScorer":
        return joblib.load(path)
