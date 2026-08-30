"""Interface for the social-engineering-channel stream (audio-deepfake,
face-forgery, document-forgery scoring), fused with the transaction stream
by a late-fusion meta-classifier per the team brief.

`ConstantStubScorer` below is a placeholder. defend/multimodal/scorer.py's
`SyntheticFeatureScorer` is a real (fit/score) implementation of this
interface, trained on synthetic placeholder features (see
generate/synth/multimodal.py) rather than real audio/video evidence — a
real ASVspoof/FaceForensics++-backed scorer drops in later by implementing
`MultimodalScorer` against real evidence, without touching any caller.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class MultimodalScorer(ABC):
    """Scores the social-engineering channel for a batch of transactions.

    A real implementation would take channel-specific evidence (an audio
    clip, a video frame, a document image) keyed to each transaction and
    return a fraud-risk probability per row. This pass has no such evidence
    available, so the interface takes the transaction DataFrame itself —
    real implementations will need to extend the input contract once
    multimodal evidence is wired into the schema.
    """

    @abstractmethod
    def score(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class ConstantStubScorer(MultimodalScorer):
    """Returns a fixed low-risk score for every row.

    Placeholder only: lets the late-fusion path exist and be wired into the
    loop before a real audio/face-forgery model is trained or fitted.
    """

    def __init__(self, constant_score: float = 0.0) -> None:
        self.constant_score = constant_score

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.constant_score, dtype=float)
