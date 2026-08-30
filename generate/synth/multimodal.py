"""Synthetic multimodal session-risk features, keyed by txn_id.

Models the social-engineering-channel stream (audio-deepfake / face-liveness
signals) as two placeholder scores in [0, 1], drawn from label-conditional
beta distributions — the same "fraud rows look different from legit rows"
pattern generate/synth/generator.py already uses for transaction fields.

This is NOT real audio/video: no clip, frame, or model is involved. It
exists so defend/multimodal's late-fusion wiring is real and testable
end-to-end ahead of swapping in an actual ASVspoof/FaceForensics++-backed
scorer behind the same MultimodalScorer interface — see that module's
docstring. Do not read these scores as evidence of real-world detector
accuracy; they are synthetic by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import MultimodalConfig

MULTIMODAL_COLUMNS: tuple[str, ...] = ("txn_id", "voice_liveness_score", "face_similarity_score")


def generate_multimodal_features(transactions: pd.DataFrame, cfg: MultimodalConfig, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    is_fraud = transactions["label"].to_numpy() == 1
    n = len(transactions)

    voice = np.where(
        is_fraud,
        rng.beta(cfg.fraud_liveness_alpha, cfg.fraud_liveness_beta, size=n),
        rng.beta(cfg.legit_liveness_alpha, cfg.legit_liveness_beta, size=n),
    )
    face = np.where(
        is_fraud,
        rng.beta(cfg.fraud_similarity_alpha, cfg.fraud_similarity_beta, size=n),
        rng.beta(cfg.legit_similarity_alpha, cfg.legit_similarity_beta, size=n),
    )

    return pd.DataFrame(
        {
            "txn_id": transactions["txn_id"].to_numpy(),
            "voice_liveness_score": voice,
            "face_similarity_score": face,
        }
    )
