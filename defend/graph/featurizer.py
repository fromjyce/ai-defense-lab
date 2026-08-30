"""Lightweight graph-structural featurizer.

The team brief's v1 fallback for time-constrained teams is explicit: cut a
full GNN and keep structural features (fan-in/out, shared-device degree)
computed directly from the transaction table. That is what this module
does, in plain pandas — no NetworkX/GNN dependency, no new graph library.
It is a real, testable implementation of GraphFeaturizer, not a stub, but
it does not claim GNN-level lift (per the brief, ~10-15 AUC points on
YelpChi/Amazon-style benchmarks come from actual graph neural nets, not
degree counts).

Not wired into defend/transaction/model.py's training pipeline this pass —
integrating a new feature block into the already-tuned, already-validated
detector/attacker/loop is a separate design decision (retuning
mutation/validity/loop parameters against a changed feature set) left for
the next pass. This module is usable standalone today.
"""

from __future__ import annotations

import pandas as pd

from defend.graph.interface import GraphFeaturizer

GRAPH_FEATURE_COLUMNS: tuple[str, ...] = (
    "payer_txn_count",
    "payee_txn_count",
    "payer_unique_payees",
    "payee_unique_payers",
    "device_shared_payers",
    "payer_payee_pair_count",
)


class StructuralGraphFeaturizer(GraphFeaturizer):
    """Fan-in/fan-out and shared-device degree counts over payer/payee/device.

    payer_txn_count / payee_txn_count: how active this payer/payee is overall.
    payer_unique_payees / payee_unique_payers: fan-out / fan-in degree.
    device_shared_payers: how many distinct payers have used this device —
        a high count is a mule/shared-device signal (echoes MuleHunter.AI's
        framing, see identify/taxonomy.yaml S2-03).
    payer_payee_pair_count: repeat-relationship strength between this
        specific payer and payee.
    """

    def featurize(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        out["payer_txn_count"] = df.groupby("payer_id")["txn_id"].transform("count")
        out["payee_txn_count"] = df.groupby("payee_id")["txn_id"].transform("count")
        out["payer_unique_payees"] = df.groupby("payer_id")["payee_id"].transform("nunique")
        out["payee_unique_payers"] = df.groupby("payee_id")["payer_id"].transform("nunique")
        out["device_shared_payers"] = df.groupby("device_id")["payer_id"].transform("nunique")
        out["payer_payee_pair_count"] = df.groupby(["payer_id", "payee_id"])["txn_id"].transform("count")
        return out[list(GRAPH_FEATURE_COLUMNS)]
