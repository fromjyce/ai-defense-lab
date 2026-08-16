"""Interface for the graph/mule-topology feature layer (fan-in/out, cycle,
scatter-gather patterns over the payer-payee money-flow graph), per the
team brief's C3 component.

Not implemented this pass. `StubGraphFeaturizer` returns an empty feature
frame so the transaction pipeline's feature set can be extended with graph
features later without changing any caller's contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class GraphFeaturizer(ABC):
    """Derives money-flow graph features for a batch of transactions.

    A real implementation would build the payer-payee graph (e.g. with
    NetworkX, per the brief's v1 plan) and compute per-transaction features
    like in-degree/out-degree of the payee, cycle membership, or
    scatter-gather pattern membership.
    """

    @abstractmethod
    def featurize(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class StubGraphFeaturizer(GraphFeaturizer):
    """Returns an empty feature frame, indexed to match the input.

    Placeholder only: lets the detector pipeline's feature set be extended
    with graph features later without changing the pipeline's shape today.
    """

    def featurize(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(index=df.index)
