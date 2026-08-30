import pandas as pd

from defend.graph.featurizer import GRAPH_FEATURE_COLUMNS, StructuralGraphFeaturizer


def test_structural_featurizer_produces_expected_columns(small_df: pd.DataFrame) -> None:
    features = StructuralGraphFeaturizer().featurize(small_df)
    assert list(features.columns) == list(GRAPH_FEATURE_COLUMNS)
    assert len(features) == len(small_df)
    assert (features >= 0).all().all()


def test_payer_txn_count_matches_actual_counts(small_df: pd.DataFrame) -> None:
    features = StructuralGraphFeaturizer().featurize(small_df)
    some_payer = small_df["payer_id"].iloc[0]
    expected = (small_df["payer_id"] == some_payer).sum()
    actual = features.loc[small_df["payer_id"] == some_payer, "payer_txn_count"].iloc[0]
    assert actual == expected
