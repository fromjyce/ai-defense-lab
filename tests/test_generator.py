import pandas as pd

from generate.synth.schema import COLUMNS, validate_schema


def test_generated_schema_is_valid(small_df: pd.DataFrame, fast_settings) -> None:
    errors = validate_schema(small_df)
    assert errors == []


def test_generated_row_count_and_columns(small_df: pd.DataFrame, fast_settings) -> None:
    assert len(small_df) == fast_settings.generator.n_rows
    assert list(small_df.columns) == list(COLUMNS)
    assert small_df["txn_id"].is_unique


def test_generated_fraud_prevalence_matches_config(small_df: pd.DataFrame, fast_settings) -> None:
    n_fraud = round(fast_settings.generator.n_rows * fast_settings.generator.fraud_prevalence)
    assert small_df["label"].sum() == n_fraud


def test_unsupported_source_raises() -> None:
    import pytest

    from config.settings import GeneratorConfig
    from generate.synth.generator import generate

    with pytest.raises(NotImplementedError):
        generate("paysim", GeneratorConfig(), seed=42)
