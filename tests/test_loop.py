import pandas as pd

from defend.transaction.model import FraudDetector
from loop.orchestrator import run_loop


def test_one_loop_generation_completes(fast_settings, small_df: pd.DataFrame, train_holdout_split, trained_detector: FraudDetector) -> None:
    fast_settings.loop.n_generations = 1
    train_df, holdout_df = train_holdout_split

    records, final_detector = run_loop(small_df, holdout_df, trained_detector, fast_settings)

    assert len(records) == 1
    record = records[0]
    assert record.loop_generation == 0
    assert 0.0 <= record.attack_success_rate_initial <= 1.0
    assert 0.0 <= record.attack_success_rate_final <= 1.0
    assert 0.0 <= record.clean_pr_auc <= 1.0
    assert 0.0 <= record.clean_roc_auc <= 1.0
    assert record.n_evaders_mined <= record.n_evaders_found
    assert record.train_pool_size >= len(train_df)
    assert isinstance(final_detector, FraudDetector)
