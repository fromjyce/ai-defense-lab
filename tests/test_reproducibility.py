import pandas as pd
from sklearn.model_selection import train_test_split

from defend.transaction.model import FraudDetector
from generate.attacker.evolutionary import EvolutionaryAttacker
from generate.synth.generator import generate_synthetic


def test_generator_is_seed_reproducible(fast_settings) -> None:
    df1 = generate_synthetic(fast_settings.generator, fast_settings.seed)
    df2 = generate_synthetic(fast_settings.generator, fast_settings.seed)
    pd.testing.assert_frame_equal(df1, df2)


def test_detector_is_seed_reproducible(fast_settings, small_df: pd.DataFrame) -> None:
    train_df, holdout_df = train_test_split(
        small_df, test_size=fast_settings.detector.test_size, stratify=small_df["label"], random_state=fast_settings.seed
    )

    d1 = FraudDetector(fast_settings.detector, fast_settings.seed).fit(train_df.reset_index(drop=True))
    d2 = FraudDetector(fast_settings.detector, fast_settings.seed).fit(train_df.reset_index(drop=True))

    scores1 = d1.score(holdout_df)
    scores2 = d2.score(holdout_df)
    assert (scores1 == scores2).all()


def test_attacker_is_seed_reproducible(fast_settings, small_df: pd.DataFrame, trained_detector: FraudDetector) -> None:
    seed_pool = small_df.sample(n=100, random_state=fast_settings.seed)

    result1 = EvolutionaryAttacker(fast_settings.attacker, fast_settings.generator, fast_settings.seed).run(
        seed_pool, trained_detector
    )
    result2 = EvolutionaryAttacker(fast_settings.attacker, fast_settings.generator, fast_settings.seed).run(
        seed_pool, trained_detector
    )

    assert [g.__dict__ for g in result1.generation_log] == [g.__dict__ for g in result2.generation_log]
    pd.testing.assert_frame_equal(result1.final_population, result2.final_population)
