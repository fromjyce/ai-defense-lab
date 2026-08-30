"""Synthetic transaction generator.

Produces a pandas DataFrame of named, interpretable transaction fields (see
generate/synth/schema.py) — no PCA components. Fraud and legitimate rows are
drawn from separate parameterized distributions so that fraud carries a
plausible, configurable signal (higher amounts, more new payees, more
country mismatches, burstier velocity, odder hours) without hand-coding a
detector-specific tell.

This is a synthetic-only generator this pass. The --source flag exists so a
derived dataset (e.g. PaySim, fetched via scripts/download_paysim.sh) can be
substituted later without changing the rest of the pipeline's interface.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import GeneratorConfig, Settings, load_config
from generate.synth.schema import (
    ALLOWED_AUTH_METHODS_BY_CHANNEL,
    COLUMNS,
    DTYPES,
)


def _weighted_choice(rng: np.random.Generator, options: list[str], weights: dict[str, float], n: int) -> np.ndarray:
    probs = np.array([weights[o] for o in options], dtype=float)
    probs = probs / probs.sum()
    return rng.choice(options, size=n, p=probs)


def _make_population(cfg: GeneratorConfig, rng: np.random.Generator) -> tuple[list[str], np.ndarray, list[str], list[str]]:
    payer_ids = [f"payer_{i:06d}" for i in range(cfg.n_payers)]
    payer_account_age = rng.integers(1, 3651, size=cfg.n_payers)
    payee_ids = [f"payee_{i:06d}" for i in range(cfg.n_payees)]
    device_ids = [f"dev_{i:07d}" for i in range(cfg.n_payers * cfg.devices_per_payer)]
    return payer_ids, payer_account_age, payee_ids, device_ids


def _sample_rows(
    n: int,
    is_fraud: bool,
    cfg: GeneratorConfig,
    rng: np.random.Generator,
    payer_ids: list[str],
    payer_account_age: np.ndarray,
    payee_ids: list[str],
    device_ids: list[str],
    start_ts: float,
    end_ts: float,
) -> pd.DataFrame:
    if n == 0:
        return pd.DataFrame(columns=COLUMNS)

    # Fraud disproportionately touches newer accounts: weight payer choice
    # inversely by account age for fraud rows, uniformly for legit rows.
    if is_fraud:
        weights = 1.0 / (payer_account_age + 30.0)
        weights = weights / weights.sum()
        payer_idx = rng.choice(len(payer_ids), size=n, p=weights)
    else:
        payer_idx = rng.choice(len(payer_ids), size=n)
    payer_id = np.array(payer_ids)[payer_idx]
    account_age = payer_account_age[payer_idx]

    payee_id = rng.choice(payee_ids, size=n)
    device_id = rng.choice(device_ids, size=n)

    currency = _weighted_choice(rng, list(cfg.currencies.keys()), cfg.currencies, n)
    mcc = rng.choice(cfg.mcc_codes, size=n)
    channel = _weighted_choice(rng, list(cfg.channels.keys()), cfg.channels, n)
    auth_method = np.array(
        [rng.choice(ALLOWED_AUTH_METHODS_BY_CHANNEL[c]) for c in channel]
    )

    new_payee_rate = cfg.fraud_new_payee_rate if is_fraud else cfg.legit_new_payee_rate
    is_new_payee = rng.binomial(1, new_payee_rate, size=n)

    mismatch_rate = cfg.fraud_country_mismatch_rate if is_fraud else cfg.legit_country_mismatch_rate
    foreign_pool = [c for c in cfg.countries if c != cfg.domestic_country]

    def _country_with_mismatch() -> np.ndarray:
        mismatch = rng.binomial(1, mismatch_rate, size=n).astype(bool)
        out = np.full(n, cfg.domestic_country, dtype=object)
        n_mismatch = int(mismatch.sum())
        if n_mismatch > 0:
            out[mismatch] = rng.choice(foreign_pool, size=n_mismatch)
        return out

    issuer_country = np.full(n, cfg.domestic_country, dtype=object)
    ip_country = _country_with_mismatch()
    merchant_country = _country_with_mismatch()

    base_amount = rng.lognormal(mean=np.log(cfg.amount_lognormal_median), sigma=cfg.amount_lognormal_sigma, size=n)
    multiplier = cfg.fraud_amount_multiplier if is_fraud else 1.0
    amount = np.round(base_amount * multiplier, 2)

    lam_1h = 0.5 * (cfg.fraud_velocity_multiplier if is_fraud else 1.0)
    lam_extra = 2.0 * (cfg.fraud_velocity_multiplier if is_fraud else 1.0)
    velocity_1h = rng.poisson(lam=lam_1h, size=n)
    velocity_24h = velocity_1h + rng.poisson(lam=lam_extra, size=n)

    day_span = (end_ts - start_ts)
    day_offset = rng.uniform(0, day_span, size=n)
    night_bias = rng.binomial(1, 0.4 if is_fraud else 0.02, size=n).astype(bool)
    hour_uniform = rng.uniform(0, 24, size=n)
    hour_night = rng.uniform(0, 5, size=n)
    hour = np.where(night_bias, hour_night, hour_uniform)
    timestamps = pd.to_datetime(start_ts + day_offset, unit="s").normalize() + pd.to_timedelta(hour, unit="h")

    label = np.full(n, 1 if is_fraud else 0, dtype=int)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "amount": amount,
            "currency": currency,
            "mcc": mcc,
            "channel": channel,
            "payer_id": payer_id,
            "payee_id": payee_id,
            "device_id": device_id,
            "ip_country": ip_country,
            "issuer_country": issuer_country,
            "merchant_country": merchant_country,
            "auth_method": auth_method,
            "is_new_payee": is_new_payee,
            "payer_account_age_days": account_age,
            "velocity_1h": velocity_1h,
            "velocity_24h": velocity_24h,
            "label": label,
        }
    )


def generate_synthetic(cfg: GeneratorConfig, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Anchored to UTC explicitly: naive .timestamp() conversion is
    # local-timezone-dependent and would silently break cross-machine
    # reproducibility for the same seed.
    start_ts = datetime.fromisoformat(cfg.start_date).replace(tzinfo=timezone.utc).timestamp()
    end_ts = datetime.fromisoformat(cfg.end_date).replace(tzinfo=timezone.utc).timestamp()

    n_fraud = round(cfg.n_rows * cfg.fraud_prevalence)
    n_legit = cfg.n_rows - n_fraud

    payer_ids, payer_account_age, payee_ids, device_ids = _make_population(cfg, rng)

    legit_df = _sample_rows(
        n_legit, False, cfg, rng, payer_ids, payer_account_age, payee_ids, device_ids, start_ts, end_ts
    )
    fraud_df = _sample_rows(
        n_fraud, True, cfg, rng, payer_ids, payer_account_age, payee_ids, device_ids, start_ts, end_ts
    )

    df = pd.concat([legit_df, fraud_df], ignore_index=True)
    df = df.sample(frac=1.0, random_state=rng.integers(0, 2**31 - 1)).reset_index(drop=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.insert(0, "txn_id", [f"txn_{i:08d}" for i in range(len(df))])

    df = df[list(COLUMNS)]
    for col, dtype in DTYPES.items():
        if dtype != "datetime64[ns]":
            df[col] = df[col].astype(dtype)

    return df


def generate(source: str, cfg: GeneratorConfig, seed: int) -> pd.DataFrame:
    if source == "synthetic":
        return generate_synthetic(cfg, seed)
    if source == "paysim":
        raise NotImplementedError(
            "PaySim-derived generation is not implemented this pass. "
            "Run scripts/download_paysim.sh to fetch the dataset under your "
            "own license agreement, then wire a PaySim adapter here."
        )
    raise ValueError(f"unknown source: {source!r}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic payment transactions.")
    parser.add_argument("--source", choices=["synthetic", "paysim"], default="synthetic")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-rows", type=int, default=None, help="override generator.n_rows")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    settings: Settings = load_config(args.config)
    cfg = settings.generator
    if args.n_rows is not None:
        cfg = cfg.model_copy(update={"n_rows": args.n_rows})

    df = generate(args.source, cfg, settings.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} rows ({df['label'].sum()} fraud) to {args.out}")


if __name__ == "__main__":
    main()
