"""Shared schema for synthetic transactions.

This is the single source of truth for column order, dtypes, and the fixed
categorical enums (channel, auth_method) that both the generator and the
evolutionary attacker's validity checker depend on. Config-driven value
sets (currencies, mcc codes, countries) live in config/default.yaml instead,
since those are tunable base rates rather than structural constraints.
"""

from __future__ import annotations

import pandas as pd

CHANNELS: tuple[str, ...] = ("card_present", "ecom", "upi_p2p", "upi_p2m", "mandate")

AUTH_METHODS: tuple[str, ...] = ("pin", "otp", "3ds", "biometric", "none")

# Which auth methods are physically/procedurally possible on each channel.
# A card_present swipe cannot carry an OTP; a UPI mandate execution has no
# interactive auth step at all, etc.
ALLOWED_AUTH_METHODS_BY_CHANNEL: dict[str, tuple[str, ...]] = {
    "card_present": ("pin", "biometric"),
    "ecom": ("otp", "3ds"),
    "upi_p2p": ("otp", "biometric", "none"),
    "upi_p2m": ("otp", "biometric", "none"),
    "mandate": ("none", "biometric"),
}

COLUMNS: tuple[str, ...] = (
    "txn_id",
    "timestamp",
    "amount",
    "currency",
    "mcc",
    "channel",
    "payer_id",
    "payee_id",
    "device_id",
    "ip_country",
    "issuer_country",
    "merchant_country",
    "auth_method",
    "is_new_payee",
    "payer_account_age_days",
    "velocity_1h",
    "velocity_24h",
    "label",
)

DTYPES: dict[str, str] = {
    "txn_id": "object",
    "timestamp": "datetime64[ns]",
    "amount": "float64",
    "currency": "object",
    "mcc": "int64",
    "channel": "object",
    "payer_id": "object",
    "payee_id": "object",
    "device_id": "object",
    "ip_country": "object",
    "issuer_country": "object",
    "merchant_country": "object",
    "auth_method": "object",
    "is_new_payee": "int64",
    "payer_account_age_days": "int64",
    "velocity_1h": "int64",
    "velocity_24h": "int64",
    "label": "int64",
}

# Fields the evolutionary attacker is allowed to mutate/crossover. Everything
# else is either an issuer-side ground-truth risk feature (velocity,
# account age, issuer_country) or a system-assigned identifier (txn_id,
# timestamp, payer_id, payee_id) that a fraudster submitting a transaction
# does not get to rewrite.
ATTACKER_CONTROLLABLE_FIELDS: tuple[str, ...] = (
    "amount",
    "currency",
    "mcc",
    "channel",
    "auth_method",
    "is_new_payee",
    "device_id",
    "ip_country",
    "merchant_country",
)


def validate_schema(df: pd.DataFrame) -> list[str]:
    """Return a list of human-readable schema violations (empty if valid)."""
    errors: list[str] = []

    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"missing columns: {missing}")
        return errors

    for col, dtype in DTYPES.items():
        actual = str(df[col].dtype)
        if dtype == "int64" and actual not in ("int64", "int32"):
            errors.append(f"{col}: expected integer dtype, got {actual}")
        elif dtype == "float64" and actual not in ("float64", "float32"):
            errors.append(f"{col}: expected float dtype, got {actual}")
        elif dtype == "object" and actual != "object":
            errors.append(f"{col}: expected object dtype, got {actual}")
        elif dtype == "datetime64[ns]" and not str(actual).startswith("datetime64"):
            errors.append(f"{col}: expected datetime dtype, got {actual}")

    if (df["amount"] <= 0).any():
        errors.append("amount: found non-positive values")
    if not df["channel"].isin(CHANNELS).all():
        errors.append("channel: found values outside CHANNELS")
    if not df["auth_method"].isin(AUTH_METHODS).all():
        errors.append("auth_method: found values outside AUTH_METHODS")
    if not df["is_new_payee"].isin([0, 1]).all():
        errors.append("is_new_payee: expected 0/1")
    if not df["label"].isin([0, 1]).all():
        errors.append("label: expected 0/1")
    if (df["velocity_1h"] > df["velocity_24h"]).any():
        errors.append("velocity_1h exceeds velocity_24h for some rows")
    if (df["payer_account_age_days"] < 0).any():
        errors.append("payer_account_age_days: found negative values")

    bad_auth = df[~df.apply(
        lambda r: r["auth_method"] in ALLOWED_AUTH_METHODS_BY_CHANNEL.get(r["channel"], ()),
        axis=1,
    )]
    if len(bad_auth) > 0:
        errors.append(f"auth_method inconsistent with channel for {len(bad_auth)} rows")

    return errors
