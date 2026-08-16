"""Validity constraints for attacker-crafted candidate transactions.

Rejects semantically impossible payments so the evolutionary search can
only "win" by finding transactions a real fraudster could plausibly
submit, not by exploiting gaps in the schema itself.
"""

from __future__ import annotations

from config.settings import GeneratorConfig
from generate.synth.schema import ALLOWED_AUTH_METHODS_BY_CHANNEL, AUTH_METHODS, CHANNELS


def validity_errors(candidate: dict, gen_cfg: GeneratorConfig) -> list[str]:
    errors: list[str] = []

    if candidate["amount"] <= 0:
        errors.append("amount must be positive")
    if candidate["currency"] not in gen_cfg.currencies:
        errors.append(f"currency {candidate['currency']!r} not in configured set")
    if candidate["mcc"] not in gen_cfg.mcc_codes:
        errors.append(f"mcc {candidate['mcc']!r} not in configured set")
    if candidate["channel"] not in CHANNELS:
        errors.append(f"channel {candidate['channel']!r} not a known channel")
    if candidate["auth_method"] not in AUTH_METHODS:
        errors.append(f"auth_method {candidate['auth_method']!r} not a known auth method")
    elif candidate["channel"] in CHANNELS and candidate["auth_method"] not in ALLOWED_AUTH_METHODS_BY_CHANNEL[candidate["channel"]]:
        errors.append(f"auth_method {candidate['auth_method']!r} inconsistent with channel {candidate['channel']!r}")
    if candidate["is_new_payee"] not in (0, 1):
        errors.append("is_new_payee must be 0 or 1")
    if candidate["ip_country"] not in gen_cfg.countries:
        errors.append(f"ip_country {candidate['ip_country']!r} not in configured set")
    if candidate["merchant_country"] not in gen_cfg.countries:
        errors.append(f"merchant_country {candidate['merchant_country']!r} not in configured set")
    if not candidate.get("device_id"):
        errors.append("device_id must be non-empty")

    # Issuer-side ground-truth fields are never mutated by the attacker, but
    # checked defensively in case a future mutation operator touches them.
    if candidate["velocity_1h"] > candidate["velocity_24h"]:
        errors.append("velocity_1h exceeds velocity_24h")
    if candidate["velocity_1h"] < 0 or candidate["velocity_24h"] < 0:
        errors.append("velocity fields must be non-negative")
    if candidate["payer_account_age_days"] < 0:
        errors.append("payer_account_age_days must be non-negative")

    return errors


def is_valid(candidate: dict, gen_cfg: GeneratorConfig) -> bool:
    return len(validity_errors(candidate, gen_cfg)) == 0
