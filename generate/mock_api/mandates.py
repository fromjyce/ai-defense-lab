"""Minimal AP2-style mandate signing/verification for the mock payment API.

Real AP2 mandates are signed W3C Verifiable Credentials; that machinery is
out of scope here. This models the same three security properties a
verifier must check — signature integrity, expiry, and scope/limit
enforcement — using HMAC-SHA256 over a canonical JSON payload instead of a
full VC/JWT stack. That is enough to demo mandate forgery/replay
(identify/taxonomy.yaml S4-01) and scope-creep (S4-03) detection without a
new dependency.

DEV-ONLY: _MOCK_ISSUER_SECRET is a fixed, published constant baked into
this file. It is a sandbox signing key for a localhost demo, not a
production secret — see ETHICS.md. Never reuse it, or this signing scheme,
outside this sandbox.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass

_MOCK_ISSUER_SECRET = b"ai-defense-lab-mock-issuer-secret-do-not-use-in-prod"


@dataclass
class Mandate:
    mandate_id: str
    mandate_type: str  # "intent" | "cart" | "payment"
    payer_id: str
    max_amount: float
    currency: str
    allowed_mcc: list[int]
    issued_at: float
    expires_at: float
    signature: str = ""

    def _canonical_payload(self) -> bytes:
        payload = {
            "mandate_id": self.mandate_id,
            "mandate_type": self.mandate_type,
            "payer_id": self.payer_id,
            "max_amount": self.max_amount,
            "currency": self.currency,
            "allowed_mcc": sorted(self.allowed_mcc),
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
        }
        return json.dumps(payload, sort_keys=True).encode("utf-8")

    def sign(self) -> "Mandate":
        self.signature = hmac.new(_MOCK_ISSUER_SECRET, self._canonical_payload(), hashlib.sha256).hexdigest()
        return self


def verify_mandate(mandate: Mandate, proposed_amount: float, proposed_mcc: int, now: float | None = None) -> list[str]:
    """Return verification failures (empty list means the mandate authorizes this transaction)."""
    now = now if now is not None else time.time()
    errors: list[str] = []

    expected_signature = hmac.new(_MOCK_ISSUER_SECRET, mandate._canonical_payload(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected_signature, mandate.signature):
        errors.append("signature invalid")
    if now > mandate.expires_at:
        errors.append("mandate expired")
    if proposed_amount > mandate.max_amount:
        errors.append(f"amount {proposed_amount} exceeds mandate cap {mandate.max_amount}")
    if proposed_mcc not in mandate.allowed_mcc:
        errors.append(f"mcc {proposed_mcc} outside mandate scope {mandate.allowed_mcc}")

    return errors
