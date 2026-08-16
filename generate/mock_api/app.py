"""Sandboxed mock payment API — stub this pass.

Per ETHICS.md, this is the *only* HTTP surface the attacker will ever be
allowed to target once implemented, and it never leaves localhost. Right
now it's a runnable FastAPI skeleton with routes that raise
NotImplementedError: the current vertical slice scores transactions
in-process (see defend/transaction/model.py) rather than over HTTP, so
there is nothing to route to yet.

Future work: a /transactions/score route that wraps FraudDetector.score(),
and an AP2-style /mandates/verify route (signature, expiry, scope checks)
for the agentic-payment demo (component C4 in the team brief).
"""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="ai-defense-lab mock payment API", version="0.0.1-stub")


@app.get("/health")
def health() -> dict:
    return {"status": "stub", "implemented_routes": []}


@app.post("/transactions/score")
def score_transaction(transaction: dict) -> dict:
    raise NotImplementedError(
        "Mock payment API scoring endpoint not implemented this pass. "
        "Use defend.transaction.model.FraudDetector.score() in-process instead."
    )


@app.post("/mandates/verify")
def verify_mandate(mandate: dict) -> dict:
    raise NotImplementedError(
        "AP2-style mandate verification (signature/expiry/scope checks) "
        "is not implemented this pass."
    )
