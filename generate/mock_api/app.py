"""Sandboxed mock payment API.

Per ETHICS.md, this is the *only* HTTP surface an attacker is ever allowed
to target in this repo, and it never leaves localhost. Implements two
routes on top of components already built elsewhere in this repo:

  /transactions/score  wraps the trained FraudDetector in-process.
  /mandates/verify      AP2-style mandate signature/expiry/scope checks
                         (see generate/mock_api/mandates.py) — catches
                         identify/taxonomy.yaml rows S4-01 (mandate
                         forgery/replay) and S4-03 (scope creep).

The closed loop (loop/orchestrator.py) still scores in-process, not over
HTTP — this app is a separate, optional demo surface, not on the
attacker's or the loop's critical path.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

from config.settings import Settings, load_config
from defend.transaction.model import FraudDetector
from generate.mock_api.mandates import Mandate, verify_mandate
from generate.synth.schema import COLUMNS

app = FastAPI(title="ai-defense-lab mock payment API", version="0.2.0")

_TRANSACTION_FIELDS = [c for c in COLUMNS if c != "label"]


@lru_cache(maxsize=1)
def _settings() -> Settings:
    return load_config()


@lru_cache(maxsize=1)
def _detector() -> FraudDetector:
    settings = _settings()
    model_path = Path(settings.paths.model_dir) / "detector.joblib"
    if not model_path.exists():
        raise RuntimeError(f"no trained detector at {model_path}; run `make train` first")
    return FraudDetector.load(model_path)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "implemented_routes": ["/transactions/score", "/mandates/verify"]}


@app.post("/transactions/score")
def score_transaction(transaction: dict) -> dict:
    missing = [c for c in _TRANSACTION_FIELDS if c not in transaction]
    if missing:
        raise HTTPException(status_code=422, detail=f"missing fields: {missing}")

    row = {c: transaction[c] for c in _TRANSACTION_FIELDS}
    df = pd.DataFrame([row])
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    try:
        score = float(_detector().score(df)[0])
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    threshold = _settings().attacker.deployed_threshold
    return {"score": score, "flagged": score >= threshold, "threshold": threshold}


@app.post("/mandates/verify")
def verify_mandate_route(payload: dict) -> dict:
    required = ["mandate", "proposed_amount", "proposed_mcc"]
    missing = [f for f in required if f not in payload]
    if missing:
        raise HTTPException(status_code=422, detail=f"missing fields: {missing}")

    mandate_fields = payload["mandate"]
    required_mandate_fields = [
        "mandate_id", "mandate_type", "payer_id", "max_amount", "currency",
        "allowed_mcc", "issued_at", "expires_at", "signature",
    ]
    missing_mandate = [f for f in required_mandate_fields if f not in mandate_fields]
    if missing_mandate:
        raise HTTPException(status_code=422, detail=f"missing mandate fields: {missing_mandate}")

    mandate = Mandate(**{f: mandate_fields[f] for f in required_mandate_fields})
    errors = verify_mandate(mandate, payload["proposed_amount"], payload["proposed_mcc"])

    return {"authorized": len(errors) == 0, "errors": errors}
