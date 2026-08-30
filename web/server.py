"""Web prototype: a single-page dashboard over the closed loop's own
output, per the team brief's C4/deliverable requirement ("presentable UI
showing the closed loop live: attacker evolving, detector catching, a
forged mandate being blocked, metrics updating").

This is a read-only view onto files `make loop`/`make eval`/`make
fidelity` already produce in results/, plus the sandboxed mock payment API
(generate/mock_api/app.py) mounted at /api/mock so the dashboard can drive
live scoring and mandate-verification demos without duplicating that
detector-loading logic. No new attack surface: everything this app can do,
generate/mock_api/app.py could already do directly.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config.settings import REPO_ROOT, Settings, load_config
from generate.mock_api.app import app as mock_api_app
from generate.mock_api.mandates import Mandate, verify_mandate

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="ai-defense-lab dashboard")
app.mount("/api/mock", mock_api_app)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _settings() -> Settings:
    return load_config()


def _results_dir() -> Path:
    return REPO_ROOT / _settings().paths.results_dir


def _read_json(name: str) -> dict | list:
    path = _results_dir() / name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{name} not found — run the relevant make target first")
    with open(path) as f:
        return json.load(f)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/evasion-curve")
def evasion_curve() -> dict:
    return {"records": _read_json("evasion_curve.json")}


@app.get("/api/attack-curve")
def attack_curve() -> dict:
    path = _results_dir() / "attack_generation_curve.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="attack_generation_curve.csv not found — run `make loop` first")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {"rows": rows}


@app.get("/api/metrics")
def metrics() -> dict:
    return {"metrics": _read_json("metrics.json")}


@app.get("/api/fidelity")
def fidelity() -> dict:
    return {"fidelity": _read_json("fidelity_report.json")}


@app.get("/api/taxonomy")
def taxonomy() -> dict:
    path = REPO_ROOT / "identify" / "taxonomy.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return {"rows": data["rows"]}


@app.get("/api/sample-transaction")
def sample_transaction(label: int = 1) -> dict:
    """A real row from the generated dataset, so the dashboard's scoring
    demo doesn't require hand-typing 16 transaction fields."""
    path = REPO_ROOT / _settings().paths.data_dir / "transactions.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="data/transactions.csv not found — run `make data` first")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    candidates = df[df["label"] == label]
    if len(candidates) == 0:
        raise HTTPException(status_code=404, detail=f"no rows with label={label}")
    row = candidates.sample(n=1).iloc[0].to_dict()
    row["timestamp"] = row["timestamp"].isoformat()
    del row["label"]
    return row


@app.post("/api/mandate-demo")
def mandate_demo() -> dict:
    """Signs a valid mandate and verifies it (authorized), then replays the
    same mandate against an out-of-scope transaction (blocked) — the
    "forged mandate being blocked" panel the brief's deliverable asks for.
    """
    now = time.time()
    mandate = Mandate(
        mandate_id="demo-mandate-001",
        mandate_type="payment",
        payer_id="demo-payer",
        max_amount=500.0,
        currency="INR",
        allowed_mcc=[5411, 5812],
        issued_at=now,
        expires_at=now + 3600,
    ).sign()

    within_scope = verify_mandate(mandate, proposed_amount=200.0, proposed_mcc=5411)

    tampered = Mandate(**{**mandate.__dict__, "max_amount": 50000.0})  # signature now stale
    scope_creep = verify_mandate(mandate, proposed_amount=200.0, proposed_mcc=7995)  # mcc outside allowed_mcc
    replay_over_cap = verify_mandate(mandate, proposed_amount=50000.0, proposed_mcc=5411)
    forged = verify_mandate(tampered, proposed_amount=200.0, proposed_mcc=5411)

    return {
        "mandate": mandate.__dict__,
        "scenarios": [
            {"name": "Valid payment within mandate scope", "errors": within_scope, "authorized": len(within_scope) == 0},
            {"name": "Amount exceeds mandate cap (replay at higher value)", "errors": replay_over_cap, "authorized": len(replay_over_cap) == 0},
            {"name": "Merchant category outside mandate scope", "errors": scope_creep, "authorized": len(scope_creep) == 0},
            {"name": "Tampered mandate (signature no longer matches)", "errors": forged, "authorized": len(forged) == 0},
        ],
    }
