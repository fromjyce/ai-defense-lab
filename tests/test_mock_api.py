"""Exercises the mock API's route functions directly (not over HTTP) so we
don't need to add an httpx/TestClient dependency just for tests — the route
functions are plain Python functions under FastAPI's decorators.
"""

import time

import pytest
from fastapi import HTTPException

from generate.mock_api import app as app_module
from generate.mock_api.mandates import Mandate


@pytest.fixture
def wired_app(fast_settings, trained_detector, monkeypatch):
    monkeypatch.setattr(app_module, "_settings", lambda: fast_settings)
    monkeypatch.setattr(app_module, "_detector", lambda: trained_detector)
    return app_module


def test_score_transaction_returns_valid_score(wired_app, small_df) -> None:
    row = small_df.iloc[0].to_dict()
    row["timestamp"] = row["timestamp"].isoformat()
    del row["label"]

    result = wired_app.score_transaction(row)

    assert 0.0 <= result["score"] <= 1.0
    assert result["flagged"] == (result["score"] >= result["threshold"])


def test_score_transaction_rejects_missing_fields(wired_app) -> None:
    with pytest.raises(HTTPException) as exc_info:
        wired_app.score_transaction({"amount": 10.0})
    assert exc_info.value.status_code == 422


def test_verify_mandate_route_authorizes_valid_mandate(wired_app) -> None:
    mandate = Mandate(
        mandate_id="m1",
        mandate_type="payment",
        payer_id="p1",
        max_amount=100.0,
        currency="INR",
        allowed_mcc=[5411],
        issued_at=time.time(),
        expires_at=time.time() + 3600,
    ).sign()

    payload = {
        "mandate": mandate.__dict__,
        "proposed_amount": 50.0,
        "proposed_mcc": 5411,
    }
    result = wired_app.verify_mandate_route(payload)

    assert result["authorized"] is True
    assert result["errors"] == []


def test_verify_mandate_route_rejects_scope_violation(wired_app) -> None:
    mandate = Mandate(
        mandate_id="m1",
        mandate_type="payment",
        payer_id="p1",
        max_amount=100.0,
        currency="INR",
        allowed_mcc=[5812],
        issued_at=time.time(),
        expires_at=time.time() + 3600,
    ).sign()

    payload = {
        "mandate": mandate.__dict__,
        "proposed_amount": 50.0,
        "proposed_mcc": 5411,
    }
    result = wired_app.verify_mandate_route(payload)

    assert result["authorized"] is False
    assert any("scope" in e for e in result["errors"])
