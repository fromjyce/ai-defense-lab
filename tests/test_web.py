"""Exercises web/server.py's route functions directly, mirroring
tests/test_mock_api.py's approach. Only covers routes that don't depend on
results/ artifacts already existing (evasion-curve/metrics/fidelity need
`make loop`/`eval`/`fidelity` to have run first, which is out of scope for
a fast smoke test) — taxonomy and the mandate demo are self-contained.
"""

import pytest
from fastapi import HTTPException

from web import server as web_server


def test_taxonomy_route_returns_all_rows() -> None:
    result = web_server.taxonomy()
    assert len(result["rows"]) >= 15
    assert all({"surface", "vector", "rail", "severity", "source"} <= set(r.keys()) for r in result["rows"])


def test_mandate_demo_shows_valid_and_blocked_scenarios() -> None:
    result = web_server.mandate_demo()
    scenarios = {s["name"]: s for s in result["scenarios"]}

    assert scenarios["Valid payment within mandate scope"]["authorized"] is True
    assert scenarios["Amount exceeds mandate cap (replay at higher value)"]["authorized"] is False
    assert scenarios["Merchant category outside mandate scope"]["authorized"] is False
    assert scenarios["Tampered mandate (signature no longer matches)"]["authorized"] is False


def test_evasion_curve_route_404s_without_results(monkeypatch, tmp_path) -> None:
    empty_results_dir = tmp_path / "results"
    empty_results_dir.mkdir()
    monkeypatch.setattr(web_server, "_results_dir", lambda: empty_results_dir)

    with pytest.raises(HTTPException) as exc_info:
        web_server.evasion_curve()
    assert exc_info.value.status_code == 404
