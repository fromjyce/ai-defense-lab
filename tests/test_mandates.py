import time

from generate.mock_api.mandates import Mandate, verify_mandate


def _make_mandate(**overrides) -> Mandate:
    fields = dict(
        mandate_id="m1",
        mandate_type="payment",
        payer_id="p1",
        max_amount=100.0,
        currency="INR",
        allowed_mcc=[5411, 5812],
        issued_at=time.time(),
        expires_at=time.time() + 3600,
    )
    fields.update(overrides)
    return Mandate(**fields).sign()


def test_valid_mandate_authorizes_matching_transaction() -> None:
    mandate = _make_mandate()
    assert verify_mandate(mandate, proposed_amount=50.0, proposed_mcc=5411) == []


def test_tampered_mandate_fails_signature_check() -> None:
    mandate = _make_mandate()
    mandate.max_amount = 100000.0  # tamper after signing, without re-signing
    errors = verify_mandate(mandate, proposed_amount=100.0, proposed_mcc=5411)
    assert "signature invalid" in errors


def test_expired_mandate_rejected() -> None:
    mandate = _make_mandate(expires_at=time.time() - 10)
    errors = verify_mandate(mandate, proposed_amount=50.0, proposed_mcc=5411)
    assert "mandate expired" in errors


def test_amount_over_cap_rejected() -> None:
    mandate = _make_mandate(max_amount=10.0)
    errors = verify_mandate(mandate, proposed_amount=50.0, proposed_mcc=5411)
    assert any("exceeds mandate cap" in e for e in errors)


def test_mcc_outside_scope_rejected() -> None:
    mandate = _make_mandate(allowed_mcc=[5812])
    errors = verify_mandate(mandate, proposed_amount=50.0, proposed_mcc=5411)
    assert any("outside mandate scope" in e for e in errors)
