# Ethics & responsible-AI statement

## Scope

This project builds a stress-testing capability for payment-fraud defenses,
not an attack tool. The evolutionary attacker in `generate/attacker/` and,
once implemented, the mock payment API in `generate/mock_api/` are the entire
attack surface this repo touches — both are our own, sandboxed, in-process
components.

- **No live systems.** Adversarial testing targets only our own in-process
  detector (`defend/transaction/`) and, once implemented, our own sandboxed
  mock payment API (`generate/mock_api/`). No code in this repo makes network
  calls to any external host at runtime, and none ever will.
- **No real data.** Synthetic, anonymised, or openly/research-licensed data
  only. No real cardholder data, no PII, no production payment data, under
  any circumstances.
- **No operational attack tooling.** The evolutionary attacker mutates
  numeric and categorical transaction *features* only — amount, channel,
  auth method, and similar fields. There are no scam scripts, phishing
  templates, or social-engineering content anywhere in this repository, and
  there never will be.
- **Attacker constraints are load-bearing, not decorative.** The attacker's
  validity constraints (`generate/attacker/validity.py`) reject semantically
  impossible transactions and the attacker cannot rewrite fields that
  represent the issuer's own ground truth about an account (e.g. account
  age, historical velocity) — it can only manipulate what a real fraudster
  could plausibly control at submission time.

## What we deliberately withheld

- Nothing in this repo names, targets, or attempts to fingerprint any real
  production fraud-detection system, bank, or payment network.
- No dataset requiring restricted or licensed access (e.g. FaceForensics++,
  DFDC) is committed to this repository. Where such datasets are used, only
  a download script under `scripts/` is provided, to be run manually under
  the user's own license agreement.

## Responsible disclosure

This is a research/competition prototype, not a deployed system, so there is
no live vulnerability-disclosure process to run. If a technique demonstrated
here (an evasion pattern, a validity-constraint gap) has an analogue a reader
recognizes in a real production system, we ask that it be reported to that
system's own security or fraud team rather than reproduced against it.

## Attribution

Every third-party dataset, model, or sample referenced by this project keeps
its original license and attribution. See `README.md` for data provenance
and `scripts/download_*.sh` headers for per-dataset license terms.
