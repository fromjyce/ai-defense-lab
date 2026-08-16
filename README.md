# ai-defense-lab

Closed-loop red-team / blue-team payment fraud system, built for the
Mastercard Innovation Challenge @ GFF 2026.

The loop is the product: a synthetic-transaction generator feeds a LightGBM
fraud detector; an evolutionary attacker mutates transactions against the
detector's own score until it evades; successful evasions are mined back into
the training set; the detector retrains; the cycle repeats. We track both the
attacker's success rate and the detector's clean-set PR-AUC per generation so
the detector can't "win" by degrading into a blanket blocker.

## Status: scaffold + vertical slice

This repo is deliberately built in passes. The current pass wires one
end-to-end path that actually runs:

```
synthetic transactions -> LightGBM detector -> evolutionary attacker
    -> evasions mined -> detector retrains -> evasion curve -> results/
```

Everything else in the full design (multimodal deepfake fusion, the
graph/mule layer, the AP2 mandate demo, the web UI) is stubbed behind a
clean interface (`NotImplementedError` or a constant-returning stub) so the
loop is demoable today and each stream can be filled in independently later.

| Component | This pass |
|---|---|
| `generate/synth` | Implemented — named, interpretable fields, configurable base rates |
| `generate/attacker` | Implemented — evolutionary search over attacker-controllable fields |
| `generate/mock_api` | Stub — FastAPI skeleton, routes raise `NotImplementedError` |
| `defend/transaction` | Implemented — sklearn Pipeline, LightGBM, isotonic calibration |
| `defend/multimodal` | Stub interface — constant-score stub |
| `defend/graph` | Stub interface — empty-feature stub |
| `defend/eval` | Implemented — PR-AUC, ROC-AUC, F1, recall@FPR, precision@k, Brier, latency |
| `loop/orchestrator.py` | Implemented — runs the full attack/retrain cycle |
| `web/` | Empty placeholder |

## Sandbox statement

All adversarial testing in this repository runs against our own in-process
detector and, once implemented, our own sandboxed mock payment API only.
Nothing here connects to, tests against, or is intended for use against any
live payment system, real cardholder data, or third-party infrastructure.

## Data provenance

All data used or planned for use in this repo is synthetic, anonymised, or
released under a research/open license. No real cardholder data, no PII, no
production payment data — ever. The default generator (`generate/synth`)
produces fully synthetic transactions from configured base rates; it does not
read or derive from any real dataset. A `--source` flag exists to later swap
in a derived dataset (e.g. PaySim); the fetch scripts for any such dataset
live in `scripts/download_*.sh` and are run manually, under the user's own
license agreement, never automatically.

## Setup

Requires Python 3.11. If `python3.11` isn't your default `python3`, install
it first (e.g. `brew install python@3.11` on macOS).

```
make setup   # creates .venv with python3.11, installs pinned requirements
make data    # generates a synthetic transaction dataset into data/
make train   # trains and calibrates the LightGBM detector
make loop    # runs the closed-loop attacker/detector co-evolution
make eval    # writes evaluation metrics to results/
make test    # runs the pytest smoke suite
make demo    # placeholder — see web/
make clean   # removes data/, results/ contents, and the venv
```

`make loop` run twice with the same `config/default.yaml` produces identical
numbers — the seed in `config/default.yaml` is threaded through numpy,
Python's `random`, and LightGBM.

## Configuration

All tunable parameters (seed, generator base rates, LightGBM hyperparameters,
evolutionary-attacker settings, loop generation count, metric thresholds)
live in `config/default.yaml`, loaded via `config/settings.py`
(pydantic-settings). No magic numbers in code.

## Repository layout

See `config/default.yaml` for parameters and `identify/taxonomy.yaml` for the
attack taxonomy (schema only this pass). Component-level detail is in the
table above.
