"""Synthetic-data fidelity harness: marginal similarity, correlation
similarity, and a distance-to-closest-record (DCR) privacy check, per the
team brief's "Is our data realistic?" metric group (TSTR + marginal &
correlation similarity + a privacy check).

Two distinct uses:

1. `compare_synthetic_runs` — an internal-consistency check requiring no
   external data: two independently-seeded runs of our own generator
   should be highly similar to each other (same generative process,
   different draws). This runs today, with no blockers, and is what
   `make fidelity` computes by default.

2. `evaluate_against_real` — the actual "is this realistic compared to the
   real world" check the brief asks for (train-on-synthetic/test-on-real
   fidelity evidence). This needs a real dataset (PaySim per the brief) on
   a caller-supplied column mapping, since PaySim's schema does not match
   ours field-for-field (see generate/synth/schema.py's docstring on why
   we generate our own richer schema rather than PaySim's own columns).
   Nothing in this repo downloads or reads real data automatically —
   scripts/download_paysim.sh is a manual, user-run, license-accepted
   step. Until that has been run, this function is simply not invoked;
   no number is invented in its place.

TSTR trap this harness guards against explicitly (per the brief): synthetic
float columns compared against real integer columns of the same logical
quantity silently zero out similarity/predictive power if left uncast.
`enforce_common_dtypes` casts every shared comparison column to float64
before any statistic is computed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from config.settings import GeneratorConfig, Settings, load_config
from generate.synth.generator import generate_synthetic

# Raw numeric columns compared by default. Categorical/identifier columns
# (currency, mcc, channel, payer_id, ...) are out of scope for this v1
# harness -- comparing them needs a frequency-distribution metric, not a
# continuous-distribution one, and isn't required to unblock the brief's
# metric.
FIDELITY_NUMERIC_COLUMNS: tuple[str, ...] = (
    "amount",
    "payer_account_age_days",
    "velocity_1h",
    "velocity_24h",
)


@dataclass
class FidelityReport:
    columns: list[str]
    marginal_similarity: dict[str, float]
    marginal_similarity_mean: float
    correlation_similarity: float
    dcr_median_ratio: float
    n_candidate_rows: int
    n_reference_rows: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def enforce_common_dtypes(a: pd.DataFrame, b: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cast shared comparison columns to float64 on both frames.

    Guards against the specific TSTR trap the brief calls out: a synthetic
    float column compared against a real int column of the same logical
    quantity (or vice versa) silently destroys similarity/predictive power
    if left uncast.
    """
    a2, b2 = a.copy(), b.copy()
    for col in columns:
        a2[col] = a2[col].astype("float64")
        b2[col] = b2[col].astype("float64")
    return a2, b2


def _empirical_cdf_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (max CDF gap), computed by
    hand so this harness has no new dependency beyond what's already
    pinned (numpy/pandas/sklearn)."""
    combined = np.sort(np.concatenate([x, y]))
    cdf_x = np.searchsorted(np.sort(x), combined, side="right") / len(x)
    cdf_y = np.searchsorted(np.sort(y), combined, side="right") / len(y)
    return float(np.max(np.abs(cdf_x - cdf_y)))


def marginal_similarity(a: pd.DataFrame, b: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    """Per-column similarity in [0, 1]: 1 - KS distance. 1.0 means the two
    empirical distributions are indistinguishable by KS statistic."""
    return {col: 1.0 - _empirical_cdf_distance(a[col].to_numpy(), b[col].to_numpy()) for col in columns}


def correlation_similarity(a: pd.DataFrame, b: pd.DataFrame, columns: list[str]) -> float:
    """1 - normalized Frobenius distance between the two correlation
    matrices, clipped to [0, 1]. 1.0 means identical linear structure."""
    if len(columns) < 2:
        return 1.0
    corr_a = a[columns].corr().to_numpy()
    corr_b = b[columns].corr().to_numpy()
    corr_a = np.nan_to_num(corr_a)
    corr_b = np.nan_to_num(corr_b)
    frobenius_diff = float(np.linalg.norm(corr_a - corr_b))
    max_possible = float(np.linalg.norm(np.full_like(corr_a, 2.0)))  # every entry off by the max possible (2.0)
    return float(np.clip(1.0 - frobenius_diff / max_possible, 0.0, 1.0))


def dcr_privacy_check(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    columns: list[str],
    sample_size: int = 1000,
    seed: int = 42,
) -> float:
    """Median distance-to-closest-record ratio: for each candidate row,
    (distance to nearest reference row) / (distance to nearest other
    candidate row). A ratio well below 1.0 means candidate rows sit closer
    to reference rows than to each other -- a memorization/privacy red
    flag. Distances computed on z-scored columns so differently-scaled
    fields don't dominate.
    """
    rng = np.random.default_rng(seed)
    cand = candidate.sample(n=min(sample_size, len(candidate)), random_state=int(rng.integers(0, 2**31 - 1)))
    ref = reference.sample(n=min(sample_size, len(reference)), random_state=int(rng.integers(0, 2**31 - 1)))

    combined_mean = pd.concat([cand[columns], ref[columns]]).mean()
    combined_std = pd.concat([cand[columns], ref[columns]]).std().replace(0, 1.0)
    cand_z = ((cand[columns] - combined_mean) / combined_std).to_numpy()
    ref_z = ((ref[columns] - combined_mean) / combined_std).to_numpy()

    dist_to_reference = pairwise_distances(cand_z, ref_z).min(axis=1)

    self_dist = pairwise_distances(cand_z, cand_z)
    np.fill_diagonal(self_dist, np.inf)
    dist_to_other_candidates = self_dist.min(axis=1)

    ratios = dist_to_reference / np.where(dist_to_other_candidates == 0, 1e-9, dist_to_other_candidates)
    return float(np.median(ratios))


def _build_report(candidate: pd.DataFrame, reference: pd.DataFrame, columns: list[str], notes: list[str]) -> FidelityReport:
    cand, ref = enforce_common_dtypes(candidate, reference, columns)
    marginals = marginal_similarity(cand, ref, columns)
    return FidelityReport(
        columns=list(columns),
        marginal_similarity=marginals,
        marginal_similarity_mean=float(np.mean(list(marginals.values()))),
        correlation_similarity=correlation_similarity(cand, ref, columns),
        dcr_median_ratio=dcr_privacy_check(cand, ref, columns),
        n_candidate_rows=len(candidate),
        n_reference_rows=len(reference),
        notes=notes,
    )


def compare_synthetic_runs(cfg: GeneratorConfig, seed_a: int, seed_b: int) -> FidelityReport:
    """Internal-consistency check: two independent draws from our own
    generator, same config. High similarity here means the generator is a
    stable statistical process -- it does NOT mean the process resembles
    real-world payment behavior. See evaluate_against_real for that."""
    df_a = generate_synthetic(cfg, seed_a)
    df_b = generate_synthetic(cfg, seed_b)
    return _build_report(
        df_a,
        df_b,
        list(FIDELITY_NUMERIC_COLUMNS),
        notes=[
            "This is an internal-consistency check (synthetic vs synthetic, "
            "different seeds), not a fidelity-to-real-world measurement. "
            "See evaluate_against_real() / TODO(data) for the real comparison."
        ],
    )


def evaluate_against_real(synthetic_df: pd.DataFrame, real_df: pd.DataFrame, column_map: dict[str, str]) -> FidelityReport:
    """The brief's actual TSTR-adjacent fidelity check: synthetic vs a real
    dataset (e.g. PaySim), on a caller-supplied column_map of
    {our_column_name: real_column_name} for whichever columns are
    genuinely comparable across the two schemas.
    """
    renamed_real = real_df.rename(columns={v: k for k, v in column_map.items()})
    columns = list(column_map.keys())
    return _build_report(
        synthetic_df,
        renamed_real,
        columns,
        notes=[f"Compared against real data via column_map={column_map}."],
    )


def write_json(report: FidelityReport, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic-data fidelity harness.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--seed-b-offset", type=int, default=1, help="second run uses settings.seed + this offset")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--real", type=Path, default=None, help="optional real dataset CSV, e.g. downloaded PaySim")
    parser.add_argument(
        "--column-map",
        type=str,
        default=None,
        help='JSON object mapping our column names to the real dataset\'s column names, e.g. \'{"amount": "amount"}\'',
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    settings: Settings = load_config(args.config)

    if args.real is not None:
        if args.column_map is None:
            raise SystemExit("--real requires --column-map")
        synthetic_df = generate_synthetic(settings.generator, settings.seed)
        real_df = pd.read_csv(args.real)
        report = evaluate_against_real(synthetic_df, real_df, json.loads(args.column_map))
    else:
        report = compare_synthetic_runs(settings.generator, settings.seed, settings.seed + args.seed_b_offset)
        report.notes.append(
            "TODO(data): no real dataset supplied. Run scripts/download_paysim.sh "
            "under your own Kaggle account/license, then re-run with --real and "
            "--column-map for a genuine fidelity-to-real-world number."
        )

    write_json(report, args.out)
    print(
        f"marginal similarity (mean) {report.marginal_similarity_mean:.4f}  "
        f"correlation similarity {report.correlation_similarity:.4f}  "
        f"DCR median ratio {report.dcr_median_ratio:.4f}  wrote {args.out}"
    )


if __name__ == "__main__":
    main()
