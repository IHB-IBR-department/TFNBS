"""ABIDE Age → FC under analyze(harmonize='auto') — v2.1 exemplar (Strategy B).

This is the v2.1 successor to `run_age_combat.py`. The legacy script runs
the analysis in two steps (manual ComBat via `harmonize.py` → manual
`compute_p_val_glm`); the v2.1 entry point collapses both into one call:

    analyze(Y, interest=..., confounds=..., sites=..., harmonize='auto')

This invokes Strategy B (auto-preserve = `interest + confounds`,
site auto-strata for the permutation null) — see
[[Projects/NetworkStatistics/_wiki/examples_refactor_2026-05-22|examples_refactor_2026-05-22]]
Item M / Item N for the validation schema covered by this exemplar.

Results go to `results/combat_v2/{variant}.npz` plus an atlas-annotated
edge table at `results/combat_v2/{variant}_edges.csv`. Numerical
equivalence against the v0 outputs (`results/combat/age/{variant}.npz`)
is the acceptance test for the PR.

Usage:
    python examples/abide_validation/run_age_combat_v2.py --variant age
    python examples/abide_validation/run_age_combat_v2.py --quick
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

from conninfpy import AtlasInfo, analyze


HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "abide_prepared.npz"
OUT_DIR = HERE / "results" / "combat_v2"


def _load() -> dict:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"{DATA_FILE} not found — run `python prepare_data.py` first."
        )
    d = np.load(DATA_FILE, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _build_atlas(data: dict) -> AtlasInfo:
    """Wrap the prepared-data ROI metadata in an AtlasInfo so the
    output CSV carries ROI names and Yeo-7 networks."""
    network_order = list(data["network_order"])
    net_labels_int = np.asarray(data["net_labels"], dtype=int)
    networks_per_roi = [network_order[i] for i in net_labels_int]
    return AtlasInfo(
        labels=[str(x) for x in data["roi_names"]],
        networks=networks_per_roi,
        source="ABIDE I — Schaefer-100 / Yeo-7 (from abide_prepared.npz)",
    )


def _summary(label: str, result, alpha: float = 0.05) -> None:
    nsig = result.n_significant(alpha=alpha)
    print(f"  {label} sig@α={alpha}: positive={nsig['positive']}, "
          f"negative={nsig['negative']}")
    if result.combat_diagnostics is not None:
        ratio_before = result.combat_diagnostics.get(
            "between_site_variance_ratio_before"
        )
        ratio_after = result.combat_diagnostics.get(
            "between_site_variance_ratio_after"
        )
        if ratio_before is not None and ratio_after is not None:
            print(f"  ComBat between-site var: "
                  f"{ratio_before:.4f} → {ratio_after:.4f}")


def run_variant(
    data: dict,
    atlas: AtlasInfo,
    *,
    interest_name: str,
    confound_names: list[str],
    label: str,
    n_perm: int,
    seed: int,
) -> None:
    print("\n" + "=" * 72)
    print(f"{label}: interest={interest_name}, confounds={confound_names}")
    print(f"  harmonize='auto' (Strategy B), sites=site auto-strata")
    print("=" * 72)

    Y = data["connectivity_z"]                      # raw Fisher-z; analyze runs ComBat
    interest = data[interest_name].astype(float)
    confounds = np.column_stack(
        [data[c].astype(float) for c in confound_names]
    )
    sites = data["site"]
    print(f"  N={Y.shape[0]}, confounds shape {confounds.shape}, "
          f"K sites={len(np.unique(sites))}")

    t0 = time.time()
    out = analyze(
        Y,
        interest=interest,
        confounds=confounds,
        sites=sites,
        harmonize="auto",                            # Strategy B
        fisher_z=False,                              # Y is already Fisher-z
        method="tfnbs",
        e=0.4, h=3.0, n=10,                          # Hao 2024 regime
        n_permutations=n_perm,
        acceleration=None,                           # empirical to match v0
        rng=seed,
        use_mp=True,
    )
    elapsed = time.time() - t0
    print(f"  time: {elapsed:.1f}s")
    _summary(label, out.inference)
    for flag in out.flags:
        print(f"  flag: {flag}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_DIR / f"{label}.npz",
        positive=out.inference["positive"],
        negative=out.inference["negative"],
    )
    out.to_csv(
        OUT_DIR / f"{label}_edges.csv",
        atlas=atlas,
        alpha=0.05,
        sort="network_pair",
    )
    print(f"  saved: {OUT_DIR / f'{label}.npz'}")
    print(f"  saved: {OUT_DIR / f'{label}_edges.csv'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--variant", choices=["age", "age_dx", "fd", "all"],
                    default="all")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    n_perm = 500 if args.quick else 5000

    data = _load()
    atlas = _build_atlas(data)

    if args.variant in ("age", "all"):
        run_variant(data, atlas,
                    interest_name="age", confound_names=["sex", "mean_fd"],
                    label="age_tfnbs", n_perm=n_perm, seed=args.seed)
    if args.variant in ("age_dx", "all"):
        run_variant(data, atlas,
                    interest_name="age",
                    confound_names=["sex", "mean_fd", "group"],
                    label="age_controlling_dx_tfnbs",
                    n_perm=n_perm, seed=args.seed)
    if args.variant in ("fd", "all"):
        run_variant(data, atlas,
                    interest_name="mean_fd",
                    confound_names=["age", "sex", "group"],
                    label="fd_tfnbs", n_perm=n_perm, seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
