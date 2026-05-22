"""ABIDE Age → FC under analyze(harmonize='nuisance_only') — Strategy D.

Sibling exemplar to `run_age_combat_v2.py`. Same scientific question,
different harmonization strategy: ComBat fits with `preserve=confounds`
only (the tested variable is *deliberately omitted* from ComBat) and
site dummies are appended to the GLM nuisance design. Removes the
Nygaard 2016 two-step label leak that Strategy B can introduce when
the tested variable is correlated with site.

Methodological background:
[[Projects/NetworkStatistics/paper/paper_combat_resolution_strategies|paper_combat_resolution_strategies]]
and [[Projects/NetworkStatistics/_wiki/examples_refactor_2026-05-22|examples_refactor_2026-05-22]]
§3.3 (Strategy B vs D vs E ablation).

Side-by-side comparison vs `run_age_combat_v2.py`:

| Run                         | preserve            | site in GLM nuisance |
|-----------------------------|---------------------|----------------------|
| run_age_combat_v2.py        | interest + confounds | no                   |
| run_age_combat_v2_d.py      | confounds only      | yes (auto-appended)  |

Pre-registered expectation: block-mass Pearson B vs D ≥ 0.85 on
ABIDE Age. If the strategies disagree on the *direction* of effects,
that's a Nygaard-style inflation signal in Strategy B.

Results go to `results/combat_v2_d/{variant}.npz` + an atlas-annotated
edge table. Pair with `run_age_combat_v2.py` outputs for the §3.3
ablation summary.

Usage:
    python examples/abide_validation/run_age_combat_v2_d.py --variant age
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
OUT_DIR = HERE / "results" / "combat_v2_d"


def _load() -> dict:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"{DATA_FILE} not found — run `python prepare_data.py` first."
        )
    d = np.load(DATA_FILE, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _build_atlas(data: dict) -> AtlasInfo:
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
    diag = result.combat_diagnostics
    if diag is not None:
        strategy = diag.get("strategy", "?")
        rb = diag.get("between_site_variance_ratio_before")
        ra = diag.get("between_site_variance_ratio_after")
        if rb is not None and ra is not None:
            print(f"  ComBat (strategy {strategy}) "
                  f"between-site var: {rb:.4f} → {ra:.4f}")


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
    print(f"  harmonize='nuisance_only' (Strategy D), site dummies in GLM")
    print("=" * 72)

    Y = data["connectivity_z"]
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
        harmonize="nuisance_only",                   # Strategy D
        fisher_z=False,
        method="tfnbs",
        e=0.4, h=3.0, n=10,
        n_permutations=n_perm,
        acceleration=None,
        rng=seed,
        use_mp=True,
    )
    print(f"  time: {time.time() - t0:.1f}s")
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
