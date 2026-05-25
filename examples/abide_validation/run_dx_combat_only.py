"""ABIDE Diagnosis — ComBat Only (No Site Dummies in GLM).

Isolates the effect of the ComBat transformation:
- ComBat is fit on residuals of interest (nuisance-only).
- NO site dummies are included in the GLM.
- NO stratified permutations (standard shuffle).

This test determines if the ComBat data transformation is sufficient
by itself to neutralize site effects in ABIDE.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from conninfpy import AtlasInfo, analyze
from conninfpy.harmonize import combat_harmonize

HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "results" / "abide_prepared.npz"
OUT_DIR = HERE / "results" / "diagnosis" / "combat_only"


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


def run_dx_combat_only(data: dict, atlas: AtlasInfo, n_perm: int, seed: int) -> None:
    print("\n" + "=" * 72)
    print("Diagnosis Contrast: ASD vs. HC (ComBat ONLY - No Site Dummies)")
    print("=" * 72)

    Y = data["connectivity_z"]
    group = data["group"].astype(float)
    confounds = np.column_stack([
        data["age"].astype(float),
        data["sex"].astype(float),
        data["mean_fd"].astype(float)
    ])
    sites = data["site"]

    # 1. Manually apply ComBat (Strategy D style: preserve bio, not diagnosis)
    print("  Applying ComBat (nuisance-only)...")
    combat_out = combat_harmonize(Y, sites, preserve=confounds)
    Y_harm = combat_out.Y_adjusted

    # 2. Run analysis on harmonized data WITHOUT site info
    t0 = time.time()
    out = analyze(
        Y_harm,
        interest=group,
        confounds=confounds,
        sites=None,      # No site dummies
        harmonize=None,  # Already harmonized
        fisher_z=False,
        method="tfnbs",
        n_permutations=n_perm,
        rng=seed,
        use_mp=True,
    )
    print(f"  time: {time.time() - t0:.1f}s")
    
    nsig = out.inference.n_significant()
    print(f"  sig@α=0.05: positive={nsig['positive']} (ASD>HC), "
          f"negative={nsig['negative']} (HC>ASD)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(
        OUT_DIR / "dx_combat_only_edges.csv",
        atlas=atlas,
        alpha=0.05,
        sort="network_pair",
    )
    print(f"  saved edge table: {OUT_DIR / 'dx_combat_only_edges.csv'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    n_perm = 500 if args.quick else 5000

    data = _load()
    atlas = _build_atlas(data)
    run_dx_combat_only(data, atlas, n_perm, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
