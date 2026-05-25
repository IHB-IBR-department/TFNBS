"""ABIDE Diagnosis Strategy E — No ComBat, but Site Dummies in GLM.

Baseline for Plot 7:
- No ComBat harmonization.
- Site dummies ARE included in GLM.
- Stratified permutations.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from conninfpy import AtlasInfo, analyze

HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "results" / "abide_prepared.npz"
OUT_DIR = HERE / "results" / "diagnosis" / "strategy_e"


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


def run_dx_strat_e(data: dict, atlas: AtlasInfo, n_perm: int, seed: int) -> None:
    print("\n" + "=" * 72)
    print("Diagnosis Contrast: ASD vs. HC (Strategy E - No ComBat)")
    print("=" * 72)

    Y = data["connectivity_z"]
    group = data["group"].astype(float)
    confounds = np.column_stack([
        data["age"].astype(float),
        data["sex"].astype(float),
        data["mean_fd"].astype(float)
    ])
    sites = data["site"]

    t0 = time.time()
    out = analyze(
        Y,
        interest=group,
        confounds=confounds,
        sites=sites,
        harmonize=None,  # Strategy E (No ComBat)
        fisher_z=False,
        method="tfnbs",
        n_permutations=n_perm,
        rng=seed,
        use_mp=True,
    )
    print(f"  time: {time.time() - t0:.1f}s")
    
    nsig = out.inference.n_significant()
    print(f"  sig@α=0.05: positive={nsig['positive']}, negative={nsig['negative']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(
        OUT_DIR / "dx_strategy_e_edges.csv",
        atlas=atlas,
        alpha=0.05,
        sort="network_pair",
    )
    print(f"  saved edge table: {OUT_DIR / 'dx_strategy_e_edges.csv'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    n_perm = 500 if args.quick else 5000

    data = _load()
    atlas = _build_atlas(data)
    run_dx_strat_e(data, atlas, n_perm, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
