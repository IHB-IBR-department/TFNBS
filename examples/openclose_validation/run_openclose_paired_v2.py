"""Open-Close paired TFNBS under analyze(test_type='paired') — v2.1 exemplar.

v2.1 successor to `run_openclose_paired.py`. Same three analyses (IHB
paired, China paired-run0, China test-retest run0 vs run1) but routed
through the `analyze()` wrapper instead of `compute_p_val(...)`
directly. With no `sites=` argument the wrapper skips ComBat and just
runs the paired sign-flip permutation, but the v2.1 plumbing —
`AtlasInfo`-aware exports, `significant_edges` / `to_csv` for the edge
table, optional `summary_figure` rendering — applies uniformly.

Acceptance criterion: numerical equivalence against the v0 outputs
(`results/{ihb_paired_tfnbs,china_paired_tfnbs_run0,china_retest_tfnbs}.npz`)
under the same `rng=` seed.

Usage:
    python examples/openclose_validation/run_openclose_paired_v2.py --quick
    python examples/openclose_validation/run_openclose_paired_v2.py --analyses ihb
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from conninfpy import AtlasInfo, analyze
from examples.openclose_validation.openclose_loader import OpenCloseDataset


HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE / "results" / "paired_v2"


def _build_atlas(net_labels: np.ndarray, names: list[str]) -> AtlasInfo:
    """Construct a Schaefer-200/182 AtlasInfo from the loader output.

    The loader exposes integer-coded `net_labels` plus a `names` list
    indexed by those integers; `AtlasInfo` wants per-ROI network names.
    """
    net_labels = np.asarray(net_labels, dtype=int)
    networks_per_roi = [names[i] for i in net_labels]
    return AtlasInfo(
        labels=[f"roi_{i:03d}" for i in range(len(net_labels))],
        networks=networks_per_roi,
        source="Open-Close Schaefer-200 / 182 ROIs (loader-provided net_labels)",
    )


def _summary(label: str, result, alpha: float = 0.05) -> None:
    nsig = result.n_significant(alpha=alpha)
    print(f"  {label} sig@α={alpha}: positive={nsig['positive']}, "
          f"negative={nsig['negative']}")


def _run_paired(
    g1: np.ndarray,
    g2: np.ndarray,
    atlas: AtlasInfo,
    label: str,
    out_stem: str,
    n_perm: int,
    seed: int,
) -> None:
    print(f"\n{'=' * 72}\n{label} — {g1.shape[0]} subjects, "
          f"{g1.shape[1]}² edges\n{'=' * 72}")
    t0 = time.time()
    out = analyze(
        group1=g1, group2=g2,
        test_type="paired",
        fisher_z=False,                              # caller already Fisher-z'd
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

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_ROOT / f"{out_stem}.npz",
        positive=out.inference["positive"],
        negative=out.inference["negative"],
    )
    out.to_csv(
        OUT_ROOT / f"{out_stem}_edges.csv",
        atlas=atlas,
        alpha=0.05,
        sort="network_pair",
    )
    print(f"  saved: {OUT_ROOT / f'{out_stem}.npz'}")
    print(f"  saved: {OUT_ROOT / f'{out_stem}_edges.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="500 perms for smoke testing (~30 s each)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--analyses", nargs="+",
                    choices=["ihb", "china", "china_retest", "all"],
                    default=["all"])
    args = ap.parse_args()
    n_perm = 500 if args.quick else 5000
    targets = ["ihb", "china", "china_retest"] if "all" in args.analyses \
        else args.analyses

    print(f"Output → {OUT_ROOT}")
    print(f"n_permutations = {n_perm}")

    # ---- §3.5 IHB paired open vs close --------------------------------
    if "ihb" in targets:
        ds = OpenCloseDataset.load("ihb")
        atlas = _build_atlas(ds.net_labels, ds.get_label_names())
        o_z, c_z = ds.connectivity_z(run=0)
        _run_paired(
            o_z, c_z, atlas,
            label="§3.5 IHB paired open vs close",
            out_stem="ihb_paired_tfnbs_v2",
            n_perm=n_perm, seed=args.seed,
        )

    # ---- §3.6 China paired (run 0) -----------------------------------
    if "china" in targets:
        ds = OpenCloseDataset.load("china")
        atlas = _build_atlas(ds.net_labels, ds.get_label_names())
        o_z, c_z = ds.connectivity_z(run=0)
        _run_paired(
            o_z, c_z, atlas,
            label="§3.6 China paired (close run 0)",
            out_stem="china_paired_tfnbs_run0_v2",
            n_perm=n_perm, seed=args.seed,
        )

    # ---- §3.6 China test-retest: close run 0 vs run 1 ----------------
    if "china_retest" in targets:
        ds = OpenCloseDataset.load("china")
        atlas = _build_atlas(ds.net_labels, ds.get_label_names())
        mask = ds.retest_subject_mask()
        excluded = [s for s, m in zip(ds.subject_ids, mask) if not m]
        print(f"\n§3.6 China test-retest — excluding "
              f"{(~mask).sum()} subject(s): {excluded}")
        _, c0 = ds.connectivity_z(run=0)
        _, c1 = ds.connectivity_z(run=1)
        c0 = c0[mask]
        c1 = c1[mask]
        _run_paired(
            c0, c1, atlas,
            label="§3.6 China test-retest (close run 0 vs run 1)",
            out_stem="china_retest_tfnbs_v2",
            n_perm=n_perm, seed=args.seed,
        )


if __name__ == "__main__":
    main()
