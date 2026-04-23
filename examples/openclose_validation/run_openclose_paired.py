"""
ValidationPlan §3.5 + §3.6 — Open-Close paired TFNBS + China test-retest.

Uses the Schaefer-200 (182-ROI) GSR version of the open-close data loaded
via :class:`examples.openclose_validation.openclose_loader.OpenCloseDataset`.

Runs:
  - §3.5 IHB paired: open vs close (n=84)
  - §3.6 China paired: open vs close run-0 (n=48)
  - §3.6 China test-retest: close-run-0 vs close-run-1 (n=47, sub-3258811 excluded)

Outputs under :file:`examples/openclose_validation/results/` per experiment.

Time budget per call (with ``use_mp=True`` on Apple M5):
  IHB paired TFNBS, 182×182, 5000 perms ≈ 10–20 min
  China paired TFNBS, same shape, 5000 perms ≈ 6–12 min
  China test-retest, same ≈ 6–12 min

The three runs are independent — can be run in parallel if the host has
CPU headroom, but run sequentially here to keep parallelism inside
``compute_p_val``.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from conninfpy import compute_p_val
from examples.openclose_validation.openclose_loader import OpenCloseDataset


HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE / "results"


def _report(tag: str, p: dict, net_labels: np.ndarray, names: list, alpha: float = 0.05) -> None:
    N = p[list(p.keys())[0]].shape[0]
    iu = np.triu_indices(N, k=1)
    for k, v in p.items():
        vec = v[iu]
        n_sig = int((vec < alpha).sum())
        print(f"  {tag} [{k}]: n_sig={n_sig}/{len(vec)}  min_p={vec.min():.4f}")
        if n_sig > 0:
            sig = vec < alpha
            counts = {}
            for ii, jj in zip(iu[0][sig], iu[1][sig]):
                pair = tuple(sorted([names[net_labels[ii]], names[net_labels[jj]]]))
                counts[pair] = counts.get(pair, 0) + 1
            top = sorted(counts.items(), key=lambda x: -x[1])[:8]
            for (a, b), c in top:
                pref = "within" if a == b else "between"
                print(f"    {pref} {a:>22s} × {b:<22s}  {c}")


def _run_paired(g1: np.ndarray, g2: np.ndarray, label: str,
                n_perm: int, seed: int) -> dict:
    print(f"\n{'=' * 72}\n{label} — {g1.shape[0]} subjects, {g1.shape[1]}² edges\n{'=' * 72}")
    t0 = time.time()
    p = compute_p_val(
        g1, g2,
        test_type="paired", method="tfnbs",
        n_permutations=n_perm, e=0.4, h=3.0, n=10,
        use_mp=True, random_state=seed,
    )
    print(f"  time: {time.time() - t0:.1f}s")
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="500 perms for smoke testing (~30s each)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--analyses", nargs="+",
                    choices=["ihb", "china", "china_retest", "all"],
                    default=["all"])
    args = ap.parse_args()
    n_perm = 500 if args.quick else 5000
    targets = ["ihb", "china", "china_retest"] if "all" in args.analyses else args.analyses

    os.makedirs(OUT_ROOT, exist_ok=True)
    print(f"Output → {OUT_ROOT}")
    print(f"n_permutations = {n_perm}")

    # ---- §3.5 IHB paired open vs close ----------------------------
    if "ihb" in targets:
        ds_ihb = OpenCloseDataset.load("ihb")
        names = ds_ihb.get_label_names()
        o_z, c_z = ds_ihb.connectivity_z(run=0)
        p = _run_paired(o_z, c_z, "§3.5 IHB paired open vs close", n_perm, args.seed)
        _report("IHB", p, ds_ihb.net_labels, names)
        out = OUT_ROOT / "ihb_paired_tfnbs.npz"
        np.savez(out, **p)
        print(f"  saved: {out}")

    # ---- §3.6 China paired (run 0) --------------------------------
    if "china" in targets:
        ds_cn = OpenCloseDataset.load("china")
        names = ds_cn.get_label_names()
        o_z, c_z = ds_cn.connectivity_z(run=0)
        p = _run_paired(o_z, c_z, "§3.6 China paired (close run 0)", n_perm, args.seed)
        _report("China run0", p, ds_cn.net_labels, names)
        out = OUT_ROOT / "china_paired_tfnbs_run0.npz"
        np.savez(out, **p)
        print(f"  saved: {out}")

    # ---- §3.6 China test-retest: close run 0 vs run 1 ------------
    if "china_retest" in targets:
        ds_cn = OpenCloseDataset.load("china")
        names = ds_cn.get_label_names()
        mask = ds_cn.retest_subject_mask()
        print(f"\n§3.6 China test-retest — excluding {(~mask).sum()} subject(s): "
              f"{[s for s, m in zip(ds_cn.subject_ids, mask) if not m]}")
        # Fisher-z on both runs, then restrict to valid subjects
        _, c0 = ds_cn.connectivity_z(run=0)
        _, c1 = ds_cn.connectivity_z(run=1)
        c0 = c0[mask]
        c1 = c1[mask]
        p = _run_paired(c0, c1, "§3.6 China test-retest (close run 0 vs run 1)", n_perm, args.seed)
        _report("China retest", p, ds_cn.net_labels, names)
        out = OUT_ROOT / "china_retest_tfnbs.npz"
        np.savez(out, **p)
        print(f"  saved: {out}")


if __name__ == "__main__":
    main()
