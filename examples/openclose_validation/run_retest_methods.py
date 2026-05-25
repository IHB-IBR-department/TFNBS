"""
Per-method run of the China close-run0 vs close-run1 paired contrast.

Counterbalanced ordering across subjects ⇒ this contrast acts as a
matched-design empirical null for the open-vs-close state contrast.
Running every method in the SELECTORS bundle on this contrast lets us
build a specificity panel (state vs empirical-null per-method edge
counts at FWER α = 0.05).

The state-contrast pmaps already live in ``results/ml/pmap_china_*.npz``;
this script writes the matching retest pmaps as
``results/ml/pmap_china_retest_{method}.npz`` so the downstream plot
script can pair them up by method name.

Subject mask: ``ds.retest_subject_mask()`` (excludes ``sub-3258811``
whose run-3 is incomplete; see openclose_loader docstring).

Runtime: ~1–2 min per perm-based method × 6 methods ≈ 10 min total on
M5 with use_mp=True at n_permutations=1000.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from conninfpy import compute_p_val
from examples.openclose_validation.openclose_loader import OpenCloseDataset


HERE = Path(__file__).resolve().parent
ML_DIR = HERE / "results" / "ml"

# Matches examples/openclose_validation/ml/run_ml_feature_selection.py
SELECTORS = {
    "tstat":    dict(method="tstat"),
    "tfnbs":    dict(method="tfnbs", e=0.4, h=3.0, n=10),
    "nbs_20":   dict(method="nbs", threshold=2.0),
    "nbs_30":   dict(method="nbs", threshold=3.0),
    "cnbs":     dict(method="cnbs"),
    "ni_tfnbs": dict(method="ni_tfnbs", e=0.4, h=3.0, n=10),
    "fbc_tfnbs":dict(method="fbc_tfnbs"),
    "bh_fdr":   dict(method="bh_fdr"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-permutations", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--methods", nargs="+", default=list(SELECTORS.keys()))
    args = ap.parse_args()

    ML_DIR.mkdir(parents=True, exist_ok=True)

    ds = OpenCloseDataset.load("china")
    mask = ds.retest_subject_mask()
    excluded = [s for s, m in zip(ds.subject_ids, mask) if not m]
    print(f"China retest — excluding {(~mask).sum()} subject(s): {excluded}")
    _, c0 = ds.connectivity_z(run=0)
    _, c1 = ds.connectivity_z(run=1)
    c0 = c0[mask]
    c1 = c1[mask]
    print(f"n_paired={c0.shape[0]}, N={c0.shape[1]}, n_perm={args.n_permutations}")
    net_labels = ds.net_labels

    for name in args.methods:
        if name not in SELECTORS:
            print(f"skip unknown method: {name}")
            continue
        kwargs = dict(SELECTORS[name])
        if kwargs["method"] in ("cnbs", "ni_tfnbs", "fbc_tfnbs"):
            kwargs["net_labels"] = net_labels

        t0 = time.time()
        p = compute_p_val(
            c0, c1,
            test_type="paired",
            n_permutations=args.n_permutations,
            use_mp=True,
            random_state=args.seed,
            **kwargs,
        )
        dt = time.time() - t0
        keys = list(p.keys())
        iu = np.triu_indices(c0.shape[1], k=1)
        n_sig_a = int((p[keys[0]][iu] < 0.05).sum())
        n_sig_b = int((p[keys[1]][iu] < 0.05).sum())
        print(f"  {name:10s}  {dt:5.1f}s  "
              f"n_sig(α=0.05) {keys[0]}={n_sig_a}, {keys[1]}={n_sig_b}")

        out_path = ML_DIR / f"pmap_china_retest_{name}.npz"
        np.savez(out_path, **{k: p[k] for k in p})
        print(f"    saved: {out_path}")


if __name__ == "__main__":
    main()
