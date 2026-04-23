"""
ABIDE Analyses 1 + 2 under the Path-1 ComBat design (§3.1 in ValidationPlan.md).

Loads ``abide_harmonized.npz`` (produced by :mod:`harmonize`), which contains
the same fields as ``abide_prepared.npz`` plus ``connectivity_z_harm``.
Runs the two group-comparison analyses with site-variance already removed
from the features, so no site dummies in ``confounds=``.

Results land in ``results/combat/`` so they can be compared side-by-side
with the existing Fisher-z + FL-with-site-dummies results in ``results/``.

Usage
-----
    python examples/abide_validation/run_analyses_combat.py
    python examples/abide_validation/run_analyses_combat.py --quick
    python examples/abide_validation/run_analyses_combat.py --analysis 2
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

from conninfpy import compute_p_val, compute_p_val_glm


HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(HERE, "abide_harmonized.npz")
RESULTS_DIR = os.path.join(HERE, "results", "combat")


def load_data():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"{DATA_FILE} not found — run `python harmonize.py` first."
        )
    data = np.load(DATA_FILE, allow_pickle=True)
    return {k: data[k] for k in data.files}


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def report_significant(p_dict, alpha, label, net_labels=None, network_order=None):
    """Print number of significant edges per tail + top network pairs."""
    N = p_dict[list(p_dict.keys())[0]].shape[0]
    triu = np.triu_indices(N, k=1)
    for key in p_dict:
        sig = p_dict[key][triu] < alpha
        n_sig = int(sig.sum())
        min_p = float(p_dict[key][triu].min())
        print(f"  {label} [{key}]: {n_sig} / {len(triu[0])} edges at p<{alpha}, min_p={min_p:.4f}")
        if n_sig > 0 and net_labels is not None and network_order is not None:
            sig_i = triu[0][sig]; sig_j = triu[1][sig]
            net_counts = {}
            for ii, jj in zip(sig_i, sig_j):
                pair = tuple(sorted([network_order[net_labels[ii]], network_order[net_labels[jj]]]))
                net_counts[pair] = net_counts.get(pair, 0) + 1
            for (n1, n2), count in sorted(net_counts.items(), key=lambda x: -x[1])[:5]:
                prefix = "within" if n1 == n2 else "between"
                print(f"    {prefix} {n1}-{n2}: {count} edges")


# =============================================================================
# Analysis 1 (ComBat): Naive group comparison on harmonised features
# =============================================================================


def analysis_1_naive_combat(data, n_perm, random_state=42):
    print("\n" + "=" * 72)
    print("ANALYSIS 1 (ComBat): naive group comparison on harmonised features")
    print("=" * 72)

    conn = data["connectivity_z_harm"]          # ComBat-harmonised
    group = data["group"]

    hc = conn[group == 0]; asd = conn[group == 1]
    print(f"HC: {hc.shape[0]}, ASD: {asd.shape[0]}, features: {conn.shape[1]}×{conn.shape[2]}")

    t0 = time.time()
    p_naive = compute_p_val(
        hc, asd,
        test_type="two-sample", method="tfnbs",
        n_permutations=n_perm, e=0.4, h=3.0, n=10,
        use_mp=True, random_state=random_state,
    )
    print(f"Time: {time.time() - t0:.1f}s")

    report_significant(p_naive, 0.05, "Naive TFNBS (ComBat)",
                       data["net_labels"], data["network_order"])

    out = os.path.join(RESULTS_DIR, "analysis_1_naive_combat.npz")
    np.savez(out, **p_naive)
    print(f"Saved: {out}")
    return p_naive


# =============================================================================
# Analysis 2 (ComBat): GLM without site dummies
# =============================================================================


def analysis_2_glm_combat(data, n_perm, random_state=42):
    print("\n" + "=" * 72)
    print("ANALYSIS 2 (ComBat): GLM on harmonised features (no site dummies)")
    print("=" * 72)

    conn = data["connectivity_z_harm"]
    group = data["group"]
    age = data["age"]; sex = data["sex"]; mean_fd = data["mean_fd"]

    # No site dummies — ComBat harmonised site variance out of the features.
    confounds = np.column_stack([age, sex, mean_fd])
    print(f"Subjects: {len(group)}, confounds: {confounds.shape[1]} (age + sex + mean_fd)")

    t0 = time.time()
    p_glm = compute_p_val_glm(
        conn,
        interest=group, confounds=confounds,
        method="tfnbs", n_permutations=n_perm,
        e=0.4, h=3.0, n=10,
        use_mp=True, random_state=random_state,
    )
    print(f"Time: {time.time() - t0:.1f}s")

    report_significant(p_glm, 0.05, "GLM TFNBS (ComBat)",
                       data["net_labels"], data["network_order"])

    out = os.path.join(RESULTS_DIR, "analysis_2_glm_combat.npz")
    np.savez(out, **p_glm)
    print(f"Saved: {out}")
    return p_glm


# =============================================================================
# CLI
# =============================================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="500 permutations instead of 5000 (dev runs).")
    ap.add_argument("--analysis", choices=["1", "2", "all"], default="all")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    n_perm = 500 if args.quick else 5000
    print(f"Permutations: {n_perm}")

    ensure_results_dir()
    data = load_data()

    print(f"\nLoaded: {DATA_FILE}")
    print(f"  N subjects: {len(data['group'])}")
    print(f"  ComBat smooth_terms: {list(data['combat_smooth_terms'])}")
    print(f"  ComBat feature count: {data['combat_feature_count']}")
    print(
        f"  Between-site variance: raw={float(data['combat_between_site_var_raw']):.5f}, "
        f"harm={float(data['combat_between_site_var_harm']):.5f}"
    )

    if args.analysis in ("1", "all"):
        analysis_1_naive_combat(data, n_perm, random_state=args.seed)
    if args.analysis in ("2", "all"):
        analysis_2_glm_combat(data, n_perm, random_state=args.seed)


if __name__ == "__main__":
    main()
