"""
ABIDE validation analyses for ConnInfPy.

Runs the analyses from ABIDEValidation.md:
1. Naive group comparison (no confounds)
2. GLM group comparison (with age, sex, motion, site confounds)
3. ADOS severity regression (ASD only)
4. Method comparison
5. Acceleration validation (GPD vs full permutation)

Usage:
    python examples/abide_validation/run_analyses.py
    python examples/abide_validation/run_analyses.py --quick    # fewer perms
    python examples/abide_validation/run_analyses.py --analysis 2  # single analysis
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

from conninfpy import compute_p_val, compute_p_val_glm, fisher_r_to_z

# =============================================================================
# Load prepared data
# =============================================================================

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "abide_prepared.npz")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load_data():
    data = np.load(DATA_FILE, allow_pickle=True)
    return {k: data[k] for k in data.files}


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def report_significant(p_dict, alpha, label, net_labels=None, network_order=None):
    """Report significant edges and network breakdown."""
    N = p_dict[list(p_dict.keys())[0]].shape[0]
    triu = np.triu_indices(N, k=1)

    for key in p_dict:
        sig = p_dict[key][triu] < alpha
        n_sig = np.sum(sig)
        print(f"  {label} [{key}]: {n_sig} / {len(triu[0])} edges (p < {alpha})")

        if n_sig > 0 and net_labels is not None and network_order is not None:
            # Network breakdown
            sig_i = triu[0][sig]
            sig_j = triu[1][sig]
            net_counts = {}
            for ii, jj in zip(sig_i, sig_j):
                ni = network_order[net_labels[ii]]
                nj = network_order[net_labels[jj]]
                pair = tuple(sorted([ni, nj]))
                net_counts[pair] = net_counts.get(pair, 0) + 1

            top = sorted(net_counts.items(), key=lambda x: -x[1])[:10]
            for (n1, n2), count in top:
                prefix = "within" if n1 == n2 else "between"
                print(f"    {prefix} {n1}-{n2}: {count} edges")


# =============================================================================
# Analysis 1: Naive group comparison
# =============================================================================

def analysis_1_naive(data, n_perm, random_state=42):
    """Group comparison WITHOUT confounds."""
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Naive group comparison (no confounds)")
    print("=" * 70)

    conn_z = data["connectivity_z"]
    group = data["group"]

    hc = conn_z[group == 0]
    asd = conn_z[group == 1]
    print(f"HC: {hc.shape[0]}, ASD: {asd.shape[0]}")

    t0 = time.time()
    p_naive = compute_p_val(
        hc, asd,
        test_type="two-sample",
        method="tfnbs",
        n_permutations=n_perm,
        e=0.4, h=3.0, n=10,
        use_mp=True,
        random_state=random_state,
    )
    elapsed = time.time() - t0
    print(f"Time: {elapsed:.1f}s")

    report_significant(p_naive, 0.05, "Naive TFNBS",
                       data["net_labels"], data["network_order"])

    np.savez(os.path.join(RESULTS_DIR, "analysis_1_naive.npz"), **p_naive)
    return p_naive


# =============================================================================
# Analysis 2: GLM with confounds
# =============================================================================

def analysis_2_glm(data, n_perm, random_state=42):
    """Group comparison WITH age, sex, motion, site confounds."""
    print("\n" + "=" * 70)
    print("ANALYSIS 2: GLM group comparison (with confounds)")
    print("=" * 70)

    conn_z = data["connectivity_z"]
    group = data["group"]
    age = data["age"]
    sex = data["sex"]
    mean_fd = data["mean_fd"]
    site_dummies = data["site_dummies"]

    confounds = np.column_stack([age, sex, mean_fd, site_dummies])
    print(f"Subjects: {len(group)}, Confounds: {confounds.shape[1]} "
          f"(age + sex + motion + {site_dummies.shape[1]} site dummies)")

    t0 = time.time()
    p_glm = compute_p_val_glm(
        conn_z,
        interest=group,
        confounds=confounds,
        method="tfnbs",
        n_permutations=n_perm,
        e=0.4, h=3.0, n=10,
        use_mp=True,
        random_state=random_state,
    )
    elapsed = time.time() - t0
    print(f"Time: {elapsed:.1f}s")

    report_significant(p_glm, 0.05, "GLM TFNBS",
                       data["net_labels"], data["network_order"])

    np.savez(os.path.join(RESULTS_DIR, "analysis_2_glm.npz"), **p_glm)
    return p_glm


# =============================================================================
# Analysis 3: ADOS severity regression (ASD only)
# =============================================================================

def analysis_3_severity(data, n_perm, random_state=42):
    """ADOS Total regression within ASD subjects."""
    print("\n" + "=" * 70)
    print("ANALYSIS 3: ADOS severity regression (ASD only)")
    print("=" * 70)

    conn_z = data["connectivity_z"]
    group = data["group"]
    age = data["age"]
    sex = data["sex"]
    mean_fd = data["mean_fd"]
    ados_total = data["ados_total"]
    ados_module = data["ados_module"]
    site_dummies = data["site_dummies"]

    # ASD subjects with valid ADOS Total
    asd_mask = (group == 1) & (~np.isnan(ados_total))
    asd_conn = conn_z[asd_mask]
    asd_ados = ados_total[asd_mask]
    asd_age = age[asd_mask]
    asd_sex = sex[asd_mask]
    asd_fd = mean_fd[asd_mask]
    asd_site = data["site"][asd_mask]
    asd_module = ados_module[asd_mask]

    # Recompute site dummies within ASD subset (avoid singular columns)
    asd_site_dummies = pd.get_dummies(asd_site, drop_first=True).values.astype(float)
    # Drop columns with zero variance (sites with only 1 subject)
    col_var = asd_site_dummies.var(axis=0)
    asd_site_dummies = asd_site_dummies[:, col_var > 0]

    # ADOS module dummies (control for module effect — Torres 2020)
    asd_module_clean = np.where(np.isnan(asd_module), -1, asd_module)
    unique_modules = np.unique(asd_module_clean[asd_module_clean >= 0])
    confound_list = [asd_age, asd_sex, asd_fd, asd_site_dummies]

    if len(unique_modules) > 1:
        module_dummies = pd.get_dummies(
            asd_module_clean, drop_first=True
        ).values.astype(float)
        # Remove the -1 (missing) dummy if present
        mod_var = module_dummies.var(axis=0)
        module_dummies = module_dummies[:, mod_var > 0]
        if module_dummies.shape[1] > 0:
            confound_list.append(module_dummies)

    confounds = np.column_stack(confound_list)

    print(f"ASD with ADOS Total: {len(asd_ados)}")
    print(f"ADOS range: [{asd_ados.min():.0f}, {asd_ados.max():.0f}], "
          f"mean={asd_ados.mean():.1f}")
    print(f"Confounds: {confounds.shape[1]}")

    t0 = time.time()
    p_severity = compute_p_val_glm(
        asd_conn,
        interest=asd_ados,
        confounds=confounds,
        method="tfnbs",
        n_permutations=n_perm,
        e=0.4, h=3.0, n=10,
        use_mp=True,
        random_state=random_state,
    )
    elapsed = time.time() - t0
    print(f"Time: {elapsed:.1f}s")

    report_significant(p_severity, 0.05, "ADOS severity",
                       data["net_labels"], data["network_order"])

    np.savez(os.path.join(RESULTS_DIR, "analysis_3_severity.npz"), **p_severity)
    return p_severity


# =============================================================================
# Analysis 4: Method comparison
# =============================================================================

def analysis_4_methods(data, n_perm, random_state=42):
    """Compare all enhancement methods on the same data (Analysis 2 design)."""
    print("\n" + "=" * 70)
    print("ANALYSIS 4: Method comparison")
    print("=" * 70)

    conn_z = data["connectivity_z"]
    group = data["group"]
    age = data["age"]
    sex = data["sex"]
    mean_fd = data["mean_fd"]
    site_dummies = data["site_dummies"]
    net_labels = data["net_labels"]

    confounds = np.column_stack([age, sex, mean_fd, site_dummies])

    methods = [
        ("tstat", {}),
        ("tfnbs", {"e": 0.4, "h": 3.0, "n": 10}),
        ("nbs", {"threshold": 2.0}),
        ("nbs", {"threshold": 3.0}),
        ("cnbs", {"net_labels": net_labels}),
        ("ni_tfnbs", {"net_labels": net_labels, "e": 0.4, "h": 3.0, "n": 10}),
        ("fbc_tfnbs", {"net_labels": net_labels, "e": 0.4, "h": 3.0, "n": 10,
                       "min_cluster_size": 3}),
    ]

    results = {}
    N = conn_z.shape[1]
    triu = np.triu_indices(N, k=1)

    for method_name, method_kwargs in methods:
        label = method_name
        if method_name == "nbs":
            label = f"nbs_t{method_kwargs['threshold']}"

        print(f"\n  Running {label}...")
        t0 = time.time()
        p = compute_p_val_glm(
            conn_z,
            interest=group,
            confounds=confounds,
            method=method_name,
            n_permutations=n_perm,
            use_mp=True,
            random_state=random_state,
            **method_kwargs,
        )
        elapsed = time.time() - t0

        n_pos = np.sum(p["positive"][triu] < 0.05)
        n_neg = np.sum(p["negative"][triu] < 0.05)
        print(f"  {label}: {n_pos} pos + {n_neg} neg significant "
              f"(p<0.05), {elapsed:.1f}s")

        results[label] = p
        np.savez(os.path.join(RESULTS_DIR, f"analysis_4_{label}.npz"), **p)

    # Overlap matrix
    print("\n  Jaccard overlap (positive tail):")
    method_names = list(results.keys())
    for i, m1 in enumerate(method_names):
        for j, m2 in enumerate(method_names):
            if j <= i:
                continue
            sig1 = results[m1]["positive"][triu] < 0.05
            sig2 = results[m2]["positive"][triu] < 0.05
            intersection = np.sum(sig1 & sig2)
            union = np.sum(sig1 | sig2)
            jaccard = intersection / union if union > 0 else 0
            if intersection > 0:
                print(f"    {m1} vs {m2}: J={jaccard:.3f} "
                      f"({intersection}/{union})")

    return results


# =============================================================================
# Analysis 5: Acceleration validation
# =============================================================================

def analysis_5_acceleration(data, random_state=42):
    """Compare GPD acceleration vs full permutation."""
    print("\n" + "=" * 70)
    print("ANALYSIS 5: Acceleration validation")
    print("=" * 70)

    conn_z = data["connectivity_z"]
    group = data["group"]
    age = data["age"]
    sex = data["sex"]
    mean_fd = data["mean_fd"]
    site_dummies = data["site_dummies"]

    confounds = np.column_stack([age, sex, mean_fd, site_dummies])
    N = conn_z.shape[1]
    triu = np.triu_indices(N, k=1)

    configs = [
        ("ref_5000", 5000, None),
        ("gpd_200", 200, "gpd"),
        ("gamma_200", 200, "gamma"),
        ("emp_200", 200, None),
    ]

    results = {}
    for label, n_perm, accel in configs:
        print(f"\n  Running {label} ({n_perm} perms, accel={accel})...")
        t0 = time.time()
        p = compute_p_val_glm(
            conn_z,
            interest=group,
            confounds=confounds,
            method="tstat",
            n_permutations=n_perm,
            acceleration=accel,
            use_mp=True,
            random_state=random_state,
        )
        elapsed = time.time() - t0

        n_pos = np.sum(p["positive"][triu] < 0.05)
        print(f"    {label}: {n_pos} pos significant, {elapsed:.1f}s")
        results[label] = p

    # Compare with reference
    ref = results["ref_5000"]
    for label in ["gpd_200", "gamma_200", "emp_200"]:
        p_test = results[label]
        for key in ["positive", "negative"]:
            r = np.corrcoef(ref[key][triu], p_test[key][triu])[0, 1]
            mae = np.mean(np.abs(ref[key][triu] - p_test[key][triu]))
            # Agreement on significance
            sig_ref = ref[key][triu] < 0.05
            sig_test = p_test[key][triu] < 0.05
            agree = np.mean(sig_ref == sig_test)
            print(f"    {label} [{key}]: r={r:.4f}, MAE={mae:.4f}, "
                  f"agree={100*agree:.1f}%")

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="ABIDE validation analyses")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run (200 perms)")
    parser.add_argument("--analysis", type=int, default=0,
                        help="Run single analysis (1-5), 0=all")
    args = parser.parse_args()

    n_perm = 200 if args.quick else 5000

    print(f"ABIDE Validation — ConnInfPy")
    print(f"Permutations: {n_perm}")
    print(f"Data: {DATA_FILE}")

    data = load_data()
    ensure_results_dir()

    if args.analysis == 0 or args.analysis == 1:
        analysis_1_naive(data, n_perm)

    if args.analysis == 0 or args.analysis == 2:
        analysis_2_glm(data, n_perm)

    if args.analysis == 0 or args.analysis == 3:
        analysis_3_severity(data, n_perm)

    if args.analysis == 0 or args.analysis == 4:
        analysis_4_methods(data, n_perm)

    if args.analysis == 0 or args.analysis == 5:
        analysis_5_acceleration(data)

    print("\n" + "=" * 70)
    print("All analyses complete. Results saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
