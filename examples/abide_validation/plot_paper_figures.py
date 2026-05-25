import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

from conninfpy.harmonize import combat_harmonize
from conninfpy.glm_stats import build_design_matrix, compute_glm_stat

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True, parents=True)

def load_data():
    data = np.load(RESULTS_DIR / "abide_prepared.npz", allow_pickle=True)
    return {k: data[k] for k in data.files}

def draw_module_boundaries(ax, labels):
    boundaries = np.where(labels[:-1] != labels[1:])[0] + 0.5
    for b in boundaries:
        ax.axhline(b, color="black", linestyle="--", linewidth=0.5, alpha=0.3)
        ax.axvline(b, color="black", linestyle="--", linewidth=0.5, alpha=0.3)

def plot1_site_variance(data):
    print("Generating Comprehensive Diagnostic...")
    Y_raw = data["connectivity_z"]
    sites = data["site"]
    confounds = np.column_stack([data["age"].astype(float), data["sex"].astype(float), data["mean_fd"].astype(float)])
    net_labels = data["net_labels"]
    
    def get_resid(Y_flat, X):
        X_full = np.column_stack([np.ones(X.shape[0]), X])
        beta = np.linalg.pinv(X_full) @ Y_flat
        return Y_flat - X_full @ beta

    iu = np.triu_indices(Y_raw.shape[1], k=1)
    Y_raw_flat = Y_raw[:, iu[0], iu[1]]
    resid_raw = get_resid(Y_raw_flat, confounds)
    
    combat_out = combat_harmonize(Y_raw, sites, preserve=confounds)
    Y_harm_flat = combat_out.Y_adjusted[:, iu[0], iu[1]]
    resid_harm = get_resid(Y_harm_flat, confounds)
    
    def get_f_matrix(resid_flat):
        unique_sites = np.unique(sites)
        n_edges = resid_flat.shape[1]
        f_vec = np.zeros(n_edges)
        for e in range(n_edges):
            groups = [resid_flat[sites == s, e] for s in unique_sites]
            f, _ = stats.f_oneway(*groups)
            f_vec[e] = f
        mat = np.zeros((Y_raw.shape[1], Y_raw.shape[1]))
        mat[iu] = f_vec
        return mat + mat.T

    f_mat_before = get_f_matrix(resid_raw)
    f_mat_after = get_f_matrix(resid_harm)
    
    fig = plt.figure(figsize=(20, 10))
    ax_box = plt.subplot2grid((2, 3), (0, 0), colspan=3)
    df_box = pd.DataFrame({
        "Site": np.concatenate([sites, sites]),
        "Mean Residual FC": np.concatenate([np.mean(resid_raw, axis=1), np.mean(resid_harm, axis=1)]),
        "Status": ["Before Harmonization"] * len(sites) + ["After ComBat (Nuisance-Only)"] * len(sites)
    })
    sns.boxplot(data=df_box, x="Site", y="Mean Residual FC", hue="Status", palette="Set2", ax=ax_box)
    ax_box.set_title("Panel A: Global Mean FC Alignment across 15 sites", fontsize=14)
    ax_box.axhline(0, color="black", linestyle="--", alpha=0.3)
    ax_box.tick_params(axis='x', rotation=30)
    
    ax_f1 = plt.subplot2grid((2, 3), (1, 0))
    vmax = np.percentile(f_mat_before, 99)
    im1 = ax_f1.imshow(f_mat_before, cmap="YlOrRd", vmin=0, vmax=vmax)
    draw_module_boundaries(ax_f1, net_labels)
    ax_f1.set_title(f"Panel B: Site-Effect F-stats (BEFORE)\nMean F={np.mean(f_mat_before[iu]):.2f}", fontsize=12)
    plt.colorbar(im1, ax=ax_f1, label="F-statistic")
    
    ax_f2 = plt.subplot2grid((2, 3), (1, 1))
    im2 = ax_f2.imshow(f_mat_after, cmap="YlOrRd", vmin=0, vmax=vmax)
    draw_module_boundaries(ax_f2, net_labels)
    ax_f2.set_title(f"Panel C: Site-Effect F-stats (AFTER)\nMean F={np.mean(f_mat_after[iu]):.2f}", fontsize=12)
    plt.colorbar(im2, ax=ax_f2, label="F-statistic")

    plt.suptitle("Comprehensive Site-Effect Neutralization Diagnostic\nMethod: ComBat (Nuisance-only preservation)", fontsize=18, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(PLOTS_DIR / "plot1_combat_site_variance.png", dpi=150)
    plt.close()

def plot_diagnosis_methodology_quadrant(data):
    print("Generating Diagnosis Methodology Comparison (2x2)...")
    network_order = list(data["network_order"])
    n_nets = len(network_order)
    
    def aggregate(csv_path):
        if not csv_path.exists(): return np.zeros((n_nets, n_nets))
        df = pd.read_csv(csv_path); net_labels = data["net_labels"]
        mat = np.zeros((n_nets, n_nets))
        for _, row in df.iterrows():
            i, j = int(row["roi_i"]), int(row["roi_j"])
            ni, nj = net_labels[i], net_labels[j]
            mat[ni, nj] += 1
            if ni != nj: mat[nj, ni] += 1
        return mat

    paths = {
        "Naive (No Site Control)": RESULTS_DIR / "diagnosis/naive/dx_naive_edges.csv",
        "ComBat ONLY (No Dummies)": RESULTS_DIR / "diagnosis/combat_only/dx_combat_only_edges.csv",
        "Site Dummies ONLY (No ComBat)": RESULTS_DIR / "diagnosis/strategy_e/dx_strategy_e_edges.csv",
        "ComBat + Site Dummies (Strategy D)": RESULTS_DIR / "diagnosis/dx_strategy_d_edges.csv"
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten(); cmaps = ["Oranges", "Blues", "Purples", "Greens"]
    
    for i, (label, path) in enumerate(paths.items()):
        mat = aggregate(path); n_sig = int(mat.sum()/2 + np.diag(mat).sum()/2)
        sns.heatmap(mat, annot=True, fmt="g", cmap=cmaps[i], xticklabels=network_order, yticklabels=network_order, ax=axes[i])
        axes[i].set_title(f"{label}\n(n_sig={n_sig})")
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')

    plt.suptitle("The Evolution of Site-Aware Inference in ABIDE (TF-NBS: E=0.4, H=3.0)\nComBat: Strategy D (Nuisance-only)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(PLOTS_DIR / "plot6_diagnosis_methodology.png", dpi=150); plt.close()

def plot2_effect_size(data):
    print("Generating Effect Size Distribution...")
    Y = data["connectivity_z"]; group = data["group"].astype(float)
    confounds = np.column_stack([data["age"].astype(float), data["sex"].astype(float), data["mean_fd"].astype(float)])
    sites = data["site"]; combat_out = combat_harmonize(Y, sites, preserve=confounds); Y_harm = combat_out.Y_adjusted
    site_dummies = pd.get_dummies(sites, drop_first=True).values.astype(float)
    X, contrast = build_design_matrix(group, np.column_stack([confounds, site_dummies]))
    t_dict = compute_glm_stat(Y_harm, X, contrast, stat_type="tstat"); t_signed = t_dict["positive"] - t_dict["negative"]
    iu = np.triu_indices(Y.shape[1], k=1); t_vec = t_signed[iu[0], iu[1]]
    r_partial = t_vec / np.sqrt(t_vec**2 + (Y_harm.shape[0] - X.shape[1]))
    plt.figure(figsize=(10, 6)); sns.histplot(r_partial, bins=100, color="gray", alpha=0.5)
    plt.xlabel("Partial Pearson r (Effect Size)"); plt.ylabel("Edge Count")
    plt.title("Diagnosis Effect Size Distribution\nMethod: GLM t-stat | ComBat: Strategy D (Nuisance-only)")
    plt.tight_layout(); plt.savefig(PLOTS_DIR / "plot2_dx_effect_size_distribution.png", dpi=150); plt.close()

def plot3_network_block_mass(data):
    print("Generating Network-Level Block-Mass...")
    csv_file = RESULTS_DIR / "age_development" / "age_strategy_d_edges.csv"
    if not csv_file.exists(): return
    df = pd.read_csv(csv_file); network_order = list(data["network_order"]); n_nets = len(network_order)
    mat_pos = np.zeros((n_nets, n_nets)); mat_neg = np.zeros((n_nets, n_nets))
    for _, row in df.iterrows():
        n_i = network_order.index(row["roi_i_network"]); n_j = network_order.index(row["roi_j_network"])
        if row["tail"] == "positive": mat_pos[n_i, n_j] += 1; mat_pos[n_j, n_i] += 1
        else: mat_neg[n_i, n_j] += 1; mat_neg[n_j, n_i] += 1
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(mat_pos, annot=True, fmt="g", cmap="Reds", xticklabels=network_order, yticklabels=network_order, ax=axes[0])
    sns.heatmap(mat_neg, annot=True, fmt="g", cmap="Blues", xticklabels=network_order, yticklabels=network_order, ax=axes[1])
    for ax in axes: ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.suptitle("Network-Level Block-Mass (Age Control, TF-NBS: E=0.4, H=3.0)\nComBat: Strategy D (Nuisance-only)")
    plt.tight_layout(); plt.savefig(PLOTS_DIR / "plot3_network_block_mass.png", dpi=150); plt.close()

def plot4_method_sensitivity():
    print("Generating Method Sensitivity...")
    methods_dir = RESULTS_DIR / "age_development" / "methods"
    csv_files = list(methods_dir.glob("age_*_edges.csv"))
    if not csv_files: return
    method_names = []; edge_sets = {}; full_stats = {}
    for f in csv_files:
        name = f.stem.replace("age_", "").replace("_edges", ""); method_names.append(name)
        df = pd.read_csv(f); edges = set()
        for _, row in df.iterrows(): edges.add(tuple(sorted([int(row["roi_i"]), int(row["roi_j"])])))
        edge_sets[name] = edges
        npz_path = methods_dir / f"age_{name}.npz"
        if npz_path.exists():
            with np.load(npz_path) as d:
                if "positive" in d and "negative" in d:
                    pos = 1.0 - d["positive"]; neg = 1.0 - d["negative"]
                    iu = np.triu_indices(pos.shape[0], k=1); full_stats[name] = pos[iu] - neg[iu]
    pref = ["tstat", "nbs_2.0", "nbs_3.0", "tfnbs", "ni_tfnbs", "fbc_tfnbs", "cnbs"]
    method_names = [m for m in pref if m in method_names] + sorted([m for m in method_names if m not in pref])
    n = len(method_names); j_mat = np.zeros((n, n)); s_mat = np.zeros((n, n))
    for i, m1 in enumerate(method_names):
        for j, m2 in enumerate(method_names):
            s1, s2 = edge_sets[m1], edge_sets[m2]
            j_mat[i, j] = len(s1.intersection(s2)) / len(s1.union(s2)) if s1 or s2 else 1.0
            if m1 in full_stats and m2 in full_stats:
                s_mat[i, j], _ = stats.spearmanr(full_stats[m1], full_stats[m2])
            else: s_mat[i, j] = np.nan
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    sns.heatmap(j_mat, annot=True, fmt=".2f", xticklabels=method_names, yticklabels=method_names, ax=ax1)
    sns.heatmap(s_mat, annot=True, fmt=".2f", xticklabels=method_names, yticklabels=method_names, ax=ax2)
    plt.suptitle("Method Sensitivity & Convergence (Age Control, TF-NBS: E=0.4, H=3.0)\nComBat: Strategy D (Nuisance-only)")
    plt.tight_layout(); plt.savefig(PLOTS_DIR / "plot4_dual.png", dpi=150); plt.close()

if __name__ == "__main__":
    data = load_data()
    plot1_site_variance(data); plot2_effect_size(data); plot3_network_block_mass(data)
    plot4_method_sensitivity(); plot_diagnosis_methodology_quadrant(data)
