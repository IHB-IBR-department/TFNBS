"""
Regenerate all simulation figures for the paper.

This script pulls from results in examples/simulation_validation/results
(aggregated by analysis type) and produces publication-ready figures.

Figures generated:
1.  FWER Calibration (Clopper-Pearson interval box plot)
2.  TPR/FDR Heatmap (Topologies vs Methods)
3.  Effect Size Sweep (TPR curves)
4.  TFNBS (E,H) Sweep (TPR/FDR heatmaps)
5.  Prior Misspecification Impact

GLM figures are intentionally skipped until the GLM validation run is
regenerated.

Usage:
    python examples/simulation_validation/regenerate_paper_figures.py
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from fpr.plot_fpr_results import plot_paper_fwer_calibration
from power.tfnbs_eh_sweep import plot_metric_rows
from topology_gallery import plot_topology_gallery

# Constants & Styling
RESULTS_DIR = Path("examples/simulation_validation/results")
OUTPUT_DIR = RESULTS_DIR / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "figure.titlesize": 11,
})

METHOD_ORDER = [
    "tstat", "bonferroni", "bh_fdr",
    "nbs_extent_2.0", "nbs_intensity_2.0",
    "nbs_extent_3.0", "nbs_intensity_3.0",
    "cnbs", "tfnbs", "ni_tfnbs", "fbc_tfnbs",
]
METHOD_LABELS = {
    "tstat": "T-stat (uncorr)",
    "bonferroni": "Bonferroni",
    "bh_fdr": "BH-FDR",
    "nbs_extent": r"NBS Ext ($\tau=2$)",
    "nbs_intensity": r"NBS Int ($\tau=2$)",
    "nbs_extent_2.0": r"NBS Ext ($\tau=2$)",
    "nbs_intensity_2.0": r"NBS Int ($\tau=2$)",
    "nbs_extent_3.0": r"NBS Ext ($\tau=3$)",
    "nbs_intensity_3.0": r"NBS Int ($\tau=3$)",
    "cnbs": "cNBS",
    "tfnbs": r"TFNBS ($E=.4, H=3$)",
    "ni_tfnbs": r"NI-TFNBS ($E=.4, H=3$)",
    "fbc_tfnbs": r"FBC-TFNBS ($E=.4, H=3$)",
}

GENERATE_GLM_FIGURES = False
PAPER_TFNBS_E = 0.4
PAPER_TFNBS_H = 3.0
TFNBS_METHODS = {"tfnbs", "ni_tfnbs", "fbc_tfnbs"}

TOPOLOGY_ORDER = [
    "hub", "rich_club", "chain", "within_module_dense", "between_modules_dense",
    "within_plus_between",
    "partial_bipartite_between_modules", "gradient_core_periphery_within_module",
    "scattered_cross_block", "cross_block_connected_chain", "fragmented_within_module"
]
TOPOLOGY_LABELS = {
    "hub": "Hub",
    "rich_club": "Rich-club",
    "chain": "Chain",
    "within_module_dense": "Within-module dense",
    "within_plus_between": "Within+Between",
    "between_modules_dense": "Between-module dense",
    "partial_bipartite_between_modules": "Partial bipartite",
    "gradient_core_periphery_within_module": "Core-periphery",
    "scattered_cross_block": "Scattered cross-block",
    "cross_block_connected_chain": "Cross-block chain",
    "fragmented_within_module": "Fragmented",
}

# =============================================================================
# FIGURE 1: FWER CALIBRATION
# =============================================================================

def plot_fwer_calibration():
    """Plot FWER calibration via the reusable FPR plotting module."""
    # Specifically use the folder containing the full run
    res_dir = RESULTS_DIR / "fpr_calibration"
    
    if not (res_dir / "fpr_results.csv").exists():
        print(f"Skipping FWER plot: {res_dir / 'fpr_results.csv'} not found.")
        return

    print(f"Using calibration results from: {res_dir}")
    plot_paper_fwer_calibration(
        res_dir,
        OUTPUT_DIR,
        filename="fig1_fwer_calibration.png",
        method_order=METHOD_ORDER,
        method_labels=METHOD_LABELS,
    )

# =============================================================================
# FIGURE 2: TOPOLOGICAL POWER (HEATMAP)
# =============================================================================

def _parse_method_param(method_params: object, key: str) -> float | None:
    """Extract a numeric method parameter from strings like 'e=0.4,h=3.0'."""
    if not isinstance(method_params, str):
        return None
    match = re.search(rf"\b{key}=([\d.]+)", method_params)
    return float(match.group(1)) if match else None


def _filter_paper_tfnbs_params(df: pd.DataFrame) -> pd.DataFrame:
    """Keep TFNBS-family rows at the paper E/H setting and all other methods."""
    df = df.copy()
    is_tfnbs = df["method"].isin(TFNBS_METHODS)
    e_vals = pd.to_numeric(
        df["method_params"].map(lambda value: _parse_method_param(value, "e")),
        errors="coerce",
    )
    h_vals = pd.to_numeric(
        df["method_params"].map(lambda value: _parse_method_param(value, "h")),
        errors="coerce",
    )
    is_paper_eh = np.isclose(e_vals, PAPER_TFNBS_E, equal_nan=False) & np.isclose(
        h_vals,
        PAPER_TFNBS_H,
        equal_nan=False,
    )
    filtered = df[~is_tfnbs | is_paper_eh]
    missing = sorted(TFNBS_METHODS.difference(filtered.loc[is_tfnbs & is_paper_eh, "method"].unique()))
    if missing:
        print(f"Warning: missing TFNBS paper-parameter rows for: {missing}")
    return filtered


def plot_power_heatmap(effect_size: float = 0.25):
    """Plot TPR and FDR heatmap for a specific effect size."""
    candidates = [
        RESULTS_DIR / "power" / "power_analysis_1000_30",
        RESULTS_DIR / "power_analysis_1000_30",
    ]
    res_dir = next((path for path in candidates if (path / "power_vs_effect_size.csv").exists()), None)
    if res_dir is None:
        print("Skipping Power Heatmap: no power_vs_effect_size.csv found.")
        return

    print(f"Using power results from: {res_dir}")
    df_path = res_dir / "power_vs_effect_size.csv"
    df = pd.read_csv(df_path)
    
    # Try requested ES, if not there pick closest or middle
    es_vals = df["effect_size"].unique()
    if effect_size not in es_vals:
        effect_size = es_vals[len(es_vals)//2]
        
    sub = df[np.isclose(df["effect_size"], effect_size)]
    sub = _filter_paper_tfnbs_params(sub)
    
    if sub.empty:
        print(f"No results for effect_size={effect_size}. Available: {es_vals}")
        return

    # Filter methods to those in METHOD_ORDER and present in data
    plot_methods = [m for m in METHOD_ORDER if m in sub["method"].values]

    # Pivot for TPR
    tpr_pivot = sub.pivot_table(index="scenario", columns="method", values="TPR", aggfunc="mean")
    tpr_pivot = tpr_pivot.reindex(index=TOPOLOGY_ORDER, columns=plot_methods).dropna(axis=0, how='all').dropna(axis=1, how='all')
    tpr_pivot.index = [TOPOLOGY_LABELS.get(i, i) for i in tpr_pivot.index]
    tpr_pivot.columns = [METHOD_LABELS.get(c, c) for c in tpr_pivot.columns]

    # Pivot for FDR
    fdr_pivot = sub.pivot_table(index="scenario", columns="method", values="FDR", aggfunc="mean")
    fdr_pivot = fdr_pivot.reindex(index=TOPOLOGY_ORDER, columns=plot_methods).dropna(axis=0, how='all').dropna(axis=1, how='all')
    fdr_pivot.index = [TOPOLOGY_LABELS.get(i, i) for i in fdr_pivot.index]
    fdr_pivot.columns = [METHOD_LABELS.get(c, c) for c in fdr_pivot.columns]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 11), sharex=True)
    
    sns.heatmap(tpr_pivot, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax1, vmin=0, vmax=1, cbar_kws={'label': 'TPR'})
    ax1.set_title(f"True Positive Rate (TPR) @ d={effect_size}", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Scenario")
    
    sns.heatmap(fdr_pivot, annot=True, fmt=".2f", cmap="RdYlGn_r", ax=ax2, vmin=0, vmax=0.2, cbar_kws={'label': 'FDR'})
    ax2.set_title(f"False Discovery Rate (FDR) @ d={effect_size}", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Scenario")
    
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / f"fig2_power_heatmap_d{effect_size}.png", dpi=300, bbox_inches="tight")
    print(f"Saved Fig 2 (2-row layout) to {OUTPUT_DIR}")
    plt.close(fig)

# =============================================================================
# FIGURE 3: EFFECT SIZE SWEEP
# =============================================================================

def plot_effect_size_sweep():
    """Plot TPR vs Effect Size for all methods across topologies."""
    candidates = [
        RESULTS_DIR / "power" / "power_analysis_1000_30",
        RESULTS_DIR / "power_analysis_1000_30",
    ]
    res_dir = next((path for path in candidates if (path / "power_vs_effect_size.csv").exists()), None)
    if res_dir is None:
        return

    df_path = res_dir / "power_vs_effect_size.csv"
    df = pd.read_csv(df_path)
    
    # Identify topologies present in data
    topos_in_data = [t for t in TOPOLOGY_ORDER if t in df["scenario"].values]
    if not topos_in_data:
        return
        
    n_topo = len(topos_in_data)
    n_cols = 5
    n_rows = (n_topo + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3.5), sharey=True)
    axes_flat = axes.flatten() if n_topo > 1 else [axes]
    
    for i, topo in enumerate(topos_in_data):
        ax = axes_flat[i]
        sub = df[df["scenario"] == topo]
        
        for method in METHOD_ORDER:
            m_sub = sub[sub["method"] == method]
            if m_sub.empty:
                continue
            
            # Group by effect size and aggregate
            summary = m_sub.groupby("effect_size")["TPR"].agg(["mean", "std", "count"]).reset_index()
            summary["se"] = summary["std"] / np.sqrt(summary["count"])
            
            ax.plot(summary["effect_size"], summary["mean"], label=METHOD_LABELS.get(method, method), marker='o', markersize=3)
            ax.fill_between(summary["effect_size"], summary["mean"] - 1.96*summary["se"], 
                            summary["mean"] + 1.96*summary["se"], alpha=0.1)

        ax.set_title(TOPOLOGY_LABELS.get(topo, topo), fontsize=9)
        ax.set_xlabel("Effect Size (Cohen's d)")
        if i % n_cols == 0:
            ax.set_ylabel("True Positive Rate (TPR)")
        ax.grid(True, alpha=0.3)
    
    # Common legend
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.15, 0.5), fontsize=8)
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig3_effect_size_sweep.png", dpi=300, bbox_inches="tight")
    print(f"Saved Fig 3 to {OUTPUT_DIR}")
    plt.close(fig)

# =============================================================================
# FIGURE 4: TFNBS (E,H) SWEEP
# =============================================================================

def plot_tfnbs_eh_sweep(effect_size: float = 0.25):
    """Plot TFNBS TPR/FDR over the (E,H) grid for first-row topologies."""
    summary_path = (
        RESULTS_DIR
        / "power"
        / "power_analysis_1000_30"
        / "derived"
        / "tfnbs_eh_sweep_effect_size.csv"
    )
    if not summary_path.exists():
        print(f"Skipping TFNBS E/H sweep: {summary_path} not found.")
        return

    plot_metric_rows(
        summary_path=summary_path,
        output_path=OUTPUT_DIR / "fig4_tfnbs_eh_tpr_fdr.png",
        selector_col="effect_size",
        selector_value=effect_size,
        method="tfnbs",
        metrics=["TPR", "FDR"],
        scenarios=TOPOLOGY_ORDER[:5],
        paper_e=PAPER_TFNBS_E,
        paper_h=PAPER_TFNBS_H,
        mark_paper_choice=True,
        dpi=300,
    )


# =============================================================================
# AUXILIARY: SENSITIVITY (MMIN)
# =============================================================================

def plot_mmin_sensitivity():
    """Plot TPR/FDR vs m_min from sensitivity results."""
    # Find mmin directories
    SENS_DIR = RESULTS_DIR / "sensitivity" / "mmin"
    mmin_dirs = sorted(SENS_DIR.glob("mmin_*_1000"))
    if not mmin_dirs:
        mmin_dirs = sorted(SENS_DIR.glob("mmin_*"))
    
    if not mmin_dirs:
        print(f"Skipping mmin sensitivity plot: no results found in {SENS_DIR}")
        return

    all_data = []
    for d in mmin_dirs:
        try:
            mmin_val = int(d.name.split('_')[1])
        except (IndexError, ValueError):
            continue
            
        df_path = d / "power_vs_effect_size.csv"
        if df_path.exists():
            df = pd.read_csv(df_path)
            df["m_min"] = mmin_val
            all_data.append(df)
    
    if not all_data:
        return
        
    df_all = pd.concat(all_data)
    # Focus on FBC-TFNBS
    df_fbc = df_all[df_all["method"] == "fbc_tfnbs"]
    
    # Representative effect size
    es_vals = df_fbc["effect_size"].unique()
    if len(es_vals) > 0:
        es = es_vals[len(es_vals)//2] # Middle one
    else:
        return
        
    sub = df_fbc[np.isclose(df_fbc["effect_size"], es)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    for topo in TOPOLOGY_ORDER:
        t_sub = sub[sub["scenario"] == topo]
        if t_sub.empty:
            continue
        summary = t_sub.groupby("m_min")["TPR"].agg(["mean", "std", "count"]).reset_index()
        summary["se"] = summary["std"] / np.sqrt(summary["count"])
        ax1.errorbar(summary["m_min"], summary["mean"], yerr=1.96*summary["se"], label=TOPOLOGY_LABELS.get(topo, topo), marker='o')
        
        summary_fdr = t_sub.groupby("m_min")["FDR"].agg(["mean", "std", "count"]).reset_index()
        summary_fdr["se"] = summary_fdr["std"] / np.sqrt(summary_fdr["count"])
        ax2.errorbar(summary_fdr["m_min"], summary_fdr["mean"], yerr=1.96*summary_fdr["se"], label=TOPOLOGY_LABELS.get(topo, topo), marker='s')

    ax1.set_title(f"TPR vs $m_{{min}}$ (FBC-TFNBS, d={es:.2f})")
    ax1.set_xlabel("$m_{min}$")
    ax1.set_ylabel("TPR")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.set_title(f"FDR vs $m_{{min}}$ (FBC-TFNBS, d={es:.2f})")
    ax2.set_xlabel("$m_{min}$")
    ax2.set_ylabel("FDR")
    ax2.axhline(0.05, color='red', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "mmin_sensitivity.png", dpi=300, bbox_inches="tight")
    print(f"Saved m_min sensitivity to {OUTPUT_DIR}")
    plt.close(fig)

# =============================================================================
# FIGURE 5: PRIOR MISSPECIFICATION
# =============================================================================

def plot_prior_misspecification():
    """Plot impact of prior label misspecification on constrained methods."""
    conditions = {
        "true": "True (4 blocks)",
        "7blocks": "Over-segmented (7 blocks)",
        "2blocks": "Under-segmented (2 blocks)"
    }
    
    SENS_DIR = RESULTS_DIR / "sensitivity" / "prior_misspec"
    all_data = []
    for cond_id, label in conditions.items():
        res_dir = SENS_DIR / f"prior_misspec_{cond_id}_1000"
        if not res_dir.exists():
            res_dir = SENS_DIR / f"prior_misspec_{cond_id}"
        
        if res_dir.exists():
            df_path = res_dir / "power_vs_effect_size.csv"
            if df_path.exists():
                df = pd.read_csv(df_path)
                df["Condition"] = label
                all_data.append(df)
                
    if not all_data:
        print(f"Skipping prior misspecification plot: no results found in {SENS_DIR}")
        return
        
    df_all = pd.concat(all_data)
    # Focus on constrained methods vs TFNBS (baseline)
    methods = ["cnbs", "ni_tfnbs", "fbc_tfnbs", "tfnbs"]
    df_sub = df_all[df_all["method"].isin(methods)]
    
    # Average across topologies for a clear summary
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # TPR plot
    sns.barplot(data=df_sub, x="method", y="TPR", hue="Condition", ax=axes[0], order=methods)
    axes[0].set_title("Sensitivity to Prior Misspecification (TPR)")
    axes[0].set_xticks(np.arange(len(methods)))
    axes[0].set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], rotation=45, ha="right")
    
    # FDR plot
    sns.barplot(data=df_sub, x="method", y="FDR", hue="Condition", ax=axes[1], order=methods)
    axes[1].set_title("Sensitivity to Prior Misspecification (FDR)")
    axes[1].axhline(0.05, color='red', linestyle='--', alpha=0.5)
    axes[1].set_xticks(np.arange(len(methods)))
    axes[1].set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], rotation=45, ha="right")
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig5_prior_misspecification.png", dpi=300, bbox_inches="tight")
    print(f"Saved Fig 5 to {OUTPUT_DIR}")
    plt.close(fig)

# =============================================================================
# AUXILIARY: COMPUTATIONAL EFFICIENCY
# =============================================================================

def plot_computational_efficiency():
    """Plot elapsed time vs TPR/Method."""
    candidates = [
        RESULTS_DIR / "power" / "power_analysis_1000_30",
        RESULTS_DIR / "power_analysis_1000_30",
    ]
    res_dir = next((path for path in candidates if (path / "power_vs_effect_size.csv").exists()), None)
    if res_dir is None:
        return

    df_path = res_dir / "power_vs_effect_size.csv"
    df = pd.read_csv(df_path)
    
    # Filter to main methods
    main_methods = [m for m in METHOD_ORDER if m in df["method"].values]
    df_sub = df[df["method"].isin(main_methods)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Average time per method
    sns.barplot(data=df_sub, x="method", y="elapsed_time", ax=ax1, order=main_methods, palette="magma", hue="method", legend=False)
    ax1.set_yscale("log")
    ax1.set_title("Computational Cost (per repeat)")
    ax1.set_ylabel("Elapsed Time (s, log-scale)")
    ax1.set_xticks(np.arange(len(main_methods)))
    ax1.set_xticklabels([METHOD_LABELS.get(m, m) for m in main_methods], rotation=45, ha="right")
    
    # TPR vs Time (Scatter)
    summary = df_sub.groupby("method").agg({
        "TPR": "mean",
        "elapsed_time": "mean"
    }).reset_index()
    
    for _, row in summary.iterrows():
        ax2.scatter(row["elapsed_time"], row["TPR"], s=100, label=METHOD_LABELS.get(row["method"], row["method"]))
    
    ax2.set_xscale("log")
    ax2.set_xlabel("Average Time (s, log-scale)")
    ax2.set_ylabel("Average TPR")
    ax2.set_title("Efficiency Trade-off")
    ax2.legend(fontsize=6)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "computational_efficiency.png", dpi=300, bbox_inches="tight")
    print(f"Saved computational efficiency to {OUTPUT_DIR}")
    plt.close(fig)

# =============================================================================
# FIGURE 7: GLM FWER CALIBRATION
# =============================================================================

def plot_glm_fwer_calibration():
    """Plot FWER calibration for the GLM tested regressor."""
    candidates = [
        RESULTS_DIR / "_archive" / "calibration" / "glm_validation_1000",
        RESULTS_DIR / "glm_validation_1000",
    ]
    res_dir = next((path for path in candidates if (path / "glm_fwer_results.csv").exists()), None)
    if res_dir is None:
        print("Skipping GLM FWER plot: no glm_fwer_results.csv found.")
        return

    print(f"Using GLM calibration results from: {res_dir}")
    # To reuse plot_paper_fwer_calibration, we need a standard filename 'fpr_results.csv'
    # in a dedicated directory.
    import shutil
    tmp_dir = res_dir / "tmp_fwer_plot"
    tmp_dir.mkdir(exist_ok=True)
    shutil.copy(res_dir / "glm_fwer_results.csv", tmp_dir / "fpr_results.csv")

    try:
        plot_paper_fwer_calibration(
            tmp_dir,
            OUTPUT_DIR,
            filename="fig7_glm_fwer_calibration.png",
            method_order=METHOD_ORDER,
            method_labels=METHOD_LABELS,
        )
    finally:
        import shutil
        shutil.rmtree(tmp_dir)

# =============================================================================
# FIGURE 8: GLM CONTINUOUS COVARIATE POWER
# =============================================================================

def plot_glm_power():
    """Plot TPR for continuous covariate in GLM."""
    candidates = [
        RESULTS_DIR / "_archive" / "calibration" / "glm_validation_1000",
        RESULTS_DIR / "glm_validation_1000",
    ]
    res_dir = next((path for path in candidates if (path / "glm_power_results.csv").exists()), None)
    if res_dir is None:
        return

    df = pd.read_csv(res_dir / "glm_power_results.csv")
    
    # Unified column naming: use n_samples if available, fallback to n_subjects
    if "n_samples" in df.columns:
        n_col = "n_samples"
    elif "n_subjects" in df.columns:
        n_col = "n_subjects"
    else:
        n_col = None

    if n_col:
        n_vals = df[n_col].unique()
        n_sel = n_vals[-1]
        sub_df = df[df[n_col] == n_sel]
    else:
        sub_df = df
        n_sel = "all"

    # TPR Heatmap at a moderate effect size
    es_vals = sub_df["effect_size"].unique()
    if len(es_vals) > 0:
        es_sel = es_vals[len(es_vals)//2]
        heatmap_df = sub_df[np.isclose(sub_df["effect_size"], es_sel)]
        
        tpr_pivot = heatmap_df.pivot_table(index="scenario", columns="method", values="TPR", aggfunc="mean")
        tpr_pivot = tpr_pivot.reindex(index=TOPOLOGY_ORDER, columns=METHOD_ORDER).dropna(axis=0, how='all').dropna(axis=1, how='all')
        tpr_pivot.index = [TOPOLOGY_LABELS.get(i, i) for i in tpr_pivot.index]
        tpr_pivot.columns = [METHOD_LABELS.get(c, c) for c in tpr_pivot.columns]

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(tpr_pivot, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax, vmin=0, vmax=1)
        ax.set_title(f"GLM Continuous Power (TPR) @ d={es_sel}, N={n_sel}")
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / "fig8_glm_power_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved Fig 8 to {OUTPUT_DIR}")

# =============================================================================
# TOPOLOGY GALLERY
# =============================================================================

def plot_topology_gallery_representative(effect_size: float = 0.3, n_samples: int = 30):
    """Plot representative ground truth and t-stats for all topologies."""
    print(f"Generating Topology Gallery (es={effect_size}, n={n_samples})...")
    plot_topology_gallery(
        effect_size=effect_size,
        n_samples=n_samples,
        output_dir=OUTPUT_DIR,
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Regenerate all simulation paper figures from existing result tables."""
    print(f"Regenerating paper figures into {OUTPUT_DIR}...")
    
    plot_topology_gallery_representative(effect_size=0.3, n_samples=30)
    plot_fwer_calibration()
    plot_power_heatmap(effect_size=0.25)
    plot_effect_size_sweep()
    plot_tfnbs_eh_sweep(effect_size=0.25)
    plot_prior_misspecification()
    
    if GENERATE_GLM_FIGURES:
        plot_glm_fwer_calibration()
        plot_glm_power()
    else:
        print("Skipping GLM figures: GLM validation has not been regenerated.")
    
    print("\nAll figures regenerated successfully.")


if __name__ == "__main__":
    main()
