"""
ABIDE §3.1 deliverables audit.

Reads the existing analysis_*.npz (Fisher-z + FL-with-site-dummies) and the
new ComBat-based analysis_*_combat.npz (§3.1 under ValidationPlan.md).
Computes §2.5 agreement metrics and §3.1 diagnostics.

Deliverables produced
---------------------
results/audit/summary.csv      one row per (design, analysis, method, tail)
results/audit/figures/*.png    per-comparison figures
results/audit/design_diagnostics.txt   VIFs, condition numbers

Designed to be idempotent: rerun after new analyses are added and it picks
them up from the filesystem.
"""

from __future__ import annotations

import argparse
import os
import sys
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Shared agreement helpers (examples/_audit_utils.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _audit_utils import (  # noqa: E402
    upper_triangle,
    jaccard,
    jaccard_random_baseline,
    fisher_exact_2x2,
    spearman_neglog10,
    topk_concordance,
    block_mass,
    block_mass_correlation,
)


HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
AUDIT_DIR = os.path.join(RESULTS_DIR, "audit")
FIG_DIR = os.path.join(AUDIT_DIR, "figures")
PREPARED = os.path.join(HERE, "abide_prepared.npz")

TAIL_CANON = {
    "g2>g1": "positive",   # t-test pipeline → GLM-style naming
    "g1>g2": "negative",
    "positive": "positive",
    "negative": "negative",
}


def load_pmap(path: str) -> Dict[str, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    out = {}
    for k in d.files:
        if k in TAIL_CANON:
            out[TAIL_CANON[k]] = d[k]
    return out


# =============================================================================
# §2.5 agreement metrics — most helpers come from examples/_audit_utils.py.
# The agreement_quartet() wrapper below renames keys to this script's
# historical schema (n_sig_a / topk_10 / etc.) for backward compatibility.
# =============================================================================


def agreement_quartet(
    p_a_full: np.ndarray, p_b_full: np.ndarray,
    net_labels: np.ndarray, alpha: float = 0.05,
) -> Dict[str, float]:
    p_a = upper_triangle(p_a_full); p_b = upper_triangle(p_b_full)
    sig_a = p_a < alpha; sig_b = p_b < alpha
    k_a = int(sig_a.sum()); k_b = int(sig_b.sum())
    return {
        "n_sig_a": k_a,
        "n_sig_b": k_b,
        "jaccard": jaccard(sig_a, sig_b),
        "jaccard_random_null": jaccard_random_baseline(k_a, k_b, p_a.size),
        "fisher_exact_p": fisher_exact_2x2(sig_a, sig_b),
        "spearman_neglog10": spearman_neglog10(p_a, p_b),
        "topk_10":  topk_concordance(p_a, p_b, 10),
        "topk_50":  topk_concordance(p_a, p_b, 50),
        "topk_100": topk_concordance(p_a, p_b, 100),
        "topk_500": topk_concordance(p_a, p_b, 500),
        "block_mass_pearson": block_mass_correlation(p_a_full, p_b_full, net_labels),
    }


# =============================================================================
# Design-matrix diagnostics (§3.1 deliverable)
# =============================================================================


def vif(X: np.ndarray) -> np.ndarray:
    """Variance Inflation Factor per column (excluding the column itself as response)."""
    from numpy.linalg import lstsq, LinAlgError
    n, p = X.shape
    vifs = np.full(p, np.nan)
    for j in range(p):
        y = X[:, j]
        Xj = np.delete(X, j, axis=1)
        try:
            beta, *_ = lstsq(Xj, y, rcond=None)
            yhat = Xj @ beta
            ss_res = ((y - yhat) ** 2).sum()
            ss_tot = ((y - y.mean()) ** 2).sum()
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            vifs[j] = 1.0 / (1 - r2) if r2 < 1 else np.inf
        except LinAlgError:
            vifs[j] = np.inf
    return vifs


def design_diagnostics(prepared_npz: str) -> pd.DataFrame:
    d = np.load(prepared_npz, allow_pickle=True)
    age = d["age"]; sex = d["sex"]; fd = d["mean_fd"]
    site_dummies = d["site_dummies"]
    group = d["group"]

    # Old design: age + sex + FD + 14 site dummies + intercept (as if full model)
    X = np.column_stack([
        np.ones_like(age), group, age, sex, fd, site_dummies
    ])
    col_names = ["intercept", "group", "age", "sex", "mean_fd"] + \
                [f"site_{s}" for s in d["site_names"]]

    # VIFs (skip intercept)
    vifs = vif(X[:, 1:])   # skip intercept for VIF
    vif_row = [np.nan] + list(vifs)

    # Condition number
    cond = float(np.linalg.cond(X))

    # Correlation of each confound with group
    from scipy.stats import pointbiserialr
    corrs = []
    for col in range(X.shape[1]):
        if col == 0 or col == 1:
            corrs.append(np.nan)  # intercept, group
        else:
            r, _ = pointbiserialr(group, X[:, col])
            corrs.append(float(r))

    df = pd.DataFrame({
        "column": col_names,
        "vif": vif_row,
        "corr_with_group": corrs,
    })
    # Also return the scalar condition number via attributes
    df.attrs["condition_number"] = cond
    return df


# =============================================================================
# Orchestration
# =============================================================================


def discover_pmap_files() -> Dict[str, str]:
    """Return {label: filepath} for all known result files."""
    files = {}
    # Old design (Fisher-z + FL-with-site-dummies)
    for p in sorted(glob(os.path.join(RESULTS_DIR, "analysis_*.npz"))):
        base = os.path.basename(p)
        if "_combat" in base:
            continue
        label = base.removesuffix(".npz")
        files[f"fz_dummies/{label}"] = p
    # New design (ComBat)
    combat_dir = os.path.join(RESULTS_DIR, "combat")
    if os.path.isdir(combat_dir):
        for p in sorted(glob(os.path.join(combat_dir, "analysis_*.npz"))):
            base = os.path.basename(p)
            label = base.removesuffix(".npz")
            files[f"combat/{label}"] = p
    return files


def per_file_summary(label: str, path: str, alpha: float = 0.05) -> List[Dict]:
    pmaps = load_pmap(path)
    rows = []
    for tail, p_full in pmaps.items():
        p_vec = upper_triangle(p_full)
        sig = p_vec < alpha
        rows.append({
            "design_analysis": label,
            "tail": tail,
            "n_edges": int(p_vec.size),
            "n_sig_005": int(sig.sum()),
            "min_p": float(p_vec.min()),
            "median_neglog10p": float(np.median(-np.log10(p_vec + 1e-12))),
        })
    return rows


def pairwise_agreement(
    files: Dict[str, str], net_labels: np.ndarray,
    pairs: Iterable[Tuple[str, str]], alpha: float = 0.05,
) -> List[Dict]:
    rows = []
    for a, b in pairs:
        if a not in files or b not in files:
            continue
        pa = load_pmap(files[a]); pb = load_pmap(files[b])
        common_tails = set(pa) & set(pb)
        for tail in sorted(common_tails):
            metrics = agreement_quartet(pa[tail], pb[tail], net_labels, alpha)
            rows.append({"a": a, "b": b, "tail": tail, **metrics})
    return rows


# =============================================================================
# Plots
# =============================================================================


def plot_design_diagnostics(df: pd.DataFrame, outpath: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df_plot = df.dropna(subset=["vif"])
    axes[0].barh(df_plot["column"], df_plot["vif"].clip(upper=50))
    axes[0].axvline(5, linestyle="--", color="orange", label="VIF=5 (warning)")
    axes[0].axvline(10, linestyle="--", color="red", label="VIF=10 (strong)")
    axes[0].set_xlabel("VIF (clipped at 50)"); axes[0].set_title("Variance Inflation Factors")
    axes[0].legend(fontsize=8)

    df_plot2 = df.dropna(subset=["corr_with_group"])
    axes[1].barh(df_plot2["column"], df_plot2["corr_with_group"])
    axes[1].axvline(0, color="k", lw=0.5)
    axes[1].set_xlabel("point-biserial r"); axes[1].set_title("Confound correlation with group (ASD=1)")
    for ax in axes:
        ax.tick_params(axis="y", labelsize=8)
    cond = df.attrs.get("condition_number", float("nan"))
    fig.suptitle(f"ABIDE design-matrix diagnostics  (condition number = {cond:.1f})")
    fig.tight_layout(); fig.savefig(outpath, dpi=140); plt.close(fig)


def plot_agreement_heatmap(pair_rows: List[Dict], outpath: str) -> None:
    if not pair_rows:
        return
    df = pd.DataFrame(pair_rows)
    # One row per (a, b) pair with average across tails for visualisation
    agg = df.groupby(["a", "b"]).agg(
        jaccard=("jaccard", "mean"),
        spearman=("spearman_neglog10", "mean"),
        topk_100=("topk_100", "mean"),
        block_corr=("block_mass_pearson", "mean"),
    ).reset_index()
    if agg.empty:
        return
    metrics = ["jaccard", "spearman", "topk_100", "block_corr"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 3.5))
    for ax, m in zip(axes, metrics):
        pivot = agg.pivot(index="a", columns="b", values=m)
        im = ax.imshow(pivot.values, cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels(pivot.columns, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(pivot.shape[0])); ax.set_yticklabels(pivot.index, fontsize=7)
        ax.set_title(m)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"§2.5 agreement quartet (averaged over tails; α=0.05)")
    fig.tight_layout(); fig.savefig(outpath, dpi=140); plt.close(fig)


# =============================================================================
# Main
# =============================================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)

    # 1. Design diagnostics on prepared data
    print("=" * 70)
    print("Design-matrix diagnostics")
    print("=" * 70)
    diag_df = design_diagnostics(PREPARED)
    diag_path = os.path.join(AUDIT_DIR, "design_diagnostics.txt")
    with open(diag_path, "w") as f:
        f.write(f"ABIDE design-matrix diagnostics\n")
        f.write(f"Condition number: {diag_df.attrs['condition_number']:.2f}\n\n")
        f.write(diag_df.to_string(index=False, float_format="%.3f"))
    print(diag_df.to_string(index=False, float_format="%.3f"))
    print(f"Condition number: {diag_df.attrs['condition_number']:.2f}")
    plot_design_diagnostics(diag_df, os.path.join(FIG_DIR, "design_diagnostics.png"))

    # 2. Per-file summary
    files = discover_pmap_files()
    print(f"\nFound {len(files)} p-map files:")
    for k, v in files.items():
        print(f"  {k:40s} -> {os.path.relpath(v, RESULTS_DIR)}")

    summary_rows = []
    for label, path in files.items():
        summary_rows.extend(per_file_summary(label, path, alpha=args.alpha))
    sum_df = pd.DataFrame(summary_rows)
    sum_path = os.path.join(AUDIT_DIR, "summary.csv")
    sum_df.to_csv(sum_path, index=False)
    print(f"\nPer-file summary → {sum_path}")
    print(sum_df.to_string(index=False))

    # 3. Pairwise agreement (§2.5 quartet)
    prep = np.load(PREPARED, allow_pickle=True)
    net_labels = prep["net_labels"]

    pairs = []
    # Old design: naive vs GLM across methods
    if "fz_dummies/analysis_1_naive" in files and "fz_dummies/analysis_2_glm" in files:
        pairs.append(("fz_dummies/analysis_1_naive", "fz_dummies/analysis_2_glm"))
    # New vs old: ComBat GLM vs FL-dummies GLM (the §3.1.X sensitivity check)
    if "combat/analysis_2_glm_combat" in files and "fz_dummies/analysis_2_glm" in files:
        pairs.append(("combat/analysis_2_glm_combat", "fz_dummies/analysis_2_glm"))
    if "combat/analysis_1_naive_combat" in files and "fz_dummies/analysis_1_naive" in files:
        pairs.append(("combat/analysis_1_naive_combat", "fz_dummies/analysis_1_naive"))
    # Method comparison (old design, analysis 4)
    method_keys = [k for k in files if k.startswith("fz_dummies/analysis_4_")]
    for i, a in enumerate(method_keys):
        for b in method_keys[i+1:]:
            pairs.append((a, b))

    print(f"\n{len(pairs)} pairwise agreement comparisons")
    agreement_rows = pairwise_agreement(files, net_labels, pairs, alpha=args.alpha)
    agr_df = pd.DataFrame(agreement_rows)
    agr_path = os.path.join(AUDIT_DIR, "agreement.csv")
    agr_df.to_csv(agr_path, index=False)
    print(f"Agreement quartet → {agr_path}")
    print(agr_df.to_string(index=False, float_format="%.3f"))

    plot_agreement_heatmap(agreement_rows, os.path.join(FIG_DIR, "agreement_heatmap.png"))
    print(f"\nFigures → {FIG_DIR}")


if __name__ == "__main__":
    main()
