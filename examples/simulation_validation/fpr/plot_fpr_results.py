"""
Plot null FWER calibration results.

Reads CSV output from fpr_calibration.py and produces:
1. Calibration overview — 4-panel plot: FWER Clopper-Pearson interval
   boxes, two-sided FWER interval boxes, min p-value distributions,
   calibration summary table.
   One plot per sample size when data contains multiple n_samples.
2. Per-method detail — per-run edge-FPR distribution (box plots).
   One plot per sample size.
3. FWER directions — FWER interval boxes for pos/neg/either direction
   and two-sided alpha/2.
   One plot per sample size.
4. FWER vs sample size — line plot (one line per method, with CI bands).
   Only generated when data contains multiple sample sizes.

Usage:
    # Full calibration overview
    python plot_fpr_results.py examples/simulation_validation/results/fpr_calibration_local

    # Plot specific methods only (loads per-method CSVs)
    python plot_fpr_results.py examples/simulation_validation/results/fpr_calibration_local \
        --method tstat tfnbs cnbs

    # List available per-method CSVs
    python plot_fpr_results.py examples/simulation_validation/results/fpr_calibration_local --list-methods

    # Specific plot type
    python plot_fpr_results.py examples/simulation_validation/results/fpr_calibration_local --plot overview

    # FWER vs sample size (requires multi-n_samples data)
    python plot_fpr_results.py examples/simulation_validation/results/fpr_calibration_local --plot fwer-vs-n

    # Custom output directory
    python plot_fpr_results.py examples/simulation_validation/results/fpr_calibration_local --output-dir plots/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# DATA LOADING
# =============================================================================

def list_available_methods(results_dir: Path) -> List[str]:
    """List methods that have per-method CSV files."""
    pm_dir = results_dir / "per_method"
    if not pm_dir.exists():
        return []
    methods = []
    for f in sorted(pm_dir.glob("fpr_results_*.csv")):
        name = f.stem.replace("fpr_results_", "")
        methods.append(name)
    return methods


def load_results(
    results_dir: Path,
    methods: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load null-calibration results from combined or per-method CSVs.

    If *methods* is given, loads only those from per_method/ subdir
    (much faster for large runs). Otherwise loads the full combined CSV.
    """
    if methods:
        pm_dir = results_dir / "per_method"
        frames = []
        for m in methods:
            path = pm_dir / f"fpr_results_{m}.csv"
            if not path.exists():
                print(f"  WARNING: {path} not found, skipping")
                continue
            frames.append(pd.read_csv(path))
        if not frames:
            sys.exit(f"No per-method CSVs found in {pm_dir}")
        df = pd.concat(frames, ignore_index=True)
        print(f"Loaded {len(methods)} per-method CSVs: {len(df)} rows")
    else:
        combined = results_dir / "fpr_results.csv"
        if not combined.exists():
            sys.exit(f"Results file not found: {combined}")
        df = pd.read_csv(combined)
        print(f"Loaded {combined.name}: {len(df)} rows")
    return df


# =============================================================================
# STATISTICS (duplicated from fpr_calibration to keep this script standalone)
# =============================================================================

def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Clopper-Pearson exact binomial confidence interval."""
    from scipy import stats

    if k == 0:
        ci_low = 0.0
    else:
        ci_low = stats.beta.ppf(alpha / 2, k, n - k + 1)

    if k == n:
        ci_high = 1.0
    else:
        ci_high = stats.beta.ppf(1 - alpha / 2, k + 1, n - k)

    return ci_low, ci_high


def _ci_not_liberal(k: int, n: int, target_alpha: float) -> Tuple[float, float, bool]:
    """95% CP interval plus validity for an upper-error-rate claim."""
    ci_low, ci_high = clopper_pearson_ci(k, n, alpha=0.05)
    return ci_low, ci_high, bool(ci_low <= target_alpha)


def _plot_cp_interval_boxes(
    ax,
    positions,
    estimates,
    ci_lows,
    ci_highs,
    colors,
    *,
    width: float = 0.55,
    label: Optional[str] = None,
) -> None:
    """Draw box-style binomial intervals from Clopper-Pearson CIs.

    The box spans the exact 95% CI and the black marker shows the empirical
    FWER estimate. This avoids implying a continuous sample distribution.
    """
    stats = [
        {
            "label": "",
            "med": float(est),
            "q1": float(lo),
            "q3": float(hi),
            "whislo": float(lo),
            "whishi": float(hi),
            "fliers": [],
        }
        for est, lo, hi in zip(estimates, ci_lows, ci_highs)
    ]
    artists = ax.bxp(
        stats,
        positions=np.asarray(positions, dtype=float),
        widths=width,
        showfliers=False,
        patch_artist=True,
        manage_ticks=False,
    )

    for box, color in zip(artists["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.45)
        box.set_edgecolor("black")
        box.set_linewidth(1.1)
    if label and artists["boxes"]:
        artists["boxes"][0].set_label(label)
    for key in ("whiskers", "caps", "medians"):
        for artist in artists[key]:
            artist.set_color("black")
            artist.set_linewidth(1.1 if key != "medians" else 1.8)

    ax.scatter(
        positions,
        estimates,
        s=28,
        c="black",
        edgecolors="white",
        linewidths=0.5,
        zorder=4,
    )


def _summarise_group(mdf: pd.DataFrame, method: str, alpha: float) -> dict:
    """Compute summary stats for one (method[, n_samples]) group."""
    from scipy import stats as sp_stats

    n_runs = len(mdf)
    n_edges = mdf["n_edges"].iloc[0]

    n_fp_pos = int(mdf["any_significant_pos"].sum())
    fwer_pos = n_fp_pos / n_runs
    n_fp_neg = int(mdf["any_significant_neg"].sum())
    fwer_neg = n_fp_neg / n_runs
    fwer_pos_ci_low, fwer_pos_ci_high, fwer_pos_valid = _ci_not_liberal(
        n_fp_pos, n_runs, alpha
    )
    fwer_neg_ci_low, fwer_neg_ci_high, fwer_neg_valid = _ci_not_liberal(
        n_fp_neg, n_runs, alpha
    )

    if fwer_neg > fwer_pos:
        fwer_max_tail = fwer_neg
        fwer_max_tail_name = "negative"
        n_fp_max_tail = n_fp_neg
        fwer_ci_low = fwer_neg_ci_low
        fwer_ci_high = fwer_neg_ci_high
    else:
        fwer_max_tail = fwer_pos
        fwer_max_tail_name = "positive"
        n_fp_max_tail = n_fp_pos
        fwer_ci_low = fwer_pos_ci_low
        fwer_ci_high = fwer_pos_ci_high

    directional_fwer_valid = bool(fwer_pos_valid and fwer_neg_valid)

    n_fp_either_alpha = int(mdf["any_significant"].sum())
    fwer_either_at_alpha = n_fp_either_alpha / n_runs
    fwer_either_ci_low, fwer_either_ci_high = clopper_pearson_ci(
        n_fp_either_alpha, n_runs, alpha=0.05
    )

    two_sided_alpha_per_tail = alpha / 2.0
    two_sided_hits = (
        (mdf["min_p_pos"] <= two_sided_alpha_per_tail)
        | (mdf["min_p_neg"] <= two_sided_alpha_per_tail)
    )
    n_fp_two_sided = int(two_sided_hits.sum())
    fwer_two_sided_bonf = n_fp_two_sided / n_runs
    (
        fwer_two_sided_ci_low,
        fwer_two_sided_ci_high,
        fwer_two_sided_valid,
    ) = _ci_not_liberal(n_fp_two_sided, n_runs, alpha)

    mean_fpr_pos = mdf["fpr_pos"].mean()
    mean_fpr_neg = mdf["fpr_neg"].mean()
    mean_fpr_total = mdf["fpr_total"].mean()
    if mean_fpr_neg > mean_fpr_pos:
        edge_fpr_col = "fpr_neg"
        mean_fpr_max_tail = mean_fpr_neg
    else:
        edge_fpr_col = "fpr_pos"
        mean_fpr_max_tail = mean_fpr_pos
    se_fpr_max_tail = mdf[edge_fpr_col].std() / np.sqrt(n_runs)

    t_crit = sp_stats.t.ppf(0.975, df=n_runs - 1)
    fpr_ci_low = max(0, mean_fpr_max_tail - t_crit * se_fpr_max_tail)
    fpr_ci_high = min(1, mean_fpr_max_tail + t_crit * se_fpr_max_tail)
    edge_fpr_not_liberal = bool(fpr_ci_low <= alpha)

    return {
        "method": method,
        "n_null_runs": n_runs,
        "n_edges": n_edges,
        "fwer_pos": fwer_pos,
        "fwer_neg": fwer_neg,
        "fwer_both": fwer_either_at_alpha,
        "fwer_either_at_alpha": fwer_either_at_alpha,
        "fwer_either_at_alpha_ci_low": fwer_either_ci_low,
        "fwer_either_at_alpha_ci_high": fwer_either_ci_high,
        "fwer_either_expected_upper": min(1.0, 2.0 * alpha),
        "fwer_either_independent_expected": 1.0 - (1.0 - alpha) ** 2,
        "fwer_pos_ci_low": fwer_pos_ci_low,
        "fwer_pos_ci_high": fwer_pos_ci_high,
        "fwer_neg_ci_low": fwer_neg_ci_low,
        "fwer_neg_ci_high": fwer_neg_ci_high,
        "fwer_max_tail": fwer_max_tail,
        "fwer_max_tail_name": fwer_max_tail_name,
        "n_runs_with_fp_max_tail": n_fp_max_tail,
        "fwer_ci_low": fwer_ci_low,
        "fwer_ci_high": fwer_ci_high,
        "fwer_calibrated": directional_fwer_valid,
        "directional_fwer_valid": directional_fwer_valid,
        "two_sided_alpha_per_tail": two_sided_alpha_per_tail,
        "fwer_two_sided_bonf": fwer_two_sided_bonf,
        "fwer_two_sided_bonf_ci_low": fwer_two_sided_ci_low,
        "fwer_two_sided_bonf_ci_high": fwer_two_sided_ci_high,
        "fwer_two_sided_bonf_valid": fwer_two_sided_valid,
        "mean_fpr_pos": mean_fpr_pos,
        "mean_fpr_neg": mean_fpr_neg,
        "mean_fpr_max_tail": mean_fpr_max_tail,
        "mean_fpr_total": mean_fpr_total,
        "fpr_se": se_fpr_max_tail,
        "fpr_ci_low": fpr_ci_low,
        "fpr_ci_high": fpr_ci_high,
        "fpr_calibrated": edge_fpr_not_liberal,
        "expected": alpha,
        "mean_n_fp_pos": mdf["n_significant_pos"].mean(),
        "mean_n_fp_neg": mdf["n_significant_neg"].mean(),
        "mean_elapsed_time": mdf["elapsed_time"].mean(),
    }


def compute_fpr_summary_single(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Compute FWER summary for a single-n_samples slice (grouped by method only)."""
    rows = []
    for method, mdf in df.groupby("method"):
        rows.append(_summarise_group(mdf, method, alpha))
    return pd.DataFrame(rows)


def compute_fpr_summary(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Compute FWER summary, grouped by (method, n_samples) if possible."""
    has_n_samples = "n_samples" in df.columns
    
    if not has_n_samples:
        return compute_fpr_summary_single(df, alpha)

    rows = []
    for (method, n_samples), mdf in df.groupby(["method", "n_samples"]):
        row = _summarise_group(mdf, method, alpha)
        row["n_samples"] = int(n_samples)
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# PLOT: CALIBRATION OVERVIEW (4-panel) — single n_samples
# =============================================================================

def plot_overview(
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    alpha: float,
    output_dir: Path,
    suffix: str = "",
    title_extra: str = "",
) -> None:
    """4-panel overview: CP interval boxes, min-p distributions, and table."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    methods = summary_df["method"].tolist()
    colors_base = plt.cm.Set2(np.linspace(0, 1, len(methods)))

    # --- Panel 1: directional FWER Clopper-Pearson interval boxes ---
    ax = axes[0, 0]
    x = np.arange(len(methods))

    fwer_vals = summary_df["fwer_max_tail"].values
    fwer_ci_lows = summary_df["fwer_ci_low"].values
    fwer_ci_highs = summary_df["fwer_ci_high"].values

    box_colors = ["#4CAF50" if row["directional_fwer_valid"] else "#F44336"
                  for _, row in summary_df.iterrows()]

    _plot_cp_interval_boxes(
        ax,
        x,
        fwer_vals,
        fwer_ci_lows,
        fwer_ci_highs,
        box_colors,
        width=0.58,
        label="Empirical FWER",
    )
    ax.axhline(alpha, color="blue", linestyle="--", linewidth=2, label=f"Target <= {alpha}")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Directional FWER", fontsize=11)
    ax.set_title("Worst-Tail FWER  max P(any FP in one tail | H0)", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, max(0.15, fwer_ci_highs.max() * 1.3))
    ax.grid(axis="y", alpha=0.3)

    # --- Panel 2: two-sided Bonferroni-over-tails FWER CP interval boxes ---
    ax = axes[0, 1]
    two_vals = summary_df["fwer_two_sided_bonf"].values
    two_ci_lows = summary_df["fwer_two_sided_bonf_ci_low"].values
    two_ci_highs = summary_df["fwer_two_sided_bonf_ci_high"].values

    box_colors = ["#4CAF50" if row["fwer_two_sided_bonf_valid"] else "#F44336"
                  for _, row in summary_df.iterrows()]

    _plot_cp_interval_boxes(
        ax,
        x,
        two_vals,
        two_ci_lows,
        two_ci_highs,
        box_colors,
        width=0.58,
        label="Empirical FWER",
    )
    ax.axhline(alpha, color="blue", linestyle="--", linewidth=2, label=f"Target <= {alpha}")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Two-sided FWER", fontsize=11)
    ax.set_title(f"Either Tail with alpha/2 per tail ({alpha/2:.3f})", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, max(0.15, two_ci_highs.max() * 1.3) if two_ci_highs.max() > 0 else 0.1)
    ax.grid(axis="y", alpha=0.3)

    # --- Panel 3: Min p-value distributions ---
    ax = axes[1, 0]
    for i, method in enumerate(methods):
        method_df = df[df["method"] == method]
        min_ps = method_df["min_p_pos"].values
        counts, bins = np.histogram(min_ps, bins=20, range=(0, 1))
        counts = counts / counts.sum() if counts.sum() > 0 else counts
        ax.step(bins[:-1], counts, where="post", label=method,
                color=colors_base[i], linewidth=2, alpha=0.8)
    ax.axvline(alpha, color="red", linestyle="--", linewidth=2, label=f"alpha = {alpha}")
    ax.set_xlabel("Minimum p-value", fontsize=11)
    ax.set_ylabel("Proportion", fontsize=11)
    ax.set_title("Distribution of Min P-values", fontsize=12)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.3)

    # --- Panel 4: Calibration summary table ---
    ax = axes[1, 1]
    ax.axis("off")
    table_data = []
    for _, row in summary_df.iterrows():
        fwer_status = "OK" if row["fwer_calibrated"] else "FAIL"
        table_data.append([
            row["method"],
            f"{row['fwer_max_tail']:.3f}",
            fwer_status,
            f"{row['fwer_two_sided_bonf']:.3f}",
            "OK" if row["fwer_two_sided_bonf_valid"] else "FAIL",
        ])
    table = ax.table(
        cellText=table_data,
        colLabels=["Method", "Max-tail", "Cal.", "2-sided", "Cal."],
        loc="center", cellLoc="center",
        colWidths=[0.25, 0.18, 0.12, 0.18, 0.12],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    for i, (_, row) in enumerate(summary_df.iterrows()):
        fwer_color = "#C8E6C9" if row["fwer_calibrated"] else "#FFCDD2"
        fpr_color = "#C8E6C9" if row["fwer_two_sided_bonf_valid"] else "#FFCDD2"
        table[(i + 1, 0)].set_facecolor("#FFFFFF")
        table[(i + 1, 1)].set_facecolor(fwer_color)
        table[(i + 1, 2)].set_facecolor(fwer_color)
        table[(i + 1, 3)].set_facecolor(fpr_color)
        table[(i + 1, 4)].set_facecolor(fpr_color)
    ax.set_title(f"Calibration Summary (alpha = {alpha})", fontsize=12, pad=20)

    fig.suptitle(f"FWER Calibration Results{title_extra}",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    out = output_dir / f"fpr_calibration_overview{suffix}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out.name}")


# =============================================================================
# PLOT: PER-METHOD DETAIL — single n_samples
# =============================================================================

def plot_per_method(
    df: pd.DataFrame,
    alpha: float,
    output_dir: Path,
    suffix: str = "",
    title_extra: str = "",
) -> None:
    """Per-method FPR distribution: box plots for each metric."""
    methods = sorted(df["method"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(5 * 3, 5))

    metrics = [
        ("fpr_pos", "Edge-wise FPR (g2>g1)"),
        ("fpr_neg", "Edge-wise FPR (g1>g2)"),
        ("min_p_pos", "Min p-value (g2>g1)"),
    ]

    for ax, (col, title) in zip(axes, metrics):
        sns.boxplot(data=df, x="method", y=col, hue="method", order=methods,
                    hue_order=methods, ax=ax, fliersize=2, palette="Set2",
                    legend=False)
        if col.startswith("fpr"):
            ax.axhline(alpha, color="red", linestyle="--", linewidth=1.5,
                       label=f"alpha = {alpha}")
            ax.legend(fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel(col)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle(f"Per-Method Edge-FPR Distributions{title_extra}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    out = output_dir / f"fpr_per_method_detail{suffix}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out.name}")


# =============================================================================
# PLOT: FWER BOTH DIRECTIONS — single n_samples
# =============================================================================

def plot_fwer_directions(
    summary_df: pd.DataFrame,
    alpha: float,
    output_dir: Path,
    suffix: str = "",
    title_extra: str = "",
) -> None:
    """CP interval boxes for positive, negative, either-tail, and two-sided FWER."""
    methods = summary_df["method"].tolist()
    x = np.arange(len(methods))
    width = 0.20

    fig, ax = plt.subplots(figsize=(12, 5))
    series = [
        (
            "g2>g1",
            -1.5 * width,
            "fwer_pos",
            "fwer_pos_ci_low",
            "fwer_pos_ci_high",
            "#42A5F5",
        ),
        (
            "g1>g2",
            -0.5 * width,
            "fwer_neg",
            "fwer_neg_ci_low",
            "fwer_neg_ci_high",
            "#EF5350",
        ),
        (
            "Either, alpha/2 per tail",
            0.5 * width,
            "fwer_two_sided_bonf",
            "fwer_two_sided_bonf_ci_low",
            "fwer_two_sided_bonf_ci_high",
            "#66BB6A",
        ),
        (
            "Either, unadjusted alpha",
            1.5 * width,
            "fwer_either_at_alpha",
            "fwer_either_at_alpha_ci_low",
            "fwer_either_at_alpha_ci_high",
            "#BDBDBD",
        ),
    ]
    max_ci_high = 0.0
    for label, offset, value_col, low_col, high_col, color in series:
        values = summary_df[value_col].values
        ci_lows = summary_df[low_col].values
        ci_highs = summary_df[high_col].values
        max_ci_high = max(max_ci_high, float(np.max(ci_highs)))
        _plot_cp_interval_boxes(
            ax,
            x + offset,
            values,
            ci_lows,
            ci_highs,
            [color] * len(methods),
            width=width * 0.82,
            label=label,
        )
    ax.axhline(alpha, color="black", linestyle="--", linewidth=1.5,
               label=f"target alpha = {alpha}")
    ax.axhline(min(1.0, 2.0 * alpha), color="gray", linestyle=":", linewidth=1.2,
               label=f"diagnostic 2*alpha = {min(1.0, 2.0 * alpha):.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("FWER", fontsize=11)
    ax.set_title(f"Directional vs Two-Sided FWER (95% Clopper-Pearson CI){title_extra}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(0.15, max_ci_high * 1.25))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out = output_dir / f"fpr_fwer_directions{suffix}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out.name}")


# =============================================================================
# PLOT: FWER vs SAMPLE SIZE (multi-n only)
# =============================================================================

METHOD_COLORS = {
    "tstat": "#1f77b4",
    "tfnbs": "#ff7f0e",
    "ni_tfnbs": "#2ca02c",
    "fbc_tfnbs": "#d62728",
    "nbs_extent": "#9467bd",
    "nbs_intensity": "#8c564b",
    "nbs_extent_2.0": "#9467bd",
    "nbs_intensity_2.0": "#8c564b",
    "nbs_extent_3.0": "#7b5db5",
    "nbs_intensity_3.0": "#6f4038",
    "cnbs": "#e377c2",
    "bonferroni": "#7f7f7f",
    "bh_fdr": "#bcbd22",
}


def plot_fwer_vs_sample_size(
    summary_df: pd.DataFrame,
    alpha: float,
    output_dir: Path,
) -> None:
    """Directional and Bonferroni two-sided FWER vs sample size."""
    if "n_samples" not in summary_df.columns:
        print("  Skipping: data has no n_samples dimension")
        return

    methods = sorted(summary_df["method"].unique())
    sample_sizes = sorted(summary_df["n_samples"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Panel 1: worst-tail directional FWER vs n_samples ---
    ax = axes[0]
    for method in methods:
        mdf = summary_df[summary_df["method"] == method].sort_values("n_samples")
        color = METHOD_COLORS.get(method, None)
        ax.plot(mdf["n_samples"], mdf["fwer_max_tail"], "o-", label=method,
                color=color, linewidth=2, markersize=6)
        ax.fill_between(
            mdf["n_samples"], mdf["fwer_ci_low"], mdf["fwer_ci_high"],
            alpha=0.15, color=color,
        )
    ax.axhline(alpha, color="black", linestyle="--", linewidth=1.5,
               label=f"target alpha = {alpha}")
    ax.set_xlabel("Sample size (per group)", fontsize=11)
    ax.set_ylabel("Directional FWER", fontsize=11)
    ax.set_title("Worst-Tail Directional FWER vs Sample Size", fontsize=13, fontweight="bold")
    ax.set_xticks(sample_sizes)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    # --- Panel 2: two-sided Bonferroni-over-tails FWER vs n_samples ---
    ax = axes[1]
    for method in methods:
        mdf = summary_df[summary_df["method"] == method].sort_values("n_samples")
        color = METHOD_COLORS.get(method, None)
        ax.plot(mdf["n_samples"], mdf["fwer_two_sided_bonf"], "o-", label=method,
                color=color, linewidth=2, markersize=6)
        ax.fill_between(
            mdf["n_samples"], mdf["fwer_two_sided_bonf_ci_low"], mdf["fwer_two_sided_bonf_ci_high"],
            alpha=0.15, color=color,
        )
    ax.axhline(alpha, color="black", linestyle="--", linewidth=1.5,
               label=f"target alpha = {alpha}")
    ax.set_xlabel("Sample size (per group)", fontsize=11)
    ax.set_ylabel("Two-sided FWER", fontsize=11)
    ax.set_title("Either Tail with alpha/2 per Tail", fontsize=13, fontweight="bold")
    ax.set_xticks(sample_sizes)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = output_dir / "fpr_vs_sample_size.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out.name}")


def plot_paper_fwer_calibration(
    results_dir: Path,
    output_dir: Path,
    *,
    alpha: float = 0.05,
    filename: str = "fig1_fwer_calibration.png",
    method_order: Optional[List[str]] = None,
    method_labels: Optional[dict] = None,
) -> Optional[Path]:
    """Create the compact paper Figure 1 FWER panel.

    This is the single reusable paper-facing FWER implementation. The root
    paper-figure regeneration script calls this function rather than
    duplicating Clopper-Pearson or worst-tail FWER logic.
    """
    df_path = results_dir / "fpr_results.csv"
    if not df_path.exists():
        print(f"Skipping FWER plot: {df_path} not found.")
        return None

    df = pd.read_csv(df_path)
    summary = compute_fpr_summary(df, alpha=alpha)
    if summary.empty:
        print(f"Skipping FWER plot: no rows in {df_path}.")
        return None

    if method_order:
        ordered = [m for m in method_order if m in set(summary["method"])]
        remaining = [m for m in summary["method"].unique() if m not in set(ordered)]
        order = ordered + remaining
        summary["method"] = pd.Categorical(summary["method"], categories=order, ordered=True)
        summary = summary.sort_values(["method", "n_samples"])
    else:
        order = list(summary["method"].unique())

    labels = method_labels or {}
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    if "n_samples" in summary.columns and summary["n_samples"].nunique() > 1:
        n_samples = sorted(summary["n_samples"].unique())
        x = np.arange(len(order))
        width = 0.8 / len(n_samples)
        colors = sns.color_palette("viridis", len(n_samples))
        
        for i, n in enumerate(n_samples):
            offset = (i - len(n_samples) / 2 + 0.5) * width
            mdf = summary[summary["n_samples"] == n].set_index("method").reindex(order)
            
            # Mask NaNs
            valid = ~mdf["fwer_max_tail"].isna()
            if not valid.any():
                continue
            
            # Use box-style representation for CIs
            _plot_cp_interval_boxes(
                ax,
                x[valid] + offset,
                mdf.loc[valid, "fwer_max_tail"].values,
                mdf.loc[valid, "fwer_ci_low"].values,
                mdf.loc[valid, "fwer_ci_high"].values,
                [colors[i]] * valid.sum(),
                width=width * 0.8,
                label=f"N={n}"
            )
        
        ax.set_xticks(x)
        ax.set_xticklabels([labels.get(m, m) for m in order], rotation=45, ha="right")
        # Custom legend to avoid repeating box-elements
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors[i], label=f"N={n}") for i, n in enumerate(n_samples)]
        ax.legend(handles=legend_elements, title="Sample Size", fontsize=8, title_fontsize=9)
    else:
        # Fallback to single dimension logic if n_samples not present
        x = np.arange(len(order))
        mdf = summary.set_index("method").reindex(order)
        colors = [METHOD_COLORS.get(method, "#42A5F5") for method in order]
        
        _plot_cp_interval_boxes(
            ax,
            x,
            mdf["fwer_max_tail"].values,
            mdf["fwer_ci_low"].values,
            mdf["fwer_ci_high"].values,
            colors,
            width=0.6
        )
        ax.set_xticks(x)
        ax.set_xticklabels([labels.get(m, m) for m in order], rotation=45, ha="right")

    ax.axhline(alpha, color="red", linestyle="--", alpha=0.6, label=f"Target alpha={alpha:.2f}")
    ax.set_ylim(0, max(0.1, float(summary["fwer_ci_high"].max(skipna=True)) * 1.25))
    ax.set_ylabel("Worst-tail directional FWER")
    ax.set_title("Directional Family-Wise Error Rate Control (95% CI)")
    ax.grid(axis="y", alpha=0.3)
    
    # Only add alpha to legend if not already there
    handles, legends = ax.get_legend_handles_labels()
    if f"Target alpha={alpha:.2f}" not in legends:
        handles.append(plt.Line2D([0], [0], color="red", linestyle="--", alpha=0.6))
        legends.append(f"Target alpha={alpha:.2f}")
        ax.legend(handles, legends, fontsize=8)

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / filename
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Fig 1 to {out}")
    return out


# =============================================================================
# HELPERS: per-sample-size dispatch
# =============================================================================

def _get_sample_sizes(df: pd.DataFrame) -> List[int]:
    """Return sorted unique sample sizes, or [None] if column missing."""
    if "n_samples" in df.columns:
        return sorted(df["n_samples"].unique())
    return [None]


def _n_suffix(n_samples) -> str:
    return f"_n{n_samples}" if n_samples is not None else ""


def _n_title(n_samples) -> str:
    return f"  (n_samples={n_samples})" if n_samples is not None else ""


# =============================================================================
# CLI
# =============================================================================

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot null FWER calibration results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "results_dir", type=Path,
        help="Directory containing fpr_results.csv (output of fpr_calibration.py)",
    )
    parser.add_argument(
        "--plot", nargs="+",
        choices=["overview", "per-method", "fwer-directions", "fwer-vs-n", "all"],
        default=["all"],
        help="Which plots to generate (default: all)",
    )
    parser.add_argument(
        "--method", nargs="+", default=None,
        help="Load only these methods (from per_method/ CSVs)",
    )
    parser.add_argument(
        "--list-methods", action="store_true",
        help="List available per-method CSVs and exit",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Significance threshold (default: 0.05)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for plots (default: <results_dir>/plots)",
    )

    args = parser.parse_args(argv)
    results_dir = args.results_dir

    if args.list_methods:
        methods = list_available_methods(results_dir)
        if methods:
            print(f"Available per-method CSVs in {results_dir / 'per_method'}:")
            for m in methods:
                print(f"  {m}")
        else:
            print(f"No per-method CSVs found in {results_dir / 'per_method'}")
        return 0

    # Load data
    df = load_results(results_dir, methods=args.method)
    alpha = args.alpha

    # Output directory
    output_dir = args.output_dir or results_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}\n")

    # Determine which plots to generate
    plot_types = set(args.plot)
    do_all = "all" in plot_types

    sample_sizes = _get_sample_sizes(df)
    has_multi_n = len(sample_sizes) > 1 and sample_sizes[0] is not None

    # --- Per-sample-size plots: overview, per-method, fwer-directions ---
    for n_samples in sample_sizes:
        if n_samples is not None:
            df_n = df[df["n_samples"] == n_samples]
        else:
            df_n = df
        summary_n = compute_fpr_summary_single(df_n, alpha=alpha)
        suffix = _n_suffix(n_samples)
        title_extra = _n_title(n_samples)

        if do_all or "overview" in plot_types:
            print(f"--- Overview{title_extra} ---")
            plot_overview(df_n, summary_n, alpha, output_dir, suffix, title_extra)

        if do_all or "per-method" in plot_types:
            print(f"--- Per-method detail{title_extra} ---")
            plot_per_method(df_n, alpha, output_dir, suffix, title_extra)

        if do_all or "fwer-directions" in plot_types:
            print(f"--- FWER directions{title_extra} ---")
            plot_fwer_directions(summary_n, alpha, output_dir, suffix, title_extra)

    # --- Cross-sample-size plot (only when multi-n) ---
    if (do_all and has_multi_n) or "fwer-vs-n" in plot_types:
        summary_full = compute_fpr_summary(df, alpha=alpha)
        print("--- FWER vs sample size ---")
        plot_fwer_vs_sample_size(summary_full, alpha, output_dir)

    print(f"\nAll plots saved to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
