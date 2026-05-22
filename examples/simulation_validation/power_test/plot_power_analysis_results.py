"""
Plot Power Analysis Results.

Reads CSV output from power_analysis.py and produces:
1. Method comparison — all methods on one plot per scenario, metric
   averaged (with std) across all parameter variants
2. Per-method curves — one figure per method, all parameter variants
   as separate lines (top 10 for TFNBS methods with 176 variants)
3. TFNBS (e,h) heatmaps — metric as color, e on x-axis, h on y-axis,
   one heatmap per (tfnbs_method, scenario, effect_size) combination

Usage:
    # All plots from a results directory
    python plot_power_analysis_results.py examples/simulation_validation/results/power_analysis_local

    # Method comparison only
    python plot_power_analysis_results.py examples/simulation_validation/results/power_analysis_local \
        --plot comparison

    # Plot specific scenarios from per-scenario CSVs (fast, no full CSV load)
    python plot_power_analysis_results.py examples/simulation_validation/results/power_analysis_local \
        --scenario hub within_module_dense

    # List available per-scenario CSVs
    python plot_power_analysis_results.py examples/simulation_validation/results/power_analysis_local \
        --list-scenarios

    # Only (e,h) heatmaps for a specific effect size
    python plot_power_analysis_results.py examples/simulation_validation/results/power_analysis_local \
        --plot eh-heatmap --effect-size 0.25

    # Custom metric and output directory
    python plot_power_analysis_results.py examples/simulation_validation/results/power_analysis_local \
        --metric FDR --output-dir plots/
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TFNBS_METHODS = {"tfnbs", "ni_tfnbs", "fbc_tfnbs"}


def parse_param(method_params: str, key: str) -> Optional[float]:
    """Extract a numeric parameter from a method_params string like 'e=0.5,h=2.0,...'."""
    if not isinstance(method_params, str):
        return None
    m = re.search(rf"\b{key}=([\d.]+)", method_params)
    return float(m.group(1)) if m else None


def add_eh_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'e' and 'h' columns parsed from method_params."""
    df = df.copy()
    df["e"] = df["method_params"].apply(lambda s: parse_param(s, "e"))
    df["h"] = df["method_params"].apply(lambda s: parse_param(s, "h"))
    return df


def load_results(results_dir: Path, scenarios: Optional[List[str]] = None):
    """Load effect-size and sample-size CSVs.

    Parameters
    ----------
    results_dir : Path
        Directory with power analysis output.
    scenarios : list of str, optional
        If given, load only these scenarios from per_scenario/ subdir
        instead of the full combined CSVs.
    """
    df_effect = None
    df_sample = None

    if scenarios:
        per_dir = results_dir / "per_scenario"
        if not per_dir.exists():
            print(f"Error: {per_dir} does not exist. Run power_analysis.py first.")
            return None, None

        effect_parts, sample_parts = [], []
        for scenario in scenarios:
            p = per_dir / f"power_vs_effect_size_{scenario}.csv"
            if p.exists():
                effect_parts.append(pd.read_csv(p))
                print(f"  Loaded {p.name}: {len(effect_parts[-1])} rows")
            p = per_dir / f"power_vs_sample_size_{scenario}.csv"
            if p.exists():
                sample_parts.append(pd.read_csv(p))
                print(f"  Loaded {p.name}: {len(sample_parts[-1])} rows")

        if effect_parts:
            df_effect = pd.concat(effect_parts, ignore_index=True)
            print(f"Combined effect-size: {len(df_effect)} rows ({len(effect_parts)} scenarios)")
        if sample_parts:
            df_sample = pd.concat(sample_parts, ignore_index=True)
            print(f"Combined sample-size: {len(df_sample)} rows ({len(sample_parts)} scenarios)")
    else:
        p = results_dir / "power_vs_effect_size.csv"
        if p.exists():
            df_effect = pd.read_csv(p)
            print(f"Loaded {p.name}: {len(df_effect)} rows")
        p = results_dir / "power_vs_sample_size.csv"
        if p.exists():
            df_sample = pd.read_csv(p)
            print(f"Loaded {p.name}: {len(df_sample)} rows")

    return df_effect, df_sample


def list_available_scenarios(results_dir: Path) -> List[str]:
    """List scenarios available in per_scenario/ directory."""
    per_dir = results_dir / "per_scenario"
    if not per_dir.exists():
        return []
    names = set()
    for p in per_dir.glob("power_vs_*_*.csv"):
        # power_vs_effect_size_<scenario>.csv or power_vs_sample_size_<scenario>.csv
        stem = p.stem
        for prefix in ("power_vs_effect_size_", "power_vs_sample_size_"):
            if stem.startswith(prefix):
                names.add(stem[len(prefix):])
    return sorted(names)


# ---------------------------------------------------------------------------
# Plot 1: Per-method curves (all parameter variants on one plot)
# ---------------------------------------------------------------------------

MAX_LEGEND_VARIANTS = 15  # show legend only when variants <= this


def _clean_param_label(method: str, method_params) -> str:
    """Short label for per-method legends: keep only varying params."""
    if pd.isna(method_params) or not method_params:
        return "default"
    s = str(method_params)
    # Remove fixed params that don't vary for cleaner legends
    parts = s.split(",")
    cleaned = [p for p in parts if not p.strip().startswith(("start_thres=", "n=", "min_cluster_size="))]
    return ",".join(cleaned) if cleaned else "default"


def plot_curves_per_method(
    df: pd.DataFrame,
    x_col: str,
    metric: str,
    output_dir: Path,
    top_n: int = 10,
):
    """One figure per method, all parameter variants as separate lines.

    For methods with many variants (TFNBS: 176), shows top_n by mean metric
    to keep plots readable. Use (e,h) heatmaps for the full grid.
    """
    per_method_dir = output_dir / "per_method"
    per_method_dir.mkdir(parents=True, exist_ok=True)

    scenarios = sorted(df["scenario"].unique())
    methods = sorted(df["method"].unique())
    x_label = "Effect Size" if x_col == "effect_size" else "Sample Size (per group)"

    for method in methods:
        mdf = df[df["method"] == method].copy()
        mdf["label"] = mdf["method_params"].apply(lambda p: _clean_param_label(method, p))
        n_variants = mdf["label"].nunique()

        # For methods with many variants, keep top_n by mean metric
        truncated = False
        if n_variants > top_n:
            ranking = mdf.groupby("label")[metric].mean()
            if metric == "FDR":
                top_labels = ranking.nsmallest(top_n).index
            else:
                top_labels = ranking.nlargest(top_n).index
            mdf = mdf[mdf["label"].isin(top_labels)]
            truncated = True

        labels = sorted(mdf["label"].unique())
        palette = dict(zip(labels, sns.color_palette("husl", len(labels))))
        n = len(scenarios)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)

        for idx, scenario in enumerate(scenarios):
            ax = axes[0, idx]
            sdf = mdf[mdf["scenario"] == scenario]

            for label in labels:
                ldf = sdf[sdf["label"] == label]
                if ldf.empty:
                    continue
                summary = ldf.groupby(x_col)[metric].agg(["mean", "std"]).reset_index()
                show_legend = len(labels) <= MAX_LEGEND_VARIANTS
                ax.errorbar(
                    summary[x_col], summary["mean"], yerr=summary["std"],
                    label=label if show_legend else None,
                    color=palette[label],
                    marker="o", capsize=3, linewidth=2, alpha=0.8,
                )

            if metric == "TPR":
                ax.axhline(0.8, color="gray", ls="--", alpha=0.5)
                ax.set_ylim(-0.02, 1.05)
            elif metric == "FDR":
                ax.axhline(0.05, color="gray", ls="--", alpha=0.5)
                ax.set_ylim(-0.02, 0.6)
            else:
                ax.set_ylim(-0.02, 1.05)

            ax.set_xlabel(x_label)
            ax.set_ylabel(metric)
            ax.set_title(scenario)
            if len(labels) <= MAX_LEGEND_VARIANTS:
                ax.legend(loc="best", fontsize=7)
            ax.grid(True, alpha=0.3)

        title = f"{method}: {metric} vs {x_label}"
        if truncated:
            best_word = "lowest" if metric == "FDR" else "highest"
            title += f" (top {top_n} by {best_word} mean {metric})"
        title += f" [{n_variants} variants]"
        fig.suptitle(title, y=1.02, fontsize=13, fontweight="bold")
        fig.tight_layout()
        fname = f"{method}_{metric.lower()}_vs_{x_col}.png"
        fig.savefig(per_method_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved per_method/{fname}")


# ---------------------------------------------------------------------------
# Plot 2: Method comparison — one line per method, averaged across params
# ---------------------------------------------------------------------------

METHOD_COLORS = {
    "tstat": "#888888",
    "tfnbs": "#1f77b4",
    "ni_tfnbs": "#ff7f0e",
    "fbc_tfnbs": "#2ca02c",
    "nbs_extent": "#d62728",
    "nbs_intensity": "#9467bd",
    "cnbs": "#8c564b",
}


def plot_method_comparison(
    df: pd.DataFrame,
    x_col: str,
    metric: str,
    output_dir: Path,
):
    """All methods on one plot per scenario, averaged across parameter variants.

    For each method, the metric is averaged (with std) over all parameter
    variants and repeats at each x value. This gives a single curve per
    method, enabling direct comparison.
    """
    scenarios = sorted(df["scenario"].unique())
    methods = sorted(df["method"].unique())
    x_label = "Effect Size" if x_col == "effect_size" else "Sample Size (per group)"

    n = len(scenarios)
    n_cols = min(n, 5)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6 * n_cols, 5 * n_rows),
        squeeze=False,
    )

    for idx, scenario in enumerate(scenarios):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        sdf = df[df["scenario"] == scenario]

        for method in methods:
            mdf = sdf[sdf["method"] == method]
            if mdf.empty:
                continue
            # Average across all parameter variants and repeats
            summary = mdf.groupby(x_col)[metric].agg(["mean", "std"]).reset_index()
            color = METHOD_COLORS.get(method, None)
            ax.errorbar(
                summary[x_col], summary["mean"], yerr=summary["std"],
                label=method, color=color,
                marker="o", capsize=3, linewidth=2, alpha=0.85,
            )

        if metric == "TPR":
            ax.axhline(0.8, color="gray", ls="--", alpha=0.5)
            ax.set_ylim(-0.02, 1.05)
        elif metric == "FDR":
            ax.axhline(0.05, color="gray", ls="--", alpha=0.5)
            ax.set_ylim(-0.02, max(0.3, ax.get_ylim()[1]))
        else:
            ax.set_ylim(-0.02, 1.05)

        ax.set_xlabel(x_label)
        ax.set_ylabel(metric)
        ax.set_title(scenario, fontsize=10)
        ax.legend(loc="best", fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle(
        f"Method Comparison: {metric} vs {x_label} (avg over params)",
        y=1.01, fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    fname = f"method_comparison_{metric.lower()}_vs_{x_col}.png"
    fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


# ---------------------------------------------------------------------------
# Plot 3: (e, h) heatmaps for TFNBS methods
# ---------------------------------------------------------------------------

def plot_eh_heatmaps(
    df: pd.DataFrame,
    metric: str,
    output_dir: Path,
    x_col: str = "effect_size",
    fixed_values: Optional[List[float]] = None,
):
    """Draw (e,h) heatmaps for each TFNBS method.

    Creates a grid: rows = fixed_values (e.g. effect sizes), cols = scenarios.
    One figure per TFNBS method.

    Parameters
    ----------
    df : DataFrame
        Full results with method_params containing e=... and h=...
    metric : str
        'TPR', 'FDR', 'precision', etc.
    output_dir : Path
        Where to save plots.
    x_col : str
        Column with the swept variable ('effect_size' or 'n_samples').
    fixed_values : list of float, optional
        Which values of x_col to plot. If None, plots all.
    """
    df = add_eh_columns(df)
    tfnbs_df = df[df["method"].isin(TFNBS_METHODS) & df["e"].notna()].copy()
    if tfnbs_df.empty:
        print("  No TFNBS results with (e,h) parameters — skipping heatmaps")
        return

    scenarios = sorted(tfnbs_df["scenario"].unique())
    tfnbs_methods = sorted(tfnbs_df["method"].unique())

    if fixed_values is None:
        fixed_values = sorted(tfnbs_df[x_col].unique())

    x_label = "Effect Size" if x_col == "effect_size" else "Sample Size"

    for method in tfnbs_methods:
        mdf = tfnbs_df[tfnbs_df["method"] == method]
        n_rows = len(fixed_values)
        n_cols = len(scenarios)

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(5 * n_cols + 1, 4 * n_rows),
            squeeze=False,
        )

        if metric == "FDR":
            vmin, vmax = 0, 0.5
            cmap = "Reds"
        elif metric == "TPR":
            vmin, vmax = 0, 1.0
            cmap = "YlGnBu"
        else:
            vmin, vmax = 0, 1.0
            cmap = "viridis"

        for row, fv in enumerate(fixed_values):
            for col, scenario in enumerate(scenarios):
                ax = axes[row, col]
                sub = mdf[(mdf[x_col] == fv) & (mdf["scenario"] == scenario)]

                if sub.empty:
                    ax.set_visible(False)
                    continue

                pivot = sub.groupby(["h", "e"])[metric].mean().unstack("e")
                # Sort axes
                pivot = pivot.sort_index(ascending=False)  # h descending (top=high)
                pivot = pivot[sorted(pivot.columns)]        # e ascending (left=low)

                sns.heatmap(
                    pivot, ax=ax, vmin=vmin, vmax=vmax, cmap=cmap,
                    annot=True, fmt=".2f", annot_kws={"size": 7},
                    cbar_kws={"label": metric, "shrink": 0.8},
                )
                ax.set_xlabel("e (extent exponent)")
                ax.set_ylabel("h (height exponent)")
                ax.set_title(f"{scenario} | {x_label}={fv}")

        fig.suptitle(
            f"{method}: {metric} over (e, h) grid",
            y=1.01, fontsize=14, fontweight="bold",
        )
        fig.tight_layout()
        fname = f"eh_heatmap_{metric.lower()}_{method}_{x_col}.png"
        fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Plot power analysis results from CSV files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "results_dir", type=Path,
        help="Directory containing power_vs_effect_size.csv and/or power_vs_sample_size.csv",
    )
    parser.add_argument(
        "--plot", nargs="+",
        choices=["per-method", "eh-heatmap", "comparison", "all"],
        default=["all"],
        help="Which plot types to generate (default: all)",
    )
    parser.add_argument(
        "--metric", nargs="+", default=["TPR", "FDR"],
        help="Metrics to plot (default: TPR FDR)",
    )
    parser.add_argument(
        "--effect-size", type=float, nargs="*", default=None,
        help="Effect sizes for (e,h) heatmaps (default: all available)",
    )
    parser.add_argument(
        "--sample-size", type=int, nargs="*", default=None,
        help="Sample sizes for (e,h) heatmaps (default: all available)",
    )
    parser.add_argument(
        "--scenario", nargs="+", default=None,
        help="Load only these scenarios from per_scenario/ CSVs "
             "(faster than loading full combined files). "
             "Use --list-scenarios to see available names.",
    )
    parser.add_argument(
        "--list-scenarios", action="store_true",
        help="List available per-scenario CSVs and exit.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for plots (default: results_dir/plots)",
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="Resolution for saved figures",
    )

    args = parser.parse_args(argv)

    if not args.results_dir.exists():
        print(f"Error: {args.results_dir} does not exist")
        return 1

    if args.list_scenarios:
        available = list_available_scenarios(args.results_dir)
        if available:
            print(f"Available scenarios in {args.results_dir / 'per_scenario'}:")
            for s in available:
                print(f"  {s}")
        else:
            print("No per-scenario CSVs found. Run power_analysis.py first.")
        return 0

    output_dir = args.output_dir or args.results_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    do_all = "all" in args.plot
    df_effect, df_sample = load_results(args.results_dir, scenarios=args.scenario)

    if df_effect is None and df_sample is None:
        print("No CSV files found in results directory.")
        return 1

    print(f"Output: {output_dir}")
    print()

    for metric in args.metric:
        print(f"--- {metric} ---")

        # Method comparison: all methods on one plot per scenario
        if do_all or "comparison" in args.plot:
            if df_effect is not None:
                plot_method_comparison(df_effect, "effect_size", metric, output_dir)
            if df_sample is not None:
                plot_method_comparison(df_sample, "n_samples", metric, output_dir)

        # Per-method: separate plot per method with all parameter variants
        if do_all or "per-method" in args.plot:
            if df_effect is not None:
                plot_curves_per_method(df_effect, "effect_size", metric, output_dir)
            if df_sample is not None:
                plot_curves_per_method(df_sample, "n_samples", metric, output_dir)

        # (e,h) heatmaps for TFNBS methods
        if do_all or "eh-heatmap" in args.plot:
            if df_effect is not None:
                plot_eh_heatmaps(
                    df_effect, metric, output_dir,
                    x_col="effect_size",
                    fixed_values=args.effect_size,
                )
            if df_sample is not None:
                plot_eh_heatmaps(
                    df_sample, metric, output_dir,
                    x_col="n_samples",
                    fixed_values=[float(s) for s in args.sample_size] if args.sample_size else None,
                )

        print()

    print(f"All plots saved to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
