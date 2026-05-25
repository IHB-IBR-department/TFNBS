"""
Prepare and plot compact TFNBS (E,H) sweep summaries.

The full power tables are useful as raw records, but they are too bulky for
quick replotting. This script extracts TFNBS-family methods, parses the
``method_params`` field into E/H columns, writes compact summary CSVs, and
plots TPR/FDR heatmaps from those summaries.

Usage:
    python examples/simulation_validation/power/tfnbs_eh_sweep.py

    python examples/simulation_validation/power/tfnbs_eh_sweep.py \
        --results-dir examples/simulation_validation/results/power/power_analysis_1000_30 \
        --effect-size 0.25 \
        --sample-size 30

    # Replot from existing summary CSVs only:
    python examples/simulation_validation/power/tfnbs_eh_sweep.py --plot-only

    # Paper panel: one TFNBS image with TPR above FDR:
    python examples/simulation_validation/power/tfnbs_eh_sweep.py \
        --plot-only --metric-rows --method tfnbs --topology-set first-row \
        --plot-dir examples/simulation_validation/results/plots
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_RESULTS_DIR = Path("examples/simulation_validation/results/power/power_analysis_1000_30")
TFNBS_METHODS = ("tfnbs", "ni_tfnbs", "fbc_tfnbs")
METRICS = ("TPR", "FDR", "precision", "FPR")
GROUP_COLS = ["method", "scenario", "effect_size", "n_samples", "e", "h"]
PAPER_CHOICE_E = 0.4
PAPER_CHOICE_H = 3.0

METHOD_LABELS = {
    "tfnbs": "TFNBS",
    "ni_tfnbs": "NI-TFNBS",
    "fbc_tfnbs": "FBC-TFNBS",
}

TOPOLOGY_ORDER = [
    "hub",
    "rich_club",
    "chain",
    "within_module_dense",
    "between_modules_dense",
    "within_plus_between",
    "partial_bipartite_between_modules",
    "gradient_core_periphery_within_module",
    "scattered_cross_block",
    "cross_block_connected_chain",
    "fragmented_within_module",
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

TOPOLOGY_SETS = {
    "all": TOPOLOGY_ORDER,
    "first-row": TOPOLOGY_ORDER[:5],
}


def _parse_param(method_params: object, key: str) -> float | None:
    """Extract a numeric method parameter from strings like ``e=0.3,h=3.0``."""
    if not isinstance(method_params, str):
        return None
    match = re.search(rf"\b{key}=([\d.]+)", method_params)
    return float(match.group(1)) if match else None


def _summarize_power_table(path: Path) -> pd.DataFrame:
    """Load one full power table and return compact TFNBS E/H summaries."""
    usecols = [
        "method",
        "method_params",
        "scenario",
        "effect_size",
        "n_samples",
        "repeat_id",
        *METRICS,
    ]
    df = pd.read_csv(path, usecols=usecols)
    df = df[df["method"].isin(TFNBS_METHODS)].copy()
    df["e"] = df["method_params"].map(lambda value: _parse_param(value, "e"))
    df["h"] = df["method_params"].map(lambda value: _parse_param(value, "h"))
    df = df.dropna(subset=["e", "h"])

    aggregations = {
        "repeat_id": "nunique",
    }
    for metric in METRICS:
        aggregations[metric] = ["mean", "std"]

    summary = df.groupby(GROUP_COLS, as_index=False).agg(aggregations)
    summary.columns = [
        "_".join(str(part) for part in col if part)
        if isinstance(col, tuple)
        else str(col)
        for col in summary.columns
    ]
    summary = summary.rename(columns={"repeat_id_nunique": "n_repeats"})

    ordered_cols = [
        *GROUP_COLS,
        "n_repeats",
        "TPR_mean",
        "TPR_std",
        "FDR_mean",
        "FDR_std",
        "precision_mean",
        "precision_std",
        "FPR_mean",
        "FPR_std",
    ]
    return summary[ordered_cols].sort_values(GROUP_COLS).reset_index(drop=True)


def build_summary_csvs(results_dir: Path, summary_dir: Path) -> list[Path]:
    """Write compact summary CSVs for effect-size and sample-size sweeps."""
    summary_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for sweep_name, filename in (
        ("effect_size", "power_vs_effect_size.csv"),
        ("sample_size", "power_vs_sample_size.csv"),
    ):
        source = results_dir / filename
        if not source.exists():
            print(f"Skipping {sweep_name}: {source} not found.")
            continue

        summary = _summarize_power_table(source)
        out_path = summary_dir / f"tfnbs_eh_sweep_{sweep_name}.csv"
        summary.to_csv(out_path, index=False)
        outputs.append(out_path)
        print(f"Wrote {out_path} ({len(summary):,} rows)")

    return outputs


def _available_scenarios(df: pd.DataFrame, requested: list[str] | None = None) -> list[str]:
    present = set(df["scenario"].unique())
    order = requested or TOPOLOGY_ORDER
    ordered = [scenario for scenario in order if scenario in present]
    if requested is None:
        ordered.extend(sorted(present.difference(ordered)))
    return ordered


def _metric_limits(metric: str) -> tuple[float, float, str]:
    if metric == "FDR":
        return 0.0, 0.2, "RdYlGn_r"
    if metric == "TPR":
        return 0.0, 1.0, "YlGnBu"
    return 0.0, 1.0, "viridis"


def _add_paper_choice_marker(
    ax,
    pivot: pd.DataFrame,
    paper_e: float,
    paper_h: float,
) -> None:
    """Draw the red default-parameter square used in validation sweeps."""
    e_values = [float(value) for value in pivot.columns]
    h_values = [float(value) for value in pivot.index]

    e_idx = next((idx for idx, value in enumerate(e_values) if abs(value - paper_e) < 1e-12), None)
    h_idx = next((idx for idx, value in enumerate(h_values) if abs(value - paper_h) < 1e-12), None)
    if e_idx is None or h_idx is None:
        return

    ax.add_patch(
        plt.Rectangle(
            (e_idx, h_idx),
            1,
            1,
            fill=False,
            edgecolor="red",
            linewidth=1.6,
            clip_on=False,
        )
    )


def _plot_one_heatmap_grid(
    df: pd.DataFrame,
    metric: str,
    method: str,
    selector_label: str,
    selector_value: float,
    plot_dir: Path,
    dpi: int,
    scenarios: list[str] | None,
    paper_e: float,
    paper_h: float,
    mark_paper_choice: bool,
) -> Path | None:
    """Plot one method/metric/value grid with scenarios as facets."""
    method_df = df[df["method"] == method]
    if method_df.empty:
        return None

    plot_scenarios = _available_scenarios(method_df, scenarios)
    if not plot_scenarios:
        return None

    n_cols = min(5, len(plot_scenarios))
    n_rows = (len(plot_scenarios) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.2 * n_cols + 0.8, 2.9 * n_rows),
        squeeze=False,
    )
    vmin, vmax, cmap = _metric_limits(metric)
    metric_col = f"{metric}_mean"

    for idx, scenario in enumerate(plot_scenarios):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        sub = method_df[method_df["scenario"] == scenario]
        pivot = sub.pivot_table(index="h", columns="e", values=metric_col, aggfunc="mean")
        pivot = pivot.sort_index(ascending=False)
        pivot = pivot[sorted(pivot.columns)]

        sns.heatmap(
            pivot,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            annot=False,
            cbar=idx == len(plot_scenarios) - 1,
            cbar_kws={"label": metric, "shrink": 0.75},
        )
        if mark_paper_choice:
            _add_paper_choice_marker(ax, pivot, paper_e, paper_h)
        ax.set_title(TOPOLOGY_LABELS.get(scenario, scenario), fontsize=8)
        ax.set_xlabel("E")
        ax.set_ylabel("H")

    for idx in range(len(plot_scenarios), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    title = (
        f"{METHOD_LABELS.get(method, method)} {metric} over (E,H), "
        f"{selector_label}={selector_value:g}"
    )
    fig.suptitle(title, y=1.01, fontsize=11, fontweight="bold")
    fig.tight_layout()

    safe_selector = f"{selector_value:g}".replace(".", "p")
    filename = (
        f"tfnbs_eh_{metric.lower()}_{method}_"
        f"{selector_label}_{safe_selector}.png"
    )
    out_path = plot_dir / filename
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_metric_rows(
    summary_path: Path,
    output_path: Path,
    selector_col: str,
    selector_value: float,
    method: str,
    metrics: list[str],
    scenarios: list[str] | None,
    paper_e: float,
    paper_h: float,
    mark_paper_choice: bool,
    dpi: int,
) -> Path:
    """Plot one method as metric rows and topology columns."""
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    df = pd.read_csv(summary_path)
    selected = df[
        (df["method"] == method)
        & (df[selector_col].astype(float).sub(float(selector_value)).abs() < 1e-12)
    ]
    if selected.empty:
        available = sorted(df[selector_col].unique())
        raise ValueError(
            f"No rows for method={method!r}, {selector_col}={selector_value}. "
            f"Available {selector_col}: {available}"
        )

    plot_scenarios = _available_scenarios(selected, scenarios)
    n_cols = len(plot_scenarios)
    n_rows = len(metrics)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.25 * n_cols + 0.8, 2.75 * n_rows),
        squeeze=False,
    )

    for row, metric in enumerate(metrics):
        vmin, vmax, cmap = _metric_limits(metric)
        metric_col = f"{metric}_mean"

        for col, scenario in enumerate(plot_scenarios):
            ax = axes[row, col]
            sub = selected[selected["scenario"] == scenario]
            pivot = sub.pivot_table(index="h", columns="e", values=metric_col, aggfunc="mean")
            pivot = pivot.sort_index(ascending=False)
            pivot = pivot[sorted(pivot.columns)]

            sns.heatmap(
                pivot,
                ax=ax,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                annot=False,
                cbar=col == n_cols - 1,
                cbar_kws={"label": metric, "shrink": 0.75},
            )
            if mark_paper_choice:
                _add_paper_choice_marker(ax, pivot, paper_e, paper_h)
            if row == 0:
                ax.set_title(TOPOLOGY_LABELS.get(scenario, scenario), fontsize=9)
            ax.set_xlabel("E")
            ax.set_ylabel(f"{metric}\nH" if col == 0 else "H")

    fig.suptitle(
        f"{METHOD_LABELS.get(method, method)} (E,H) sweep, {selector_col}={selector_value:g}",
        y=1.02,
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")
    return output_path


def _plot_summary_values(
    summary_path: Path,
    selector_col: str,
    selector_values: Iterable[float],
    metrics: Iterable[str],
    methods: Iterable[str],
    plot_dir: Path,
    dpi: int,
    scenarios: list[str] | None,
    paper_e: float,
    paper_h: float,
    mark_paper_choice: bool,
) -> list[Path]:
    """Plot selected values from one compact summary CSV."""
    if not summary_path.exists():
        print(f"Skipping plots: {summary_path} not found.")
        return []

    df = pd.read_csv(summary_path)
    outputs = []
    for selector_value in selector_values:
        selected = df[df[selector_col].astype(float).sub(float(selector_value)).abs() < 1e-12]
        if selected.empty:
            available = sorted(df[selector_col].unique())
            print(f"No rows for {selector_col}={selector_value}. Available: {available}")
            continue

        for metric in metrics:
            for method in methods:
                out_path = _plot_one_heatmap_grid(
                    selected,
                    metric,
                    method,
                    selector_col,
                    float(selector_value),
                    plot_dir,
                    dpi,
                    scenarios,
                    paper_e,
                    paper_h,
                    mark_paper_choice,
                )
                if out_path is not None:
                    outputs.append(out_path)
                    print(f"Wrote {out_path}")
    return outputs


def plot_from_summaries(
    summary_dir: Path,
    plot_dir: Path,
    effect_sizes: list[float],
    sample_sizes: list[float],
    metrics: list[str],
    methods: list[str],
    scenarios: list[str] | None,
    paper_e: float,
    paper_h: float,
    mark_paper_choice: bool,
    dpi: int,
) -> list[Path]:
    """Plot selected TFNBS E/H heatmaps from compact summary CSVs."""
    plot_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    outputs.extend(
        _plot_summary_values(
            summary_dir / "tfnbs_eh_sweep_effect_size.csv",
            "effect_size",
            effect_sizes,
            metrics,
            methods,
            plot_dir,
            dpi,
            scenarios,
            paper_e,
            paper_h,
            mark_paper_choice,
        )
    )
    if sample_sizes:
        outputs.extend(
            _plot_summary_values(
                summary_dir / "tfnbs_eh_sweep_sample_size.csv",
                "n_samples",
                sample_sizes,
                metrics,
                methods,
                plot_dir,
                dpi,
                scenarios,
                paper_e,
                paper_h,
                mark_paper_choice,
            )
        )
    return outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build compact TFNBS (E,H) sweep CSVs and plot TPR/FDR heatmaps.",
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--summary-dir", type=Path, default=None)
    parser.add_argument("--plot-dir", type=Path, default=None)
    parser.add_argument("--effect-size", type=float, nargs="+", default=[0.25])
    parser.add_argument("--sample-size", type=float, nargs="*", default=[])
    parser.add_argument("--metric", nargs="+", default=["TPR", "FDR"])
    parser.add_argument("--method", nargs="+", choices=TFNBS_METHODS, default=list(TFNBS_METHODS))
    parser.add_argument("--topology-set", choices=sorted(TOPOLOGY_SETS), default="all")
    parser.add_argument("--scenario", nargs="+", default=None)
    parser.add_argument("--paper-e", type=float, default=PAPER_CHOICE_E)
    parser.add_argument("--paper-h", type=float, default=PAPER_CHOICE_H)
    parser.add_argument("--no-paper-marker", action="store_true")
    parser.add_argument("--metric-rows", action="store_true")
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary_dir = args.summary_dir or args.results_dir / "derived"
    plot_dir = args.plot_dir or args.results_dir / "plots"
    metrics = [metric.upper() for metric in args.metric]
    invalid = sorted(set(metrics).difference(METRICS))
    if invalid:
        raise ValueError(f"Unsupported metric(s): {invalid}. Choose from {METRICS}.")
    scenarios = args.scenario or TOPOLOGY_SETS[args.topology_set]

    if not args.plot_only:
        build_summary_csvs(args.results_dir, summary_dir)

    if not args.summary_only:
        if args.metric_rows:
            if len(args.method) != 1:
                raise ValueError("--metric-rows expects exactly one --method value.")
            if len(args.effect_size) != 1:
                raise ValueError("--metric-rows expects exactly one --effect-size value.")
            output_file = args.output_file or plot_dir / "fig4_tfnbs_eh_tpr_fdr.png"
            plot_metric_rows(
                summary_dir / "tfnbs_eh_sweep_effect_size.csv",
                output_file,
                "effect_size",
                args.effect_size[0],
                args.method[0],
                metrics,
                scenarios,
                args.paper_e,
                args.paper_h,
                not args.no_paper_marker,
                args.dpi,
            )
        else:
            outputs = plot_from_summaries(
                summary_dir,
                plot_dir,
                args.effect_size,
                args.sample_size,
                metrics,
                list(args.method),
                scenarios,
                args.paper_e,
                args.paper_h,
                not args.no_paper_marker,
                args.dpi,
            )
            print(f"Generated {len(outputs)} heatmap figure(s).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
