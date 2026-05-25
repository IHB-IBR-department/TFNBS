"""
Visualize benchmark CSV outputs.

Reads CSVs from ``examples/benchmarks/res_benchmarks/`` (or a user-specified
directory), produces one or two PNG plots per benchmark. Writes PNGs into
the same directory.

Each plot answers the scientific question of its benchmark:

    timing_benchmark   — how does wall-clock scale with N?
    precompsum         — how much does the sums fast path win, and where?
    precompsum_composed — how do pre-computation and GPD acceleration compose?
    acceleration       — do accelerated p-values match the reference?
    glm_pipeline       — which method is fastest, and which detects the planted effect?
    glm_stat           — how does GLM stat computation throughput scale with N?

Usage
-----

    # Default: reads res_benchmarks/, writes PNGs there
    python examples/benchmarks/plot_results.py

    # Custom input/output directories
    python examples/benchmarks/plot_results.py \\
        --csv-dir path/to/csvs --output-dir path/to/pngs

    # Only plot a subset (matching filenames without .csv)
    python examples/benchmarks/plot_results.py --only timing_benchmark precompsum

Missing CSVs are skipped with a note. This makes the script safe to run
after partial benchmark runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_DIR = Path(__file__).parent / "res_benchmarks"


# -----------------------------------------------------------------------------
# timing_benchmark.csv
# -----------------------------------------------------------------------------

def plot_timing_benchmark(df: pd.DataFrame, out_path: Path) -> None:
    """Two panels: wall-clock bars per method × N, and per-perm cost log-log."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: wall-clock bar chart
    ax = axes[0]
    methods = df["method"].unique()
    n_nodes_list = sorted(df["n_nodes"].unique())
    x = np.arange(len(n_nodes_list))
    width = 0.8 / len(methods)

    for i, m in enumerate(methods):
        sub = df[df["method"] == m]
        # If multiple n_perms present, pick the largest for the bar chart
        sub = sub[sub["n_permutations"] == sub["n_permutations"].max()]
        sub = sub.groupby("n_nodes")["time_mean_s"].mean().reindex(n_nodes_list)
        ax.bar(x + i * width, sub.values, width, label=m)

    ax.set_xlabel("n_nodes")
    ax.set_ylabel("wall-clock (s)")
    ax.set_title(f"Wall-clock time (n_perms={df['n_permutations'].max()})")
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(n_nodes_list)
    # ax.set_yscale("log")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 2: per-permutation scaling (log-log)
    ax = axes[1]
    for m in methods:
        sub = df[df["method"] == m]
        if (sub["per_perm_ms"] == 0).all():
            continue  # parametric methods; no permutations
        sub = sub.groupby("n_nodes")["per_perm_ms"].mean().reindex(n_nodes_list)
        ax.plot(n_nodes_list, sub.values, marker="o", label=m)

    ax.set_xlabel("n_nodes")
    ax.set_ylabel("per-permutation time (ms)")
    ax.set_title("Per-permutation scaling (log-log)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# precompsum.csv
# -----------------------------------------------------------------------------

def plot_precompsum(df: pd.DataFrame, out_path: Path) -> None:
    """Two panels: speedup per case × topology; speedup vs N averaged across topologies."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: speedup grouped by topology, split by case
    ax = axes[0]
    # Average over (N, n_sub, n_perm) to get one speedup per (case, topology)
    agg = df.groupby(["case", "topology"])["speedup"].mean().unstack("topology")
    agg.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylabel("mean speedup (slow / fast)")
    ax.set_xlabel("case")
    ax.set_title("Fast-path speedup by topology and case")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(title="topology", fontsize=8, loc="best")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.5)
    ax.grid(True, axis="y", alpha=0.3)

    # Panel 2: speedup vs N for each case (averaged over topologies + n_sub + n_perm)
    ax = axes[1]
    for case, sub in df.groupby("case"):
        agg_N = sub.groupby("N")["speedup"].mean()
        ax.plot(agg_N.index, agg_N.values, marker="o", label=case)
    ax.set_xlabel("N (nodes)")
    ax.set_ylabel("mean speedup")
    ax.set_title("Fast-path speedup vs N (averaged over topologies)")
    ax.set_xscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# precompsum_composed.csv
# -----------------------------------------------------------------------------

def plot_precompsum_composed(df: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar: baseline, +gpd, +precomp, +both per (N, n_sub)."""
    fig, ax = plt.subplots(figsize=(11, 5))

    df = df.copy()
    df["label"] = df.apply(lambda r: f"N={int(r['N'])}, n={int(r['n_sub'])}", axis=1)
    cols = ["t_baseline", "t_gpd_only", "t_precomp_only", "t_both"]
    names = ["baseline", "+GPD only", "+precomp only", "both"]
    x = np.arange(len(df))
    width = 0.2
    for i, (c, n) in enumerate(zip(cols, names)):
        ax.bar(x + i * width, df[c].values, width, label=n)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(df["label"].values, rotation=15)
    ax.set_ylabel("wall-clock (s, log scale)")
    ax.set_yscale("log")
    ax.set_title("Pre-computation × GPD acceleration (two-sample, within_module_dense)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# acceleration.csv
# -----------------------------------------------------------------------------

def plot_acceleration(df: pd.DataFrame, out_path: Path) -> None:
    """Two panels: p_planted (accel) vs p_planted (ref); MAE per method × accel × topology."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: scatter of accelerated p_planted against reference p_planted
    ax = axes[0]
    # Pair each accelerated row with its reference (same N × topology × method)
    ref = df[df["accel"] == "none"].set_index(["N", "topology", "method"])
    accel_rows = df[df["accel"].isin(["gpd", "gamma", "emp"])]

    colors = {"gpd": "C0", "gamma": "C1", "emp": "C2"}
    for accel_name, sub in accel_rows.groupby("accel"):
        p_refs = []
        p_accels = []
        for _, row in sub.iterrows():
            key = (row["N"], row["topology"], row["method"])
            if key in ref.index:
                p_refs.append(ref.loc[key, "p_planted"])
                p_accels.append(row["p_planted"])
        ax.scatter(p_refs, p_accels, c=colors.get(accel_name, "gray"),
                   label=accel_name, alpha=0.7, s=40)
    # y=x reference line
    vmin = df["p_planted"].min()
    vmax = df["p_planted"].max()
    ax.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=0.6, alpha=0.5)
    ax.set_xlabel("reference p_planted (5000 perms)")
    ax.set_ylabel("accelerated p_planted (200 perms)")
    ax.set_title("Planted-edge p-values: accelerated vs reference")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: MAE bar chart per method × accel (averaged over topology × N)
    ax = axes[1]
    mae_agg = (accel_rows.groupby(["method", "accel"])["mae"].mean()
               .unstack("accel"))
    mae_agg.plot(kind="bar", ax=ax, width=0.75)
    ax.set_ylabel("mean |p_accel − p_ref|  (averaged over topologies, N)")
    ax.set_xlabel("method")
    ax.set_title("Accelerated-vs-reference p-value MAE")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(title="accel", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# glm_pipeline.csv
# -----------------------------------------------------------------------------

def plot_glm_pipeline(df: pd.DataFrame, out_path: Path) -> None:
    """Two panels: time per method × N (bar); planted-p heatmap topology × method."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: wall-clock per method × N (averaged over topologies)
    ax = axes[0]
    time_agg = df.groupby(["N", "method"])["time_s"].mean().unstack("method")
    time_agg.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylabel("wall-clock (s, log)")
    ax.set_xlabel("N")
    ax.set_yscale("log")
    ax.set_title(f"GLM pipeline time per method × N (n_perms={df['n_perms'].max()})")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(title="method", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3, which="both")

    # Panel 2: p_planted heatmap (topology × method), at largest N
    ax = axes[1]
    N_max = df["N"].max()
    sub = df[df["N"] == N_max]
    heat = sub.pivot_table(
        index="topology", columns="method", values="p_planted", aggfunc="min",
    )
    im = ax.imshow(heat.values, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=0.1)
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_xticklabels(heat.columns, rotation=30, ha="right")
    ax.set_yticklabels(heat.index)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            v = heat.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        color="white" if v > 0.05 else "black", fontsize=8)
    ax.set_title(f"planted-edge p-value at N={N_max} (lower = better detection)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="p_planted")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# glm_stat.csv
# -----------------------------------------------------------------------------

def plot_glm_stat(df: pd.DataFrame, out_path: Path) -> None:
    """Log-log: edges/s vs N, one line per topology."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for topo, sub in df.groupby("topology"):
        sub = sub.sort_values("N")
        ax.plot(sub["N"], sub["edges_per_s"], marker="o", label=topo)
    ax.set_xlabel("N (nodes)")
    ax.set_ylabel("edges / second")
    ax.set_title("compute_glm_stat throughput vs N")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

PLOTTERS = {
    "timing_benchmark": plot_timing_benchmark,
    "precompsum": plot_precompsum,
    "precompsum_composed": plot_precompsum_composed,
    "acceleration": plot_acceleration,
    "glm_pipeline": plot_glm_pipeline,
    "glm_stat": plot_glm_stat,
}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--csv-dir", type=Path, default=DEFAULT_DIR,
                        help=f"Directory with benchmark CSVs (default: {DEFAULT_DIR})")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory to write PNGs (default: same as --csv-dir)")
    parser.add_argument("--only", nargs="+", choices=sorted(PLOTTERS.keys()),
                        help="Only plot these benchmarks (by name, without .csv)")
    args = parser.parse_args(argv)

    csv_dir = args.csv_dir
    out_dir = args.output_dir or csv_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = args.only or list(PLOTTERS.keys())

    made = 0
    for name in targets:
        csv_path = csv_dir / f"{name}.csv"
        if not csv_path.exists():
            print(f"[skip] {csv_path} not found")
            continue
        png_path = out_dir / f"{name}.png"
        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"[skip] {csv_path} is empty")
            continue
        PLOTTERS[name](df, png_path)
        print(f"[ok]   {png_path}  ({len(df)} rows)")
        made += 1

    if made == 0:
        print("No plots generated. Run the benchmarks first:")
        print("  python examples/benchmarks/benchmark_precompsum.py")
        print("  python examples/benchmarks/timing_benchmark.py")
        print("  python examples/benchmarks/benchmark_acceleration.py")
        print("  python examples/benchmarks/benchmark_glm.py")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
