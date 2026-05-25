"""
Exp 3: Analyze FBC-TFNBS m_min sensitivity results.

Shows TPR/FDR vs min_cluster_size for FBC-TFNBS across topologies.

Usage:
    # After running 5 sweeps with different fbc_min_cluster values:
    python examples/simulation_validation/sensitivity/analyze_mmin_sensitivity.py \
        --results-dirs examples/simulation_validation/results/mmin_1 examples/simulation_validation/results/mmin_2 examples/simulation_validation/results/mmin_3 examples/simulation_validation/results/mmin_5 examples/simulation_validation/results/mmin_10 \
        --mmin-values 1 2 3 5 10 \
        --output-dir examples/simulation_validation/results/mmin_comparison

Run the sweeps with:
    for M in 1 2 3 5 10; do
        python examples/simulation_validation/power/power_analysis.py \
            --config examples/simulation_validation/configs/sweep_config_mmin.yaml \
            --fbc-min-cluster $M \
            --output-dir examples/simulation_validation/results/mmin_$M
    done
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TOPOLOGIES = ["hub", "rich_club", "chain", "between_modules_dense"]

TOPO_LABELS = {
    "hub": "Hub",
    "rich_club": "Rich-club",
    "chain": "Chain",
    "between_modules_dense": "Between-module dense",
}


def load_and_tag(results_dir: Path, mmin_value: int) -> pd.DataFrame:
    """Load results and add m_min column."""
    path = results_dir / "power_vs_effect_size.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    df = pd.read_csv(path)
    df["m_min"] = mmin_value
    return df


def plot_mmin_sensitivity(df: pd.DataFrame, output_dir: Path):
    """Plot TPR and FDR vs m_min for each topology and effect size."""
    effect_sizes = sorted(df["effect_size"].unique())
    topologies = [t for t in TOPOLOGIES if t in df["scenario"].unique()]

    fig, axes = plt.subplots(
        2, len(topologies),
        figsize=(3.5 * len(topologies), 6),
        squeeze=False,
    )

    for col, topo in enumerate(topologies):
        sub = df[(df["scenario"] == topo) & (df["method"] == "fbc_tfnbs")]

        for es in effect_sizes:
            es_sub = sub[sub["effect_size"] == es]
            summary = es_sub.groupby("m_min").agg(
                TPR_mean=("TPR", "mean"),
                TPR_se=("TPR", lambda x: x.std() / np.sqrt(len(x))),
                FDR_mean=("FDR", "mean"),
                FDR_se=("FDR", lambda x: x.std() / np.sqrt(len(x))),
            ).reset_index()

            # TPR
            ax = axes[0, col]
            ax.errorbar(
                summary["m_min"], summary["TPR_mean"],
                yerr=1.96 * summary["TPR_se"],
                marker="o", label=f"d={es}",
            )
            ax.set_title(TOPO_LABELS.get(topo, topo))
            ax.set_ylim(-0.05, 1.05)
            if col == 0:
                ax.set_ylabel("TPR")
            ax.grid(True, alpha=0.3)

            # FDR
            ax = axes[1, col]
            ax.errorbar(
                summary["m_min"], summary["FDR_mean"],
                yerr=1.96 * summary["FDR_se"],
                marker="s", label=f"d={es}",
            )
            ax.axhline(0.05, color="black", linestyle=":", alpha=0.5)
            ax.set_ylim(-0.05, 0.5)
            ax.set_xlabel("$m_{\\mathrm{min}}$")
            if col == 0:
                ax.set_ylabel("FDR")
            ax.grid(True, alpha=0.3)

    axes[0, -1].legend(fontsize=7, loc="lower right")
    fig.suptitle("FBC-TFNBS: Sensitivity to $m_{\\mathrm{min}}$", fontsize=11)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        path = output_dir / f"mmin_sensitivity.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze m_min sensitivity")
    parser.add_argument("--results-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--mmin-values", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("examples/simulation_validation/results/mmin_comparison"))
    args = parser.parse_args()

    if len(args.results_dirs) != len(args.mmin_values):
        raise ValueError("--results-dirs and --mmin-values must have the same length")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    dfs = []
    for rdir, mmin in zip(args.results_dirs, args.mmin_values):
        dfs.append(load_and_tag(rdir, mmin))
        print(f"Loaded {rdir} (m_min={mmin}, {len(dfs[-1])} rows)")

    df_all = pd.concat(dfs, ignore_index=True)

    # Summary table
    summary = df_all[df_all["method"] == "fbc_tfnbs"].groupby(
        ["m_min", "scenario", "effect_size"]
    ).agg(
        TPR_mean=("TPR", "mean"),
        TPR_se=("TPR", lambda x: x.std() / np.sqrt(len(x))),
        FDR_mean=("FDR", "mean"),
        FDR_se=("FDR", lambda x: x.std() / np.sqrt(len(x))),
        n_repeats=("TPR", "count"),
    ).reset_index()

    out_path = args.output_dir / "mmin_sensitivity.csv"
    summary.to_csv(out_path, index=False)
    print(f"\nSaved {out_path} ({len(summary)} rows)")

    plot_mmin_sensitivity(df_all, args.output_dir)


if __name__ == "__main__":
    main()
