"""
Exp 4: Analyze NI-TFNBS normalization ablation results.

Compares 3 normalization variants for NI-TFNBS:
  - sqrt: W = k / sqrt(M_b)  (current default)
  - linear: W = k / M_b  (density normalization)
  - none: W = k  (raw active count)

Usage:
    # After running 3 sweeps with different ni_normalization:
    python examples/simulation_validation/sensitivity/analyze_norm_ablation.py \
        --sqrt-dir examples/simulation_validation/results/norm_sqrt \
        --linear-dir examples/simulation_validation/results/norm_linear \
        --none-dir examples/simulation_validation/results/norm_none \
        --output-dir examples/simulation_validation/results/norm_ablation_comparison

Run the sweeps with:
    for NORM in sqrt linear none; do
        python examples/simulation_validation/power/power_analysis.py \
            --config examples/simulation_validation/configs/sweep_config_norm_ablation.yaml \
            --ni-normalization $NORM \
            --output-dir examples/simulation_validation/results/norm_$NORM
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

NORM_LABELS = {
    "sqrt": r"$k / \sqrt{M_b}$",
    "linear": r"$k / M_b$",
    "none": r"$k$ (raw)",
}

NORM_COLORS = {
    "sqrt": "#2ca02c",
    "linear": "#d62728",
    "none": "#1f77b4",
}


def load_and_tag(results_dir: Path, norm_name: str) -> pd.DataFrame:
    """Load results and add normalization column."""
    path = results_dir / "power_vs_effect_size.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    df = pd.read_csv(path)
    df["normalization"] = norm_name
    return df


def plot_norm_ablation(df: pd.DataFrame, output_dir: Path):
    """Plot TPR and FDR vs effect size for each normalization and topology."""
    topologies = [t for t in TOPOLOGIES if t in df["scenario"].unique()]

    fig, axes = plt.subplots(
        2, len(topologies),
        figsize=(3.5 * len(topologies), 5),
        squeeze=False,
    )

    for col, topo in enumerate(topologies):
        for norm in ("sqrt", "linear", "none"):
            sub = df[
                (df["scenario"] == topo) &
                (df["method"] == "ni_tfnbs") &
                (df["normalization"] == norm)
            ]
            if sub.empty:
                continue

            summary = sub.groupby("effect_size").agg(
                TPR_mean=("TPR", "mean"),
                TPR_se=("TPR", lambda x: x.std() / np.sqrt(len(x))),
                FDR_mean=("FDR", "mean"),
                FDR_se=("FDR", lambda x: x.std() / np.sqrt(len(x))),
            ).reset_index()

            color = NORM_COLORS[norm]
            label = NORM_LABELS[norm]

            ax = axes[0, col]
            ax.errorbar(
                summary["effect_size"], summary["TPR_mean"],
                yerr=1.96 * summary["TPR_se"],
                marker="o", color=color, label=label,
            )
            ax.set_title(TOPO_LABELS.get(topo, topo))
            ax.set_ylim(-0.05, 1.05)
            if col == 0:
                ax.set_ylabel("TPR")
            ax.grid(True, alpha=0.3)

            ax = axes[1, col]
            ax.errorbar(
                summary["effect_size"], summary["FDR_mean"],
                yerr=1.96 * summary["FDR_se"],
                marker="s", color=color, label=label,
            )
            ax.axhline(0.05, color="black", linestyle=":", alpha=0.5)
            ax.set_ylim(-0.05, 0.5)
            ax.set_xlabel("Cohen's $d$")
            if col == 0:
                ax.set_ylabel("FDR")
            ax.grid(True, alpha=0.3)

    axes[0, -1].legend(fontsize=7, loc="lower right")
    fig.suptitle("NI-TFNBS: Normalization Ablation", fontsize=11)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        path = output_dir / f"norm_ablation.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze NI-TFNBS normalization ablation")
    parser.add_argument("--sqrt-dir", type=Path, required=True)
    parser.add_argument("--linear-dir", type=Path, required=True)
    parser.add_argument("--none-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("examples/simulation_validation/results/norm_ablation_comparison"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df_sqrt = load_and_tag(args.sqrt_dir, "sqrt")
    df_linear = load_and_tag(args.linear_dir, "linear")
    df_none = load_and_tag(args.none_dir, "none")

    print(f"sqrt: {len(df_sqrt)}, linear: {len(df_linear)}, none: {len(df_none)}")

    df_all = pd.concat([df_sqrt, df_linear, df_none], ignore_index=True)

    # Summary table
    ni_sub = df_all[df_all["method"] == "ni_tfnbs"]
    summary = ni_sub.groupby(
        ["normalization", "scenario", "effect_size"]
    ).agg(
        TPR_mean=("TPR", "mean"),
        TPR_se=("TPR", lambda x: x.std() / np.sqrt(len(x))),
        FDR_mean=("FDR", "mean"),
        FDR_se=("FDR", lambda x: x.std() / np.sqrt(len(x))),
        n_repeats=("TPR", "count"),
    ).reset_index()

    out_path = args.output_dir / "norm_ablation.csv"
    summary.to_csv(out_path, index=False)
    print(f"\nSaved {out_path} ({len(summary)} rows)")

    # Print comparison table
    print("\n=== NI-TFNBS Normalization Comparison ===\n")
    for topo in TOPOLOGIES:
        for es in sorted(ni_sub["effect_size"].unique()):
            print(f"  {topo} (d={es}):")
            for norm in ("sqrt", "linear", "none"):
                row = summary[
                    (summary["normalization"] == norm) &
                    (summary["scenario"] == topo) &
                    (summary["effect_size"] == es)
                ]
                if row.empty:
                    continue
                r = row.iloc[0]
                print(f"    {norm:8s}: TPR={r['TPR_mean']:.3f} +/- {r['TPR_se']:.3f}, "
                      f"FDR={r['FDR_mean']:.3f} +/- {r['FDR_se']:.3f}")
            print()

    plot_norm_ablation(df_all, args.output_dir)


if __name__ == "__main__":
    main()
