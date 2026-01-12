"""
Power Analysis for TFNBS Methods.

This script computes statistical power curves for all methods across:
1. Effect sizes (at fixed sample size)
2. Sample sizes (at fixed effect size)

Power is defined as the proportion of true edges that are detected
(True Positive Rate / Sensitivity) at the nominal alpha level.

Usage:
    # Power vs Effect Size
    python power_analysis.py --mode effect-size \
        --effect-sizes 0.10 0.15 0.20 0.25 0.30 0.40 0.50 \
        --n-repeats 50 --n-permutations 500

    # Power vs Sample Size
    python power_analysis.py --mode sample-size \
        --sample-sizes 10 15 20 30 50 \
        --effect-size 0.25 \
        --n-repeats 50

    # Both analyses
    python power_analysis.py --mode both

Output:
    - results/power_analysis/power_vs_effect_size.csv
    - results/power_analysis/power_vs_sample_size.csv
    - results/power_analysis/power_curves.png
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tfnbs.pairwise_stats import compute_p_val
from tfnbs.synth_datasets import ModularDatasetGenerator


@dataclass
class PowerResult:
    """Result from a single power simulation."""

    method: str
    scenario: str
    effect_size: float
    n_samples: int
    repeat_id: int
    n_true: int       # Number of true effect edges
    n_null: int       # Number of null edges
    TP: int           # True Positives
    FP: int           # False Positives
    FN: int           # False Negatives
    TN: int           # True Negatives
    TPR: float        # Sensitivity/Power
    FPR: float        # False Positive Rate
    precision: float
    F1: float
    elapsed_time: float


def create_effect_mask(scenario: str, n_nodes: int, n_modules: int) -> np.ndarray:
    """Create effect mask for different scenarios."""

    nodes_per_module = n_nodes // n_modules
    mask = np.zeros((n_nodes, n_nodes), dtype=bool)

    if scenario == "within_module_dense":
        # Dense block within first module
        mask[:nodes_per_module, :nodes_per_module] = True

    elif scenario == "chain":
        # Chain of edges spanning multiple modules
        for i in range(n_nodes - 1):
            if i % 2 == 0:  # Create chain pattern
                mask[i, i + 1] = True
                mask[i + 1, i] = True

    elif scenario == "hub":
        # Hub node connected to many others
        hub_node = 0
        for i in range(1, min(n_nodes // 2, 30)):
            mask[hub_node, i] = True
            mask[i, hub_node] = True

    elif scenario == "scattered_cross_block":
        # Scattered edges across blocks
        rng = np.random.default_rng(42)
        n_edges = 60
        for _ in range(n_edges):
            i = rng.integers(0, n_nodes)
            j = rng.integers(0, n_nodes)
            if i != j:
                mask[i, j] = True
                mask[j, i] = True

    elif scenario == "between_modules":
        # Dense block between first two modules
        mask[:nodes_per_module, nodes_per_module:2*nodes_per_module] = True
        mask[nodes_per_module:2*nodes_per_module, :nodes_per_module] = True

    else:
        # Default: within first module
        mask[:nodes_per_module, :nodes_per_module] = True

    # Ensure symmetric and zero diagonal
    mask = mask | mask.T
    np.fill_diagonal(mask, False)

    return mask


def run_single_power_test(
    method: str,
    scenario: str,
    effect_size: float,
    n_samples: int,
    repeat_id: int,
    n_nodes: int = 60,
    n_modules: int = 4,
    n_permutations: int = 500,
    alpha: float = 0.05,
    use_mp: bool = False,
    seed_base: int = 42,
    method_kwargs: dict = None,
) -> PowerResult:
    """Run a single power simulation."""

    start_time = time.time()

    # Create effect mask
    effect_mask = create_effect_mask(scenario, n_nodes, n_modules)

    # Generate data with effect
    seed = seed_base + repeat_id * 1000 + hash(scenario) % 10000
    generator = ModularDatasetGenerator(
        n_nodes=n_nodes,
        n_modules=n_modules,
        intra_corr=0.3,
        inter_corr=0.05,
        noise_level=0.05,
        seed=seed,
    )

    group1, group2, net_labels = generator.generate_data(
        effect_size=effect_size,
        n_samples=n_samples,
        effect_mask=effect_mask,
    )

    # Apply Fisher z-transform
    group1_z = np.arctanh(np.clip(group1, -0.999, 0.999))
    group2_z = np.arctanh(np.clip(group2, -0.999, 0.999))

    # Prepare method kwargs
    kwargs = method_kwargs or {}
    if method in ["cnbs", "ni_tfnbs", "fbc_tfnbs"]:
        kwargs["net_labels"] = net_labels

    # Compute p-values
    method_key = method.split("_")[0] if method.startswith("nbs_") else method
    if method.startswith("nbs_"):
        kwargs["nbs_stat"] = method.split("_")[1]

    try:
        p_vals = compute_p_val(
            group1_z, group2_z,
            test_type="two-sample",
            method=method_key,
            n_permutations=n_permutations,
            use_mp=use_mp,
            random_state=seed,
            **kwargs,
        )

        # Get p-values for positive direction (group2 > group1)
        p_pos = np.array(p_vals["g2>g1"])

        # Get upper triangle
        triu_idx = np.triu_indices(n_nodes, k=1)
        p_upper = p_pos[triu_idx]
        mask_upper = effect_mask[triu_idx]

        # Compute metrics
        sig_mask = p_upper < alpha

        TP = int(np.sum(sig_mask & mask_upper))
        FP = int(np.sum(sig_mask & ~mask_upper))
        FN = int(np.sum(~sig_mask & mask_upper))
        TN = int(np.sum(~sig_mask & ~mask_upper))

        n_true = int(np.sum(mask_upper))
        n_null = int(np.sum(~mask_upper))

        TPR = TP / n_true if n_true > 0 else 0.0
        FPR = FP / n_null if n_null > 0 else 0.0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        F1 = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0.0

    except Exception as e:
        print(f"  ERROR in {method} repeat {repeat_id}: {e}")
        n_true = int(np.sum(effect_mask[np.triu_indices(n_nodes, k=1)]))
        n_null = n_nodes * (n_nodes - 1) // 2 - n_true
        TP, FP, FN, TN = 0, 0, n_true, n_null
        TPR, FPR, precision, F1 = 0.0, 0.0, 0.0, 0.0

    elapsed = time.time() - start_time

    return PowerResult(
        method=method,
        scenario=scenario,
        effect_size=effect_size,
        n_samples=n_samples,
        repeat_id=repeat_id,
        n_true=n_true,
        n_null=n_null,
        TP=TP, FP=FP, FN=FN, TN=TN,
        TPR=TPR, FPR=FPR,
        precision=precision, F1=F1,
        elapsed_time=elapsed,
    )


def get_method_configs() -> Dict[str, dict]:
    """Get default configuration for each method."""
    return {
        "tstat": {},
        "tfnbs": {"e": 0.5, "h": 2.0, "n": 50, "start_thres": 1.65},
        "ni_tfnbs": {"e": 0.5, "h": 2.0, "n": 50, "start_thres": 1.65},
        "fbc_tfnbs": {"e": 0.5, "h": 2.0, "n": 50, "start_thres": 1.65, "min_cluster_size": 3},
        "nbs_extent": {"threshold": 2.0},
        "nbs_intensity": {"threshold": 2.0},
        "cnbs": {},
    }


def run_power_vs_effect_size(
    methods: List[str],
    scenarios: List[str],
    effect_sizes: List[float],
    n_samples: int = 20,
    n_repeats: int = 50,
    n_nodes: int = 60,
    n_modules: int = 4,
    n_permutations: int = 500,
    alpha: float = 0.05,
    use_mp: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """Run power analysis across effect sizes."""

    method_configs = get_method_configs()
    results = []

    total_runs = len(methods) * len(scenarios) * len(effect_sizes) * n_repeats
    run_count = 0

    print(f"=== Power vs Effect Size ===")
    print(f"Methods: {methods}")
    print(f"Scenarios: {scenarios}")
    print(f"Effect sizes: {effect_sizes}")
    print(f"Repeats per condition: {n_repeats}")
    print(f"Total runs: {total_runs}")
    print()

    for scenario in scenarios:
        print(f"\nScenario: {scenario}")

        for effect_size in effect_sizes:
            print(f"  Effect size: {effect_size}")

            for method in methods:
                kwargs = method_configs.get(method, {})

                for repeat_id in range(n_repeats):
                    run_count += 1

                    result = run_single_power_test(
                        method=method,
                        scenario=scenario,
                        effect_size=effect_size,
                        n_samples=n_samples,
                        repeat_id=repeat_id,
                        n_nodes=n_nodes,
                        n_modules=n_modules,
                        n_permutations=n_permutations,
                        alpha=alpha,
                        use_mp=use_mp,
                        seed_base=seed,
                        method_kwargs=kwargs,
                    )
                    results.append(result)

                # Print progress
                method_results = [r for r in results
                                  if r.method == method
                                  and r.scenario == scenario
                                  and r.effect_size == effect_size]
                mean_tpr = np.mean([r.TPR for r in method_results])
                print(f"    {method}: TPR={mean_tpr:.3f} ({run_count}/{total_runs})")

    return pd.DataFrame([vars(r) for r in results])


def run_power_vs_sample_size(
    methods: List[str],
    scenarios: List[str],
    sample_sizes: List[int],
    effect_size: float = 0.25,
    n_repeats: int = 50,
    n_nodes: int = 60,
    n_modules: int = 4,
    n_permutations: int = 500,
    alpha: float = 0.05,
    use_mp: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """Run power analysis across sample sizes."""

    method_configs = get_method_configs()
    results = []

    total_runs = len(methods) * len(scenarios) * len(sample_sizes) * n_repeats
    run_count = 0

    print(f"=== Power vs Sample Size ===")
    print(f"Methods: {methods}")
    print(f"Scenarios: {scenarios}")
    print(f"Sample sizes: {sample_sizes}")
    print(f"Effect size: {effect_size}")
    print(f"Repeats per condition: {n_repeats}")
    print()

    for scenario in scenarios:
        print(f"\nScenario: {scenario}")

        for n_samples in sample_sizes:
            print(f"  Sample size: {n_samples}")

            for method in methods:
                kwargs = method_configs.get(method, {})

                for repeat_id in range(n_repeats):
                    run_count += 1

                    result = run_single_power_test(
                        method=method,
                        scenario=scenario,
                        effect_size=effect_size,
                        n_samples=n_samples,
                        repeat_id=repeat_id,
                        n_nodes=n_nodes,
                        n_modules=n_modules,
                        n_permutations=n_permutations,
                        alpha=alpha,
                        use_mp=use_mp,
                        seed_base=seed,
                        method_kwargs=kwargs,
                    )
                    results.append(result)

                method_results = [r for r in results
                                  if r.method == method
                                  and r.scenario == scenario
                                  and r.n_samples == n_samples]
                mean_tpr = np.mean([r.TPR for r in method_results])
                print(f"    {method}: TPR={mean_tpr:.3f} ({run_count}/{total_runs})")

    return pd.DataFrame([vars(r) for r in results])


def summarize_power_results(df: pd.DataFrame, groupby_cols: List[str]) -> pd.DataFrame:
    """Summarize power results by grouping columns."""

    summary = df.groupby(groupby_cols).agg({
        "TPR": ["mean", "std", "count"],
        "FPR": ["mean", "std"],
        "precision": ["mean", "std"],
        "F1": ["mean", "std"],
    }).reset_index()

    # Flatten column names
    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]

    return summary


def plot_power_curves(
    df_effect: pd.DataFrame,
    df_sample: pd.DataFrame,
    output_dir: Path,
):
    """Create power curve plots."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not available, skipping plots")
        return

    # Color palette for methods
    methods = df_effect["method"].unique()
    palette = dict(zip(methods, sns.color_palette("husl", len(methods))))

    scenarios = df_effect["scenario"].unique()

    # Plot 1: Power vs Effect Size (one subplot per scenario)
    n_scenarios = len(scenarios)
    fig, axes = plt.subplots(1, n_scenarios, figsize=(5 * n_scenarios, 4), squeeze=False)

    for idx, scenario in enumerate(scenarios):
        ax = axes[0, idx]
        scenario_df = df_effect[df_effect["scenario"] == scenario]

        for method in methods:
            method_df = scenario_df[scenario_df["method"] == method]
            summary = method_df.groupby("effect_size")["TPR"].agg(["mean", "std"]).reset_index()
            ax.errorbar(
                summary["effect_size"], summary["mean"],
                yerr=summary["std"],
                label=method, color=palette[method],
                marker="o", capsize=3, linewidth=2,
            )

        ax.axhline(0.8, color="gray", linestyle="--", alpha=0.5, label="80% power")
        ax.set_xlabel("Effect Size")
        ax.set_ylabel("Power (TPR)")
        ax.set_title(f"{scenario}")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Power vs Effect Size by Scenario", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "power_vs_effect_size.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 2: Power vs Sample Size
    if df_sample is not None and len(df_sample) > 0:
        fig, axes = plt.subplots(1, n_scenarios, figsize=(5 * n_scenarios, 4), squeeze=False)

        for idx, scenario in enumerate(scenarios):
            ax = axes[0, idx]
            scenario_df = df_sample[df_sample["scenario"] == scenario]

            for method in methods:
                method_df = scenario_df[scenario_df["method"] == method]
                summary = method_df.groupby("n_samples")["TPR"].agg(["mean", "std"]).reset_index()
                ax.errorbar(
                    summary["n_samples"], summary["mean"],
                    yerr=summary["std"],
                    label=method, color=palette[method],
                    marker="o", capsize=3, linewidth=2,
                )

            ax.axhline(0.8, color="gray", linestyle="--", alpha=0.5)
            ax.set_xlabel("Sample Size (per group)")
            ax.set_ylabel("Power (TPR)")
            ax.set_title(f"{scenario}")
            ax.set_ylim(0, 1.05)
            ax.legend(loc="lower right", fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.suptitle("Power vs Sample Size by Scenario", y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / "power_vs_sample_size.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Plot 3: Combined summary heatmap (F1 by method x scenario at ES=0.25)
    if "effect_size" in df_effect.columns:
        pivot_df = df_effect[df_effect["effect_size"] == 0.25].groupby(
            ["method", "scenario"]
        )["F1"].mean().unstack()

        if not pivot_df.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                pivot_df, annot=True, fmt=".2f", cmap="RdYlGn",
                vmin=0, vmax=1, ax=ax, cbar_kws={"label": "F1 Score"}
            )
            ax.set_title("F1 Score by Method and Scenario (Effect Size = 0.25)")
            plt.tight_layout()
            plt.savefig(output_dir / "f1_heatmap.png", dpi=150, bbox_inches="tight")
            plt.close()

    print(f"Saved plots to {output_dir}")


def compute_required_sample_size(
    df: pd.DataFrame,
    target_power: float = 0.8,
) -> pd.DataFrame:
    """Estimate required sample size to achieve target power."""

    results = []

    for (method, scenario, effect_size), group in df.groupby(["method", "scenario", "effect_size"]):
        power_by_n = group.groupby("n_samples")["TPR"].mean()

        # Find smallest n with power >= target
        required_n = None
        for n in sorted(power_by_n.index):
            if power_by_n[n] >= target_power:
                required_n = n
                break

        results.append({
            "method": method,
            "scenario": scenario,
            "effect_size": effect_size,
            "target_power": target_power,
            "required_n": required_n,
            "achieved_power": power_by_n.get(required_n, np.nan) if required_n else np.nan,
        })

    return pd.DataFrame(results)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Power analysis for TFNBS methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--mode", choices=["effect-size", "sample-size", "both"],
        default="both", help="Analysis mode",
    )
    parser.add_argument(
        "--effect-sizes", type=float, nargs="+",
        default=[0.10, 0.15, 0.20, 0.25, 0.30, 0.40],
        help="Effect sizes to test",
    )
    parser.add_argument(
        "--sample-sizes", type=int, nargs="+",
        default=[10, 15, 20, 30, 50],
        help="Sample sizes to test",
    )
    parser.add_argument(
        "--effect-size", type=float, default=0.25,
        help="Fixed effect size for sample size analysis",
    )
    parser.add_argument(
        "--n-samples", type=int, default=20,
        help="Fixed sample size for effect size analysis",
    )
    parser.add_argument(
        "--scenarios", nargs="+",
        default=["within_module_dense", "chain", "hub", "scattered_cross_block"],
        help="Scenarios to test",
    )
    parser.add_argument(
        "--methods", nargs="+",
        default=["tstat", "tfnbs", "ni_tfnbs", "fbc_tfnbs", "nbs_extent", "cnbs"],
        help="Methods to test",
    )
    parser.add_argument(
        "--n-repeats", type=int, default=50,
        help="Repeats per condition",
    )
    parser.add_argument(
        "--n-permutations", type=int, default=500,
        help="Permutations per test",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Significance threshold",
    )
    parser.add_argument(
        "--n-nodes", type=int, default=60,
        help="Number of nodes",
    )
    parser.add_argument(
        "--n-modules", type=int, default=4,
        help="Number of modules",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--use-mp", action="store_true",
        help="Use multiprocessing",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).parent.parent / "results" / "power_analysis",
        help="Output directory",
    )

    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df_effect = None
    df_sample = None

    # Run effect size analysis
    if args.mode in ["effect-size", "both"]:
        df_effect = run_power_vs_effect_size(
            methods=args.methods,
            scenarios=args.scenarios,
            effect_sizes=args.effect_sizes,
            n_samples=args.n_samples,
            n_repeats=args.n_repeats,
            n_nodes=args.n_nodes,
            n_modules=args.n_modules,
            n_permutations=args.n_permutations,
            alpha=args.alpha,
            use_mp=args.use_mp,
            seed=args.seed,
        )
        df_effect.to_csv(args.output_dir / "power_vs_effect_size.csv", index=False)

        # Summary
        summary_effect = summarize_power_results(
            df_effect, ["method", "scenario", "effect_size"]
        )
        summary_effect.to_csv(args.output_dir / "power_vs_effect_size_summary.csv", index=False)

    # Run sample size analysis
    if args.mode in ["sample-size", "both"]:
        df_sample = run_power_vs_sample_size(
            methods=args.methods,
            scenarios=args.scenarios,
            sample_sizes=args.sample_sizes,
            effect_size=args.effect_size,
            n_repeats=args.n_repeats,
            n_nodes=args.n_nodes,
            n_modules=args.n_modules,
            n_permutations=args.n_permutations,
            alpha=args.alpha,
            use_mp=args.use_mp,
            seed=args.seed,
        )
        df_sample.to_csv(args.output_dir / "power_vs_sample_size.csv", index=False)

        # Summary
        summary_sample = summarize_power_results(
            df_sample, ["method", "scenario", "n_samples"]
        )
        summary_sample.to_csv(args.output_dir / "power_vs_sample_size_summary.csv", index=False)

        # Required sample size
        required_n = compute_required_sample_size(df_sample, target_power=0.8)
        required_n.to_csv(args.output_dir / "required_sample_sizes.csv", index=False)

    # Create plots
    if df_effect is not None:
        plot_power_curves(df_effect, df_sample, args.output_dir)

    print(f"\nResults saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
