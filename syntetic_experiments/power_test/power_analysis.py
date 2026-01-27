"""
Power Analysis for TFNBS Methods.

This script computes statistical power curves for all methods across:
1. Effect sizes (at fixed sample size)
2. Sample sizes (at fixed effect size)

Power is defined as the proportion of true edges that are detected
(True Positive Rate / Sensitivity) at the nominal alpha level.

Scenario names follow syntetic_experiments/sim_methods/sim_topology_examples.py so power analysis
matches the method-comparison topologies exactly.

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

    # Use YAML config
    python power_analysis.py --config sweep_config_power.yaml

    # Resume interrupted run
    python power_analysis.py --config sweep_config_power.yaml --resume

Output:
    - results/power_analysis/power_vs_effect_size.csv
    - results/power_analysis/power_vs_sample_size.csv
    - results/power_analysis/power_curves.png
    - results/power_analysis/checkpoint_*.csv (for resume)
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import itertools

import numpy as np
import pandas as pd

try:
    import yaml
    HAS_YAML = True
except ImportError:  
    HAS_YAML = False

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import sim_topology_examples as topo
from tfnbs.pairwise_stats import compute_p_val


@dataclass
class PowerResult:
    """Result from a single power simulation."""

    method: str
    method_params: str
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
    FDR: float
    elapsed_time: float


def run_single_power_test(
    method: str,
    method_params: str,
    scenario: str,
    effect_size: float,
    n_samples: int,
    repeat_id: int,
    n_nodes: int = 60,
    n_modules: int = 4,
    time_points: int = 30,
    n_permutations: int = 500,
    alpha: float = 0.05,
    use_mp: bool = False,
    seed_base: int = 42,
    method_kwargs: dict = None,
    intra_corr: float = 0.3,
    inter_corr: float = 0.05,
    uniform_corr: float = 0.15,
    noise_level: float = 0.05,
) -> PowerResult:
    """Run a single power simulation."""

    start_time = time.time()

    # Generate data using the same topology definitions as sim_method_comparisons.py.
    seed = seed_base + repeat_id * 1000
    generator = topo.TopologyDatasetGenerator(
        n_nodes=n_nodes,
        n_modules=n_modules,
        intra_corr=intra_corr,
        inter_corr=inter_corr,
        uniform_corr=uniform_corr,
        noise_level=noise_level,
        seed=seed,
    )
    dataset = generator.generate(
        scenario,
        effect_size=effect_size,
        n_samples=n_samples,
        time_points=time_points,
    )
    group1_z, group2_z = dataset.fisher_z()
    effect_mask = dataset.effect_mask
    net_labels = dataset.net_labels

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
        mask_upper = effect_mask[triu_idx] != 0

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
        FDR = FP / (TP + FP) if (TP + FP) > 0 else 0.0

    except Exception as e:
        print(f"  ERROR in {method} repeat {repeat_id}: {e}")
        n_true = int(np.sum(effect_mask[np.triu_indices(n_nodes, k=1)] != 0))
        n_null = n_nodes * (n_nodes - 1) // 2 - n_true
        TP, FP, FN, TN = 0, 0, n_true, n_null
        TPR, FPR, precision, FDR = 0.0, 0.0, 0.0, 0.0

    elapsed = time.time() - start_time

    return PowerResult(
        method=method,
        method_params=method_params,
        scenario=scenario,
        effect_size=effect_size,
        n_samples=n_samples,
        repeat_id=repeat_id,
        n_true=n_true,
        n_null=n_null,
        TP=TP, FP=FP, FN=FN, TN=TN,
        TPR=TPR, FPR=FPR,
        precision=precision, FDR=FDR,
        elapsed_time=elapsed,
    )


def _normalize_tfnbs_options(options: Optional[dict]) -> dict:
    if not options:
        return {}
    if not isinstance(options, dict):
        raise TypeError("tfnbs_options must be a dict.")
    normalized = dict(options)
    if "n_thresholds" in normalized and "n" not in normalized:
        normalized["n"] = normalized.pop("n_thresholds")
    if "fbc_min_cluster_size" in normalized and "min_cluster_size" not in normalized:
        normalized["min_cluster_size"] = normalized.pop("fbc_min_cluster_size")
    return normalized


def _normalize_nbs_options(options: Optional[dict]) -> dict:
    if not options:
        return {}
    if not isinstance(options, dict):
        raise TypeError("nbs_options must be a dict.")
    return dict(options)


def _expand_options(base: dict, overrides: dict) -> List[dict]:
    base_config = dict(base)
    sweep_keys = []
    sweep_values = []
    fixed = {}
    for key, value in overrides.items():
        if isinstance(value, (list, tuple)):
            sweep_keys.append(key)
            sweep_values.append(list(value))
        else:
            fixed[key] = value
    base_config.update(fixed)
    if not sweep_keys:
        return [base_config]
    configs = []
    for combo in itertools.product(*sweep_values):
        cfg = dict(base_config)
        for key, value in zip(sweep_keys, combo):
            cfg[key] = value
        configs.append(cfg)
    return configs


# =============================================================================
# CHECKPOINTING
# =============================================================================

def save_checkpoint(results: List[PowerResult], checkpoint_path: Path) -> None:
    """Save intermediate results to checkpoint file."""
    df = pd.DataFrame([vars(r) for r in results])
    df.to_csv(checkpoint_path, index=False)


def load_checkpoint(checkpoint_path: Path) -> Tuple[List[PowerResult], set]:
    """Load results from checkpoint and return completed run keys."""
    if not checkpoint_path.exists():
        return [], set()

    df = pd.read_csv(checkpoint_path)
    results = []
    completed = set()

    for _, row in df.iterrows():
        result = PowerResult(
            method=row["method"],
            method_params=row["method_params"],
            scenario=row["scenario"],
            effect_size=float(row["effect_size"]),
            n_samples=int(row["n_samples"]),
            repeat_id=int(row["repeat_id"]),
            n_true=int(row["n_true"]),
            n_null=int(row["n_null"]),
            TP=int(row["TP"]),
            FP=int(row["FP"]),
            FN=int(row["FN"]),
            TN=int(row["TN"]),
            TPR=float(row["TPR"]),
            FPR=float(row["FPR"]),
            precision=float(row["precision"]),
            FDR=float(row["FDR"]),
            elapsed_time=float(row["elapsed_time"]),
        )
        results.append(result)
        # Key: (method, method_params, scenario, effect_size, n_samples, repeat_id)
        completed.add((
            row["method"],
            row["method_params"],
            row["scenario"],
            float(row["effect_size"]),
            int(row["n_samples"]),
            int(row["repeat_id"]),
        ))

    return results, completed


def _format_method_params(method: str, method_kwargs: dict) -> str:
    if not method_kwargs:
        return ""
    if method in {"tfnbs", "ni_tfnbs"}:
        keys = ("e", "h", "n", "start_thres")
    elif method == "fbc_tfnbs":
        keys = ("e", "h", "n", "start_thres", "min_cluster_size")
    elif method in {"nbs_extent", "nbs_intensity"}:
        keys = ("threshold",)
    else:
        keys = tuple(method_kwargs.keys())
    parts = []
    for key in keys:
        if key in method_kwargs:
            parts.append(f"{key}={method_kwargs[key]}")
    return ",".join(parts)


def get_method_configs(
    *,
    tfnbs_options: Optional[dict] = None,
    nbs_options: Optional[dict] = None,
) -> Dict[str, dict]:
    """Get default configuration for each method."""
    tfnbs_overrides = _normalize_tfnbs_options(tfnbs_options)
    nbs_overrides = _normalize_nbs_options(nbs_options)

    tfnbs_defaults = {"e": 0.5, "h": 2.0, "n": 50, "start_thres": 1.65}
    tfnbs_configs = _expand_options(
        tfnbs_defaults,
        {k: v for k, v in tfnbs_overrides.items() if k != "min_cluster_size"},
    )

    fbc_defaults = dict(tfnbs_defaults)
    fbc_defaults.setdefault("min_cluster_size", 3)
    fbc_configs = _expand_options(
        fbc_defaults,
        {k: v for k, v in tfnbs_overrides.items()},
    )

    nbs_defaults = {"threshold": 2.0}
    nbs_configs = _expand_options(nbs_defaults, nbs_overrides)

    return {
        "tstat": {},
        "tfnbs": tfnbs_configs,
        "ni_tfnbs": tfnbs_configs,
        "fbc_tfnbs": fbc_configs,
        "nbs_extent": nbs_configs,
        "nbs_intensity": nbs_configs,
        "cnbs": {},
    }


def _iter_method_variants(
    methods: List[str],
    method_configs: Dict[str, object],
) -> List[Tuple[str, dict]]:
    variants = []
    for method in methods:
        configs = method_configs.get(method, {})
        if isinstance(configs, list):
            variants.extend((method, cfg) for cfg in configs)
        elif isinstance(configs, dict):
            variants.append((method, configs))
        else:
            raise TypeError(f"Invalid method config for {method}: {type(configs).__name__}")
    return variants


def run_power_vs_effect_size(
    methods: List[str],
    scenarios: List[str],
    effect_sizes: List[float],
    n_samples: int = 20,
    n_repeats: int = 50,
    n_nodes: int = 60,
    n_modules: int = 4,
    time_points: int = 30,
    n_permutations: int = 500,
    alpha: float = 0.05,
    use_mp: bool = False,
    seed: int = 42,
    intra_corr: float = 0.3,
    inter_corr: float = 0.05,
    uniform_corr: float = 0.15,
    noise_level: float = 0.05,
    method_configs: Optional[Dict[str, dict]] = None,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_every: int = 10,
    resume: bool = False,
) -> pd.DataFrame:
    """Run power analysis across effect sizes with checkpointing."""

    method_configs = method_configs or get_method_configs()

    # Setup checkpointing
    checkpoint_path = None
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "checkpoint_effect_size.csv"

    # Load checkpoint if resuming
    if resume and checkpoint_path and checkpoint_path.exists():
        results, completed = load_checkpoint(checkpoint_path)
        print(f"Resuming from checkpoint: {len(completed)} runs already completed")
    else:
        results = []
        completed = set()

    method_variants = _iter_method_variants(methods, method_configs)

    # Build task list (skipping completed)
    tasks = []
    for scenario in scenarios:
        for effect_size in effect_sizes:
            for method, kwargs in method_variants:
                method_params = _format_method_params(method, kwargs)
                for repeat_id in range(n_repeats):
                    key = (method, method_params, scenario, effect_size, n_samples, repeat_id)
                    if key not in completed:
                        tasks.append((method, method_params, kwargs, scenario, effect_size, repeat_id))

    total_runs = len(tasks) + len(completed)
    remaining_runs = len(tasks)

    if remaining_runs == 0:
        print("All runs already completed!")
        return pd.DataFrame([vars(r) for r in results])

    print(f"=== Power vs Effect Size ===")
    print(f"Methods: {methods} (variants: {len(method_variants)})")
    print(f"Scenarios: {scenarios}")
    print(f"Effect sizes: {effect_sizes}")
    print(f"Repeats per condition: {n_repeats}")
    print(f"Total runs: {total_runs}, Remaining: {remaining_runs}")
    print()

    start_time = time.time()
    run_times = []  # Track individual run times for estimation
    run_count = len(completed)

    for i, (method, method_params, kwargs, scenario, effect_size, repeat_id) in enumerate(tasks):
        run_start = time.time()
        run_count += 1

        result = run_single_power_test(
            method=method,
            method_params=method_params,
            scenario=scenario,
            effect_size=effect_size,
            n_samples=n_samples,
            repeat_id=repeat_id,
            n_nodes=n_nodes,
            n_modules=n_modules,
            time_points=time_points,
            n_permutations=n_permutations,
            alpha=alpha,
            use_mp=use_mp,
            seed_base=seed,
            method_kwargs=kwargs,
            intra_corr=intra_corr,
            inter_corr=inter_corr,
            uniform_corr=uniform_corr,
            noise_level=noise_level,
        )
        results.append(result)

        # Track run time
        run_elapsed = time.time() - run_start
        run_times.append(run_elapsed)

        # Progress and ETA (every checkpoint_every runs or on last run)
        if (i + 1) % checkpoint_every == 0 or (i + 1) == remaining_runs:
            elapsed = time.time() - start_time
            avg_time = np.mean(run_times[-50:])  # Rolling average of last 50 runs
            eta_seconds = avg_time * (remaining_runs - i - 1)

            # Method stats for current scenario/effect_size
            method_results = [
                r for r in results
                if r.method == method
                and r.method_params == method_params
                and r.scenario == scenario
                and r.effect_size == effect_size
            ]
            mean_tpr = np.mean([r.TPR for r in method_results]) if method_results else 0.0
            label = f"{method} [{method_params}]" if method_params else method

            print(
                f"  [{run_count}/{total_runs}] {scenario} ES={effect_size} {label}: "
                f"TPR={mean_tpr:.3f} | {run_elapsed:.1f}s/run | "
                f"Elapsed: {elapsed/60:.1f}min | ETA: {eta_seconds/60:.1f}min"
            )

            # Save checkpoint
            if checkpoint_path:
                save_checkpoint(results, checkpoint_path)

    return pd.DataFrame([vars(r) for r in results])


def run_power_vs_sample_size(
    methods: List[str],
    scenarios: List[str],
    sample_sizes: List[int],
    effect_size: float = 0.25,
    n_repeats: int = 50,
    n_nodes: int = 60,
    n_modules: int = 4,
    time_points: int = 30,
    n_permutations: int = 500,
    alpha: float = 0.05,
    use_mp: bool = False,
    seed: int = 42,
    intra_corr: float = 0.3,
    inter_corr: float = 0.05,
    uniform_corr: float = 0.15,
    noise_level: float = 0.05,
    method_configs: Optional[Dict[str, dict]] = None,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_every: int = 10,
    resume: bool = False,
) -> pd.DataFrame:
    """Run power analysis across sample sizes with checkpointing."""

    method_configs = method_configs or get_method_configs()

    # Setup checkpointing
    checkpoint_path = None
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "checkpoint_sample_size.csv"

    # Load checkpoint if resuming
    if resume and checkpoint_path and checkpoint_path.exists():
        results, completed = load_checkpoint(checkpoint_path)
        print(f"Resuming from checkpoint: {len(completed)} runs already completed")
    else:
        results = []
        completed = set()

    method_variants = _iter_method_variants(methods, method_configs)

    # Build task list (skipping completed)
    tasks = []
    for scenario in scenarios:
        for n_samples in sample_sizes:
            for method, kwargs in method_variants:
                method_params = _format_method_params(method, kwargs)
                for repeat_id in range(n_repeats):
                    key = (method, method_params, scenario, effect_size, n_samples, repeat_id)
                    if key not in completed:
                        tasks.append((method, method_params, kwargs, scenario, n_samples, repeat_id))

    total_runs = len(tasks) + len(completed)
    remaining_runs = len(tasks)

    if remaining_runs == 0:
        print("All runs already completed!")
        return pd.DataFrame([vars(r) for r in results])

    print(f"=== Power vs Sample Size ===")
    print(f"Methods: {methods} (variants: {len(method_variants)})")
    print(f"Scenarios: {scenarios}")
    print(f"Sample sizes: {sample_sizes}")
    print(f"Effect size: {effect_size}")
    print(f"Repeats per condition: {n_repeats}")
    print(f"Total runs: {total_runs}, Remaining: {remaining_runs}")
    print()

    start_time = time.time()
    run_times = []  # Track individual run times for estimation
    run_count = len(completed)

    for i, (method, method_params, kwargs, scenario, n_samples, repeat_id) in enumerate(tasks):
        run_start = time.time()
        run_count += 1

        result = run_single_power_test(
            method=method,
            method_params=method_params,
            scenario=scenario,
            effect_size=effect_size,
            n_samples=n_samples,
            repeat_id=repeat_id,
            n_nodes=n_nodes,
            n_modules=n_modules,
            time_points=time_points,
            n_permutations=n_permutations,
            alpha=alpha,
            use_mp=use_mp,
            seed_base=seed,
            method_kwargs=kwargs,
            intra_corr=intra_corr,
            inter_corr=inter_corr,
            uniform_corr=uniform_corr,
            noise_level=noise_level,
        )
        results.append(result)

        # Track run time
        run_elapsed = time.time() - run_start
        run_times.append(run_elapsed)

        # Progress and ETA (every checkpoint_every runs or on last run)
        if (i + 1) % checkpoint_every == 0 or (i + 1) == remaining_runs:
            elapsed = time.time() - start_time
            avg_time = np.mean(run_times[-50:])  # Rolling average of last 50 runs
            eta_seconds = avg_time * (remaining_runs - i - 1)

            # Method stats for current scenario/n_samples
            method_results = [
                r for r in results
                if r.method == method
                and r.method_params == method_params
                and r.scenario == scenario
                and r.n_samples == n_samples
            ]
            mean_tpr = np.mean([r.TPR for r in method_results]) if method_results else 0.0
            label = f"{method} [{method_params}]" if method_params else method

            print(
                f"  [{run_count}/{total_runs}] {scenario} N={n_samples} {label}: "
                f"TPR={mean_tpr:.3f} | {run_elapsed:.1f}s/run | "
                f"Elapsed: {elapsed/60:.1f}min | ETA: {eta_seconds/60:.1f}min"
            )

            # Save checkpoint
            if checkpoint_path:
                save_checkpoint(results, checkpoint_path)

    return pd.DataFrame([vars(r) for r in results])


def summarize_power_results(df: pd.DataFrame, groupby_cols: List[str]) -> pd.DataFrame:
    """Summarize power results by grouping columns."""

    summary = df.groupby(groupby_cols).agg({
        "TPR": ["mean", "std", "count"],
        "FPR": ["mean", "std"],
        "precision": ["mean", "std"],
        "FDR": ["mean", "std"],
        "elapsed_time": ["mean", "std"],
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

    df_effect_plot = df_effect.copy()
    df_sample_plot = df_sample.copy() if df_sample is not None else None
    if "method_params" in df_effect_plot.columns:
        df_effect_plot["method_display"] = df_effect_plot.apply(
            lambda row: f"{row['method']} [{row['method_params']}]" if row["method_params"] else row["method"],
            axis=1,
        )
    else:
        df_effect_plot["method_display"] = df_effect_plot["method"]
    if df_sample_plot is not None:
        if "method_params" in df_sample_plot.columns:
            df_sample_plot["method_display"] = df_sample_plot.apply(
                lambda row: f"{row['method']} [{row['method_params']}]" if row["method_params"] else row["method"],
                axis=1,
            )
        else:
            df_sample_plot["method_display"] = df_sample_plot["method"]

    # Color palette for method variants
    method_labels = df_effect_plot["method_display"].unique()
    palette = dict(zip(method_labels, sns.color_palette("husl", len(method_labels))))

    scenarios = df_effect_plot["scenario"].unique()

    # Plot 1: Power vs Effect Size (one subplot per scenario)
    n_scenarios = len(scenarios)
    fig, axes = plt.subplots(1, n_scenarios, figsize=(5 * n_scenarios, 4), squeeze=False)

    for idx, scenario in enumerate(scenarios):
        ax = axes[0, idx]
        scenario_df = df_effect_plot[df_effect_plot["scenario"] == scenario]

        for label in method_labels:
            method_df = scenario_df[scenario_df["method_display"] == label]
            summary = method_df.groupby("effect_size")["TPR"].agg(["mean", "std"]).reset_index()
            ax.errorbar(
                summary["effect_size"], summary["mean"],
                yerr=summary["std"],
                label=label, color=palette[label],
                marker="o", capsize=3, linewidth=2,
            )

        ax.axhline(0.8, color="gray", linestyle="--", alpha=0.5, label="80% power")
        ax.set_xlabel("Effect Size")
        ax.set_ylabel("Power (TPR)")
        ax.set_title(f"{scenario}")
        ax.set_ylim(0, 1.)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Power vs Effect Size by Scenario", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "power_vs_effect_size.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 1b: FDR vs Effect Size (one subplot per scenario)
    fig, axes = plt.subplots(1, n_scenarios, figsize=(5 * n_scenarios, 4), squeeze=False)

    for idx, scenario in enumerate(scenarios):
        ax = axes[0, idx]
        scenario_df = df_effect_plot[df_effect_plot["scenario"] == scenario]

        for label in method_labels:
            method_df = scenario_df[scenario_df["method_display"] == label]
            summary = method_df.groupby("effect_size")["FDR"].agg(["mean", "std"]).reset_index()
            ax.errorbar(
                summary["effect_size"], summary["mean"],
                yerr=summary["std"],
                label=label, color=palette[label],
                marker="o", capsize=3, linewidth=2,
            )

        ax.set_xlabel("Effect Size")
        ax.set_ylabel("False Discovery Rate (FDR)")
        ax.set_title(f"{scenario}")
        ax.set_ylim(0, 0.5)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("FDR vs Effect Size by Scenario", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "fdr_vs_effect_size.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 2: Power vs Sample Size
    if df_sample_plot is not None and len(df_sample_plot) > 0:
        fig, axes = plt.subplots(1, n_scenarios, figsize=(5 * n_scenarios, 4), squeeze=False)

        for idx, scenario in enumerate(scenarios):
            ax = axes[0, idx]
            scenario_df = df_sample_plot[df_sample_plot["scenario"] == scenario]

            for label in method_labels:
                method_df = scenario_df[scenario_df["method_display"] == label]
                summary = method_df.groupby("n_samples")["TPR"].agg(["mean", "std"]).reset_index()
                ax.errorbar(
                    summary["n_samples"], summary["mean"],
                    yerr=summary["std"],
                    label=label, color=palette[label],
                    marker="o", capsize=3, linewidth=2,
                )

            ax.axhline(0.8, color="gray", linestyle="--", alpha=0.5)
            ax.set_xlabel("Sample Size (per group)")
            ax.set_ylabel("Power (TPR)")
            ax.set_title(f"{scenario}")
            ax.set_ylim(0, 1.)
            ax.legend(loc="lower right", fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.suptitle("Power vs Sample Size by Scenario", y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / "power_vs_sample_size.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Plot 2b: FDR vs Sample Size
        fig, axes = plt.subplots(1, n_scenarios, figsize=(5 * n_scenarios, 4), squeeze=False)

        for idx, scenario in enumerate(scenarios):
            ax = axes[0, idx]
            scenario_df = df_sample_plot[df_sample_plot["scenario"] == scenario]

            for label in method_labels:
                method_df = scenario_df[scenario_df["method_display"] == label]
                summary = method_df.groupby("n_samples")["FDR"].agg(["mean", "std"]).reset_index()
                ax.errorbar(
                    summary["n_samples"], summary["mean"],
                    yerr=summary["std"],
                    label=label, color=palette[label],
                    marker="o", capsize=3, linewidth=2,
                )

            ax.set_xlabel("Sample Size (per group)")
            ax.set_ylabel("False Discovery Rate (FDR)")
            ax.set_title(f"{scenario}")
            ax.set_ylim(0, 0.5)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.suptitle("FDR vs Sample Size by Scenario", y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / "fdr_vs_sample_size.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Plot 3: Combined summary heatmap (FDR by method x scenario at ES=0.25)
    if "effect_size" in df_effect_plot.columns:
        pivot_df = df_effect_plot[df_effect_plot["effect_size"] == 0.25].groupby(
            ["method_display", "scenario"]
        )["FDR"].mean().unstack()

        if not pivot_df.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                pivot_df, annot=True, fmt=".2f", cmap="Reds",
                vmin=0, vmax=1, ax=ax, cbar_kws={"label": "FDR"}
            )
            ax.set_title("FDR by Method and Scenario (Effect Size = 0.25)")
            plt.tight_layout()
            plt.savefig(output_dir / "fdr_heatmap.png", dpi=150, bbox_inches="tight")
            plt.close()

    print(f"Saved plots to {output_dir}")


def _clean_method_params_for_display(params) -> str:
    """Clean method params string for plot legend by removing start_thres and n."""
    if pd.isna(params) or not params:
        return "default"
    params = str(params)
    # Remove start_thres and n parameters for cleaner display
    parts = params.split(",")
    cleaned = [p for p in parts if not p.strip().startswith(("start_thres=", "n="))]
    return ",".join(cleaned) if cleaned else "default"


def plot_power_curves_per_method(
    df_effect: pd.DataFrame,
    df_sample: pd.DataFrame,
    output_dir: Path,
):
    """Create separate power curve plots for each base method."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not available, skipping plots")
        return

    df_effect_plot = df_effect.copy()
    df_sample_plot = df_sample.copy() if df_sample is not None else None

    # Add method_display column (cleaned for better readability)
    if "method_params" in df_effect_plot.columns:
        df_effect_plot["method_display"] = df_effect_plot["method_params"].apply(
            _clean_method_params_for_display
        )
    else:
        df_effect_plot["method_display"] = "default"

    if df_sample_plot is not None:
        if "method_params" in df_sample_plot.columns:
            df_sample_plot["method_display"] = df_sample_plot["method_params"].apply(
                _clean_method_params_for_display
            )
        else:
            df_sample_plot["method_display"] = "default"

    # Get unique base methods and scenarios
    base_methods = df_effect_plot["method"].unique()
    scenarios = df_effect_plot["scenario"].unique()
    n_scenarios = len(scenarios)

    # Create per-method subdirectory
    per_method_dir = output_dir / "per_method"
    per_method_dir.mkdir(parents=True, exist_ok=True)

    for method in base_methods:
        method_df_effect = df_effect_plot[df_effect_plot["method"] == method]
        method_df_sample = (
            df_sample_plot[df_sample_plot["method"] == method]
            if df_sample_plot is not None else None
        )

        # Get unique variants for this method
        variants = method_df_effect["method_display"].unique()
        n_variants = len(variants)

        # Use color palette for variants
        palette = dict(zip(variants, sns.color_palette("husl", n_variants)))

        # Plot 1: Power vs Effect Size
        _fig, axes = plt.subplots(1, n_scenarios, figsize=(5 * n_scenarios, 4), squeeze=False)

        for idx, scenario in enumerate(scenarios):
            ax = axes[0, idx]
            scenario_df = method_df_effect[method_df_effect["scenario"] == scenario]

            for variant in variants:
                variant_df = scenario_df[scenario_df["method_display"] == variant]
                if len(variant_df) == 0:
                    continue
                summary = variant_df.groupby("effect_size")["TPR"].agg(["mean", "std"]).reset_index()
                ax.errorbar(
                    summary["effect_size"], summary["mean"],
                    yerr=summary["std"],
                    label=variant, color=palette[variant],
                    marker="o", capsize=3, linewidth=2,
                )

            ax.axhline(0.8, color="gray", linestyle="--", alpha=0.5, label="80% power")
            ax.set_xlabel("Effect Size")
            ax.set_ylabel("Power (TPR)")
            ax.set_title(f"{scenario}")
            ax.set_ylim(0, 1.)
            ax.legend(loc="lower right", fontsize=7)
            ax.grid(True, alpha=0.3)

        plt.suptitle(f"{method}: Power vs Effect Size", y=1.02, fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(per_method_dir / f"power_vs_effect_size_{method}.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Plot 2: FDR vs Effect Size
        _fig, axes = plt.subplots(1, n_scenarios, figsize=(5 * n_scenarios, 4), squeeze=False)

        for idx, scenario in enumerate(scenarios):
            ax = axes[0, idx]
            scenario_df = method_df_effect[method_df_effect["scenario"] == scenario]

            for variant in variants:
                variant_df = scenario_df[scenario_df["method_display"] == variant]
                if len(variant_df) == 0:
                    continue
                summary = variant_df.groupby("effect_size")["FDR"].agg(["mean", "std"]).reset_index()
                ax.errorbar(
                    summary["effect_size"], summary["mean"],
                    yerr=summary["std"],
                    label=variant, color=palette[variant],
                    marker="o", capsize=3, linewidth=2,
                )

            ax.axhline(0.05, color="gray", linestyle="--", alpha=0.5, label="alpha=0.05")
            ax.set_xlabel("Effect Size")
            ax.set_ylabel("False Discovery Rate (FDR)")
            ax.set_title(f"{scenario}")
            ax.set_ylim(0, 0.6)
            ax.legend(loc="upper right", fontsize=7)
            ax.grid(True, alpha=0.3)

        plt.suptitle(f"{method}: FDR vs Effect Size", y=1.02, fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(per_method_dir / f"fdr_vs_effect_size_{method}.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Plot 3: Power vs Sample Size
        if method_df_sample is not None and len(method_df_sample) > 0:
            _fig, axes = plt.subplots(1, n_scenarios, figsize=(5 * n_scenarios, 4), squeeze=False)

            for idx, scenario in enumerate(scenarios):
                ax = axes[0, idx]
                scenario_df = method_df_sample[method_df_sample["scenario"] == scenario]

                for variant in variants:
                    variant_df = scenario_df[scenario_df["method_display"] == variant]
                    if len(variant_df) == 0:
                        continue
                    summary = variant_df.groupby("n_samples")["TPR"].agg(["mean", "std"]).reset_index()
                    ax.errorbar(
                        summary["n_samples"], summary["mean"],
                        yerr=summary["std"],
                        label=variant, color=palette[variant],
                        marker="o", capsize=3, linewidth=2,
                    )

                ax.axhline(0.8, color="gray", linestyle="--", alpha=0.5)
                ax.set_xlabel("Sample Size (per group)")
                ax.set_ylabel("Power (TPR)")
                ax.set_title(f"{scenario}")
                ax.set_ylim(0, 1.)
                ax.legend(loc="lower right", fontsize=7)
                ax.grid(True, alpha=0.3)

            plt.suptitle(f"{method}: Power vs Sample Size", y=1.02, fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.savefig(per_method_dir / f"power_vs_sample_size_{method}.png", dpi=150, bbox_inches="tight")
            plt.close()

            # Plot 4: FDR vs Sample Size
            _fig, axes = plt.subplots(1, n_scenarios, figsize=(5 * n_scenarios, 4), squeeze=False)

            for idx, scenario in enumerate(scenarios):
                ax = axes[0, idx]
                scenario_df = method_df_sample[method_df_sample["scenario"] == scenario]

                for variant in variants:
                    variant_df = scenario_df[scenario_df["method_display"] == variant]
                    if len(variant_df) == 0:
                        continue
                    summary = variant_df.groupby("n_samples")["FDR"].agg(["mean", "std"]).reset_index()
                    ax.errorbar(
                        summary["n_samples"], summary["mean"],
                        yerr=summary["std"],
                        label=variant, color=palette[variant],
                        marker="o", capsize=3, linewidth=2,
                    )

                ax.axhline(0.05, color="gray", linestyle="--", alpha=0.5, label="alpha=0.05")
                ax.set_xlabel("Sample Size (per group)")
                ax.set_ylabel("False Discovery Rate (FDR)")
                ax.set_title(f"{scenario}")
                ax.set_ylim(0, 0.6)
                ax.legend(loc="upper right", fontsize=7)
                ax.grid(True, alpha=0.3)

            plt.suptitle(f"{method}: FDR vs Sample Size", y=1.02, fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.savefig(per_method_dir / f"fdr_vs_sample_size_{method}.png", dpi=150, bbox_inches="tight")
            plt.close()

    print(f"Saved per-method plots to {per_method_dir}")


def compute_required_sample_size(
    df: pd.DataFrame,
    target_power: float = 0.8,
) -> pd.DataFrame:
    """Estimate required sample size to achieve target power."""

    results = []

    group_cols = ["method", "scenario", "effect_size"]
    if "method_params" in df.columns:
        group_cols.append("method_params")

    for group_keys, group in df.groupby(group_cols):
        method, scenario, effect_size = group_keys[:3]
        method_params = group_keys[3] if len(group_keys) > 3 else ""
        power_by_n = group.groupby("n_samples")["TPR"].mean()

        # Find smallest n with power >= target
        required_n = None
        for n in sorted(power_by_n.index):
            if power_by_n[n] >= target_power:
                required_n = n
                break

        results.append({
            "method": method,
            "method_params": method_params,
            "scenario": scenario,
            "effect_size": effect_size,
            "target_power": target_power,
            "required_n": required_n,
            "achieved_power": power_by_n.get(required_n, np.nan) if required_n else np.nan,
        })

    return pd.DataFrame(results)


def load_yaml_config(path: Path) -> dict:
    """Load configuration from YAML file."""
    if not HAS_YAML:
        raise ImportError("PyYAML is required for --config. Install with: pip install pyyaml")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a dict in {path}, got {type(data).__name__}")
    return data


def apply_config_defaults(parser: argparse.ArgumentParser, config: dict) -> None:
    """Apply YAML config values as parser defaults."""
    valid_keys = {action.dest for action in parser._actions}
    defaults = {}
    for key, value in config.items():
        if key in {"tfnbs_options", "nbs_options"}:
            continue
        if key not in valid_keys:
            print(f"Warning: ignoring unknown config key '{key}'")
            continue
        if key == "output_dir" and value is not None:
            defaults[key] = Path(value)
        else:
            defaults[key] = value
    if defaults:
        parser.set_defaults(**defaults)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Power analysis for TFNBS methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--config", type=Path, default=None,
        help="Load power analysis configuration from YAML file.",
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
        "--time-points", type=int, default=30,
        help="Time points per subject (data generator)",
    )
    parser.add_argument(
        "--intra-corr", type=float, default=0.3,
        help="Within-module base correlation",
    )
    parser.add_argument(
        "--inter-corr", type=float, default=0.05,
        help="Between-module base correlation",
    )
    parser.add_argument(
        "--uniform-corr", type=float, default=0.15,
        help="Uniform base correlation for uniform scenarios",
    )
    parser.add_argument(
        "--noise-level", type=float, default=0.05,
        help="Noise level for effect variability",
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
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint if available",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=10,
        help="Save checkpoint every N runs",
    )

    # Pre-parse to load YAML defaults if provided.
    pre_args, _ = parser.parse_known_args(argv)
    config = None
    if pre_args.config is not None:
        config = load_yaml_config(pre_args.config)
        apply_config_defaults(parser, config)

    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df_effect = None
    df_sample = None

    tfnbs_options = None
    nbs_options = None
    if config:
        tfnbs_options = config.get("tfnbs_options")
        nbs_options = config.get("nbs_options")
    method_configs = get_method_configs(
        tfnbs_options=tfnbs_options,
        nbs_options=nbs_options,
    )
    if tfnbs_options or nbs_options:
        print("Using method options from config:")
        if tfnbs_options:
            print(f"  tfnbs_options: {tfnbs_options}")
        if nbs_options:
            print(f"  nbs_options: {nbs_options}")

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
            time_points=args.time_points,
            n_permutations=args.n_permutations,
            alpha=args.alpha,
            use_mp=args.use_mp,
            seed=args.seed,
            intra_corr=args.intra_corr,
            inter_corr=args.inter_corr,
            uniform_corr=args.uniform_corr,
            noise_level=args.noise_level,
            method_configs=method_configs,
            checkpoint_dir=args.output_dir,
            checkpoint_every=args.checkpoint_every,
            resume=args.resume,
        )
        df_effect.to_csv(args.output_dir / "power_vs_effect_size.csv", index=False)

        # Summary
        summary_effect = summarize_power_results(
            df_effect, ["method", "method_params", "scenario", "effect_size"]
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
            time_points=args.time_points,
            n_permutations=args.n_permutations,
            alpha=args.alpha,
            use_mp=args.use_mp,
            seed=args.seed,
            intra_corr=args.intra_corr,
            inter_corr=args.inter_corr,
            uniform_corr=args.uniform_corr,
            noise_level=args.noise_level,
            method_configs=method_configs,
            checkpoint_dir=args.output_dir,
            checkpoint_every=args.checkpoint_every,
            resume=args.resume,
        )
        df_sample.to_csv(args.output_dir / "power_vs_sample_size.csv", index=False)

        # Summary
        summary_sample = summarize_power_results(
            df_sample, ["method", "method_params", "scenario", "n_samples"]
        )
        summary_sample.to_csv(args.output_dir / "power_vs_sample_size_summary.csv", index=False)

        # Required sample size
        required_n = compute_required_sample_size(df_sample, target_power=0.8)
        required_n.to_csv(args.output_dir / "required_sample_sizes.csv", index=False)

    # Create plots
    if df_effect is not None:
        plot_power_curves(df_effect, df_sample, args.output_dir)

    # Clean up checkpoints on successful completion
    checkpoint_effect = args.output_dir / "checkpoint_effect_size.csv"
    checkpoint_sample = args.output_dir / "checkpoint_sample_size.csv"
    if checkpoint_effect.exists():
        checkpoint_effect.unlink()
        print("Removed effect size checkpoint (run completed)")
    if checkpoint_sample.exists():
        checkpoint_sample.unlink()
        print("Removed sample size checkpoint (run completed)")

    print(f"\nResults saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
