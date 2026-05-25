"""
Power Analysis for TFNBS Methods.

Computes statistical power (TPR), FDR, precision, and FPR for all methods
across effect sizes and/or sample sizes using synthetic modular networks
with known ground-truth effects.

Methods compared:
    tstat, tfnbs, ni_tfnbs, fbc_tfnbs, nbs_extent, nbs_intensity, cnbs,
    bonferroni, bh_fdr

Key features:
    - Batched 3D mode: TFNBS methods compute all (e,h) parameter combos
      in a single compute_p_val call (e.g. 176 combos with no extra cost).
    - Numba-accelerated TFNBS scoring (auto backend).
    - Parallel execution via mp.Pool with spawn context and BLAS thread
      pinning (1 thread per worker) to avoid mutex deadlocks.
    - Checkpointing with --resume support for interrupted runs.
    - Per-scenario CSV output for easy downstream analysis.

Scenarios are defined in conninfpy/topologies.py (e.g.
within_module_dense, hub, chain, scattered_cross_block, etc.).

Usage:
    # Run both analyses with a YAML config (recommended)
    python examples/simulation_validation/power/power_analysis.py \\
        --config examples/simulation_validation/configs/sweep_config_power_1000.yaml

    # Power vs effect size only
    python examples/simulation_validation/power/power_analysis.py \\
        --config examples/simulation_validation/configs/sweep_config_power_1000.yaml \\
        --mode effect-size

    # Power vs sample size only
    python examples/simulation_validation/power/power_analysis.py \\
        --config examples/simulation_validation/configs/sweep_config_power_1000.yaml \\
        --mode sample-size

    # Resume an interrupted run
    python examples/simulation_validation/power/power_analysis.py \\
        --config examples/simulation_validation/configs/sweep_config_power_1000.yaml --resume

    # CLI overrides (no YAML)
    python examples/simulation_validation/power/power_analysis.py \\
        --effect-sizes 0.15 0.25 0.50 \\
        --sample-sizes 10 30 50 \\
        --scenarios within_module_dense hub \\
        --methods tstat tfnbs nbs_extent \\
        --n-repeats 10 --n-permutations 500

Output (saved to --output-dir, default examples/simulation_validation/results/power_analysis):
    power_vs_effect_size.csv          — raw results (all methods × scenarios × repeats)
    power_vs_effect_size_summary.csv  — aggregated mean/std per condition
    power_vs_sample_size.csv          — raw results
    power_vs_sample_size_summary.csv  — aggregated mean/std per condition
    required_sample_sizes.csv         — min N for 80% power per method/scenario
    per_scenario/                     — one CSV per scenario for each mode
    checkpoint_*.csv                  — intermediate (removed on completion)

Plotting:
    Use plot_power_analysis_results.py to visualize results:
        python examples/simulation_validation/power/plot_power_analysis_results.py \\
            examples/simulation_validation/results/power_analysis_local
"""

from __future__ import annotations

import os

# Pin each worker process to 1 BLAS/LAPACK thread.
# MUST be set before `import numpy` — libraries read these at load time only.
#
# Why: mp.Pool spawns N workers doing NumPy math. Without this, each worker
# spawns its own BLAS thread pool (default = all cores), causing
# N_workers × N_cores threads fighting for N_cores CPUs.
# With these = 1: N workers × 1 thread = N threads on N cores (optimal).
#
# OMP_NUM_THREADS      — OpenMP (covers GCC/Clang BLAS, most Linux distros)
# OPENBLAS_NUM_THREADS — OpenBLAS (common NumPy backend on Linux servers)
# MKL_NUM_THREADS      — Intel MKL (conda-installed NumPy on Intel CPUs)
# VECLIB_MAXIMUM_THREADS — Apple Accelerate (macOS only)
# NUMBA_NUM_THREADS    — Numba's own thread pool (prange, parallel=True)
_SINGLE_THREAD = "1"
os.environ["OMP_NUM_THREADS"] = _SINGLE_THREAD
os.environ["OPENBLAS_NUM_THREADS"] = _SINGLE_THREAD
os.environ["MKL_NUM_THREADS"] = _SINGLE_THREAD
os.environ["VECLIB_MAXIMUM_THREADS"] = _SINGLE_THREAD
os.environ["NUMBA_NUM_THREADS"] = _SINGLE_THREAD


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

from tqdm import tqdm

from conninfpy import topologies as topo
from conninfpy.pairwise_stats import compute_p_val, get_available_cores


import multiprocessing as mp


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


def _apply_label_override(net_labels, label_override, n_nodes):
    """Apply label override for prior misspecification experiments.

    Parameters
    ----------
    net_labels : ndarray
        True network labels from the dataset.
    label_override : str or None
        Override mode: 'true' (no change), '7blocks', '2blocks', or None.
    n_nodes : int
        Number of nodes.

    Returns
    -------
    ndarray
        Possibly overridden labels.
    """
    if label_override is None or label_override == "true":
        return net_labels
    if label_override == "7blocks":
        return np.sort(np.arange(n_nodes) % 7)
    elif label_override == "2blocks":
        return np.sort(np.arange(n_nodes) % 2)
    else:
        raise ValueError(f"Unknown label_override: '{label_override}'")


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
    label_override: str = None,
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
    net_labels = _apply_label_override(dataset.net_labels, label_override, n_nodes)

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
            use_mp=False,
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
def _parallel_worker(task):
    (method, method_params, kwargs, scenario, effect_size, repeat_id,
     n_nodes, n_modules, time_points, n_permutations, alpha, seed,
     intra_corr, inter_corr, uniform_corr, noise_level, n_samples,
     label_override) = task

    return run_single_power_test(
        method=method, method_params=method_params, scenario=scenario,
        effect_size=effect_size, n_samples=n_samples, repeat_id=repeat_id,
        n_nodes=n_nodes, n_modules=n_modules, time_points=time_points,
        n_permutations=n_permutations, alpha=alpha,
        use_mp=False,  # CRITICAL: Each core handles one whole test
        seed_base=seed, method_kwargs=kwargs,
        intra_corr=intra_corr, inter_corr=inter_corr,
        uniform_corr=uniform_corr, noise_level=noise_level,
        label_override=label_override,
    )


# =============================================================================
# BATCHED WORKER — generates data once, computes all (e,h) variants at once
# =============================================================================

TFNBS_METHODS = {"tfnbs", "ni_tfnbs", "fbc_tfnbs"}


def run_single_power_test_batched(
    method: str,
    all_method_params: List[Tuple[str, dict]],
    scenario: str,
    effect_size: float,
    n_samples: int,
    repeat_id: int,
    n_nodes: int = 60,
    n_modules: int = 4,
    time_points: int = 30,
    n_permutations: int = 500,
    alpha: float = 0.05,
    seed_base: int = 42,
    intra_corr: float = 0.3,
    inter_corr: float = 0.05,
    uniform_corr: float = 0.15,
    noise_level: float = 0.05,
    label_override: str = None,
) -> List[PowerResult]:
    """Run a single power simulation with batched (e,h) parameter variants.

    For TFNBS-family methods, generates data once and calls compute_p_val
    with all (e,h) pairs as lists, producing 3D p-value output.
    For other methods, runs each variant separately (usually just 1).

    Parameters
    ----------
    method : str
        Method name (e.g. 'tfnbs', 'ni_tfnbs', 'nbs_extent').
    all_method_params : list of (str, dict)
        List of (param_label, kwargs) pairs to batch.
    scenario : str
        Topology scenario name.
    effect_size : float
        Effect size for data generation.
    n_samples : int
        Samples per group.
    repeat_id : int
        Repeat identifier (used for seeding).
    Other parameters match run_single_power_test.

    Returns
    -------
    list of PowerResult
        One result per parameter variant.
    """
    start_time = time.time()

    # Generate data ONCE
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
    net_labels = _apply_label_override(dataset.net_labels, label_override, n_nodes)
    triu_idx = np.triu_indices(n_nodes, k=1)
    mask_upper = effect_mask[triu_idx] != 0
    n_true = int(np.sum(mask_upper))
    n_null = int(np.sum(~mask_upper))

    results = []
    method_key = method.split("_")[0] if method.startswith("nbs_") else method

    if method in TFNBS_METHODS and len(all_method_params) > 1:
        # BATCHED: collect all (e,h) pairs into lists for 3D mode
        e_list = [kw["e"] for _, kw in all_method_params]
        h_list = [kw["h"] for _, kw in all_method_params]
        # Non-e/h params should be consistent across variants
        base_kw = {k: v for k, v in all_method_params[0][1].items()
                   if k not in ("e", "h")}
        if method in ("cnbs", "ni_tfnbs", "fbc_tfnbs"):
            base_kw["net_labels"] = net_labels

        try:
            p_vals = compute_p_val(
                group1_z, group2_z,
                test_type="two-sample",
                method=method_key,
                n_permutations=n_permutations,
                use_mp=False,
                random_state=seed,
                e=e_list, h=h_list,
                **base_kw,
            )
            elapsed = time.time() - start_time
            avg_elapsed = elapsed / len(all_method_params)

            p_pos_3d = np.array(p_vals["g2>g1"])  # (N, N, n_params)

            for i, (param_label, _) in enumerate(all_method_params):
                if p_pos_3d.ndim == 3:
                    p_upper = p_pos_3d[:, :, i][triu_idx]
                else:
                    # Scalar fallback (single param)
                    p_upper = p_pos_3d[triu_idx]

                sig_mask = p_upper < alpha
                TP = int(np.sum(sig_mask & mask_upper))
                FP = int(np.sum(sig_mask & ~mask_upper))
                FN = int(np.sum(~sig_mask & mask_upper))
                TN = int(np.sum(~sig_mask & ~mask_upper))
                TPR = TP / n_true if n_true > 0 else 0.0
                FPR = FP / n_null if n_null > 0 else 0.0
                precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
                FDR = FP / (TP + FP) if (TP + FP) > 0 else 0.0

                results.append(PowerResult(
                    method=method, method_params=param_label,
                    scenario=scenario, effect_size=effect_size,
                    n_samples=n_samples, repeat_id=repeat_id,
                    n_true=n_true, n_null=n_null,
                    TP=TP, FP=FP, FN=FN, TN=TN,
                    TPR=TPR, FPR=FPR,
                    precision=precision, FDR=FDR,
                    elapsed_time=avg_elapsed,
                ))

        except Exception as exc:
            print(f"  ERROR in batched {method} repeat {repeat_id}: {exc}")
            elapsed = time.time() - start_time
            for param_label, _ in all_method_params:
                results.append(PowerResult(
                    method=method, method_params=param_label,
                    scenario=scenario, effect_size=effect_size,
                    n_samples=n_samples, repeat_id=repeat_id,
                    n_true=n_true, n_null=n_null,
                    TP=0, FP=0, FN=n_true, TN=n_null,
                    TPR=0.0, FPR=0.0, precision=0.0, FDR=0.0,
                    elapsed_time=elapsed / len(all_method_params),
                ))
    else:
        # NON-BATCHED: run each variant separately (tstat, nbs, cnbs, or single-param TFNBS)
        for param_label, kwargs in all_method_params:
            kw = dict(kwargs)
            if method.startswith("nbs_"):
                kw["nbs_stat"] = method.split("_")[1]
            if method in ("cnbs", "ni_tfnbs", "fbc_tfnbs"):
                kw["net_labels"] = net_labels

            try:
                p_vals = compute_p_val(
                    group1_z, group2_z,
                    test_type="two-sample",
                    method=method_key,
                    n_permutations=n_permutations,
                    use_mp=False,
                    random_state=seed,
                    **kw,
                )
                p_pos = np.array(p_vals["g2>g1"])
                p_upper = p_pos[triu_idx]
                sig_mask = p_upper < alpha
                TP = int(np.sum(sig_mask & mask_upper))
                FP = int(np.sum(sig_mask & ~mask_upper))
                FN = int(np.sum(~sig_mask & mask_upper))
                TN = int(np.sum(~sig_mask & ~mask_upper))
                TPR = TP / n_true if n_true > 0 else 0.0
                FPR = FP / n_null if n_null > 0 else 0.0
                precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
                FDR = FP / (TP + FP) if (TP + FP) > 0 else 0.0
            except Exception as exc:
                print(f"  ERROR in {method} repeat {repeat_id}: {exc}")
                TP, FP, FN, TN = 0, 0, n_true, n_null
                TPR, FPR, precision, FDR = 0.0, 0.0, 0.0, 0.0

            elapsed = time.time() - start_time
            results.append(PowerResult(
                method=method, method_params=param_label,
                scenario=scenario, effect_size=effect_size,
                n_samples=n_samples, repeat_id=repeat_id,
                n_true=n_true, n_null=n_null,
                TP=TP, FP=FP, FN=FN, TN=TN,
                TPR=TPR, FPR=FPR,
                precision=precision, FDR=FDR,
                elapsed_time=elapsed,
            ))

    return results


def _parallel_worker_batched(task):
    """Unpack a batched task tuple and call run_single_power_test_batched."""
    (method, all_method_params, scenario, effect_size, repeat_id,
     n_nodes, n_modules, time_points, n_permutations, alpha, seed,
     intra_corr, inter_corr, uniform_corr, noise_level, n_samples,
     label_override) = task

    return run_single_power_test_batched(
        method=method,
        all_method_params=all_method_params,
        scenario=scenario,
        effect_size=effect_size,
        n_samples=n_samples,
        repeat_id=repeat_id,
        n_nodes=n_nodes,
        n_modules=n_modules,
        time_points=time_points,
        n_permutations=n_permutations,
        alpha=alpha,
        seed_base=seed,
        intra_corr=intra_corr,
        inter_corr=inter_corr,
        uniform_corr=uniform_corr,
        noise_level=noise_level,
        label_override=label_override,
    )

# =============================================================================
# TIME ESTIMATION — benchmarked with numba backend, 60 nodes, n_thresholds=30
# =============================================================================

# Per-task seconds at 100 permutations (measured on Apple M-series, numba)
_BENCH_PERMS = 100
_TASK_SECS = {
    "tstat": 0.9,
    "tfnbs": 5.5,           # batched 176 (e,h) combos
    "ni_tfnbs": 4.8,        # batched 176 (e,h) combos
    "fbc_tfnbs": 4.7,       # batched 176 (e,h) combos
    "nbs_extent": 1.2,      # per threshold variant
    "nbs_intensity": 1.5,   # per threshold variant
    "cnbs": 1.3,
    "bonferroni": 0.02,     # parametric, no permutations
    "bh_fdr": 0.02,         # parametric, no permutations
    "bh_fdr_perm": 0.9,     # permutation-based, same as tstat
}


def estimate_runtime(
    batched_groups: List[Tuple[str, list]],
    n_conditions: int,
    n_permutations: int,
    n_cores: int,
) -> Tuple[float, float]:
    """Estimate sequential and parallel runtime in seconds.

    Parameters
    ----------
    batched_groups : list of (method, param_list)
        Output from _group_variants_for_batching.
    n_conditions : int
        Number of (scenario × sweep_value × repeat) combinations.
    n_permutations : int
        Permutations per test.
    n_cores : int
        Available CPU cores for parallel execution.

    Returns
    -------
    (sequential_seconds, parallel_seconds)
    """
    perm_scale = n_permutations / _BENCH_PERMS

    total_task_secs = 0.0
    for method, param_list in batched_groups:
        base_time = _TASK_SECS.get(method, 2.0)
        total_task_secs += base_time * perm_scale

    sequential = total_task_secs * n_conditions
    n_tasks = n_conditions * len(batched_groups)
    parallel = sequential / max(n_cores, 1)

    return sequential, parallel


def print_time_estimate(
    mode: str,
    batched_groups: List[Tuple[str, list]],
    n_conditions: int,
    n_tasks: int,
    n_permutations: int,
    n_cores: int,
):
    """Print estimated runtime before launching."""
    seq, par = estimate_runtime(batched_groups, n_conditions, n_permutations, n_cores)

    def _fmt(secs):
        if secs < 120:
            return f"{secs:.0f}s"
        elif secs < 7200:
            return f"{secs/60:.0f}min"
        else:
            return f"{secs/3600:.1f}h"

    print(f"Estimated runtime ({mode}):")
    print(f"  {n_tasks} tasks, {n_conditions} conditions, {n_permutations} perms")
    print(f"  Sequential: {_fmt(seq)} | Parallel ({n_cores} cores): {_fmt(par)}")
    print()


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
        "bonferroni": {},
        "bh_fdr": {},
        "bh_fdr_perm": {},
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


def _group_variants_for_batching(
    method_variants: List[Tuple[str, dict]],
) -> List[Tuple[str, List[Tuple[str, dict]]]]:
    """Group method variants for batched execution.

    TFNBS-family methods (tfnbs, ni_tfnbs, fbc_tfnbs) are grouped by base method,
    so all (e,h) parameter combos run in a single compute_p_val call (3D mode).
    Other methods remain as individual tasks (group of 1).

    Returns
    -------
    list of (method, [(param_label, kwargs), ...])
        Each element is a method with its list of parameter variants.
    """
    from collections import OrderedDict

    groups: Dict[str, List[Tuple[str, dict]]] = OrderedDict()

    for method, kwargs in method_variants:
        if method in TFNBS_METHODS:
            key = method  # group all variants of same TFNBS method
        else:
            # Each non-TFNBS variant is its own group
            param_label = _format_method_params(method, kwargs)
            key = f"{method}|{param_label}"

        if key not in groups:
            groups[key] = (method, [])
        param_label = _format_method_params(method, kwargs)
        groups[key][1].append((param_label, kwargs))

    return [(method, params) for method, params in groups.values()]


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
    label_override: str = None,
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
    batched_groups = _group_variants_for_batching(method_variants)

    # Build batched task list (skipping completed)
    tasks = []
    total_result_count = 0
    for scenario in scenarios:
        for effect_size in effect_sizes:
            for method, param_list in batched_groups:
                for repeat_id in range(n_repeats):
                    # Filter out already-completed parameter variants
                    pending_params = []
                    for param_label, kwargs in param_list:
                        key = (method, param_label, scenario, effect_size, n_samples, repeat_id)
                        if key not in completed:
                            pending_params.append((param_label, kwargs))
                    total_result_count += len(param_list)
                    if pending_params:
                        tasks.append((method, pending_params, scenario, effect_size, repeat_id))

    total_runs = total_result_count
    remaining_results = sum(len(params) for _, params, _, _, _ in tasks)

    if remaining_results == 0:
        print("All runs already completed!")
        return pd.DataFrame([vars(r) for r in results])

    print(f"=== Power vs Effect Size (BATCHED) ===")
    print(f"Methods: {methods} (variants: {len(method_variants)})")
    print(f"Batched groups: {len(batched_groups)} "
          f"({', '.join(f'{m}[{len(p)}]' for m, p in batched_groups)})")
    print(f"Scenarios: {scenarios}")
    print(f"Effect sizes: {effect_sizes}")
    print(f"Repeats per condition: {n_repeats}")
    print(f"Total results: {total_runs}, Remaining: {remaining_results}")
    print(f"Batched tasks: {len(tasks)} (vs {remaining_results} unbatched)")
    print()

    n_conditions = len(scenarios) * len(effect_sizes) * n_repeats
    n_cores = get_available_cores()
    print_time_estimate("effect-size", batched_groups, n_conditions,
                        len(tasks), n_permutations, n_cores)

    full_tasks = [
        (method, pending_params, scenario, effect_size, repeat_id,
         n_nodes, n_modules, time_points, n_permutations, alpha, seed,
         intra_corr, inter_corr, uniform_corr, noise_level, n_samples,
         label_override)
        for (method, pending_params, scenario, effect_size, repeat_id) in tasks
    ]

    print(f"Launching parallel execution on {n_cores} cores...")

    ctx = mp.get_context('spawn')

    with ctx.Pool(processes=n_cores) as pool:
        pbar = tqdm(
            pool.imap_unordered(_parallel_worker_batched, full_tasks),
            total=len(full_tasks),
            desc="Effect-size",
            unit="task",
            miniters=1,
        )
        for i, batch_results in enumerate(pbar, 1):
            results.extend(batch_results)
            pbar.set_postfix_str(
                f"{batch_results[0].scenario[:12]} {batch_results[0].method} "
                f"ES={batch_results[0].effect_size}"
            )

            if checkpoint_path and (i % checkpoint_every == 0 or i == len(full_tasks)):
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
    label_override: str = None,
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
    batched_groups = _group_variants_for_batching(method_variants)

    # Build batched task list (skipping completed)
    tasks = []
    total_result_count = 0
    for scenario in scenarios:
        for n_samples in sample_sizes:
            for method, param_list in batched_groups:
                for repeat_id in range(n_repeats):
                    pending_params = []
                    for param_label, kwargs in param_list:
                        key = (method, param_label, scenario, effect_size, n_samples, repeat_id)
                        if key not in completed:
                            pending_params.append((param_label, kwargs))
                    total_result_count += len(param_list)
                    if pending_params:
                        tasks.append((method, pending_params, scenario, n_samples, repeat_id))

    total_runs = total_result_count
    remaining_results = sum(len(params) for _, params, _, _, _ in tasks)

    if remaining_results == 0:
        print("All runs already completed!")
        return pd.DataFrame([vars(r) for r in results])

    print(f"=== Power vs Sample Size (BATCHED) ===")
    print(f"Methods: {methods} (variants: {len(method_variants)})")
    print(f"Batched groups: {len(batched_groups)} "
          f"({', '.join(f'{m}[{len(p)}]' for m, p in batched_groups)})")
    print(f"Scenarios: {scenarios}")
    print(f"Sample sizes: {sample_sizes}")
    print(f"Effect size: {effect_size}")
    print(f"Repeats per condition: {n_repeats}")
    print(f"Total results: {total_runs}, Remaining: {remaining_results}")
    print(f"Batched tasks: {len(tasks)} (vs {remaining_results} unbatched)")
    print()

    n_conditions = len(scenarios) * len(sample_sizes) * n_repeats
    n_cores = get_available_cores()
    print_time_estimate("sample-size", batched_groups, n_conditions,
                        len(tasks), n_permutations, n_cores)

    full_tasks = [
        (method, pending_params, scenario, effect_size, repeat_id,
         n_nodes, n_modules, time_points, n_permutations, alpha, seed,
         intra_corr, inter_corr, uniform_corr, noise_level, n_samples,
         label_override)
        for (method, pending_params, scenario, n_samples, repeat_id) in tasks
    ]

    print(f"Launching parallel execution on {n_cores} cores...")

    ctx = mp.get_context('spawn')

    with ctx.Pool(processes=n_cores) as pool:
        pbar = tqdm(
            pool.imap_unordered(_parallel_worker_batched, full_tasks),
            total=len(full_tasks),
            desc="Sample-size",
            unit="task",
            miniters=1,
        )
        for i, batch_results in enumerate(pbar, 1):
            results.extend(batch_results)
            pbar.set_postfix_str(
                f"{batch_results[0].scenario[:12]} {batch_results[0].method} "
                f"N={batch_results[0].n_samples}"
            )

            if checkpoint_path and (i % checkpoint_every == 0 or i == len(full_tasks)):
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


def save_per_scenario(df: pd.DataFrame, output_dir: Path, prefix: str) -> None:
    """Save separate CSV files for each scenario."""
    scenario_dir = output_dir / "per_scenario"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    for scenario, sdf in df.groupby("scenario"):
        path = scenario_dir / f"{prefix}_{scenario}.csv"
        sdf.to_csv(path, index=False)
    print(f"  Saved {df['scenario'].nunique()} per-scenario files to {scenario_dir}/")


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

def setup_multiprocessing():
    """setup multiprocessing method"""
    try:
        mp.set_start_method('spawn', force=True)
        print(f"Multiprocessing start method set to: 'spawn' (forced)")
        print(f"   System: {sys.platform}, Default would be: {mp.get_start_method()}")
    except RuntimeError as e:

        current = mp.get_start_method()
        print(f"ℹMultiprocessing start method already set to: '{current}'")
        print(f"   (Attempted to set 'spawn' but was ignored)")

def main(argv: Optional[List[str]] = None) -> int:
    print('Thread pinning (1 BLAS thread per worker):')
    for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"]:
        value = os.environ.get(var, 'NOT SET')
        print(f"  {var}: {value}")

    print(f'Available cores: {get_available_cores()}')

    setup_multiprocessing()


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
        default=["tstat", "tfnbs", "ni_tfnbs", "fbc_tfnbs", "nbs_extent", "cnbs",
                 "bonferroni", "bh_fdr", "bh_fdr_perm"],
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
        default=Path(__file__).parent.parent.parent / "results" / "power_analysis",
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
    parser.add_argument(
        "--label-override", type=str, default=None,
        choices=["true", "7blocks", "2blocks"],
        help="Override network labels for prior misspecification experiments. "
             "'true' uses ground-truth labels, '7blocks'/'2blocks' use synthetic equi-sized blocks.",
    )
    parser.add_argument(
        "--fbc-min-cluster", type=int, default=None,
        help="Override fbc_min_cluster_size in tfnbs_options (for m_min sweep).",
    )
    parser.add_argument(
        "--ni-normalization", type=str, default=None,
        choices=["sqrt", "linear", "none"],
        help="Override NI-TFNBS normalization variant (for ablation study).",
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
    # Apply CLI overrides for fbc_min_cluster and ni_normalization
    if args.fbc_min_cluster is not None:
        if tfnbs_options is None:
            tfnbs_options = {}
        tfnbs_options["fbc_min_cluster_size"] = args.fbc_min_cluster
        print(f"CLI override: fbc_min_cluster_size = {args.fbc_min_cluster}")

    method_configs = get_method_configs(
        tfnbs_options=tfnbs_options,
        nbs_options=nbs_options,
    )

    # Apply ni_normalization override to ni_tfnbs method configs
    if args.ni_normalization is not None:
        for method_key in ("ni_tfnbs",):
            cfg = method_configs.get(method_key)
            if isinstance(cfg, list):
                for c in cfg:
                    c["normalization"] = args.ni_normalization
            elif isinstance(cfg, dict):
                cfg["normalization"] = args.ni_normalization
        print(f"CLI override: ni_normalization = {args.ni_normalization}")

    if tfnbs_options or nbs_options:
        print("Using method options from config:")
        if tfnbs_options:
            print(f"  tfnbs_options: {tfnbs_options}")
        if nbs_options:
            print(f"  nbs_options: {nbs_options}")

    if args.label_override:
        print(f"Label override: {args.label_override}")

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
            use_mp=False,
            seed=args.seed,
            intra_corr=args.intra_corr,
            inter_corr=args.inter_corr,
            uniform_corr=args.uniform_corr,
            noise_level=args.noise_level,
            method_configs=method_configs,
            checkpoint_dir=args.output_dir,
            checkpoint_every=args.checkpoint_every,
            resume=args.resume,
            label_override=args.label_override,
        )
        df_effect.to_csv(args.output_dir / "power_vs_effect_size.csv", index=False)
        save_per_scenario(df_effect, args.output_dir, "power_vs_effect_size")

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
            use_mp=False,
            seed=args.seed,
            intra_corr=args.intra_corr,
            inter_corr=args.inter_corr,
            uniform_corr=args.uniform_corr,
            noise_level=args.noise_level,
            method_configs=method_configs,
            checkpoint_dir=args.output_dir,
            checkpoint_every=args.checkpoint_every,
            resume=args.resume,
            label_override=args.label_override,
        )
        df_sample.to_csv(args.output_dir / "power_vs_sample_size.csv", index=False)
        save_per_scenario(df_sample, args.output_dir, "power_vs_sample_size")

        # Summary
        summary_sample = summarize_power_results(
            df_sample, ["method", "method_params", "scenario", "n_samples"]
        )
        summary_sample.to_csv(args.output_dir / "power_vs_sample_size_summary.csv", index=False)

        # Required sample size
        required_n = compute_required_sample_size(df_sample, target_power=0.8)
        required_n.to_csv(args.output_dir / "required_sample_sizes.csv", index=False)

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
    main()
