"""
False Positive Rate (FPR) Calibration Test for TFNBS Methods.

This script validates that all statistical methods maintain the nominal
false positive rate (FPR) under the null hypothesis (no true effect).

General protocol:
1. Generate N null datasets (effect_size=0)
2. Run each method with full permutation testing
3. Check if any edge is significant at alpha
4. Verify FPR = alpha +/- sampling_error

Usage:
    # Quick test (20 null runs) - for development
    python fpr_calibration.py --config fpr_config_quick.yaml

    # Full validation (1000 null runs) - for publication
    python fpr_calibration.py --config fpr_config.yaml

    # Resume interrupted run
    python fpr_calibration.py --config fpr_config.yaml --resume

    # Specific methods only
    python fpr_calibration.py --methods tstat tfnbs cnbs --n-null 100

Output:
    - results/fpr_calibration/fpr_results.csv
    - results/fpr_calibration/fpr_summary.csv
    - results/fpr_calibration/fpr_calibration_plot.png
    - results/fpr_calibration/checkpoint.csv (for resume)
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tfnbs.pairwise_stats import compute_p_val
from tfnbs.synth_datasets import ModularDatasetGenerator


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class FPRResult:
    """Result from a single null simulation."""

    method: str
    null_run_id: int
    n_edges: int              # Total edges tested (upper triangle)
    # Positive direction (g2 > g1)
    any_significant_pos: bool  # Any p < alpha (FWER)
    n_significant_pos: int     # Count of significant edges
    fpr_pos: float             # n_significant_pos / n_edges (edge-wise FPR)
    min_p_pos: float
    # Negative direction (g1 > g2)
    any_significant_neg: bool
    n_significant_neg: int
    fpr_neg: float             # n_significant_neg / n_edges
    min_p_neg: float
    # Combined (either direction) - NOTE: this is for two-sided testing
    any_significant: bool      # Either direction has any significant
    n_significant_total: int   # Total significant edges (both directions)
    fpr_total: float           # Combined edge-wise FPR
    elapsed_time: float


@dataclass
class FPRConfig:
    """Configuration for FPR calibration run."""

    # Methods to test
    methods: List[str]

    # Null simulation parameters
    n_null: int = 100
    n_permutations: int = 500
    alpha: float = 0.05

    # Data generation parameters
    n_nodes: int = 60
    n_modules: int = 4
    n_samples: int = 20
    intra_corr: float = 0.3
    inter_corr: float = 0.05
    noise_level: float = 0.05

    # Method-specific parameters
    tfnbs_e: float = 0.5
    tfnbs_h: float = 2.0
    tfnbs_n: int = 50
    tfnbs_start_thres: float = 1.65
    nbs_threshold: float = 2.0
    fbc_min_cluster: int = 3

    # Execution options
    seed: int = 42
    use_mp: bool = False
    n_jobs: int = -1            # -1 = use all cores for parallel null runs
    checkpoint_every: int = 10  # Save checkpoint every N runs per method

    # Output
    output_dir: Path = Path("results/fpr_calibration")

    @classmethod
    def from_yaml(cls, path: Path) -> "FPRConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        # Handle output_dir as Path
        if "output_dir" in data:
            data["output_dir"] = Path(data["output_dir"])

        return cls(**data)

    def get_method_kwargs(self, method: str) -> dict:
        """Get method-specific kwargs."""
        configs = {
            "tstat": {},
            "tfnbs": {
                "e": self.tfnbs_e,
                "h": self.tfnbs_h,
                "n": self.tfnbs_n,
                "start_thres": self.tfnbs_start_thres,
            },
            "ni_tfnbs": {
                "e": self.tfnbs_e,
                "h": self.tfnbs_h,
                "n": self.tfnbs_n,
                "start_thres": self.tfnbs_start_thres,
            },
            "fbc_tfnbs": {
                "e": self.tfnbs_e,
                "h": self.tfnbs_h,
                "n": self.tfnbs_n,
                "start_thres": self.tfnbs_start_thres,
                "min_cluster_size": self.fbc_min_cluster,
            },
            "nbs_extent": {"threshold": self.nbs_threshold, "nbs_stat": "extent"},
            "nbs_intensity": {"threshold": self.nbs_threshold, "nbs_stat": "intensity"},
            "cnbs": {},
        }
        return configs.get(method, {})

    def get_compute_method(self, method: str) -> str:
        """Map display method name to compute_p_val method name."""
        if method.startswith("nbs_"):
            return "nbs"
        return method


# =============================================================================
# NULL DATA GENERATION
# =============================================================================

def generate_null_dataset(
    n_nodes: int = 60,
    n_modules: int = 4,
    n_samples: int = 20,
    intra_corr: float = 0.3,
    inter_corr: float = 0.05,
    noise_level: float = 0.05,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a null dataset with no true effect."""

    generator = ModularDatasetGenerator(
        N=n_nodes,  # ModularDatasetGenerator uses 'N' for number of nodes
        n_modules=n_modules,
        intra_corr=intra_corr,
        inter_corr=inter_corr,
        noise_level=noise_level,
        seed=seed,
    )

    # effect size = 0 (i.e. no true difference)
    # effect_mask must be provided but effect_size=0 means no effect
    effect_mask = np.zeros((n_nodes, n_nodes))
    group1, group2, net_labels = generator.generate_data(
        effect_mask=effect_mask,
        effect_size=0.0,  # NULL hypothesis
        n_samples_g1=n_samples,
        n_samples_g2=n_samples,
    )

    return group1, group2, net_labels


# =============================================================================
# SINGLE NULL TEST
# =============================================================================

def run_single_null_test(
    method: str,
    null_run_id: int,
    config: FPRConfig,
) -> FPRResult:
    """Run a single null hypothesis test for one method."""

    start_time = time.time()

    # Generate null data with unique seed
    seed = config.seed + null_run_id * 1000
    group1, group2, net_labels = generate_null_dataset(
        n_nodes=config.n_nodes,
        n_modules=config.n_modules,
        n_samples=config.n_samples,
        intra_corr=config.intra_corr,
        inter_corr=config.inter_corr,
        noise_level=config.noise_level,
        seed=seed,
    )

    # Apply Fisher z-transform
    group1_z = np.arctanh(np.clip(group1, -0.999, 0.999))
    group2_z = np.arctanh(np.clip(group2, -0.999, 0.999))

    # Prepare method kwargs
    kwargs = config.get_method_kwargs(method)
    compute_method = config.get_compute_method(method)

    if compute_method in ["cnbs", "ni_tfnbs", "fbc_tfnbs"]:
        kwargs["net_labels"] = net_labels

    # Number of edges in upper triangle
    n_edges = config.n_nodes * (config.n_nodes - 1) // 2

    # Compute p-values
    try:
        p_vals = compute_p_val(
            group1_z, group2_z,
            test_type="two-sample",
            method=compute_method,
            n_permutations=config.n_permutations,
            use_mp=config.use_mp,
            random_state=seed,
            **kwargs,
        )

        p_pos = np.array(p_vals["g2>g1"])
        p_neg = np.array(p_vals["g1>g2"])

        # Get upper triangle only (symmetric matrix)
        triu_idx = np.triu_indices(config.n_nodes, k=1)
        p_pos_upper = p_pos[triu_idx]
        p_neg_upper = p_neg[triu_idx]

        # Positive direction metrics
        sig_mask_pos = p_pos_upper < config.alpha
        any_sig_pos = bool(np.any(sig_mask_pos))
        n_sig_pos = int(np.sum(sig_mask_pos))
        fpr_pos = n_sig_pos / n_edges
        min_p_pos = float(np.min(p_pos_upper))

        # Negative direction metrics
        sig_mask_neg = p_neg_upper < config.alpha
        any_sig_neg = bool(np.any(sig_mask_neg))
        n_sig_neg = int(np.sum(sig_mask_neg))
        fpr_neg = n_sig_neg / n_edges
        min_p_neg = float(np.min(p_neg_upper))

        # Combined metrics (both directions)
        n_sig_total = n_sig_pos + n_sig_neg
        fpr_total = n_sig_total / (2 * n_edges)  # Denominator is 2*n_edges for two-sided

    except Exception as e:
        print(f"  ERROR in {method} run {null_run_id}: {e}")
        any_sig_pos = False
        any_sig_neg = False
        n_sig_pos = 0
        n_sig_neg = 0
        fpr_pos = 0.0
        fpr_neg = 0.0
        n_sig_total = 0
        fpr_total = 0.0
        min_p_pos = 1.0
        min_p_neg = 1.0

    elapsed = time.time() - start_time

    return FPRResult(
        method=method,
        null_run_id=null_run_id,
        n_edges=n_edges,
        any_significant_pos=any_sig_pos,
        n_significant_pos=n_sig_pos,
        fpr_pos=fpr_pos,
        min_p_pos=min_p_pos,
        any_significant_neg=any_sig_neg,
        n_significant_neg=n_sig_neg,
        fpr_neg=fpr_neg,
        min_p_neg=min_p_neg,
        any_significant=any_sig_pos or any_sig_neg,
        n_significant_total=n_sig_total,
        fpr_total=fpr_total,
        elapsed_time=elapsed,
    )


def _run_null_test_wrapper(args: tuple) -> FPRResult:
    """Wrapper for parallel execution."""
    method, null_run_id, config = args
    return run_single_null_test(method, null_run_id, config)


# =============================================================================
# CHECKPOINTING
# =============================================================================

def save_checkpoint(results: List[FPRResult], checkpoint_path: Path) -> None:
    """Save intermediate results to checkpoint file."""
    df = pd.DataFrame([asdict(r) for r in results])
    df.to_csv(checkpoint_path, index=False)


def load_checkpoint(checkpoint_path: Path) -> Tuple[List[FPRResult], set]:
    """Load results from checkpoint and return completed (method, run_id) pairs."""
    if not checkpoint_path.exists():
        return [], set()

    df = pd.read_csv(checkpoint_path)
    results = []
    completed = set()

    for _, row in df.iterrows():
        result = FPRResult(
            method=row["method"],
            null_run_id=int(row["null_run_id"]),
            n_edges=int(row["n_edges"]),
            any_significant_pos=bool(row["any_significant_pos"]),
            n_significant_pos=int(row["n_significant_pos"]),
            fpr_pos=float(row["fpr_pos"]),
            min_p_pos=float(row["min_p_pos"]),
            any_significant_neg=bool(row["any_significant_neg"]),
            n_significant_neg=int(row["n_significant_neg"]),
            fpr_neg=float(row["fpr_neg"]),
            min_p_neg=float(row["min_p_neg"]),
            any_significant=bool(row["any_significant"]),
            n_significant_total=int(row["n_significant_total"]),
            fpr_total=float(row["fpr_total"]),
            elapsed_time=float(row["elapsed_time"]),
        )
        results.append(result)
        completed.add((row["method"], int(row["null_run_id"])))

    return results, completed


# =============================================================================
# MAIN CALIBRATION RUNNER
# =============================================================================

def run_fpr_calibration(
    config: FPRConfig,
    resume: bool = False,
) -> pd.DataFrame:
    """Run full FPR calibration test with optional parallelization and checkpointing."""

    checkpoint_path = config.output_dir / "checkpoint.csv"
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint if resuming
    if resume:
        results, completed = load_checkpoint(checkpoint_path)
        print(f"Resuming from checkpoint: {len(completed)} runs already completed")
    else:
        results = []
        completed = set()

    # Build task list
    tasks = []
    for method in config.methods:
        for null_id in range(config.n_null):
            if (method, null_id) not in completed:
                tasks.append((method, null_id, config))

    if not tasks:
        print("All runs already completed!")
        return pd.DataFrame([asdict(r) for r in results])

    total_runs = len(tasks)
    print(f"=== FPR Calibration Test ===")
    print(f"Methods: {config.methods}")
    print(f"Null datasets per method: {config.n_null}")
    print(f"Permutations per test: {config.n_permutations}")
    print(f"Alpha: {config.alpha}")
    print(f"Parallel jobs: {config.n_jobs}")
    print(f"Remaining runs: {total_runs}")
    print(f"Expected FPR: {config.alpha:.3f} +/- {1.96 * np.sqrt(config.alpha * (1-config.alpha) / config.n_null):.3f}")
    print()

    # Check if joblib is available for parallel execution
    use_parallel = config.n_jobs != 1 and total_runs > 1
    if use_parallel:
        try:
            from joblib import Parallel, delayed
        except ImportError:
            print("joblib not available, falling back to sequential execution")
            use_parallel = False

    start_time = time.time()

    if use_parallel:
        # Parallel execution with progress updates
        n_jobs = config.n_jobs if config.n_jobs > 0 else -1

        # Process in batches for checkpointing
        batch_size = config.checkpoint_every * len(config.methods)

        for batch_start in range(0, len(tasks), batch_size):
            batch_tasks = tasks[batch_start:batch_start + batch_size]

            batch_results = Parallel(n_jobs=n_jobs, verbose=0)(
                delayed(_run_null_test_wrapper)(task) for task in batch_tasks
            )

            results.extend(batch_results)

            # Progress update
            completed_count = len(results)
            elapsed = time.time() - start_time
            eta = elapsed / completed_count * (total_runs + len(completed) - completed_count)

            # Compute current FPR per method
            print(f"\n[{completed_count}/{total_runs + len(completed)}] Elapsed: {elapsed/60:.1f}min, ETA: {eta/60:.1f}min")
            for method in config.methods:
                method_results = [r for r in results if r.method == method]
                if method_results:
                    fpr = np.mean([r.any_significant for r in method_results])
                    print(f"  {method}: FPR = {fpr:.4f} (n={len(method_results)})")

            # Save checkpoint
            save_checkpoint(results, checkpoint_path)

    else:
        # Sequential execution with progress
        for i, (method, null_id, cfg) in enumerate(tasks):
            result = run_single_null_test(method, null_id, cfg)
            results.append(result)

            if (i + 1) % config.checkpoint_every == 0:
                elapsed = time.time() - start_time
                eta = elapsed / (i + 1) * (total_runs - i - 1)
                method_results = [r for r in results if r.method == method]
                fpr = np.mean([r.any_significant for r in method_results])
                print(f"  [{i + 1}/{total_runs}] {method} FPR={fpr:.3f}, ETA={eta/60:.1f}min")

                # Save checkpoint
                save_checkpoint(results, checkpoint_path)

    # Final checkpoint save
    save_checkpoint(results, checkpoint_path)

    return pd.DataFrame([asdict(r) for r in results])


# =============================================================================
# STATISTICAL SUMMARY
# =============================================================================

def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Compute Clopper-Pearson exact binomial confidence interval.

    This is the "exact" confidence interval for binomial proportion,
    more accurate than the Wald interval for small samples or extreme proportions.
    """
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


def compute_fpr_summary(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Compute FPR summary statistics by method.

    Reports two key metrics:
    1. FWER (Family-Wise Error Rate): P(at least 1 FP | H0)
       - Proportion of null runs with any significant edge
       - For FWER-controlling methods, this should equal alpha

    2. Edge-wise FPR: E[proportion of FP edges | H0]
       - Mean proportion of edges falsely declared significant
       - For uncorrected tests, this should equal alpha
       - For FWER-controlling methods, this is << alpha
    """
    summary_rows = []

    for method in df["method"].unique():
        method_df = df[df["method"] == method]
        n_runs = len(method_df)
        n_edges = method_df["n_edges"].iloc[0]

        # =====================================================================
        # FWER: Proportion of runs with ANY significant edge
        # =====================================================================
        # Positive direction
        n_runs_with_fp_pos = int(method_df["any_significant_pos"].sum())
        fwer_pos = n_runs_with_fp_pos / n_runs

        # Negative direction
        n_runs_with_fp_neg = int(method_df["any_significant_neg"].sum())
        fwer_neg = n_runs_with_fp_neg / n_runs

        # Either direction (for two-sided test)
        n_runs_with_fp = int(method_df["any_significant"].sum())
        fwer = n_runs_with_fp / n_runs

        # Exact binomial CI for FWER (one-sided, positive direction)
        fwer_ci_low, fwer_ci_high = clopper_pearson_ci(n_runs_with_fp_pos, n_runs, alpha=0.05)

        # Check if FWER is calibrated (one-sided)
        fwer_calibrated = fwer_ci_low <= alpha <= fwer_ci_high

        # =====================================================================
        # Edge-wise FPR: Mean proportion of significant edges
        # =====================================================================
        # This is the key metric for edge-wise calibration
        mean_fpr_pos = method_df["fpr_pos"].mean()
        mean_fpr_neg = method_df["fpr_neg"].mean()
        mean_fpr_total = method_df["fpr_total"].mean()

        # Standard error for edge-wise FPR
        se_fpr_pos = method_df["fpr_pos"].std() / np.sqrt(n_runs)

        # CI for edge-wise FPR (using t-distribution for mean)
        from scipy import stats
        t_crit = stats.t.ppf(0.975, df=n_runs - 1)
        fpr_ci_low = max(0, mean_fpr_pos - t_crit * se_fpr_pos)
        fpr_ci_high = min(1, mean_fpr_pos + t_crit * se_fpr_pos)

        # Check if edge-wise FPR is calibrated
        fpr_calibrated = fpr_ci_low <= alpha <= fpr_ci_high

        # =====================================================================
        # Additional metrics
        # =====================================================================
        mean_n_fp_pos = method_df["n_significant_pos"].mean()
        mean_n_fp_neg = method_df["n_significant_neg"].mean()

        summary_rows.append({
            "method": method,
            "n_null_runs": n_runs,
            "n_edges": n_edges,
            # FWER metrics (one-sided, positive direction)
            "fwer_pos": fwer_pos,
            "fwer_neg": fwer_neg,
            "fwer_both": fwer,
            "fwer_ci_low": fwer_ci_low,
            "fwer_ci_high": fwer_ci_high,
            "fwer_calibrated": fwer_calibrated,
            # Edge-wise FPR metrics (one-sided, positive direction)
            "mean_fpr_pos": mean_fpr_pos,
            "mean_fpr_neg": mean_fpr_neg,
            "mean_fpr_total": mean_fpr_total,
            "fpr_se": se_fpr_pos,
            "fpr_ci_low": fpr_ci_low,
            "fpr_ci_high": fpr_ci_high,
            "fpr_calibrated": fpr_calibrated,
            # Expected value
            "expected": alpha,
            # Additional info
            "mean_n_fp_pos": mean_n_fp_pos,
            "mean_n_fp_neg": mean_n_fp_neg,
            "mean_elapsed_time": method_df["elapsed_time"].mean(),
        })

    return pd.DataFrame(summary_rows)


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_fpr_results(
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    alpha: float,
    output_path: Path,
) -> None:
    """Create FPR calibration plot with improved visualization."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    methods = summary_df["method"].tolist()
    colors_base = plt.cm.Set2(np.linspace(0, 1, len(methods)))

    # ==========================================================================
    # Plot 1: FWER by method (top-left)
    # ==========================================================================
    ax = axes[0, 0]
    x = np.arange(len(methods))

    fwer_vals = summary_df["fwer_pos"].tolist()
    fwer_ci_lows = summary_df["fwer_ci_low"].tolist()
    fwer_ci_highs = summary_df["fwer_ci_high"].tolist()

    bar_colors = ["#4CAF50" if row["fwer_calibrated"] else "#F44336"
                  for _, row in summary_df.iterrows()]

    ax.bar(x, fwer_vals, color=bar_colors, alpha=0.7, edgecolor="black", linewidth=1.2)
    ax.errorbar(
        x, fwer_vals,
        yerr=[np.array(fwer_vals) - np.array(fwer_ci_lows),
              np.array(fwer_ci_highs) - np.array(fwer_vals)],
        fmt="none", color="black", capsize=5, capthick=1.5, linewidth=1.5,
    )
    ax.axhline(alpha, color="blue", linestyle="--", linewidth=2, label=f"Expected = {alpha}")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("FWER", fontsize=11)
    ax.set_title("Family-Wise Error Rate (FWER)\nP(any FP | H0)", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, max(0.15, max(fwer_ci_highs) * 1.3))
    ax.grid(axis="y", alpha=0.3)

    # ==========================================================================
    # Plot 2: Edge-wise FPR by method (top-right)
    # ==========================================================================
    ax = axes[0, 1]

    fpr_vals = summary_df["mean_fpr_pos"].tolist()
    fpr_ci_lows = summary_df["fpr_ci_low"].tolist()
    fpr_ci_highs = summary_df["fpr_ci_high"].tolist()

    bar_colors = ["#4CAF50" if row["fpr_calibrated"] else "#F44336"
                  for _, row in summary_df.iterrows()]

    ax.bar(x, fpr_vals, color=bar_colors, alpha=0.7, edgecolor="black", linewidth=1.2)
    ax.errorbar(
        x, fpr_vals,
        yerr=[np.array(fpr_vals) - np.array(fpr_ci_lows),
              np.array(fpr_ci_highs) - np.array(fpr_vals)],
        fmt="none", color="black", capsize=5, capthick=1.5, linewidth=1.5,
    )
    ax.axhline(alpha, color="blue", linestyle="--", linewidth=2, label=f"Expected = {alpha}")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Edge-wise FPR", fontsize=11)
    ax.set_title("Edge-wise False Positive Rate\nE[n_FP / n_edges | H0]", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, max(0.15, max(fpr_ci_highs) * 1.3) if max(fpr_ci_highs) > 0 else 0.1)
    ax.grid(axis="y", alpha=0.3)

    # ==========================================================================
    # Plot 3: Distribution of minimum p-values (bottom-left)
    # ==========================================================================
    ax = axes[1, 0]

    for i, method in enumerate(methods):
        method_df = df[df["method"] == method]
        min_ps = method_df["min_p_pos"].values

        counts, bins = np.histogram(min_ps, bins=20, range=(0, 1))
        counts = counts / counts.sum() if counts.sum() > 0 else counts

        ax.step(bins[:-1], counts, where="post", label=method,
                color=colors_base[i], linewidth=2, alpha=0.8)

    ax.axvline(alpha, color="red", linestyle="--", linewidth=2, label=f"α = {alpha}")
    ax.set_xlabel("Minimum p-value", fontsize=11)
    ax.set_ylabel("Proportion", fontsize=11)
    ax.set_title("Distribution of Min P-values\nUnder Null Hypothesis", fontsize=12)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.3)

    # ==========================================================================
    # Plot 4: Calibration summary table (bottom-right)
    # ==========================================================================
    ax = axes[1, 1]
    ax.axis("off")

    table_data = []
    for _, row in summary_df.iterrows():
        fwer_status = "✓" if row["fwer_calibrated"] else "✗"
        fpr_status = "✓" if row["fpr_calibrated"] else "✗"
        table_data.append([
            row["method"],
            f"{row['fwer_pos']:.3f}",
            fwer_status,
            f"{row['mean_fpr_pos']:.4f}",
            fpr_status,
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=["Method", "FWER", "Cal.", "Edge FPR", "Cal."],
        loc="center",
        cellLoc="center",
        colWidths=[0.25, 0.18, 0.12, 0.18, 0.12],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    for i, (_, row) in enumerate(summary_df.iterrows()):
        fwer_color = "#C8E6C9" if row["fwer_calibrated"] else "#FFCDD2"
        fpr_color = "#C8E6C9" if row["fpr_calibrated"] else "#FFCDD2"
        table[(i + 1, 0)].set_facecolor("#FFFFFF")
        table[(i + 1, 1)].set_facecolor(fwer_color)
        table[(i + 1, 2)].set_facecolor(fwer_color)
        table[(i + 1, 3)].set_facecolor(fpr_color)
        table[(i + 1, 4)].set_facecolor(fpr_color)

    ax.set_title(f"Calibration Summary (α = {alpha})", fontsize=12, pad=20)

    fig.suptitle("FPR Calibration Results", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


# =============================================================================
# CLI params
# =============================================================================

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="FPR calibration test for TFNBS methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--config", type=Path,
        help="YAML configuration file",
    )
    parser.add_argument(
        "--n-null", type=int, default=None,
        help="Number of null datasets (overrides config)",
    )
    parser.add_argument(
        "--n-permutations", type=int, default=None,
        help="Permutations per test (overrides config)",
    )
    parser.add_argument(
        "--methods", nargs="+", default=None,
        help="Methods to test (overrides config)",
    )
    parser.add_argument(
        "--alpha", type=float, default=None,
        help="Significance threshold (overrides config)",
    )
    parser.add_argument(
        "--n-nodes", type=int, default=None,
        help="Number of nodes (overrides config)",
    )
    parser.add_argument(
        "--n-modules", type=int, default=None,
        help="Number of modules (overrides config)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=None,
        help="Samples per group (overrides config)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Base random seed (overrides config)",
    )
    parser.add_argument(
        "--use-mp", action="store_true",
        help="Use multiprocessing for permutations within each test",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=None,
        help="Number of parallel jobs for null runs (-1=all cores, 1=sequential)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint if available",
    )

    args = parser.parse_args(argv)

    # Load config from file or use defaults
    if args.config:
        config = FPRConfig.from_yaml(args.config)
    else:
        # Default configuration
        config = FPRConfig(
            methods=["tstat", "tfnbs", "ni_tfnbs", "fbc_tfnbs", "nbs_extent", "nbs_intensity", "cnbs"],
        )

    # Override config with CLI arguments
    if args.n_null is not None:
        config.n_null = args.n_null
    if args.n_permutations is not None:
        config.n_permutations = args.n_permutations
    if args.methods is not None:
        config.methods = args.methods
    if args.alpha is not None:
        config.alpha = args.alpha
    if args.n_nodes is not None:
        config.n_nodes = args.n_nodes
    if args.n_modules is not None:
        config.n_modules = args.n_modules
    if args.n_samples is not None:
        config.n_samples = args.n_samples
    if args.seed is not None:
        config.seed = args.seed
    if args.use_mp:
        config.use_mp = True
    if args.n_jobs is not None:
        config.n_jobs = args.n_jobs
    if args.output_dir is not None:
        config.output_dir = args.output_dir

    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Run calibration
    df = run_fpr_calibration(config, resume=args.resume)

    # Save results
    df.to_csv(config.output_dir / "fpr_results.csv", index=False)
    print(f"\nSaved results to {config.output_dir / 'fpr_results.csv'}")

    # Compute and save summary
    summary_df = compute_fpr_summary(df, alpha=config.alpha)
    summary_df.to_csv(config.output_dir / "fpr_summary.csv", index=False)

    # Print summary
    print("\n" + "=" * 70)
    print("FPR CALIBRATION SUMMARY")
    print("=" * 70)
    print(f"Alpha = {config.alpha}")
    print(f"N edges = {summary_df['n_edges'].iloc[0]}")
    print()

    # Header
    print(f"{'Method':<15} {'FWER':>8} {'95% CI':>20} {'Edge FPR':>10} {'Status':>12}")
    print("-" * 70)

    all_fwer_calibrated = True
    all_fpr_calibrated = True
    for _, row in summary_df.iterrows():
        fwer_ok = row["fwer_calibrated"]
        fpr_ok = row["fpr_calibrated"]
        if not fwer_ok:
            all_fwer_calibrated = False
        if not fpr_ok:
            all_fpr_calibrated = False

        status = "OK" if fwer_ok else "FWER FAIL"
        ci_str = f"[{row['fwer_ci_low']:.3f}, {row['fwer_ci_high']:.3f}]"
        print(f"  {row['method']:<13} {row['fwer_pos']:>8.4f} {ci_str:>20} "
              f"{row['mean_fpr_pos']:>10.5f} {status:>12}")

    # Save text summary
    with open(config.output_dir / "fpr_summary.txt", "w") as f:
        f.write("FPR Calibration Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write("Parameters:\n")
        f.write(f"  n_null_runs = {config.n_null}\n")
        f.write(f"  n_permutations = {config.n_permutations}\n")
        f.write(f"  alpha = {config.alpha}\n")
        f.write(f"  n_nodes = {config.n_nodes}\n")
        f.write(f"  n_edges = {summary_df['n_edges'].iloc[0]}\n")
        f.write(f"  n_samples = {config.n_samples}\n\n")

        f.write("Metrics Explained:\n")
        f.write("  FWER: Family-Wise Error Rate = P(any FP | H0)\n")
        f.write("        For FWER-controlling methods, this should = alpha\n")
        f.write("  Edge FPR: Mean proportion of FP edges = E[n_FP / n_edges | H0]\n")
        f.write("        For uncorrected tests, this should = alpha\n")
        f.write("        For FWER methods, this is << alpha\n\n")

        f.write(f"{'Method':<15} {'FWER':>8} {'95% CI':>22} {'Edge FPR':>12} {'FWER Cal.':>10}\n")
        f.write("-" * 70 + "\n")
        for _, row in summary_df.iterrows():
            status = "OK" if row["fwer_calibrated"] else "FAIL"
            ci_str = f"[{row['fwer_ci_low']:.4f}, {row['fwer_ci_high']:.4f}]"
            f.write(f"  {row['method']:<13} {row['fwer_pos']:>8.4f} {ci_str:>22} "
                    f"{row['mean_fpr_pos']:>12.6f} {status:>10}\n")
        f.write(f"\nOverall FWER: {'PASS' if all_fwer_calibrated else 'FAIL'}\n")

    # Create plot
    plot_fpr_results(
        df, summary_df, config.alpha,
        config.output_dir / "fpr_calibration_plot.png"
    )

    print()
    print("=" * 70)
    if all_fwer_calibrated:
        print("OVERALL RESULT: PASS - All methods have calibrated FWER")
    else:
        print("OVERALL RESULT: FAIL - Some methods have miscalibrated FWER")
    print("=" * 70)

    # Clean up checkpoint on success
    checkpoint_path = config.output_dir / "checkpoint.csv"
    if all_fwer_calibrated and checkpoint_path.exists():
        checkpoint_path.unlink()
        print("Removed checkpoint file (run completed successfully)")

    return 0 if all_fwer_calibrated else 1


if __name__ == "__main__":
    raise SystemExit(main())
