"""
Reproducibility Analysis for TFNBS Methods.

This script computes reproducibility metrics comparing results across:
1. Different experiments (IHB vs RMET)
2. Split-half stability within experiments
3. Bootstrap resampling stability

Metrics computed:
- Jaccard overlap of significant edge masks
- Dice coefficient
- T-map correlation (Pearson/Spearman)
- Significant edge overlap at multiple thresholds

Usage:
    # Compare IHB vs RMET
    python reproducibility_analysis.py --mode cross-experiment

    # Split-half stability
    python reproducibility_analysis.py --mode split-half --n-splits 100

    # All analyses
    python reproducibility_analysis.py --mode all

Output:
    - results/reproducibility/cross_experiment.csv
    - results/reproducibility/split_half_stability.csv
    - results/reproducibility/reproducibility_plots.png
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
from scipy import stats

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tfnbs.pairwise_stats import compute_p_val, compute_t_stat


# =============================================================================
# REPRODUCIBILITY METRICS
# =============================================================================

def jaccard_overlap(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute Jaccard index between two binary masks.

    J = |A ∩ B| / |A ∪ B|
    """
    intersection = np.sum(mask1 & mask2)
    union = np.sum(mask1 | mask2)
    return intersection / union if union > 0 else 0.0


def dice_overlap(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute Dice coefficient between two binary masks.

    D = 2|A ∩ B| / (|A| + |B|)
    """
    intersection = np.sum(mask1 & mask2)
    total = np.sum(mask1) + np.sum(mask2)
    return 2 * intersection / total if total > 0 else 0.0


def tmap_correlation(
    tmap1: np.ndarray,
    tmap2: np.ndarray,
    method: str = "spearman",
) -> float:
    """
    Compute correlation between two t-statistic maps.

    Parameters
    ----------
    tmap1, tmap2 : (n_nodes, n_nodes) arrays of t-statistics
    method : "pearson" or "spearman"
    """
    # Get upper triangle
    triu_idx = np.triu_indices(tmap1.shape[0], k=1)
    vec1 = tmap1[triu_idx]
    vec2 = tmap2[triu_idx]

    if method == "pearson":
        r, _ = stats.pearsonr(vec1, vec2)
    else:
        r, _ = stats.spearmanr(vec1, vec2)

    return float(r)


def overlap_at_threshold(
    pvals1: np.ndarray,
    pvals2: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[float, float, int, int]:
    """
    Compute overlap metrics at a specific threshold.

    Returns: (jaccard, dice, n_sig1, n_sig2)
    """
    mask1 = pvals1 < alpha
    mask2 = pvals2 < alpha

    # Ensure symmetric and zero diagonal
    mask1 = mask1 | mask1.T
    mask2 = mask2 | mask2.T
    np.fill_diagonal(mask1, False)
    np.fill_diagonal(mask2, False)

    # Upper triangle only
    triu_idx = np.triu_indices(mask1.shape[0], k=1)
    mask1_upper = mask1[triu_idx]
    mask2_upper = mask2[triu_idx]

    return (
        jaccard_overlap(mask1_upper, mask2_upper),
        dice_overlap(mask1_upper, mask2_upper),
        int(np.sum(mask1_upper)),
        int(np.sum(mask2_upper)),
    )


# =============================================================================
# CROSS-EXPERIMENT REPRODUCIBILITY
# =============================================================================

@dataclass
class CrossExperimentResult:
    """Results from cross-experiment comparison."""

    method: str
    alpha: float
    jaccard_pos: float  # g2>g1 direction
    jaccard_neg: float  # g1>g2 direction
    dice_pos: float
    dice_neg: float
    tmap_correlation: float
    n_sig_exp1_pos: int
    n_sig_exp1_neg: int
    n_sig_exp2_pos: int
    n_sig_exp2_neg: int


def run_cross_experiment_comparison(
    open_exp1: np.ndarray,
    close_exp1: np.ndarray,
    open_exp2: np.ndarray,
    close_exp2: np.ndarray,
    methods: List[str],
    net_labels: np.ndarray = None,
    alphas: List[float] = None,
    n_permutations: int = 500,
    use_mp: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compare reproducibility across two experiments (e.g., IHB vs RMET).

    Parameters
    ----------
    open_exp1, close_exp1 : Experiment 1 data (n_subjects, n_nodes, n_nodes)
    open_exp2, close_exp2 : Experiment 2 data
    methods : List of methods to compare
    net_labels : Network labels (required for cnbs, ni_tfnbs, fbc_tfnbs)
    alphas : List of significance thresholds
    """
    if alphas is None:
        alphas = [0.05, 0.01, 0.001]

    results = []

    # Compute t-maps for correlation
    t_exp1 = compute_t_stat(open_exp1, close_exp1, test_type="paired")
    t_exp2 = compute_t_stat(open_exp2, close_exp2, test_type="paired")

    signed_t_exp1 = t_exp1["g2>g1"] - t_exp1["g1>g2"]
    signed_t_exp2 = t_exp2["g2>g1"] - t_exp2["g1>g2"]

    tmap_r = tmap_correlation(signed_t_exp1, signed_t_exp2, method="spearman")

    if verbose:
        print(f"T-map Spearman correlation: {tmap_r:.3f}")

    for method in methods:
        if verbose:
            print(f"\nMethod: {method}")

        # Prepare kwargs
        kwargs = {}
        if method in ["cnbs", "ni_tfnbs", "fbc_tfnbs"]:
            if net_labels is None:
                print(f"  Skipping {method} - net_labels required")
                continue
            kwargs["net_labels"] = net_labels

        # Compute p-values for both experiments
        try:
            pvals_exp1 = compute_p_val(
                open_exp1, close_exp1,
                test_type="paired",
                method=method,
                n_permutations=n_permutations,
                use_mp=use_mp,
                **kwargs,
            )
            pvals_exp2 = compute_p_val(
                open_exp2, close_exp2,
                test_type="paired",
                method=method,
                n_permutations=n_permutations,
                use_mp=use_mp,
                **kwargs,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        for alpha in alphas:
            # Positive direction (close > open)
            jac_pos, dice_pos, n1_pos, n2_pos = overlap_at_threshold(
                pvals_exp1["g2>g1"], pvals_exp2["g2>g1"], alpha
            )

            # Negative direction (open > close)
            jac_neg, dice_neg, n1_neg, n2_neg = overlap_at_threshold(
                pvals_exp1["g1>g2"], pvals_exp2["g1>g2"], alpha
            )

            if verbose:
                print(f"  alpha={alpha}: Jaccard(pos)={jac_pos:.3f}, Dice(pos)={dice_pos:.3f}")

            results.append(CrossExperimentResult(
                method=method,
                alpha=alpha,
                jaccard_pos=jac_pos,
                jaccard_neg=jac_neg,
                dice_pos=dice_pos,
                dice_neg=dice_neg,
                tmap_correlation=tmap_r,
                n_sig_exp1_pos=n1_pos,
                n_sig_exp1_neg=n1_neg,
                n_sig_exp2_pos=n2_pos,
                n_sig_exp2_neg=n2_neg,
            ))

    return pd.DataFrame([vars(r) for r in results])


# =============================================================================
# SPLIT-HALF STABILITY
# =============================================================================

@dataclass
class SplitHalfResult:
    """Results from split-half stability analysis."""

    method: str
    split_id: int
    alpha: float
    jaccard_pos: float
    jaccard_neg: float
    dice_pos: float
    dice_neg: float
    tmap_correlation: float


def run_split_half_stability(
    open_data: np.ndarray,
    close_data: np.ndarray,
    methods: List[str],
    net_labels: np.ndarray = None,
    n_splits: int = 100,
    alpha: float = 0.05,
    n_permutations: int = 200,
    use_mp: bool = False,
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Assess split-half stability by randomly splitting subjects.

    For each split:
    1. Randomly divide subjects into two halves
    2. Compute p-values for each half
    3. Compare overlap between halves
    """
    n_subjects = open_data.shape[0]
    rng = np.random.default_rng(seed)

    results = []

    if verbose:
        print(f"=== Split-Half Stability ===")
        print(f"Subjects: {n_subjects}")
        print(f"Splits: {n_splits}")
        print()

    for split_id in range(n_splits):
        # Random permutation of subjects
        perm = rng.permutation(n_subjects)
        half = n_subjects // 2

        idx1 = perm[:half]
        idx2 = perm[half:2*half]

        # Split data
        open1, close1 = open_data[idx1], close_data[idx1]
        open2, close2 = open_data[idx2], close_data[idx2]

        # Compute t-maps
        t1 = compute_t_stat(open1, close1, test_type="paired")
        t2 = compute_t_stat(open2, close2, test_type="paired")
        signed_t1 = t1["g2>g1"] - t1["g1>g2"]
        signed_t2 = t2["g2>g1"] - t2["g1>g2"]
        tmap_r = tmap_correlation(signed_t1, signed_t2, method="spearman")

        for method in methods:
            kwargs = {}
            if method in ["cnbs", "ni_tfnbs", "fbc_tfnbs"]:
                if net_labels is None:
                    continue
                kwargs["net_labels"] = net_labels

            try:
                pvals1 = compute_p_val(
                    open1, close1,
                    test_type="paired",
                    method=method,
                    n_permutations=n_permutations,
                    use_mp=use_mp,
                    random_state=seed + split_id,
                    **kwargs,
                )
                pvals2 = compute_p_val(
                    open2, close2,
                    test_type="paired",
                    method=method,
                    n_permutations=n_permutations,
                    use_mp=use_mp,
                    random_state=seed + split_id + 1000,
                    **kwargs,
                )
            except Exception as e:
                if verbose and split_id == 0:
                    print(f"  {method}: ERROR - {e}")
                continue

            jac_pos, dice_pos, _, _ = overlap_at_threshold(
                pvals1["g2>g1"], pvals2["g2>g1"], alpha
            )
            jac_neg, dice_neg, _, _ = overlap_at_threshold(
                pvals1["g1>g2"], pvals2["g1>g2"], alpha
            )

            results.append(SplitHalfResult(
                method=method,
                split_id=split_id,
                alpha=alpha,
                jaccard_pos=jac_pos,
                jaccard_neg=jac_neg,
                dice_pos=dice_pos,
                dice_neg=dice_neg,
                tmap_correlation=tmap_r,
            ))

        if verbose and (split_id + 1) % 10 == 0:
            print(f"  Completed {split_id + 1}/{n_splits} splits")

    return pd.DataFrame([vars(r) for r in results])


# =============================================================================
# SUMMARY AND VISUALIZATION
# =============================================================================

def summarize_split_half(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize split-half results by method."""

    summary = df.groupby("method").agg({
        "jaccard_pos": ["mean", "std"],
        "jaccard_neg": ["mean", "std"],
        "dice_pos": ["mean", "std"],
        "dice_neg": ["mean", "std"],
        "tmap_correlation": ["mean", "std"],
    }).reset_index()

    # Flatten columns
    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]

    return summary


def plot_reproducibility(
    cross_exp_df: pd.DataFrame,
    split_half_df: pd.DataFrame,
    output_path: Path,
):
    """Create reproducibility visualization."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not available, skipping plots")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Cross-experiment Jaccard by method
    if cross_exp_df is not None and len(cross_exp_df) > 0:
        ax = axes[0]
        cross_05 = cross_exp_df[cross_exp_df["alpha"] == 0.05]
        if len(cross_05) > 0:
            methods = cross_05["method"].unique()
            x = np.arange(len(methods))
            width = 0.35

            jac_pos = [cross_05[cross_05["method"] == m]["jaccard_pos"].values[0] for m in methods]
            jac_neg = [cross_05[cross_05["method"] == m]["jaccard_neg"].values[0] for m in methods]

            ax.bar(x - width/2, jac_pos, width, label="close > open", color="red", alpha=0.7)
            ax.bar(x + width/2, jac_neg, width, label="open > close", color="blue", alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=45, ha="right")
            ax.set_ylabel("Jaccard Index")
            ax.set_title("Cross-Experiment Reproducibility (alpha=0.05)")
            ax.legend()
            ax.set_ylim(0, 1)

    # Panel 2: Split-half stability distribution
    if split_half_df is not None and len(split_half_df) > 0:
        ax = axes[1]
        sns.boxplot(data=split_half_df, x="method", y="jaccard_pos", ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_ylabel("Jaccard Index (close > open)")
        ax.set_title("Split-Half Stability Distribution")
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reproducibility analysis for TFNBS methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--mode", choices=["cross-experiment", "split-half", "all"],
        default="all", help="Analysis mode",
    )
    parser.add_argument(
        "--methods", nargs="+",
        default=["tstat", "tfnbs", "ni_tfnbs", "fbc_tfnbs", "nbs", "cnbs"],
        help="Methods to analyze",
    )
    parser.add_argument(
        "--n-splits", type=int, default=100,
        help="Number of split-half iterations",
    )
    parser.add_argument(
        "--n-permutations", type=int, default=200,
        help="Permutations per test",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Significance threshold",
    )
    parser.add_argument(
        "--use-mp", action="store_true",
        help="Use multiprocessing",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).parent.parent / "results" / "reproducibility",
        help="Output directory",
    )

    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Try to load OpenClose data
    try:
        from ml_transfer.openclose_loader import OpenCloseDataset
        ds_ihb = OpenCloseDataset.hcp("ihb")
        ds_rmet = OpenCloseDataset.hcp("rmet")
        open_ihb, close_ihb = ds_ihb.fisher_z()
        open_rmet, close_rmet = ds_rmet.fisher_z()
        net_labels = ds_ihb.get_net_labels("cole")
        has_data = True
    except Exception as e:
        print(f"Could not load OpenClose data: {e}")
        print("Using synthetic data for demonstration")
        has_data = False

        # Generate synthetic data
        n_subjects, n_nodes = 30, 50
        rng = np.random.default_rng(42)
        open_ihb = rng.standard_normal((n_subjects, n_nodes, n_nodes))
        close_ihb = open_ihb + rng.standard_normal((n_subjects, n_nodes, n_nodes)) * 0.1
        close_ihb[:, :10, :10] += 0.3
        open_rmet = rng.standard_normal((n_subjects, n_nodes, n_nodes))
        close_rmet = open_rmet + rng.standard_normal((n_subjects, n_nodes, n_nodes)) * 0.1
        close_rmet[:, :10, :10] += 0.3
        net_labels = np.repeat(np.arange(5), 10)

        # Symmetrize
        for data in [open_ihb, close_ihb, open_rmet, close_rmet]:
            for i in range(data.shape[0]):
                data[i] = (data[i] + data[i].T) / 2
                np.fill_diagonal(data[i], 0)

    cross_exp_df = None
    split_half_df = None

    # Cross-experiment analysis
    if args.mode in ["cross-experiment", "all"]:
        print("\n=== Cross-Experiment Reproducibility ===")
        cross_exp_df = run_cross_experiment_comparison(
            open_ihb, close_ihb,
            open_rmet, close_rmet,
            methods=args.methods,
            net_labels=net_labels,
            n_permutations=args.n_permutations,
            use_mp=args.use_mp,
        )
        cross_exp_df.to_csv(args.output_dir / "cross_experiment.csv", index=False)

    # Split-half stability
    if args.mode in ["split-half", "all"]:
        print("\n=== Split-Half Stability ===")
        split_half_df = run_split_half_stability(
            open_ihb, close_ihb,
            methods=args.methods,
            net_labels=net_labels,
            n_splits=args.n_splits,
            alpha=args.alpha,
            n_permutations=args.n_permutations,
            use_mp=args.use_mp,
            seed=args.seed,
        )
        split_half_df.to_csv(args.output_dir / "split_half_stability.csv", index=False)

        # Summary
        summary = summarize_split_half(split_half_df)
        summary.to_csv(args.output_dir / "split_half_summary.csv", index=False)
        print("\nSplit-Half Summary:")
        print(summary.to_string(index=False))

    # Visualization
    plot_reproducibility(
        cross_exp_df, split_half_df,
        args.output_dir / "reproducibility_plots.png"
    )

    print(f"\nResults saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
