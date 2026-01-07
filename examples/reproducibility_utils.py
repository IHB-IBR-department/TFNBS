"""
Reproducibility metrics for comparing significance masks and effect maps.

Provides functions for:
- Edge mask overlap (Jaccard, Dice)
- T-map correlation (Spearman, Pearson)
- Split-half stability analysis
- Block-level reproducibility

Usage:
    from reproducibility_utils import jaccard_overlap, tmap_correlation, split_half_stability
"""

from __future__ import annotations

from typing import Callable, Dict, List, Literal, Optional

import numpy as np
from scipy.stats import pearsonr, spearmanr


def jaccard_overlap(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute Jaccard index between two binary masks.

    Jaccard = |A ∩ B| / |A ∪ B|

    Parameters
    ----------
    mask1, mask2 : (n_nodes, n_nodes) boolean arrays

    Returns
    -------
    float
        Jaccard index in [0, 1]
    """
    tri = np.triu_indices(mask1.shape[0], k=1)
    m1, m2 = mask1[tri].astype(bool), mask2[tri].astype(bool)

    intersection = np.sum(m1 & m2)
    union = np.sum(m1 | m2)

    return float(intersection / union) if union > 0 else 0.0


def dice_overlap(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute Dice coefficient between two binary masks.

    Dice = 2|A ∩ B| / (|A| + |B|)

    Parameters
    ----------
    mask1, mask2 : (n_nodes, n_nodes) boolean arrays

    Returns
    -------
    float
        Dice coefficient in [0, 1]
    """
    tri = np.triu_indices(mask1.shape[0], k=1)
    m1, m2 = mask1[tri].astype(bool), mask2[tri].astype(bool)

    intersection = np.sum(m1 & m2)
    total = np.sum(m1) + np.sum(m2)

    return float(2 * intersection / total) if total > 0 else 0.0


def tmap_correlation(
    t1: np.ndarray,
    t2: np.ndarray,
    method: Literal["spearman", "pearson"] = "spearman",
) -> float:
    """
    Compute correlation between two t-statistic maps.

    Parameters
    ----------
    t1, t2 : (n_nodes, n_nodes) arrays
        Signed t-statistic maps
    method : str
        "spearman" or "pearson"

    Returns
    -------
    float
        Correlation coefficient
    """
    tri = np.triu_indices(t1.shape[0], k=1)
    v1, v2 = t1[tri].ravel(), t2[tri].ravel()

    if method == "spearman":
        return float(spearmanr(v1, v2)[0])
    return float(pearsonr(v1, v2)[0])


def signed_overlap(
    mask1: np.ndarray,
    mask2: np.ndarray,
    sign1: np.ndarray,
    sign2: np.ndarray,
) -> float:
    """
    Compute directional agreement among union of discovered edges.

    Parameters
    ----------
    mask1, mask2 : (n_nodes, n_nodes) boolean arrays
        Significance masks
    sign1, sign2 : (n_nodes, n_nodes) arrays
        Signed effect maps (e.g., t-statistics)

    Returns
    -------
    float
        Proportion of edges with same sign direction in [0, 1]
    """
    tri = np.triu_indices(mask1.shape[0], k=1)
    union = (mask1 | mask2)[tri]

    if not union.any():
        return 0.0

    s1 = np.sign(sign1[tri][union])
    s2 = np.sign(sign2[tri][union])

    return float(np.mean(s1 == s2))


def split_half_stability(
    data_open: np.ndarray,
    data_close: np.ndarray,
    compute_mask_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_splits: int = 50,
    seed: int = 42,
) -> Dict[str, float | List[float]]:
    """
    Compute split-half stability of significance masks.

    Parameters
    ----------
    data_open : (n_subjects, n_nodes, n_nodes)
    data_close : (n_subjects, n_nodes, n_nodes)
    compute_mask_fn : callable
        Function (open, close) -> binary mask
    n_splits : int
        Number of random splits
    seed : int
        Random seed

    Returns
    -------
    dict
        {"mean": float, "std": float, "values": list}
    """
    rng = np.random.default_rng(seed)
    n = data_open.shape[0]
    jaccards = []

    for _ in range(n_splits):
        idx = rng.permutation(n)
        half = n // 2
        idx1, idx2 = idx[:half], idx[half : 2 * half]

        mask1 = compute_mask_fn(data_open[idx1], data_close[idx1])
        mask2 = compute_mask_fn(data_open[idx2], data_close[idx2])

        jaccards.append(jaccard_overlap(mask1, mask2))

    return {
        "mean": float(np.mean(jaccards)),
        "std": float(np.std(jaccards)),
        "values": jaccards,
    }


def bootstrap_stability(
    data_open: np.ndarray,
    data_close: np.ndarray,
    compute_mask_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_bootstrap: int = 100,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Compute bootstrap selection frequency for each edge.

    Parameters
    ----------
    data_open : (n_subjects, n_nodes, n_nodes)
    data_close : (n_subjects, n_nodes, n_nodes)
    compute_mask_fn : callable
        Function (open, close) -> binary mask
    n_bootstrap : int
        Number of bootstrap samples
    seed : int
        Random seed

    Returns
    -------
    dict
        {"selection_freq": (n_nodes, n_nodes) array,
         "mean_selected": float}
    """
    rng = np.random.default_rng(seed)
    n = data_open.shape[0]
    n_nodes = data_open.shape[1]

    selection_count = np.zeros((n_nodes, n_nodes), dtype=np.float64)

    for _ in range(n_bootstrap):
        # Sample with replacement, keeping pairs together
        idx = rng.choice(n, size=n, replace=True)

        mask = compute_mask_fn(data_open[idx], data_close[idx])
        selection_count += mask.astype(float)

    selection_freq = selection_count / n_bootstrap

    tri = np.triu_indices(n_nodes, k=1)
    mean_selected = float(np.mean(selection_count[tri] > 0))

    return {
        "selection_freq": selection_freq,
        "mean_selected": mean_selected,
    }


def compare_experiments(
    mask_ihb: np.ndarray,
    mask_rmet: np.ndarray,
    t_ihb: Optional[np.ndarray] = None,
    t_rmet: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute all reproducibility metrics between two experiments.

    Parameters
    ----------
    mask_ihb, mask_rmet : (n_nodes, n_nodes) boolean arrays
    t_ihb, t_rmet : (n_nodes, n_nodes) arrays, optional
        Signed t-statistics for correlation and signed overlap

    Returns
    -------
    dict
        All reproducibility metrics
    """
    results = {
        "jaccard": jaccard_overlap(mask_ihb, mask_rmet),
        "dice": dice_overlap(mask_ihb, mask_rmet),
        "n_sig_ihb": int(np.sum(mask_ihb) // 2),  # Upper tri only
        "n_sig_rmet": int(np.sum(mask_rmet) // 2),
    }

    if t_ihb is not None and t_rmet is not None:
        results["tmap_spearman"] = tmap_correlation(t_ihb, t_rmet, "spearman")
        results["tmap_pearson"] = tmap_correlation(t_ihb, t_rmet, "pearson")
        results["signed_overlap"] = signed_overlap(mask_ihb, mask_rmet, t_ihb, t_rmet)

    return results


# Interpretation guidelines
JACCARD_INTERPRETATION = {
    (0.6, 1.0): "High reproducibility",
    (0.3, 0.6): "Moderate reproducibility",
    (0.0, 0.3): "Low reproducibility",
}

TMAP_INTERPRETATION = {
    (0.7, 1.0): "Strong pattern similarity",
    (0.4, 0.7): "Moderate similarity",
    (0.0, 0.4): "Weak similarity",
}


def interpret_jaccard(value: float) -> str:
    """Interpret Jaccard value."""
    for (low, high), label in JACCARD_INTERPRETATION.items():
        if low <= value < high:
            return label
    return "High reproducibility" if value >= 0.6 else "Unknown"


def interpret_tmap_r(value: float) -> str:
    """Interpret t-map correlation value."""
    value = abs(value)
    for (low, high), label in TMAP_INTERPRETATION.items():
        if low <= value < high:
            return label
    return "Strong pattern similarity" if value >= 0.7 else "Unknown"


if __name__ == "__main__":
    # Quick test with random data
    n_nodes = 50
    rng = np.random.default_rng(42)

    mask1 = rng.random((n_nodes, n_nodes)) > 0.9
    mask1 = mask1 | mask1.T  # Symmetrize

    mask2 = rng.random((n_nodes, n_nodes)) > 0.9
    mask2 = mask2 | mask2.T

    t1 = rng.standard_normal((n_nodes, n_nodes))
    t1 = (t1 + t1.T) / 2

    t2 = t1 + rng.standard_normal((n_nodes, n_nodes)) * 0.5
    t2 = (t2 + t2.T) / 2

    results = compare_experiments(mask1, mask2, t1, t2)
    print("Test results:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
