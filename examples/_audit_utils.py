"""Shared agreement-metric helpers for the validation audit scripts.

The four audit scripts under ``abide_validation/`` and
``openclose_validation/`` each used to re-implement the same
agreement quartet with minor formatting variations. This module collects
the canonical implementations so the audits stay consistent.

Functions
---------
upper_triangle
    1D upper-triangle (``k=1``) extract for a symmetric matrix.
jaccard
    Bare Jaccard on two boolean masks.
jaccard_random_baseline
    Expected Jaccard under independence at matched marginal counts.
fisher_exact_2x2
    One-sided greater Fisher's exact on the 2 × 2 mask contingency.
spearman_neglog10
    Spearman rank correlation on ``−log10(p + eps)``; robust to ties
    and to a non-significant tail of identical empirical p-values.
topk_concordance
    Fraction of the top-``k`` strongest edges shared between two
    p-value vectors (rank-based).
block_mass
    K × K block-mass matrix summing ``−log10 p`` over edges binned by
    pairs of network labels.
block_mass_correlation
    Pearson on the upper-triangle of two block-mass matrices.
agreement_quartet
    The full §2.5 bundle: Jaccard (observed + random baseline +
    Fisher exact), Spearman on ``−log10 p``, top-{10, 50, 100, 500}
    concordance, and block-mass Pearson. Returns a single dict.

These helpers operate on the full symmetric matrices (``(N, N)``) by
convention; the upper-triangle extraction happens internally so
callers don't need to remember to ``triu_indices`` first.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import numpy.typing as npt
from scipy import stats


_EPS = 1e-12
_TOPK_DEFAULTS = (10, 50, 100, 500)


def upper_triangle(arr: npt.NDArray) -> npt.NDArray:
    """Return the strict upper triangle (k=1) of a square matrix as a 1D vector."""
    return arr[np.triu_indices(arr.shape[0], k=1)]


def jaccard(mask_a: npt.NDArray[np.bool_], mask_b: npt.NDArray[np.bool_]) -> float:
    """Jaccard coefficient |A ∩ B| / |A ∪ B|; 0 when both empty."""
    inter = int(np.logical_and(mask_a, mask_b).sum())
    union = int(np.logical_or(mask_a, mask_b).sum())
    return float(inter / union) if union > 0 else 0.0


def jaccard_random_baseline(k_a: int, k_b: int, n_edges: int) -> float:
    """Expected Jaccard under independence at matched marginal counts.

    For ``k_a`` significant edges in A and ``k_b`` in B drawn at random
    from ``n_edges`` total, the expected intersection is ``k_a · k_b /
    n_edges`` and the expected Jaccard is
    ``E[|A∩B|] / E[|A∪B|]``.
    """
    if k_a + k_b == 0:
        return 0.0
    exp_inter = k_a * k_b / n_edges
    exp_union = k_a + k_b - exp_inter
    return float(exp_inter / exp_union) if exp_union > 0 else 0.0


def fisher_exact_2x2(
    sig_a: npt.NDArray[np.bool_],
    sig_b: npt.NDArray[np.bool_],
    *,
    alternative: str = "greater",
) -> float:
    """One-sided Fisher's exact on the 2 × 2 mask co-occurrence table."""
    tp = int(np.logical_and(sig_a, sig_b).sum())
    fp = int(np.logical_and(sig_a, ~sig_b).sum())
    fn = int(np.logical_and(~sig_a, sig_b).sum())
    tn = int(np.logical_and(~sig_a, ~sig_b).sum())
    _, p = stats.fisher_exact([[tp, fp], [fn, tn]], alternative=alternative)
    return float(p)


def spearman_neglog10(
    p_a: npt.NDArray[np.float64], p_b: npt.NDArray[np.float64]
) -> float:
    """Spearman rank correlation on ``−log10(p + eps)``.

    Operates on 1D vectors. Use :func:`upper_triangle` first if the
    inputs are full ``(N, N)`` matrices.
    """
    r, _ = stats.spearmanr(-np.log10(p_a + _EPS), -np.log10(p_b + _EPS))
    return float(r)


def topk_concordance(
    p_a: npt.NDArray[np.float64], p_b: npt.NDArray[np.float64], k: int
) -> float:
    """Fraction of the top-``k`` smallest-p edges shared between vectors.

    Returns ``nan`` when ``k`` is out of range. 1D inputs.
    """
    if k <= 0 or k > p_a.size:
        return float("nan")
    ranks_a = np.argsort(p_a)[:k]
    ranks_b = np.argsort(p_b)[:k]
    return float(len(np.intersect1d(ranks_a, ranks_b)) / k)


def block_mass(
    p_full: npt.NDArray[np.float64], net_labels: npt.NDArray[np.int_]
) -> npt.NDArray[np.float64]:
    """K × K block-mass matrix summing ``−log10 p`` over edges binned by
    pairs of network labels.

    Symmetric: ``M[i, j] == M[j, i]``. Diagonal entries are within-network
    sums of ``−log10 p`` over the upper triangle of the corresponding
    block.
    """
    N = p_full.shape[0]
    K = int(net_labels.max() + 1)
    neglog = -np.log10(p_full + _EPS)
    M = np.zeros((K, K), dtype=np.float64)
    iu, ju = np.triu_indices(N, k=1)
    bi = net_labels[iu]
    bj = net_labels[ju]
    lo = np.minimum(bi, bj)
    hi = np.maximum(bi, bj)
    np.add.at(M, (lo, hi), neglog[iu, ju])
    # Mirror to symmetric form (off-diagonal only).
    M = M + np.triu(M, 1).T
    return M


def block_mass_correlation(
    p_a_full: npt.NDArray[np.float64],
    p_b_full: npt.NDArray[np.float64],
    net_labels: npt.NDArray[np.int_],
) -> float:
    """Pearson on the upper-triangle of two block-mass matrices."""
    M_a = block_mass(p_a_full, net_labels)
    M_b = block_mass(p_b_full, net_labels)
    tri = np.triu_indices_from(M_a)
    r, _ = stats.pearsonr(M_a[tri], M_b[tri])
    return float(r)


def agreement_quartet(
    p_a_full: npt.NDArray[np.float64],
    p_b_full: npt.NDArray[np.float64],
    net_labels: npt.NDArray[np.int_],
    *,
    alpha: float = 0.05,
    topk_values: tuple = _TOPK_DEFAULTS,
) -> Dict[str, float]:
    """The §2.5 agreement bundle on two full ``(N, N)`` p-value maps.

    Returns
    -------
    dict
        Keys: ``jaccard``, ``jaccard_random``, ``fisher_exact_p``,
        ``spearman``, ``top10``/``top50``/``top100``/``top500``
        (per the ``topk_values`` argument), ``block_corr``,
        ``ka``, ``kb`` (per-tail significant-edge counts under
        ``alpha``).
    """
    p_a = upper_triangle(p_a_full)
    p_b = upper_triangle(p_b_full)
    sig_a = p_a < alpha
    sig_b = p_b < alpha
    n_edges = p_a.size
    k_a = int(sig_a.sum())
    k_b = int(sig_b.sum())

    out: Dict[str, float] = {
        "jaccard": jaccard(sig_a, sig_b),
        "jaccard_random": jaccard_random_baseline(k_a, k_b, n_edges),
        "fisher_exact_p": fisher_exact_2x2(sig_a, sig_b),
        "spearman": spearman_neglog10(p_a, p_b),
        "block_corr": block_mass_correlation(
            p_a_full, p_b_full, net_labels
        ),
        "ka": k_a,
        "kb": k_b,
    }
    for k in topk_values:
        out[f"top{k}"] = topk_concordance(p_a, p_b, k)
    return out


__all__ = [
    "upper_triangle",
    "jaccard",
    "jaccard_random_baseline",
    "fisher_exact_2x2",
    "spearman_neglog10",
    "topk_concordance",
    "block_mass",
    "block_mass_correlation",
    "agreement_quartet",
]
