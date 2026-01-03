"""
Network-Based Statistics (NBS) scoring module.

Provides classical NBS cluster scoring (extent or intensity) and constrained
cNBS scoring in a single module, mirroring the role of tfnbs_score.py.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


__all__ = [
    "DEFAULT_NBS_THRESHOLD",
    "DEFAULT_NBS_STAT",
    "get_nbs_score",
    "get_cnbs_score",
    "nbs_bct",
]


# =============================================================================
# Constants
# =============================================================================

DEFAULT_NBS_THRESHOLD: float = 2.0
DEFAULT_NBS_STAT: str = "extent"


# =============================================================================
# Helper utilities
# =============================================================================

ArrayF = npt.NDArray[np.float64]


def _is_symmetric(matrix: npt.NDArray[np.floating], rtol: float = 1e-10) -> bool:
    return np.allclose(matrix, matrix.T, rtol=rtol, atol=0)


def _get_edges(
    t_stats: npt.NDArray[np.floating],
    min_threshold: float,
    symmetric: bool,
) -> Tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], npt.NDArray[np.floating]]:
    if symmetric:
        rows, cols = np.triu_indices_from(t_stats, k=1)
    else:
        nroi = t_stats.shape[0]
        rows, cols = np.where(~np.eye(nroi, dtype=bool))

    weights = t_stats[rows, cols]
    valid_mask = weights > min_threshold
    return rows[valid_mask], cols[valid_mask], weights[valid_mask]


def _check_square(t_stat: npt.NDArray[np.floating]) -> None:
    if t_stat.ndim != 2 or t_stat.shape[0] != t_stat.shape[1]:
        raise ValueError("t_stat must be a square (N, N) matrix.")
    if not np.allclose(np.diag(t_stat), 0.0):
        raise ValueError("Diagonal elements must be zero (no self-connections).")


def _validate_net_labels(
    net_labels: npt.NDArray[np.integer],
    n_nodes: int,
) -> npt.NDArray[np.integer]:
    if net_labels.shape[0] != n_nodes:
        raise ValueError(
            f"net_labels shape {net_labels.shape} does not match number of nodes {n_nodes}."
        )
    _, inverse = np.unique(net_labels, return_inverse=True)
    return inverse


def _compute_canonical_block_ids(
    edge_rows: npt.NDArray[np.intp],
    edge_cols: npt.NDArray[np.intp],
    node_labels: npt.NDArray[np.integer],
    n_networks: int,
) -> npt.NDArray[np.intp]:
    labels_i = node_labels[edge_rows]
    labels_j = node_labels[edge_cols]
    min_labels = np.minimum(labels_i, labels_j)
    max_labels = np.maximum(labels_i, labels_j)
    return min_labels * n_networks + max_labels


# =============================================================================
# NBS scoring
# =============================================================================

def get_nbs_score(
    t_stats: npt.NDArray[np.floating],
    threshold: float = DEFAULT_NBS_THRESHOLD,
    stat_type: str = DEFAULT_NBS_STAT,
) -> npt.NDArray[np.floating]:
    """
    Compute NBS cluster statistics for a t-statistic matrix.

    Parameters
    ----------
    t_stats : ndarray of shape (N, N)
        Non-negative t-statistic matrix (one tail, already separated).
    threshold : float
        T-statistic threshold for edge inclusion.
    stat_type : {'extent', 'intensity'}
        - 'extent': cluster size (# edges).
        - 'intensity': sum of t-values within the cluster.

    Returns
    -------
    ndarray of shape (N, N)
        Matrix where each suprathreshold edge is assigned its cluster statistic.
        Non-suprathreshold edges are 0.
    """
    if stat_type not in ("extent", "intensity"):
        raise ValueError("stat_type must be 'extent' or 'intensity'.")
    _check_square(t_stats)

    # Round to avoid float precision issues at threshold boundaries
    t_stats = np.round(t_stats, decimals=10)

    nroi = t_stats.shape[0]
    scores = np.zeros((nroi, nroi), dtype=np.float64)

    is_symm = _is_symmetric(t_stats)
    edge_rows, edge_cols, edge_weights = _get_edges(t_stats, threshold, symmetric=is_symm)
    if edge_rows.size == 0:
        return scores

    mask = np.zeros((nroi, nroi), dtype=bool)
    mask[edge_rows, edge_cols] = True
    if is_symm:
        mask[edge_cols, edge_rows] = True

    mask_for_cc = mask | mask.T
    sparse_mat = csr_matrix(mask_for_cc)
    n_components, node_labels = connected_components(sparse_mat, directed=False)

    edge_component_ids = node_labels[edge_rows]

    if stat_type == "extent":
        component_stats = np.bincount(edge_component_ids, minlength=n_components).astype(np.float64)
    else:
        component_stats = np.bincount(
            edge_component_ids,
            weights=edge_weights.astype(np.float64),
            minlength=n_components,
        )

    edge_stats = component_stats[edge_component_ids]
    scores[edge_rows, edge_cols] = edge_stats
    if is_symm:
        scores[edge_cols, edge_rows] = edge_stats

    return scores


# =============================================================================
# cNBS scoring
# =============================================================================

def get_cnbs_score(
    t_stat: npt.NDArray[np.float64],
    net_labels: npt.NDArray[np.int_],
) -> npt.NDArray[np.float64]:
    """
    Compute cNBS score: mean t-stat per subnetwork.

    Parameters
    ----------
    t_stat : ndarray of shape (N, N)
        Non-negative t-statistic matrix (one tail, already separated).
    net_labels : ndarray of shape (N,)
        Network label for each node (0..K-1).

    Returns
    -------
    ndarray of shape (N, N)
        Matrix where each edge has its subnetwork's mean t-stat.
        Edges below threshold (t_stat=0) get score=0.
    """
    _check_square(t_stat)
    if net_labels.ndim != 1 or net_labels.shape[0] != t_stat.shape[0]:
        raise ValueError("net_labels must have shape (N,) matching t_stat.")

    # Round to avoid float precision issues at threshold boundaries
    t_stat = np.round(t_stat, decimals=10)

    is_symm = _is_symmetric(t_stat)
    normalized_labels = _validate_net_labels(np.asarray(net_labels), t_stat.shape[0])
    n_networks = int(np.max(normalized_labels) + 1)

    edge_rows, edge_cols, edge_weights = _get_edges(t_stat, 0.0, symmetric=is_symm)
    if edge_rows.size == 0:
        return np.zeros_like(t_stat, dtype=np.float64)

    block_ids = _compute_canonical_block_ids(edge_rows, edge_cols, normalized_labels, n_networks)
    max_block_id = n_networks * n_networks

    sums = np.bincount(block_ids, weights=edge_weights.astype(np.float64), minlength=max_block_id)
    counts = np.bincount(block_ids, minlength=max_block_id)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)

    edge_scores = means[block_ids]
    scores = np.zeros_like(t_stat, dtype=np.float64)
    scores[edge_rows, edge_cols] = edge_scores
    if is_symm:
        scores[edge_cols, edge_rows] = edge_scores

    return scores


# =============================================================================
# Reference NBS (comparison/testing only)
# =============================================================================

def _score_nbs_from_diffs(
    diffs: npt.NDArray[np.float64],
    threshold: float,
    stat_type: str,
) -> Dict[str, npt.NDArray[np.float64]]:
    from .pairwise_stats import compute_t_stat_diff

    t_stat_dict = compute_t_stat_diff(diffs)
    return {
        "g2>g1": get_nbs_score(t_stat_dict["g2>g1"], threshold=threshold, stat_type=stat_type),
        "g1>g2": get_nbs_score(t_stat_dict["g1>g2"], threshold=threshold, stat_type=stat_type),
    }


def _score_nbs_two_sample(
    group1: npt.NDArray[np.float64],
    group2: npt.NDArray[np.float64],
    test_type: str,
    threshold: float,
    stat_type: str,
) -> Dict[str, npt.NDArray[np.float64]]:
    from .pairwise_stats import compute_t_stat

    t_stat_dict = compute_t_stat(group1, group2, test_type=test_type)
    return {
        "g2>g1": get_nbs_score(t_stat_dict["g2>g1"], threshold=threshold, stat_type=stat_type),
        "g1>g2": get_nbs_score(t_stat_dict["g1>g2"], threshold=threshold, stat_type=stat_type),
    }


def nbs_bct(
    group1: npt.NDArray[np.float64],
    group2: Optional[npt.NDArray[np.float64]] = None,
    threshold: float = DEFAULT_NBS_THRESHOLD,
    stat_type: str = DEFAULT_NBS_STAT,
    n_permutations: int = 100,
    test_type: str = "paired",
    use_mp: bool = True,
    random_state: Optional[int] = None,
    n_processes: Optional[int] = None,
    **kwargs,
) -> Tuple[
    Dict[str, npt.NDArray[np.float64]],
    Dict[str, npt.NDArray[np.uint8]],
    Dict[str, npt.NDArray[np.float64]],
]:
    """
    Compute classical NBS with permutation testing.

    Returns p-values, adjacency matrices (thresholded t-stats), and null maxima.
    """
    from .pairwise_stats import compute_null_dist, compute_t_stat, compute_t_stat_diff, compute_diffs

    if test_type == "paired":
        diffs = compute_diffs(group1, group2)
        emp_t_dict = _score_nbs_from_diffs(diffs, threshold=threshold, stat_type=stat_type)
        t_func = _score_nbs_from_diffs

    elif test_type == "two-sample":
        emp_t_dict = _score_nbs_two_sample(group1, group2, test_type="two-sample", threshold=threshold, stat_type=stat_type)
        t_func = _score_nbs_two_sample

    elif test_type == "one-sample":
        if group2 is not None:
            raise ValueError("group2 must be None for one-sample tests.")
        if group1.ndim != 3:
            raise ValueError("Dimensions of group 1 data should be: (subjects, N, N).")
        emp_t_dict = _score_nbs_from_diffs(group1, threshold=threshold, stat_type=stat_type)
        t_func = _score_nbs_from_diffs

    else:
        raise ValueError("test_type must be one of {'paired', 'one-sample', 'two-sample'}.")

    # Build adjacency masks for reporting (raw thresholding on t-stats)
    adj_matrices: Dict[str, npt.NDArray[np.uint8]] = {}
    if test_type == "paired":
        raw_t = compute_t_stat_diff(compute_diffs(group1, group2))
    elif test_type == "two-sample":
        raw_t = compute_t_stat(group1, group2, test_type="two-sample", **kwargs)
    else:
        raw_t = compute_t_stat_diff(group1)

    for key in raw_t:
        adj = (raw_t[key] > threshold).astype(np.uint8)
        if adj.shape[-1] == adj.shape[-2]:
            adj = np.triu(adj, 1)
            adj = adj + adj.T
        adj_matrices[key] = adj

    # Null distribution for max cluster statistic
    group2_for_null = group2 if test_type != "one-sample" else None
    max_null_dict = compute_null_dist(
        group1,
        group2_for_null,
        t_func,
        n_permutations=n_permutations,
        test_type=test_type,
        use_mp=use_mp,
        random_state=random_state,
        n_processes=n_processes,
        threshold=threshold,
        stat_type=stat_type,
    )

    # P-values via max-stat (cluster extent or intensity)
    keys = list(emp_t_dict.keys())
    p_values: Dict[str, npt.NDArray[np.float64]] = {}
    for key in keys:
        emp_t = emp_t_dict[key][..., np.newaxis]
        p_values[key] = np.mean(emp_t < max_null_dict[key], axis=-1)

    return p_values, adj_matrices, max_null_dict
