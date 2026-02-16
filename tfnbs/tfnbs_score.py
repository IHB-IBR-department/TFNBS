"""
Threshold-Free Network-Based Statistics (TFNBS) scoring module.

This module provides implementations of TFCE (Threshold-Free Cluster Enhancement)
adapted for network/connectivity analysis. The main function is `get_tfnbs_score()`
which transforms t-statistic matrices into TFNBS scores.

Functions
---------
get_tfnbs_score : Main optimized TFNBS implementation using scipy
get_tfnbs_score_baseline : Non-optimized version for benchmarking
get_tfnbs_score_networkx : Legacy networkx implementation for comparison with other packages
"""

from __future__ import annotations

import logging
from typing import Optional, Union, List, Tuple

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        """Fallback decorator when numba is not installed."""
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


__all__ = [
    "get_tfnbs_score",
    "get_tfnbs_score_baseline",
    "get_tfnbs_score_networkx",
    "get_network_informed_tfnbs_score",
    "get_fbc_tfnbs_score",
    "HAS_NUMBA",
    "DEFAULT_START_THRESHOLD",
    "DEFAULT_EXTENT_EXPONENT",
    "DEFAULT_HEIGHT_EXPONENT",
    "DEFAULT_N_THRESHOLDS",
    "DEFAULT_MIN_CLUSTER_SIZE",
]

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

DEFAULT_START_THRESHOLD: float = 1.65
"""Default initial threshold for cluster formation (corresponds to p < 0.05 one-tailed)."""

DEFAULT_EXTENT_EXPONENT: float = 0.5
"""Default extent exponent (E) for TFCE."""

DEFAULT_HEIGHT_EXPONENT: float = 2.0
"""Default height exponent (H) for TFCE."""

DEFAULT_N_THRESHOLDS: int = 100
"""Default number of threshold integration steps."""

DEFAULT_MIN_CLUSTER_SIZE: int = 3
"""Default minimum edges in a functional block to form a cluster (for FBC-TFNBS)."""


# =============================================================================
# Type aliases
# =============================================================================

ArrayLike = Union[float, List[float], npt.NDArray[np.floating]]


# =============================================================================
# Helper functions
# =============================================================================


def _validate_params(
    t_stats: npt.NDArray[np.floating],
    e: ArrayLike,
    h: ArrayLike
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], bool]:
    """
    Validate input dimensions and unify scalar/array parameters.

    Parameters
    ----------
    t_stats : ndarray of shape (N, N)
        Input statistical matrix.
    e : float or array-like
        Extent parameter(s).
    h : float or array-like
        Height parameter(s).

    Returns
    -------
    e_arr : ndarray
        Extent parameter as 1D array.
    h_arr : ndarray
        Height parameter as 1D array.
    scalar_mode : bool
        True if both e and h were originally scalars.

    Raises
    ------
    ValueError
        If diagonal elements are non-zero or e/h shapes don't match.
    """
    if not np.all(np.diag(t_stats) == 0):
        raise ValueError("Diagonal elements of the connectivity matrix must be zero (no self-connections).")

    scalar_mode = np.isscalar(e) and np.isscalar(h)

    e_arr = np.atleast_1d(e)
    h_arr = np.atleast_1d(h)

    if e_arr.shape != h_arr.shape:
        raise ValueError("Parameters 'e' and 'h' must have the same shape.")

    return e_arr, h_arr, scalar_mode


def _compute_thresholds(
    t_stats: npt.NDArray[np.floating],
    n: int,
    start_thres: float
) -> Tuple[Optional[npt.NDArray[np.floating]], float]:
    """
    Compute threshold range for TFCE integration.

    Parameters
    ----------
    t_stats : ndarray of shape (N, N)
        Statistical matrix.
    n : int
        Number of threshold steps.
    start_thres : float
        Initial threshold.

    Returns
    -------
    thresholds : ndarray or None
        Array of threshold values, or None if no valid range exists.
    dh : float
        Step size for integration.
    """
    max_stat = np.max(t_stats)
    dh = (max_stat - start_thres) / n

    if dh <= 0:
        return None, 0

    thresholds = np.linspace(start_thres + dh, max_stat, n)
    return thresholds, dh


def _is_symmetric(matrix: npt.NDArray[np.floating], rtol: float = 1e-10) -> bool:
    """Check if matrix is symmetric within given tolerance."""
    return np.allclose(matrix, matrix.T, rtol=rtol, atol=0)


def _get_edges(
    t_stats: npt.NDArray[np.floating],
    start_thres: float,
    symmetric: bool
) -> Tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], npt.NDArray[np.floating]]:
    """
    Extract edges with values above threshold.

    For symmetric matrices, extracts only upper triangle for efficiency.
    For asymmetric matrices, extracts all off-diagonal elements.

    Parameters
    ----------
    t_stats : ndarray of shape (N, N)
        Statistical matrix.
    start_thres : float
        Minimum threshold for edge inclusion.
    symmetric : bool
        Whether matrix is symmetric (use upper triangle only).

    Returns
    -------
    row_indices : ndarray
        Row indices of valid edges.
    col_indices : ndarray
        Column indices of valid edges.
    edge_weights : ndarray
        Edge weight values.
    """
    if symmetric:
        rows, cols = np.triu_indices_from(t_stats, k=1)
    else:
        nroi = t_stats.shape[0]
        rows, cols = np.where(~np.eye(nroi, dtype=bool))

    weights = t_stats[rows, cols]
    valid_mask = weights > start_thres
    return rows[valid_mask], cols[valid_mask], weights[valid_mask]


def get_tfnbs_score(
    t_stats: npt.NDArray[np.floating],
    e: ArrayLike,
    h: ArrayLike,
    n: int,
    start_thres: float = DEFAULT_START_THRESHOLD,
    backend: str = 'auto'
) -> npt.NDArray[np.floating]:
    """
    Transform the connectivity matrix using Threshold-Free Network-Based Statistics.

    Optimized implementation using scipy's csgraph module with:

    - Pre-extracted edges to avoid repeated masking
    - Buffer reuse to minimize memory allocations
    - Vectorized cluster size computation

    Parameters
    ----------
    t_stats : ndarray of shape (N, N)
        Statistical matrix to be transformed.
    e : float or array-like
        Extent exponent. Can be a scalar or list of values for parameter sweep.
    h : float or array-like
        Height exponent. Can be a scalar or list of values for parameter sweep.
    n : int
        Number of threshold steps between start_thres and max(t_stats).
    start_thres : float, default=1.65
        Initial threshold for cluster formation.
    backend : {'auto', 'numba', 'scipy'}, default='auto'
        Computation backend. 'auto' uses numba if available, otherwise scipy.
        'numba' uses JIT-compiled union-find for connected components
        (8-30x faster). Falls back to 'scipy' if numba is not installed.

    Returns
    -------
    tfnbs : ndarray of shape (N, N) or (N, N, num_params)
        TFNBS score matrix. Shape is (N, N) if e and h are scalars,
        otherwise (N, N, num_params).

    Examples
    --------
    >>> t = np.array([[0, 2.1, 0.5],[2.1, 0, 2.5],[0.5, 2.5, 0]])
    >>> np.round(get_tfnbs_score(t, e=0.5, h=2.0, n=10), 2)
    array([[0.  , 2.19, 0.  ],
           [2.19, 0.  , 4.5 ],
           [0.  , 4.5 , 0.  ]])
    """
    use_numba = (backend == 'numba') or (backend == 'auto' and HAS_NUMBA)
    if use_numba:
        if not HAS_NUMBA:
            logger.warning("Numba not installed, falling back to scipy backend.")
        else:
            return _get_tfnbs_score_numba(t_stats, e, h, n, start_thres)
    e_arr, h_arr, scalar_mode = _validate_params(t_stats, e, h)

    # Round to avoid float precision issues at threshold boundaries
    t_stats = np.round(t_stats, decimals=10)

    nroi = t_stats.shape[0]
    num_params = len(e_arr)
    tfnbs_shape = (nroi, nroi) if scalar_mode else (nroi, nroi, num_params)
    tfnbs = np.zeros(tfnbs_shape)

    threshs, dh = _compute_thresholds(t_stats, n, start_thres)
    if threshs is None:
        return tfnbs

    # Check if matrix is symmetric for optimization
    is_symm = _is_symmetric(t_stats)
    edge_rows, edge_cols, edge_weights = _get_edges(t_stats, start_thres, symmetric=is_symm)

    if len(edge_rows) == 0:
        return tfnbs

    if scalar_mode:
        e_bc = e_arr[0]
        h_bc = h_arr[0]
    else:
        e_bc = e_arr
        h_bc = h_arr

    # Pre-allocate reusable buffers
    mask = np.zeros((nroi, nroi), dtype=bool)
    clustsize = np.zeros((nroi, nroi), dtype=np.float64)

    for threshold in threshs:
        active_edges = edge_weights >= threshold

        if not np.any(active_edges):
            continue

        active_rows = edge_rows[active_edges]
        active_cols = edge_cols[active_edges]

        # Build mask for active edges
        mask.fill(False)
        mask[active_rows, active_cols] = True
        if is_symm:
            # For symmetric matrices, mirror to lower triangle
            mask[active_cols, active_rows] = True

        # For connected_components, always use symmetric mask (undirected graph)
        mask_for_cc = mask | mask.T
        sparse_mat = csr_matrix(mask_for_cc)
        n_components, node_labels = connected_components(sparse_mat, directed=False)

        # Count edges per component
        edge_component_ids = node_labels[active_rows]
        component_edge_counts = np.bincount(edge_component_ids, minlength=n_components)
        edge_sizes = component_edge_counts[edge_component_ids].astype(np.float64)

        clustsize.fill(0)
        clustsize[active_rows, active_cols] = edge_sizes
        if is_symm:
            # For symmetric matrices, copy to lower triangle
            clustsize[active_cols, active_rows] = edge_sizes

        if scalar_mode:
            tfnbs += (clustsize ** e_bc) * (threshold ** h_bc)
        else:
            tfnbs += (clustsize[..., np.newaxis] ** e_bc) * (threshold ** h_bc)

    tfnbs *= dh
    return tfnbs


# =============================================================================
# Numba JIT backend
# =============================================================================

@njit(cache=True)
def _union_find_labels(rows, cols, n_nodes):
    """Union-Find connected components with path compression and union-by-rank.

    Parameters
    ----------
    rows, cols : int64 arrays
        Edge list (row[i], col[i]) for each edge.
    n_nodes : int
        Total number of nodes in the graph.

    Returns
    -------
    labels : int32 array of shape (n_nodes,)
        Component label for each node (compressed so root = label).
    """
    parent = np.arange(n_nodes, dtype=np.int32)
    rank = np.zeros(n_nodes, dtype=np.int32)

    for i in range(len(rows)):
        # find root of rows[i]
        a = rows[i]
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        # find root of cols[i]
        b = cols[i]
        while parent[b] != b:
            parent[b] = parent[parent[b]]
            b = parent[b]
        if a != b:
            if rank[a] < rank[b]:
                parent[a] = b
            elif rank[a] > rank[b]:
                parent[b] = a
            else:
                parent[b] = a
                rank[a] += 1

    # Flatten labels
    labels = np.empty(n_nodes, dtype=np.int32)
    for i in range(n_nodes):
        x = i
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        labels[i] = x
    return labels


@njit(cache=True)
def _tfnbs_core_numba_scalar(
    edge_rows, edge_cols, edge_weights,
    thresholds, e, h, dh, nroi, is_symm
):
    """JIT-compiled TFNBS inner loop for scalar (e, h).

    Returns a flat array of shape (nroi * nroi,) representing the score matrix.
    """
    tfnbs = np.zeros(nroi * nroi, dtype=np.float64)

    for t_idx in range(len(thresholds)):
        threshold = thresholds[t_idx]

        # Count active edges
        n_active = 0
        for k in range(len(edge_weights)):
            if edge_weights[k] >= threshold:
                n_active += 1

        if n_active == 0:
            continue

        # Collect active edges
        active_rows = np.empty(n_active, dtype=np.int64)
        active_cols = np.empty(n_active, dtype=np.int64)
        idx = 0
        for k in range(len(edge_weights)):
            if edge_weights[k] >= threshold:
                active_rows[idx] = edge_rows[k]
                active_cols[idx] = edge_cols[k]
                idx += 1

        # Build symmetric edge list for union-find
        if is_symm:
            sym_rows = np.empty(n_active * 2, dtype=np.int64)
            sym_cols = np.empty(n_active * 2, dtype=np.int64)
            for k in range(n_active):
                sym_rows[k] = active_rows[k]
                sym_cols[k] = active_cols[k]
                sym_rows[n_active + k] = active_cols[k]
                sym_cols[n_active + k] = active_rows[k]
            labels = _union_find_labels(sym_rows, sym_cols, nroi)
        else:
            # For asymmetric, symmetrize for connected components
            sym_rows = np.empty(n_active * 2, dtype=np.int64)
            sym_cols = np.empty(n_active * 2, dtype=np.int64)
            for k in range(n_active):
                sym_rows[k] = active_rows[k]
                sym_cols[k] = active_cols[k]
                sym_rows[n_active + k] = active_cols[k]
                sym_cols[n_active + k] = active_rows[k]
            labels = _union_find_labels(sym_rows, sym_cols, nroi)

        # Count edges per component (using the original active edges)
        # First, find max label for bincount size
        max_label = 0
        for k in range(n_active):
            lbl = labels[active_rows[k]]
            if lbl > max_label:
                max_label = lbl
        comp_edge_counts = np.zeros(max_label + 1, dtype=np.int64)
        for k in range(n_active):
            comp_edge_counts[labels[active_rows[k]]] += 1

        # Accumulate scores
        h_power = threshold ** h
        for k in range(n_active):
            edge_size = float(comp_edge_counts[labels[active_rows[k]]])
            score_inc = (edge_size ** e) * h_power
            r, c = active_rows[k], active_cols[k]
            tfnbs[r * nroi + c] += score_inc
            if is_symm:
                tfnbs[c * nroi + r] += score_inc

    # Multiply by dh
    for i in range(len(tfnbs)):
        tfnbs[i] *= dh

    return tfnbs


@njit(cache=True)
def _tfnbs_core_numba_multi(
    edge_rows, edge_cols, edge_weights,
    thresholds, e_arr, h_arr, dh, nroi, n_params, is_symm
):
    """JIT-compiled TFNBS inner loop for multiple (e, h) parameter pairs.

    Returns a flat array of shape (nroi * nroi * n_params,).
    """
    tfnbs = np.zeros(nroi * nroi * n_params, dtype=np.float64)

    for t_idx in range(len(thresholds)):
        threshold = thresholds[t_idx]

        # Count active edges
        n_active = 0
        for k in range(len(edge_weights)):
            if edge_weights[k] >= threshold:
                n_active += 1

        if n_active == 0:
            continue

        # Collect active edges
        active_rows = np.empty(n_active, dtype=np.int64)
        active_cols = np.empty(n_active, dtype=np.int64)
        idx = 0
        for k in range(len(edge_weights)):
            if edge_weights[k] >= threshold:
                active_rows[idx] = edge_rows[k]
                active_cols[idx] = edge_cols[k]
                idx += 1

        # Build symmetric edge list for union-find
        sym_rows = np.empty(n_active * 2, dtype=np.int64)
        sym_cols = np.empty(n_active * 2, dtype=np.int64)
        for k in range(n_active):
            sym_rows[k] = active_rows[k]
            sym_cols[k] = active_cols[k]
            sym_rows[n_active + k] = active_cols[k]
            sym_cols[n_active + k] = active_rows[k]
        labels = _union_find_labels(sym_rows, sym_cols, nroi)

        # Count edges per component
        max_label = 0
        for k in range(n_active):
            lbl = labels[active_rows[k]]
            if lbl > max_label:
                max_label = lbl
        comp_edge_counts = np.zeros(max_label + 1, dtype=np.int64)
        for k in range(n_active):
            comp_edge_counts[labels[active_rows[k]]] += 1

        # Accumulate scores for all parameter pairs
        for k in range(n_active):
            edge_size = float(comp_edge_counts[labels[active_rows[k]]])
            r, c = active_rows[k], active_cols[k]
            for p in range(n_params):
                score_inc = (edge_size ** e_arr[p]) * (threshold ** h_arr[p])
                tfnbs[(r * nroi + c) * n_params + p] += score_inc
                if is_symm:
                    tfnbs[(c * nroi + r) * n_params + p] += score_inc

    # Multiply by dh
    for i in range(len(tfnbs)):
        tfnbs[i] *= dh

    return tfnbs


def _get_tfnbs_score_numba(
    t_stats: npt.NDArray[np.floating],
    e: ArrayLike,
    h: ArrayLike,
    n: int,
    start_thres: float = DEFAULT_START_THRESHOLD
) -> npt.NDArray[np.floating]:
    """TFNBS scoring using Numba JIT backend.

    Same interface as get_tfnbs_score() but uses JIT-compiled union-find
    for connected components instead of scipy.sparse.csgraph.

    Parameters
    ----------
    t_stats : ndarray of shape (N, N)
        Statistical matrix to be transformed.
    e : float or array-like
        Extent exponent.
    h : float or array-like
        Height exponent.
    n : int
        Number of threshold steps.
    start_thres : float, default=1.65
        Initial threshold for cluster formation.

    Returns
    -------
    tfnbs : ndarray of shape (N, N) or (N, N, num_params)
        TFNBS score matrix.
    """
    e_arr, h_arr, scalar_mode = _validate_params(t_stats, e, h)
    t_stats = np.round(t_stats, decimals=10)

    nroi = t_stats.shape[0]
    num_params = len(e_arr)

    threshs, dh = _compute_thresholds(t_stats, n, start_thres)
    if threshs is None:
        tfnbs_shape = (nroi, nroi) if scalar_mode else (nroi, nroi, num_params)
        return np.zeros(tfnbs_shape)

    is_symm = _is_symmetric(t_stats)
    edge_rows, edge_cols, edge_weights = _get_edges(t_stats, start_thres, symmetric=is_symm)

    if len(edge_rows) == 0:
        tfnbs_shape = (nroi, nroi) if scalar_mode else (nroi, nroi, num_params)
        return np.zeros(tfnbs_shape)

    # Ensure int64 for numba
    edge_rows = edge_rows.astype(np.int64)
    edge_cols = edge_cols.astype(np.int64)
    edge_weights = edge_weights.astype(np.float64)
    e_arr = e_arr.astype(np.float64)
    h_arr = h_arr.astype(np.float64)

    if scalar_mode:
        flat = _tfnbs_core_numba_scalar(
            edge_rows, edge_cols, edge_weights,
            threshs, e_arr[0], h_arr[0], dh, nroi, is_symm
        )
        return flat.reshape(nroi, nroi)
    else:
        flat = _tfnbs_core_numba_multi(
            edge_rows, edge_cols, edge_weights,
            threshs, e_arr, h_arr, dh, nroi, num_params, is_symm
        )
        return flat.reshape(nroi, nroi, num_params)


def get_tfnbs_score_baseline(
    t_stats: npt.NDArray[np.floating],
    e: ArrayLike,
    h: ArrayLike,
    n: int,
    start_thres: float = DEFAULT_START_THRESHOLD
) -> npt.NDArray[np.floating]:
    """
    Baseline (non-optimized) implementation of TFNBS for performance comparison.

    This is the original scipy-based implementation without optimizations.
    Use `get_tfnbs_score()` for production - it is ~1.5x faster.

    Parameters
    ----------
    t_stats : ndarray of shape (N, N)
        Statistical matrix to be transformed.
    e : float or array-like
        Extent exponent. Can be a scalar or list of values.
    h : float or array-like
        Height exponent. Can be a scalar or list of values.
    n : int
        Number of threshold steps between start_thres and max(t_stats).
    start_thres : float, default=1.65
        Initial threshold for cluster formation.

    Returns
    -------
    tfnbs : ndarray of shape (N, N) or (N, N, num_params)
        TFNBS score matrix.

    Examples
    --------
    >>> t = np.array([[0, 2.1, 0.5],[2.1, 0, 2.5],[0.5, 2.5, 0]])
    >>> np.round(get_tfnbs_score_baseline(t, e=0.5, h=2.0, n=10), 2)
    array([[0.  , 2.19, 0.  ],
           [2.19, 0.  , 4.5 ],
           [0.  , 4.5 , 0.  ]])
    """
    if not np.all(np.diag(t_stats) == 0):
        raise ValueError("Diagonal elements of the connectivity matrix must be zero (no self-connections).")

    # Round to avoid float precision issues at threshold boundaries
    t_stats = np.round(t_stats, decimals=10)

    scalar_mode = np.isscalar(e) and np.isscalar(h)
    if scalar_mode:
        e, h = [e], [h]

    e = np.array(e)
    h = np.array(h)
    if e.shape != h.shape:
        raise ValueError("e and h must have the same shape!")

    nroi = t_stats.shape[0]
    num_params = len(e)
    tfnbs_shape = (nroi, nroi) if scalar_mode else (nroi, nroi, num_params)
    tfnbs = np.zeros(tfnbs_shape)

    max_stat = np.max(t_stats)
    dh = (max_stat - start_thres) / n
    if dh == 0:
        return tfnbs
    threshs = np.linspace(start_thres + dh, max_stat, n)

    if not scalar_mode:
        e = e.reshape(1, 1, num_params)
        h = h.reshape(1, 1, num_params)

    for threshold in threshs:
        mask = t_stats >= threshold
        np.fill_diagonal(mask, False)
        n_components, labels = connected_components(mask.astype(int), directed=False)

        unique, counts = np.unique(labels, return_counts=True)
        clustsize = 1. * mask.copy()

        for lbl, size in zip(unique, counts):
            if size >= 2:
                sz_links = np.sum(mask[np.ix_(labels == lbl, labels == lbl)]) / 2
                clustsize[np.ix_(labels == lbl, labels == lbl)] *= sz_links

        np.fill_diagonal(clustsize, 0)

        if scalar_mode:
            tfnbs += (clustsize ** e[0]) * (threshold ** h[0])
        else:
            tfnbs += (clustsize[..., np.newaxis] ** e) * (threshold ** h)

    tfnbs *= dh
    return tfnbs


def get_tfnbs_score_networkx(
    t_stats: npt.NDArray[np.floating],
    e: ArrayLike,
    h: ArrayLike,
    n: int,
    start_thres: float = DEFAULT_START_THRESHOLD
) -> npt.NDArray[np.floating]:
    """
    Transform the connectivity matrix using Threshold-Free Network-Based Statistics.

    Legacy networkx implementation - use `get_tfnbs_score()` for better performance.
    This function is kept for comparison with other packages that use networkx.

    Parameters
    ----------
    t_stats : ndarray of shape (N, N)
        Statistical matrix to be transformed.
    e : float or array-like
        Extent exponent. Can be a scalar or list of values.
    h : float or array-like
        Height exponent. Can be a scalar or list of values.
    n : int
        Number of threshold steps between start_thres and max(t_stats).
    start_thres : float, default=1.65
        Initial threshold for cluster formation.

    Returns
    -------
    tfnbs : ndarray of shape (N, N) or (N, N, num_params)
        TFNBS score matrix.

    Examples
    --------
    >>> t = np.array([[0, 2.1, 0.5],[2.1, 0, 2.5],[0.5, 2.5, 0]])
    >>> np.round(get_tfnbs_score_networkx(t, e=0.5, h=2.0, n=10), 2)
    array([[0.  , 2.19, 0.  ],
           [2.19, 0.  , 4.5 ],
           [0.  , 4.5 , 0.  ]])
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError(
            "networkx is required for get_tfnbs_score_networkx(). "
            "Use get_tfnbs_score() instead."
        )

    e_arr, h_arr, scalar_mode = _validate_params(t_stats, e, h)

    nroi = t_stats.shape[0]
    num_params = len(e_arr)
    tfnbs_shape = (nroi, nroi) if scalar_mode else (nroi, nroi, num_params)
    tfnbs = np.zeros(tfnbs_shape)

    threshs, dh = _compute_thresholds(t_stats, n, start_thres)
    if threshs is None:
        return tfnbs

    if scalar_mode:
        e_bc = e_arr[0]
        h_bc = h_arr[0]
    else:
        e_bc = e_arr.reshape(1, 1, num_params)
        h_bc = h_arr.reshape(1, 1, num_params)

    clustsize = np.zeros((nroi, nroi), dtype=np.float64)

    for threshold in threshs:
        mask = t_stats >= threshold
        np.fill_diagonal(mask, False)

        if not np.any(mask):
            continue

        G = nx.from_numpy_array(mask)
        components = list(nx.connected_components(G))

        clustsize.fill(0)
        clustsize[mask] = 1.0

        for component in components:
            if len(component) >= 2:
                component_list = list(component)
                sz_links = np.sum(mask[np.ix_(component_list, component_list)]) / 2
                clustsize[np.ix_(component_list, component_list)] *= sz_links

        np.fill_diagonal(clustsize, 0)

        if scalar_mode:
            tfnbs += (clustsize ** e_bc) * (threshold ** h_bc)
        else:
            tfnbs += (clustsize[..., np.newaxis] ** e_bc) * (threshold ** h_bc)

    tfnbs *= dh
    return tfnbs


"""
Network-Informed TFNBS scoring module.

This module extends TFNBS to incorporate functional network architecture.
It weights topological clusters based on the density of the functional blocks
they span, increasing sensitivity to effects that align with known
functional anatomy.
"""


# =============================================================================
# Specific Helper Functions
# =============================================================================

def _validate_net_labels(
    net_labels: npt.NDArray[np.integer], 
    n_nodes: int
) -> npt.NDArray[np.integer]:
    """
    Validate and normalize network labels.
    
    Parameters
    ----------
    net_labels : ndarray of shape (N,)
        Network assignments.
    n_nodes : int
        Expected number of nodes.
        
    Returns
    -------
    normalized_labels : ndarray
        Labels mapped to 0..K-1 range.
    """
    if net_labels.shape[0] != n_nodes:
        raise ValueError(f"net_labels shape {net_labels.shape} does not match "
                         f"number of nodes {n_nodes}.")
    
    # Map to continuous range 0..K-1 for efficient bincount indexing
    _, inverse = np.unique(net_labels, return_inverse=True)
    return inverse


def _precompute_block_metadata(
    edge_rows: npt.NDArray[np.intp],
    edge_cols: npt.NDArray[np.intp],
    node_labels: npt.NDArray[np.integer],
) -> Tuple[npt.NDArray[np.intp], npt.NDArray[np.floating]]:
    """
    Pre-compute block assignments and capacity factors for all extractable edges.
    
    This avoids re-calculating block IDs and capacities inside the threshold loop.
    
    Parameters
    ----------
    edge_rows : ndarray
        Row indices of all potential suprathreshold edges.
    edge_cols : ndarray
        Column indices of all potential suprathreshold edges.
    node_labels : ndarray
        Normalized network labels (0..K-1) for each node.
        
    Returns
    -------
    edge_block_ids : ndarray
        The Block ID assigned to each edge in `edge_rows`.
    sqrt_capacities : ndarray
        Lookup table for sqrt(M_total) indexed by Block ID.
    """
    n_nets = np.max(node_labels) + 1
    
    # Calculate Block ID for each edge: Block_ID = Label_i * N_nets + Label_j
    # Note: For symmetric analysis (triu), this implicitly defines blocks 
    # based on the upper triangle connections.
    labels_i = node_labels[edge_rows]
    labels_j = node_labels[edge_cols]
    
    # Ensure canonical block ID for undirected edges (min, max) to handle symmetry
    # effectively if needed, though simple mapping usually suffices if we stick 
    # to upper triangle. Here we stick to direct mapping.
    edge_block_ids = labels_i * n_nets + labels_j
    
    # Total number of possible blocks
    n_blocks_total = n_nets * n_nets
    
    # Calculate Capacity (M_total) for each block based on the provided edges.
    # If we are working with just upper triangle edges, this counts the 
    # capacity of the upper triangle block, which is correct for the density formula.
    block_capacities = np.bincount(edge_block_ids, minlength=n_blocks_total).astype(np.float64)
    
    # Pre-calculate sqrt(Capacity) for the weighting formula: W = k / sqrt(M)
    # Avoid division by zero
    sqrt_capacities = np.sqrt(block_capacities)
    sqrt_capacities[sqrt_capacities == 0] = 1.0
    
    return edge_block_ids, sqrt_capacities


def _compute_edge_block_weights(
    active_edge_indices: npt.NDArray[np.bool_],
    edge_block_ids: npt.NDArray[np.intp],
    sqrt_capacities: npt.NDArray[np.floating]
) -> npt.NDArray[np.float64]:
    """
    Compute edge-level weights based on functional block density.

    Each edge's weight depends only on its functional block density,
    with no topological clustering (Edge-Level Weighting approach).

    Formula: W_edge = k_active_in_block / sqrt(M_total_in_block)

    Parameters
    ----------
    active_edge_indices : ndarray
        Boolean mask indicating which pre-extracted edges are active.
    edge_block_ids : ndarray
        Block ID for every pre-extracted edge.
    sqrt_capacities : ndarray
        Sqrt of total edges per block (precomputed).

    Returns
    -------
    edge_weights : ndarray
        Weight for each active edge based on its block density.
    """
    # 1. Get block IDs for active edges
    active_block_ids = edge_block_ids[active_edge_indices]

    # 2. Count active edges (k) per block
    block_active_counts = np.bincount(active_block_ids, minlength=len(sqrt_capacities))

    # 3. Calculate weight per block: W = k / sqrt(M)
    block_weights = block_active_counts / sqrt_capacities

    # 4. Return weight for each active edge
    return block_weights[active_block_ids]


# =============================================================================
# Main Function
# =============================================================================

def get_network_informed_tfnbs_score(
    t_stats: npt.NDArray[np.floating],
    net_labels: npt.NDArray[np.integer],
    e: ArrayLike,
    h: ArrayLike,
    n: int,
    start_thres: float = DEFAULT_START_THRESHOLD
) -> npt.NDArray[np.floating]:
    """
    Transform connectivity matrix using Network-Informed TFNBS (Edge-Level Weighting).

    This method uses Edge-Level Weighting: each edge's support depends only on
    the density of its functional block, with NO topological clustering.
    This prevents "signal bleeding" where noise edges inherit scores from
    signal clusters they happen to connect to.

    Weighting Logic:
        S_edge = k_active_in_block / sqrt(M_total_edges_in_block)

    Key difference from standard TFNBS:
        - Standard TFNBS: edges cluster if they share a node (topological)
        - NI-TFNBS: each edge scored independently by block density

    Parameters
    ----------
    t_stats : ndarray of shape (N, N)
        Statistical matrix to be transformed.
    net_labels : ndarray of shape (N,)
        Integer labels assigning each node to a functional network.
    e : float or array-like
        Extent exponent.
    h : float or array-like
        Height exponent.
    n : int
        Number of threshold steps.
    start_thres : float, default=1.65
        Initial threshold for cluster formation.

    Returns
    -------
    tfnbs : ndarray of shape (N, N) or (N, N, num_params)
        Network-Informed TFNBS score matrix.
    """
    # 1. Validation
    e_arr, h_arr, scalar_mode = _validate_params(t_stats, e, h)
    nroi = t_stats.shape[0]
    
    normalized_net_labels = _validate_net_labels(np.asarray(net_labels), nroi)
    
    # Round to avoid float precision issues at threshold boundaries
    t_stats = np.round(t_stats, decimals=10)
    
    # Initialize output
    num_params = len(e_arr)
    tfnbs_shape = (nroi, nroi) if scalar_mode else (nroi, nroi, num_params)
    tfnbs = np.zeros(tfnbs_shape)

    # 2. Thresholds
    threshs, dh = _compute_thresholds(t_stats, n, start_thres)
    if threshs is None:
        return tfnbs

    # 3. Pre-processing
    # Check symmetry to optimize edge extraction
    is_symm = _is_symmetric(t_stats)
    
    # Extract ALL potential edges > start_thres. 
    # We do this ONCE to avoid repeated masking of the full NxN matrix.
    edge_rows, edge_cols, edge_weights = _get_edges(t_stats, start_thres, symmetric=is_symm)

    if len(edge_rows) == 0:
        return tfnbs

    # Pre-compute Block Metadata for these specific edges
    # This gives us the Block ID for every edge in our list and the Capacity of blocks
    edge_block_ids, sqrt_capacities = _precompute_block_metadata(
        edge_rows, edge_cols, normalized_net_labels
    )

    # Prepare broadcasting parameters
    if scalar_mode:
        e_bc = e_arr[0]
        h_bc = h_arr[0]
    else:
        e_bc = e_arr
        h_bc = h_arr

    # 4. Main Integration Loop (NO topological clustering - edge-level weighting)
    for threshold in threshs:
        # Determine which of our pre-extracted edges are active at this level
        active_indices_mask = edge_weights >= threshold

        if not np.any(active_indices_mask):
            continue

        # Get active edge indices
        active_rows = edge_rows[active_indices_mask]
        active_cols = edge_cols[active_indices_mask]

        # Compute edge-level weights based on block density
        # Each edge's weight = k_active_in_block / sqrt(M_total_in_block)
        edge_supports = _compute_edge_block_weights(
            active_indices_mask,
            edge_block_ids,
            sqrt_capacities
        )

        # Calculate score increment: h^H * support^E * dh
        if scalar_mode:
            increment = (threshold ** h_bc) * (edge_supports ** e_bc) * dh
            tfnbs[active_rows, active_cols] += increment
            if is_symm:
                tfnbs[active_cols, active_rows] += increment
        else:
            # Handle parameter sweep (broadcast dimensions)
            increment = (threshold ** h_bc) * (edge_supports[..., np.newaxis] ** e_bc) * dh
            tfnbs[active_rows, active_cols, :] += increment
            if is_symm:
                tfnbs[active_cols, active_rows, :] += increment

    return tfnbs


# =============================================================================
# Functional Block Clustering TFNBS (FBC-TFNBS)
# =============================================================================

def _compute_canonical_block_ids(
    edge_rows: npt.NDArray[np.intp],
    edge_cols: npt.NDArray[np.intp],
    node_labels: npt.NDArray[np.integer],
    n_networks: int
) -> npt.NDArray[np.intp]:
    """
    Compute canonical block IDs for edges (handles undirected networks).

    For undirected networks, Block(A→B) == Block(B→A), so we use
    min(label_i, label_j) * n_networks + max(label_i, label_j).

    Parameters
    ----------
    edge_rows : ndarray
        Row indices of edges.
    edge_cols : ndarray
        Column indices of edges.
    node_labels : ndarray
        Network label for each node (0..K-1).
    n_networks : int
        Number of distinct networks.

    Returns
    -------
    block_ids : ndarray
        Canonical block ID for each edge.
    """
    labels_i = node_labels[edge_rows]
    labels_j = node_labels[edge_cols]

    # Canonical ordering for undirected: min * n_networks + max
    min_labels = np.minimum(labels_i, labels_j)
    max_labels = np.maximum(labels_i, labels_j)

    return min_labels * n_networks + max_labels


def get_fbc_tfnbs_score(
    t_stats: npt.NDArray[np.floating],
    net_labels: npt.NDArray[np.integer],
    e: ArrayLike,
    h: ArrayLike,
    n: int,
    start_thres: float = DEFAULT_START_THRESHOLD,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE
) -> npt.NDArray[np.floating]:
    """
    Functional Block Clustering TFNBS (FBC-TFNBS).

    This method clusters edges by functional block membership rather than
    topological connectivity. At each threshold:

    1. Edges are grouped by their functional block (e.g., Visual-Visual, DMN-DMN)
    2. If a block has k >= min_cluster_size edges, those edges support each other
       (support = k)
    3. If a block has fewer edges, they are considered "isolated" and suppressed
       (support = 0)

    Key difference from standard TFNBS:
        - Standard TFNBS: edges cluster if they share a node (topological)
        - FBC-TFNBS: edges cluster if they're in the same functional block

    Key difference from NI-TFNBS:
        - NI-TFNBS: topological clustering with density weighting (spreads to noise)
        - FBC-TFNBS: no topological clustering, isolated edges are suppressed

    Parameters
    ----------
    t_stats : ndarray of shape (N, N)
        Statistical matrix to be transformed (usually absolute t-values).
    net_labels : ndarray of shape (N,)
        Integer labels assigning each node to a functional network.
    e : float or array-like
        Extent exponent. Controls how much cluster size matters.
    h : float or array-like
        Height exponent. Controls how much threshold height matters.
    n : int
        Number of threshold steps between start_thres and max(t_stats).
    start_thres : float, default=1.65
        Initial threshold for cluster formation.
    min_cluster_size : int, default=3
        Minimum number of edges in a functional block to form a cluster.
        Blocks with fewer edges get support=0 (suppressed).

    Returns
    -------
    tfnbs : ndarray of shape (N, N) or (N, N, num_params)
        FBC-TFNBS score matrix.

    Notes
    -----
    The key insight is that edges in the same functional block should support
    each other even if they don't share any nodes. This captures the biological
    reality that effects within a functional system (e.g., Visual cortex) are
    likely related, while isolated edges are more likely noise.

    Examples
    --------
    >>> import numpy as np
    >>> # Create a simple t-statistic matrix
    >>> t = np.array([[0, 2.5, 1.8, 0.5],
    ...               [2.5, 0, 2.2, 0.3],
    ...               [1.8, 2.2, 0, 2.0],
    ...               [0.5, 0.3, 2.0, 0]])
    >>> # Nodes 0,1 in network 0; nodes 2,3 in network 1
    >>> labels = np.array([0, 0, 1, 1])
    >>> scores = get_fbc_tfnbs_score(t, labels, e=0.5, h=2.0, n=20,
    ...                               start_thres=1.5, min_cluster_size=2)
    """
    # 1. Validation
    e_arr, h_arr, scalar_mode = _validate_params(t_stats, e, h)
    nroi = t_stats.shape[0]

    normalized_net_labels = _validate_net_labels(np.asarray(net_labels), nroi)
    n_networks = np.max(normalized_net_labels) + 1

    # Round to avoid float precision issues
    t_stats = np.round(t_stats, decimals=10)

    # Initialize output
    num_params = len(e_arr)
    tfnbs_shape = (nroi, nroi) if scalar_mode else (nroi, nroi, num_params)
    tfnbs = np.zeros(tfnbs_shape)

    # 2. Compute thresholds
    threshs, dh = _compute_thresholds(t_stats, n, start_thres)
    if threshs is None:
        return tfnbs

    # 3. Pre-processing
    is_symm = _is_symmetric(t_stats)

    # Extract edges above start threshold
    edge_rows, edge_cols, edge_weights = _get_edges(t_stats, start_thres, symmetric=is_symm)

    if len(edge_rows) == 0:
        return tfnbs

    # Compute canonical block IDs for all extracted edges
    edge_block_ids = _compute_canonical_block_ids(
        edge_rows, edge_cols, normalized_net_labels, n_networks
    )

    # Maximum possible block ID (for bincount)
    max_block_id = n_networks * n_networks

    # Prepare broadcasting parameters
    if scalar_mode:
        e_bc = e_arr[0]
        h_bc = h_arr[0]
    else:
        e_bc = e_arr
        h_bc = h_arr

    # 4. Main Integration Loop
    for threshold in threshs:
        # Find active edges at this threshold
        active_mask = edge_weights >= threshold

        if not np.any(active_mask):
            continue

        active_indices = np.where(active_mask)[0]
        active_rows = edge_rows[active_indices]
        active_cols = edge_cols[active_indices]
        active_block_ids = edge_block_ids[active_indices]

        # Count edges per functional block at this threshold
        block_counts = np.bincount(active_block_ids, minlength=max_block_id)

        # Compute support for each active edge
        # Support = block_count if block_count >= min_cluster_size, else 0
        edge_block_counts = block_counts[active_block_ids]
        edge_supports = np.where(
            edge_block_counts >= min_cluster_size,
            edge_block_counts,
            0
        ).astype(np.float64)

        # Calculate score increment: h^H * support^E * dh
        # Note: support^E where support=0 gives 0, which is correct (suppressed)
        if scalar_mode:
            increment = (threshold ** h_bc) * (edge_supports ** e_bc) * dh
            tfnbs[active_rows, active_cols] += increment
            if is_symm:
                tfnbs[active_cols, active_rows] += increment
        else:
            # Handle parameter sweep
            increment = (threshold ** h_bc) * (edge_supports[..., np.newaxis] ** e_bc) * dh
            tfnbs[active_rows, active_cols, :] += increment
            if is_symm:
                tfnbs[active_cols, active_rows, :] += increment

    return tfnbs