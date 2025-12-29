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


__all__ = [
    "get_tfnbs_score",
    "get_tfnbs_score_baseline",
    "get_tfnbs_score_networkx",
    "DEFAULT_START_THRESHOLD",
    "DEFAULT_EXTENT_EXPONENT",
    "DEFAULT_HEIGHT_EXPONENT",
    "DEFAULT_N_THRESHOLDS",
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


def _compute_weighted_cluster_sizes_vectorized(
    n_nodes: int,
    rows: npt.NDArray[np.intp],
    cols: npt.NDArray[np.intp],
    mask: npt.NDArray[np.bool_],
    node_labels: npt.NDArray[np.intp],
    n_components: int,
    weight_map: npt.NDArray[np.floating]
) -> npt.NDArray[np.float64]:
    """
    Compute weighted cluster sizes using vectorized operations.

    Parameters
    ----------
    n_nodes : int
        Number of nodes in the graph.
    rows : ndarray
        Row indices of edges.
    cols : ndarray
        Column indices of edges.
    mask : ndarray of bool
        Boolean mask of active edges.
    node_labels : ndarray
        Component labels for each node.
    n_components : int
        Number of connected components.
    weight_map : ndarray of shape (N, N)
        Weight matrix for edges.

    Returns
    -------
    edge_sizes : ndarray
        Weighted cluster size for each edge.
    """
    active_mask = mask[rows, cols]

    if not np.any(active_mask):
        return np.zeros(len(rows), dtype=np.float64)

    edge_components = node_labels[rows]
    edge_weights = weight_map[rows, cols]

    weighted_sums = np.bincount(
        edge_components[active_mask],
        weights=edge_weights[active_mask],
        minlength=n_components
    )

    edge_sizes = np.zeros(len(rows), dtype=np.float64)
    edge_sizes[active_mask] = weighted_sums[edge_components[active_mask]]

    return edge_sizes


def get_tfnbs_score(
    t_stats: npt.NDArray[np.floating],
    e: ArrayLike,
    h: ArrayLike,
    n: int,
    start_thres: float = DEFAULT_START_THRESHOLD,
    weight_map: Optional[npt.NDArray[np.floating]] = None
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
    weight_map : ndarray of shape (N, N), optional
        Prior weights for edges. If provided, cluster sizes are weighted.

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
    e_arr, h_arr, scalar_mode = _validate_params(t_stats, e, h)

    # Round to avoid float precision issues at threshold boundaries
    t_stats = np.round(t_stats, decimals=10)

    if weight_map is not None:
        weight_map = np.asarray(weight_map, dtype=np.float64)
        if weight_map.shape != t_stats.shape:
            raise ValueError("`weight_map` must have the same shape as `t_stats` (N, N)")

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

        if weight_map is not None:
            edge_sizes = _compute_weighted_cluster_sizes_vectorized(
                nroi, active_rows, active_cols, mask, node_labels,
                n_components, weight_map
            )
        else:
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
