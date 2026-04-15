"""
Pairwise statistical testing module for network analysis.

This module provides functions for computing t-statistics and p-values
using permutation testing with optional TFCE (Threshold-Free Cluster Enhancement)
transformation.

Main Functions
--------------
compute_p_val : Compute p-values using permutation testing
compute_t_stat : Compute t-statistics for paired, one-sample, or two-sample tests
compute_t_stat_tfnbs : Compute TFNBS-enhanced t-statistics
compute_null_dist : Compute null distribution via permutation testing
"""

from __future__ import annotations
import os
import logging
import multiprocessing
from enum import Enum
from functools import partial
from multiprocessing import Pool
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from scipy import stats

from .defaults import (
    DEFAULT_EXTENT_EXPONENT,
    DEFAULT_HEIGHT_EXPONENT,
    DEFAULT_N_THRESHOLDS_PERMUTATION as DEFAULT_N_THRESHOLDS,
    DEFAULT_N_PERMUTATIONS,
    DEFAULT_START_THRESHOLD,
    DEFAULT_MIN_CLUSTER_SIZE,
    DEFAULT_NBS_THRESHOLD,
    DEFAULT_NBS_STAT,
)
from .nbs_score import get_cnbs_score, get_nbs_score
from .tfnbs_score import (
    get_tfnbs_score,
    get_network_informed_tfnbs_score,
    get_fbc_tfnbs_score,
)
from .acceleration import compute_p_values_accelerated


__all__ = [
    # Enums and constants
    "TestType",
    "StatMethod",
    "DEFAULT_N_PERMUTATIONS",
    "DEFAULT_EXTENT_EXPONENT",
    "DEFAULT_HEIGHT_EXPONENT",
    "DEFAULT_N_THRESHOLDS",
    # Main functions
    "compute_p_val",
    "compute_t_stat",
    "compute_t_stat_tfnbs",
    "compute_t_stat_tfnbs_diffs",
    "compute_null_dist",
    # T-statistic functions
    "compute_t_stat_diff",
    "compute_t_stat_ind",
    "compute_diffs",
    "get_available_cores",
    "_is_worker_process",
]

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class TestType(str, Enum):
    """Statistical test types for permutation testing."""

    PAIRED = "paired"
    """Paired samples t-test (within-subjects design)."""

    ONE_SAMPLE = "one-sample"
    """One-sample t-test against zero."""

    TWO_SAMPLE = "two-sample"
    """Independent samples t-test (between-subjects design)."""


class StatMethod(str, Enum):
    """Statistical method for network analysis."""

    TFNBS = "tfnbs"
    """Threshold-Free Network-Based Statistics (TFCE-style)."""

    TSTAT = "tstat"
    """Raw t-statistics without enhancement."""

    NBS = "nbs"
    """Classical Network-Based Statistics with fixed threshold."""

    CNBS = "cnbs"
    """Constrained NBS with predefined network partitions."""

    NI_TFNBS = "ni_tfnbs"
    """Network-Informed TFNBS with functional block density weighting."""

    FBC_TFNBS = "fbc_tfnbs"
    """Functional Block Clustering TFNBS (block-defined clustering)."""

    BONFERRONI = "bonferroni"
    """Parametric Bonferroni correction (no permutation testing)."""

    BH_FDR = "bh_fdr"
    """Parametric Benjamini-Hochberg FDR correction (no permutation testing)."""

    BH_FDR_PERM = "bh_fdr_perm"
    """Permutation-based BH-FDR correction (per-edge permutation p-values)."""


CONSTRAINED_METHODS = {StatMethod.CNBS, StatMethod.NI_TFNBS, StatMethod.FBC_TFNBS}
PARAMETRIC_METHODS = {StatMethod.BONFERRONI, StatMethod.BH_FDR}


# =============================================================================
# Helper functions for permutation testing
# =============================================================================

def _extract_max_stats(
    stat_dict: Dict[str, npt.NDArray],
    reference_shape: Tuple[int, ...]
) -> Dict[str, np.float64]:
    """
    Extract maximum statistics from a stat dictionary.

    Handles both 2D matrices and multi-parameter 3D arrays.
    Key-agnostic: works with any dictionary keys (e.g., 'g1>g2'/'g2>g1'
    for t-test pipeline, 'positive'/'negative' for GLM pipeline).

    Parameters
    ----------
    stat_dict : dict
        Dictionary with statistic arrays (any string keys).
    reference_shape : tuple
        Shape of a single sample for determining dimensionality.

    Returns
    -------
    dict
        Dictionary with maximum statistics for each key.
    """
    result = {}
    for key, arr in stat_dict.items():
        if arr.shape == reference_shape:
            result[key] = np.max(arr).astype(np.float64)
        else:
            # Multi-parameter case: max over spatial dims, keep param dim
            spatial_axes = tuple(range(arr.ndim - 1))
            result[key] = np.max(arr, axis=spatial_axes).astype(np.float64)
    return result


def _permutation_task_ind(
    full_group: npt.NDArray[np.float64],
    func: Callable[..., Any],
    n1: int,
    seed: int,
    **func_kwargs
) -> Dict[str, Union[float, npt.NDArray[np.float64]]]:
    """
    Compute maximum t-statistic for a single permutation (two-sample test).

    Parameters
    ----------
    full_group : ndarray
        Concatenated data array of shape (n_samples_1 + n_samples_2, *dims).
    func : callable
        Function to compute the t-statistic.
    n1 : int
        Number of samples in group 1.
    seed : int
        Random seed for this permutation.
    **func_kwargs
        Additional keyword arguments passed to func.

    Returns
    -------
    dict
        Dictionary with max statistics for 'g1>g2' and 'g2>g1' directions.
    """
    rng = np.random.RandomState(seed)
    idx = rng.permutation(full_group.shape[0])
    new_group1 = full_group[idx[:n1]]
    new_group2 = full_group[idx[n1:]]
    perm_stat_dict = func(new_group1, new_group2, test_type='two-sample', **func_kwargs)
    return _extract_max_stats(perm_stat_dict, full_group[0].shape)


def _permutation_task_paired(
    diffs: npt.NDArray[np.float64],
    func: Callable[..., Any],
    seed: Optional[int] = None,
    **func_kwargs
) -> Dict[str, Union[float, npt.NDArray[np.float64]]]:
    """
    Compute maximum t-statistic for a single permutation (paired/one-sample test).

    Parameters
    ----------
    diffs : ndarray
        Array of shape (n_samples, *dims) containing paired differences.
    func : callable
        Function to compute the t-statistic.
    seed : int, optional
        Random seed for this permutation.
    **func_kwargs
        Additional keyword arguments passed to func.

    Returns
    -------
    dict
        Dictionary with max statistics for 'g1>g2' and 'g2>g1' directions.
    """
    n_dims = len(diffs.shape) - 1
    faked_dims = [1] * n_dims
    rng = np.random.RandomState(seed)
    signs = rng.choice([1, -1], diffs.shape[0]).reshape(-1, *faked_dims)
    new_diffs = signs * diffs
    perm_stat_dict = func(new_diffs, **func_kwargs)
    return _extract_max_stats(perm_stat_dict, diffs[0].shape)


def _collect_results_to_arrays(
    results: List[Dict[str, Any]],
    n_permutations: int
) -> Dict[str, npt.NDArray[np.float64]]:
    """
    Efficiently collect permutation results into numpy arrays.

    Uses pre-allocation and direct indexing instead of Python loops.

    Parameters
    ----------
    results : list of dict
        List of permutation result dictionaries.
    n_permutations : int
        Number of permutations.

    Returns
    -------
    dict
        Dictionary with arrays of shape (n_permutations,) or (n_permutations, n_params).
    """
    group_keys = list(results[0].keys())
    first_val = results[0][group_keys[0]]
    output_shape = first_val.shape if hasattr(first_val, 'shape') else ()

    t_maxes_dict = {
        key: np.empty((n_permutations, *output_shape), dtype=np.float64)
        for key in group_keys
    }

    for key in group_keys:
        for i, perm_dict in enumerate(results):
            t_maxes_dict[key][i] = perm_dict[key]

    return t_maxes_dict


# =============================================================================
# Scoring wrappers for different methods
# =============================================================================

def _score_tfnbs_from_diffs(
    diffs: npt.NDArray[np.float64],
    e: Union[float, List[float]] = DEFAULT_EXTENT_EXPONENT,
    h: Union[float, List[float]] = DEFAULT_HEIGHT_EXPONENT,
    n: int = DEFAULT_N_THRESHOLDS,
    start_thres: float = DEFAULT_START_THRESHOLD,
    **kwargs
) -> Dict[str, npt.NDArray[np.float64]]:
    """Compute TFNBS scores from difference matrices."""
    t_stat_dict = compute_t_stat_diff(diffs)
    score_pos = get_tfnbs_score(t_stat_dict["g2>g1"], e, h, n, start_thres=start_thres)
    score_neg = get_tfnbs_score(t_stat_dict["g1>g2"], e, h, n, start_thres=start_thres)
    return {"g2>g1": score_pos, "g1>g2": score_neg}


def _score_tfnbs_two_sample(
    group1: npt.NDArray[np.float64],
    group2: npt.NDArray[np.float64],
    test_type: str,
    e: Union[float, List[float]] = DEFAULT_EXTENT_EXPONENT,
    h: Union[float, List[float]] = DEFAULT_HEIGHT_EXPONENT,
    n: int = DEFAULT_N_THRESHOLDS,
    **kwargs
) -> Dict[str, npt.NDArray[np.float64]]:
    """Compute TFNBS scores for two-sample test."""
    t_stat_dict = compute_t_stat(group1, group2, test_type=test_type)
    score_pos = get_tfnbs_score(t_stat_dict["g2>g1"], e, h, n)
    score_neg = get_tfnbs_score(t_stat_dict["g1>g2"], e, h, n)
    return {"g2>g1": score_pos, "g1>g2": score_neg}


def _score_nbs_from_diffs(
    diffs: npt.NDArray[np.float64],
    threshold: float = DEFAULT_NBS_THRESHOLD,
    nbs_stat: str = DEFAULT_NBS_STAT,
    **kwargs
) -> Dict[str, npt.NDArray[np.float64]]:
    """Compute NBS scores from difference matrices."""
    t_stat_dict = compute_t_stat_diff(diffs)
    score_pos = get_nbs_score(t_stat_dict["g2>g1"], threshold=threshold, stat_type=nbs_stat)
    score_neg = get_nbs_score(t_stat_dict["g1>g2"], threshold=threshold, stat_type=nbs_stat)
    return {"g2>g1": score_pos, "g1>g2": score_neg}


def _score_nbs_two_sample(
    group1: npt.NDArray[np.float64],
    group2: npt.NDArray[np.float64],
    test_type: str,
    threshold: float = DEFAULT_NBS_THRESHOLD,
    nbs_stat: str = DEFAULT_NBS_STAT,
    **kwargs
) -> Dict[str, npt.NDArray[np.float64]]:
    """Compute NBS scores for two-sample test."""
    t_stat_dict = compute_t_stat(group1, group2, test_type=test_type)
    score_pos = get_nbs_score(t_stat_dict["g2>g1"], threshold=threshold, stat_type=nbs_stat)
    score_neg = get_nbs_score(t_stat_dict["g1>g2"], threshold=threshold, stat_type=nbs_stat)
    return {"g2>g1": score_pos, "g1>g2": score_neg}


def _score_cnbs_from_diffs(
    diffs: npt.NDArray[np.float64],
    net_labels: npt.NDArray[np.int_],
    **kwargs
) -> Dict[str, npt.NDArray[np.float64]]:
    """Compute cNBS scores from difference matrices."""
    t_stat_dict = compute_t_stat_diff(diffs)
    score_pos = get_cnbs_score(t_stat_dict["g2>g1"], net_labels)
    score_neg = get_cnbs_score(t_stat_dict["g1>g2"], net_labels)
    return {"g2>g1": score_pos, "g1>g2": score_neg}


def _score_cnbs_two_sample(
    group1: npt.NDArray[np.float64],
    group2: npt.NDArray[np.float64],
    test_type: str,
    net_labels: npt.NDArray[np.int_],
    **kwargs
) -> Dict[str, npt.NDArray[np.float64]]:
    """Compute cNBS scores for two-sample test."""
    t_stat_dict = compute_t_stat(group1, group2, test_type=test_type)
    score_pos = get_cnbs_score(t_stat_dict["g2>g1"], net_labels)
    score_neg = get_cnbs_score(t_stat_dict["g1>g2"], net_labels)
    return {"g2>g1": score_pos, "g1>g2": score_neg}


def _score_ni_tfnbs_from_diffs(
    diffs: npt.NDArray[np.float64],
    net_labels: npt.NDArray[np.int_],
    e: Union[float, List[float]] = DEFAULT_EXTENT_EXPONENT,
    h: Union[float, List[float]] = DEFAULT_HEIGHT_EXPONENT,
    n: int = DEFAULT_N_THRESHOLDS,
    start_thres: float = DEFAULT_START_THRESHOLD,
    normalization: str = "sqrt",
    **kwargs
) -> Dict[str, npt.NDArray[np.float64]]:
    """Compute NI-TFNBS scores from difference matrices."""
    t_stat_dict = compute_t_stat_diff(diffs)
    score_pos = get_network_informed_tfnbs_score(
        t_stat_dict["g2>g1"], net_labels, e, h, n,
        start_thres=start_thres, normalization=normalization,
    )
    score_neg = get_network_informed_tfnbs_score(
        t_stat_dict["g1>g2"], net_labels, e, h, n,
        start_thres=start_thres, normalization=normalization,
    )
    return {"g2>g1": score_pos, "g1>g2": score_neg}


def _score_ni_tfnbs_two_sample(
    group1: npt.NDArray[np.float64],
    group2: npt.NDArray[np.float64],
    test_type: str,
    net_labels: npt.NDArray[np.int_],
    e: Union[float, List[float]] = DEFAULT_EXTENT_EXPONENT,
    h: Union[float, List[float]] = DEFAULT_HEIGHT_EXPONENT,
    n: int = DEFAULT_N_THRESHOLDS,
    normalization: str = "sqrt",
    **kwargs
) -> Dict[str, npt.NDArray[np.float64]]:
    """Compute NI-TFNBS scores for two-sample test."""
    t_stat_dict = compute_t_stat(group1, group2, test_type=test_type)
    score_pos = get_network_informed_tfnbs_score(
        t_stat_dict["g2>g1"], net_labels, e, h, n, normalization=normalization,
    )
    score_neg = get_network_informed_tfnbs_score(
        t_stat_dict["g1>g2"], net_labels, e, h, n, normalization=normalization,
    )
    return {"g2>g1": score_pos, "g1>g2": score_neg}


def _score_fbc_tfnbs_from_diffs(
    diffs: npt.NDArray[np.float64],
    net_labels: npt.NDArray[np.int_],
    e: Union[float, List[float]] = DEFAULT_EXTENT_EXPONENT,
    h: Union[float, List[float]] = DEFAULT_HEIGHT_EXPONENT,
    n: int = DEFAULT_N_THRESHOLDS,
    start_thres: float = DEFAULT_START_THRESHOLD,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    **kwargs
) -> Dict[str, npt.NDArray[np.float64]]:
    """Compute FBC-TFNBS scores from difference matrices."""
    t_stat_dict = compute_t_stat_diff(diffs)
    score_pos = get_fbc_tfnbs_score(
        t_stat_dict["g2>g1"], net_labels, e, h, n,
        start_thres=start_thres, min_cluster_size=min_cluster_size
    )
    score_neg = get_fbc_tfnbs_score(
        t_stat_dict["g1>g2"], net_labels, e, h, n,
        start_thres=start_thres, min_cluster_size=min_cluster_size
    )
    return {"g2>g1": score_pos, "g1>g2": score_neg}


def _score_fbc_tfnbs_two_sample(
    group1: npt.NDArray[np.float64],
    group2: npt.NDArray[np.float64],
    test_type: str,
    net_labels: npt.NDArray[np.int_],
    e: Union[float, List[float]] = DEFAULT_EXTENT_EXPONENT,
    h: Union[float, List[float]] = DEFAULT_HEIGHT_EXPONENT,
    n: int = DEFAULT_N_THRESHOLDS,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    **kwargs
) -> Dict[str, npt.NDArray[np.float64]]:
    """Compute FBC-TFNBS scores for two-sample test."""
    t_stat_dict = compute_t_stat(group1, group2, test_type=test_type)
    score_pos = get_fbc_tfnbs_score(
        t_stat_dict["g2>g1"], net_labels, e, h, n, min_cluster_size=min_cluster_size
    )
    score_neg = get_fbc_tfnbs_score(
        t_stat_dict["g1>g2"], net_labels, e, h, n, min_cluster_size=min_cluster_size
    )
    return {"g2>g1": score_pos, "g1>g2": score_neg}


# =============================================================================
# Helper functions for multiprocessing
# =============================================================================

def _is_worker_process() -> bool:
    """Check if running inside a multiprocessing worker.

    Returns True when the current process was spawned by a Pool,
    preventing nested Pool creation which causes deadlocks.
    """
    return multiprocessing.current_process().name != 'MainProcess'


def get_available_cores():
    try:
        # Linux
        affinity = os.sched_getaffinity(0)
        return len(affinity)
    except AttributeError:
        # Fallback Windows/Mac
        return multiprocessing.cpu_count()


# =============================================================================
# Null distribution computation
# =============================================================================

def compute_null_dist(
    group1: npt.NDArray[np.float64],
    group2: Optional[npt.NDArray[np.float64]] = None,
    func: Optional[Callable[..., Any]] = None,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    test_type: Union[str, TestType] = TestType.PAIRED,
    random_state: Optional[int] = None,
    n_processes: Optional[int] = None,
    use_mp: bool = True,
    **func_kwargs
) -> Dict[str, npt.NDArray[np.float64]]:
    """
    Compute null distribution of maximum t-statistics via permutation testing.

    Optimized implementation with:

    - Fixed indexing bug in sequential mode
    - Efficient result collection
    - Context-aware multiprocessing (auto-disables inside worker processes)

    Parameters
    ----------
    group1 : ndarray of shape (n_samples_1, *dims)
        Data array for group 1.
    group2 : ndarray of shape (n_samples_2, *dims), optional
        Data array for group 2. Required for 'paired' and 'two-sample' tests.
    func : callable, optional
        Function to compute the t-statistic.
    n_permutations : int, default=1000
        Number of permutations.
    test_type : {'paired', 'one-sample', 'two-sample'} or TestType
        Type of statistical test.
    random_state : int, optional
        Seed for reproducibility.
    n_processes : int, optional
        Number of parallel processes. Defaults to CPU count.
    use_mp : bool, default=True
        Whether to use multiprocessing. Automatically disabled when called
        from inside a multiprocessing worker to prevent nested pools.
    **func_kwargs
        Additional keyword arguments passed to func.

    Returns
    -------
    dict
        Dictionary with 'g1>g2' and 'g2>g1' arrays of shape (n_permutations,)
        or (n_permutations, n_params) for multi-parameter TFCE.

    Raises
    ------
    ValueError
        If test_type is invalid or required group2 is missing.
    """
    # Normalize test_type to string for comparison
    test_type_str = test_type.value if isinstance(test_type, TestType) else test_type

    # Input validation
    if test_type_str in (TestType.PAIRED.value, TestType.TWO_SAMPLE.value):
        if group2 is None:
            raise ValueError(f"group2 is required for test_type='{test_type_str}'")
        if group1.shape[1:] != group2.shape[1:]:
            raise ValueError("Trailing dimensions of group1 and group2 must match.")
        n1, n2 = group1.shape[0], group2.shape[0]
        if n1 < 2 or n2 < 2:
            raise ValueError("Each group must have at least 2 samples.")

    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1.")

    # Prepare data for permutation
    if test_type_str == TestType.PAIRED.value:
        array_to_permute = compute_diffs(group1, group2)
        task_func = partial(_permutation_task_paired, array_to_permute, func, **func_kwargs)

    elif test_type_str == TestType.TWO_SAMPLE.value:
        array_to_permute = np.concatenate((group1, group2), axis=0)
        task_func = partial(_permutation_task_ind, array_to_permute, func, n1, **func_kwargs)

    elif test_type_str == TestType.ONE_SAMPLE.value:
        group2_zeros = np.zeros(group1.shape)
        array_to_permute = compute_diffs(group1, group2_zeros)
        task_func = partial(_permutation_task_paired, array_to_permute, func, **func_kwargs)
    else:
        raise ValueError(
            f"Invalid test_type: '{test_type_str}'. "
            f"Must be one of: {[t.value for t in TestType]}"
        )

    # Generate seeds for reproducibility
    rng = np.random.RandomState(random_state)
    seeds = rng.randint(0, 2**32 - 1, size=n_permutations, dtype=np.int64)
    
    # Context-aware multiprocessing: disable inside worker processes
    # to prevent nested pools that cause deadlocks
    _use_mp = use_mp and not _is_worker_process()

    if _use_mp:
        if n_processes is None:
            n_processes = get_available_cores()
        n_processes = min(n_processes, n_permutations)

        with Pool(processes=n_processes) as pool:
            results = pool.map(task_func, seeds)
    else:
        # Sequential computation
        results = [task_func(seed) for seed in seeds]

    return _collect_results_to_arrays(results, n_permutations)


# =============================================================================
# Per-edge permutation helpers (for BH-FDR-perm)
# =============================================================================

def _permutation_task_ind_full(
    full_group: npt.NDArray[np.float64],
    func: Callable[..., Any],
    n1: int,
    seed: int,
    **func_kwargs
) -> Dict[str, npt.NDArray[np.float64]]:
    """Return full per-edge t-stats for a single permutation (two-sample)."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(full_group.shape[0])
    new_group1 = full_group[idx[:n1]]
    new_group2 = full_group[idx[n1:]]
    return func(new_group1, new_group2, test_type='two-sample', **func_kwargs)


def _permutation_task_paired_full(
    diffs: npt.NDArray[np.float64],
    func: Callable[..., Any],
    seed: Optional[int] = None,
    **func_kwargs
) -> Dict[str, npt.NDArray[np.float64]]:
    """Return full per-edge t-stats for a single permutation (paired/one-sample)."""
    n_dims = len(diffs.shape) - 1
    faked_dims = [1] * n_dims
    rng = np.random.RandomState(seed)
    signs = rng.choice([1, -1], diffs.shape[0]).reshape(-1, *faked_dims)
    new_diffs = signs * diffs
    return func(new_diffs, **func_kwargs)


def _compute_bh_fdr_perm_p_values(
    group1: npt.NDArray[np.float64],
    group2: Optional[npt.NDArray[np.float64]],
    test_type_str: str,
    n_permutations: int,
    random_state: Optional[int],
    use_mp: bool,
    n_processes: Optional[int],
) -> Dict[str, npt.NDArray[np.float64]]:
    """
    Compute BH-FDR corrected p-values using permutation-based per-edge nulls.

    1. Compute observed t-stats per edge.
    2. For each permutation, compute per-edge t-stats.
    3. For each edge, p = (# perm t >= observed t) / n_perm.
    4. Apply BH-FDR correction to per-edge p-values.
    """
    # Compute observed t-stats
    if test_type_str == TestType.PAIRED.value:
        diffs = compute_diffs(group1, group2)
        emp_t_dict = compute_t_stat_diff(diffs)
    elif test_type_str == TestType.ONE_SAMPLE.value:
        emp_t_dict = compute_t_stat_diff(group1)
    elif test_type_str == TestType.TWO_SAMPLE.value:
        emp_t_dict = compute_t_stat(group1, group2, test_type=test_type_str)
    else:
        raise ValueError(f"Invalid test_type: '{test_type_str}'")

    N = emp_t_dict["g2>g1"].shape[0]
    triu_idx = np.triu_indices(N, k=1)
    n_edges = len(triu_idx[0])

    # Generate seeds
    rng = np.random.RandomState(random_state)
    seeds = rng.randint(0, 2**32 - 1, size=n_permutations, dtype=np.int64)

    # Prepare permutation task function (returns full t-stat matrices)
    if test_type_str == TestType.PAIRED.value:
        diffs = compute_diffs(group1, group2)
        task_func = partial(
            _permutation_task_paired_full, diffs, compute_t_stat_diff
        )
    elif test_type_str == TestType.ONE_SAMPLE.value:
        group2_zeros = np.zeros(group1.shape)
        diffs = compute_diffs(group1, group2_zeros)
        task_func = partial(
            _permutation_task_paired_full, diffs, compute_t_stat_diff
        )
    else:  # two-sample
        full_group = np.concatenate((group1, group2), axis=0)
        n1 = group1.shape[0]
        task_func = partial(
            _permutation_task_ind_full, full_group, compute_t_stat, n1
        )

    _use_mp = use_mp and not _is_worker_process()

    if _use_mp:
        if n_processes is None:
            n_processes = get_available_cores()
        n_processes = min(n_processes, n_permutations)
        with Pool(processes=n_processes) as pool:
            perm_results = pool.map(task_func, seeds)
    else:
        perm_results = [task_func(seed) for seed in seeds]

    # Compute per-edge p-values and apply BH correction
    p_values = {}
    for key in ("g2>g1", "g1>g2"):
        emp_upper = emp_t_dict[key][triu_idx]

        # Count how many permutation t-stats >= observed for each edge
        count_ge = np.zeros(n_edges, dtype=np.float64)
        for perm_dict in perm_results:
            perm_upper = perm_dict[key][triu_idx]
            count_ge += (perm_upper >= emp_upper).astype(np.float64)

        per_edge_p = count_ge / n_permutations

        # Apply BH-FDR correction
        corrected_p = _bh_fdr_correction(per_edge_p)

        # Reconstruct full symmetric matrix
        p_mat = np.ones((N, N), dtype=np.float64)
        p_mat[triu_idx] = corrected_p
        p_mat[(triu_idx[1], triu_idx[0])] = corrected_p
        p_values[key] = p_mat

    return p_values


# =============================================================================
# P-value computation
# =============================================================================

def _compute_p_values_from_null(
    emp_t_dict: Dict[str, npt.NDArray],
    max_null_dict: Dict[str, npt.NDArray]
) -> Dict[str, npt.NDArray]:
    """
    Compute p-values by comparing empirical statistics to null distribution.

    Parameters
    ----------
    emp_t_dict : dict
        Dictionary with empirical t-statistic arrays.
    max_null_dict : dict
        Dictionary with null distribution arrays.

    Returns
    -------
    dict
        Dictionary with p-value arrays for each direction.
    """
    keys = list(emp_t_dict.keys())
    p_values = {}

    is_2d = len(emp_t_dict[keys[0]].shape) == 2

    for key in keys:
        emp_t = emp_t_dict[key]
        null_dist = max_null_dict[key]

        if is_2d:
            # Shape: (N, N) vs (n_permutations,)
            emp_t_expanded = emp_t[..., np.newaxis]
            p_values[key] = np.mean(emp_t_expanded < null_dist, axis=-1)
        else:
            # Multi-param: (N, N, n_params) vs (n_permutations, n_params)
            emp_t_expanded = emp_t[..., np.newaxis]
            null_reshaped = null_dist.swapaxes(0, 1)[None, None, ...]
            p_values[key] = np.mean(emp_t_expanded < null_reshaped, axis=-1)

    return p_values


def _compute_degrees_of_freedom(
    n1: int,
    n2: int,
    test_type_str: str
) -> int:
    """Compute degrees of freedom for a t-test.

    Parameters
    ----------
    n1 : int
        Number of samples in group 1.
    n2 : int
        Number of samples in group 2 (0 for one-sample).
    test_type_str : str
        Test type string ('paired', 'one-sample', 'two-sample').

    Returns
    -------
    int
        Degrees of freedom.
    """
    if test_type_str == TestType.TWO_SAMPLE.value:
        return n1 + n2 - 2
    else:
        # paired or one-sample: df = n - 1
        return n1 - 1


def _compute_parametric_p_values(
    group1: npt.NDArray[np.float64],
    group2: Optional[npt.NDArray[np.float64]],
    test_type_str: str,
    method_enum: "StatMethod",
    alpha: float = 0.05,
) -> Dict[str, npt.NDArray[np.float64]]:
    """
    Compute parametric p-values with Bonferroni or BH-FDR correction.

    Parameters
    ----------
    group1 : ndarray of shape (n_subjects, N, N)
        Connectivity matrices for group 1.
    group2 : ndarray of shape (n_subjects, N, N), optional
        Connectivity matrices for group 2.
    test_type_str : str
        Test type ('paired', 'one-sample', 'two-sample').
    method_enum : StatMethod
        BONFERRONI or BH_FDR.
    alpha : float, default=0.05
        Significance level (used for BH-FDR step-up).

    Returns
    -------
    dict
        Dictionary with 'g1>g2' and 'g2>g1' p-value arrays of shape (N, N).
    """
    # Compute t-statistics using existing infrastructure
    if test_type_str == TestType.PAIRED.value:
        diffs = compute_diffs(group1, group2)
        t_dict = compute_t_stat_diff(diffs)
        df = _compute_degrees_of_freedom(group1.shape[0], 0, test_type_str)
    elif test_type_str == TestType.ONE_SAMPLE.value:
        t_dict = compute_t_stat_diff(group1)
        df = _compute_degrees_of_freedom(group1.shape[0], 0, test_type_str)
    elif test_type_str == TestType.TWO_SAMPLE.value:
        t_dict = compute_t_stat_ind(group1, group2)
        df = _compute_degrees_of_freedom(group1.shape[0], group2.shape[0], test_type_str)
    else:
        raise ValueError(f"Invalid test_type: '{test_type_str}'")

    p_values = {}
    for key in ("g1>g2", "g2>g1"):
        t_vals = t_dict[key]  # Non-negative (one-tailed)
        N = t_vals.shape[0]

        # Compute one-tailed p-values from t-distribution
        raw_p = stats.t.sf(t_vals, df)

        # Extract upper triangle (unique edges for symmetric matrices)
        triu_idx = np.triu_indices(N, k=1)
        p_upper = raw_p[triu_idx]

        # Number of unique comparisons
        m = len(p_upper)

        if method_enum == StatMethod.BONFERRONI:
            # Bonferroni: multiply by number of comparisons, cap at 1.0
            p_corrected_upper = np.minimum(p_upper * m, 1.0)
        elif method_enum == StatMethod.BH_FDR:
            # Benjamini-Hochberg step-up procedure
            p_corrected_upper = _bh_fdr_correction(p_upper)
        else:
            raise ValueError(f"Unsupported parametric method: {method_enum}")

        # Reconstruct full symmetric matrix
        p_corrected = np.ones((N, N), dtype=np.float64)
        p_corrected[triu_idx] = p_corrected_upper
        p_corrected[(triu_idx[1], triu_idx[0])] = p_corrected_upper

        p_values[key] = p_corrected

    return p_values


def _bh_fdr_correction(
    p_values: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Apply Benjamini-Hochberg FDR correction to a 1D array of p-values.

    Parameters
    ----------
    p_values : ndarray of shape (m,)
        Uncorrected p-values.

    Returns
    -------
    ndarray of shape (m,)
        BH-corrected p-values (adjusted so that thresholding at alpha
        controls FDR at level alpha).
    """
    m = len(p_values)
    if m == 0:
        return p_values.copy()

    # Sort p-values and track original indices
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    # BH adjustment: p_adj[i] = p[i] * m / rank[i]
    ranks = np.arange(1, m + 1, dtype=np.float64)
    adjusted = sorted_p * m / ranks

    # Enforce monotonicity (step-up): working backwards,
    # each adjusted p-value must be <= the next one
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]

    # Cap at 1.0
    adjusted = np.minimum(adjusted, 1.0)

    # Restore original order
    result = np.empty(m, dtype=np.float64)
    result[sorted_idx] = adjusted

    return result


def compute_p_val(
    group1: npt.NDArray[np.float64],
    group2: Optional[npt.NDArray[np.float64]] = None,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    test_type: Union[str, TestType] = TestType.PAIRED,
    method: Union[str, StatMethod] = StatMethod.TFNBS,
    use_mp: bool = True,
    random_state: Optional[int] = None,
    n_processes: Optional[int] = None,
    acceleration: Optional[str] = None,
    # Method-specific parameters
    net_labels: Optional[npt.NDArray[np.int_]] = None,
    threshold: float = DEFAULT_NBS_THRESHOLD,
    nbs_stat: str = DEFAULT_NBS_STAT,
    e: Union[float, List[float]] = DEFAULT_EXTENT_EXPONENT,
    h: Union[float, List[float]] = DEFAULT_HEIGHT_EXPONENT,
    n: int = DEFAULT_N_THRESHOLDS,
    start_thres: float = DEFAULT_START_THRESHOLD,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    normalization: str = "sqrt",
    **kwargs
) -> Dict[str, npt.NDArray[np.float64]]:
    """
    Compute p-values using permutation testing with various network-based methods.

    Supports multiple statistical methods: tfnbs, tstat, nbs, cnbs, ni_tfnbs,
    fbc_tfnbs, bonferroni, bh_fdr, bh_fdr_perm.

    .. note:: **cNBS null distribution**

       This implementation uses the **max-statistic** null distribution for cNBS,
       where each permutation contributes its global maximum cNBS score to the
       null. This is the same family-wise error rate (FWER) control strategy used
       by classical NBS (Zalesky et al. 2010) and TFNBS.

       Noble & Scheinost (2020) originally proposed computing **per-block** null
       distributions with Bonferroni correction across blocks. The max-statistic
       approach used here is more conservative (controls FWER globally) but
       provides a consistent framework across all methods in this package.

    .. note:: **Bonferroni and BH-FDR methods**

       ``StatMethod.BONFERRONI`` and ``StatMethod.BH_FDR`` are parametric baselines
       that do **not** use permutation testing. They compute p-values from the
       t-distribution and apply multiple comparison corrections. The
       ``n_permutations`` parameter is ignored for these methods.

    Parameters
    ----------
    group1 : ndarray of shape (n_subjects_g1, N, N)
        Input connectivity matrices for group 1.
    group2 : ndarray of shape (n_subjects_g2, N, N), optional
        Input connectivity matrices for group 2. Required for paired/two-sample tests.
    n_permutations : int, default=1000
        Number of permutations for null distribution.
    test_type : {'paired', 'one-sample', 'two-sample'} or TestType, default='paired'
        Type of statistical test.
    method : {'tfnbs', 'tstat', 'nbs', 'cnbs', 'ni_tfnbs', 'fbc_tfnbs'} or StatMethod, default='tfnbs'
        Statistical method to use for scoring.
    use_mp : bool, default=True
        Use multiprocessing for permutation testing. Automatically disabled
        when called from inside a multiprocessing worker to prevent deadlocks.
    random_state : int, optional
        Random seed for reproducibility.
    n_processes : int, optional
        Number of CPU cores for parallel computing.
    net_labels : ndarray of shape (N,), optional
        Network labels for each node. Required for cnbs, ni_tfnbs, and fbc_tfnbs.
    threshold : float, default=2.0
        T-statistic threshold for NBS (only used when method='nbs').
    nbs_stat : {'extent', 'intensity'}, default='extent'
        Cluster statistic for NBS (only used when method='nbs').
    e : float or list, default=0.4
        Extent exponent for TFNBS-based methods.
    h : float or list, default=3.0
        Height exponent for TFNBS-based methods.
    n : int, default=10
        Number of threshold steps for TFNBS-based methods.
    start_thres : float, default=1.65
        Starting threshold for TFNBS integration.
    min_cluster_size : int, default=3
        Minimum cluster size for FBC-TFNBS (only used when method='fbc_tfnbs').
    normalization : {'sqrt', 'linear', 'none'}, default='sqrt'
        Block density normalization for NI-TFNBS (only used when method='ni_tfnbs').
    **kwargs
        Additional keyword arguments (for future extensions).

    Returns
    -------
    dict
        Dictionary with p-value arrays:

        - 'g1>g2': P-values for group 1 > group 2.
        - 'g2>g1': P-values for group 2 > group 1.

    Raises
    ------
    ValueError
        If constrained methods (cnbs, ni_tfnbs, fbc_tfnbs) are used without net_labels.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(2)
    >>> group1 = np.random.rand(5, 3, 3)
    >>> for arr in group1: np.fill_diagonal(arr, 0)
    >>> group2 = np.random.rand(8, 3, 3)
    >>> for arr in group2: np.fill_diagonal(arr, 0)
    >>> # Standard t-test
    >>> p_vals = compute_p_val(group1, group2, n_permutations=10,
    ...                        test_type='two-sample', method='tstat',
    ...                        use_mp=False, random_state=0)
    >>> # TFNBS
    >>> p_vals = compute_p_val(group1, group2, n_permutations=10,
    ...                        test_type='two-sample', method='tfnbs',
    ...                        use_mp=False, random_state=0)
    >>> # cNBS with network labels
    >>> labels = np.array([0, 0, 1])
    >>> p_vals = compute_p_val(group1, group2, n_permutations=10,
    ...                        test_type='two-sample', method='cnbs',
    ...                        net_labels=labels, use_mp=False, random_state=0)
    """
    # Normalize inputs
    test_type_str = test_type.value if isinstance(test_type, TestType) else test_type
    method_str = method.value if isinstance(method, StatMethod) else method

    # Validate method
    try:
        method_enum = StatMethod(method_str)
    except ValueError:
        valid_methods = [m.value for m in StatMethod]
        raise ValueError(
            f"Invalid method: '{method_str}'. Must be one of: {valid_methods}"
        )

    # Validate constrained methods require net_labels
    if method_enum in CONSTRAINED_METHODS and net_labels is None:
        raise ValueError(
            f"Method '{method_str}' requires net_labels to be provided. "
            f"Constrained methods are: {[m.value for m in CONSTRAINED_METHODS]}"
        )

    # Parametric methods: compute p-values directly from t-distribution
    if method_enum in PARAMETRIC_METHODS:
        return _compute_parametric_p_values(
            group1, group2, test_type_str, method_enum
        )

    # BH-FDR with permutation p-values: separate code path
    if method_enum == StatMethod.BH_FDR_PERM:
        return _compute_bh_fdr_perm_p_values(
            group1, group2, test_type_str,
            n_permutations=n_permutations,
            random_state=random_state,
            use_mp=use_mp,
            n_processes=n_processes,
        )

    # Select appropriate scorer function based on method and test type
    if test_type_str == TestType.PAIRED.value:
        diffs = compute_diffs(group1, group2)
        scorer_map = {
            StatMethod.TSTAT: compute_t_stat_diff,
            StatMethod.TFNBS: _score_tfnbs_from_diffs,
            StatMethod.NBS: _score_nbs_from_diffs,
            StatMethod.CNBS: _score_cnbs_from_diffs,
            StatMethod.NI_TFNBS: _score_ni_tfnbs_from_diffs,
            StatMethod.FBC_TFNBS: _score_fbc_tfnbs_from_diffs,
        }
        t_func = scorer_map[method_enum]

        # Build kwargs for scorer
        scorer_kwargs = {}
        if method_enum == StatMethod.NBS:
            scorer_kwargs = {"threshold": threshold, "nbs_stat": nbs_stat}
        elif method_enum in {StatMethod.TFNBS, StatMethod.NI_TFNBS, StatMethod.FBC_TFNBS}:
            scorer_kwargs = {"e": e, "h": h, "n": n, "start_thres": start_thres}
            if method_enum in CONSTRAINED_METHODS:
                scorer_kwargs["net_labels"] = net_labels
            if method_enum == StatMethod.FBC_TFNBS:
                scorer_kwargs["min_cluster_size"] = min_cluster_size
            if method_enum == StatMethod.NI_TFNBS:
                scorer_kwargs["normalization"] = normalization
        elif method_enum == StatMethod.CNBS:
            scorer_kwargs = {"net_labels": net_labels}

        emp_t_dict = t_func(diffs, **scorer_kwargs)

    elif test_type_str == TestType.TWO_SAMPLE.value:
        scorer_map = {
            StatMethod.TSTAT: compute_t_stat,
            StatMethod.TFNBS: _score_tfnbs_two_sample,
            StatMethod.NBS: _score_nbs_two_sample,
            StatMethod.CNBS: _score_cnbs_two_sample,
            StatMethod.NI_TFNBS: _score_ni_tfnbs_two_sample,
            StatMethod.FBC_TFNBS: _score_fbc_tfnbs_two_sample,
        }
        t_func = scorer_map[method_enum]

        # Build kwargs for scorer
        scorer_kwargs = {"test_type": TestType.TWO_SAMPLE.value}
        if method_enum == StatMethod.NBS:
            scorer_kwargs.update({"threshold": threshold, "nbs_stat": nbs_stat})
        elif method_enum in {StatMethod.TFNBS, StatMethod.NI_TFNBS, StatMethod.FBC_TFNBS}:
            scorer_kwargs.update({"e": e, "h": h, "n": n})
            if method_enum in CONSTRAINED_METHODS:
                scorer_kwargs["net_labels"] = net_labels
            if method_enum == StatMethod.FBC_TFNBS:
                scorer_kwargs["min_cluster_size"] = min_cluster_size
            if method_enum == StatMethod.NI_TFNBS:
                scorer_kwargs["normalization"] = normalization
        elif method_enum == StatMethod.CNBS:
            scorer_kwargs["net_labels"] = net_labels

        emp_t_dict = t_func(group1, group2, **scorer_kwargs)

    elif test_type_str == TestType.ONE_SAMPLE.value:
        scorer_map = {
            StatMethod.TSTAT: compute_t_stat_diff,
            StatMethod.TFNBS: _score_tfnbs_from_diffs,
            StatMethod.NBS: _score_nbs_from_diffs,
            StatMethod.CNBS: _score_cnbs_from_diffs,
            StatMethod.NI_TFNBS: _score_ni_tfnbs_from_diffs,
            StatMethod.FBC_TFNBS: _score_fbc_tfnbs_from_diffs,
        }
        t_func = scorer_map[method_enum]

        # Build kwargs for scorer
        scorer_kwargs = {}
        if method_enum == StatMethod.NBS:
            scorer_kwargs = {"threshold": threshold, "nbs_stat": nbs_stat}
        elif method_enum in {StatMethod.TFNBS, StatMethod.NI_TFNBS, StatMethod.FBC_TFNBS}:
            scorer_kwargs = {"e": e, "h": h, "n": n, "start_thres": start_thres}
            if method_enum in CONSTRAINED_METHODS:
                scorer_kwargs["net_labels"] = net_labels
            if method_enum == StatMethod.FBC_TFNBS:
                scorer_kwargs["min_cluster_size"] = min_cluster_size
            if method_enum == StatMethod.NI_TFNBS:
                scorer_kwargs["normalization"] = normalization
        elif method_enum == StatMethod.CNBS:
            scorer_kwargs = {"net_labels": net_labels}

        emp_t_dict = t_func(group1, **scorer_kwargs)
    else:
        raise ValueError(
            f"Invalid test_type: '{test_type_str}'. "
            f"Must be one of: {[t.value for t in TestType]}"
        )

    # Compute null distribution
    group2_for_null = group2 if test_type_str != TestType.ONE_SAMPLE.value else None

    # Remove test_type from scorer_kwargs if present (compute_null_dist has it as explicit param)
    null_kwargs = {k: v for k, v in scorer_kwargs.items() if k != 'test_type'}

    max_null_dict = compute_null_dist(
        group1, group2_for_null, t_func,
        n_permutations=n_permutations,
        test_type=test_type,
        use_mp=use_mp,
        random_state=random_state,
        n_processes=n_processes,
        **null_kwargs
    )

    if acceleration is not None:
        return compute_p_values_accelerated(
            emp_t_dict, max_null_dict, method=acceleration,
        )
    return _compute_p_values_from_null(emp_t_dict, max_null_dict)


# =============================================================================
# T-statistic computation functions
# =============================================================================

def compute_t_stat_tfnbs(
    group1: npt.NDArray[np.float64],
    group2: npt.NDArray[np.float64],
    test_type: Union[str, TestType] = TestType.TWO_SAMPLE,
    e: Union[float, List[float]] = DEFAULT_EXTENT_EXPONENT,
    h: Union[float, List[float]] = DEFAULT_HEIGHT_EXPONENT,
    n: int = DEFAULT_N_THRESHOLDS
) -> Dict[str, npt.NDArray[np.float64]]:
    """
    Compute TFNBS-enhanced t-statistics for independent groups.

    Returns separate scores for positive (g2 > g1) and negative (g1 > g2) effects.

    Parameters
    ----------
    group1 : ndarray of shape (n_samples_1, N, N)
        Data array for group 1.
    group2 : ndarray of shape (n_samples_2, N, N)
        Data array for group 2.
    test_type : {'two-sample'} or TestType, default='two-sample'
        Statistical test type.
    e : float or list, default=0.4
        Extent exponent for TFNBS.
    h : float or list, default=3
        Height exponent for TFNBS.
    n : int, default=10
        Number of integration steps.

    Returns
    -------
    dict
        Dictionary with 'g2>g1' and 'g1>g2' TFNBS score arrays.
    """
    t_stat_dict = compute_t_stat(group1, group2, test_type=test_type)
    score_pos = get_tfnbs_score(t_stat_dict["g2>g1"], e, h, n)
    score_neg = get_tfnbs_score(t_stat_dict["g1>g2"], e, h, n)
    return {"g2>g1": score_pos, "g1>g2": score_neg}


def compute_t_stat_tfnbs_diffs(
    diffs: npt.NDArray[np.float64],
    e: Union[float, List[float]] = DEFAULT_EXTENT_EXPONENT,
    h: Union[float, List[float]] = DEFAULT_HEIGHT_EXPONENT,
    n: int = DEFAULT_N_THRESHOLDS,
    start_thres: float = DEFAULT_START_THRESHOLD
) -> Dict[str, npt.NDArray[np.float64]]:
    """
    Compute TFNBS-enhanced t-statistics from difference matrices.

    Returns separate scores for positive (g2 > g1) and negative (g1 > g2) effects.

    Parameters
    ----------
    diffs : ndarray of shape (n_samples, N, N)
        Array of pairwise differences.
    e : float or list, default=0.4
        Extent exponent for TFNBS.
    h : float or list, default=3
        Height exponent for TFNBS.
    n : int, default=10
        Number of integration steps.
    start_thres : float, default=1.65
        Initial threshold for TFNBS.

    Returns
    -------
    dict
        Dictionary with 'g2>g1' and 'g1>g2' TFNBS score arrays.
    """
    t_stat_dict = compute_t_stat_diff(diffs)
    score_pos = get_tfnbs_score(t_stat_dict["g2>g1"], e, h, n, start_thres=start_thres)
    score_neg = get_tfnbs_score(t_stat_dict["g1>g2"], e, h, n, start_thres=start_thres)
    return {"g2>g1": score_pos, "g1>g2": score_neg}


def compute_t_stat(
    group1: npt.NDArray[np.float64],
    group2: Optional[npt.NDArray[np.float64]] = None,
    test_type: Union[str, TestType] = TestType.PAIRED
) -> Dict[str, npt.NDArray[np.float64]]:
    """
    Compute t-statistics for paired, one-sample, or two-sample tests.

    Parameters
    ----------
    group1 : ndarray of shape (n_samples_1, N, N)
        Data array for group 1.
    group2 : ndarray of shape (n_samples_2, N, N), optional
        Data array for group 2. Required for paired/two-sample tests.
    test_type : {'paired', 'one-sample', 'two-sample'} or TestType
        Type of statistical test.

    Returns
    -------
    dict
        Dictionary with 'g2>g1' and 'g1>g2' t-statistic arrays.

    Raises
    ------
    ValueError
        If test_type is invalid or dimensions don't match.
    """
    # Normalize test_type to string
    test_type_str = test_type.value if isinstance(test_type, TestType) else test_type

    if test_type_str == TestType.ONE_SAMPLE.value:
        if group2 is not None and not isinstance(group2, int):
            logger.warning("Group 1 input will be considered for one-sample T-test.")
        if group1.ndim != 3:
            raise ValueError("Dimensions of group 1 data should be: (subjects, N, N).")
        return compute_t_stat_diff(group1)

    elif test_type_str == TestType.TWO_SAMPLE.value:
        if group1.shape[1:] != group2.shape[1:]:
            raise ValueError("Trailing dimensions of group1 and group2 must match.")
        return compute_t_stat_ind(group1, group2)

    elif test_type_str == TestType.PAIRED.value:
        diffs = compute_diffs(group1, group2)
        return compute_t_stat_diff(diffs)

    else:
        raise ValueError(
            f"Invalid test_type: '{test_type_str}'. "
            f"Must be one of: {[t.value for t in TestType]}"
        )


def compute_diffs(
    group1: npt.NDArray[np.float64],
    group2: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Compute differences between paired samples (group2 - group1).

    Parameters
    ----------
    group1 : ndarray of shape (n_samples, *dims)
        Data array for group 1.
    group2 : ndarray of shape (n_samples, *dims)
        Data array for group 2.

    Returns
    -------
    ndarray
        Array of differences with same shape.
    """
    return group2 - group1


def compute_t_stat_diff(
    diff: npt.NDArray[np.float64]
) -> Dict[str, npt.NDArray[np.float64]]:
    """
    Compute t-statistics for paired differences.

    Parameters
    ----------
    diff : ndarray of shape (n_samples, *dims)
        Array containing paired differences.

    Returns
    -------
    dict
        Dictionary with:

        - 'g2>g1': Positive t-values (where group 2 > group 1).
        - 'g1>g2': Negative t-values converted to positive.

    Raises
    ------
    ValueError
        If fewer than 2 samples are provided.
    """
    n = diff.shape[0]
    if n < 2:
        raise ValueError("At least 2 samples required for t-statistic.")

    x_mean = np.mean(diff, axis=0)
    x_std = np.std(diff, axis=0, ddof=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        t_stat = x_mean / (x_std / np.sqrt(n))
        t_stat = np.where(x_std == 0, 0, t_stat)

    pos_t = np.where(t_stat > 0, t_stat, 0)
    neg_t = np.where(t_stat < 0, -t_stat, 0)

    return {"g2>g1": pos_t, "g1>g2": neg_t}


def compute_t_stat_ind(
    group1: npt.NDArray[np.float64],
    group2: npt.NDArray[np.float64]
) -> Dict[str, npt.NDArray[np.float64]]:
    """
    Compute Welch's t-statistics for independent samples.

    Parameters
    ----------
    group1 : ndarray of shape (n_samples_1, *dims)
        Data array for group 1.
    group2 : ndarray of shape (n_samples_2, *dims)
        Data array for group 2.

    Returns
    -------
    dict
        Dictionary with:

        - 'g2>g1': Positive t-values (where group 2 > group 1).
        - 'g1>g2': Negative t-values converted to positive.

    Raises
    ------
    ValueError
        If either group has fewer than 2 samples.
    """
    n1, n2 = group1.shape[0], group2.shape[0]
    if n1 < 2 or n2 < 2:
        raise ValueError("Each group must have at least 2 samples.")

    x_mean_1 = np.mean(group1, axis=0)
    x_mean_2 = np.mean(group2, axis=0)
    x_var_1 = np.var(group1, axis=0, ddof=1) / n1
    x_var_2 = np.var(group2, axis=0, ddof=1) / n2

    denominator = np.sqrt(x_var_1 + x_var_2)

    with np.errstate(divide='ignore', invalid='ignore'):
        t_stat = (x_mean_2 - x_mean_1) / denominator
        t_stat = np.where(denominator == 0, 0, t_stat)

    pos_t = np.where(t_stat > 0, t_stat, 0)
    neg_t = np.where(t_stat < 0, -t_stat, 0)

    return {"g2>g1": pos_t, "g1>g2": neg_t}

