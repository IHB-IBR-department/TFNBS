"""
Pairwise statistical testing module for network analysis.

This module provides functions for computing t-statistics and p-values
using permutation testing, with optional enhancement (TFNBS, NBS, cNBS,
NI-TFNBS, FBC-TFNBS) applied via shared ``apply_*`` wrappers from
``conninfpy._enhancement``.

Main Functions
--------------
compute_p_val : Compute p-values using permutation testing (with enhancement)
compute_t_stat : Compute t-statistics for paired, one-sample, or two-sample tests
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
from ._enhancement import (
    apply_cnbs,
    apply_fbc_tfnbs,
    apply_nbs,
    apply_ni_tfnbs,
    apply_tfnbs,
)
import time

from .acceleration import compute_p_values_accelerated
from ._compat import TailResult, make_tail_result, normalize_keys
from ._progress import run_permutations
from ._result import InferenceResult, make_inference_result
from ._rng import RngLike, resolve_seed, warn_legacy_random_state


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
    "compute_null_dist",
    # T-statistic primitive (accepts pre-computed diffs)
    "compute_t_stat_diff",
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

# Enhancement-function registry. Maps StatMethod → pure `stat_dict → stat_dict`
# wrapper from ._enhancement (shared with glm_stats.compute_p_val_glm). Methods
# without enhancement (raw t-stat, parametric baselines, bh_fdr_perm) map to None.
_ENHANCE_MAP = {
    StatMethod.TSTAT: None,
    StatMethod.TFNBS: apply_tfnbs,
    StatMethod.NBS: apply_nbs,
    StatMethod.CNBS: apply_cnbs,
    StatMethod.NI_TFNBS: apply_ni_tfnbs,
    StatMethod.FBC_TFNBS: apply_fbc_tfnbs,
    StatMethod.BONFERRONI: None,
    StatMethod.BH_FDR: None,
    StatMethod.BH_FDR_PERM: None,
}
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
    Key-agnostic: works with any dictionary keys (e.g., 'negative'/'positive'
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
    result: Dict[str, Any] = {}
    for key, arr in stat_dict.items():
        if arr.shape == reference_shape:
            result[key] = np.max(arr).astype(np.float64)
        else:
            # Multi-parameter case: max over spatial dims, keep param dim
            spatial_axes = tuple(range(arr.ndim - 1))
            result[key] = np.max(arr, axis=spatial_axes).astype(np.float64)
    if set(result.keys()) == {"positive", "negative"}:
        return make_tail_result(result["positive"], result["negative"])
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
        Dictionary with max statistics for 'negative' and 'positive' directions.
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
        Dictionary with max statistics for 'negative' and 'positive' directions.
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

    if set(t_maxes_dict.keys()) == {"positive", "negative"}:
        return make_tail_result(t_maxes_dict["positive"], t_maxes_dict["negative"])
    return t_maxes_dict


# =============================================================================
# Pre-computed sums fast path (method='tstat' and 'bh_fdr_perm')
# =============================================================================
# Exploits the algebraic variance identity Var(x) = (Σx² − (Σx)²/n)/(n−1) with
# the sign-flip-invariant Σx² pre-computed once, so each permutation reduces to
# a matrix-vector dot product over upper-triangle edges (~6-8x faster per perm
# vs. recomputing mean/std from full (nSub, N, N) arrays).


def _onesample_tstat_from_sums(sum_vec, sumsq_vec, n):
    """One-sample t-statistic from pre-computed sums. Returns (t_pos, t_neg) edge vectors."""
    mean = sum_vec / n
    var = np.maximum((sumsq_vec - sum_vec ** 2 / n) / (n - 1), 0)
    se = np.sqrt(var / n)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = mean / se
        t = np.where(se == 0, 0.0, t)
    return np.maximum(t, 0.0), np.maximum(-t, 0.0)


def _twosample_tstat_from_sums(sum1, sumsq1, n1, sum2, sumsq2, n2):
    """Welch two-sample t-statistic from pre-computed sums. Returns (t_pos, t_neg) edge vectors."""
    mean1 = sum1 / n1
    mean2 = sum2 / n2
    var1 = np.maximum((sumsq1 - sum1 ** 2 / n1) / (n1 - 1), 0)
    var2 = np.maximum((sumsq2 - sum2 ** 2 / n2) / (n2 - 1), 0)
    denom = np.sqrt(var1 / n1 + var2 / n2)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (mean2 - mean1) / denom
        t = np.where(denom == 0, 0.0, t)
    return np.maximum(t, 0.0), np.maximum(-t, 0.0)


def _precompute_edge_sums(data_3d):
    """Extract upper-triangle edges + sum-of-squares for sign-flip fast path.

    Returns (X, sumsq_all): X is (n_samples, n_edges) contiguous; sumsq_all is
    (n_edges,) and is sign-flip invariant since s² = 1 for s ∈ {+1, −1}.
    """
    N = data_3d.shape[1]
    triu_idx = np.triu_indices(N, k=1)
    X = np.ascontiguousarray(data_3d[:, triu_idx[0], triu_idx[1]])
    sumsq_all = np.sum(X ** 2, axis=0)
    return X, sumsq_all


def _precompute_twosample_sums(Xall_3d):
    """Extract upper-triangle edges + pooled sums for group-label fast path.

    Returns (Xall, Xall2, sum_all, sumsq_all). Xall2 caches element-wise square
    so per-permutation cost is two matvecs + two subtractions.
    """
    N = Xall_3d.shape[1]
    triu_idx = np.triu_indices(N, k=1)
    Xall = np.ascontiguousarray(Xall_3d[:, triu_idx[0], triu_idx[1]])
    Xall2 = Xall ** 2
    sum_all = np.sum(Xall, axis=0)
    sumsq_all = np.sum(Xall2, axis=0)
    return Xall, Xall2, sum_all, sumsq_all


def _fast_permutation_task_paired(X, sumsq_all, seed):
    """Paired/one-sample permutation via sign-flip + sums. Returns max stats."""
    rng = np.random.RandomState(seed)
    n_sub = X.shape[0]
    signs = rng.choice([1, -1], n_sub).astype(np.float64)
    sum_perm = signs @ X
    t_pos, t_neg = _onesample_tstat_from_sums(sum_perm, sumsq_all, n_sub)
    return {
        "positive": np.float64(np.max(t_pos)),
        "negative": np.float64(np.max(t_neg)),
    }


def _fast_permutation_task_ind(Xall, Xall2, sum_all, sumsq_all, n1, seed):
    """Two-sample permutation via group-label shuffle + sums. Returns max stats."""
    rng = np.random.RandomState(seed)
    n_all = Xall.shape[0]
    n2 = n_all - n1
    g = np.zeros(n_all, dtype=np.float64)
    g[rng.choice(n_all, n1, replace=False)] = 1.0
    sum1 = g @ Xall
    sumsq1 = g @ Xall2
    sum2 = sum_all - sum1
    sumsq2 = sumsq_all - sumsq1
    t_pos, t_neg = _twosample_tstat_from_sums(sum1, sumsq1, n1, sum2, sumsq2, n2)
    return {
        "positive": np.float64(np.max(t_pos)),
        "negative": np.float64(np.max(t_neg)),
    }


def _fast_permutation_task_paired_edges(X, sumsq_all, seed):
    """Paired/one-sample permutation returning upper-triangle t-stat vectors (for BH-FDR-perm)."""
    rng = np.random.RandomState(seed)
    n_sub = X.shape[0]
    signs = rng.choice([1, -1], n_sub).astype(np.float64)
    sum_perm = signs @ X
    t_pos, t_neg = _onesample_tstat_from_sums(sum_perm, sumsq_all, n_sub)
    return {"positive": t_pos, "negative": t_neg}


def _fast_permutation_task_ind_edges(Xall, Xall2, sum_all, sumsq_all, n1, seed):
    """Two-sample permutation returning upper-triangle t-stat vectors (for BH-FDR-perm)."""
    rng = np.random.RandomState(seed)
    n_all = Xall.shape[0]
    n2 = n_all - n1
    g = np.zeros(n_all, dtype=np.float64)
    g[rng.choice(n_all, n1, replace=False)] = 1.0
    sum1 = g @ Xall
    sumsq1 = g @ Xall2
    sum2 = sum_all - sum1
    sumsq2 = sumsq_all - sumsq1
    t_pos, t_neg = _twosample_tstat_from_sums(sum1, sumsq1, n1, sum2, sumsq2, n2)
    return {"positive": t_pos, "negative": t_neg}


# =============================================================================
# Enhancement-method fast permutation tasks (sums trick + shared enhancers)
# =============================================================================

def _tstat_edges_to_matrix(t_pos, t_neg, triu_idx, N):
    """Expand upper-triangle t-stat vectors to symmetric (N, N) matrices."""
    t_pos_mat = np.zeros((N, N))
    t_pos_mat[triu_idx] = t_pos
    t_pos_mat = t_pos_mat + t_pos_mat.T
    t_neg_mat = np.zeros((N, N))
    t_neg_mat[triu_idx] = t_neg
    t_neg_mat = t_neg_mat + t_neg_mat.T
    return t_pos_mat, t_neg_mat


def _fast_perm_task_enhanced_paired(
    X, sumsq_all, triu_idx, N, enhance_func, enhance_kwargs, seed,
):
    """Paired/one-sample permutation: sign-flip + fast t-stat + enhancement → max."""
    rng = np.random.RandomState(seed)
    n_sub = X.shape[0]
    signs = rng.choice([1, -1], n_sub).astype(np.float64)
    sum_perm = signs @ X
    t_pos, t_neg = _onesample_tstat_from_sums(sum_perm, sumsq_all, n_sub)
    t_pos_mat, t_neg_mat = _tstat_edges_to_matrix(t_pos, t_neg, triu_idx, N)
    stat_dict = {"positive": t_pos_mat, "negative": t_neg_mat}
    if enhance_func is not None:
        stat_dict = enhance_func(stat_dict, **enhance_kwargs)
    return _extract_max_stats(stat_dict, (N, N))


def _fast_perm_task_enhanced_ind(
    Xall, Xall2, sum_all, sumsq_all, n1, triu_idx, N,
    enhance_func, enhance_kwargs, seed,
):
    """Two-sample permutation: group shuffle + fast t-stat + enhancement → max."""
    rng = np.random.RandomState(seed)
    n_all = Xall.shape[0]
    n2 = n_all - n1
    g = np.zeros(n_all, dtype=np.float64)
    g[rng.choice(n_all, n1, replace=False)] = 1.0
    sum1 = g @ Xall
    sumsq1 = g @ Xall2
    sum2 = sum_all - sum1
    sumsq2 = sumsq_all - sumsq1
    t_pos, t_neg = _twosample_tstat_from_sums(sum1, sumsq1, n1, sum2, sumsq2, n2)
    t_pos_mat, t_neg_mat = _tstat_edges_to_matrix(t_pos, t_neg, triu_idx, N)
    stat_dict = {"positive": t_pos_mat, "negative": t_neg_mat}
    if enhance_func is not None:
        stat_dict = enhance_func(stat_dict, **enhance_kwargs)
    return _extract_max_stats(stat_dict, (N, N))


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
    verbose: bool = False,
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
    group1 : ndarray of shape ``(n_samples_1, *dims)``
        Data array for group 1.
    group2 : ndarray of shape ``(n_samples_2, *dims)``, optional
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
        Dictionary with 'negative' and 'positive' arrays of shape (n_permutations,)
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

    # Fast path: raw t-statistic with no enhancement — uses pre-computed sums
    # trick (upper-triangle only, sign-flip-invariant Σx²). ~6-8x faster per perm.
    use_fast_tstat = (
        func in (compute_t_stat_diff, _compute_t_stat_ind, compute_t_stat)
        and not func_kwargs
        and group1.ndim == 3
        and group1.shape[1] == group1.shape[2]
    )

    # Prepare data for permutation
    if test_type_str == TestType.PAIRED.value:
        if use_fast_tstat:
            diffs = group2 - group1
            X, sumsq_all = _precompute_edge_sums(diffs)
            task_func = partial(_fast_permutation_task_paired, X, sumsq_all)
        else:
            array_to_permute = group2 - group1
            task_func = partial(_permutation_task_paired, array_to_permute, func, **func_kwargs)

    elif test_type_str == TestType.TWO_SAMPLE.value:
        if use_fast_tstat:
            Xall_3d = np.concatenate((group1, group2), axis=0)
            Xall, Xall2, sum_all, sumsq_all = _precompute_twosample_sums(Xall_3d)
            task_func = partial(_fast_permutation_task_ind, Xall, Xall2, sum_all, sumsq_all, n1)
        else:
            array_to_permute = np.concatenate((group1, group2), axis=0)
            task_func = partial(_permutation_task_ind, array_to_permute, func, n1, **func_kwargs)

    elif test_type_str == TestType.ONE_SAMPLE.value:
        if use_fast_tstat:
            X, sumsq_all = _precompute_edge_sums(group1)
            task_func = partial(_fast_permutation_task_paired, X, sumsq_all)
        else:
            group2_zeros = np.zeros(group1.shape)
            array_to_permute = group2_zeros - group1
            task_func = partial(_permutation_task_paired, array_to_permute, func, **func_kwargs)
    else:
        raise ValueError(
            f"Invalid test_type: '{test_type_str}'. "
            f"Must be one of: {[t.value for t in TestType]}"
        )

    # Generate seeds for reproducibility
    rng = np.random.RandomState(random_state)
    seeds = rng.randint(0, 2**32 - 1, size=n_permutations, dtype=np.int64)

    _use_mp = use_mp and not _is_worker_process()
    if _use_mp and n_processes is None:
        n_processes = get_available_cores()
    if _use_mp and n_processes is not None:
        n_processes = min(n_processes, n_permutations)

    results = run_permutations(
        task_func, list(seeds),
        use_mp=_use_mp, n_processes=n_processes,
        verbose=verbose, desc="compute_null_dist",
    )

    return _collect_results_to_arrays(results, n_permutations)


def _compute_enhanced_null_dist(
    group1: npt.NDArray[np.float64],
    group2: Optional[npt.NDArray[np.float64]],
    test_type_str: str,
    enhance_func: Callable,
    enhance_kwargs: Dict[str, Any],
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    random_state: Optional[int] = None,
    n_processes: Optional[int] = None,
    use_mp: bool = True,
    verbose: bool = False,
) -> Dict[str, npt.NDArray[np.float64]]:
    """Null distribution for enhancement methods via pre-computed sums fast path.

    Each permutation: sign-flip or group-shuffle → fast t-stat via sums →
    reconstruct (N, N) → apply shared enhancement wrapper → extract max.
    """
    N = group1.shape[1]
    triu_idx = np.triu_indices(N, k=1)

    if test_type_str == TestType.PAIRED.value:
        diffs = group2 - group1
        X, sumsq_all = _precompute_edge_sums(diffs)
        task_func = partial(
            _fast_perm_task_enhanced_paired,
            X, sumsq_all, triu_idx, N, enhance_func, enhance_kwargs,
        )
    elif test_type_str == TestType.ONE_SAMPLE.value:
        X, sumsq_all = _precompute_edge_sums(group1)
        task_func = partial(
            _fast_perm_task_enhanced_paired,
            X, sumsq_all, triu_idx, N, enhance_func, enhance_kwargs,
        )
    elif test_type_str == TestType.TWO_SAMPLE.value:
        n1 = group1.shape[0]
        Xall_3d = np.concatenate((group1, group2), axis=0)
        Xall, Xall2, sum_all, sumsq_all = _precompute_twosample_sums(Xall_3d)
        task_func = partial(
            _fast_perm_task_enhanced_ind,
            Xall, Xall2, sum_all, sumsq_all, n1, triu_idx, N,
            enhance_func, enhance_kwargs,
        )
    else:
        raise ValueError(
            f"Invalid test_type: '{test_type_str}'. "
            f"Must be one of: {[t.value for t in TestType]}"
        )

    rng = np.random.RandomState(random_state)
    seeds = rng.randint(0, 2**32 - 1, size=n_permutations, dtype=np.int64)

    _use_mp = use_mp and not _is_worker_process()
    if _use_mp and n_processes is None:
        n_processes = get_available_cores()
    if _use_mp and n_processes is not None:
        n_processes = min(n_processes, n_permutations)

    results = run_permutations(
        task_func, list(seeds),
        use_mp=_use_mp, n_processes=n_processes,
        verbose=verbose, desc="enhanced_null",
    )

    return _collect_results_to_arrays(results, n_permutations)


def _compute_bh_fdr_perm_p_values(
    group1: npt.NDArray[np.float64],
    group2: Optional[npt.NDArray[np.float64]],
    test_type_str: str,
    n_permutations: int,
    random_state: Optional[int],
    use_mp: bool,
    n_processes: Optional[int],
    verbose: bool = False,
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
        diffs = group2 - group1
        emp_t_dict = compute_t_stat_diff(diffs)
    elif test_type_str == TestType.ONE_SAMPLE.value:
        emp_t_dict = compute_t_stat_diff(group1)
    elif test_type_str == TestType.TWO_SAMPLE.value:
        emp_t_dict = compute_t_stat(group1, group2, test_type=test_type_str)
    else:
        raise ValueError(f"Invalid test_type: '{test_type_str}'")

    N = emp_t_dict["positive"].shape[0]
    triu_idx = np.triu_indices(N, k=1)
    n_edges = len(triu_idx[0])

    # Generate seeds
    rng = np.random.RandomState(random_state)
    seeds = rng.randint(0, 2**32 - 1, size=n_permutations, dtype=np.int64)

    # Prepare permutation task — pre-computed sums fast path (edge vectors).
    # Skips the redundant (N, N) reshape and triu extraction inside the perm loop.
    if test_type_str == TestType.PAIRED.value:
        diffs = group2 - group1
        X, sumsq_all = _precompute_edge_sums(diffs)
        task_func = partial(_fast_permutation_task_paired_edges, X, sumsq_all)
    elif test_type_str == TestType.ONE_SAMPLE.value:
        X, sumsq_all = _precompute_edge_sums(group1)
        task_func = partial(_fast_permutation_task_paired_edges, X, sumsq_all)
    else:  # two-sample
        Xall_3d = np.concatenate((group1, group2), axis=0)
        Xall, Xall2, sum_all, sumsq_all = _precompute_twosample_sums(Xall_3d)
        n1 = group1.shape[0]
        task_func = partial(
            _fast_permutation_task_ind_edges, Xall, Xall2, sum_all, sumsq_all, n1
        )

    _use_mp = use_mp and not _is_worker_process()
    if _use_mp and n_processes is None:
        n_processes = get_available_cores()
    if _use_mp and n_processes is not None:
        n_processes = min(n_processes, n_permutations)

    perm_results = run_permutations(
        task_func, list(seeds),
        use_mp=_use_mp, n_processes=n_processes,
        verbose=verbose, desc="bh_fdr_perm",
    )

    # Compute per-edge p-values (with +1 correction — Phipson & Smyth 2010)
    # and apply BH-FDR correction
    p_values = {}
    for key in ("positive", "negative"):
        emp_upper = emp_t_dict[key][triu_idx]

        # Count how many permutation t-stats >= observed for each edge
        count_ge = np.zeros(n_edges, dtype=np.float64)
        for perm_dict in perm_results:
            # perm_dict[key] is already the upper-triangle vector (fast path)
            count_ge += (perm_dict[key] >= emp_upper).astype(np.float64)

        # +1 correction: prevents p = 0 with finite permutations
        per_edge_p = (count_ge + 1.0) / (n_permutations + 1.0)

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
            n_perm = null_dist.shape[0]
            emp_t_expanded = emp_t[..., np.newaxis]
            count = np.sum(emp_t_expanded < null_dist, axis=-1)
        else:
            # Multi-param: (N, N, n_params) vs (n_permutations, n_params)
            n_perm = null_dist.shape[0]
            emp_t_expanded = emp_t[..., np.newaxis]
            null_reshaped = null_dist.swapaxes(0, 1)[None, None, ...]
            count = np.sum(emp_t_expanded < null_reshaped, axis=-1)

        # +1 correction (Phipson & Smyth 2010): prevents p = 0 with finite perms
        p_values[key] = (count + 1.0) / (n_perm + 1.0)

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
        Dictionary with 'negative' and 'positive' p-value arrays of shape (N, N).
    """
    # Compute t-statistics using existing infrastructure
    if test_type_str == TestType.PAIRED.value:
        diffs = group2 - group1
        t_dict = compute_t_stat_diff(diffs)
        df = _compute_degrees_of_freedom(group1.shape[0], 0, test_type_str)
    elif test_type_str == TestType.ONE_SAMPLE.value:
        t_dict = compute_t_stat_diff(group1)
        df = _compute_degrees_of_freedom(group1.shape[0], 0, test_type_str)
    elif test_type_str == TestType.TWO_SAMPLE.value:
        t_dict = _compute_t_stat_ind(group1, group2)
        df = _compute_degrees_of_freedom(group1.shape[0], group2.shape[0], test_type_str)
    else:
        raise ValueError(f"Invalid test_type: '{test_type_str}'")

    p_values = {}
    for key in ("negative", "positive"):
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
    rng: RngLike = None,
    random_state: Optional[int] = None,  # deprecated alias for `rng`
    n_processes: Optional[int] = None,
    acceleration: Optional[str] = None,
    verbose: bool = False,
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

        - 'negative': P-values for group 1 > group 2.
        - 'positive': P-values for group 2 > group 1.

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
    if random_state is not None and rng is None:
        warn_legacy_random_state("random_state")
    random_state = resolve_seed(rng, legacy_random_state=random_state)

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

    _t0 = time.perf_counter()

    # Parametric methods: compute p-values directly from t-distribution
    if method_enum in PARAMETRIC_METHODS:
        result = _compute_parametric_p_values(
            group1, group2, test_type_str, method_enum
        )
        return InferenceResult(
            result["positive"], result["negative"],
            method=method_str, n_permutations=0, acceleration=None,
            wall_time_s=time.perf_counter() - _t0,
        )

    # BH-FDR with permutation p-values: separate code path
    if method_enum == StatMethod.BH_FDR_PERM:
        result = _compute_bh_fdr_perm_p_values(
            group1, group2, test_type_str,
            n_permutations=n_permutations,
            random_state=random_state,
            use_mp=use_mp,
            n_processes=n_processes,
            verbose=verbose,
        )
        return InferenceResult(
            result["positive"], result["negative"],
            method=method_str, n_permutations=n_permutations,
            acceleration=None,
            wall_time_s=time.perf_counter() - _t0,
        )

    # Resolve enhancement wrapper (None for raw t-stat)
    enhance_func = _ENHANCE_MAP[method_enum]

    # Build enhancement kwargs once (method-specific)
    enhance_kwargs = {}
    if method_enum == StatMethod.NBS:
        enhance_kwargs = {"threshold": threshold, "nbs_stat": nbs_stat}
    elif method_enum in {StatMethod.TFNBS, StatMethod.NI_TFNBS, StatMethod.FBC_TFNBS}:
        enhance_kwargs = {"e": e, "h": h, "n": n, "start_thres": start_thres}
        if method_enum in CONSTRAINED_METHODS:
            enhance_kwargs["net_labels"] = net_labels
        if method_enum == StatMethod.FBC_TFNBS:
            enhance_kwargs["min_cluster_size"] = min_cluster_size
        if method_enum == StatMethod.NI_TFNBS:
            enhance_kwargs["normalization"] = normalization
    elif method_enum == StatMethod.CNBS:
        enhance_kwargs = {"net_labels": net_labels}

    # Compute observed t-stat once, then apply enhancement (if any)
    if test_type_str == TestType.PAIRED.value:
        diffs = group2 - group1
        emp_t_dict = compute_t_stat_diff(diffs)
    elif test_type_str == TestType.ONE_SAMPLE.value:
        emp_t_dict = compute_t_stat_diff(group1)
    elif test_type_str == TestType.TWO_SAMPLE.value:
        emp_t_dict = _compute_t_stat_ind(group1, group2)
    else:
        raise ValueError(
            f"Invalid test_type: '{test_type_str}'. "
            f"Must be one of: {[t.value for t in TestType]}"
        )

    if enhance_func is not None:
        emp_t_dict = enhance_func(emp_t_dict, **enhance_kwargs)

    # Compute null distribution
    # - method='tstat' (no enhancement): use compute_null_dist fast path
    # - enhancement methods: use fast enhancement perm tasks (sums + enhancement)
    if enhance_func is None:
        group2_for_null = group2 if test_type_str != TestType.ONE_SAMPLE.value else None
        tstat_func = (
            _compute_t_stat_ind
            if test_type_str == TestType.TWO_SAMPLE.value
            else compute_t_stat_diff
        )
        max_null_dict = compute_null_dist(
            group1, group2_for_null, tstat_func,
            n_permutations=n_permutations,
            test_type=test_type,
            use_mp=use_mp,
            random_state=random_state,
            n_processes=n_processes,
            verbose=verbose,
        )
    else:
        max_null_dict = _compute_enhanced_null_dist(
            group1, group2, test_type_str,
            enhance_func=enhance_func,
            enhance_kwargs=enhance_kwargs,
            n_permutations=n_permutations,
            random_state=random_state,
            use_mp=use_mp,
            n_processes=n_processes,
            verbose=verbose,
        )

    if acceleration is not None:
        result = compute_p_values_accelerated(
            emp_t_dict, max_null_dict, method=acceleration,
        )
    else:
        result = _compute_p_values_from_null(emp_t_dict, max_null_dict)
    return InferenceResult(
        result["positive"], result["negative"],
        method=method_str, n_permutations=n_permutations,
        acceleration=acceleration,
        wall_time_s=time.perf_counter() - _t0,
    )


# =============================================================================
# T-statistic computation functions
# =============================================================================


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
        Dictionary with 'positive' and 'negative' t-statistic arrays.

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
        return _compute_t_stat_ind(group1, group2)

    elif test_type_str == TestType.PAIRED.value:
        diffs = group2 - group1
        return compute_t_stat_diff(diffs)

    else:
        raise ValueError(
            f"Invalid test_type: '{test_type_str}'. "
            f"Must be one of: {[t.value for t in TestType]}"
        )


def compute_t_stat_diff(
    diff: npt.NDArray[np.float64]
) -> Dict[str, npt.NDArray[np.float64]]:
    """
    Compute t-statistics for paired differences.

    Parameters
    ----------
    diff : ndarray of shape ``(n_samples, *dims)``
        Array containing paired differences.

    Returns
    -------
    dict
        Dictionary with:

        - 'positive': Positive t-values (where group 2 > group 1).
        - 'negative': Negative t-values converted to positive.

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

    return make_tail_result(pos_t, neg_t)


def _compute_t_stat_ind(
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

        - 'positive': Positive t-values (where group 2 > group 1).
        - 'negative': Negative t-values converted to positive.

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

    return make_tail_result(pos_t, neg_t)

