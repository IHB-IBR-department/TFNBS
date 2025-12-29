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

import logging
import multiprocessing
from enum import Enum
from functools import partial
from multiprocessing import Pool
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from .tfnbs_score import get_tfnbs_score, DEFAULT_START_THRESHOLD


__all__ = [
    # Enums and constants
    "TestType",
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


DEFAULT_N_PERMUTATIONS: int = 1000
"""Default number of permutations for null distribution."""

DEFAULT_EXTENT_EXPONENT: float = 0.4
"""Default extent exponent (E) for TFCE in permutation testing."""

DEFAULT_HEIGHT_EXPONENT: float = 3.0
"""Default height exponent (H) for TFCE in permutation testing."""

DEFAULT_N_THRESHOLDS: int = 10
"""Default number of threshold integration steps for TFCE."""


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

    Parameters
    ----------
    stat_dict : dict
        Dictionary with 'g1>g2' and 'g2>g1' statistic arrays.
    reference_shape : tuple
        Shape of a single sample for determining dimensionality.

    Returns
    -------
    dict
        Dictionary with maximum statistics for each direction.
    """
    if stat_dict["g1>g2"].shape == reference_shape:
        return {
            "g1>g2": np.max(stat_dict["g1>g2"]).astype(np.float64),
            "g2>g1": np.max(stat_dict["g2>g1"]).astype(np.float64)
        }
    else:
        # Multi-parameter case: max over spatial dims, keep param dim
        spatial_axes = tuple(range(stat_dict["g1>g2"].ndim - 1))
        return {
            "g1>g2": np.max(stat_dict["g1>g2"], axis=spatial_axes).astype(np.float64),
            "g2>g1": np.max(stat_dict["g2>g1"], axis=spatial_axes).astype(np.float64)
        }


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
    use_mp: bool = False,
    **func_kwargs
) -> Dict[str, npt.NDArray[np.float64]]:
    """
    Compute null distribution of maximum t-statistics via permutation testing.

    Optimized implementation with:

    - Fixed indexing bug in sequential mode
    - Efficient result collection
    - Better memory management for multiprocessing

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
    use_mp : bool, default=False
        Whether to use multiprocessing.
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

    if use_mp:
        # Parallel computation
        if n_processes is None:
            n_processes = multiprocessing.cpu_count()
        n_processes = min(n_processes, n_permutations)

        with Pool(processes=n_processes) as pool:
            results = pool.map(task_func, seeds)
    else:
        # Sequential computation
        results = [task_func(seed) for seed in seeds]

    return _collect_results_to_arrays(results, n_permutations)


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


def compute_p_val(
    group1: npt.NDArray[np.float64],
    group2: Optional[npt.NDArray[np.float64]] = None,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    test_type: Union[str, TestType] = TestType.PAIRED,
    tf: bool = True,
    use_mp: bool = True,
    random_state: Optional[int] = None,
    n_processes: Optional[int] = None,
    **kwargs
) -> Dict[str, npt.NDArray[np.float64]]:
    """
    Compute p-values using permutation testing with TFNBS or standard t-statistics.

    Parameters
    ----------
    group1 : ndarray of shape (n_subjects_g1, N, N)
        Input connectivity matrices for group 1.
    group2 : ndarray of shape (n_subjects_g2, N, N), optional
        Input connectivity matrices for group 2. Required for paired/two-sample tests.
    n_permutations : int, default=1000
        Number of permutations for null distribution.
    test_type : {'paired', 'one-sample', 'two-sample'} or TestType
        Type of statistical test.
    tf : bool, default=True
        Use TFCE transformation (True) or standard t-test (False).
    use_mp : bool, default=True
        Use multiprocessing for permutation testing.
    random_state : int, optional
        Random seed for reproducibility.
    n_processes : int, optional
        Number of CPU cores for parallel computing.
    **kwargs
        Additional arguments passed to TFCE (e, h, n, start_thres).

    Returns
    -------
    dict
        Dictionary with p-value arrays:

        - 'g1>g2': P-values for group 1 > group 2.
        - 'g2>g1': P-values for group 2 > group 1.

    Examples
    --------
    >>> np.random.seed(2)
    >>> group1 = np.random.rand(5, 3, 3)
    >>> for arr in group1: np.fill_diagonal(arr, 1)
    >>> group2 = np.random.rand(8, 3, 3)
    >>> for arr in group2: np.fill_diagonal(arr, 1)
    >>> p_vals = compute_p_val(group1, group2, n_permutations=10, test_type='two-sample', tf=False, use_mp=False, random_state=0)
    >>> (p_vals['g2>g1'].mean() < p_vals['g1>g2'].mean()).all() is np.True_
    True
    """
    # Normalize test_type to string
    test_type_str = test_type.value if isinstance(test_type, TestType) else test_type

    # Select appropriate t-statistic function
    if test_type_str == TestType.PAIRED.value:
        t_func = compute_t_stat_tfnbs_diffs if tf else compute_t_stat_diff
        emp_t_dict = t_func(compute_diffs(group1, group2), **kwargs)

    elif test_type_str == TestType.TWO_SAMPLE.value:
        t_func = compute_t_stat_tfnbs if tf else compute_t_stat
        emp_t_dict = t_func(group1, group2, test_type=TestType.TWO_SAMPLE.value, **kwargs)

    elif test_type_str == TestType.ONE_SAMPLE.value:
        t_func = compute_t_stat_tfnbs_diffs if tf else compute_t_stat_diff
        emp_t_dict = t_func(group1, **kwargs)
    else:
        raise ValueError(
            f"Invalid test_type: '{test_type_str}'. "
            f"Must be one of: {[t.value for t in TestType]}"
        )

    # Compute null distribution
    group2_for_null = group2 if test_type_str != TestType.ONE_SAMPLE.value else None

    max_null_dict = compute_null_dist(
        group1, group2_for_null, t_func,
        n_permutations=n_permutations,
        test_type=test_type,
        use_mp=use_mp,
        random_state=random_state,
        n_processes=n_processes,
        **kwargs
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


