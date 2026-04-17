"""
Shared enhancement wrappers for both t-test and GLM pipelines.

Each wrapper is a pure transformation: `stat_dict → score_dict`. It takes a
per-direction statistic dict (e.g. `{'g2>g1': t_pos, 'g1>g2': t_neg}` for the
t-test pipeline, or `{'positive': ..., 'negative': ...}` for GLM) and applies
the corresponding network-based enhancement to each direction independently.

Wrappers do NOT compute statistics internally — the caller is responsible for
providing the raw statistic (t-stat, β, etc.). This matches the design pattern
documented in developer guide: enhancement methods accept pre-computed statistics.
"""

from __future__ import annotations

from typing import Dict, List, Union

import numpy.typing as npt
import numpy as np

from .defaults import (
    DEFAULT_EXTENT_EXPONENT,
    DEFAULT_HEIGHT_EXPONENT,
    DEFAULT_MIN_CLUSTER_SIZE,
    DEFAULT_NBS_STAT,
    DEFAULT_NBS_THRESHOLD,
    DEFAULT_N_THRESHOLDS_PERMUTATION as DEFAULT_N_THRESHOLDS,
    DEFAULT_START_THRESHOLD,
)
from .nbs_score import get_cnbs_score, get_nbs_score
from .tfnbs_score import (
    get_fbc_tfnbs_score,
    get_network_informed_tfnbs_score,
    get_tfnbs_score,
)


def apply_tfnbs(
    stat_dict: Dict[str, npt.NDArray[np.float64]],
    e: Union[float, List[float]] = DEFAULT_EXTENT_EXPONENT,
    h: Union[float, List[float]] = DEFAULT_HEIGHT_EXPONENT,
    n: int = DEFAULT_N_THRESHOLDS,
    start_thres: float = DEFAULT_START_THRESHOLD,
    **kwargs,
) -> Dict[str, npt.NDArray[np.float64]]:
    """Apply TFNBS (threshold-free cluster enhancement) to each direction."""
    return {
        key: get_tfnbs_score(arr, e, h, n, start_thres=start_thres)
        for key, arr in stat_dict.items()
    }


def apply_nbs(
    stat_dict: Dict[str, npt.NDArray[np.float64]],
    threshold: float = DEFAULT_NBS_THRESHOLD,
    nbs_stat: str = DEFAULT_NBS_STAT,
    **kwargs,
) -> Dict[str, npt.NDArray[np.float64]]:
    """Apply classical NBS (fixed threshold) to each direction."""
    return {
        key: get_nbs_score(arr, threshold=threshold, stat_type=nbs_stat)
        for key, arr in stat_dict.items()
    }


def apply_cnbs(
    stat_dict: Dict[str, npt.NDArray[np.float64]],
    net_labels: npt.NDArray[np.int_],
    **kwargs,
) -> Dict[str, npt.NDArray[np.float64]]:
    """Apply constrained NBS (block-constrained scoring) to each direction."""
    return {
        key: get_cnbs_score(arr, net_labels)
        for key, arr in stat_dict.items()
    }


def apply_ni_tfnbs(
    stat_dict: Dict[str, npt.NDArray[np.float64]],
    net_labels: npt.NDArray[np.int_],
    e: Union[float, List[float]] = DEFAULT_EXTENT_EXPONENT,
    h: Union[float, List[float]] = DEFAULT_HEIGHT_EXPONENT,
    n: int = DEFAULT_N_THRESHOLDS,
    start_thres: float = DEFAULT_START_THRESHOLD,
    normalization: str = "sqrt",
    **kwargs,
) -> Dict[str, npt.NDArray[np.float64]]:
    """Apply network-informed TFNBS (block-density weighted) to each direction."""
    return {
        key: get_network_informed_tfnbs_score(
            arr, net_labels, e, h, n,
            start_thres=start_thres, normalization=normalization,
        )
        for key, arr in stat_dict.items()
    }


def apply_fbc_tfnbs(
    stat_dict: Dict[str, npt.NDArray[np.float64]],
    net_labels: npt.NDArray[np.int_],
    e: Union[float, List[float]] = DEFAULT_EXTENT_EXPONENT,
    h: Union[float, List[float]] = DEFAULT_HEIGHT_EXPONENT,
    n: int = DEFAULT_N_THRESHOLDS,
    start_thres: float = DEFAULT_START_THRESHOLD,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    **kwargs,
) -> Dict[str, npt.NDArray[np.float64]]:
    """Apply functional-block-clustering TFNBS to each direction."""
    return {
        key: get_fbc_tfnbs_score(
            arr, net_labels, e, h, n,
            start_thres=start_thres, min_cluster_size=min_cluster_size,
        )
        for key, arr in stat_dict.items()
    }
