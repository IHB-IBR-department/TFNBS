"""
GLM-based statistical testing for brain connectivity networks.

Implements edge-wise General Linear Model with Freedman-Lane permutation
for confound-aware inference on connectivity matrices. Supports all
enhancement methods (TFNBS, NBS, cNBS, NI-TFNBS, FBC-TFNBS).

Main Functions
--------------
compute_glm_stat : Compute GLM t-statistics for connectivity data
compute_p_val_glm : Full GLM pipeline with permutation testing
build_design_matrix : Convenience builder for design matrix + contrast

References
----------
Freedman & Lane (1983). A nonstochastic interpretation of reported significance levels.
Anderson & Legendre (1999). An empirical comparison of permutation methods.
Winkler et al. (2014). Permutation inference for the general linear model.
"""

from __future__ import annotations

import logging
from enum import Enum
from functools import partial
from multiprocessing import Pool
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

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
from .pairwise_stats import (
    _extract_max_stats,
    _collect_results_to_arrays,
    _compute_p_values_from_null,
    _is_worker_process,
    get_available_cores,
    StatMethod,
    CONSTRAINED_METHODS,
)


__all__ = [
    "GLMStatType",
    "compute_glm_stat",
    "compute_p_val_glm",
    "build_design_matrix",
]

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class GLMStatType(str, Enum):
    """Statistic type for GLM inference."""

    TSTAT = "tstat"
    """t-statistic: beta / SE(beta). Scale-invariant, start_thres=1.65 valid."""

    BETA = "beta"
    """Raw regression coefficient. Interpretable, needs adapted threshold."""


# =============================================================================
# OLS precomputation
# =============================================================================

def _precompute_ols(
    X: npt.NDArray[np.float64],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Precompute OLS quantities that are constant across permutations.

    Parameters
    ----------
    X : ndarray of shape (n_subjects, p)
        Design matrix (must include intercept if desired).

    Returns
    -------
    X_pinv : ndarray of shape (p, n_subjects)
        Pseudoinverse: (X'X)^{-1} X'.
    XtX_inv_diag : ndarray of shape (p,)
        Diagonal of (X'X)^{-1}, needed for SE computation.
    """
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    X_pinv = XtX_inv @ X.T
    XtX_inv_diag = np.diag(XtX_inv)
    return X_pinv, XtX_inv_diag


# =============================================================================
# Core GLM computation
# =============================================================================

def compute_glm_stat(
    Y: npt.NDArray[np.float64],
    X: npt.NDArray[np.float64],
    contrast: npt.NDArray[np.float64],
    stat_type: Union[str, GLMStatType] = GLMStatType.TSTAT,
    X_pinv: Optional[npt.NDArray[np.float64]] = None,
    XtX_inv_diag: Optional[npt.NDArray[np.float64]] = None,
) -> Dict[str, npt.NDArray[np.float64]]:
    """
    Compute edge-wise GLM statistics for connectivity matrices.

    For each edge (i,j): Y_ij = X @ beta_ij + epsilon_ij.
    Returns the statistic for the contrast of interest, separated into
    positive and negative effects.

    Parameters
    ----------
    Y : ndarray of shape (n_subjects, N, N)
        Connectivity matrices (symmetric, zero diagonal).
    X : ndarray of shape (n_subjects, p)
        Design matrix (should include intercept column if needed).
    contrast : ndarray of shape (p,)
        Contrast vector. Must have exactly one non-zero entry targeting
        the predictor of interest (e.g., [0, 1, 0] for column 1).
    stat_type : {'tstat', 'beta'} or GLMStatType, default='tstat'
        Type of statistic to compute.
    X_pinv : ndarray of shape (p, n_subjects), optional
        Precomputed pseudoinverse. If None, computed internally.
    XtX_inv_diag : ndarray of shape (p,), optional
        Precomputed diagonal of (X'X)^{-1}. Required when stat_type='tstat'
        and X_pinv is provided.

    Returns
    -------
    dict
        Dictionary with:
        - 'positive': non-negative statistic array (N, N) for positive effects
        - 'negative': non-negative statistic array (N, N) for negative effects
    """
    stat_type_str = stat_type.value if isinstance(stat_type, GLMStatType) else stat_type

    n, N, _ = Y.shape
    contrast = np.asarray(contrast, dtype=np.float64)
    p = X.shape[1]

    # Precompute if not provided
    if X_pinv is None:
        X_pinv, XtX_inv_diag = _precompute_ols(X)

    # Compute betas: (p, N, N)
    # Y reshaped to (n, N*N), then beta = X_pinv @ Y_flat → (p, N*N) → (p, N, N)
    Y_flat = Y.reshape(n, -1)  # (n, N*N)
    beta_flat = X_pinv @ Y_flat  # (p, N*N)
    beta = beta_flat.reshape(p, N, N)  # (p, N, N)

    # Contrast beta: scalar statistic per edge
    # contrast @ beta → (N, N)
    c_beta = np.tensordot(contrast, beta, axes=([0], [0]))  # (N, N)

    if stat_type_str == GLMStatType.BETA.value:
        stat = c_beta
    elif stat_type_str == GLMStatType.TSTAT.value:
        # Residuals: (n, N*N)
        residuals_flat = Y_flat - X @ beta_flat  # (n, N*N)

        # Residual variance per edge: sigma^2 = RSS / (n - p)
        df = n - p
        sigma2_flat = np.sum(residuals_flat ** 2, axis=0) / df  # (N*N,)
        sigma2 = sigma2_flat.reshape(N, N)

        # SE of contrast beta: sqrt(c' (X'X)^{-1} c * sigma^2)
        c_XtX_inv_c = contrast @ np.linalg.inv(X.T @ X) @ contrast  # scalar
        se = np.sqrt(c_XtX_inv_c * sigma2)

        with np.errstate(divide='ignore', invalid='ignore'):
            stat = c_beta / se
            stat = np.where(se == 0, 0, stat)
    else:
        raise ValueError(
            f"Invalid stat_type: '{stat_type_str}'. "
            f"Must be one of: {[s.value for s in GLMStatType]}"
        )

    # Separate positive and negative effects
    positive = np.where(stat > 0, stat, 0.0)
    negative = np.where(stat < 0, -stat, 0.0)

    return {"positive": positive, "negative": negative}


# =============================================================================
# Design matrix builder
# =============================================================================

def build_design_matrix(
    interest: npt.NDArray[np.float64],
    confounds: Optional[npt.NDArray[np.float64]] = None,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Build design matrix and contrast vector from interest and confound variables.

    Constructs X = [intercept, confounds, interest] and contrast = [0, ..., 0, 1]
    targeting the last column (interest variable).

    Parameters
    ----------
    interest : ndarray of shape (n_subjects,)
        Variable of interest (continuous or binary).
    confounds : ndarray of shape (n_subjects,) or (n_subjects, q), optional
        Confound variables. If 1D, treated as single confound.

    Returns
    -------
    X : ndarray of shape (n_subjects, p)
        Design matrix with intercept, confounds, and interest.
    contrast : ndarray of shape (p,)
        Contrast vector [0, ..., 0, 1] targeting the interest column.
    """
    n = len(interest)
    interest = np.asarray(interest, dtype=np.float64)
    if interest.ndim == 1:
        interest = interest[:, np.newaxis]

    intercept = np.ones((n, 1), dtype=np.float64)

    if confounds is not None:
        confounds = np.asarray(confounds, dtype=np.float64)
        if confounds.ndim == 1:
            confounds = confounds[:, np.newaxis]
        X = np.hstack([intercept, confounds, interest])
    else:
        X = np.hstack([intercept, interest])

    contrast = np.zeros(X.shape[1], dtype=np.float64)
    contrast[-1] = 1.0

    return X, contrast


# =============================================================================
# Freedman-Lane permutation helpers
# =============================================================================

def _compute_reduced_model(
    X: npt.NDArray[np.float64],
    contrast: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Compute the reduced design matrix for Freedman-Lane permutation.

    The reduced model contains all columns NOT targeted by the contrast.

    Parameters
    ----------
    X : ndarray of shape (n_subjects, p)
        Full design matrix.
    contrast : ndarray of shape (p,)
        Contrast vector (non-zero entries identify columns of interest).

    Returns
    -------
    X_reduced : ndarray of shape (n_subjects, p_reduced)
        Reduced design matrix (confounds + intercept only).
    """
    keep_cols = np.where(contrast == 0)[0]
    if len(keep_cols) == 0:
        # No confounds — reduced model is just intercept
        return np.ones((X.shape[0], 1), dtype=np.float64)
    return X[:, keep_cols]


def _freedman_lane_permutation_task(
    Y_hat_reduced: npt.NDArray[np.float64],
    residuals_reduced: npt.NDArray[np.float64],
    X: npt.NDArray[np.float64],
    contrast: npt.NDArray[np.float64],
    X_pinv: npt.NDArray[np.float64],
    XtX_inv_diag: npt.NDArray[np.float64],
    reference_shape: Tuple[int, ...],
    stat_type: str,
    enhance_func: Optional[Callable] = None,
    seed: Optional[int] = None,
    **enhance_kwargs,
) -> Dict[str, np.float64]:
    """
    Single Freedman-Lane permutation step.

    1. Permute reduced-model residuals across subjects
    2. Reconstruct Y_perm = Y_hat_reduced + permuted residuals
    3. Fit full model to Y_perm
    4. Optionally apply enhancement
    5. Return max statistics

    Parameters
    ----------
    Y_hat_reduced : ndarray of shape (n_subjects, N, N)
        Predicted values from reduced model.
    residuals_reduced : ndarray of shape (n_subjects, N, N)
        Residuals from reduced model.
    X, contrast, X_pinv, XtX_inv_diag : precomputed GLM quantities
    reference_shape : tuple
        Shape (N, N) for _extract_max_stats.
    stat_type : str
        'tstat' or 'beta'.
    enhance_func : callable, optional
        Enhancement function to apply to GLM stats.
    seed : int
        Random seed for this permutation.
    **enhance_kwargs
        Keyword arguments for enhance_func.

    Returns
    -------
    dict
        Maximum statistics per direction.
    """
    rng = np.random.RandomState(seed)
    perm_idx = rng.permutation(residuals_reduced.shape[0])
    residuals_perm = residuals_reduced[perm_idx]
    Y_perm = Y_hat_reduced + residuals_perm

    stat_dict = compute_glm_stat(
        Y_perm, X, contrast,
        stat_type=stat_type,
        X_pinv=X_pinv,
        XtX_inv_diag=XtX_inv_diag,
    )

    if enhance_func is not None:
        stat_dict = enhance_func(stat_dict, **enhance_kwargs)

    return _extract_max_stats(stat_dict, reference_shape)


# =============================================================================
# Enhancement wrappers for GLM pipeline
# =============================================================================

def _enhance_tfnbs(
    stat_dict: Dict[str, npt.NDArray],
    e: Union[float, List[float]] = DEFAULT_EXTENT_EXPONENT,
    h: Union[float, List[float]] = DEFAULT_HEIGHT_EXPONENT,
    n: int = DEFAULT_N_THRESHOLDS,
    start_thres: float = DEFAULT_START_THRESHOLD,
    **kwargs,
) -> Dict[str, npt.NDArray]:
    """Apply TFNBS enhancement to GLM statistics."""
    return {
        key: get_tfnbs_score(arr, e, h, n, start_thres=start_thres)
        for key, arr in stat_dict.items()
    }


def _enhance_nbs(
    stat_dict: Dict[str, npt.NDArray],
    threshold: float = DEFAULT_NBS_THRESHOLD,
    nbs_stat: str = DEFAULT_NBS_STAT,
    **kwargs,
) -> Dict[str, npt.NDArray]:
    """Apply NBS enhancement to GLM statistics."""
    return {
        key: get_nbs_score(arr, threshold=threshold, stat_type=nbs_stat)
        for key, arr in stat_dict.items()
    }


def _enhance_cnbs(
    stat_dict: Dict[str, npt.NDArray],
    net_labels: npt.NDArray[np.int_],
    **kwargs,
) -> Dict[str, npt.NDArray]:
    """Apply cNBS enhancement to GLM statistics."""
    return {
        key: get_cnbs_score(arr, net_labels)
        for key, arr in stat_dict.items()
    }


def _enhance_ni_tfnbs(
    stat_dict: Dict[str, npt.NDArray],
    net_labels: npt.NDArray[np.int_],
    e: Union[float, List[float]] = DEFAULT_EXTENT_EXPONENT,
    h: Union[float, List[float]] = DEFAULT_HEIGHT_EXPONENT,
    n: int = DEFAULT_N_THRESHOLDS,
    start_thres: float = DEFAULT_START_THRESHOLD,
    normalization: str = "sqrt",
    **kwargs,
) -> Dict[str, npt.NDArray]:
    """Apply NI-TFNBS enhancement to GLM statistics."""
    return {
        key: get_network_informed_tfnbs_score(
            arr, net_labels, e, h, n,
            start_thres=start_thres, normalization=normalization,
        )
        for key, arr in stat_dict.items()
    }


def _enhance_fbc_tfnbs(
    stat_dict: Dict[str, npt.NDArray],
    net_labels: npt.NDArray[np.int_],
    e: Union[float, List[float]] = DEFAULT_EXTENT_EXPONENT,
    h: Union[float, List[float]] = DEFAULT_HEIGHT_EXPONENT,
    n: int = DEFAULT_N_THRESHOLDS,
    start_thres: float = DEFAULT_START_THRESHOLD,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    **kwargs,
) -> Dict[str, npt.NDArray]:
    """Apply FBC-TFNBS enhancement to GLM statistics."""
    return {
        key: get_fbc_tfnbs_score(
            arr, net_labels, e, h, n,
            start_thres=start_thres, min_cluster_size=min_cluster_size,
        )
        for key, arr in stat_dict.items()
    }


# Enhancement function registry
_ENHANCE_MAP = {
    StatMethod.TSTAT: None,  # No enhancement
    StatMethod.TFNBS: _enhance_tfnbs,
    StatMethod.NBS: _enhance_nbs,
    StatMethod.CNBS: _enhance_cnbs,
    StatMethod.NI_TFNBS: _enhance_ni_tfnbs,
    StatMethod.FBC_TFNBS: _enhance_fbc_tfnbs,
}


# =============================================================================
# Main orchestrator
# =============================================================================

def compute_p_val_glm(
    Y: npt.NDArray[np.float64],
    # --- Advanced API ---
    design_matrix: Optional[npt.NDArray[np.float64]] = None,
    contrast: Optional[npt.NDArray[np.float64]] = None,
    # --- Convenience API ---
    interest: Optional[npt.NDArray[np.float64]] = None,
    confounds: Optional[npt.NDArray[np.float64]] = None,
    # --- Common parameters ---
    stat_type: Union[str, GLMStatType] = GLMStatType.TSTAT,
    method: Union[str, StatMethod] = StatMethod.TFNBS,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    two_tailed: bool = False,
    use_mp: bool = True,
    random_state: Optional[int] = None,
    n_processes: Optional[int] = None,
    # --- Method-specific parameters ---
    net_labels: Optional[npt.NDArray[np.int_]] = None,
    threshold: float = DEFAULT_NBS_THRESHOLD,
    nbs_stat: str = DEFAULT_NBS_STAT,
    e: Union[float, List[float]] = DEFAULT_EXTENT_EXPONENT,
    h: Union[float, List[float]] = DEFAULT_HEIGHT_EXPONENT,
    n: int = DEFAULT_N_THRESHOLDS,
    start_thres: float = DEFAULT_START_THRESHOLD,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    normalization: str = "sqrt",
) -> Dict[str, npt.NDArray[np.float64]]:
    """
    Compute p-values for connectivity data using GLM with Freedman-Lane permutation.

    Supports two APIs that share the same core engine:

    - **Advanced API**: provide ``design_matrix`` and ``contrast`` directly.
    - **Convenience API**: provide ``interest`` (and optional ``confounds``);
      the design matrix and contrast are built automatically.

    Parameters
    ----------
    Y : ndarray of shape (n_subjects, N, N)
        Connectivity matrices (symmetric, zero diagonal).
    design_matrix : ndarray of shape (n_subjects, p), optional
        Full design matrix (advanced API). Mutually exclusive with ``interest``.
    contrast : ndarray of shape (p,), optional
        Contrast vector (advanced API). Required with ``design_matrix``.
    interest : ndarray of shape (n_subjects,), optional
        Variable of interest (convenience API). Mutually exclusive with
        ``design_matrix``.
    confounds : ndarray of shape (n_subjects,) or (n_subjects, q), optional
        Confound variables (convenience API).
    stat_type : {'tstat', 'beta'} or GLMStatType, default='tstat'
        Type of statistic to compute.
    method : str or StatMethod, default='tfnbs'
        Enhancement method.
    n_permutations : int, default=1000
        Number of permutations for null distribution.
    two_tailed : bool, default=False
        If False (default), per-tail FWER control (separate null for positive
        and negative). If True, joint null from max(max_positive, max_negative).
    use_mp : bool, default=True
        Use multiprocessing for permutation testing.
    random_state : int, optional
        Random seed for reproducibility.
    n_processes : int, optional
        Number of CPU cores for parallel computing.
    net_labels : ndarray of shape (N,), optional
        Network labels (required for cnbs, ni_tfnbs, fbc_tfnbs).
    threshold, nbs_stat, e, h, n, start_thres, min_cluster_size, normalization
        Method-specific parameters (see ``compute_p_val`` docstring).

    Returns
    -------
    dict
        Dictionary with 'positive' and 'negative' p-value arrays of shape (N, N).

    Raises
    ------
    ValueError
        If API arguments are inconsistent or contrast is invalid.
    """
    # ---- Resolve API mode ----
    if design_matrix is not None and interest is not None:
        raise ValueError(
            "Provide either (design_matrix, contrast) or (interest, confounds), "
            "not both."
        )
    if design_matrix is None and interest is None:
        raise ValueError(
            "Must provide either design_matrix or interest."
        )

    if interest is not None:
        X, contrast_vec = build_design_matrix(interest, confounds)
    else:
        X = np.asarray(design_matrix, dtype=np.float64)
        if contrast is None:
            raise ValueError("contrast is required when using design_matrix.")
        contrast_vec = np.asarray(contrast, dtype=np.float64)

    # ---- Validate inputs ----
    n_subjects = Y.shape[0]
    N = Y.shape[1]
    p = X.shape[1]

    if X.shape[0] != n_subjects:
        raise ValueError(
            f"Design matrix has {X.shape[0]} rows but Y has {n_subjects} subjects."
        )
    if len(contrast_vec) != p:
        raise ValueError(
            f"Contrast vector length ({len(contrast_vec)}) must match "
            f"number of columns in design matrix ({p})."
        )
    if np.all(contrast_vec == 0):
        raise ValueError("Contrast vector must have at least one non-zero entry.")

    stat_type_str = stat_type.value if isinstance(stat_type, GLMStatType) else stat_type
    method_str = method.value if isinstance(method, StatMethod) else method

    try:
        method_enum = StatMethod(method_str)
    except ValueError:
        valid = [m.value for m in StatMethod if m in _ENHANCE_MAP]
        raise ValueError(f"Invalid method: '{method_str}'. Must be one of: {valid}")

    if method_enum not in _ENHANCE_MAP:
        raise ValueError(
            f"Method '{method_str}' is not supported in the GLM pipeline. "
            f"Use compute_p_val() for parametric baselines."
        )

    if method_enum in CONSTRAINED_METHODS and net_labels is None:
        raise ValueError(
            f"Method '{method_str}' requires net_labels."
        )

    # ---- Build enhancement kwargs ----
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

    enhance_func = _ENHANCE_MAP[method_enum]

    # ---- Precompute OLS ----
    X_pinv, XtX_inv_diag = _precompute_ols(X)

    # ---- Compute observed statistics ----
    emp_stat_dict = compute_glm_stat(
        Y, X, contrast_vec,
        stat_type=stat_type_str,
        X_pinv=X_pinv,
        XtX_inv_diag=XtX_inv_diag,
    )
    if enhance_func is not None:
        emp_stat_dict = enhance_func(emp_stat_dict, **enhance_kwargs)

    # ---- Freedman-Lane: compute reduced model ----
    X_reduced = _compute_reduced_model(X, contrast_vec)
    X_red_pinv, _ = _precompute_ols(X_reduced)

    Y_flat = Y.reshape(n_subjects, -1)
    Y_hat_reduced_flat = X_reduced @ (X_red_pinv @ Y_flat)
    Y_hat_reduced = Y_hat_reduced_flat.reshape(n_subjects, N, N)
    residuals_reduced = Y - Y_hat_reduced

    reference_shape = (N, N)

    # ---- Generate seeds ----
    rng = np.random.RandomState(random_state)
    seeds = rng.randint(0, 2**32 - 1, size=n_permutations, dtype=np.int64)

    # ---- Build permutation task ----
    task_func = partial(
        _freedman_lane_permutation_task,
        Y_hat_reduced,
        residuals_reduced,
        X,
        contrast_vec,
        X_pinv,
        XtX_inv_diag,
        reference_shape,
        stat_type_str,
        enhance_func,
        **enhance_kwargs,
    )

    # ---- Run permutations ----
    _use_mp = use_mp and not _is_worker_process()

    if _use_mp:
        if n_processes is None:
            n_processes = get_available_cores()
        n_processes = min(n_processes, n_permutations)
        with Pool(processes=n_processes) as pool:
            results = pool.map(task_func, seeds)
    else:
        results = [task_func(seed) for seed in seeds]

    max_null_dict = _collect_results_to_arrays(results, n_permutations)

    # ---- Two-tailed FWER ----
    if two_tailed:
        # Build joint null from max(max_positive, max_negative)
        joint_null = np.maximum(
            max_null_dict["positive"],
            max_null_dict["negative"],
        )
        max_null_dict = {
            "positive": joint_null,
            "negative": joint_null,
        }

    return _compute_p_values_from_null(emp_stat_dict, max_null_dict)
