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
from ._enhancement import (
    apply_cnbs,
    apply_fbc_tfnbs,
    apply_nbs,
    apply_ni_tfnbs,
    apply_tfnbs,
)
from .pairwise_stats import (
    _extract_max_stats,
    _collect_results_to_arrays,
    _compute_p_values_from_null,
    _is_worker_process,
    compute_p_val,
    get_available_cores,
    StatMethod,
    CONSTRAINED_METHODS,
)
from .acceleration import compute_p_values_accelerated


__all__ = [
    "GLMStatType",
    "compute_glm_stat",
    "compute_p_val_glm",
    "compute_p_val_paired_glm",
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

    FSTAT = "fstat"
    """F-statistic for an omnibus (multi-row) contrast.

    Non-negative by construction — the enhancement pipeline sees a single
    ``'omnibus'`` key instead of a ``positive``/``negative`` split. For a
    single-row contrast, F == t² (same rejection regions, same rankings)."""


# =============================================================================
# OLS precomputation
# =============================================================================

def _precompute_ols(
    X: npt.NDArray[np.float64],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
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
    XtX_inv : ndarray of shape (p, p)
        Full (X'X)^{-1}, needed for contrast SE computation.

    Raises
    ------
    np.linalg.LinAlgError
        If X'X is singular (e.g., perfectly collinear regressors).
    """
    XtX = X.T @ X
    cond = np.linalg.cond(XtX)
    if cond > 1e12:
        logger.warning(
            f"Design matrix is ill-conditioned (cond={cond:.1e}). "
            f"Results may be numerically unstable."
        )
    XtX_inv = np.linalg.inv(XtX)
    X_pinv = XtX_inv @ X.T
    XtX_inv_diag = np.diag(XtX_inv)
    return X_pinv, XtX_inv_diag, XtX_inv


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
    XtX_inv: Optional[npt.NDArray[np.float64]] = None,
) -> Dict[str, npt.NDArray[np.float64]]:
    """
    Compute edge-wise GLM statistics for connectivity matrices.

    For each edge (i,j): Y_ij = X @ beta_ij + epsilon_ij.

    - ``tstat``/``beta``: single-column contrast, returns ``{positive, negative}``.
    - ``fstat``: multi-row contrast (omnibus F), returns ``{omnibus}`` — F is
      non-negative by construction, so there is no direction to split.

    Parameters
    ----------
    Y : ndarray of shape (n_subjects, N, N)
        Connectivity matrices (symmetric, zero diagonal).
    X : ndarray of shape (n_subjects, p)
        Design matrix (should include intercept column if needed).
    contrast : ndarray of shape (p,) or (r, p)
        Contrast vector (for ``tstat``/``beta``: 1D) or contrast matrix (for
        ``fstat``: 2D with ``r`` rows). A 1D contrast passed with ``stat_type
        ='fstat'`` is treated as a 1×p matrix (F == t² at each edge).
    stat_type : {'tstat', 'beta', 'fstat'} or GLMStatType, default='tstat'
        Type of statistic to compute.
    X_pinv : ndarray of shape (p, n_subjects), optional
        Precomputed pseudoinverse. If None, computed internally.
    XtX_inv_diag : ndarray of shape (p,), optional
        Precomputed diagonal of (X'X)^{-1}. Required when stat_type='tstat'
        and X_pinv is provided.

    Returns
    -------
    dict
        For ``tstat``/``beta``: ``{'positive': (N, N), 'negative': (N, N)}``,
        both non-negative.
        For ``fstat``: ``{'omnibus': (N, N)}``, non-negative.
    """
    stat_type_str = stat_type.value if isinstance(stat_type, GLMStatType) else stat_type

    n, N, _ = Y.shape
    contrast = np.asarray(contrast, dtype=np.float64)
    p = X.shape[1]

    # Precompute if not provided
    if X_pinv is None:
        X_pinv, XtX_inv_diag, XtX_inv = _precompute_ols(X)

    # Compute betas: (p, N, N)
    # Y reshaped to (n, N*N), then beta = X_pinv @ Y_flat → (p, N*N) → (p, N, N)
    Y_flat = Y.reshape(n, -1)  # (n, N*N)
    beta_flat = X_pinv @ Y_flat  # (p, N*N)
    beta = beta_flat.reshape(p, N, N)  # (p, N, N)

    if stat_type_str == GLMStatType.FSTAT.value:
        # --- F-statistic: (Cβ)' [C(X'X)⁻¹C']⁻¹ (Cβ) / (r · σ²) ---
        if contrast.ndim == 1:
            C = contrast.reshape(1, -1)
        elif contrast.ndim == 2:
            C = contrast
        else:
            raise ValueError(
                f"F-stat contrast must be 1D or 2D, got ndim={contrast.ndim}."
            )
        r = C.shape[0]
        if C.shape[1] != p:
            raise ValueError(
                f"Contrast has {C.shape[1]} columns but design matrix has {p}."
            )

        # Residual variance per edge
        residuals_flat = Y_flat - X @ beta_flat  # (n, N*N)
        df = n - p
        sigma2_flat = np.sum(residuals_flat ** 2, axis=0) / df
        sigma2 = sigma2_flat.reshape(N, N)

        if XtX_inv is None:
            XtX_inv = np.linalg.inv(X.T @ X)

        # M = C (X'X)⁻¹ C' shape (r, r); invert once (small)
        M = C @ XtX_inv @ C.T
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError as err:
            raise ValueError(
                "C(X'X)⁻¹C' is singular — contrast rows are not linearly "
                "independent on the column space of X."
            ) from err

        # Cβ shape (r, N, N); quadratic form per edge via einsum
        Cbeta = np.tensordot(C, beta, axes=([1], [0]))  # (r, N, N)
        quad = np.einsum("ixy,ij,jxy->xy", Cbeta, M_inv, Cbeta)  # (N, N)

        with np.errstate(divide="ignore", invalid="ignore"):
            F = quad / (r * sigma2)
            F = np.where(sigma2 == 0, 0.0, F)
            F = np.where(F < 0, 0.0, F)  # guard numerical noise near zero

        return {"omnibus": F}

    # --- t-stat / beta path: single-column contrast, signed output ---
    if contrast.ndim != 1:
        raise ValueError(
            f"stat_type='{stat_type_str}' requires a 1D contrast; got "
            f"ndim={contrast.ndim}. Use stat_type='fstat' for multi-row contrasts."
        )

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
        if XtX_inv is None:
            XtX_inv = np.linalg.inv(X.T @ X)
        c_XtX_inv_c = contrast @ XtX_inv @ contrast  # scalar
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
    For a 2D contrast matrix (F-test), a column is targeted if ANY row of
    the contrast has a non-zero entry for it.

    Parameters
    ----------
    X : ndarray of shape (n_subjects, p)
        Full design matrix.
    contrast : ndarray of shape (p,) or (r, p)
        Contrast vector (1D, t-test) or contrast matrix (2D, F-test).

    Returns
    -------
    X_reduced : ndarray of shape (n_subjects, p_reduced)
        Reduced design matrix (columns not touched by the contrast).
    """
    contrast = np.asarray(contrast)
    if contrast.ndim == 1:
        keep_cols = np.where(contrast == 0)[0]
    else:
        # 2D: keep columns that are zero across ALL contrast rows
        keep_cols = np.where(np.all(contrast == 0, axis=0))[0]

    if len(keep_cols) == 0:
        # All columns targeted — reduced model is just intercept
        return np.ones((X.shape[0], 1), dtype=np.float64)
    return X[:, keep_cols]


def _freedman_lane_permutation_task(
    Y_hat_reduced: npt.NDArray[np.float64],
    residuals_reduced: npt.NDArray[np.float64],
    X: npt.NDArray[np.float64],
    contrast: npt.NDArray[np.float64],
    X_pinv: npt.NDArray[np.float64],
    XtX_inv_diag: npt.NDArray[np.float64],
    XtX_inv: npt.NDArray[np.float64],
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
    X, contrast, X_pinv, XtX_inv_diag, XtX_inv : precomputed GLM quantities
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
        XtX_inv=XtX_inv,
    )

    if enhance_func is not None:
        stat_dict = enhance_func(stat_dict, **enhance_kwargs)

    return _extract_max_stats(stat_dict, reference_shape)


# =============================================================================
# Enhancement registry
# =============================================================================
# Wrappers live in _enhancement.py and are shared with the t-test pipeline —
# see pairwise_stats.compute_p_val.

_ENHANCE_MAP = {
    StatMethod.TSTAT: None,  # No enhancement
    StatMethod.TFNBS: apply_tfnbs,
    StatMethod.NBS: apply_nbs,
    StatMethod.CNBS: apply_cnbs,
    StatMethod.NI_TFNBS: apply_ni_tfnbs,
    StatMethod.FBC_TFNBS: apply_fbc_tfnbs,
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
    acceleration: Optional[str] = None,
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
    contrast : ndarray of shape (p,) or (r, p), optional
        Contrast vector (1D, t/beta) or contrast matrix (2D, F-stat).
        Required with ``design_matrix``. For ``stat_type='fstat'``, pass a
        2D matrix whose rows encode the linear combinations being tested
        jointly (e.g., ``[[1, -1, 0], [1, 0, -1]]`` tests β₁=β₂=β₃).
    interest : ndarray of shape (n_subjects,), optional
        Variable of interest (convenience API). Mutually exclusive with
        ``design_matrix``. Produces a single-column contrast — use the
        advanced API for multi-row F-contrasts.
    confounds : ndarray of shape (n_subjects,) or (n_subjects, q), optional
        Confound variables (convenience API).
    stat_type : {'tstat', 'beta', 'fstat'} or GLMStatType, default='tstat'
        Type of statistic to compute. ``'fstat'`` enables an omnibus F-test
        for a multi-row contrast matrix and returns a single ``'omnibus'``
        p-value map (no positive/negative split).
    method : str or StatMethod, default='tfnbs'
        Enhancement method.
    n_permutations : int, default=1000
        Number of permutations for null distribution.
    two_tailed : bool, default=False
        If False (default), per-tail FWER control (separate null for positive
        and negative). If True, joint null from max(max_positive, max_negative).
    acceleration : {'gpd', 'gamma'} or None, default=None
        Permutation acceleration method (Winkler et al., 2016). If None,
        use standard empirical p-values. 'gpd' fits a Generalized Pareto
        Distribution to the tail of the null distribution. 'gamma' fits
        a gamma distribution using method of moments. Both allow accurate
        p-values with fewer permutations (~200 instead of ~5000).
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
        For ``stat_type`` in ``{'tstat', 'beta'}``: ``{'positive', 'negative'}``
        p-value arrays of shape (N, N). For ``stat_type='fstat'``: a single
        ``{'omnibus'}`` p-value array of shape (N, N).

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

    # Contrast may be 1D (t/beta) or 2D (F-stat). Check the column dimension.
    if contrast_vec.ndim == 1:
        if contrast_vec.shape[0] != p:
            raise ValueError(
                f"Contrast vector length ({contrast_vec.shape[0]}) must match "
                f"number of columns in design matrix ({p})."
            )
    elif contrast_vec.ndim == 2:
        if contrast_vec.shape[1] != p:
            raise ValueError(
                f"Contrast matrix has {contrast_vec.shape[1]} columns but design "
                f"matrix has {p}."
            )
    else:
        raise ValueError(
            f"Contrast must be 1D or 2D, got ndim={contrast_vec.ndim}."
        )
    if np.all(contrast_vec == 0):
        raise ValueError("Contrast must have at least one non-zero entry.")

    if acceleration is not None and acceleration not in ("gpd", "gamma"):
        raise ValueError(
            f"Invalid acceleration: '{acceleration}'. Must be 'gpd', 'gamma', or None."
        )

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
    X_pinv, XtX_inv_diag, XtX_inv = _precompute_ols(X)

    # ---- Compute observed statistics ----
    emp_stat_dict = compute_glm_stat(
        Y, X, contrast_vec,
        stat_type=stat_type_str,
        X_pinv=X_pinv,
        XtX_inv_diag=XtX_inv_diag,
        XtX_inv=XtX_inv,
    )
    if enhance_func is not None:
        emp_stat_dict = enhance_func(emp_stat_dict, **enhance_kwargs)

    # ---- Freedman-Lane: compute reduced model ----
    X_reduced = _compute_reduced_model(X, contrast_vec)
    X_red_pinv, _, _ = _precompute_ols(X_reduced)

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
        XtX_inv,
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
        if stat_type_str == GLMStatType.FSTAT.value:
            # F is non-negative; there is no sign to split and nothing to pool.
            # Silently ignore rather than raise — keeps the API uniform for
            # callers that pass two_tailed=True by default.
            pass
        else:
            # Build joint null from max(max_positive, max_negative)
            joint_null = np.maximum(
                max_null_dict["positive"],
                max_null_dict["negative"],
            )
            max_null_dict = {
                "positive": joint_null,
                "negative": joint_null,
            }

    # ---- Compute p-values ----
    if acceleration is not None:
        return compute_p_values_accelerated(
            emp_stat_dict, max_null_dict, method=acceleration,
        )
    return _compute_p_values_from_null(emp_stat_dict, max_null_dict)


# =============================================================================
# Paired-difference GLM convenience wrapper
# =============================================================================

def compute_p_val_paired_glm(
    Y_A: npt.NDArray[np.float64],
    Y_B: npt.NDArray[np.float64],
    confounds_A: Optional[npt.NDArray[np.float64]] = None,
    confounds_B: Optional[npt.NDArray[np.float64]] = None,
    **kwargs,
) -> Dict[str, npt.NDArray[np.float64]]:
    """
    Paired-difference GLM: test ``Y_A`` vs ``Y_B`` within-subject, optionally
    controlling for per-subject differences in confounds.

    Two routes depending on ``confounds_*``:

    - **No confounds** → delegates to
      :func:`conninfpy.compute_p_val` with ``test_type='paired'`` (sign-flip
      permutation). This is numerically equivalent to a one-sample GLM on
      the differences and avoids the Freedman-Lane degeneracy that occurs
      when the contrast targets the only column in the design matrix.
    - **With confounds** → builds ``Δ_Y = Y_A − Y_B`` and
      ``Δ_C = confounds_A − confounds_B``, then calls
      :func:`compute_p_val_glm` with ``design_matrix = [1, Δ_C]``,
      ``contrast = [1, 0, ..., 0]`` (intercept as the effect of interest).

    Use this when the same subjects are measured under two conditions (task
    A vs task B, pre vs post, open vs close) and a confound whose value
    *differs between the two conditions for the same subject* (e.g.,
    motion, arousal, response time) needs to be partialled out.

    Parameters
    ----------
    Y_A, Y_B : ndarray of shape (n_subjects, N, N)
        Aligned per-subject connectivity matrices for the two conditions.
        ``Y_A[s]`` and ``Y_B[s]`` must correspond to the same subject.
    confounds_A, confounds_B : ndarray of shape (n_subjects,) or (n_subjects, k), optional
        Aligned per-subject confound values for the two conditions. Either
        pass both or neither. If passed, the *difference*
        ``confounds_A - confounds_B`` enters the GLM as a nuisance regressor.
    **kwargs
        Forwarded to the downstream call (``compute_p_val`` or
        ``compute_p_val_glm``). ``test_type`` is fixed to ``'paired'`` when
        routing to the t-test pipeline and must not be supplied.

    Returns
    -------
    dict
        For the no-confound path: ``{'g2>g1', 'g1>g2'}`` (matches
        ``compute_p_val(test_type='paired')`` output; ``g2>g1`` is the
        ``Y_B > Y_A`` tail, i.e. condition B has greater connectivity).
        For the with-confound path: ``{'positive', 'negative'}`` referring
        to the sign of the ``Y_A - Y_B`` intercept, adjusted for ``Δ_C``.

    Raises
    ------
    ValueError
        If ``Y_A`` and ``Y_B`` shapes mismatch, only one of the confound
        arrays is supplied, or confound shapes mismatch.
    """
    Y_A = np.asarray(Y_A, dtype=np.float64)
    Y_B = np.asarray(Y_B, dtype=np.float64)
    if Y_A.shape != Y_B.shape:
        raise ValueError(
            f"Y_A and Y_B must have identical shapes, got {Y_A.shape} vs {Y_B.shape}."
        )
    if Y_A.ndim != 3 or Y_A.shape[1] != Y_A.shape[2]:
        raise ValueError(
            f"Y_A/Y_B must have shape (n_subjects, N, N), got {Y_A.shape}."
        )

    has_A = confounds_A is not None
    has_B = confounds_B is not None
    if has_A != has_B:
        raise ValueError(
            "Provide either both confounds_A and confounds_B, or neither. "
            "Paired-difference inference requires matched confound values "
            "in both conditions."
        )

    # --- No-confound path: delegate to sign-flip paired test ---
    if not has_A:
        if "test_type" in kwargs:
            raise ValueError(
                "test_type is fixed to 'paired' by compute_p_val_paired_glm; "
                "do not pass it explicitly."
            )
        return compute_p_val(Y_A, Y_B, test_type="paired", **kwargs)

    # --- With-confound path: one-sample GLM on differences ---
    confounds_A = np.asarray(confounds_A, dtype=np.float64)
    confounds_B = np.asarray(confounds_B, dtype=np.float64)
    if confounds_A.shape != confounds_B.shape:
        raise ValueError(
            f"confounds_A and confounds_B must have identical shapes, "
            f"got {confounds_A.shape} vs {confounds_B.shape}."
        )
    if confounds_A.shape[0] != Y_A.shape[0]:
        raise ValueError(
            f"confounds have {confounds_A.shape[0]} subjects but Y_A has "
            f"{Y_A.shape[0]}."
        )

    Delta_Y = Y_A - Y_B
    Delta_C = confounds_A - confounds_B
    if Delta_C.ndim == 1:
        Delta_C = Delta_C[:, np.newaxis]

    n_subjects = Y_A.shape[0]
    intercept = np.ones((n_subjects, 1), dtype=np.float64)
    X = np.hstack([intercept, Delta_C])
    contrast = np.zeros(X.shape[1], dtype=np.float64)
    contrast[0] = 1.0  # test the intercept (= mean of Δ_Y)

    # Disallow caller from overriding the design; these keys conflict.
    for forbidden in ("design_matrix", "contrast", "interest", "confounds"):
        if forbidden in kwargs:
            raise ValueError(
                f"compute_p_val_paired_glm builds the design internally; "
                f"do not pass '{forbidden}'."
            )

    return compute_p_val_glm(
        Delta_Y,
        design_matrix=X,
        contrast=contrast,
        **kwargs,
    )
