"""
Permutation acceleration methods for faster inference.

Implements tail approximation via Generalized Pareto Distribution (GPD)
and gamma approximation, following Winkler et al. (2016) "Faster
permutation inference in brain imaging", NeuroImage.

These methods reduce the number of permutations needed for accurate
p-value estimation from ~5000 to ~200, yielding 10-100x speedup
while maintaining exact error rates for FWER-corrected inference.

References
----------
Winkler et al. (2016). Faster permutation inference in brain imaging.
    NeuroImage, doi:10.1016/j.neuroimage.2016.05.068
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


__all__ = [
    "fit_gpd_tail",
    "fit_gamma_tail",
    "compute_p_values_accelerated",
]


def _fit_gpd_mom(
    exceedances: npt.NDArray[np.float64],
) -> tuple[float, float, bool]:
    """
    Fit GPD parameters by method of moments.

    Parameters
    ----------
    exceedances : ndarray
        Positive values y = T - u for T > u (exceedances above threshold u).

    Returns
    -------
    sigma : float
        Scale parameter.
    xi : float
        Shape parameter.
    valid : bool
        Whether the fit is valid (enough exceedances, finite parameters).
    """
    n = len(exceedances)
    if n < 10:
        return 0.0, 0.0, False

    y_bar = np.mean(exceedances)
    s2 = np.var(exceedances, ddof=1)

    if y_bar <= 0 or s2 <= 0:
        return 0.0, 0.0, False

    # Method of moments (Hosking & Wallis, 1987)
    ratio = y_bar ** 2 / s2
    xi = (ratio - 1) / 2
    sigma = y_bar * (ratio + 1) / 2

    if sigma <= 0 or not np.isfinite(xi) or not np.isfinite(sigma):
        return 0.0, 0.0, False

    return sigma, xi, True


def _gpd_sf(
    x: npt.NDArray[np.float64],
    sigma: float,
    xi: float,
) -> npt.NDArray[np.float64]:
    """
    GPD survival function P(Y > x).

    Parameters
    ----------
    x : ndarray
        Values at which to evaluate.
    sigma : float
        Scale parameter.
    xi : float
        Shape parameter.

    Returns
    -------
    ndarray
        Survival function values.
    """
    x = np.asarray(x, dtype=np.float64)

    if abs(xi) < 1e-10:
        # Exponential case (xi → 0)
        return np.exp(-x / sigma)

    z = 1 + xi * x / sigma
    # Ensure z > 0 for valid domain
    z = np.maximum(z, 0.0)
    return np.power(z, -1.0 / xi)


def _anderson_darling_gpd(
    exceedances: npt.NDArray[np.float64],
    sigma: float,
    xi: float,
) -> float:
    """
    Anderson-Darling test statistic for GPD fit.

    Lower values indicate better fit. A rough threshold of 2.5
    indicates acceptable fit (Choulakian & Stephens, 2001).

    Parameters
    ----------
    exceedances : sorted ndarray
        Exceedances in ascending order.
    sigma, xi : float
        GPD parameters.

    Returns
    -------
    float
        Anderson-Darling statistic.
    """
    n = len(exceedances)
    if n < 5:
        return np.inf

    # CDF values: F(y) = 1 - SF(y)
    F = 1.0 - _gpd_sf(exceedances, sigma, xi)
    F = np.clip(F, 1e-15, 1 - 1e-15)

    i = np.arange(1, n + 1, dtype=np.float64)
    A2 = -n - np.sum((2 * i - 1) * (np.log(F) + np.log(1 - F[::-1]))) / n

    return A2


def fit_gpd_tail(
    null_dist: npt.NDArray[np.float64],
    observed: npt.NDArray[np.float64],
    n_thresholds: int = 5,
    ad_threshold: float = 2.5,
) -> npt.NDArray[np.float64]:
    """
    Compute p-values using GPD tail approximation.

    Fits a GPD to the upper tail of the null distribution and uses
    the fitted distribution to compute p-values for the observed
    statistics. Falls back to empirical p-values when GPD fit is poor.

    Algorithm (Winkler et al., 2016, Section 2.2.3):

    1. Set initial threshold ``u`` at upper quartile of ``null_dist``
    2. Fit GPD to exceedances ``y = T_star_j - u`` by method of moments
    3. Test fit with Anderson-Darling; if poor, increase ``u`` and retry
    4. Compute ``p = (k / J) * GPD_sf(T - u)`` where ``k`` = #exceedances

    Parameters
    ----------
    null_dist : ndarray of shape (J,)
        Max-statistic null distribution from permutations.
    observed : ndarray of shape ``(*spatial_dims,)``
        Observed statistics (e.g., shape (N, N) for connectivity).
    n_thresholds : int, default=5
        Maximum number of threshold increases to try for good GPD fit.
    ad_threshold : float, default=2.5
        Anderson-Darling threshold for acceptable fit.

    Returns
    -------
    p_values : ndarray of same shape as observed
        P-values. Uses GPD where fit is good, empirical otherwise.
    """
    J = len(null_dist)
    null_sorted = np.sort(null_dist)

    # Empirical p-values as fallback (with +1 correction — Phipson & Smyth 2010)
    count = np.sum(observed[..., np.newaxis] < null_dist, axis=-1)
    p_empirical = (count + 1.0) / (J + 1.0)

    # Try increasing thresholds for GPD fit
    quantiles = np.linspace(0.50, 0.90, n_thresholds)

    for q in quantiles:
        u = np.quantile(null_sorted, q)
        exceedances = null_sorted[null_sorted > u] - u

        if len(exceedances) < 10:
            continue

        sigma, xi, valid = _fit_gpd_mom(exceedances)
        if not valid:
            continue

        # Check fit quality
        ad_stat = _anderson_darling_gpd(exceedances, sigma, xi)
        if ad_stat > ad_threshold:
            continue

        # Good fit found — compute p-values from GPD
        k = len(exceedances)  # number of exceedances
        tail_fraction = k / J  # P(T > u)

        # For observed values > u: p = tail_fraction * GPD_sf(T - u)
        # For observed values <= u: use empirical p-value
        above_u = observed > u
        p_gpd = np.copy(p_empirical)

        if np.any(above_u):
            gpd_tail_p = tail_fraction * _gpd_sf(observed[above_u] - u, sigma, xi)
            # Ensure p-values are in [1/J, 1] (can't be more precise than 1/J)
            gpd_tail_p = np.clip(gpd_tail_p, 1.0 / J, 1.0)
            p_gpd[above_u] = gpd_tail_p

        logger.debug(
            f"GPD fit: u={u:.3f}, k={k}, xi={xi:.4f}, "
            f"sigma={sigma:.4f}, AD={ad_stat:.3f}"
        )
        return p_gpd

    # No good fit found — fall back to empirical
    logger.debug("GPD fit failed at all thresholds, using empirical p-values")
    return p_empirical


def fit_gamma_tail(
    null_dist: npt.NDArray[np.float64],
    observed: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Compute p-values using gamma (Pearson type III) approximation.

    Fits a gamma distribution to the null distribution using method
    of moments (first three moments) and computes p-values from the
    fitted distribution. Simpler and more robust than GPD.

    Parameters
    ----------
    null_dist : ndarray of shape (J,)
        Max-statistic null distribution.
    observed : ndarray of shape ``(*spatial_dims,)``
        Observed statistics.

    Returns
    -------
    p_values : ndarray of same shape as observed
        P-values from fitted gamma distribution.
    """
    from scipy import stats

    J = len(null_dist)

    mean = np.mean(null_dist)
    var = np.var(null_dist, ddof=1)
    skew_val = float(stats.skew(null_dist))

    if var <= 0 or abs(skew_val) < 1e-10:
        # Can't fit gamma, fall back to empirical (with +1 correction)
        count = np.sum(observed[..., np.newaxis] < null_dist, axis=-1)
        return (count + 1.0) / (J + 1.0)

    # Gamma parameters from moments
    # shape a = 4/skew^2, scale b = sqrt(var / a), loc = mean - a*b
    a = 4.0 / (skew_val ** 2)
    b = np.sqrt(var / a)
    loc = mean - a * b

    # P-values from gamma survival function
    p_values = stats.gamma.sf(observed, a, loc=loc, scale=b)
    p_values = np.clip(p_values, 1.0 / J, 1.0)

    return p_values


def compute_p_values_accelerated(
    emp_stat_dict: Dict[str, npt.NDArray],
    max_null_dict: Dict[str, npt.NDArray],
    method: str = "gpd",
) -> Dict[str, npt.NDArray]:
    """
    Compute p-values using accelerated tail approximation.

    Drop-in replacement for _compute_p_values_from_null that uses
    GPD or gamma fitting instead of pure empirical counting.

    Parameters
    ----------
    emp_stat_dict : dict
        Dictionary with empirical statistic arrays (any keys).
    max_null_dict : dict
        Dictionary with null distribution arrays.
    method : {'gpd', 'gamma'}, default='gpd'
        Acceleration method.

    Returns
    -------
    dict
        Dictionary with p-value arrays for each key.
    """
    fit_func = fit_gpd_tail if method == "gpd" else fit_gamma_tail
    p_values = {}

    for key in emp_stat_dict:
        emp = emp_stat_dict[key]
        null = max_null_dict[key]

        if emp.ndim == 2:
            # Standard case: (N, N) vs (J,)
            p_values[key] = fit_func(null, emp)
        else:
            # Multi-param: (N, N, n_params) vs (J, n_params)
            n_params = emp.shape[-1]
            p_list = []
            for p_idx in range(n_params):
                p_slice = fit_func(null[:, p_idx], emp[..., p_idx])
                p_list.append(p_slice)
            p_values[key] = np.stack(p_list, axis=-1)

    return p_values
