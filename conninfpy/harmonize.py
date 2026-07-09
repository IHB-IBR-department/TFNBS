"""
Multi-site harmonization for connectivity data.

Implements parametric empirical-Bayes ComBat (Johnson, Li & Rabinovic 2007;
Fortin et al. 2017) directly in numpy — no dependency on ``neuroHarmonize``
or ``neurocombat``. Plus design-matrix diagnostics (VIF, condition number).

ComBat model per feature j:

    Y_ij_s = α_j + X_i β_j + γ_sj + δ_sj · ε_ij_s,   ε ~ N(0, σ_j²)

where ``s`` indexes site, ``i`` indexes subject within site, and ``X_i`` holds
covariates to preserve (age, sex, diagnosis). The parameterization follows
Fortin 2017 / neuroCombat: site dummies are encoded one-hot (no reference
site dropped); ``α`` is the sample-size-weighted grand mean of per-site
intercepts; ``σ²`` uses the biased ``var.pooled`` denominator (divide by
``n``, not ``n − p``). This matches the canonical neuroCombat reference
implementation to machine precision under EB and without — validated in
``tests/test_combat_equivalence.py``.

Estimation:

1. Fit OLS with site dummies + preserved covariates to obtain per-site
   intercepts, ``β̂``, and pooled ``σ̂_j``.
2. Standardize residuals per feature.
3. Estimate per-(site, feature) location ``γ̂`` and scale ``δ̂²`` from the
   standardized data.
4. Empirical Bayes shrinkage using site-level conjugate priors, iterated to
   convergence.
5. Adjust: subtract the shrunken site effect, re-add the preserved-covariate
   fit and the pooled scale.

Use this when your cohort pools subjects from multiple scanners/sites and
you want to remove the site-aligned variance that is *orthogonal* to the
biological covariates you care about. Fortin 2018 discusses the limits.

Typical use
-----------
>>> from conninfpy import combat_harmonize
>>> result = combat_harmonize(Y, sites=site_labels, preserve=covariates)
>>> Y_adj = result.Y_adjusted   # same shape as Y
>>> result.diagnostics           # ratio of between-site variance before/after

References
----------
Johnson WE, Li C, Rabinovic A (2007). Adjusting batch effects in microarray
expression data using empirical Bayes methods. Biostatistics 8(1):118-27.

Fortin J-P, Parker D, Tunç B, et al. (2017). Harmonization of multi-site
diffusion tensor imaging data. NeuroImage 161:149-170.

Fortin J-P et al. (2018). Harmonization of cortical thickness measurements
across scanners and sites. NeuroImage 167:104-120.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt


# Floor applied to variance estimates inside ComBat. Matches the
# `np.maximum(·, _VAR_FLOOR)` calls in `combat_fit`. Below this value we
# emit a RuntimeWarning so catastrophic cancellations (e.g. Fisher-z near
# saturation, |r|→1, where within-site var collapses to zero) don't pass
# silently. Clip behavior is unchanged.
_VAR_FLOOR = 1e-12


def _warn_variance_floor(
    var: npt.NDArray[np.float64],
    stage: str,
    *,
    site_label: Optional[Any] = None,
) -> None:
    """Emit a RuntimeWarning listing how many entries fell at/below the floor.

    ``stage`` names the call site ('pooled sigma', 'per-site delta',
    'EB-shrunken delta'); ``site_label`` is the human-readable site name if
    the offending entries are per-site.
    """
    bad = np.asarray(var <= _VAR_FLOOR)
    if not bad.any():
        return
    n_bad = int(bad.sum())
    finite = var[np.isfinite(var)]
    min_val = float(finite.min()) if finite.size else float("nan")
    where = f" at site {site_label!r}" if site_label is not None else ""
    warnings.warn(
        f"ComBat: {n_bad} feature(s){where} had variance ≤ {_VAR_FLOOR:.1e} "
        f"at '{stage}' (min value = {min_val:.3e}); clipped to {_VAR_FLOOR:.1e}. "
        "This commonly indicates catastrophic cancellation (Fisher-z near "
        "saturation |r|→1) or a within-site constant feature.",
        RuntimeWarning,
        stacklevel=3,
    )


__all__ = [
    "ComBatModel",
    "CombatResult",
    "combat_harmonize",
    "combat_fit",
    "combat_apply",
    "compute_vif",
    "design_diagnostics",
    "flatten_upper",
    "unflatten_upper",
    "block_mass",
]


# =============================================================================
# Flat <-> matrix conversions (promoted from examples/abide_validation)
# =============================================================================

def flatten_upper(
    Y: npt.NDArray[np.float64],
) -> Tuple[npt.NDArray[np.float64], Tuple[npt.NDArray, npt.NDArray], int]:
    """
    Flatten the upper triangle (k=1) of a (n, N, N) connectivity stack.

    Parameters
    ----------
    Y : ndarray of shape (n_subjects, N, N)
        Symmetric connectivity matrices with zero diagonal.

    Returns
    -------
    features : ndarray of shape (n_subjects, N*(N-1)/2)
        Flattened upper triangles — one feature per edge.
    triu_idx : tuple of ndarray
        ``(rows, cols)`` index tuple usable as ``M[triu_idx]`` to invert.
    N : int
        Side length of the original matrices.
    """
    if Y.ndim != 3 or Y.shape[1] != Y.shape[2]:
        raise ValueError(f"Y must have shape (n, N, N), got {Y.shape}.")
    N = Y.shape[1]
    triu_idx = np.triu_indices(N, k=1)
    features = Y[:, triu_idx[0], triu_idx[1]]
    return features, triu_idx, N


def unflatten_upper(
    features: npt.NDArray[np.float64],
    triu_idx: Tuple[npt.NDArray, npt.NDArray],
    N: int,
) -> npt.NDArray[np.float64]:
    """
    Reconstruct (n, N, N) symmetric matrices (zero diagonal) from flattened
    upper-triangle features.
    """
    n = features.shape[0]
    Y = np.zeros((n, N, N), dtype=features.dtype)
    Y[:, triu_idx[0], triu_idx[1]] = features
    Y[:, triu_idx[1], triu_idx[0]] = features
    return Y


# =============================================================================
# Site encoding
# =============================================================================

def _encode_sites(sites: Sequence[Any]) -> Tuple[npt.NDArray[np.int_], List[Any]]:
    """Return integer site codes and the ordered list of unique site labels."""
    unique, inverse = np.unique(np.asarray(sites), return_inverse=True)
    return inverse.astype(np.int_), list(unique)


def _site_dummies(site_codes: npt.NDArray[np.int_], n_sites: int) -> npt.NDArray[np.float64]:
    """One-hot site dummies: ``n_sites`` columns, no reference dropped.

    Canonical Fortin 2017 / Johnson 2007 parameterization — the per-site
    intercepts in the OLS fit are then per-site feature means, and the
    grand-mean α is recovered as a sample-size-weighted average of them.
    Combined with `preserve` columns, the design ``[D | preserve]`` is
    full-rank (the one-hot dummies sum to ones but the preserve block
    typically has nontrivial variation; no separate intercept needed).
    """
    n = site_codes.shape[0]
    D = np.zeros((n, n_sites), dtype=np.float64)
    for s in range(n_sites):
        D[:, s] = (site_codes == s).astype(np.float64)
    return D


# =============================================================================
# ComBat model (dataclass state for fit → apply)
# =============================================================================

@dataclass
class ComBatModel:
    """Fitted parametric ComBat state, for later application to new data."""

    site_labels: List[Any]
    alpha: npt.NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    beta: Optional[npt.NDArray[np.float64]] = None
    sigma: npt.NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    gamma_star: npt.NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    delta_star: npt.NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    eb: bool = True
    preserve_n_cols: int = 0


@dataclass
class CombatResult:
    """Return value of :func:`combat_harmonize`."""

    Y_adjusted: npt.NDArray[np.float64]
    model: ComBatModel
    diagnostics: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Empirical Bayes priors (method of moments)
# =============================================================================

def _aprior(delta_sq: npt.NDArray[np.float64]) -> float:
    """Method-of-moments shape λ for inverse-gamma prior on δ²."""
    m = float(np.mean(delta_sq))
    v = float(np.var(delta_sq, ddof=1)) if delta_sq.size > 1 else 0.0
    if v <= 0:
        return np.inf
    return (2.0 * v + m ** 2) / v


def _bprior(delta_sq: npt.NDArray[np.float64]) -> float:
    """Method-of-moments scale θ for inverse-gamma prior on δ²."""
    m = float(np.mean(delta_sq))
    v = float(np.var(delta_sq, ddof=1)) if delta_sq.size > 1 else 0.0
    if v <= 0:
        return 0.0
    return (m * v + m ** 3) / v


def _eb_update(
    Z_s: npt.NDArray[np.float64],
    gamma_hat: npt.NDArray[np.float64],
    delta_sq_hat: npt.NDArray[np.float64],
    gamma_bar: float,
    tau_sq: float,
    lam: float,
    theta: float,
    tol: float = 1e-4,
    max_iter: int = 500,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Johnson 2007 Appendix iterative EB estimator for (γ*, δ²*) at one site.

    Z_s : (n_s, p) standardized residuals for subjects in site s.
    gamma_hat, delta_sq_hat : (p,) starting estimates (method of moments).
    """
    n_s = Z_s.shape[0]
    gamma_old = gamma_hat.copy()
    delta_sq_old = delta_sq_hat.copy()
    for _ in range(max_iter):
        # γ update
        gamma_new = (n_s * tau_sq * gamma_hat + delta_sq_old * gamma_bar) / (
            n_s * tau_sq + delta_sq_old
        )
        # δ² update
        resid = Z_s - gamma_new[np.newaxis, :]
        ss = np.sum(resid ** 2, axis=0)
        delta_sq_new = (theta + 0.5 * ss) / (n_s / 2.0 + lam - 1.0)

        change = max(
            float(np.max(np.abs(gamma_new - gamma_old))),
            float(np.max(np.abs(delta_sq_new - delta_sq_old))),
        )
        gamma_old, delta_sq_old = gamma_new, delta_sq_new
        if change < tol:
            break
    return gamma_old, delta_sq_old


# =============================================================================
# Core fit + apply
# =============================================================================

def _prepare_inputs(
    Y: npt.NDArray[np.float64],
    preserve: Optional[npt.NDArray[np.float64]],
) -> Tuple[npt.NDArray[np.float64], Optional[npt.NDArray[np.float64]], Tuple[int, ...], Optional[Tuple]]:
    """Flatten (n, N, N) → (n, p) if needed; validate ``preserve``."""
    Y = np.asarray(Y, dtype=np.float64)
    matrix_shape = None
    triu_idx = None
    if Y.ndim == 3:
        if Y.shape[1] != Y.shape[2]:
            raise ValueError(f"3D Y must have shape (n, N, N), got {Y.shape}.")
        features, triu_idx, N = flatten_upper(Y)
        matrix_shape = (N, N)
    elif Y.ndim == 2:
        features = Y
    else:
        raise ValueError(f"Y must be 2D (n, p) or 3D (n, N, N); got ndim={Y.ndim}.")

    if preserve is not None:
        preserve = np.asarray(preserve, dtype=np.float64)
        if preserve.ndim == 1:
            preserve = preserve[:, np.newaxis]
        if preserve.shape[0] != features.shape[0]:
            raise ValueError(
                f"preserve has {preserve.shape[0]} rows but Y has "
                f"{features.shape[0]} subjects."
            )
    return features, preserve, matrix_shape, triu_idx


def combat_fit(
    Y: npt.NDArray[np.float64],
    sites: Sequence[Any],
    preserve: Optional[npt.NDArray[np.float64]] = None,
    eb: bool = True,
) -> ComBatModel:
    """
    Fit parametric ComBat on a training cohort.

    Parameters
    ----------
    Y : ndarray of shape (n, N, N) or (n, p)
        Connectivity matrices (3D) or pre-flattened features (2D).
    sites : sequence of length n
        Site label per subject. Any hashable (strings, ints).
    preserve : ndarray of shape (n,) or (n, k), optional
        Covariates whose effect on Y should be preserved (e.g. diagnosis,
        age, sex). Their fitted coefficients survive harmonization.
    eb : bool, default=True
        If True (standard ComBat), apply empirical-Bayes shrinkage on the
        per-(site, feature) effects. If False, use the raw method-of-moments
        estimates — faster but noisier for small per-site sample sizes.

    Returns
    -------
    ComBatModel
        Fitted state, usable with :func:`combat_apply`.
    """
    features, preserve, _, _ = _prepare_inputs(Y, preserve)
    n, p = features.shape
    site_codes, site_labels = _encode_sites(sites)
    n_sites = len(site_labels)
    if n_sites < 2:
        raise ValueError(
            f"ComBat needs ≥ 2 sites, got {n_sites}. If you have one site, "
            f"no harmonization is needed."
        )

    # Build design: [site_dummies (all K, one-hot) | preserve_cols].
    # Canonical Fortin 2017 / neuroCombat parameterization — no reference
    # site dropped; α is recovered as the sample-size-weighted average of
    # the per-site intercepts. No separate intercept column needed (one-hot
    # dummies already span the constants).
    D = _site_dummies(site_codes, n_sites)              # (n, n_sites)
    cols = [D]
    k = 0
    if preserve is not None:
        cols.append(preserve)
        k = preserve.shape[1]
    X_full = np.hstack(cols)

    # OLS fit. First n_sites rows of beta_full = per-site intercepts
    # (per-site feature mean after preserve adjustment); next k rows = β̂.
    beta_full, *_ = np.linalg.lstsq(X_full, features, rcond=None)

    # α̂_j = weighted grand mean = Σ_s (n_s/n) · per_site_intercept_s
    n_per_site = np.array(
        [int(np.sum(site_codes == s)) for s in range(n_sites)],
        dtype=np.float64,
    )
    site_weights = n_per_site / float(n)                 # (n_sites,)
    alpha = site_weights @ beta_full[:n_sites]           # (p,)
    beta_cov = beta_full[n_sites:n_sites + k] if k > 0 else None

    # Pooled residual variance — Fortin 2017 var.pooled (biased MLE,
    # divide by n). This matches neuroCombat's reference convention.
    preds = X_full @ beta_full
    residuals = features - preds
    sigma_sq = np.sum(residuals ** 2, axis=0) / float(n)
    _warn_variance_floor(sigma_sq, "pooled sigma")
    sigma = np.sqrt(np.maximum(sigma_sq, _VAR_FLOOR))

    # Standardize: Z_ij = (Y_ij - α̂_j - X_i β̂_j) / σ̂_j
    # Note: site effects are NOT subtracted here — we want Z to retain them.
    preserve_fit = np.zeros_like(features)
    if beta_cov is not None:
        preserve_fit = preserve @ beta_cov
    Z = (features - alpha[np.newaxis, :] - preserve_fit) / sigma[np.newaxis, :]

    # Method-of-moments per (site, feature)
    gamma_hat = np.zeros((n_sites, p), dtype=np.float64)
    delta_sq_hat = np.ones((n_sites, p), dtype=np.float64)
    for s in range(n_sites):
        mask = site_codes == s
        if np.sum(mask) < 2:
            # Too few subjects to estimate scale; keep δ²=1
            gamma_hat[s] = np.mean(Z[mask], axis=0) if np.any(mask) else 0.0
            delta_sq_hat[s] = 1.0
            continue
        gamma_hat[s] = np.mean(Z[mask], axis=0)
        delta_sq_hat[s] = np.var(Z[mask], axis=0, ddof=1)
        _warn_variance_floor(
            delta_sq_hat[s], "per-site delta", site_label=site_labels[s]
        )
        delta_sq_hat[s] = np.maximum(delta_sq_hat[s], _VAR_FLOOR)

    if eb:
        gamma_star = np.zeros_like(gamma_hat)
        delta_star_sq = np.zeros_like(delta_sq_hat)
        for s in range(n_sites):
            mask = site_codes == s
            if np.sum(mask) < 2:
                gamma_star[s] = gamma_hat[s]
                delta_star_sq[s] = delta_sq_hat[s]
                continue
            g_bar = float(np.mean(gamma_hat[s]))
            t_sq = float(np.var(gamma_hat[s], ddof=1))
            lam = _aprior(delta_sq_hat[s])
            theta = _bprior(delta_sq_hat[s])
            if not np.isfinite(lam) or t_sq <= 0:
                # Degenerate prior → fall back to MoM
                gamma_star[s] = gamma_hat[s]
                delta_star_sq[s] = delta_sq_hat[s]
                continue
            gamma_star[s], delta_star_sq[s] = _eb_update(
                Z[mask], gamma_hat[s], delta_sq_hat[s],
                g_bar, t_sq, lam, theta,
            )
        _warn_variance_floor(delta_star_sq, "EB-shrunken delta")
        delta_star = np.sqrt(np.maximum(delta_star_sq, _VAR_FLOOR))
    else:
        gamma_star = gamma_hat
        delta_star = np.sqrt(delta_sq_hat)

    return ComBatModel(
        site_labels=site_labels,
        alpha=alpha,
        beta=beta_cov,
        sigma=sigma,
        gamma_star=gamma_star,
        delta_star=delta_star,
        eb=eb,
        preserve_n_cols=k,
    )


def combat_apply(
    model: ComBatModel,
    Y: npt.NDArray[np.float64],
    sites: Sequence[Any],
    preserve: Optional[npt.NDArray[np.float64]] = None,
) -> npt.NDArray[np.float64]:
    """
    Apply a fitted ComBat model to new data. Sites must all be present in
    the training cohort; unseen sites raise ValueError.
    """
    features, preserve, matrix_shape, triu_idx = _prepare_inputs(Y, preserve)
    n = features.shape[0]
    label_to_code = {lab: i for i, lab in enumerate(model.site_labels)}

    try:
        site_codes = np.array([label_to_code[s] for s in sites], dtype=np.int_)
    except KeyError as err:
        raise ValueError(
            f"Site {err.args[0]!r} was not present at fit time. "
            f"Known sites: {model.site_labels}"
        ) from None
    if site_codes.shape[0] != n:
        raise ValueError(f"sites length ({site_codes.shape[0]}) != n ({n}).")

    preserve_fit = np.zeros_like(features)
    if model.beta is not None:
        if preserve is None:
            raise ValueError("Model was fit with preserve; new data missing it.")
        if preserve.shape[1] != model.preserve_n_cols:
            raise ValueError(
                f"preserve has {preserve.shape[1]} cols, expected "
                f"{model.preserve_n_cols}."
            )
        preserve_fit = preserve @ model.beta

    # Standardize new data with fit-time α, β, σ
    Z = (features - model.alpha[np.newaxis, :] - preserve_fit) / model.sigma[np.newaxis, :]

    # Remove shrunken site effect
    gamma_per_sub = model.gamma_star[site_codes]
    delta_per_sub = model.delta_star[site_codes]
    Z_adj = (Z - gamma_per_sub) / delta_per_sub

    # Rescale back
    adjusted = Z_adj * model.sigma[np.newaxis, :] + model.alpha[np.newaxis, :] + preserve_fit

    if matrix_shape is not None:
        return unflatten_upper(adjusted, triu_idx, matrix_shape[0])
    return adjusted


def combat_harmonize(
    Y: npt.NDArray[np.float64],
    sites: Sequence[Any],
    preserve: Optional[npt.NDArray[np.float64]] = None,
    eb: bool = True,
    return_diagnostics: bool = True,
) -> CombatResult:
    """
    Fit + transform: parametric empirical-Bayes ComBat on the full cohort.

    See module docstring for the model and references. Handles either
    ``(n, N, N)`` connectivity matrices or ``(n, p)`` pre-flattened features.

    Parameters
    ----------
    Y : ndarray of shape (n, N, N) or (n, p)
        Input data.
    sites : sequence of length n
        Site label per subject.
    preserve : ndarray of shape (n,) or (n, k), optional
        Covariates whose variance should be preserved through ComBat.
        Pass the GLM nuisance design (age, sex, motion, ...) here and
        omit the variable that downstream inference will test — that
        avoids the Nygaard 2016 label leak in which the harmonization
        fit absorbs label-aligned variance the permutation null cannot
        recover. See :func:`~conninfpy.analyze`, which sets
        ``preserve = confounds`` automatically under the ``combat_only`` and
        ``combat_site_dummies_glm`` strategies, and
        [[paper_03_functionality]] / [[paper_04_use_cases]] for the
        software-paper strategy description and ABIDE comparison.
    eb : bool, default=True
        Apply empirical-Bayes shrinkage on site effects (recommended).
    return_diagnostics : bool, default=True
        If True, compute a between-site / within-site variance ratio before
        and after correction, plus the explicit
        ``between_site_variance_ratio_after_over_before`` key used by
        :func:`conninfpy.analyze` for residual-site-variance flags.

    Returns
    -------
    CombatResult
        ``Y_adjusted`` (same shape as input), ``model``, and ``diagnostics``
        (between-site variance ratio before/after, after/before ratio,
        ratio reduction, and per-site sample sizes).
    """
    model = combat_fit(Y, sites, preserve=preserve, eb=eb)
    Y_adj = combat_apply(model, Y, sites, preserve=preserve)

    diagnostics: Dict[str, Any] = {}
    if return_diagnostics:
        features_before, _, _, _ = _prepare_inputs(Y, None)
        features_after, _, _, _ = _prepare_inputs(Y_adj, None)
        site_codes, _ = _encode_sites(sites)

        def _between_over_total(feats):
            n_sites = int(site_codes.max()) + 1
            site_means = np.stack(
                [feats[site_codes == s].mean(axis=0) for s in range(n_sites)]
            )  # (n_sites, p)
            grand = feats.mean(axis=0)
            between = np.mean(np.var(site_means, axis=0, ddof=0))
            total = float(np.mean(np.var(feats, axis=0, ddof=0)))
            return between / max(total, 1e-12)

        ratio_before = _between_over_total(features_before)
        ratio_after = _between_over_total(features_after)
        ratio_after_over_before = ratio_after / max(ratio_before, 1e-12)
        diagnostics = {
            "between_site_variance_ratio_before": float(ratio_before),
            "between_site_variance_ratio_after": float(ratio_after),
            "between_site_variance_ratio_after_over_before": float(ratio_after_over_before),
            "ratio_reduction": float(1.0 - ratio_after_over_before),
            "site_labels": model.site_labels,
            "per_site_n": [int(np.sum(site_codes == s)) for s in range(len(model.site_labels))],
        }

    return CombatResult(Y_adjusted=Y_adj, model=model, diagnostics=diagnostics)


# =============================================================================
# Design diagnostics (VIF + condition number + correlation)
# =============================================================================

def compute_vif(X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Variance inflation factor per design-matrix column.

    ``VIF_j = 1 / (1 - R²_j)`` where ``R²_j`` is from regressing column j on
    all other columns. ``VIF > 5`` is a common concern threshold; ``> 10``
    is strong collinearity (Kutner et al. 2004).

    Constant columns (e.g. intercept) get ``VIF = 1`` by convention.

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Design matrix.

    Returns
    -------
    ndarray of shape (p,)
        VIF per column.
    """
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    vif = np.ones(p, dtype=np.float64)
    for j in range(p):
        x_j = X[:, j]
        if np.var(x_j) == 0:
            # Constant column (e.g. intercept) → R² undefined; report 1
            vif[j] = 1.0
            continue
        others = np.delete(X, j, axis=1)
        if others.shape[1] == 0:
            vif[j] = 1.0
            continue
        # Add intercept to the "others" block to get the right R²
        others_full = np.hstack([np.ones((n, 1)), others])
        beta, *_ = np.linalg.lstsq(others_full, x_j, rcond=None)
        pred = others_full @ beta
        ss_res = float(np.sum((x_j - pred) ** 2))
        ss_tot = float(np.sum((x_j - x_j.mean()) ** 2))
        r_sq = 1.0 - ss_res / max(ss_tot, 1e-12)
        r_sq = min(max(r_sq, 0.0), 1.0 - 1e-12)
        vif[j] = 1.0 / (1.0 - r_sq)
    return vif


def design_diagnostics(
    X: npt.NDArray[np.float64],
    names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Diagnostic report for a GLM design matrix.

    Returns a dict with:
      - ``condition_number``: condition number of ``X'X`` (> 30 is caution,
        > 100 is strong concern)
      - ``vif``: VIF per column
      - ``vif_max``, ``vif_max_col``: most collinear column
      - ``correlation``: pairwise Pearson correlation matrix (p, p)
      - ``flags``: list of plain-English warnings when thresholds are crossed

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Design matrix (should include intercept if used in the GLM).
    names : sequence of str, optional
        Column names for reporting. Defaults to ``['x0', 'x1', ...]``.
    """
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    if names is None:
        names = [f"x{i}" for i in range(p)]
    elif len(names) != p:
        raise ValueError(f"names length ({len(names)}) must match ncols ({p}).")

    XtX = X.T @ X
    try:
        cond = float(np.linalg.cond(XtX))
    except np.linalg.LinAlgError:
        cond = np.inf

    vif = compute_vif(X)
    vif_argmax = int(np.argmax(vif))

    # Pairwise correlation (intercept → NaN row/col; report as 0)
    with np.errstate(invalid='ignore'):
        corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)

    flags = []
    if cond > 100:
        flags.append(
            f"Condition number {cond:.1f} > 100: strong multicollinearity — "
            f"coefficient estimates may be unstable."
        )
    elif cond > 30:
        flags.append(
            f"Condition number {cond:.1f} > 30: moderate multicollinearity."
        )
    if np.max(vif) > 10:
        flags.append(
            f"VIF for '{names[vif_argmax]}' = {vif[vif_argmax]:.1f} > 10: "
            f"strong collinearity."
        )
    elif np.max(vif) > 5:
        flags.append(
            f"VIF for '{names[vif_argmax]}' = {vif[vif_argmax]:.1f} > 5: "
            f"moderate collinearity."
        )

    return {
        "condition_number": cond,
        "vif": vif,
        "vif_max": float(vif[vif_argmax]),
        "vif_max_col": names[vif_argmax],
        "correlation": corr,
        "names": list(names),
        "flags": flags,
    }


# =============================================================================
# Block-mass aggregation (promoted from examples/abide_validation)
# =============================================================================

def block_mass(
    p_full: npt.NDArray[np.float64],
    net_labels: npt.NDArray[np.int_],
    alpha: float = 0.05,
    *,
    return_upper: bool = True,
) -> npt.NDArray[np.int_]:
    r"""Aggregate edge-level survival counts into a network-block matrix.

    For an upper-triangular set of significant edges (``p <= alpha``),
    count how many fall within each unordered pair of network labels
    ``(i, j)``. The diagonal counts within-network edges.

    Parameters
    ----------
    p_full : ndarray of shape (N, N)
        Edge-wise p-value matrix (typically symmetric, with diagonal=1).
    net_labels : ndarray of shape (N,)
        Integer network assignment per node, ``0..K-1``. The Yeo-7
        partition stored in ``abide_prepared.npz['net_labels']`` is the
        canonical use case.
    alpha : float, default ``0.05``
        Significance threshold; edges with ``p <= alpha`` are counted.
    return_upper : bool, default ``True``
        If ``True``, the returned matrix is upper-triangular (lower
        triangle zero-filled). If ``False``, the matrix is symmetric.

    Returns
    -------
    M : ndarray of shape (K, K), int
        Block-mass matrix where ``M[i, j]`` is the number of significant
        edges with one endpoint in network ``i`` and the other in ``j``.

    Examples
    --------
    >>> import numpy as np
    >>> from conninfpy import block_mass
    >>> p = np.array([[1, 0.01, 0.5], [0.01, 1, 0.6], [0.5, 0.6, 1]])
    >>> labels = np.array([0, 0, 1])
    >>> block_mass(p, labels, alpha=0.05)
    array([[1, 0],
           [0, 0]])
    """
    if p_full.ndim != 2 or p_full.shape[0] != p_full.shape[1]:
        raise ValueError(
            f"p_full must be (N, N), got {p_full.shape}."
        )
    if net_labels.shape[0] != p_full.shape[0]:
        raise ValueError(
            f"net_labels has {net_labels.shape[0]} entries but p_full is "
            f"{p_full.shape[0]} × {p_full.shape[0]}."
        )

    K = int(net_labels.max() + 1)
    M = np.zeros((K, K), dtype=np.int64)

    sig = (p_full <= alpha) & np.isfinite(p_full)
    np.fill_diagonal(sig, False)

    iu, ju = np.triu_indices_from(p_full, k=1)
    sig_idx = sig[iu, ju]
    iu, ju = iu[sig_idx], ju[sig_idx]

    bi = net_labels[iu]
    bj = net_labels[ju]
    lo = np.minimum(bi, bj)
    hi = np.maximum(bi, bj)
    np.add.at(M, (lo, hi), 1)

    if not return_upper:
        # mirror to lower triangle (off-diagonal only)
        M = M + np.triu(M, 1).T

    return M
