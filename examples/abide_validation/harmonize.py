"""
ComBat harmonisation for ABIDE connectivity (ValidationPlan §3.1).

Now uses the in-package :mod:`conninfpy.harmonize` ComBat implementation —
parametric empirical-Bayes (Johnson 2007) in pure numpy, no external
``neuroHarmonize`` / ``neurocombat`` dependency.

Path 1 (one-shot): fit ComBat on the full cohort while preserving biological
covariates (DX, AGE, SEX), then return harmonised connectivity matrices for
downstream GLM. Site-aligned variance orthogonal to the preserved covariates
is removed; site-aligned variance that co-varies with DX is (mostly) kept
(see Fortin 2018 for the theoretical argument and its limits).

Path 2 (fit/apply, cross-site ML): :func:`harmonize_fit` on training-site
subjects, then :func:`harmonize_apply` on held-out sites. Small-batch
empirical-Bayes gets unreliable below ~10 subjects per site; caller is
responsible for QC.

API kept for backward compatibility with earlier scripts. The ``smooth_terms``
argument is accepted but ignored (in-package ComBat is linear-only in the
preserved covariates); pass ``strict_smooth=True`` to make non-empty
``smooth_terms`` raise instead.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from conninfpy.harmonize import (
    combat_apply,
    combat_fit,
    combat_harmonize,
    ComBatModel,
    flatten_upper as _pkg_flatten,
    unflatten_upper as _pkg_unflatten,
)


__all__ = [
    "HarmonizedFeatures",
    "flatten_upper",
    "unflatten_upper",
    "build_combat_covars",
    "harmonize_fit_transform",
    "harmonize_fit",
    "harmonize_apply",
]


# =============================================================================
# Flat <-> matrix conversions (delegate to package helpers; kept for compat)
# =============================================================================


def flatten_upper(conn: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """(n_subjects, N, N) → (n_subjects, N*(N-1)/2) upper-triangle features.

    Returns ``(flat, iu)`` matching the original API. Delegates to
    :func:`conninfpy.harmonize.flatten_upper` for the math.
    """
    flat, iu, _ = _pkg_flatten(conn)
    return flat, iu


def unflatten_upper(flat: np.ndarray, N: int) -> np.ndarray:
    """Inverse of :func:`flatten_upper`; symmetric matrices with zero diagonal."""
    iu = np.triu_indices(N, k=1)
    return _pkg_unflatten(flat, iu, N)


# =============================================================================
# Covariate helpers (preserve = numpy stack, as package ComBat expects)
# =============================================================================


def build_combat_covars(
    site: np.ndarray,
    dx: Optional[np.ndarray] = None,
    age: Optional[np.ndarray] = None,
    sex: Optional[np.ndarray] = None,
    extra: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Build a covariates DataFrame compatible with the old neuroHarmonize API.

    Kept for backward compatibility — downstream callers that already assemble
    covariates this way still work. The ``harmonize_*`` functions below pull
    SITE + non-SITE columns apart and feed them to :mod:`conninfpy.harmonize`.

    Parameters
    ----------
    site : (n_subjects,) array of site labels.
    dx, age, sex : optional preserved covariates, one value per subject.
    extra : optional dict of ``column_name -> (n_subjects,)`` arrays.

    Returns
    -------
    covars : DataFrame with columns ``[SITE, (DX), (AGE), (SEX), ...]``.
    """
    n = len(site)
    cols = {"SITE": np.asarray(site)}
    if dx is not None:
        cols["DX"] = np.asarray(dx).astype(float)
    if age is not None:
        cols["AGE"] = np.asarray(age).astype(float)
    if sex is not None:
        cols["SEX"] = np.asarray(sex).astype(float)
    if extra:
        for k, v in extra.items():
            arr = np.asarray(v)
            if arr.shape[0] != n:
                raise ValueError(f"extra[{k}] length {arr.shape[0]} != n_subjects {n}")
            cols[k] = arr.astype(float)
    df = pd.DataFrame(cols)
    if df.isna().any().any():
        raise ValueError(
            "Covariates contain NaN — ComBat does not handle missing values. "
            "Impute or drop subjects first."
        )
    return df


def _split_site_and_preserve(covars: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """DataFrame → ``(sites, preserve_matrix)`` for :mod:`conninfpy.harmonize`."""
    if "SITE" not in covars.columns:
        raise ValueError("covars must contain a 'SITE' column.")
    sites = covars["SITE"].values
    other = covars.drop(columns=["SITE"])
    if other.shape[1] == 0:
        return sites, None
    # All non-SITE columns are coerced to float — the old API guaranteed this.
    return sites, other.values.astype(np.float64)


def _check_smooth(smooth_terms: Iterable[str], strict: bool) -> None:
    """In-package ComBat is linear-only; warn or raise on non-empty smooths."""
    terms = list(smooth_terms)
    if not terms:
        return
    msg = (
        f"smooth_terms={terms!r} is not supported by the in-package ComBat "
        f"(linear preserved covariates only). "
        f"For age non-linearity, add polynomial/spline columns to preserve."
    )
    if strict:
        raise NotImplementedError(msg)
    warnings.warn(msg, stacklevel=3)


# =============================================================================
# Path-1 one-shot harmonisation
# =============================================================================


@dataclass
class HarmonizedFeatures:
    """Container for ComBat-harmonised connectivity + fit metadata.

    Preserves the shape of the original API so downstream code does not break.
    The ``model`` field now holds a :class:`conninfpy.harmonize.ComBatModel`
    instead of the neuroHarmonize dict.
    """

    conn_harm: np.ndarray
    model: ComBatModel
    feature_count: Tuple[int, int]
    covar_columns: Tuple[str, ...]


def harmonize_fit_transform(
    conn: np.ndarray,
    covars: pd.DataFrame,
    smooth_terms: Iterable[str] = (),
    ref_batch: Optional[str] = None,
    eb: bool = True,
    strict_smooth: bool = False,
) -> HarmonizedFeatures:
    """
    One-shot ComBat: fit on full cohort, return harmonised matrices.

    Parameters
    ----------
    conn : (n_subjects, N, N) Fisher-z connectivity.
    covars : DataFrame with a mandatory ``SITE`` column and optional
        biological covariates to preserve (DX, AGE, SEX, ...).
    smooth_terms : accepted for backward compat; ignored (linear-only ComBat).
        Pass ``strict_smooth=True`` to make non-empty input raise.
    ref_batch : accepted for backward compat; ignored (our ComBat uses
        grand-mean coding, no single reference batch).
    eb : empirical-Bayes shrinkage (default True).
    strict_smooth : raise ``NotImplementedError`` when ``smooth_terms`` is
        non-empty. Default warns.
    """
    _check_smooth(smooth_terms, strict=strict_smooth)
    if ref_batch is not None:
        warnings.warn(
            f"ref_batch={ref_batch!r} ignored — in-package ComBat uses "
            f"grand-mean coding across all sites.",
            stacklevel=2,
        )
    sites, preserve = _split_site_and_preserve(covars)
    flat, iu = flatten_upper(conn)
    n_in = flat.shape[1]

    result = combat_harmonize(
        flat, sites=sites, preserve=preserve, eb=eb, return_diagnostics=False,
    )
    flat_harm = result.Y_adjusted
    if flat_harm.shape[1] != n_in:
        raise RuntimeError(
            f"Feature-count mismatch after ComBat: in={n_in}, out={flat_harm.shape[1]}."
        )
    conn_harm = unflatten_upper(flat_harm, conn.shape[1])

    return HarmonizedFeatures(
        conn_harm=conn_harm,
        model=result.model,
        feature_count=(n_in, flat_harm.shape[1]),
        covar_columns=tuple(covars.columns),
    )


# =============================================================================
# Path-2 fit/apply (for cross-site ML)
# =============================================================================


def harmonize_fit(
    conn_train: np.ndarray,
    covars_train: pd.DataFrame,
    smooth_terms: Iterable[str] = (),
    eb: bool = True,
    strict_smooth: bool = False,
) -> Tuple[np.ndarray, ComBatModel]:
    """Fit ComBat on training-site subjects. Returns ``(conn_harm, model)``."""
    _check_smooth(smooth_terms, strict=strict_smooth)
    sites, preserve = _split_site_and_preserve(covars_train)
    flat, _ = flatten_upper(conn_train)
    model = combat_fit(flat, sites=sites, preserve=preserve, eb=eb)
    flat_harm = combat_apply(model, flat, sites=sites, preserve=preserve)
    conn_harm = unflatten_upper(flat_harm, conn_train.shape[1])
    return conn_harm, model


def harmonize_apply(
    conn_new: np.ndarray,
    covars_new: pd.DataFrame,
    model: ComBatModel,
) -> np.ndarray:
    """Apply a fitted :class:`ComBatModel` to new subjects from known sites."""
    sites, preserve = _split_site_and_preserve(covars_new)
    flat, _ = flatten_upper(conn_new)
    flat_harm = combat_apply(model, flat, sites=sites, preserve=preserve)
    return unflatten_upper(flat_harm, conn_new.shape[1])


# =============================================================================
# Command-line smoke test (run directly on the prepared ABIDE data)
# =============================================================================


def _run_combat_and_save(
    prepared_npz: str,
    output_npz: str,
    smooth_age: bool = False,
) -> None:
    """Harmonise the prepared ABIDE data and write to disk.

    Output file mirrors ``abide_prepared.npz`` plus:
      - ``connectivity_z_harm``: (n_subjects, N, N) ComBat-harmonised features
      - ``combat_smooth_terms``: list of columns that would be GAM smooths
        (in-package ComBat is linear-only, so this is informational)
      - ``combat_feature_count``: (n_in, n_out) — should match
      - ``combat_between_site_var_raw`` / ``_harm``: variance diagnostics
    """
    print(f"Loading {prepared_npz}")
    data = np.load(prepared_npz, allow_pickle=True)

    conn = data["connectivity_z"]
    site = data["site"]
    dx = data["group"]
    age = data["age"]
    sex = data["sex"]
    print(f"  conn: {conn.shape}, n_subjects={conn.shape[0]}")
    print(f"  sites: {len(set(site))} unique")

    covars = build_combat_covars(site=site, dx=dx, age=age, sex=sex)
    print(f"  covars columns: {list(covars.columns)}")

    smooth_terms = ["AGE"] if smooth_age else []
    if smooth_terms:
        print(
            f"  smooth_terms: {smooth_terms}  "
            f"(ignored — in-package ComBat is linear-only)"
        )

    import time
    t0 = time.time()
    result = harmonize_fit_transform(
        conn=conn, covars=covars, smooth_terms=smooth_terms,
    )
    dt = time.time() - t0
    print(f"ComBat fit+transform: {dt:.1f}s")

    conn_harm = result.conn_harm

    # --- Diagnostics: between-site variance reduction --------------------
    iu = np.triu_indices(conn.shape[1], k=1)
    raw_flat = conn[:, iu[0], iu[1]]
    harm_flat = conn_harm[:, iu[0], iu[1]]
    sites_unique = sorted(set(site))
    raw_site_means = np.array([raw_flat[site == s].mean(axis=0) for s in sites_unique])
    harm_site_means = np.array([harm_flat[site == s].mean(axis=0) for s in sites_unique])
    raw_bsv = raw_site_means.var(axis=0).mean()
    harm_bsv = harm_site_means.var(axis=0).mean()
    reduction = (1 - harm_bsv / raw_bsv) * 100
    print(
        f"Between-site variance of per-edge means: "
        f"raw={raw_bsv:.5f}, harm={harm_bsv:.5f} (reduction: {reduction:.1f}%)"
    )

    # --- Save: copy prepared_npz fields + harmonised connectivity --------
    print(f"Saving to {output_npz}")
    out_payload = {k: data[k] for k in data.files}
    out_payload["connectivity_z_harm"] = conn_harm
    out_payload["combat_smooth_terms"] = np.array(smooth_terms, dtype=object)
    out_payload["combat_feature_count"] = np.array(result.feature_count)
    out_payload["combat_between_site_var_raw"] = np.float64(raw_bsv)
    out_payload["combat_between_site_var_harm"] = np.float64(harm_bsv)
    np.savez_compressed(output_npz, **out_payload)
    print(f"  wrote {output_npz}")


if __name__ == "__main__":
    import argparse
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description="Fit ComBat on prepared ABIDE and save harmonised features."
    )
    ap.add_argument("--prepared", default=os.path.join(here, "abide_prepared.npz"))
    ap.add_argument("--output", default=os.path.join(here, "abide_harmonized.npz"))
    ap.add_argument(
        "--smooth-age", action="store_true",
        help="Request an AGE smooth (ignored — in-package ComBat is linear).",
    )
    args = ap.parse_args()
    _run_combat_and_save(args.prepared, args.output, smooth_age=args.smooth_age)
