"""Cross-implementation equivalence against `neuroCombat`.

`neuroCombat` (Fortin et al. 2017; https://github.com/Jfortin1/neuroCombat)
is the canonical Python reference implementation of parametric empirical-
Bayes ComBat. conninfpy reimplements the same algorithm in pure NumPy
(no pandas dependency, native (n, N, N) connectivity convention,
auditable single-file algorithm) — the deliberate design choices are
listed in ``conninfpy/harmonize.py``'s module docstring.

As of conninfpy v2.0 (post `harmonize.py` refactor 2026-05-19),
conninfpy uses the **canonical Fortin 2017 parameterization**:

  - One-hot site dummies (no reference site dropped);
  - ``α`` is the sample-size-weighted grand mean across sites;
  - ``σ²`` uses the biased ``var.pooled`` denominator (divide by ``n``).

Under this parameterization the two implementations agree at:

  - **machine precision** for ``Y_adj``, ``γ̂*``, and ``δ̂*²`` when
    EB is OFF;
  - **EB-convergence tolerance** (~1e-5) for ``Y_adj`` when EB is ON,
    bounded by the per-package iteration stopping criterion;
  - **machine precision** for ``γ̂*`` even with EB ON (the EB
    convergence affects ``δ̂*`` more than ``γ̂*``).

One unit-convention difference: neuroCombat's ``estimates['delta.star']``
is δ² (variance); conninfpy's ``model.delta_star`` is δ (sd, i.e.
sqrt of neuroCombat's). The tests below take ``sqrt`` of neuroCombat's
``delta.star`` before comparing.

Plan / scope / paper-paragraph template: see
``Projects/NetworkStatistics/combat_validation.md`` in the Obsidian
vault.

Tests skip cleanly if neuroCombat is not installed.
"""
from __future__ import annotations

import unittest
import warnings

import numpy as np

try:
    import pandas as pd
    from neuroCombat import neuroCombat
    HAS_NEUROCOMBAT = True
except ImportError:  # pragma: no cover
    HAS_NEUROCOMBAT = False

from conninfpy import combat_harmonize


SKIP_MSG = (
    "neuroCombat or pandas not installed; "
    "pip install neuroCombat to enable cross-implementation tests."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_two_site_data(n_per_site=30, m=12,
                       gamma=(0.5, -0.5), delta=(1.0, 1.5),
                       sigma=0.3, age_effect=False, seed=7):
    """Generate (n, m) two-site dataset with optional age covariate."""
    rng = np.random.RandomState(seed)
    n = 2 * n_per_site
    sites = np.array([0] * n_per_site + [1] * n_per_site)
    gamma_arr = np.array([[gamma[0]] * m, [gamma[1]] * m])
    delta_arr = np.array([[delta[0]] * m, [delta[1]] * m])

    Y = np.zeros((n, m))
    age = rng.uniform(20, 60, n) if age_effect else None
    beta_age = rng.randn(m) * 0.01 if age_effect else None
    for i in range(n):
        signal = (beta_age * age[i]) if age_effect else 0.0
        Y[i] = (signal + gamma_arr[sites[i]]
                + delta_arr[sites[i]] * rng.randn(m) * sigma)
    return Y, sites, age


def _run_neurocombat(Y, sites, age=None, eb=True):
    """Run neuroCombat with conninfpy-style argument shape."""
    covars = {'batch': sites}
    cont_cols = None
    if age is not None:
        covars['age'] = age
        cont_cols = ['age']
    covars_df = pd.DataFrame(covars)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = neuroCombat(
            dat=Y.T, covars=covars_df, batch_col='batch',
            continuous_cols=cont_cols, eb=eb, parametric=True,
        )
    return out


def _harmonize_both(Y, sites, age=None, eb=True):
    """Return (Y_cinf, Y_nc, cinf_model, nc_estimates) for both packages."""
    res = combat_harmonize(Y, sites=sites, preserve=age,
                        eb=eb, return_diagnostics=False)
    nc_out = _run_neurocombat(Y, sites, age=age, eb=eb)
    return res.Y_adjusted, nc_out['data'].T, res.model, nc_out['estimates']


@unittest.skipUnless(HAS_NEUROCOMBAT, SKIP_MSG)
class TestCombatEquivalence(unittest.TestCase):
    """conninfpy ↔ neuroCombat under the canonical Fortin 2017 parameterization."""

    # ------------------------------------------------------------------
    # Test A — Y_adj machine-precision match WITHOUT EB
    # ------------------------------------------------------------------

    def test_y_adj_matches_neurocombat_no_eb(self):
        """Test A: Y_adj agrees to machine precision when EB is OFF."""
        Y, sites, _ = _make_two_site_data(seed=11)
        Y_cinf, Y_nc, _, _ = _harmonize_both(Y, sites, eb=False)
        np.testing.assert_allclose(
            Y_cinf, Y_nc, rtol=1e-12, atol=1e-12,
            err_msg="conninfpy ≠ neuroCombat without EB — algorithm divergence.",
        )

    def test_y_adj_matches_neurocombat_no_eb_with_preserve(self):
        """Test A': Y_adj matches to OLS-solver precision with preserve, EB off.

        Tolerance is loosened from 1e-12 (Test A) to 1e-7 because the two
        packages use different OLS solvers when a preserve block is
        present: conninfpy calls ``np.linalg.lstsq`` (SVD-based
        pseudoinverse); neuroCombat explicitly forms
        ``inv(X'X) @ X' @ y`` per feature. Algebraically identical,
        but rounding paths diverge at ~1e-8 on conditioned designs.
        """
        Y, sites, age = _make_two_site_data(age_effect=True, seed=13)
        Y_cinf, Y_nc, _, _ = _harmonize_both(Y, sites, age=age, eb=False)
        np.testing.assert_allclose(
            Y_cinf, Y_nc, atol=1e-7, rtol=1e-7,
        )

    # ------------------------------------------------------------------
    # Test B — Y_adj match WITH EB (bounded by EB-iteration convergence)
    # ------------------------------------------------------------------

    def test_y_adj_matches_neurocombat_with_eb(self):
        """Test B: Y_adj matches to EB-tolerance with EB on, no preserve.

        EB iteration uses fixed-point with tol=1e-4 on both packages;
        the two convergence trajectories produce slightly different
        terminal values, bounded by that tolerance plus rounding.
        """
        Y, sites, _ = _make_two_site_data(seed=17)
        Y_cinf, Y_nc, _, _ = _harmonize_both(Y, sites, eb=True)
        # EB tolerance is 1e-4 per parameter; propagated through scale to Y_adj
        # gives ~1e-5 max abs diff with this generator
        np.testing.assert_allclose(
            Y_cinf, Y_nc, atol=1e-4, rtol=1e-4,
            err_msg="Y_adj diverges beyond EB convergence tolerance.",
        )

    def test_y_adj_matches_neurocombat_with_eb_and_preserve(self):
        """Test B': Y_adj matches with EB on AND preserve."""
        Y, sites, age = _make_two_site_data(age_effect=True, seed=19)
        Y_cinf, Y_nc, _, _ = _harmonize_both(Y, sites, age=age, eb=True)
        np.testing.assert_allclose(
            Y_cinf, Y_nc, atol=1e-4, rtol=1e-4,
        )

    # ------------------------------------------------------------------
    # Test C — γ̂* matches at machine precision
    # ------------------------------------------------------------------

    def test_gamma_star_matches_neurocombat(self):
        """Test C: γ̂* matches at machine precision with EB on, no preserve.

        γ̂* turns out to be more numerically stable than δ̂* under EB —
        both packages converge it identically.
        """
        Y, sites, _ = _make_two_site_data(seed=23)
        _, _, cinf_model, nc_est = _harmonize_both(Y, sites, eb=True)
        np.testing.assert_allclose(
            cinf_model.gamma_star, nc_est['gamma.star'],
            atol=1e-10, rtol=1e-10,
            err_msg="γ̂* differs from neuroCombat — EB iteration mismatch.",
        )

    # ------------------------------------------------------------------
    # Test D — δ̂*² matches at machine precision (unit-aligned)
    # ------------------------------------------------------------------

    def test_delta_star_squared_matches_neurocombat(self):
        """Test D: δ̂*² matches at machine precision (unit-aligned).

        neuroCombat stores ``estimates['delta.star']`` as δ² (variance);
        conninfpy stores ``model.delta_star`` as δ (sd, i.e. sqrt of nc's).
        After unit alignment: machine precision.
        """
        Y, sites, _ = _make_two_site_data(seed=29)
        _, _, cinf_model, nc_est = _harmonize_both(Y, sites, eb=False)
        # nc stores δ²; cinf stores δ. Square cinf's; compare to nc's directly.
        cinf_delta_sq = cinf_model.delta_star ** 2
        np.testing.assert_allclose(
            cinf_delta_sq, nc_est['delta.star'],
            atol=1e-12, rtol=1e-12,
            err_msg=("δ̂*² differs from neuroCombat — variance estimator "
                    "mismatch. Verify the var.pooled denominator and EB "
                    "iteration direction."),
        )

    # ------------------------------------------------------------------
    # Test E — Variance equalization (smoke test, was Test A in v1)
    # ------------------------------------------------------------------

    def test_variance_equalization_matches_neurocombat(self):
        """Test E (smoke test): both packages reduce between-site variance."""
        Y, sites, _ = _make_two_site_data(seed=31)
        Y_cinf, Y_nc, _, _ = _harmonize_both(Y, sites, eb=True)

        means_before = np.array([Y[sites == s].mean(axis=0)
                                for s in np.unique(sites)])
        var_before = means_before.var(axis=0, ddof=0).mean()
        means_after_c = np.array([Y_cinf[sites == s].mean(axis=0)
                                for s in np.unique(sites)])
        means_after_n = np.array([Y_nc[sites == s].mean(axis=0)
                                for s in np.unique(sites)])

        ratio_c = means_after_c.var(axis=0, ddof=0).mean() / var_before
        ratio_n = means_after_n.var(axis=0, ddof=0).mean() / var_before
        self.assertLess(ratio_c, 0.05)
        self.assertLess(ratio_n, 0.05)
        # Tightened from 0.02 to 0.001 — they're now numerically identical
        self.assertAlmostEqual(ratio_c, ratio_n, delta=1e-3)

    # ------------------------------------------------------------------
    # Test F — model.beta is exactly the OLS β (preserve invariant)
    # ------------------------------------------------------------------

    def test_preserve_coefficient_matches_neurocombat(self):
        """Test F: model.beta agrees with neuroCombat's preserve coefficient.

        The OLS β for the preserve column is identifiable across both
        parameterizations (different intercept conventions span the same
        column space). With the canonical refactor, conninfpy's
        ``model.beta`` equals neuroCombat's ``B_hat[n_batch:n_batch+k]``
        slot.
        """
        Y, sites, age = _make_two_site_data(age_effect=True, seed=37)
        _, _, cinf_model, _ = _harmonize_both(Y, sites, age=age, eb=False)
        # neuroCombat does not expose B_hat directly in 'estimates' but
        # the OLS β is recoverable from the same design. Reconstruct it.
        n = Y.shape[0]
        n_sites = 2
        D = np.zeros((n, n_sites))
        D[np.arange(n), sites] = 1.0
        X_full = np.column_stack([D, age.reshape(-1, 1)])
        beta_full, *_ = np.linalg.lstsq(X_full, Y, rcond=None)
        beta_ols_age = beta_full[n_sites:n_sites + 1]  # (1, m)

        np.testing.assert_allclose(
            cinf_model.beta, beta_ols_age,
            rtol=1e-12, atol=1e-12,
            err_msg=("model.beta differs from canonical-design OLS β — "
                    "combat_fit is silently modifying preserve coefficient."),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
