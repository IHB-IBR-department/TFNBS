"""
Tests for conninfpy.harmonize — parametric empirical-Bayes ComBat and
design-matrix diagnostics (VIF, condition number).
"""

import unittest

import numpy as np
import numpy.testing as npt

from conninfpy.harmonize import (
    ComBatModel,
    CombatResult,
    combat_apply,
    combat_fit,
    combat_harmonize,
    compute_vif,
    design_diagnostics,
    flatten_upper,
    unflatten_upper,
)


def _make_site_planted_data(
    n_per_site: int = 20,
    n_sites: int = 3,
    N: int = 6,
    site_scale: float = 0.8,
    noise_scale: float = 0.3,
    preserve_beta: float = 0.0,
    seed: int = 0,
):
    """
    Build a multi-site connectivity dataset where each site contributes a
    symmetric edge-level offset. Optionally plant a preserved-covariate
    effect proportional to ``preserve_beta`` at edge (0, 1).
    """
    rng = np.random.RandomState(seed)
    n = n_per_site * n_sites
    sites = np.repeat(np.arange(n_sites), n_per_site)

    # Site-level offsets (symmetric, zero-diagonal)
    site_offsets = rng.randn(n_sites, N, N)
    for s in range(n_sites):
        site_offsets[s] = (site_offsets[s] + site_offsets[s].T) / 2
        np.fill_diagonal(site_offsets[s], 0.0)

    covariate = rng.randn(n)  # e.g. age

    Y = rng.randn(n, N, N) * noise_scale
    for i in range(n):
        Y[i] = (Y[i] + Y[i].T) / 2
        np.fill_diagonal(Y[i], 0.0)
        Y[i] += site_scale * site_offsets[sites[i]]
        if preserve_beta != 0.0:
            Y[i, 0, 1] += preserve_beta * covariate[i]
            Y[i, 1, 0] += preserve_beta * covariate[i]
    return Y, sites, covariate, site_offsets


class TestFlattenUpper(unittest.TestCase):
    """Flatten/unflatten round-trip."""

    def test_round_trip(self):
        rng = np.random.RandomState(7)
        N, n = 5, 4
        Y = rng.randn(n, N, N)
        for i in range(n):
            Y[i] = (Y[i] + Y[i].T) / 2
            np.fill_diagonal(Y[i], 0.0)
        features, triu_idx, N_out = flatten_upper(Y)
        self.assertEqual(N_out, N)
        self.assertEqual(features.shape, (n, N * (N - 1) // 2))
        Y_back = unflatten_upper(features, triu_idx, N)
        npt.assert_allclose(Y_back, Y, atol=1e-12)

    def test_bad_shape_raises(self):
        with self.assertRaises(ValueError):
            flatten_upper(np.random.randn(3, 4, 5))  # not square


class TestComBat(unittest.TestCase):
    """Parametric ComBat correctness."""

    def test_reduces_between_site_variance_ratio(self):
        """
        After ComBat the between-site variance ratio should drop by >90%
        on a clean synthetic site-planted dataset.
        """
        Y, sites, _, _ = _make_site_planted_data(seed=1)
        res = combat_harmonize(Y, sites, return_diagnostics=True)
        ratio_before = res.diagnostics["between_site_variance_ratio_before"]
        ratio_after = res.diagnostics["between_site_variance_ratio_after"]
        self.assertGreater(ratio_before, 0.2,
                           f"setup bug: planted site effect too weak, "
                           f"ratio_before={ratio_before:.4f}")
        self.assertLess(ratio_after / ratio_before, 0.1,
                        f"ratio reduction < 90%: before={ratio_before:.4f}, "
                        f"after={ratio_after:.4f}")

    def test_output_shape_matches_input_3d(self):
        """(n, N, N) in → (n, N, N) out, symmetric, zero diagonal preserved."""
        Y, sites, _, _ = _make_site_planted_data(seed=2)
        res = combat_harmonize(Y, sites)
        self.assertEqual(res.Y_adjusted.shape, Y.shape)
        # Symmetric
        npt.assert_allclose(res.Y_adjusted,
                            res.Y_adjusted.transpose(0, 2, 1),
                            atol=1e-10)
        # Zero diagonal
        diag_vals = np.diagonal(res.Y_adjusted, axis1=1, axis2=2)
        npt.assert_allclose(diag_vals, 0.0, atol=1e-10)

    def test_output_shape_matches_input_2d(self):
        """(n, p) in → (n, p) out."""
        rng = np.random.RandomState(3)
        n, p = 40, 25
        sites = rng.randint(0, 3, size=n)
        X = rng.randn(n, p) + sites[:, np.newaxis] * 0.5
        res = combat_harmonize(X, sites)
        self.assertEqual(res.Y_adjusted.shape, X.shape)

    def test_preserve_covariate_effect_is_preserved(self):
        """
        A covariate-linear signal at edge (0, 1) survives ComBat with
        ``preserve``: the post-ComBat β (Y regressed on covariate) should
        be close to the pre-ComBat β at that edge.
        """
        Y, sites, cov, _ = _make_site_planted_data(
            preserve_beta=1.0, seed=4,
        )
        res = combat_harmonize(Y, sites, preserve=cov, return_diagnostics=False)

        # Beta at edge (0, 1) before vs after — simple OLS on the single edge
        def edge_beta(Yarr):
            y = Yarr[:, 0, 1]
            X = np.column_stack([np.ones(len(y)), cov])
            b, *_ = np.linalg.lstsq(X, y, rcond=None)
            return b[1]

        b_before = edge_beta(Y)
        b_after = edge_beta(res.Y_adjusted)
        self.assertGreater(b_before, 0.3, "preserved effect should be ≈ 1 before")
        self.assertAlmostEqual(
            b_after, b_before, delta=0.3,
            msg=f"preserved β drifted: before={b_before:.3f}, after={b_after:.3f}",
        )

    def test_fit_apply_equals_fit_transform(self):
        """combat_fit + combat_apply on same data = combat_harmonize."""
        Y, sites, _, _ = _make_site_planted_data(seed=5)
        res_all = combat_harmonize(Y, sites, return_diagnostics=False)
        model = combat_fit(Y, sites)
        Y_adj = combat_apply(model, Y, sites)
        npt.assert_allclose(res_all.Y_adjusted, Y_adj, atol=1e-12)

    def test_apply_to_new_subjects_from_known_sites(self):
        """
        Fit on one half, apply to the other half — shapes and site-variance
        reduction should still hold.
        """
        Y, sites, _, _ = _make_site_planted_data(n_per_site=30, seed=6)
        n = Y.shape[0]
        rng = np.random.RandomState(0)
        idx = rng.permutation(n)
        train, test = idx[: n // 2], idx[n // 2:]
        model = combat_fit(Y[train], sites[train])
        Y_test_adj = combat_apply(model, Y[test], sites[test])
        self.assertEqual(Y_test_adj.shape, Y[test].shape)

        # Between-site variance should drop on the held-out subjects too
        def ratio(arr, s):
            flat = arr.reshape(arr.shape[0], -1)
            means = np.stack([flat[s == u].mean(axis=0) for u in np.unique(s)])
            return float(np.mean(np.var(means, axis=0, ddof=0)) /
                         max(float(np.mean(np.var(flat, axis=0, ddof=0))), 1e-12))

        r_before = ratio(Y[test], sites[test])
        r_after = ratio(Y_test_adj, sites[test])
        self.assertLess(r_after, 0.5 * r_before,
                        f"fit-on-train / apply-on-test: ratio before={r_before:.3f}, "
                        f"after={r_after:.3f}")

    def test_unknown_site_on_apply_raises(self):
        """Applying to a site not seen during fit raises ValueError."""
        Y, sites, _, _ = _make_site_planted_data(seed=7)
        # Fit on sites 0, 1 only
        mask = sites != 2
        model = combat_fit(Y[mask], sites[mask])
        with self.assertRaises(ValueError):
            combat_apply(model, Y, sites)

    def test_single_site_raises(self):
        """combat_fit with only one unique site raises ValueError."""
        Y, _, _, _ = _make_site_planted_data(seed=8)
        with self.assertRaises(ValueError):
            combat_fit(Y, np.zeros(Y.shape[0]))

    def test_no_eb_still_runs(self):
        """eb=False path runs and reduces site variance (less strongly)."""
        Y, sites, _, _ = _make_site_planted_data(seed=9)
        res = combat_harmonize(Y, sites, eb=False, return_diagnostics=True)
        self.assertLess(
            res.diagnostics["between_site_variance_ratio_after"],
            res.diagnostics["between_site_variance_ratio_before"],
        )


class TestVarianceFloorWarning(unittest.TestCase):
    """Variance-floor RuntimeWarnings on catastrophic-cancellation cases.

    Build a multi-site dataset where one edge is held constant within one
    site (within-site variance is exactly zero) so the per-site delta floor
    must fire. The pooled-sigma path is exercised by holding the same edge
    constant across the *whole* dataset.
    """

    def test_per_site_delta_floor_warns(self):
        Y, sites, _, _ = _make_site_planted_data(
            n_per_site=10, n_sites=3, N=5, seed=11
        )
        # Force one edge to be a within-site constant on site 0 → δ²₀ = 0.
        mask = sites == 0
        Y[mask, 0, 1] = 0.42
        Y[mask, 1, 0] = 0.42
        with self.assertWarns(RuntimeWarning) as ctx:
            combat_harmonize(Y, sites)
        msgs = [str(w.message) for w in ctx.warnings]
        self.assertTrue(
            any("per-site delta" in m for m in msgs),
            f"expected a per-site delta warning, got: {msgs}",
        )

    def test_pooled_sigma_floor_warns(self):
        Y, sites, _, _ = _make_site_planted_data(
            n_per_site=10, n_sites=3, N=5, seed=12
        )
        # Force one edge to a global constant → residuals = 0 → σ² = 0.
        Y[:, 0, 1] = 0.13
        Y[:, 1, 0] = 0.13
        with self.assertWarns(RuntimeWarning) as ctx:
            combat_harmonize(Y, sites)
        msgs = [str(w.message) for w in ctx.warnings]
        self.assertTrue(
            any("pooled sigma" in m for m in msgs),
            f"expected a pooled-sigma warning, got: {msgs}",
        )

    def test_no_warning_on_clean_data(self):
        import warnings as _w

        Y, sites, _, _ = _make_site_planted_data(seed=13)
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            combat_harmonize(Y, sites)
        floor_warnings = [
            w for w in caught
            if issubclass(w.category, RuntimeWarning)
            and "variance" in str(w.message).lower()
        ]
        self.assertEqual(floor_warnings, [])


class TestVIF(unittest.TestCase):
    """Variance inflation factor correctness."""

    def test_independent_columns_vif_near_one(self):
        rng = np.random.RandomState(42)
        X = np.column_stack([np.ones(80), rng.randn(80), rng.randn(80), rng.randn(80)])
        vif = compute_vif(X)
        self.assertEqual(vif.shape, (4,))
        npt.assert_allclose(vif[0], 1.0)
        # Independent columns: VIF should be close to 1
        self.assertTrue(np.all(vif[1:] < 1.3),
                        f"independent columns should have VIF≈1, got {vif}")

    def test_near_collinear_columns_vif_large(self):
        rng = np.random.RandomState(0)
        x = rng.randn(50)
        x_twin = x + 1e-3 * rng.randn(50)
        X = np.column_stack([np.ones(50), x, x_twin])
        vif = compute_vif(X)
        self.assertGreater(vif[1], 100.0)
        self.assertGreater(vif[2], 100.0)

    def test_vif_matches_r_squared_formula(self):
        """VIF_j = 1/(1-R²_j) where R²_j is from regressing col j on others."""
        rng = np.random.RandomState(1)
        n = 100
        a = rng.randn(n)
        b = rng.randn(n)
        c = 0.6 * a + 0.3 * b + 0.5 * rng.randn(n)
        X = np.column_stack([np.ones(n), a, b, c])
        vif = compute_vif(X)

        # Manually: regress c on [intercept, a, b]
        X_others = np.column_stack([np.ones(n), a, b])
        beta, *_ = np.linalg.lstsq(X_others, c, rcond=None)
        pred = X_others @ beta
        ss_res = float(np.sum((c - pred) ** 2))
        ss_tot = float(np.sum((c - c.mean()) ** 2))
        r_sq = 1.0 - ss_res / ss_tot
        vif_manual = 1.0 / (1.0 - r_sq)
        npt.assert_allclose(vif[3], vif_manual, atol=1e-10)


class TestDesignDiagnostics(unittest.TestCase):
    """Integrated design-matrix diagnostic report."""

    def test_clean_design_no_flags(self):
        rng = np.random.RandomState(0)
        X = np.column_stack([np.ones(80), rng.randn(80), rng.randn(80)])
        d = design_diagnostics(X, names=["intercept", "age", "fd"])
        self.assertEqual(d["flags"], [])
        self.assertLess(d["condition_number"], 30)
        self.assertLess(d["vif_max"], 5)

    def test_collinear_design_flags(self):
        rng = np.random.RandomState(0)
        x = rng.randn(50)
        X = np.column_stack([np.ones(50), x, x + 1e-3 * rng.randn(50)])
        d = design_diagnostics(X, names=["intercept", "a", "a_twin"])
        self.assertGreater(len(d["flags"]), 0)
        self.assertEqual(d["vif_max_col"], "a")

    def test_name_length_validated(self):
        with self.assertRaises(ValueError):
            design_diagnostics(np.random.randn(10, 3), names=["a", "b"])


if __name__ == "__main__":
    unittest.main()
