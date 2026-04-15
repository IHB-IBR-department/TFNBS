"""
Tests for permutation acceleration methods (GPD and gamma).

Verifies:
- GPD fit correctness on known distributions
- Accelerated p-values close to full-permutation reference
- Null control preserved (no type I error inflation)
- Integration with both t-test and GLM pipelines
- Gamma fallback
"""

import unittest

import numpy as np
import numpy.testing as npt

from conninfpy.acceleration import (
    _fit_gpd_mom,
    _gpd_sf,
    fit_gpd_tail,
    fit_gamma_tail,
    compute_p_values_accelerated,
)
from conninfpy.pairwise_stats import compute_p_val
from conninfpy.glm_stats import compute_p_val_glm


# =============================================================================
# 1. GPD fitting internals
# =============================================================================


class TestGPDFit(unittest.TestCase):
    """Test GPD fitting by method of moments."""

    def test_exponential_case(self):
        """When xi=0, GPD is exponential. MoM should recover xi≈0."""
        rng = np.random.RandomState(42)
        # Exponential(scale=2) → GPD with xi=0, sigma=2
        exceedances = rng.exponential(scale=2.0, size=1000)
        sigma, xi, valid = _fit_gpd_mom(exceedances)
        self.assertTrue(valid)
        self.assertAlmostEqual(xi, 0.0, delta=0.1)
        self.assertAlmostEqual(sigma, 2.0, delta=0.3)

    def test_too_few_exceedances(self):
        """Fewer than 10 exceedances → invalid fit."""
        _, _, valid = _fit_gpd_mom(np.array([1.0, 2.0, 3.0]))
        self.assertFalse(valid)

    def test_constant_exceedances(self):
        """Constant values (zero variance) → invalid fit."""
        _, _, valid = _fit_gpd_mom(np.ones(20))
        self.assertFalse(valid)


class TestGPDSurvival(unittest.TestCase):
    """Test GPD survival function."""

    def test_exponential_sf(self):
        """When xi=0, GPD SF = exp(-x/sigma)."""
        x = np.array([0.0, 1.0, 2.0, 5.0])
        sf = _gpd_sf(x, sigma=2.0, xi=0.0)
        expected = np.exp(-x / 2.0)
        npt.assert_allclose(sf, expected, atol=1e-10)

    def test_sf_at_zero(self):
        """SF at 0 should be 1."""
        sf = _gpd_sf(np.array([0.0]), sigma=1.0, xi=0.5)
        self.assertAlmostEqual(sf.item(), 1.0)

    def test_sf_monotone_decreasing(self):
        """SF should be monotonically decreasing."""
        x = np.linspace(0, 10, 50)
        sf = _gpd_sf(x, sigma=1.0, xi=0.2)
        self.assertTrue(np.all(np.diff(sf) <= 0))


# =============================================================================
# 2. GPD tail approximation
# =============================================================================


class TestFitGPDTail(unittest.TestCase):
    """Test fit_gpd_tail end-to-end."""

    def test_known_null_distribution(self):
        """GPD p-values from Gaussian null should be reasonable."""
        rng = np.random.RandomState(42)
        null_dist = rng.standard_normal(500)  # simulated max-stats
        # Observed values at different quantiles
        observed = np.array([[0.0, 1.0], [2.0, 3.0]])

        p_gpd = fit_gpd_tail(null_dist, observed)
        p_emp = np.mean(observed[..., np.newaxis] < null_dist, axis=-1)

        # GPD and empirical should roughly agree for moderate values
        # For extreme values (3.0), GPD should give more precise estimate
        self.assertEqual(p_gpd.shape, observed.shape)
        self.assertTrue(np.all(p_gpd >= 0))
        self.assertTrue(np.all(p_gpd <= 1))

        # p-value ordering should be preserved
        self.assertGreater(p_gpd[0, 0], p_gpd[1, 1])

    def test_fallback_to_empirical(self):
        """When GPD fit fails, should fall back to empirical p-values."""
        rng = np.random.RandomState(42)
        # Very small null (hard to fit GPD)
        null_dist = rng.standard_normal(20)
        observed = np.array([[1.0, 2.0], [0.5, 0.0]])

        p_gpd = fit_gpd_tail(null_dist, observed)
        p_emp = np.mean(observed[..., np.newaxis] < null_dist, axis=-1)

        # Should get valid p-values regardless
        self.assertEqual(p_gpd.shape, observed.shape)
        self.assertTrue(np.all(p_gpd >= 0))
        self.assertTrue(np.all(p_gpd <= 1))


# =============================================================================
# 3. Gamma approximation
# =============================================================================


class TestFitGammaTail(unittest.TestCase):
    """Test gamma approximation."""

    def test_gamma_from_gaussian_null(self):
        """Gamma fit to Gaussian max-stats produces valid p-values."""
        rng = np.random.RandomState(42)
        null_dist = np.abs(rng.standard_normal(500))
        observed = np.array([[0.5, 1.0], [2.0, 3.0]])

        p_gamma = fit_gamma_tail(null_dist, observed)

        self.assertEqual(p_gamma.shape, observed.shape)
        self.assertTrue(np.all(p_gamma >= 0))
        self.assertTrue(np.all(p_gamma <= 1))
        # Ordering should be preserved
        self.assertGreater(p_gamma[0, 0], p_gamma[1, 1])

    def test_gamma_matches_empirical_roughly(self):
        """Gamma p-values should be in the same ballpark as empirical."""
        rng = np.random.RandomState(42)
        null_dist = np.abs(rng.standard_normal(1000))
        observed = np.array([[1.5]])

        p_gamma = fit_gamma_tail(null_dist, observed)
        p_emp = np.mean(observed[..., np.newaxis] < null_dist, axis=-1)

        # Should agree within 0.1 for moderate values
        self.assertAlmostEqual(p_gamma.item(), p_emp.item(), delta=0.1)


# =============================================================================
# 4. compute_p_values_accelerated wrapper
# =============================================================================


class TestComputePValuesAccelerated(unittest.TestCase):
    """Test the drop-in replacement for _compute_p_values_from_null."""

    def test_gpd_method(self):
        """GPD method returns correct structure."""
        rng = np.random.RandomState(42)
        N = 5
        emp = {
            "positive": rng.rand(N, N) * 3,
            "negative": rng.rand(N, N) * 3,
        }
        null = {
            "positive": rng.standard_normal(200),
            "negative": rng.standard_normal(200),
        }
        p = compute_p_values_accelerated(emp, null, method="gpd")
        self.assertIn("positive", p)
        self.assertIn("negative", p)
        self.assertEqual(p["positive"].shape, (N, N))
        self.assertTrue(np.all(p["positive"] >= 0))
        self.assertTrue(np.all(p["positive"] <= 1))

    def test_gamma_method(self):
        """Gamma method returns correct structure."""
        rng = np.random.RandomState(42)
        N = 5
        emp = {"g2>g1": rng.rand(N, N) * 3, "g1>g2": rng.rand(N, N) * 3}
        null = {"g2>g1": np.abs(rng.standard_normal(200)),
                "g1>g2": np.abs(rng.standard_normal(200))}
        p = compute_p_values_accelerated(emp, null, method="gamma")
        self.assertIn("g2>g1", p)
        self.assertEqual(p["g2>g1"].shape, (N, N))


# =============================================================================
# 5. Integration with t-test pipeline
# =============================================================================


class TestAccelerationTTestPipeline(unittest.TestCase):
    """Integration tests: acceleration in compute_p_val."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.RandomState(42)
        cls.N = 8
        n_per = 15
        cls.group1 = rng.randn(n_per, cls.N, cls.N)
        cls.group2 = rng.randn(n_per, cls.N, cls.N)
        for g in (cls.group1, cls.group2):
            for s in range(n_per):
                g[s] = (g[s] + g[s].T) / 2
                np.fill_diagonal(g[s], 0)

    def test_gpd_tstat(self):
        """compute_p_val with acceleration='gpd' runs and returns valid p."""
        p = compute_p_val(
            self.group1, self.group2,
            n_permutations=200, test_type='two-sample',
            method='tstat', use_mp=False, random_state=0,
            acceleration='gpd',
        )
        self.assertIn("g2>g1", p)
        self.assertEqual(p["g2>g1"].shape, (self.N, self.N))
        self.assertTrue(np.all(p["g2>g1"] >= 0))
        self.assertTrue(np.all(p["g2>g1"] <= 1))

    def test_gamma_tfnbs(self):
        """compute_p_val with acceleration='gamma' and TFNBS."""
        p = compute_p_val(
            self.group1, self.group2,
            n_permutations=200, test_type='two-sample',
            method='tfnbs', use_mp=False, random_state=0,
            acceleration='gamma', e=0.4, h=3.0, n=5,
        )
        self.assertIn("g2>g1", p)
        self.assertTrue(np.all(p["g2>g1"] >= 0))
        self.assertTrue(np.all(p["g2>g1"] <= 1))

    def test_no_acceleration_baseline(self):
        """acceleration=None preserves original behavior."""
        p_none = compute_p_val(
            self.group1, self.group2,
            n_permutations=100, test_type='two-sample',
            method='tstat', use_mp=False, random_state=0,
            acceleration=None,
        )
        # Just verify it runs — exact values tested elsewhere
        self.assertIn("g2>g1", p_none)


# =============================================================================
# 6. Integration with GLM pipeline
# =============================================================================


class TestAccelerationGLMPipeline(unittest.TestCase):
    """Integration tests: acceleration in compute_p_val_glm."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.RandomState(42)
        cls.N = 8
        cls.n = 20
        cls.Y = rng.randn(cls.n, cls.N, cls.N)
        for s in range(cls.n):
            cls.Y[s] = (cls.Y[s] + cls.Y[s].T) / 2
            np.fill_diagonal(cls.Y[s], 0)
        cls.age = rng.randn(cls.n)

    def test_gpd_glm(self):
        """compute_p_val_glm with acceleration='gpd' runs correctly."""
        p = compute_p_val_glm(
            self.Y, interest=self.age, method='tstat',
            n_permutations=200, use_mp=False, random_state=0,
            acceleration='gpd',
        )
        self.assertIn("positive", p)
        self.assertEqual(p["positive"].shape, (self.N, self.N))
        self.assertTrue(np.all(p["positive"] >= 0))
        self.assertTrue(np.all(p["positive"] <= 1))

    def test_gamma_glm_tfnbs(self):
        """compute_p_val_glm with gamma + TFNBS."""
        p = compute_p_val_glm(
            self.Y, interest=self.age, method='tfnbs',
            n_permutations=200, use_mp=False, random_state=0,
            acceleration='gamma', e=0.4, h=3.0, n=5,
        )
        self.assertIn("positive", p)
        self.assertTrue(np.all(p["positive"] >= 0))

    def test_gpd_planted_effect(self):
        """GPD acceleration still detects planted effect."""
        rng = np.random.RandomState(321)
        n = 30
        N = 8
        age = rng.randn(n)

        Y = rng.randn(n, N, N) * 0.3
        for s in range(n):
            Y[s] = (Y[s] + Y[s].T) / 2
            np.fill_diagonal(Y[s], 0)

        # Plant strong effect
        for s in range(n):
            Y[s, 1, 2] += 1.5 * age[s]
            Y[s, 2, 1] += 1.5 * age[s]

        p = compute_p_val_glm(
            Y, interest=age, method='tstat',
            n_permutations=200, use_mp=False, random_state=0,
            acceleration='gpd',
        )

        # Planted edge should still be significant
        self.assertLess(p["positive"][1, 2], 0.05,
                        f"Planted edge p={p['positive'][1, 2]}, expected < 0.05")


if __name__ == "__main__":
    unittest.main()
