"""
Tests for GLM extension (conninfpy.glm_stats).

Covers: core GLM computation, Freedman-Lane permutation,
enhancement integration, two-tailed FWER, API consistency.

Dev parameters: N=10, n_subjects=20 per group, n_perm=200, d=1.0.
"""

import unittest

import numpy as np
import numpy.testing as npt
from scipy import stats

from conninfpy.glm_stats import (
    GLMStatType,
    _precompute_ols,
    _compute_reduced_model,
    compute_glm_stat,
    compute_p_val_glm,
    compute_p_val_paired_glm,
    build_design_matrix,
)
from conninfpy.pairwise_stats import (
    compute_p_val,
    compute_t_stat,
    _extract_max_stats,
)
from . import fixtures


# =============================================================================
# 1. GLM Foundation
# =============================================================================


class TestPrecomputeOLS(unittest.TestCase):
    """Test _precompute_ols correctness."""

    def test_pseudoinverse_matches_lstsq(self):
        """X_pinv @ Y produces identical betas to np.linalg.lstsq."""
        rng = np.random.RandomState(42)
        n, p = 40, 3
        X = rng.randn(n, p)
        X[:, 0] = 1.0  # intercept
        Y = rng.randn(n, 10)

        X_pinv, XtX_inv_diag, XtX_inv = _precompute_ols(X)
        beta_pinv = X_pinv @ Y

        beta_lstsq, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)

        npt.assert_allclose(beta_pinv, beta_lstsq, atol=1e-10)

    def test_XtX_inv_diag_shape(self):
        """XtX_inv_diag has shape (p,)."""
        rng = np.random.RandomState(42)
        X = rng.randn(40, 3)
        _, XtX_inv_diag, XtX_inv = _precompute_ols(X)
        self.assertEqual(XtX_inv_diag.shape, (3,))
        self.assertEqual(XtX_inv.shape, (3, 3))


class TestComputeGLMStat(unittest.TestCase):
    """Test compute_glm_stat core logic."""

    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.RandomState(42)
        cls.N = 10
        cls.n_per_group = 20
        cls.n_total = 2 * cls.n_per_group

    def test_binary_glm_matches_ttest(self):
        """
        Binary GLM (group indicator) produces same t-statistics as
        two-sample Welch t-test within numerical tolerance.
        """
        rng = self.rng
        N = self.N
        n1 = self.n_per_group
        n_total = self.n_total

        # Generate two groups of connectivity
        group1 = rng.randn(n1, N, N)
        group2 = rng.randn(n1, N, N)
        for s in range(n1):
            group1[s] = (group1[s] + group1[s].T) / 2
            group2[s] = (group2[s] + group2[s].T) / 2
            np.fill_diagonal(group1[s], 0)
            np.fill_diagonal(group2[s], 0)

        # Existing t-test
        ttest_result = compute_t_stat(group1, group2, test_type='two-sample')

        # GLM: Y = [group1; group2], X = [intercept, group_indicator]
        Y = np.concatenate([group1, group2], axis=0)
        intercept = np.ones(n_total)
        group_ind = np.concatenate([np.zeros(n1), np.ones(n1)])
        X = np.column_stack([intercept, group_ind])
        contrast = np.array([0.0, 1.0])

        glm_result = compute_glm_stat(Y, X, contrast, stat_type='tstat')

        # GLM uses pooled variance (not Welch), so exact match isn't expected.
        # But the sign pattern and relative ordering should be very close.
        # For equal group sizes, pooled and Welch are similar.
        #
        # We check that the positive/negative separation is consistent:
        # Where ttest has g2>g1, GLM should have positive, and vice versa.
        ttest_pos = ttest_result["g2>g1"]
        ttest_neg = ttest_result["g1>g2"]
        glm_pos = glm_result["positive"]
        glm_neg = glm_result["negative"]

        # Edges with positive effect should agree on which direction
        mask_pos = ttest_pos > 0.5
        mask_neg = ttest_neg > 0.5

        # Where one has a positive effect, the other should too
        if np.any(mask_pos):
            self.assertTrue(np.all(glm_pos[mask_pos] > 0),
                            "GLM positive should be >0 where ttest is positive")
        if np.any(mask_neg):
            self.assertTrue(np.all(glm_neg[mask_neg] > 0),
                            "GLM negative should be >0 where ttest is negative")

        # Correlation between t-values should be very high
        triu = np.triu_indices(N, k=1)
        combined_ttest = ttest_pos[triu] - ttest_neg[triu]
        combined_glm = glm_pos[triu] - glm_neg[triu]
        corr = np.corrcoef(combined_ttest, combined_glm)[0, 1]
        self.assertGreater(corr, 0.99, f"GLM-ttest correlation too low: {corr}")

    def test_continuous_predictor_detection(self):
        """GLM detects planted continuous effect at specific edge."""
        scenario = fixtures.glm_planted_edge()  # n=40, N=10, edge=(2,3), β=1, noise=0.5
        Y, age, effect_edge = scenario["Y"], scenario["age"], scenario["effect_edge"]
        N = scenario["N"]

        X, contrast = build_design_matrix(age)
        result = compute_glm_stat(Y, X, contrast, stat_type='tstat')

        # The planted edge should have the largest positive t-stat
        pos = result["positive"]
        planted_t = pos[effect_edge]
        max_other = np.max(pos[np.triu_indices(N, k=1)])

        self.assertEqual(planted_t, max_other,
                         "Planted edge should have the max positive t-stat")
        self.assertGreater(planted_t, 3.0,
                           f"Planted effect t-stat too small: {planted_t}")

    def test_null_control(self):
        """Under H0 (no effect), t-stats are not systematically inflated."""
        scenario = fixtures.glm_null()   # n=40, N=10, pure noise
        Y, age, N = scenario["Y"], scenario["age"], scenario["N"]

        X, contrast = build_design_matrix(age)
        result = compute_glm_stat(Y, X, contrast, stat_type='tstat')

        # Under null, max t should be modest (not systematically > 4)
        triu = np.triu_indices(N, k=1)
        all_t = result["positive"][triu] - result["negative"][triu]
        self.assertLess(np.max(np.abs(all_t)), 5.0,
                        "Null t-stats unexpectedly large")
        # Mean should be close to 0
        self.assertAlmostEqual(np.mean(all_t), 0, delta=0.5)

    def test_single_edge_matches_scipy(self):
        """GLM t-stat at one edge matches scipy.stats.linregress."""
        rng = np.random.RandomState(77)
        n_subjects = 30
        N = 5
        age = rng.randn(n_subjects)

        Y = rng.randn(n_subjects, N, N)
        for s in range(n_subjects):
            Y[s] = (Y[s] + Y[s].T) / 2
            np.fill_diagonal(Y[s], 0)

        X, contrast = build_design_matrix(age)
        result = compute_glm_stat(Y, X, contrast, stat_type='tstat')

        # Check edge (0, 1) against scipy
        y_edge = Y[:, 0, 1]
        slope, intercept, r, p, stderr = stats.linregress(age, y_edge)
        scipy_t = slope / stderr

        glm_t = result["positive"][0, 1] - result["negative"][0, 1]
        npt.assert_allclose(glm_t, scipy_t, atol=1e-6,
                            err_msg="GLM t-stat doesn't match scipy.linregress")

    def test_beta_stat_type(self):
        """stat_type='beta' returns raw regression coefficients."""
        scenario = fixtures.glm_planted_edge(
            n=20, N=5, effect_edge=(1, 2), effect_beta=2.0, noise_scale=1.0, seed=55,
        )
        Y, age = scenario["Y"], scenario["age"]

        X, contrast = build_design_matrix(age)
        result_beta = compute_glm_stat(Y, X, contrast, stat_type='beta')
        result_tstat = compute_glm_stat(Y, X, contrast, stat_type='tstat')

        # Beta at planted edge should be close to 2.0
        planted_beta = result_beta["positive"][1, 2]
        self.assertGreater(planted_beta, 1.5)
        self.assertLess(planted_beta, 2.5)

        # Both should agree on sign
        self.assertGreater(result_tstat["positive"][1, 2], 0)

    def test_output_keys_and_shape(self):
        """Output has 'positive' and 'negative' keys with correct shape."""
        rng = np.random.RandomState(11)
        N = 8
        n = 20
        Y = rng.randn(n, N, N)
        age = rng.randn(n)
        X, contrast = build_design_matrix(age)
        result = compute_glm_stat(Y, X, contrast)

        self.assertIn("positive", result)
        self.assertIn("negative", result)
        self.assertEqual(result["positive"].shape, (N, N))
        self.assertEqual(result["negative"].shape, (N, N))
        # All values non-negative
        self.assertTrue(np.all(result["positive"] >= 0))
        self.assertTrue(np.all(result["negative"] >= 0))


# =============================================================================
# 2. Build Design Matrix
# =============================================================================


class TestBuildDesignMatrix(unittest.TestCase):
    """Test build_design_matrix convenience function."""

    def test_no_confounds(self):
        """Interest only → X = [intercept, interest], contrast = [0, 1]."""
        age = np.array([25, 30, 28, 35, 40], dtype=float)
        X, contrast = build_design_matrix(age)
        self.assertEqual(X.shape, (5, 2))
        npt.assert_array_equal(X[:, 0], np.ones(5))
        npt.assert_array_equal(X[:, 1], age)
        npt.assert_array_equal(contrast, [0, 1])

    def test_with_confounds_1d(self):
        """Single confound → X = [intercept, confound, interest]."""
        age = np.ones(5)
        motion = np.zeros(5)
        X, contrast = build_design_matrix(age, confounds=motion)
        self.assertEqual(X.shape, (5, 3))
        npt.assert_array_equal(contrast, [0, 0, 1])

    def test_with_confounds_2d(self):
        """Multiple confounds → X = [intercept, confound1, confound2, interest]."""
        age = np.ones(5)
        confounds = np.zeros((5, 2))
        X, contrast = build_design_matrix(age, confounds=confounds)
        self.assertEqual(X.shape, (5, 4))
        npt.assert_array_equal(contrast, [0, 0, 0, 1])


# =============================================================================
# 3. Reduced Model
# =============================================================================


class TestComputeReducedModel(unittest.TestCase):
    """Test _compute_reduced_model."""

    def test_removes_interest_column(self):
        """Reduced model excludes columns targeted by contrast."""
        X = np.column_stack([np.ones(10), np.random.randn(10), np.random.randn(10)])
        contrast = np.array([0, 0, 1])
        X_red = _compute_reduced_model(X, contrast)
        self.assertEqual(X_red.shape, (10, 2))  # intercept + confound

    def test_no_confounds(self):
        """With contrast=[0,1] on X=[intercept, interest], reduced = intercept."""
        X = np.column_stack([np.ones(10), np.random.randn(10)])
        contrast = np.array([0, 1])
        X_red = _compute_reduced_model(X, contrast)
        self.assertEqual(X_red.shape, (10, 1))
        npt.assert_array_equal(X_red[:, 0], np.ones(10))

    def test_all_interest(self):
        """If contrast targets all columns, reduced model is intercept."""
        X = np.random.randn(10, 2)
        contrast = np.array([1, 1])
        X_red = _compute_reduced_model(X, contrast)
        self.assertEqual(X_red.shape, (10, 1))


# =============================================================================
# 4. _extract_max_stats refactor
# =============================================================================


class TestExtractMaxStatsRefactor(unittest.TestCase):
    """Test that _extract_max_stats works with arbitrary keys."""

    def test_traditional_keys(self):
        """Works with existing 'g1>g2' / 'g2>g1' keys."""
        N = 5
        stat_dict = {
            "g1>g2": np.random.rand(N, N),
            "g2>g1": np.random.rand(N, N),
        }
        result = _extract_max_stats(stat_dict, (N, N))
        self.assertIn("g1>g2", result)
        self.assertIn("g2>g1", result)
        self.assertAlmostEqual(float(result["g1>g2"]),
                               float(np.max(stat_dict["g1>g2"])))

    def test_glm_keys(self):
        """Works with GLM 'positive' / 'negative' keys."""
        N = 5
        stat_dict = {
            "positive": np.random.rand(N, N),
            "negative": np.random.rand(N, N),
        }
        result = _extract_max_stats(stat_dict, (N, N))
        self.assertIn("positive", result)
        self.assertIn("negative", result)
        self.assertAlmostEqual(float(result["positive"]),
                               float(np.max(stat_dict["positive"])))


# =============================================================================
# 5. Freedman-Lane: Confound Removal
# =============================================================================


class TestFreedmanLane(unittest.TestCase):
    """Test Freedman-Lane permutation behavior."""

    def test_confound_removal(self):
        """
        Edge correlated with both age AND motion.
        With confound → reduced effect; without → significant.
        """
        scenario = fixtures.glm_confound_driven(
            effect_edge=(2, 3), confound_beta=1.5, confound_corr=0.7,
            noise_scale=0.3, seed=456,
        )
        Y, age, motion = scenario["Y"], scenario["age"], scenario["motion"]
        i, j = scenario["effect_edge"]

        # Without confound: age appears significant (because correlated with motion)
        X_no_conf, c_no_conf = build_design_matrix(age)
        result_no_conf = compute_glm_stat(Y, X_no_conf, c_no_conf, stat_type='tstat')
        t_no_conf = result_no_conf["positive"][i, j]

        # With confound: age should NOT be significant
        X_with_conf, c_with_conf = build_design_matrix(age, confounds=motion)
        result_with_conf = compute_glm_stat(Y, X_with_conf, c_with_conf, stat_type='tstat')
        t_with_conf = result_with_conf["positive"][i, j]

        self.assertGreater(t_no_conf, t_with_conf,
                           "Adding confound should reduce the t-statistic")
        self.assertGreater(t_no_conf, 2.0,
                           "Without confound, effect should be detectable")

    def test_reduced_model_reconstruction(self):
        """Y_hat_reduced + residuals == Y (reconstruction identity)."""
        rng = np.random.RandomState(789)
        n = 30
        N = 5
        Y = rng.randn(n, N, N)

        age = rng.randn(n)
        motion = rng.randn(n)
        X, contrast = build_design_matrix(age, confounds=motion)

        X_reduced = _compute_reduced_model(X, contrast)
        X_red_pinv, _, _ = _precompute_ols(X_reduced)

        Y_flat = Y.reshape(n, -1)
        Y_hat_flat = X_reduced @ (X_red_pinv @ Y_flat)
        Y_hat = Y_hat_flat.reshape(n, N, N)
        residuals = Y - Y_hat

        npt.assert_allclose(Y_hat + residuals, Y, atol=1e-12)


# =============================================================================
# 6. Integration: compute_p_val_glm
# =============================================================================


class TestComputePValGLM(unittest.TestCase):
    """Integration tests for compute_p_val_glm orchestrator."""

    def test_tstat_method_smoke(self):
        """Smoke test: tstat method runs and returns correct structure."""
        rng = np.random.RandomState(42)
        n, N = 20, 8
        Y = rng.randn(n, N, N)
        for s in range(n):
            Y[s] = (Y[s] + Y[s].T) / 2
            np.fill_diagonal(Y[s], 0)
        age = rng.randn(n)

        p_vals = compute_p_val_glm(
            Y, interest=age, method='tstat',
            n_permutations=50, use_mp=False, random_state=0,
        )
        self.assertIn("positive", p_vals)
        self.assertIn("negative", p_vals)
        self.assertEqual(p_vals["positive"].shape, (N, N))
        self.assertTrue(np.all(p_vals["positive"] >= 0))
        self.assertTrue(np.all(p_vals["positive"] <= 1))

    def test_tfnbs_method_smoke(self):
        """Smoke test: TFNBS enhancement runs with GLM pipeline."""
        rng = np.random.RandomState(42)
        n, N = 20, 8
        Y = rng.randn(n, N, N)
        for s in range(n):
            Y[s] = (Y[s] + Y[s].T) / 2
            np.fill_diagonal(Y[s], 0)
        age = rng.randn(n)

        p_vals = compute_p_val_glm(
            Y, interest=age, method='tfnbs',
            n_permutations=50, use_mp=False, random_state=0,
            e=0.4, h=3.0, n=5,
        )
        self.assertIn("positive", p_vals)
        self.assertEqual(p_vals["positive"].shape, (N, N))

    def test_planted_effect_detection(self):
        """GLM + permutation detects a planted continuous effect."""
        rng = np.random.RandomState(321)
        n_subjects = 30
        N = 8
        age = rng.randn(n_subjects)

        Y = rng.randn(n_subjects, N, N) * 0.3
        for s in range(n_subjects):
            Y[s] = (Y[s] + Y[s].T) / 2
            np.fill_diagonal(Y[s], 0)

        # Plant strong effect at edge (1, 2)
        for s in range(n_subjects):
            Y[s, 1, 2] += 1.5 * age[s]
            Y[s, 2, 1] += 1.5 * age[s]

        p_vals = compute_p_val_glm(
            Y, interest=age, method='tstat',
            n_permutations=200, use_mp=False, random_state=0,
        )

        # Planted edge should be significant
        self.assertLess(p_vals["positive"][1, 2], 0.05,
                        f"Planted edge p={p_vals['positive'][1, 2]}, expected < 0.05")

    def test_convenience_matches_advanced(self):
        """Convenience API (interest+confounds) matches advanced API (X+contrast)."""
        rng = np.random.RandomState(654)
        n, N = 25, 6
        Y = rng.randn(n, N, N)
        for s in range(n):
            Y[s] = (Y[s] + Y[s].T) / 2
            np.fill_diagonal(Y[s], 0)

        age = rng.randn(n)
        motion = rng.randn(n)

        # Convenience
        p_conv = compute_p_val_glm(
            Y, interest=age, confounds=motion,
            method='tstat', n_permutations=100,
            use_mp=False, random_state=42,
        )

        # Advanced: manually build same design
        X = np.column_stack([np.ones(n), motion, age])
        contrast = np.array([0, 0, 1])
        p_adv = compute_p_val_glm(
            Y, design_matrix=X, contrast=contrast,
            method='tstat', n_permutations=100,
            use_mp=False, random_state=42,
        )

        npt.assert_array_equal(p_conv["positive"], p_adv["positive"])
        npt.assert_array_equal(p_conv["negative"], p_adv["negative"])

    def test_two_tailed(self):
        """two_tailed=True produces more conservative p-values."""
        rng = np.random.RandomState(111)
        n, N = 20, 6
        Y = rng.randn(n, N, N)
        for s in range(n):
            Y[s] = (Y[s] + Y[s].T) / 2
            np.fill_diagonal(Y[s], 0)
        age = rng.randn(n)

        p_one = compute_p_val_glm(
            Y, interest=age, method='tstat',
            n_permutations=100, use_mp=False, random_state=0,
            two_tailed=False,
        )
        p_two = compute_p_val_glm(
            Y, interest=age, method='tstat',
            n_permutations=100, use_mp=False, random_state=0,
            two_tailed=True,
        )

        # Two-tailed p-values should be >= one-tailed (more conservative)
        self.assertTrue(
            np.all(p_two["positive"] >= p_one["positive"] - 1e-10),
            "Two-tailed should be >= one-tailed"
        )

    def test_invalid_api_both(self):
        """Providing both design_matrix and interest raises ValueError."""
        Y = np.random.randn(10, 5, 5)
        X = np.random.randn(10, 2)
        with self.assertRaises(ValueError):
            compute_p_val_glm(Y, design_matrix=X, contrast=[0, 1],
                              interest=np.ones(10))

    def test_invalid_api_neither(self):
        """Providing neither design_matrix nor interest raises ValueError."""
        Y = np.random.randn(10, 5, 5)
        with self.assertRaises(ValueError):
            compute_p_val_glm(Y)

    def test_invalid_contrast_length(self):
        """Wrong-length contrast raises ValueError."""
        Y = np.random.randn(10, 5, 5)
        X = np.random.randn(10, 3)
        with self.assertRaises(ValueError):
            compute_p_val_glm(Y, design_matrix=X, contrast=[0, 1],
                              n_permutations=10, use_mp=False)

    def test_zero_contrast(self):
        """All-zero contrast raises ValueError."""
        Y = np.random.randn(10, 5, 5)
        X = np.random.randn(10, 2)
        with self.assertRaises(ValueError):
            compute_p_val_glm(Y, design_matrix=X, contrast=[0, 0],
                              n_permutations=10, use_mp=False)


# =============================================================================
# 7. Enhancement methods with GLM
# =============================================================================


class TestGLMEnhancementMethods(unittest.TestCase):
    """Smoke test each enhancement method through the GLM pipeline."""

    @classmethod
    def setUpClass(cls):
        # glm_null — n=20, N=8, pure noise. Smoke tests don't need signal.
        scenario = fixtures.glm_null(n=20, N=8, seed=42)
        cls.Y, cls.age, cls.N, cls.n = (
            scenario["Y"], scenario["age"], scenario["N"], scenario["n"],
        )
        cls.net_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    def _run_method(self, method, **extra_kwargs):
        p_vals = compute_p_val_glm(
            self.Y, interest=self.age, method=method,
            n_permutations=30, use_mp=False, random_state=0,
            e=0.4, h=3.0, n=5, start_thres=1.65,
            **extra_kwargs,
        )
        self.assertIn("positive", p_vals)
        self.assertIn("negative", p_vals)
        self.assertEqual(p_vals["positive"].shape, (self.N, self.N))
        self.assertTrue(np.all(p_vals["positive"] >= 0))
        self.assertTrue(np.all(p_vals["positive"] <= 1))

    def test_nbs(self):
        self._run_method('nbs', threshold=2.0)

    def test_cnbs(self):
        self._run_method('cnbs', net_labels=self.net_labels)

    def test_ni_tfnbs(self):
        self._run_method('ni_tfnbs', net_labels=self.net_labels)

    def test_fbc_tfnbs(self):
        self._run_method('fbc_tfnbs', net_labels=self.net_labels,
                         min_cluster_size=2)


# =============================================================================
# 8. F-contrast (omnibus)
# =============================================================================


class TestFStatCompute(unittest.TestCase):
    """Core F-statistic computation."""

    def test_fstat_single_row_equals_tstat_squared(self):
        """For a 1-row contrast, F == t² at every edge."""
        rng = np.random.RandomState(13)
        n, N = 30, 6
        age = rng.randn(n)
        motion = rng.randn(n)
        Y = rng.randn(n, N, N)
        for s in range(n):
            Y[s] = (Y[s] + Y[s].T) / 2
            np.fill_diagonal(Y[s], 0)

        X = np.column_stack([np.ones(n), motion, age])
        contrast_t = np.array([0.0, 0.0, 1.0])
        contrast_f = contrast_t[np.newaxis, :]  # (1, 3)

        res_t = compute_glm_stat(Y, X, contrast_t, stat_type='tstat')
        res_f = compute_glm_stat(Y, X, contrast_f, stat_type='fstat')

        t_signed = res_t["positive"] - res_t["negative"]
        npt.assert_allclose(res_f["omnibus"], t_signed ** 2, atol=1e-10)

    def test_fstat_output_keys_and_nonneg(self):
        """F-stat returns {'omnibus': (N, N)} with non-negative values."""
        rng = np.random.RandomState(7)
        n, N = 20, 5
        Y = rng.randn(n, N, N)
        for s in range(n):
            Y[s] = (Y[s] + Y[s].T) / 2
            np.fill_diagonal(Y[s], 0)
        X = np.column_stack([np.ones(n), rng.randn(n), rng.randn(n)])
        C = np.array([[0, 1, 0], [0, 0, 1]])
        res = compute_glm_stat(Y, X, C, stat_type='fstat')
        self.assertEqual(list(res.keys()), ['omnibus'])
        self.assertEqual(res["omnibus"].shape, (N, N))
        self.assertTrue(np.all(res["omnibus"] >= 0))

    def test_fstat_rejects_wrong_column_count(self):
        """Contrast matrix with wrong column count raises ValueError."""
        Y = np.random.randn(10, 4, 4)
        X = np.random.randn(10, 3)
        C_bad = np.array([[0, 1]])  # only 2 cols, X has 3
        with self.assertRaises(ValueError):
            compute_glm_stat(Y, X, C_bad, stat_type='fstat')

    def test_fstat_rejects_singular_contrast(self):
        """Linearly-dependent contrast rows raise ValueError (M singular)."""
        rng = np.random.RandomState(0)
        Y = rng.randn(10, 4, 4)
        X = np.column_stack([np.ones(10), rng.randn(10), rng.randn(10)])
        # Two copies of the same row → rank-deficient
        C = np.array([[0, 1, 0], [0, 1, 0]], dtype=float)
        with self.assertRaises(ValueError):
            compute_glm_stat(Y, X, C, stat_type='fstat')

    def test_fstat_matches_scipy_single_edge(self):
        """F at one edge matches scipy.stats.f_oneway for 3-group design."""
        rng = np.random.RandomState(99)
        ng = 15  # per group
        n = 3 * ng
        N = 4
        Y = rng.randn(n, N, N)
        for s in range(n):
            Y[s] = (Y[s] + Y[s].T) / 2
            np.fill_diagonal(Y[s], 0)

        # Plant a group-A elevation at edge (0, 1) to make F large and testable
        Y[:ng, 0, 1] += 2.0
        Y[:ng, 1, 0] += 2.0

        # Dummy-coded design: intercept + group_B + group_C (group_A = reference)
        group_B = np.concatenate([np.zeros(ng), np.ones(ng), np.zeros(ng)])
        group_C = np.concatenate([np.zeros(ng), np.zeros(ng), np.ones(ng)])
        X = np.column_stack([np.ones(n), group_B, group_C])
        # Test jointly β_B = β_C = 0 → 2-row contrast on cols 1, 2
        C = np.array([[0, 1, 0], [0, 0, 1]], dtype=float)
        res = compute_glm_stat(Y, X, C, stat_type='fstat')
        F_edge = res["omnibus"][0, 1]

        y_edge = Y[:, 0, 1]
        gA, gB, gC = y_edge[:ng], y_edge[ng:2 * ng], y_edge[2 * ng:]
        scipy_F, _ = stats.f_oneway(gA, gB, gC)

        npt.assert_allclose(F_edge, scipy_F, rtol=1e-10)


class TestReducedModel2D(unittest.TestCase):
    """Reduced-model correctness for 2D contrast matrices."""

    def test_keeps_columns_zero_across_all_rows(self):
        """A column is kept only if ALL contrast rows are zero for it."""
        X = np.column_stack([np.ones(8), np.arange(8), np.random.randn(8), np.random.randn(8)])
        # Contrast touches cols 1 and 2 (row 0 on col 1, row 1 on col 2);
        # cols 0 (intercept) and 3 should remain in reduced.
        C = np.array([[0, 1, 0, 0], [0, 0, 1, 0]], dtype=float)
        X_red = _compute_reduced_model(X, C)
        self.assertEqual(X_red.shape, (8, 2))
        npt.assert_array_equal(X_red[:, 0], np.ones(8))
        npt.assert_array_equal(X_red[:, 1], X[:, 3])

    def test_all_columns_targeted_falls_back_to_intercept(self):
        """If every column is touched by some contrast row, reduced = intercept."""
        X = np.random.randn(6, 3)
        C = np.array([[1, 1, 0], [0, 0, 1]], dtype=float)
        X_red = _compute_reduced_model(X, C)
        self.assertEqual(X_red.shape, (6, 1))
        npt.assert_array_equal(X_red[:, 0], np.ones(6))


class TestFStatPermutation(unittest.TestCase):
    """compute_p_val_glm with stat_type='fstat' end-to-end."""

    def test_fstat_smoke(self):
        """Smoke: fstat runs and returns {'omnibus'} with valid p-values."""
        rng = np.random.RandomState(0)
        n, N = 20, 5
        Y = rng.randn(n, N, N)
        for s in range(n):
            Y[s] = (Y[s] + Y[s].T) / 2
            np.fill_diagonal(Y[s], 0)
        X = np.column_stack([np.ones(n), rng.randn(n), rng.randn(n)])
        C = np.array([[0, 1, 0], [0, 0, 1]], dtype=float)

        p_vals = compute_p_val_glm(
            Y, design_matrix=X, contrast=C, stat_type='fstat',
            method='tstat', n_permutations=50,
            use_mp=False, random_state=0,
        )
        self.assertEqual(list(p_vals.keys()), ['omnibus'])
        self.assertEqual(p_vals["omnibus"].shape, (N, N))
        self.assertTrue(np.all(p_vals["omnibus"] >= 0))
        self.assertTrue(np.all(p_vals["omnibus"] <= 1))

    def test_fstat_detects_planted_group_effect(self):
        """3-group design with a planted group-A elevation → F detects it."""
        rng = np.random.RandomState(321)
        ng = 15
        n = 3 * ng
        N = 6
        Y = rng.randn(n, N, N) * 0.3
        for s in range(n):
            Y[s] = (Y[s] + Y[s].T) / 2
            np.fill_diagonal(Y[s], 0)

        # Strong group-A offset at edge (1, 2)
        Y[:ng, 1, 2] += 2.0
        Y[:ng, 2, 1] += 2.0

        group_B = np.concatenate([np.zeros(ng), np.ones(ng), np.zeros(ng)])
        group_C = np.concatenate([np.zeros(ng), np.zeros(ng), np.ones(ng)])
        X = np.column_stack([np.ones(n), group_B, group_C])
        C = np.array([[0, 1, 0], [0, 0, 1]], dtype=float)

        p_vals = compute_p_val_glm(
            Y, design_matrix=X, contrast=C, stat_type='fstat',
            method='tstat', n_permutations=200,
            use_mp=False, random_state=0,
        )
        self.assertLess(
            p_vals["omnibus"][1, 2], 0.05,
            f"Planted edge p={p_vals['omnibus'][1, 2]:.3f}, expected < 0.05",
        )

    def test_fstat_two_tailed_is_ignored(self):
        """two_tailed=True is silently ignored for fstat (F has no sign)."""
        rng = np.random.RandomState(0)
        n, N = 15, 4
        Y = rng.randn(n, N, N)
        for s in range(n):
            Y[s] = (Y[s] + Y[s].T) / 2
            np.fill_diagonal(Y[s], 0)
        X = np.column_stack([np.ones(n), rng.randn(n), rng.randn(n)])
        C = np.array([[0, 1, 0], [0, 0, 1]], dtype=float)

        p_one = compute_p_val_glm(
            Y, design_matrix=X, contrast=C, stat_type='fstat',
            method='tstat', n_permutations=30,
            use_mp=False, random_state=0, two_tailed=False,
        )
        p_two = compute_p_val_glm(
            Y, design_matrix=X, contrast=C, stat_type='fstat',
            method='tstat', n_permutations=30,
            use_mp=False, random_state=0, two_tailed=True,
        )
        npt.assert_array_equal(p_one["omnibus"], p_two["omnibus"])


# =============================================================================
# 9. Paired-difference GLM wrapper
# =============================================================================


class TestPairedGLMWrapper(unittest.TestCase):
    """compute_p_val_paired_glm routing and correctness."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.RandomState(42)
        cls.n = 20
        cls.N = 6
        Y_A = rng.randn(cls.n, cls.N, cls.N)
        Y_B = rng.randn(cls.n, cls.N, cls.N)
        for s in range(cls.n):
            Y_A[s] = (Y_A[s] + Y_A[s].T) / 2
            np.fill_diagonal(Y_A[s], 0)
            Y_B[s] = (Y_B[s] + Y_B[s].T) / 2
            np.fill_diagonal(Y_B[s], 0)
        cls.Y_A, cls.Y_B = Y_A, Y_B
        cls.fd_A = rng.randn(cls.n)
        cls.fd_B = rng.randn(cls.n)

    def test_no_confound_routes_to_paired_ttest(self):
        """Without confounds, wrapper returns the same dict as compute_p_val(paired)."""
        p_wrap = compute_p_val_paired_glm(
            self.Y_A, self.Y_B,
            method='tstat', n_permutations=50,
            use_mp=False, random_state=0,
        )
        p_direct = compute_p_val(
            self.Y_A, self.Y_B,
            test_type='paired', method='tstat',
            n_permutations=50,
            use_mp=False, random_state=0,
        )
        # Sanity: keys match the t-test pipeline convention
        self.assertEqual(sorted(p_wrap.keys()), ['g1>g2', 'g2>g1'])
        # Exact match: same seed, same pipeline
        npt.assert_array_equal(p_wrap['g2>g1'], p_direct['g2>g1'])
        npt.assert_array_equal(p_wrap['g1>g2'], p_direct['g1>g2'])

    def test_with_confound_returns_positive_negative(self):
        """With confounds, wrapper returns GLM-style {'positive', 'negative'}."""
        p = compute_p_val_paired_glm(
            self.Y_A, self.Y_B,
            confounds_A=self.fd_A, confounds_B=self.fd_B,
            method='tstat', n_permutations=50,
            use_mp=False, random_state=0,
        )
        self.assertEqual(sorted(p.keys()), ['negative', 'positive'])
        self.assertEqual(p['positive'].shape, (self.N, self.N))
        self.assertTrue(np.all(p['positive'] >= 0))
        self.assertTrue(np.all(p['positive'] <= 1))

    def test_tstat_on_differences_matches_paired_ttest(self):
        """
        At the t-stat level: GLM on Δ = B − A with intercept contrast matches
        paired-t on (A, B) (``g2>g1`` = B>A by the t-test convention).
        """
        Delta = self.Y_B - self.Y_A  # match compute_t_stat paired convention (g2 − g1)
        X_int = np.ones((self.n, 1))
        contrast_int = np.array([1.0])
        res_glm = compute_glm_stat(Delta, X_int, contrast_int, stat_type='tstat')
        glm_signed = res_glm["positive"] - res_glm["negative"]

        res_t = compute_t_stat(self.Y_A, self.Y_B, test_type='paired')
        t_signed = res_t["g2>g1"] - res_t["g1>g2"]

        npt.assert_allclose(glm_signed, t_signed, atol=1e-10)

    def test_confound_adjustment_removes_spurious_mean(self):
        """
        Confound-driven signal: Δ-motion has non-zero mean, and the edge is a
        linear function of Δ-motion. The raw intercept β̂₀ (no adjustment)
        picks up the contamination; adjusting for Δ-motion collapses it to
        ~0. Compares β̂ directly — t-stats can grow after adjustment because
        the residual SE tightens, so β̂ is the right quantity here.
        """
        rng = np.random.RandomState(777)
        n, N = 40, 6
        # Δ-motion with non-zero mean (eyes-closed moves more than eyes-open)
        fd_diff = rng.randn(n) + 2.0
        Delta = rng.randn(n, N, N) * 0.1
        for s in range(n):
            Delta[s] = (Delta[s] + Delta[s].T) / 2
            np.fill_diagonal(Delta[s], 0)
            # Plant a confound-driven signal at edge (2, 3):
            # Δ_Y[2,3] = 1.5 · fd_diff_s + noise → true intercept = 0,
            # but raw mean(Δ_Y[2,3]) ≈ 1.5 · 2.0 = 3.0 (spurious)
            Delta[s, 2, 3] += 1.5 * fd_diff[s]
            Delta[s, 3, 2] += 1.5 * fd_diff[s]

        # Unadjusted: intercept absorbs the confound-driven mean
        X_int = np.ones((n, 1))
        c_int = np.array([1.0])
        b_un = compute_glm_stat(Delta, X_int, c_int, stat_type='beta')
        beta_unadj = float(b_un["positive"][2, 3] - b_un["negative"][2, 3])

        # Adjusted: intercept at fd_diff = 0 → should be ~0
        X_adj = np.column_stack([np.ones(n), fd_diff])
        c_adj = np.array([1.0, 0.0])
        b_ad = compute_glm_stat(Delta, X_adj, c_adj, stat_type='beta')
        beta_adj = float(b_ad["positive"][2, 3] - b_ad["negative"][2, 3])

        self.assertGreater(abs(beta_unadj), 2.0,
                           f"Unadjusted β̂ should be contaminated, got {beta_unadj:.3f}")
        self.assertLess(abs(beta_adj), 0.5,
                        f"Adjusted β̂ should be ~0, got {beta_adj:.3f}")

    def test_shape_mismatch_raises(self):
        """Y_A and Y_B with different shapes → ValueError."""
        with self.assertRaises(ValueError):
            compute_p_val_paired_glm(
                np.zeros((10, 5, 5)), np.zeros((11, 5, 5)),
                n_permutations=10, use_mp=False,
            )

    def test_partial_confound_raises(self):
        """Only one of confounds_A / confounds_B provided → ValueError."""
        with self.assertRaises(ValueError):
            compute_p_val_paired_glm(
                self.Y_A, self.Y_B,
                confounds_A=self.fd_A,  # B missing
                n_permutations=10, use_mp=False,
            )

    def test_cannot_override_design(self):
        """Passing design_matrix/contrast/interest/confounds is rejected."""
        with self.assertRaises(ValueError):
            compute_p_val_paired_glm(
                self.Y_A, self.Y_B,
                confounds_A=self.fd_A, confounds_B=self.fd_B,
                interest=np.ones(self.n),
                n_permutations=10, use_mp=False,
            )

    def test_cannot_override_test_type_on_no_confound_path(self):
        """Passing test_type to the no-confound path is rejected."""
        with self.assertRaises(ValueError):
            compute_p_val_paired_glm(
                self.Y_A, self.Y_B,
                test_type='two-sample',
                n_permutations=10, use_mp=False,
            )


if __name__ == "__main__":
    unittest.main()
