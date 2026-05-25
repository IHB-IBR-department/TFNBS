import unittest
from unittest import TestCase
import numpy as np
from conninfpy.pairwise_stats import compute_t_stat, compute_t_stat_diff
from conninfpy.tfnbs_score import (
    get_tfnbs_score_networkx,
    get_tfnbs_score,
    get_tfnbs_score_baseline,
    get_network_informed_tfnbs_score,
    get_fbc_tfnbs_score,
    HAS_NUMBA,
)
from conninfpy.synth_datasets import generate_fc_matrices, ModularDatasetGenerator
from conninfpy.pairwise_stats import compute_t_stat
from tests import fixtures
from conninfpy.utils import fisher_r_to_z
import time


class TestTFNBS(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.small_matrix = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
        cls.invalid_matrix = np.array([[1, 1, 2], [1, 0, 1], [2, 1, 0]])

        # Scenario: small_two_sample — N=30
        group1, group2, (cov1, cov2) = fixtures.small_two_sample()
        t_stat_30 = compute_t_stat(fisher_r_to_z(group1),
                                   fisher_r_to_z(group2), test_type='two-sample')
        cls.fc_sim_30 = {"t_stat": t_stat_30,
                         "cov1": cov1.copy(), "cov2": cov2.copy()}

        # Scenario: medium_two_sample — N=100
        group1, group2, (cov1, cov2) = fixtures.medium_two_sample()
        t_stat_100 = compute_t_stat(fisher_r_to_z(group1),
                                    fisher_r_to_z(group2), test_type='two-sample')
        cls.fc_sim_100 = {"t_stat": t_stat_100,
                          "cov1": cov1, "cov2": cov2}

    def setUp(self):
        self.E = 0.4
        self.H = 3
        self.n = 10

    def run_and_measure(self, func, matrix, n_runs=3):
        """Helper function to measure execution time of a function."""
        times = []
        for _ in range(n_runs):
            start_time = time.time()
            func(matrix, self.E, self.H, self.n)
            times.append(time.time() - start_time)
        return min(times)

    def test_small_matrix(self):
        """Test with a small 3x3 matrix and known output."""
        statsmat = self.small_matrix
        result_pos = get_tfnbs_score_networkx(statsmat, 1, 1, 2, start_thres=0)
        result_neg = get_tfnbs_score_networkx(-statsmat, 1, 1, 2, start_thres=0)

        expected_pos = np.array([[0.0, 3.0, 5.0], [3.0, 0.0, 3.0], [5.0, 3.0, 0.0]])
        expected_neg = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        np.testing.assert_allclose(result_pos, expected_pos, rtol=1e-5)
        np.testing.assert_allclose(result_pos, expected_pos, rtol=1e-5)

    def test_small_matrix_scipy(self):
        """Test with a small 3x3 matrix and known output."""
        statsmat = self.small_matrix
        result_nx = get_tfnbs_score_networkx(statsmat, 1, 1, 2, start_thres=0)
        result = get_tfnbs_score(statsmat, 1, 1, 2, start_thres=0)

        expected = np.array([[0.0, 3.0, 5.0], [3.0, 0.0, 3.0], [5.0, 3.0, 0.0]])
        np.testing.assert_allclose(result_nx, result, rtol=1e-5)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_small_matrix_diff_pars(self):
        """Test with a small 3x3 matrix and known output."""
        statsmat = self.small_matrix
        result_nx = get_tfnbs_score_networkx(statsmat, 0.4, 4, 100, start_thres=0)
        result_scipy = get_tfnbs_score(statsmat, 0.4, 4, 100, start_thres=0)

        expected = np.array([[0.0, 0.326, 6.677], [0.326, 0.0, 0.326], [6.677, 0.326, 0.0]])
        np.testing.assert_allclose(result_nx, expected, rtol=1e-2)
        np.testing.assert_allclose(result_scipy, expected, rtol=1e-2)

    def test_stat_symmetry(self):
        statsmat = self.small_matrix
        result = get_tfnbs_score_networkx(statsmat, 1, 1, 2)
        self.assertTrue(np.allclose(result, result.T))

    def test_input_no_self_connections(self):
        statsmat = self.invalid_matrix
        with self.assertRaises(ValueError):
            get_tfnbs_score_networkx(statsmat, 1, 1, 2)

    def test_tfnbs_real_matrix_30N(self):
        t_stat = self.fc_sim_30["t_stat"]
        score_pos = get_tfnbs_score(t_stat['g2>g1'], self.E, self.H, self.n, start_thres=1.7)
        score_neg = get_tfnbs_score(t_stat['g1>g2'], self.E, self.H, self.n, start_thres=1.7)

        self.assertTrue((score_pos >= 0).all())
        self.assertTrue((score_neg >= 0).all())

    def test_time_consumption_scipy_vs_networkx(self):
        """Test that scipy implementation is faster than networkx."""
        time_original = self.run_and_measure(get_tfnbs_score_networkx, self.fc_sim_100["t_stat"]['g2>g1'])
        time_scipy = self.run_and_measure(get_tfnbs_score, self.fc_sim_100["t_stat"]['g2>g1'])

        self.assertLess(time_scipy, time_original)

    def test_scipy_list_params(self):
        statsmat = self.fc_sim_30["t_stat"]['g2>g1']
        result = get_tfnbs_score(statsmat, [0.4, 0.4], [1, 2], 10)
        result_nx = get_tfnbs_score_networkx(statsmat, [0.4, 0.4], [1, 2], 10)

        self.assertTrue(result.shape[2] == 2)
        self.assertTrue(result_nx.shape[2] == 2)

    def test_baseline_correctness(self):
        """Test that baseline and optimized versions produce similar results."""
        statsmat = self.small_matrix
        result_baseline = get_tfnbs_score_baseline(statsmat, self.E, self.H, self.n, start_thres=0)
        result_optimized = get_tfnbs_score(statsmat, self.E, self.H, self.n, start_thres=0)
        np.testing.assert_allclose(result_baseline, result_optimized, rtol=1e-10)

        # Test on larger matrix - compare upper triangles only
        # Note: baseline may have minor asymmetry due to floating point order,
        # optimized version explicitly ensures symmetry
        t_stat = self.fc_sim_100["t_stat"]['g2>g1']
        result_baseline_large = get_tfnbs_score_baseline(t_stat, self.E, self.H, self.n, start_thres=1.7)
        result_optimized_large = get_tfnbs_score(t_stat, self.E, self.H, self.n, start_thres=1.7)

        # Optimized version should be symmetric
        self.assertTrue(np.allclose(result_optimized_large, result_optimized_large.T))

        # Compare upper triangles (where both should agree)
        triu_idx = np.triu_indices(100, k=1)
        np.testing.assert_allclose(
            result_baseline_large[triu_idx],
            result_optimized_large[triu_idx],
            rtol=1e-10
        )

    def test_optimization_speedup(self):
        """
        Test that optimized get_tfnbs_score is faster than baseline.

        Compares performance across different matrix sizes and threshold counts.
        Optimized version should be at least 1.2x faster.
        """
        np.random.seed(42)

        test_configs = [
            (50, 50),   # (matrix_size, n_thresholds)
            (80, 100),
        ]

        for size, n_thresh in test_configs:
            # Generate random symmetric matrix
            t_stats = np.random.randn(size, size)
            t_stats = (t_stats + t_stats.T) / 2
            np.fill_diagonal(t_stats, 0)
            t_stats = np.abs(t_stats) * 3

            # Measure baseline
            n_runs = 5
            baseline_times = []
            for _ in range(n_runs):
                start = time.perf_counter()
                result_baseline = get_tfnbs_score_baseline(t_stats, 0.5, 2.0, n_thresh)
                baseline_times.append(time.perf_counter() - start)
            time_baseline = min(baseline_times)

            # Measure optimized
            optimized_times = []
            for _ in range(n_runs):
                start = time.perf_counter()
                result_optimized = get_tfnbs_score(t_stats, 0.5, 2.0, n_thresh)
                optimized_times.append(time.perf_counter() - start)
            time_optimized = min(optimized_times)

            # Check correctness
            np.testing.assert_allclose(result_baseline, result_optimized, rtol=1e-10,
                                       err_msg=f"Results differ for size={size}, n={n_thresh}")

            # Check speedup (optimized should be faster)
            speedup = time_baseline / time_optimized
            self.assertGreater(speedup, 1.2,
                              f"Optimized version not fast enough for size={size}, n={n_thresh}. "
                              f"Speedup: {speedup:.2f}x (expected > 1.2x)")

    def test_baseline_list_params(self):
        """Test that baseline handles list parameters correctly."""
        statsmat = self.fc_sim_30["t_stat"]['g2>g1']
        e_list = [0.3, 0.4, 0.5]
        h_list = [2.0, 2.5, 3.0]

        result_baseline = get_tfnbs_score_baseline(statsmat, e_list, h_list, 10)
        result_optimized = get_tfnbs_score(statsmat, e_list, h_list, 10)

        self.assertEqual(result_baseline.shape, (30, 30, 3))
        self.assertEqual(result_optimized.shape, (30, 30, 3))
        np.testing.assert_allclose(result_baseline, result_optimized, rtol=1e-10)


class TestNetworkInformedTFNBS(TestCase):
    """
    Tests for Network-Informed TFNBS implementation.

    Uses ModularDatasetGenerator to create controlled test scenarios
    that validate the NI-TFNBS algorithm behavior.
    """

    def setUp(self):
        """Set up generator for testing."""
        self.N = 50
        self.n_modules = 5
        self.seed = 42
        self.gen = ModularDatasetGenerator(
            N=self.N,
            n_modules=self.n_modules,
            intra_corr=0.6,
            inter_corr=0.1,
            seed=self.seed
        )

    def test_basic_output_shape(self):
        """Test that NI-TFNBS returns correct output shape."""
        t_stats = np.zeros((self.N, self.N))
        t_stats[0, 1] = t_stats[1, 0] = 2.5
        t_stats[1, 2] = t_stats[2, 1] = 2.0

        labels = self.gen.labels
        result = get_network_informed_tfnbs_score(t_stats, labels, e=0.5, h=2.0, n=50)

        self.assertEqual(result.shape, (self.N, self.N))
        np.testing.assert_allclose(result, result.T)

    def test_param_sweep_shape(self):
        """Test parameter sweep returns correct shape."""
        t_stats = np.zeros((self.N, self.N))
        t_stats[0, 1] = t_stats[1, 0] = 2.5

        labels = self.gen.labels
        e_vals = [0.3, 0.5, 0.7]
        h_vals = [1.5, 2.0, 2.5]

        result = get_network_informed_tfnbs_score(t_stats, labels, e=e_vals, h=h_vals, n=50)

        self.assertEqual(result.shape, (self.N, self.N, 3))

    def test_zero_matrix_returns_zeros(self):
        """Test that zero input returns zero output."""
        t_stats = np.zeros((self.N, self.N))
        labels = self.gen.labels

        result = get_network_informed_tfnbs_score(t_stats, labels, e=0.5, h=2.0, n=50)

        np.testing.assert_allclose(result, 0)

    def test_below_threshold_returns_zeros(self):
        """Test that values below start_thres give zero scores."""
        t_stats = np.ones((self.N, self.N)) * 1.0
        np.fill_diagonal(t_stats, 0)
        labels = self.gen.labels

        result = get_network_informed_tfnbs_score(
            t_stats, labels, e=0.5, h=2.0, n=50, start_thres=1.65
        )

        np.testing.assert_allclose(result, 0)

    def test_within_module_boost(self):
        """
        Test that edges within a densely activated module receive higher scores
        compared to isolated edges spanning different modules.
        """
        mask_within = self.gen.get_mask_within_module(0)

        # Scenario A: Dense activation within Module 0
        t_stats_dense = np.zeros((self.N, self.N))
        t_stats_dense[mask_within == 1] = 2.5

        # Scenario B: Same number of edges, but scattered across modules
        n_edges_within = np.sum(mask_within) // 2
        t_stats_scattered = np.zeros((self.N, self.N))

        rng = np.random.default_rng(123)
        placed = 0
        while placed < n_edges_within:
            i = rng.integers(0, self.N)
            j = rng.integers(0, self.N)
            if i != j and self.gen.labels[i] != self.gen.labels[j] and t_stats_scattered[i, j] == 0:
                t_stats_scattered[i, j] = 2.5
                t_stats_scattered[j, i] = 2.5
                placed += 1

        labels = self.gen.labels

        score_dense = get_network_informed_tfnbs_score(t_stats_dense, labels, e=0.5, h=2.0, n=50)
        score_scattered = get_network_informed_tfnbs_score(t_stats_scattered, labels, e=0.5, h=2.0, n=50)

        ni_ratio = np.sum(score_dense) / (np.sum(score_scattered) + 1e-10)

        self.assertGreater(ni_ratio, 1.0,
                           "Within-module dense pattern should score higher than scattered")

    def test_integration_with_generated_data(self):
        """
        End-to-end test: generate modular data, compute t-stats, apply NI-TFNBS.
        """
        mask = self.gen.get_mask_within_module(0)
        g1, g2, labels = self.gen.generate_data(
            effect_mask=mask,
            effect_size=0.3,
            n_samples_g1=30,
            n_samples_g2=30,
            time_points=200
        )

        t_stat_dict = compute_t_stat(g1, g2, test_type='two-sample')
        t_stats = t_stat_dict["g2>g1"]
        np.fill_diagonal(t_stats, 0)

        ni_scores = get_network_informed_tfnbs_score(t_stats, labels, e=0.5, h=2.0, n=50)

        self.assertFalse(np.any(np.isnan(ni_scores)), "NI-TFNBS produced NaN values")
        self.assertFalse(np.any(np.isinf(ni_scores)), "NI-TFNBS produced Inf values")
        np.testing.assert_allclose(ni_scores, ni_scores.T, rtol=1e-10)
        self.assertTrue(np.all(ni_scores >= 0), "Scores should be non-negative")

    def test_label_validation(self):
        """Test that invalid labels raise appropriate errors."""
        t_stats = np.zeros((self.N, self.N))
        t_stats[0, 1] = t_stats[1, 0] = 2.5

        wrong_labels = np.zeros(self.N + 5, dtype=int)

        with self.assertRaises(ValueError):
            get_network_informed_tfnbs_score(t_stats, wrong_labels, e=0.5, h=2.0, n=50)

    def test_non_contiguous_labels(self):
        """Test that non-contiguous labels (e.g., [0, 5, 10]) work correctly."""
        t_stats = np.zeros((self.N, self.N))
        t_stats[0, 1] = t_stats[1, 0] = 2.5
        t_stats[1, 2] = t_stats[2, 1] = 2.3

        labels = np.array([0, 5, 10] * (self.N // 3) + [0] * (self.N % 3))

        result = get_network_informed_tfnbs_score(t_stats, labels, e=0.5, h=2.0, n=50)

        self.assertEqual(result.shape, (self.N, self.N))
        self.assertFalse(np.any(np.isnan(result)))

    def test_block_weights_are_roi_order_invariant(self):
        """Undirected network-pair density must not depend on ROI ordering."""
        labels = np.array([0, 1, 0, 1])
        t_stats = np.zeros((4, 4), dtype=float)
        t_stats[0, 1] = t_stats[1, 0] = 2.0
        t_stats[0, 3] = t_stats[3, 0] = 2.0
        perm = np.array([1, 0, 2, 3])
        inv_perm = np.argsort(perm)

        score = get_network_informed_tfnbs_score(
            t_stats, labels, e=1.0, h=0.0, n=5, start_thres=1.0,
        )
        score_perm = get_network_informed_tfnbs_score(
            t_stats[np.ix_(perm, perm)],
            labels[perm],
            e=1.0,
            h=0.0,
            n=5,
            start_thres=1.0,
        )[np.ix_(inv_perm, inv_perm)]

        np.testing.assert_allclose(score, score_perm)


class TestTFNBSNumbaBackend(TestCase):
    """Test that numba backend produces identical results to scipy."""

    @classmethod
    def setUpClass(cls):
        cls.small_matrix = np.array([[0, 2.1, 0.5], [2.1, 0, 2.5], [0.5, 2.5, 0]])

        # Generate a realistic t-stat matrix
        effect_size = 0.2
        group1, group2, (cov1, cov2) = generate_fc_matrices(
            60, effect_size, n_samples_group1=30, n_samples_group2=20, seed=42
        )
        t_stat = compute_t_stat(
            fisher_r_to_z(group1), fisher_r_to_z(group2), test_type='two-sample'
        )
        cls.t_stat_60 = t_stat

        # Large matrix for scale test
        group1_l, group2_l, _ = generate_fc_matrices(
            200, effect_size, n_samples_group1=30, n_samples_group2=20, seed=99
        )
        t_stat_l = compute_t_stat(
            fisher_r_to_z(group1_l), fisher_r_to_z(group2_l), test_type='two-sample'
        )
        cls.t_stat_200 = t_stat_l

    def _compare_backends(self, t_stats, e, h, n, start_thres=1.65, atol=1e-10):
        """Helper: compare scipy vs numba backends."""
        result_scipy = get_tfnbs_score(t_stats, e, h, n, start_thres=start_thres, backend='scipy')
        result_numba = get_tfnbs_score(t_stats, e, h, n, start_thres=start_thres, backend='numba')
        np.testing.assert_allclose(result_scipy, result_numba, atol=atol,
                                   err_msg=f"Mismatch for e={e}, h={h}, n={n}")
        return result_scipy, result_numba

    def test_scalar_small_matrix(self):
        """Scalar (e, h) on small 3x3 matrix."""
        self._compare_backends(self.small_matrix, e=0.5, h=2.0, n=10, start_thres=0)

    def test_scalar_realistic_matrix(self):
        """Scalar (e, h) on realistic 60-node t-stat matrix."""
        self._compare_backends(self.t_stat_60['g2>g1'], e=0.4, h=3.0, n=30)

    def test_3d_multi_param(self):
        """3D multi-param (e=[list], h=[list]) produce identical output."""
        e_list = [0.3, 0.5, 0.7, 1.0]
        h_list = [1.5, 2.0, 3.0, 4.0]
        r_scipy, r_numba = self._compare_backends(
            self.t_stat_60['g2>g1'], e=e_list, h=h_list, n=20
        )
        self.assertEqual(r_scipy.shape, (60, 60, 4))
        self.assertEqual(r_numba.shape, (60, 60, 4))

    def test_zero_matrix(self):
        """Zero matrix returns zeros."""
        t = np.zeros((10, 10))
        self._compare_backends(t, e=0.5, h=2.0, n=10, start_thres=0)

    def test_all_below_threshold(self):
        """All values below start_thres returns zeros."""
        t = np.ones((10, 10)) * 0.5
        np.fill_diagonal(t, 0)
        r_scipy, r_numba = self._compare_backends(t, e=0.5, h=2.0, n=10, start_thres=1.65)
        np.testing.assert_allclose(r_scipy, 0)
        np.testing.assert_allclose(r_numba, 0)

    def test_single_edge(self):
        """Single edge above threshold."""
        t = np.zeros((5, 5))
        t[0, 1] = t[1, 0] = 2.5
        self._compare_backends(t, e=0.5, h=2.0, n=10)

    def test_fully_connected(self):
        """Fully connected matrix above threshold."""
        t = np.ones((8, 8)) * 3.0
        np.fill_diagonal(t, 0)
        self._compare_backends(t, e=0.5, h=2.0, n=10)

    def test_asymmetric_matrix(self):
        """Asymmetric (directed) matrix."""
        np.random.seed(123)
        t = np.abs(np.random.randn(10, 10)) * 3
        np.fill_diagonal(t, 0)
        # Make asymmetric
        t[0, 1] = 5.0
        t[1, 0] = 1.0
        self._compare_backends(t, e=0.5, h=2.0, n=20, start_thres=1.0)

    def test_large_matrix_200(self):
        """N=200 matrix for correctness at scale."""
        self._compare_backends(self.t_stat_200['g2>g1'], e=0.4, h=3.0, n=30)

    def test_large_matrix_200_3d(self):
        """N=200 matrix with multiple parameter pairs."""
        e_list = [0.3, 0.5, 1.0]
        h_list = [2.0, 3.0, 5.0]
        self._compare_backends(self.t_stat_200['g2>g1'], e=e_list, h=h_list, n=20)

    def test_backend_fallback_without_numba(self):
        """Test that backend='numba' gracefully falls back when numba unavailable."""
        import conninfpy.tfnbs_score as mod
        original = mod.HAS_NUMBA
        try:
            mod.HAS_NUMBA = False
            # Should fall back to scipy and produce correct results
            t = self.small_matrix
            result = get_tfnbs_score(t, e=0.5, h=2.0, n=10, start_thres=0, backend='numba')
            expected = get_tfnbs_score(t, e=0.5, h=2.0, n=10, start_thres=0, backend='scipy')
            np.testing.assert_allclose(result, expected, atol=1e-10)
        finally:
            mod.HAS_NUMBA = original

    @unittest.skipUnless(HAS_NUMBA, "Numba not installed")
    def test_numba_speedup(self):
        """When numba is installed, it should be faster than scipy on large matrices."""
        t = self.t_stat_200['g2>g1']
        # Warmup JIT
        get_tfnbs_score(t, e=0.4, h=3.0, n=30, backend='numba')

        n_runs = 5
        scipy_times = []
        numba_times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            get_tfnbs_score(t, e=0.4, h=3.0, n=30, backend='scipy')
            scipy_times.append(time.perf_counter() - start)

            start = time.perf_counter()
            get_tfnbs_score(t, e=0.4, h=3.0, n=30, backend='numba')
            numba_times.append(time.perf_counter() - start)

        speedup = min(scipy_times) / min(numba_times)
        print(f"\nNumba speedup: {speedup:.2f}x (scipy={min(scipy_times)*1000:.1f}ms, numba={min(numba_times)*1000:.1f}ms)")
        self.assertGreater(speedup, 1.0, "Numba should be at least as fast as scipy")


class TestFBCTFNBS(TestCase):
    """
    Tests for FBC-TFNBS (Functional Block Clustering TFNBS).

    Verifies that the min_cluster_size (m_min) indicator function works
    correctly per Paper Eq. 6: blocks with k < m_min are suppressed.
    """

    def setUp(self):
        """Set up a controlled network with known block structure."""
        self.N = 12
        # 3 modules of 4 nodes each: [0,1,2,3], [4,5,6,7], [8,9,10,11]
        self.labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

        # Build a t-stat matrix with known structure:
        # Module 0: dense activation (6 edges within module 0)
        # Module 1: sparse activation (only 1 edge)
        # Module 2: moderate activation (3 edges)
        self.t_stats = np.zeros((self.N, self.N))

        # Module 0: all 6 within-module edges active
        for i in range(4):
            for j in range(i + 1, 4):
                self.t_stats[i, j] = 2.5
                self.t_stats[j, i] = 2.5

        # Module 1: only 1 within-module edge active
        self.t_stats[4, 5] = 2.5
        self.t_stats[5, 4] = 2.5

        # Module 2: exactly 3 within-module edges active
        self.t_stats[8, 9] = 2.5
        self.t_stats[9, 8] = 2.5
        self.t_stats[9, 10] = 2.5
        self.t_stats[10, 9] = 2.5
        self.t_stats[10, 11] = 2.5
        self.t_stats[11, 10] = 2.5

    def test_small_blocks_suppressed(self):
        """Blocks with fewer than m_min edges should contribute 0 to the score."""
        # m_min=3: Module 1 has 1 edge (< 3) → suppressed
        scores = get_fbc_tfnbs_score(
            self.t_stats, self.labels,
            e=0.5, h=2.0, n=20, start_thres=1.5, min_cluster_size=3
        )

        # Module 1 edges should be 0 (only 1 edge, below m_min=3)
        self.assertEqual(scores[4, 5], 0.0,
                         "Edge in block with 1 edge should be suppressed at m_min=3")
        self.assertEqual(scores[5, 4], 0.0)

    def test_large_blocks_contribute(self):
        """Blocks with >= m_min edges should contribute > 0 to the score."""
        scores = get_fbc_tfnbs_score(
            self.t_stats, self.labels,
            e=0.5, h=2.0, n=20, start_thres=1.5, min_cluster_size=3
        )

        # Module 0 has 6 edges (>= 3) → should contribute
        self.assertGreater(scores[0, 1], 0.0,
                           "Edge in block with 6 edges should contribute at m_min=3")

        # Module 2 has exactly 3 edges (>= 3) → should contribute
        self.assertGreater(scores[8, 9], 0.0,
                           "Edge in block with exactly m_min edges should contribute")

    def test_changing_m_min_changes_output(self):
        """Different m_min values should produce different results."""
        scores_m1 = get_fbc_tfnbs_score(
            self.t_stats, self.labels,
            e=0.5, h=2.0, n=20, start_thres=1.5, min_cluster_size=1
        )
        scores_m3 = get_fbc_tfnbs_score(
            self.t_stats, self.labels,
            e=0.5, h=2.0, n=20, start_thres=1.5, min_cluster_size=3
        )
        scores_m5 = get_fbc_tfnbs_score(
            self.t_stats, self.labels,
            e=0.5, h=2.0, n=20, start_thres=1.5, min_cluster_size=5
        )

        # m_min=1: all edges contribute (all blocks have >= 1 edge)
        self.assertGreater(scores_m1[4, 5], 0.0,
                           "m_min=1 should include single-edge blocks")

        # m_min=3: Module 1 (1 edge) suppressed
        self.assertEqual(scores_m3[4, 5], 0.0,
                         "m_min=3 should suppress 1-edge blocks")
        self.assertGreater(scores_m3[0, 1], 0.0,
                           "m_min=3 should keep 6-edge blocks")

        # m_min=5: Module 2 (3 edges) also suppressed
        self.assertEqual(scores_m5[8, 9], 0.0,
                         "m_min=5 should suppress 3-edge blocks")
        self.assertGreater(scores_m5[0, 1], 0.0,
                           "m_min=5 should keep 6-edge blocks")

        # Total scores should decrease as m_min increases
        self.assertGreater(np.sum(scores_m1), np.sum(scores_m3))
        self.assertGreater(np.sum(scores_m3), np.sum(scores_m5))

    def test_m_min_boundary_exact(self):
        """Block with exactly m_min edges should be included (>= not >)."""
        # Module 2 has exactly 3 edges
        scores_m3 = get_fbc_tfnbs_score(
            self.t_stats, self.labels,
            e=0.5, h=2.0, n=20, start_thres=1.5, min_cluster_size=3
        )
        scores_m4 = get_fbc_tfnbs_score(
            self.t_stats, self.labels,
            e=0.5, h=2.0, n=20, start_thres=1.5, min_cluster_size=4
        )

        # m_min=3: Module 2 included (3 >= 3)
        self.assertGreater(scores_m3[8, 9], 0.0,
                           "Block with exactly m_min edges should be included")

        # m_min=4: Module 2 excluded (3 < 4)
        self.assertEqual(scores_m4[8, 9], 0.0,
                         "Block with fewer than m_min edges should be excluded")


class TestNITFNBSNormalization(TestCase):
    """Test NI-TFNBS normalization parameter (Exp 4: ablation study)."""

    def setUp(self):
        self.N = 20
        # Create a t-stat matrix with signal in one block
        self.t_stats = np.zeros((self.N, self.N))
        # Block (0,0): nodes 0-4, strong signal
        for i in range(5):
            for j in range(i + 1, 5):
                self.t_stats[i, j] = self.t_stats[j, i] = 2.5

        self.labels = np.array([0] * 5 + [1] * 5 + [2] * 5 + [3] * 5)

    def test_sqrt_is_default(self):
        """Default normalization should equal explicit sqrt."""
        result_default = get_network_informed_tfnbs_score(
            self.t_stats, self.labels, e=0.5, h=2.0, n=20
        )
        result_sqrt = get_network_informed_tfnbs_score(
            self.t_stats, self.labels, e=0.5, h=2.0, n=20,
            normalization="sqrt"
        )
        np.testing.assert_allclose(result_default, result_sqrt)

    def test_all_variants_produce_output(self):
        """All normalization variants should produce non-zero scores for active edges."""
        for norm in ("sqrt", "linear", "none"):
            result = get_network_informed_tfnbs_score(
                self.t_stats, self.labels, e=0.5, h=2.0, n=20,
                normalization=norm,
            )
            self.assertEqual(result.shape, (self.N, self.N),
                             f"Shape mismatch for normalization='{norm}'")
            self.assertGreater(
                result[0, 1], 0.0,
                f"Active edge should have positive score with normalization='{norm}'"
            )
            # Inactive edges should be zero
            self.assertEqual(
                result[0, 10], 0.0,
                f"Inactive edge should be zero with normalization='{norm}'"
            )

    def test_none_gives_largest_scores(self):
        """'none' (raw k) should give larger scores than 'sqrt' (k/sqrt(M))."""
        result_none = get_network_informed_tfnbs_score(
            self.t_stats, self.labels, e=0.5, h=2.0, n=20,
            normalization="none",
        )
        result_sqrt = get_network_informed_tfnbs_score(
            self.t_stats, self.labels, e=0.5, h=2.0, n=20,
            normalization="sqrt",
        )
        # Raw count k is always >= k/sqrt(M) when M >= 1
        self.assertGreaterEqual(
            result_none[0, 1], result_sqrt[0, 1],
            "'none' should give scores >= 'sqrt'"
        )

    def test_linear_gives_smallest_scores(self):
        """'linear' (k/M) should give smaller scores than 'sqrt' (k/sqrt(M))."""
        result_linear = get_network_informed_tfnbs_score(
            self.t_stats, self.labels, e=0.5, h=2.0, n=20,
            normalization="linear",
        )
        result_sqrt = get_network_informed_tfnbs_score(
            self.t_stats, self.labels, e=0.5, h=2.0, n=20,
            normalization="sqrt",
        )
        # k/M <= k/sqrt(M) when M >= 1
        self.assertLessEqual(
            result_linear[0, 1], result_sqrt[0, 1],
            "'linear' should give scores <= 'sqrt'"
        )

    def test_invalid_normalization_raises(self):
        """Invalid normalization value should raise ValueError."""
        with self.assertRaises(ValueError):
            get_network_informed_tfnbs_score(
                self.t_stats, self.labels, e=0.5, h=2.0, n=20,
                normalization="invalid",
            )

    def test_param_sweep_with_normalization(self):
        """Normalization should work with parameter sweep (3D output)."""
        e_vals = [0.3, 0.5, 0.7]
        h_vals = [1.5, 2.0, 2.5]
        for norm in ("sqrt", "linear", "none"):
            result = get_network_informed_tfnbs_score(
                self.t_stats, self.labels, e=e_vals, h=h_vals, n=20,
                normalization=norm,
            )
            self.assertEqual(result.shape, (self.N, self.N, 3),
                             f"3D shape mismatch for normalization='{norm}'")
