import unittest
from unittest import TestCase
import numpy as np
from tfnbs.pairwise_stats import compute_t_stat, compute_t_stat_diff
from tfnbs.tfnbs_score import (
    get_tfnbs_score_networkx,
    get_tfnbs_score,
    get_tfnbs_score_baseline,
    get_network_informed_tfnbs_score
)
from tfnbs.synth_datasets import generate_fc_matrices, ModularDatasetGenerator
from tfnbs.pairwise_stats import compute_t_stat_ind
from tfnbs.utils import fisher_r_to_z
import time


class TestTFNBS(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.small_matrix = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
        cls.invalid_matrix = np.array([[1, 1, 2], [1, 0, 1], [2, 1, 0]])
        effect_size = 0.2
        group1, group2, (cov1, cov2) = generate_fc_matrices(30,
                                                            effect_size,
                                                            n_samples_group1=30,
                                                            n_samples_group2=20,
                                                            seed=42)
        t_stat_30 = compute_t_stat(fisher_r_to_z(group1),
                                   fisher_r_to_z(group2), test_type='two-sample')

        cls.fc_sim_30 = {"t_stat": t_stat_30,
                         "cov1": cov1.copy(), "cov2": cov2.copy()}

        group1, group2, (cov1, cov2) = generate_fc_matrices(100,
                                                            effect_size,
                                                            n_samples_group1=50,
                                                            n_samples_group2=40,
                                                            seed=42)
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

        t_stat_dict = compute_t_stat_ind(g1, g2)
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