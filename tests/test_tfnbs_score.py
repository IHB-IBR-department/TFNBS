import unittest
from unittest import TestCase
import numpy as np
from tfnbs.pairwise_stats import compute_t_stat, compute_t_stat_diff
from tfnbs.tfnbs_score import get_tfnbs_score_networkx, get_tfnbs_score, get_tfnbs_score_baseline
from tfnbs.datasets import generate_fc_matrices
from tfnbs.utils import fisher_r_to_z
import time
from tfnbs.utils import create_prior_weights


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

    def test_weighted_tfnbs_increases_scores(self):
        """
        Create a small synthetic t_stats and a prior weight_map boosting two
        networks. Verify that using the weight_map increases the overall
        TFNBS score compared to the unweighted variant and preserves symmetry.
        """
        t = np.zeros((6, 6), dtype=float)
        t[0, 1] = t[1, 0] = 3.0
        t[0, 2] = t[2, 0] = 2.8
        t[1, 2] = t[2, 1] = 2.6
        t[3, 4] = t[4, 3] = 2.5
        np.fill_diagonal(t, 0.0)

        labels = np.array([1, 1, 1, 2, 2, 3])
        weight_map = create_prior_weights(labels, boost_factor=3.0)

        E = 0.5
        H = 2.0
        n = 20
        start_thres = 1.65

        unweighted = get_tfnbs_score(t, E, H, n, start_thres=start_thres)
        weighted = get_tfnbs_score(t, E, H, n, start_thres=start_thres, weight_map=weight_map)

        self.assertEqual(unweighted.shape, weighted.shape)
        self.assertTrue(np.allclose(unweighted, unweighted.T))
        self.assertTrue(np.allclose(weighted, weighted.T))
        self.assertGreaterEqual(np.sum(weighted), np.sum(unweighted))
        self.assertTrue(np.any(weighted > unweighted + 1e-12))

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