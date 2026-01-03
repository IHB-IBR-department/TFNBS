import unittest
from unittest import TestCase

import numpy as np

from tfnbs.synth_datasets import generate_fc_matrices
from tfnbs.pairwise_stats import compute_t_stat
from tfnbs.nbs_score import get_nbs_score
from tfnbs.utils import fisher_r_to_z


class TestNBSScore(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.small_matrix = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float)
        cls.invalid_matrix = np.array([[1, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float)
        cls.asymmetric_matrix = np.array([[0, 2, 0], [0, 0, 4], [1, 0, 0]], dtype=float)

        effect_size = 0.2
        group1, group2, _ = generate_fc_matrices(
            30,
            effect_size,
            n_samples_group1=30,
            n_samples_group2=20,
            seed=42
        )
        t_stat_30 = compute_t_stat(
            fisher_r_to_z(group1),
            fisher_r_to_z(group2),
            test_type="two-sample"
        )

        cls.fc_sim_30 = {"t_stat": t_stat_30}

    def test_small_matrix_extent(self):
        """Extent = number of edges in the component."""
        statsmat = self.small_matrix
        result = get_nbs_score(statsmat, threshold=0.5, stat_type="extent")
        expected = np.array([[0.0, 3.0, 3.0], [3.0, 0.0, 3.0], [3.0, 3.0, 0.0]])
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_small_matrix_intensity(self):
        """Intensity = sum of t-values in the component."""
        statsmat = self.small_matrix
        result = get_nbs_score(statsmat, threshold=0.5, stat_type="intensity")
        expected = np.array([[0.0, 4.0, 4.0], [4.0, 0.0, 4.0], [4.0, 4.0, 0.0]])
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_asymmetric_matrix_extent(self):
        statsmat = self.asymmetric_matrix
        result = get_nbs_score(statsmat, threshold=0.5, stat_type="extent")
        expected = np.array([[0.0, 3.0, 0.0], [0.0, 0.0, 3.0], [3.0, 0.0, 0.0]])
        np.testing.assert_allclose(result, expected, rtol=1e-5)
        self.assertFalse(np.allclose(result, result.T))

    def test_invalid_matrix_diag(self):
        """Non-zero diagonal should raise."""
        with self.assertRaises(ValueError):
            get_nbs_score(self.invalid_matrix, threshold=0.5, stat_type="extent")

    def test_below_threshold_returns_zeros(self):
        statsmat = self.small_matrix
        result = get_nbs_score(statsmat, threshold=3.0, stat_type="extent")
        np.testing.assert_allclose(result, 0.0)

    def test_stat_type_invalid(self):
        with self.assertRaises(ValueError):
            get_nbs_score(self.small_matrix, threshold=0.5, stat_type="bad")

    def test_real_matrix_30N(self):
        t_stat = self.fc_sim_30["t_stat"]
        score_pos = get_nbs_score(t_stat["g2>g1"], threshold=1.7, stat_type="extent")
        score_neg = get_nbs_score(t_stat["g1>g2"], threshold=1.7, stat_type="extent")

        self.assertEqual(score_pos.shape, (30, 30))
        self.assertEqual(score_neg.shape, (30, 30))
        self.assertTrue((score_pos >= 0).all())
        self.assertTrue((score_neg >= 0).all())
        np.testing.assert_allclose(score_pos, score_pos.T, rtol=1e-10)
        np.testing.assert_allclose(score_neg, score_neg.T, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
