import unittest
from unittest import TestCase

import numpy as np

from conninfpy.pairwise_stats import compute_t_stat
from conninfpy.nbs_score import get_nbs_score, nbs_bct
from conninfpy.utils import fisher_r_to_z

from . import fixtures


class TestNBSScore(TestCase):
    """Tests for the low-level get_nbs_score (classical NBS cluster scoring)."""

    @classmethod
    def setUpClass(cls):
        cls.small_matrix = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float)
        cls.invalid_matrix = np.array([[1, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float)
        cls.asymmetric_matrix = np.array([[0, 2, 0], [0, 0, 4], [1, 0, 0]], dtype=float)

        # Scenario: small_two_sample — N=30, effect=0.2
        group1, group2, _ = fixtures.small_two_sample()
        t_stat_30 = compute_t_stat(
            fisher_r_to_z(group1),
            fisher_r_to_z(group2),
            test_type="two-sample",
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


class TestNBSBCT(TestCase):
    """Tests for the high-level nbs_bct pipeline (permutation-based NBS p-values)."""

    @classmethod
    def setUpClass(cls):
        # Scenario: small_two_sample — N=30, moderate effect. Structural tests
        # below don't require strong signal; they check shapes and invariants.
        g1, g2, _ = fixtures.small_two_sample()
        cls.group1 = fisher_r_to_z(g1)
        cls.group2 = fisher_r_to_z(g2)

    def test_output_shapes_and_keys(self):
        """Check output structure and shape of nbs_bct."""
        p_vals, adj, null = nbs_bct(
            self.group1, self.group2,
            threshold=2.1, n_permutations=100,
            test_type='two-sample', random_state=2, use_mp=False,
        )

        for key in ("g1>g2", "g2>g1"):
            self.assertIn(key, p_vals)
            self.assertIn(key, adj)
            self.assertIn(key, null)

        N = self.group1.shape[1]
        self.assertEqual(p_vals["g1>g2"].shape, (N, N))
        self.assertEqual(adj["g1>g2"].shape, (N, N))
        self.assertIn(null["g1>g2"].ndim, [1, 2])

    def test_symmetry_of_adjacency(self):
        """Adjacency matrix should be symmetric."""
        _, adj, _ = nbs_bct(
            self.group1, self.group2,
            threshold=2.0, n_permutations=50,
            test_type='two-sample', random_state=123, use_mp=False,
        )
        for key in adj:
            np.testing.assert_array_equal(adj[key], adj[key].T)

    def test_paired_behavior(self):
        """Paired test_type yields valid output for matched subjects."""
        n = min(self.group1.shape[0], self.group2.shape[0])
        g1 = self.group1[:n]
        g2 = self.group2[:n]
        p_vals, _, _ = nbs_bct(
            g1, g2,
            threshold=1.5, n_permutations=10,
            test_type='paired', use_mp=False,
        )
        self.assertIsInstance(p_vals, dict)
        self.assertIn("g1>g2", p_vals)


if __name__ == "__main__":
    unittest.main()
