import os
from unittest import TestCase
from conninfpy.utils import get_components, create_prior_weights, fisher_r_to_z
from conninfpy.synth_datasets import generate_fc_matrices
from conninfpy.pairwise_stats import compute_t_stat
import numpy as np

SHOW_PLOTS = os.getenv("CONNINFPY_TEST_PLOTS") == "1"


class Test(TestCase):

    @classmethod
    def setUpClass(cls):
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

    def test_get_components(self):
        t_stats = self.fc_sim_30["t_stat"]['g2>g1']

        adj = t_stats >= 2.47
        adj_mod = adj.copy()
        a, sz = get_components(adj_mod)
        ind_sz, = np.where(sz > 1)
        ind_sz += 1
        nr_components = np.size(ind_sz)
        sz_links = np.zeros((nr_components,))
        adj_mod = 1. * adj.copy()

        for i in range(nr_components):
            nodes, = np.where(ind_sz[i] == a)
            sz_links[i] = np.sum(adj[np.ix_(nodes, nodes)]) / 2
            adj_mod[np.ix_(nodes, nodes)] *= (i + 2)

        # subtract 1 to delete any edges not comprising a component
        adj_mod[np.where(adj_mod)] -= 1

        if SHOW_PLOTS:
            import matplotlib.pyplot as plt
            plt.imshow(adj_mod)
            plt.show()

        self.assertTrue(True)


class TestCreatePriorWeights(TestCase):
    def test_basic_boosting(self):

        labels = np.array([1, 1, 2, 2, 3])
        W = create_prior_weights(labels, boost_factor=3.0)

        # shape and diagonal
        self.assertEqual(W.shape, (5, 5))
        self.assertTrue(np.allclose(np.diag(W), 1.0))

        # intra-network pairs boosted
        self.assertEqual(W[0, 1], 3.0)
        self.assertEqual(W[1, 0], 3.0)
        self.assertEqual(W[2, 3], 3.0)

        # inter-network pairs remain background
        self.assertEqual(W[0, 2], 1.0)

    def test_target_network_only(self):

        labels = np.array([1, 1, 2, 2, 3])
        W = create_prior_weights(labels, target_network_id=2, boost_factor=4.0)

        # only nodes in network 2 (indices 2 and 3) are boosted
        self.assertEqual(W[2, 3], 4.0)
        self.assertEqual(W[0, 1], 1.0)
        self.assertEqual(W[0, 2], 1.0)
