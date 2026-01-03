from unittest import TestCase
from tfnbs.utils import binarize, get_components
from tfnbs.utils import create_prior_weights
from tfnbs.synth_datasets import generate_fc_matrices
from tfnbs.utils import fisher_r_to_z
from tfnbs.pairwise_stats import compute_t_stat, compute_t_stat_diff
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Get path to datasets directory relative to this file
DATASETS_DIR = Path(__file__).parent.parent / 'datasets' / '02_BLOCK_VAR_HRF_SNR05_CORRDIFF'


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

        from scipy.io import loadmat
        taskB = loadmat(DATASETS_DIR / 'Task_B.mat')['corrdiff_TaskB']
        taskA = loadmat(DATASETS_DIR / 'Task_A.mat')['corrdiff_TaskA']
        taskB = fisher_r_to_z(np.nan_to_num(taskB, posinf=0, neginf=0))
        taskA = fisher_r_to_z(np.nan_to_num(taskA, posinf=0, neginf=0))
        cls.taskA = taskA
        cls.taskB = taskB
        cls.t_stat = compute_t_stat_diff(taskA - taskB)

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

        self.assertTrue(True)


class TestReal(TestCase):

    @classmethod
    def setUpClass(cls):
        from scipy.io import loadmat
        taskB = loadmat(DATASETS_DIR / 'Task_B.mat')['corrdiff_TaskB']
        taskA = loadmat(DATASETS_DIR / 'Task_A.mat')['corrdiff_TaskA']
        taskB = fisher_r_to_z(np.nan_to_num(taskB, posinf=0, neginf=0)).swapaxes(0, 2)
        taskA = fisher_r_to_z(np.nan_to_num(taskA, posinf=0, neginf=0)).swapaxes(0, 2)
        cls.taskA = taskA
        cls.taskB = taskB
        cls.t_stat = compute_t_stat_diff(taskA - taskB)

    def test_get_components(self):
        t_stats = self.t_stat['g2>g1']
        np.allclose(t_stats, t_stats.T, atol=1e-8)
        adj = t_stats >= 1.75
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
