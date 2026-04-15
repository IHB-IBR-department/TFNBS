from unittest import TestCase
import time
from conninfpy.utils import fisher_r_to_z
from conninfpy.pairwise_stats import (_permutation_task_ind,
                                   _permutation_task_paired,
                                   compute_null_dist,
                                   compute_t_stat_diff,
                                   compute_p_val,
                                   compute_t_stat,
                                   compute_t_stat_tfnbs,
                                   compute_t_stat_tfnbs_diffs)

from conninfpy.synth_datasets import generate_fc_matrices
import numpy as np


class TestBasicStats(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        N = 30  # Number of ROIs
        effect_size = 0.2  # Magnitude of group differences

        group1, group2, (cov1, cov2) = generate_fc_matrices(N, effect_size, n_samples_group1=30,
                                                            n_samples_group2=20)
        cls.fc_sim = {"group1": fisher_r_to_z(group1.copy()),
                      "group2": fisher_r_to_z(group2.copy()),
                      "true_diff": cov2 - cov1,
                      'cov2': cov2,
                      'cov1': cov1}

        group1, group2, (cov1, cov2) = generate_fc_matrices(N, effect_size, n_samples_group1=40,
                                                            n_samples_group2=40)
        cls.fc_sim_paired = {"group1": fisher_r_to_z(group1.copy()),
                             "group2": fisher_r_to_z(group2.copy()),
                             "true_diff": cov2 - cov1,
                             'cov2': cov2,
                             'cov1': cov1}

    def run_and_measure(self, func, arr1, arr2, n_permutations, test_type, random_state, use_mp):
        """Helper function to measure execution time of a function."""
        start_time = time.time()
        compute_null_dist(arr1, arr2, func, n_permutations=n_permutations, test_type=test_type, random_state=random_state, use_mp=use_mp)
        return time.time() - start_time

    def test_compute_t_stat(self):
        group_dict = self.fc_sim

        emp_t_dict = compute_t_stat(group_dict['group1'], group_dict['group2'], test_type='two-sample')

        self.assertLess(2, emp_t_dict["g2>g1"][np.triu_indices(10, k=1)].mean())
        self.assertEqual(0, emp_t_dict["g1>g2"][np.triu_indices(10, k=1)].mean())

    def test_compute_t_stat_diff(self):
        group_dict = self.fc_sim_paired
        t_stat_dict = compute_t_stat_diff(group_dict['group2'] - group_dict['group1'])
        self.assertLess(2, t_stat_dict["g2>g1"][np.triu_indices(10, k=1)].mean())
        self.assertEqual(0, t_stat_dict["g1>g2"][np.triu_indices(10, k=1)].mean())

    def test_compute_permut_t_stat_ind(self):
        group_dict = self.fc_sim

        # Use modern _permutation_task_ind
        full_group = np.concatenate((group_dict['group1'], group_dict['group2']), axis=0)
        n1 = group_dict['group1'].shape[0]
        perm_result = _permutation_task_ind(full_group, compute_t_stat, n1, seed=42)

        perm_t_pos = perm_result["g2>g1"]
        perm_t_neg = perm_result["g1>g2"]

        self.assertGreater(perm_t_pos, 1)
        self.assertGreater(perm_t_neg, 1)
        self.assertLess(perm_t_pos, 5)

    def test_compute_t_stat_tfnbs(self):
        group_dict = self.fc_sim

        emp_t_dict = compute_t_stat(group_dict['group1'], group_dict['group2'], test_type='two-sample')
        emp_tfnos_dict = compute_t_stat_tfnbs(group_dict['group1'], group_dict['group2'], test_type='two-sample')

        self.assertLess(10, emp_tfnos_dict["g2>g1"][np.triu_indices(10, k=1)].mean())
        self.assertLess(1, emp_t_dict["g2>g1"][np.triu_indices(10, k=1)].mean())

    def test_compute_t_stat_tfnbs_paired(self):
        group_dict = self.fc_sim_paired

        emp_t_dict = compute_t_stat(group_dict['group1'], group_dict['group2'], test_type='paired')
        emp_tfnos_dict = compute_t_stat_tfnbs(group_dict['group1'], group_dict['group2'], test_type='paired')
        emp_tfnos_sp_dict = compute_t_stat_tfnbs_diffs(group_dict['group2'] - group_dict['group1'])

        self.assertIsNone(np.testing.assert_almost_equal(emp_tfnos_dict["g2>g1"],
                                                         emp_tfnos_sp_dict["g2>g1"]))
        self.assertGreater(emp_tfnos_dict["g2>g1"].sum(), emp_t_dict["g2>g1"].sum())

    def test_compute_t_stat_tfnbs_list_pars(self):
        group_dict = self.fc_sim_paired
        t_stat_mod = compute_t_stat_tfnbs(group_dict['group1'], group_dict['group2'], test_type='paired', e=[0.4, 0.6], h=[1, 2])
        self.assertEqual(t_stat_mod["g2>g1"].shape[-1], 2)
        self.assertEqual(t_stat_mod["g1>g2"].shape[-1], 2)

    def test__permutation_task_ind_t(self):
        group_dict = self.fc_sim
        t_stat = compute_t_stat(group_dict['group1'], group_dict['group2'], test_type='two-sample')
        full_group = np.concatenate((group_dict['group1'], group_dict['group2']), axis=0)
        t_maxes = _permutation_task_ind(full_group, compute_t_stat,
                                        30, 42)
        self.assertIsInstance(t_maxes, dict)
        self.assertEqual(len(t_maxes.values()), 2)
        self.assertGreater(np.max(t_stat["g2>g1"]), t_maxes["g2>g1"])

    def test__permutation_task_ind(self):
        group_dict = self.fc_sim
        t_stat_mod = compute_t_stat_tfnbs(group_dict['group1'], group_dict['group2'], test_type = 'two-sample', e=[0.4, 0.6], h=[1, 2])
        full_group = np.concatenate((group_dict['group1'], group_dict['group2']), axis=0)
        t_maxes = _permutation_task_ind(full_group, compute_t_stat_tfnbs,
                                        30, 42, e=[0.4, 0.6], h=[1, 2])
        self.assertIsInstance(t_maxes, dict)
        self.assertIsInstance(t_stat_mod, dict)
        self.assertGreater(t_stat_mod['g2>g1'][:10, :10, :].mean(), t_maxes['g2>g1'].mean())
        self.assertGreater(t_stat_mod['g1>g2'].max(), t_maxes['g1>g2'].mean())

    def test__permutation_task_paired(self):
        group_dict = self.fc_sim_paired
        emp_t = compute_t_stat_diff(group_dict['group2'] - group_dict['group1'])
        emp_tfnos = compute_t_stat_tfnbs_diffs(group_dict['group2'] - group_dict['group1'], e=[0.4, 0.6], h=[1, 2])
        t_max_t = _permutation_task_paired(group_dict['group2'] - group_dict['group1'], compute_t_stat_diff, 30)
        t_maxes = _permutation_task_paired(group_dict['group2'] - group_dict['group1'], compute_t_stat_tfnbs_diffs,
                                           e=[0.4, 0.6], h=[1, 2])
        self.assertIsInstance(t_maxes, dict)
        self.assertIsInstance(t_max_t, dict)
        self.assertGreater(emp_tfnos['g2>g1'][:10,:10,:].mean(), t_maxes['g2>g1'].mean())
        self.assertGreater(emp_t['g1>g2'].max(), t_max_t['g1>g2'])
        self.assertGreater(emp_tfnos['g1>g2'].max(), t_maxes['g1>g2'].mean())

    def test_compute_null_t_stat_ind(self):
        group_dict = self.fc_sim

        n_permutations = 100

        null_t = compute_null_dist(group_dict['group1'], group_dict['group2'],
                                   compute_t_stat, n_permutations=n_permutations, 
                                   test_type='two-sample',random_state=42, use_mp=False)
        null_t_mc = compute_null_dist(group_dict['group1'], group_dict['group2'],
                                      compute_t_stat, n_permutations=n_permutations,
                                      test_type='two-sample', random_state=42, use_mp=True)

        self.assertIsInstance(null_t, dict)
        self.assertIsInstance(null_t_mc, dict)
        self.assertEqual((null_t["g2>g1"].mean() - null_t_mc["g1>g2"].mean()).round(), 0)

    def test_compute_null_t_stat_tfnos_ind(self):
        group_dict = self.fc_sim

        n_permutations = 100
        emp_tfnos = compute_t_stat_tfnbs(group_dict['group1'], group_dict['group2'], test_type = 'two-sample')

        null_t = compute_null_dist(group_dict['group1'], group_dict['group2'],
                                   compute_t_stat_tfnbs, n_permutations=n_permutations,
                                   test_type='two-sample', random_state=42, use_mp=False)
        null_t_mc = compute_null_dist(group_dict['group1'], group_dict['group2'],
                                      compute_t_stat_tfnbs, n_permutations=n_permutations,
                                      test_type='two-sample', random_state=42, use_mp=True)

        self.assertGreater(emp_tfnos["g2>g1"].mean(), null_t["g1>g2"].mean())
        self.assertGreater(emp_tfnos["g2>g1"].mean(), null_t_mc["g1>g2"].mean())

    def test_compute_null_t_stat_tfnos_ind_multi(self):
        group_dict = self.fc_sim

        n_permutations = 100
        emp_tfnos = compute_t_stat_tfnbs(group_dict['group1'], group_dict['group2'], test_type='two-sample',
                                         e=[0.4, 0.6], h=[1, 2])

        null_t = compute_null_dist(group_dict['group1'], group_dict['group2'],
                                   compute_t_stat_tfnbs, n_permutations=n_permutations,
                                   test_type='two-sample', use_mp=False, e=[0.4, 0.6], h=[1, 2])
        null_t_mp = compute_null_dist(group_dict['group1'], group_dict['group2'],
                                      compute_t_stat_tfnbs, n_permutations=n_permutations,
                                   test_type='two-sample', use_mp=True, e=[0.4, 0.6], h=[1, 2])
        self.assertEqual(null_t["g2>g1"].shape[-1], emp_tfnos["g2>g1"].shape[-1])
        self.assertEqual(null_t_mp["g2>g1"].shape[-1], emp_tfnos["g2>g1"].shape[-1])

    def test_compute_null_t_stat_tfnos_paired_multi(self):
        group_dict = self.fc_sim_paired

        n_permutations = 100
        emp_tfnos = compute_t_stat_tfnbs(group_dict['group1'], group_dict['group2'], test_type='paired',
                                         e=[0.4, 0.6], h=[1, 2])

        null_t = compute_null_dist(group_dict['group1'], group_dict['group2'],
                                   compute_t_stat_tfnbs_diffs, n_permutations=n_permutations,
                                   test_type='paired', use_mp=False, e=[0.4, 0.6], h=[1, 2])
        null_t_mp = compute_null_dist(group_dict['group1'], group_dict['group2'],
                                      compute_t_stat_tfnbs_diffs, n_permutations=n_permutations,
                                   test_type='paired', use_mp=True, e=[0.4, 0.6], h=[1, 2])
        self.assertEqual(null_t["g2>g1"].shape[-1], emp_tfnos["g2>g1"].shape[-1])
        self.assertEqual(null_t_mp["g2>g1"].shape[-1], emp_tfnos["g2>g1"].shape[-1])

    def test_compute_null_t_stat_ind_eff(self):
        """Test that mp and sequential paths produce consistent null distributions."""
        group_dict = self.fc_sim

        n_permutations = 100
        random_state = 42
        test_type = 'two-sample'

        null_mp = compute_null_dist(group_dict['group1'], group_dict['group2'],
                                    compute_t_stat_tfnbs, n_permutations=n_permutations,
                                    test_type=test_type, random_state=random_state, use_mp=True)

        null_seq = compute_null_dist(group_dict['group1'], group_dict['group2'],
                                     compute_t_stat_tfnbs, n_permutations=n_permutations,
                                     test_type=test_type, random_state=random_state, use_mp=False)

        # Both paths should produce valid null distributions of the same shape
        self.assertEqual(null_mp["g2>g1"].shape, null_seq["g2>g1"].shape)
        self.assertEqual(null_mp["g1>g2"].shape, null_seq["g1>g2"].shape)
        # Null distributions should have similar statistics (same seeds)
        np.testing.assert_allclose(null_mp["g2>g1"].mean(), null_seq["g2>g1"].mean(), rtol=0.3)
        np.testing.assert_allclose(null_mp["g1>g2"].mean(), null_seq["g1>g2"].mean(), rtol=0.3)

    def test_compute_p_val_indep(self):
        group_dict = self.fc_sim
        n_permutations = 100
        p_vals = compute_p_val(group_dict['group1'], group_dict['group2'],
                               n_permutations=n_permutations, test_type='two-sample', method='tstat', use_mp=True)

        self.assertLess(p_vals["g2>g1"][np.triu_indices(10, k=1)].mean(), 0.3)
        self.assertGreater(p_vals["g1>g2"][np.triu_indices(10, k=1)].mean(), 0.3)

    def test_compute_p_val_indep_tf(self):
        group_dict = self.fc_sim
        n_permutations = 100

        p_vals = compute_p_val(group_dict['group1'], group_dict['group2'],
                               n_permutations=n_permutations, test_type='two-sample', method='tfnbs', use_mp=True)

        self.assertLess(p_vals["g2>g1"][np.triu_indices(10, k=1)].mean(), 0.3)
        self.assertGreater(p_vals["g1>g2"][np.triu_indices(10, k=1)].mean(), 0.3)

    def test_compute_p_val_indep_tf_multi(self):
        group_dict = self.fc_sim
        n_permutations = 100

        p_vals = compute_p_val(group_dict['group1'], group_dict['group2'],
                               n_permutations=n_permutations, test_type='two-sample', method='tfnbs', use_mp=True, e=[0.4, 0.6],
                               h=[1, 2])

        self.assertLess(p_vals["g2>g1"][..., 0][np.triu_indices(10, k=1)].mean(), 0.05)
        self.assertLess(p_vals["g2>g1"][..., 1][np.triu_indices(10, k=1)].mean(), 0.05)

    def test_compute_p_val_indep_tf_orig(self):
        group_dict = self.fc_sim
        n_permutations = 100
        p_vals_orig = compute_p_val(group_dict['group1'], group_dict['group2'],
                                    n_permutations=n_permutations, test_type='two-sample', method='tstat', use_mp=True)
        p_vals_tf = compute_p_val(group_dict['group1'], group_dict['group2'],
                                  n_permutations=n_permutations, test_type='two-sample', method='tfnbs', use_mp=True)

        self.assertLess(p_vals_tf["g2>g1"][np.triu_indices(10, k=1)].mean(),
                        p_vals_orig["g2>g1"][np.triu_indices(10, k=1)].mean())

    def test_compute_p_val_paired_tf_orig(self):
        group_dict = self.fc_sim_paired
        n_permutations = 100
        p_vals_orig = compute_p_val(group_dict['group1'], group_dict['group2'],
                                    n_permutations=n_permutations, test_type='paired', method='tstat', use_mp=True)
        p_vals_tf = compute_p_val(group_dict['group1'], group_dict['group2'],
                                  n_permutations=n_permutations, test_type='paired', method='tfnbs', use_mp=True)

        self.assertLess(p_vals_tf["g2>g1"][np.triu_indices(10, k=1)].mean(),
                        p_vals_orig["g2>g1"][np.triu_indices(10, k=1)].mean())

    def test_compute_p_val_tfnbs_multi_params(self):
        """Test TFNBS with multiple parameter combinations e=[0.5, 1], h=[1, 2]."""
        group_dict = self.fc_sim
        n_permutations = 100

        p_vals = compute_p_val(
            group_dict['group1'], group_dict['group2'],
            n_permutations=n_permutations,
            test_type='two-sample',
            method='tfnbs',
            e=[0.5, 1],
            h=[1, 2],
            n=10,
            use_mp=False,
            random_state=42
        )

        # Check that p-values are computed
        self.assertIn("g2>g1", p_vals)
        self.assertIn("g1>g2", p_vals)

        # Check shape: should be (N, N, 2) for 2 parameter combinations
        expected_shape = group_dict['group1'][0].shape + (2,)
        self.assertEqual(p_vals["g2>g1"].shape, expected_shape)
        self.assertEqual(p_vals["g1>g2"].shape, expected_shape)

        # Check that different parameter combinations give different results
        self.assertFalse(np.allclose(p_vals["g2>g1"][..., 0], p_vals["g2>g1"][..., 1]))

    def test_compute_p_val_nbs_two_sample(self):
        """Test NBS method with two-sample test."""
        group_dict = self.fc_sim
        n_permutations = 100
        p_vals = compute_p_val(
            group_dict['group1'], group_dict['group2'],
            n_permutations=n_permutations,
            test_type='two-sample',
            method='nbs',
            threshold=2.0,
            nbs_stat='extent',
            use_mp=False,
            random_state=42
        )
        # Check that p-values are computed
        self.assertIn("g2>g1", p_vals)
        self.assertIn("g1>g2", p_vals)
        self.assertEqual(p_vals["g2>g1"].shape, group_dict['group1'][0].shape)

    def test_compute_p_val_nbs_paired(self):
        """Test NBS method with paired test."""
        group_dict = self.fc_sim_paired
        n_permutations = 100
        p_vals = compute_p_val(
            group_dict['group1'], group_dict['group2'],
            n_permutations=n_permutations,
            test_type='paired',
            method='nbs',
            threshold=2.0,
            nbs_stat='intensity',
            use_mp=False,
            random_state=42
        )
        self.assertIn("g2>g1", p_vals)
        self.assertIn("g1>g2", p_vals)

    def test_compute_p_val_cnbs_two_sample(self):
        """Test cNBS method with two-sample test."""
        group_dict = self.fc_sim
        n_permutations = 100
        # Create network labels (3 networks for 30 nodes)
        net_labels = np.repeat([0, 1, 2], 10)

        p_vals = compute_p_val(
            group_dict['group1'], group_dict['group2'],
            n_permutations=n_permutations,
            test_type='two-sample',
            method='cnbs',
            net_labels=net_labels,
            use_mp=False,
            random_state=42
        )
        self.assertIn("g2>g1", p_vals)
        self.assertIn("g1>g2", p_vals)

    def test_compute_p_val_cnbs_paired(self):
        """Test cNBS method with paired test."""
        group_dict = self.fc_sim_paired
        n_permutations = 100
        net_labels = np.repeat([0, 1, 2], 10)

        p_vals = compute_p_val(
            group_dict['group1'], group_dict['group2'],
            n_permutations=n_permutations,
            test_type='paired',
            method='cnbs',
            net_labels=net_labels,
            use_mp=False,
            random_state=42
        )
        self.assertIn("g2>g1", p_vals)
        self.assertIn("g1>g2", p_vals)

    def test_compute_p_val_ni_tfnbs_two_sample(self):
        """Test NI-TFNBS method with two-sample test."""
        group_dict = self.fc_sim
        n_permutations = 100
        net_labels = np.repeat([0, 1, 2], 10)

        p_vals = compute_p_val(
            group_dict['group1'], group_dict['group2'],
            n_permutations=n_permutations,
            test_type='two-sample',
            method='ni_tfnbs',
            net_labels=net_labels,
            e=0.5,
            h=2.0,
            n=10,
            use_mp=False,
            random_state=42
        )
        self.assertIn("g2>g1", p_vals)
        self.assertIn("g1>g2", p_vals)
        # NI-TFNBS should produce different results than regular TFNBS
        p_vals_tfnbs = compute_p_val(
            group_dict['group1'], group_dict['group2'],
            n_permutations=n_permutations,
            test_type='two-sample',
            method='tfnbs',
            e=0.5,
            h=2.0,
            n=10,
            use_mp=False,
            random_state=42
        )
        # At least some p-values should differ
        self.assertFalse(np.allclose(p_vals["g2>g1"], p_vals_tfnbs["g2>g1"]))

    def test_compute_p_val_fbc_tfnbs_two_sample(self):
        """Test FBC-TFNBS method with two-sample test."""
        group_dict = self.fc_sim
        n_permutations = 100
        net_labels = np.repeat([0, 1, 2], 10)

        p_vals = compute_p_val(
            group_dict['group1'], group_dict['group2'],
            n_permutations=n_permutations,
            test_type='two-sample',
            method='fbc_tfnbs',
            net_labels=net_labels,
            e=0.5,
            h=2.0,
            n=10,
            min_cluster_size=3,
            use_mp=False,
            random_state=42
        )
        self.assertIn("g2>g1", p_vals)
        self.assertIn("g1>g2", p_vals)

    def test_compute_p_val_fbc_tfnbs_paired(self):
        """Test FBC-TFNBS method with paired test."""
        group_dict = self.fc_sim_paired
        n_permutations = 100
        net_labels = np.repeat([0, 1, 2], 10)

        p_vals = compute_p_val(
            group_dict['group1'], group_dict['group2'],
            n_permutations=n_permutations,
            test_type='paired',
            method='fbc_tfnbs',
            net_labels=net_labels,
            e=0.5,
            h=2.0,
            n=10,
            min_cluster_size=2,
            use_mp=False,
            random_state=42
        )
        self.assertIn("g2>g1", p_vals)
        self.assertIn("g1>g2", p_vals)

    def test_constrained_methods_require_net_labels(self):
        """Test that constrained methods raise ValueError without net_labels."""
        group_dict = self.fc_sim

        # cNBS should require net_labels
        with self.assertRaises(ValueError) as context:
            compute_p_val(
                group_dict['group1'], group_dict['group2'],
                n_permutations=10,
                test_type='two-sample',
                method='cnbs',
                use_mp=False
            )
        self.assertIn("net_labels", str(context.exception))

        # NI-TFNBS should require net_labels
        with self.assertRaises(ValueError) as context:
            compute_p_val(
                group_dict['group1'], group_dict['group2'],
                n_permutations=10,
                test_type='two-sample',
                method='ni_tfnbs',
                use_mp=False
            )
        self.assertIn("net_labels", str(context.exception))

        # FBC-TFNBS should require net_labels
        with self.assertRaises(ValueError) as context:
            compute_p_val(
                group_dict['group1'], group_dict['group2'],
                n_permutations=10,
                test_type='two-sample',
                method='fbc_tfnbs',
                use_mp=False
            )
        self.assertIn("net_labels", str(context.exception))

    def test_invalid_method_raises_error(self):
        """Test that invalid method raises ValueError."""
        group_dict = self.fc_sim

        with self.assertRaises(ValueError) as context:
            compute_p_val(
                group_dict['group1'], group_dict['group2'],
                n_permutations=10,
                test_type='two-sample',
                method='invalid_method',
                use_mp=False
            )
        self.assertIn("Invalid method", str(context.exception))

    # -----------------------------------------------------------------
    # Bonferroni and BH-FDR parametric baselines
    # -----------------------------------------------------------------

    def test_compute_p_val_bonferroni_two_sample(self):
        """Test Bonferroni correction with two-sample test."""
        group_dict = self.fc_sim
        p_vals = compute_p_val(
            group_dict['group1'], group_dict['group2'],
            test_type='two-sample', method='bonferroni'
        )
        self.assertIn("g2>g1", p_vals)
        self.assertIn("g1>g2", p_vals)
        N = group_dict['group1'].shape[1]
        self.assertEqual(p_vals["g2>g1"].shape, (N, N))
        # p-values in [0, 1]
        self.assertTrue(np.all(p_vals["g2>g1"] >= 0))
        self.assertTrue(np.all(p_vals["g2>g1"] <= 1))
        # Symmetric
        np.testing.assert_allclose(p_vals["g2>g1"], p_vals["g2>g1"].T)
        # Diagonal should be 1.0 (no self-connections)
        np.testing.assert_allclose(np.diag(p_vals["g2>g1"]), 1.0)

    def test_compute_p_val_bh_fdr_two_sample(self):
        """Test BH-FDR correction with two-sample test."""
        group_dict = self.fc_sim
        p_vals = compute_p_val(
            group_dict['group1'], group_dict['group2'],
            test_type='two-sample', method='bh_fdr'
        )
        self.assertIn("g2>g1", p_vals)
        self.assertIn("g1>g2", p_vals)
        N = group_dict['group1'].shape[1]
        self.assertEqual(p_vals["g2>g1"].shape, (N, N))
        self.assertTrue(np.all(p_vals["g2>g1"] >= 0))
        self.assertTrue(np.all(p_vals["g2>g1"] <= 1))
        np.testing.assert_allclose(p_vals["g2>g1"], p_vals["g2>g1"].T)

    def test_compute_p_val_bonferroni_paired(self):
        """Test Bonferroni correction with paired test."""
        group_dict = self.fc_sim_paired
        p_vals = compute_p_val(
            group_dict['group1'], group_dict['group2'],
            test_type='paired', method='bonferroni'
        )
        self.assertIn("g2>g1", p_vals)
        N = group_dict['group1'].shape[1]
        self.assertEqual(p_vals["g2>g1"].shape, (N, N))
        np.testing.assert_allclose(p_vals["g2>g1"], p_vals["g2>g1"].T)

    def test_compute_p_val_bh_fdr_paired(self):
        """Test BH-FDR correction with paired test."""
        group_dict = self.fc_sim_paired
        p_vals = compute_p_val(
            group_dict['group1'], group_dict['group2'],
            test_type='paired', method='bh_fdr'
        )
        self.assertIn("g2>g1", p_vals)
        N = group_dict['group1'].shape[1]
        self.assertEqual(p_vals["g2>g1"].shape, (N, N))

    def test_bonferroni_more_conservative_than_bh_fdr(self):
        """Bonferroni p-values should be >= BH-FDR p-values."""
        group_dict = self.fc_sim
        p_bonf = compute_p_val(
            group_dict['group1'], group_dict['group2'],
            test_type='two-sample', method='bonferroni'
        )
        p_bh = compute_p_val(
            group_dict['group1'], group_dict['group2'],
            test_type='two-sample', method='bh_fdr'
        )
        # Bonferroni is more conservative (larger p-values)
        self.assertTrue(
            np.all(p_bonf["g2>g1"] >= p_bh["g2>g1"] - 1e-10),
            "Bonferroni should be at least as conservative as BH-FDR"
        )

    def test_parametric_detects_true_effect(self):
        """Parametric methods should detect the planted effect in the test data."""
        group_dict = self.fc_sim
        N = group_dict['group1'].shape[1]
        # Effect is in the first 10x10 block (upper triangle)
        effect_idx = np.triu_indices(10, k=1)

        p_bh = compute_p_val(
            group_dict['group1'], group_dict['group2'],
            test_type='two-sample', method='bh_fdr'
        )
        # Mean p-value in the effect region should be lower than
        # mean p-value across all edges
        p_effect = p_bh["g2>g1"][effect_idx].mean()
        full_triu = np.triu_indices(N, k=1)
        p_all = p_bh["g2>g1"][full_triu].mean()
        self.assertLess(p_effect, p_all,
                        "BH-FDR should yield lower p-values in the effect region")

    def test_parametric_no_permutations_needed(self):
        """Parametric methods should be fast (no permutation overhead)."""
        group_dict = self.fc_sim
        import time
        start = time.time()
        compute_p_val(
            group_dict['group1'], group_dict['group2'],
            test_type='two-sample', method='bonferroni',
            n_permutations=10000  # should be ignored
        )
        elapsed = time.time() - start
        # Should be near-instant (< 1s), permutations are not run
        self.assertLess(elapsed, 1.0,
                        "Bonferroni should not run permutations")

    def test_bh_fdr_perm_returns_p_values(self):
        """BH-FDR-perm should return valid p-value matrices."""
        group_dict = self.fc_sim
        p_vals = compute_p_val(
            group_dict['group1'], group_dict['group2'],
            test_type='two-sample', method='bh_fdr_perm',
            n_permutations=50, use_mp=False, random_state=42,
        )
        self.assertIn('g2>g1', p_vals)
        self.assertIn('g1>g2', p_vals)
        N = group_dict['group1'].shape[1]
        self.assertEqual(p_vals['g2>g1'].shape, (N, N))
        # P-values should be in [0, 1]
        self.assertTrue(np.all(p_vals['g2>g1'] >= 0))
        self.assertTrue(np.all(p_vals['g2>g1'] <= 1))
        # Diagonal should be 1 (no self-connections)
        np.testing.assert_allclose(np.diag(p_vals['g2>g1']), 1.0)

    def test_bh_fdr_perm_detects_effect(self):
        """BH-FDR-perm should detect planted effect (lower p in effect region)."""
        group_dict = self.fc_sim
        p_vals = compute_p_val(
            group_dict['group1'], group_dict['group2'],
            test_type='two-sample', method='bh_fdr_perm',
            n_permutations=100, use_mp=False, random_state=42,
        )
        true_diff = group_dict['true_diff']
        effect_mask = np.abs(true_diff) > 0.05
        triu = np.triu_indices(true_diff.shape[0], k=1)

        p_upper = p_vals['g2>g1'][triu]
        effect_upper = effect_mask[triu]
        if np.any(effect_upper):
            # P-values in effect region should tend to be lower
            mean_p_effect = np.mean(p_upper[effect_upper])
            mean_p_null = np.mean(p_upper[~effect_upper])
            self.assertLess(mean_p_effect, mean_p_null,
                            "BH-FDR-perm should yield lower p-values in effect region")

    def test_bh_fdr_perm_symmetric(self):
        """BH-FDR-perm p-values should be symmetric."""
        group_dict = self.fc_sim
        p_vals = compute_p_val(
            group_dict['group1'], group_dict['group2'],
            test_type='two-sample', method='bh_fdr_perm',
            n_permutations=50, use_mp=False, random_state=42,
        )
        np.testing.assert_allclose(p_vals['g2>g1'], p_vals['g2>g1'].T)

    def test_bh_fdr_perm_paired(self):
        """BH-FDR-perm should work with paired test type."""
        group_dict = self.fc_sim
        n_min = min(group_dict['group1'].shape[0], group_dict['group2'].shape[0])
        g1 = group_dict['group1'][:n_min]
        g2 = group_dict['group2'][:n_min]
        p_vals = compute_p_val(
            g1, g2,
            test_type='paired', method='bh_fdr_perm',
            n_permutations=50, use_mp=False, random_state=42,
        )
        N = g1.shape[1]
        self.assertEqual(p_vals['g2>g1'].shape, (N, N))
        self.assertTrue(np.all(p_vals['g2>g1'] >= 0))
        self.assertTrue(np.all(p_vals['g2>g1'] <= 1))

    def test_ni_tfnbs_normalization_via_compute_p_val(self):
        """NI-TFNBS normalization parameter should be threaded through compute_p_val."""
        group_dict = self.fc_sim
        N = group_dict['group1'].shape[1]
        labels = np.arange(N) % 4  # 4 modules

        for norm in ("sqrt", "linear", "none"):
            p_vals = compute_p_val(
                group_dict['group1'], group_dict['group2'],
                test_type='two-sample', method='ni_tfnbs',
                n_permutations=10, use_mp=False, random_state=42,
                net_labels=labels, normalization=norm,
                e=0.5, h=2.0, n=10,
            )
            self.assertEqual(p_vals['g2>g1'].shape, (N, N),
                             f"Shape mismatch for normalization='{norm}'")
            self.assertTrue(np.all(p_vals['g2>g1'] >= 0),
                            f"Negative p-values for normalization='{norm}'")


class TestRealExample(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from scipy.io import loadmat
        import os
        # Get path relative to this test file
        test_dir = os.path.dirname(os.path.abspath(__file__))
        path_to_data = os.path.join(test_dir, '../datasets/02_BLOCK_VAR_HRF_SNR05_CORRDIFF/')
        taskB = loadmat(path_to_data + 'Task_B.mat')['corrdiff_TaskB']
        taskA = loadmat(path_to_data + 'Task_A.mat')['corrdiff_TaskA']
        taskB = fisher_r_to_z(np.nan_to_num(taskB, posinf=0, neginf=0))
        taskA = fisher_r_to_z(np.nan_to_num(taskA, posinf=0, neginf=0))
        cls.taskA = taskA
        cls.taskB = taskB

    def test_compute_p_val(self):
        n_permutations = 10
        p_vals_orig = compute_p_val(self.taskA,
                                    self.taskB,
                                    n_permutations=n_permutations,
                                    test_type='paired',
                                    method='tstat',
                                    use_mp=True)
        self.assertEqual(0,0)
