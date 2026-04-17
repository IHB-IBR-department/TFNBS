import os
from unittest import TestCase
from conninfpy.synth_datasets import (generate_fc_matrices, ModularDatasetGenerator)
import numpy as np

SHOW_PLOTS = os.getenv("CONNINFPY_TEST_PLOTS") == "1"


class Test(TestCase):

    def test_generate_fc_matrices(self):
        N = 30  # Number of ROIs
        effect_size = 0.2  # Magnitude of group differences
        mask = np.zeros((N, N))
        mask[0:10, 0:10] = 1  # Introduce differences in a subnetwork
        mask[10:20, 10:20] = -1  # Ensure symmetry

        group1, group2, (cov1, cov2) = generate_fc_matrices(N, effect_size,  mask, n_samples_group1=50, n_samples_group2=70)

        if SHOW_PLOTS:
            import matplotlib.pyplot as plt
            diff = group2.mean(axis=0) - group1.mean(axis=0)
            plt.subplot(141); plt.imshow(mask)
            plt.subplot(142); plt.imshow(cov1)
            plt.subplot(143); plt.imshow(cov2 - cov1)
            plt.subplot(144); plt.imshow(diff)
            plt.show()

        self.assertEqual(group1.shape, (50, N, N))


class TestModularDatasetGenerator(TestCase):
    
    def setUp(self):
        """Set up a standard generator for testing."""
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

    def test_initialization_shapes(self):
        """Test if the base covariance and labels have correct shapes."""
        self.assertEqual(self.gen.base_cov.shape, (self.N, self.N))
        self.assertEqual(len(self.gen.labels), self.N)
        # Check if diagonal is 1.0
        np.testing.assert_allclose(np.diag(self.gen.base_cov), 1.0)

    def test_spd_property(self):
        """Test if the base covariance matrix is Symmetric Positive Definite."""
        cov = self.gen.base_cov
        # Check symmetry
        np.testing.assert_allclose(cov, cov.T, err_msg="Matrix is not symmetric")
        
        # Check Positive Definiteness (Eigenvalues > 0)
        eigvals = np.linalg.eigvalsh(cov)
        self.assertTrue(np.all(eigvals > 0), "Matrix is not Positive Definite")

    def test_modular_structure(self):
        """Test if intra-module correlations are higher than inter-module correlations."""
        cov = self.gen.base_cov
        labels = self.gen.labels
        
        intra_values = []
        inter_values = []
        
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if labels[i] == labels[j]:
                    intra_values.append(cov[i, j])
                else:
                    inter_values.append(cov[i, j])
        
        mean_intra = np.mean(intra_values)
        mean_inter = np.mean(inter_values)
        
        # We expect intra to be significantly higher (e.g. ~0.6 vs ~0.1)
        self.assertGreater(mean_intra, mean_inter, 
                           f"Intra ({mean_intra:.2f}) should be > Inter ({mean_inter:.2f})")

    def test_mask_generation(self):
        """Test if masks are generated correctly and are symmetric."""
        # Test Within-Module Mask
        mask_intra = self.gen.get_mask_within_module(0)
        self.assertTrue(np.all(mask_intra == mask_intra.T))
        # Ensure mask only targets nodes in module 0
        nodes_in_0 = np.where(self.gen.labels == 0)[0]
        # Check a connection within module 0
        self.assertEqual(mask_intra[nodes_in_0[0], nodes_in_0[1]], 1)
        
        # Test Between-Module Mask
        mask_inter = self.gen.get_mask_between_modules(0, 1)
        self.assertTrue(np.all(mask_inter == mask_inter.T))
        nodes_in_1 = np.where(self.gen.labels == 1)[0]
        # Check connection between 0 and 1
        self.assertEqual(mask_inter[nodes_in_0[0], nodes_in_1[0]], 1)

    def test_data_generation_shapes(self):
        """Test shapes of generated subject data."""
        mask = self.gen.get_mask_within_module(0)
        n_g1, n_g2 = 10, 15
        g1, g2, _ = self.gen.generate_data(mask, effect_size=0.2, n_samples_g1=n_g1, n_samples_g2=n_g2)
        
        self.assertEqual(g1.shape, (n_g1, self.N, self.N))
        self.assertEqual(g2.shape, (n_g2, self.N, self.N))

    def test_effect_application(self):
        """Test if Group 2 actually has higher correlation in the masked region."""
        # Define effect
        mask = self.gen.get_mask_within_module(0)
        effect_size = 0.3
        
        # Generate with low sampling noise (high time_points) to detect mean shift clearly
        g1, g2, _ = self.gen.generate_data(
            mask, effect_size=effect_size, 
            n_samples_g1=100, n_samples_g2=100, 
            time_points=2000
        )
        
        mean_g1 = np.mean(g1, axis=0)
        mean_g2 = np.mean(g2, axis=0)
        
        # Check diff in masked region
        diff = mean_g2 - mean_g1
        masked_diff = diff[mask == 1]
        non_masked_diff = diff[mask == 0]
        
        # Average difference in mask should be close to effect_size
        # (It won't be exact due to SPD enforcement and sampling noise)
        avg_effect = np.mean(masked_diff)
        avg_noise = np.mean(non_masked_diff)
        
        print(f"\nDEBUG: Measured Effect: {avg_effect:.4f}, Expected: ~{effect_size}")
        print(f"DEBUG: Background Noise Diff: {avg_noise:.4f}")

        self.assertGreater(avg_effect, 0.15, "Effect was not applied correctly in Group 2")
        self.assertLess(abs(avg_noise), 0.05, "Background should not change significantly")