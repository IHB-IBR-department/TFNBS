"""
Tests for conninfpy.topologies — scenario registry and dataset generation.

Strategy: one subTest-parametrized check iterates the whole scenario registry
and verifies structural invariants (shape, symmetry, zero diagonal, non-trivial
effect). Edge-case tests cover the registry API and error handling.
"""

import unittest

import numpy as np

from conninfpy.topologies import (
    TopologyDataset,
    TopologyDatasetGenerator,
    TopologyScenario,
    get_scenario,
    get_scenarios,
    list_scenarios,
)


N_NODES = 60
N_MODULES = 4


# -----------------------------------------------------------------------------
# Registry API
# -----------------------------------------------------------------------------

class TestScenarioRegistry(unittest.TestCase):
    """Registry helpers: list_scenarios, get_scenarios, get_scenario."""

    def test_list_scenarios_nonempty(self):
        names = list_scenarios()
        self.assertGreaterEqual(
            len(names), 18,
            f"expected the full scenario library (>=18), got {len(names)}",
        )
        self.assertTrue(all(isinstance(n, str) for n in names))
        # No duplicates
        self.assertEqual(len(set(names)), len(names))

    def test_get_scenarios_returns_topology_scenarios(self):
        scenarios = get_scenarios()
        self.assertTrue(all(isinstance(s, TopologyScenario) for s in scenarios))
        self.assertEqual(len(scenarios), len(list_scenarios()))

    def test_get_scenario_by_name(self):
        s = get_scenario("hub")
        self.assertIsInstance(s, TopologyScenario)
        self.assertEqual(s.name, "hub")

    def test_get_scenario_accepts_object(self):
        s = get_scenario("chain")
        self.assertIs(get_scenario(s), s)  # passthrough

    def test_get_scenario_raises_on_unknown_name(self):
        with self.assertRaisesRegex(ValueError, "Unknown scenario"):
            get_scenario("not_a_real_topology")

    def test_get_scenario_raises_on_bad_type(self):
        with self.assertRaises(TypeError):
            get_scenario(42)


# -----------------------------------------------------------------------------
# Every scenario produces a structurally-valid dataset
# -----------------------------------------------------------------------------

class TestScenarioGeneration(unittest.TestCase):
    """Each scenario in the registry must produce a valid TopologyDataset."""

    @classmethod
    def setUpClass(cls):
        cls.gen = TopologyDatasetGenerator(
            n_nodes=N_NODES, n_modules=N_MODULES, seed=0,
        )

    def test_every_scenario_generates_valid_dataset(self):
        """Structural invariants for every registered scenario (subTest)."""
        for scenario_name in list_scenarios():
            with self.subTest(scenario=scenario_name):
                ds: TopologyDataset = self.gen.generate(
                    scenario_name,
                    effect_size=0.25,
                    n_samples=8,
                    time_points=40,
                )

                # Structural invariants
                self.assertEqual(ds.group1.shape, (8, N_NODES, N_NODES))
                self.assertEqual(ds.group2.shape, (8, N_NODES, N_NODES))
                self.assertEqual(ds.net_labels.shape, (N_NODES,))
                self.assertEqual(ds.effect_mask.shape, (N_NODES, N_NODES))

                # Symmetry of effect mask
                np.testing.assert_allclose(
                    ds.effect_mask, ds.effect_mask.T, atol=1e-12,
                )

                # Zero diagonal on both groups and mask
                for s in range(ds.group1.shape[0]):
                    self.assertTrue(np.all(np.diag(ds.group1[s]) == 0))
                    self.assertTrue(np.all(np.diag(ds.group2[s]) == 0))
                self.assertTrue(np.all(np.diag(ds.effect_mask) == 0))

                # At least one non-zero edge in the mask (otherwise scenario
                # is meaningless)
                self.assertTrue(
                    np.any(ds.effect_mask != 0),
                    f"{scenario_name} produced an empty mask",
                )


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------

class TestReproducibility(unittest.TestCase):

    def test_generator_is_deterministic_with_same_seed(self):
        gen1 = TopologyDatasetGenerator(n_nodes=N_NODES, n_modules=N_MODULES, seed=42)
        gen2 = TopologyDatasetGenerator(n_nodes=N_NODES, n_modules=N_MODULES, seed=42)
        ds1 = gen1.generate("chain", effect_size=0.2, n_samples=5, time_points=20)
        ds2 = gen2.generate("chain", effect_size=0.2, n_samples=5, time_points=20)
        np.testing.assert_array_equal(ds1.effect_mask, ds2.effect_mask)
        np.testing.assert_allclose(ds1.group1, ds2.group1)
        np.testing.assert_allclose(ds1.group2, ds2.group2)

    def test_generator_produces_different_data_across_seeds(self):
        gen1 = TopologyDatasetGenerator(n_nodes=N_NODES, n_modules=N_MODULES, seed=1)
        gen2 = TopologyDatasetGenerator(n_nodes=N_NODES, n_modules=N_MODULES, seed=2)
        ds1 = gen1.generate("hub", effect_size=0.2, n_samples=5, time_points=20)
        ds2 = gen2.generate("hub", effect_size=0.2, n_samples=5, time_points=20)
        self.assertFalse(np.allclose(ds1.group1, ds2.group1))


# -----------------------------------------------------------------------------
# Scenario-specific sanity checks
# -----------------------------------------------------------------------------

class TestScenarioSanityChecks(unittest.TestCase):
    """Shape / topology sanity on a few specific scenarios."""

    @classmethod
    def setUpClass(cls):
        cls.gen = TopologyDatasetGenerator(
            n_nodes=N_NODES, n_modules=N_MODULES, seed=0,
        )

    def test_hub_scenario_has_star_topology(self):
        """`hub`: at least one node with max degree in the effect mask."""
        ds = self.gen.generate("hub", effect_size=0.2, n_samples=5, time_points=20)
        mask = ds.effect_mask != 0
        degrees = mask.sum(axis=1)
        max_deg = degrees.max()
        high_deg_count = int(np.sum(degrees == max_deg))
        self.assertGreaterEqual(
            high_deg_count, 1,
            "hub scenario should have at least one high-degree node",
        )

    def test_rich_club_scenario_has_dense_core(self):
        """`rich_club`: hubs form a clique — at least one hub with degree >= 2."""
        ds = self.gen.generate(
            "rich_club", effect_size=0.2, n_samples=5, time_points=20,
        )
        mask = ds.effect_mask != 0
        degrees = mask.sum(axis=1)
        self.assertGreaterEqual(
            int(degrees.max()), 2, "rich_club should have hubs with degree >= 2",
        )

    def test_fisher_z_helper_returns_transformed_matrices(self):
        ds = self.gen.generate(
            "within_module_dense", effect_size=0.2, n_samples=5, time_points=20,
        )
        z1, z2 = ds.fisher_z()
        self.assertEqual(z1.shape, ds.group1.shape)
        # Fisher-z of a correlation matrix shouldn't equal the original
        self.assertFalse(np.allclose(z1, ds.group1))


if __name__ == "__main__":
    unittest.main()
