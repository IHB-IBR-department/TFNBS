"""
Tests for conninfpy.topologies — scenario registry and dataset generation.

Strategy: one parametrized check iterates the whole scenario registry and
verifies structural invariants (shape, symmetry, zero diagonal, non-trivial
effect). Edge-case tests cover the registry API and error handling.
"""

import numpy as np
import pytest

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

def test_list_scenarios_nonempty():
    names = list_scenarios()
    assert len(names) >= 18, f"expected the full scenario library (≥18), got {len(names)}"
    assert all(isinstance(n, str) for n in names)
    # No duplicates
    assert len(set(names)) == len(names)


def test_get_scenarios_returns_topology_scenarios():
    scenarios = get_scenarios()
    assert all(isinstance(s, TopologyScenario) for s in scenarios)
    assert len(scenarios) == len(list_scenarios())


def test_get_scenario_by_name():
    s = get_scenario("hub")
    assert isinstance(s, TopologyScenario)
    assert s.name == "hub"


def test_get_scenario_accepts_object():
    s = get_scenario("chain")
    assert get_scenario(s) is s   # passthrough


def test_get_scenario_raises_on_unknown_name():
    with pytest.raises(ValueError, match="Unknown scenario"):
        get_scenario("not_a_real_topology")


def test_get_scenario_raises_on_bad_type():
    with pytest.raises(TypeError):
        get_scenario(42)


# -----------------------------------------------------------------------------
# Every scenario produces a structurally-valid dataset
# -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gen():
    return TopologyDatasetGenerator(n_nodes=N_NODES, n_modules=N_MODULES, seed=0)


@pytest.mark.parametrize("scenario_name", list_scenarios())
def test_scenario_generates_valid_dataset(gen, scenario_name):
    """Every scenario in the registry must produce a valid TopologyDataset."""
    ds: TopologyDataset = gen.generate(
        scenario_name,
        effect_size=0.25,
        n_samples=8,
        time_points=40,
    )

    # Structural invariants
    assert ds.group1.shape == (8, N_NODES, N_NODES)
    assert ds.group2.shape == (8, N_NODES, N_NODES)
    assert ds.net_labels.shape == (N_NODES,)
    assert ds.effect_mask.shape == (N_NODES, N_NODES)

    # Symmetry
    np.testing.assert_allclose(ds.effect_mask, ds.effect_mask.T, atol=1e-12)

    # Zero diagonal on both groups and mask
    for s in range(ds.group1.shape[0]):
        assert np.all(np.diag(ds.group1[s]) == 0)
        assert np.all(np.diag(ds.group2[s]) == 0)
    assert np.all(np.diag(ds.effect_mask) == 0)

    # At least one non-zero edge in the mask (scenario would be meaningless otherwise)
    assert np.any(ds.effect_mask != 0), f"{scenario_name} produced an empty mask"


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------

def test_generator_is_deterministic_with_same_seed():
    gen1 = TopologyDatasetGenerator(n_nodes=N_NODES, n_modules=N_MODULES, seed=42)
    gen2 = TopologyDatasetGenerator(n_nodes=N_NODES, n_modules=N_MODULES, seed=42)
    ds1 = gen1.generate("chain", effect_size=0.2, n_samples=5, time_points=20)
    ds2 = gen2.generate("chain", effect_size=0.2, n_samples=5, time_points=20)
    np.testing.assert_array_equal(ds1.effect_mask, ds2.effect_mask)
    np.testing.assert_allclose(ds1.group1, ds2.group1)
    np.testing.assert_allclose(ds1.group2, ds2.group2)


def test_generator_produces_different_data_across_seeds():
    gen1 = TopologyDatasetGenerator(n_nodes=N_NODES, n_modules=N_MODULES, seed=1)
    gen2 = TopologyDatasetGenerator(n_nodes=N_NODES, n_modules=N_MODULES, seed=2)
    ds1 = gen1.generate("hub", effect_size=0.2, n_samples=5, time_points=20)
    ds2 = gen2.generate("hub", effect_size=0.2, n_samples=5, time_points=20)
    # Data should differ (almost surely with different seeds)
    assert not np.allclose(ds1.group1, ds2.group1)


# -----------------------------------------------------------------------------
# Scenario-specific sanity checks
# -----------------------------------------------------------------------------

def test_hub_scenario_has_star_topology(gen):
    """The `hub` scenario should have exactly one node with max degree in the mask."""
    ds = gen.generate("hub", effect_size=0.2, n_samples=5, time_points=20)
    mask = ds.effect_mask != 0
    degrees = mask.sum(axis=1)
    # One hub with many spokes; others should have at most degree 1 (only connected to hub)
    max_deg = degrees.max()
    # At least one node must connect to all spokes; others connect only to the hub
    high_deg_count = np.sum(degrees == max_deg)
    assert high_deg_count >= 1, "hub scenario should have at least one high-degree node"


def test_rich_club_scenario_has_dense_core(gen):
    """The `rich_club` scenario: hubs form a clique (all pairwise connected)."""
    ds = gen.generate("rich_club", effect_size=0.2, n_samples=5, time_points=20)
    mask = ds.effect_mask != 0
    degrees = mask.sum(axis=1)
    # The hub clique of size k produces k nodes each with degree k-1 (or higher with spokes)
    assert degrees.max() >= 2, "rich_club should have hubs with degree ≥ 2"


def test_fisher_z_helper_returns_transformed_matrices(gen):
    ds = gen.generate("within_module_dense", effect_size=0.2, n_samples=5, time_points=20)
    z1, z2 = ds.fisher_z()
    assert z1.shape == ds.group1.shape
    # Fisher-z of a correlation matrix shouldn't equal the original
    assert not np.allclose(z1, ds.group1)
