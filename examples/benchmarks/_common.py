"""
Shared data factories for benchmarks.

All benchmarks use biologically realistic synthetic data built on top of
:mod:`conninfpy.topologies` (modular covariance + named topology scenarios).
This replaces ad-hoc ``rng.randn`` baselines that don't reflect real BOLD
signal structure.

The **four canonical topologies** are those highlighted in Figure 1 of
the MICCAI 2026 ConnInfPy paper — they span the meaningful structural
regimes for method comparison:

    - ``within_module_dense``     — block-aligned effect; cNBS/FBC-TFNBS win
    - ``hub``                     — star topology spanning blocks; stress test for cNBS
    - ``rich_club``               — dense clique of high-degree nodes; TFNBS favored
    - ``scattered_cross_block``   — diffuse non-aligned effect; equalizes methods

Factories
---------

``make_two_sample`` — Fisher-z'd (g1, g2) pair with topology-defined effect.
    Use for t-test / enhancement benchmarks (timing, precompsum).

``make_glm_data`` — (Y, age, motion_or_None) for GLM benchmarks.
    Baseline Y from the modular covariance (null topology), plus an
    age-driven planted effect at one edge. Optional motion confound
    correlated with age.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from conninfpy.topologies import TopologyDatasetGenerator
from conninfpy.utils import fisher_r_to_z


# The 4 topologies highlighted in MICCAI 2026 Figure 1 rows 2-3.
PAPER_TOPOLOGIES: Tuple[str, ...] = (
    "within_module_dense",
    "hub",
    "rich_club",
    "scattered_cross_block",
)


def make_two_sample(
    N: int,
    n_sub: int,
    topology: str = "within_module_dense",
    effect_size: float = 0.3,
    n_modules: int = 4,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Two-group Fisher-z'd FC matrices with a topology-defined effect.

    Used by timing / enhancement / sums-path benchmarks.

    Parameters
    ----------
    N : number of nodes
    n_sub : subjects per group
    topology : scenario name from conninfpy.topologies (e.g. "hub", "rich_club")
    effect_size : Cohen's d shift at masked edges (0 = pure null)
    n_modules : number of functional blocks in the base covariance
    seed : RNG seed

    Returns
    -------
    (g1_z, g2_z) : both (n_sub, N, N) Fisher-z'd
    """
    gen = TopologyDatasetGenerator(
        n_nodes=N, n_modules=n_modules, seed=seed,
    )
    ds = gen.generate(
        topology,
        effect_size=effect_size,
        n_samples=n_sub,
        time_points=max(40, 2 * N),  # enough TS length to stabilize correlations
    )
    return fisher_r_to_z(ds.group1), fisher_r_to_z(ds.group2)


def make_glm_data(
    N: int,
    n_sub: int,
    topology: str = "within_module_dense",
    effect_edge: Tuple[int, int] = (1, 2),
    effect_beta: float = 1.0,
    confound: bool = False,
    confound_corr: float = 0.7,
    n_modules: int = 4,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """GLM-ready connectivity + continuous predictor with optional confound.

    Baseline Y comes from the modular covariance (topology scenario at
    effect_size=0 → null backbone). An age-driven planted effect is added
    at ``effect_edge``: ``Y[s, i, j] += effect_beta * age[s]``.
    When ``confound=True``, motion is generated correlated with age
    (``corr = confound_corr``) — caller can pass it as a confound to
    ``compute_p_val_glm`` to exercise Freedman-Lane partialling.

    Parameters
    ----------
    N, n_sub : dimensions
    topology : backbone topology (just for realistic modular covariance;
               effect planted at one edge regardless)
    effect_edge : (i, j) where to plant the continuous-predictor effect
    effect_beta : strength of the planted effect
    confound : if True, also return a motion vector correlated with age
    confound_corr : correlation between age and motion when confound=True
    n_modules : modules in the base covariance
    seed : RNG seed

    Returns
    -------
    (Y_z, age, motion_or_None)
        Y_z : (n_sub, N, N) Fisher-z'd connectivity
        age : (n_sub,) predictor of interest
        motion : (n_sub,) confound vector, or None if confound=False
    """
    gen = TopologyDatasetGenerator(
        n_nodes=N, n_modules=n_modules, seed=seed,
    )
    # Null backbone — only the modular covariance, no planted group effect
    ds = gen.generate(
        topology, effect_size=0.0, n_samples=n_sub,
        time_points=max(40, 2 * N),
    )
    # Plant the age-driven effect in Fisher-z space (unbounded), not on raw
    # correlations — r + β·age can overflow [-1, 1] for modest β.
    Y_z = fisher_r_to_z(ds.group1).copy()

    rng = np.random.RandomState(seed)
    age = rng.randn(n_sub)

    i, j = effect_edge
    for s in range(n_sub):
        Y_z[s, i, j] += effect_beta * age[s]
        Y_z[s, j, i] += effect_beta * age[s]

    motion = None
    if confound:
        noise = rng.randn(n_sub)
        motion = confound_corr * age + np.sqrt(1 - confound_corr ** 2) * noise

    return Y_z, age, motion
