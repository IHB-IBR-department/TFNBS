"""
Shared synthetic-data factories for tests.

Each function is a named scenario with a docstring describing what it
represents and the conditions it's designed to exercise. Callers are free
to override the defaults via kwargs — the defaults are the canonical shape.

Two families of scenarios:

**Group-comparison (t-test pipeline)** — built on
:func:`conninfpy.synth_datasets.generate_fc_matrices`. Returns a 3-tuple
``(group1, group2, (cov1, cov2))`` matching that helper's convention.

    small_two_sample       : N=30, effect=0.2, n_g1=30, n_g2=20 — quick sanity
    paired_effect_moderate : N=30, effect=0.2, n=40 per group   — paired tests
    medium_two_sample      : N=100, effect=0.2, n_g1=50, n_g2=40 — larger signal
    null_two_sample        : N=30, effect=0.0, n_g1=30, n_g2=20 — type-I / null
    tiny_two_sample        : N=15, effect=0.3, n=20/25          — fast-path tests

**GLM (continuous predictor) pipeline** — symmetric noisy Y and a
continuous predictor (age), optionally with a confound. Returns a dict
with keys ``Y``, ``age``, optionally ``motion``, and metadata (``N``, ``n``,
``effect_edge``, ``effect_beta``).

    glm_null                 : pure noise, for null-control tests
    glm_planted_edge         : strong effect planted at a single edge
    glm_confound_driven      : signal at an edge driven by a confound
                               (motion), not by the predictor of interest —
                               used to verify Freedman-Lane partials out
                               the confound

All scenarios use fixed seeds for reproducibility. Pass ``seed=...`` to
override if you need independent draws.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt

from conninfpy.synth_datasets import generate_fc_matrices


# -----------------------------------------------------------------------------
# Group-comparison scenarios (t-test pipeline)
# -----------------------------------------------------------------------------

def small_two_sample(
    seed: int = 42,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], Tuple[npt.NDArray, npt.NDArray]]:
    """N=30, effect_size=0.2, n_g1=30, n_g2=20.

    Moderate signal on modest-sized network. Canonical "quick sanity" shape
    used by most t-test correctness tests.
    """
    return generate_fc_matrices(
        30, effect_size=0.2,
        n_samples_group1=30, n_samples_group2=20, seed=seed,
    )


def paired_effect_moderate(
    seed: int = 42,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], Tuple[npt.NDArray, npt.NDArray]]:
    """N=30, effect_size=0.2, n=40 per group (matched).

    Equal-n groups for paired t-test checks. Uses the same covariance
    structure as :func:`small_two_sample` but larger sample size.
    """
    return generate_fc_matrices(
        30, effect_size=0.2,
        n_samples_group1=40, n_samples_group2=40, seed=seed,
    )


def medium_two_sample(
    seed: int = 42,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], Tuple[npt.NDArray, npt.NDArray]]:
    """N=100, effect_size=0.2, n_g1=50, n_g2=40.

    Larger network for tests that exercise the permutation loop at a
    more realistic scale (e.g. TFNBS multi-parameter sweeps).
    """
    return generate_fc_matrices(
        100, effect_size=0.2,
        n_samples_group1=50, n_samples_group2=40, seed=seed,
    )


def null_two_sample(
    seed: int = 42,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], Tuple[npt.NDArray, npt.NDArray]]:
    """N=30, effect_size=0.0, n_g1=30, n_g2=20.

    No planted effect — for type-I / null-control checks and benchmarks
    that care only about timing, not detectability.
    """
    return generate_fc_matrices(
        30, effect_size=0.0,
        n_samples_group1=30, n_samples_group2=20, seed=seed,
    )


def tiny_two_sample(
    seed: int = 42,
    n_samples_group1: int = 20,
    n_samples_group2: int = 25,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], Tuple[npt.NDArray, npt.NDArray]]:
    """N=15, effect_size=0.3.

    Small network used by fast-path unit tests where we want:
    a) enough signal for shape / bound assertions to pass reliably,
    b) quick execution for test-suite turnaround.

    Defaults to asymmetric sizes (20, 25). Pass matched sizes for paired tests.
    """
    return generate_fc_matrices(
        15, effect_size=0.3,
        n_samples_group1=n_samples_group1,
        n_samples_group2=n_samples_group2,
        seed=seed,
    )


# -----------------------------------------------------------------------------
# GLM scenarios (continuous-predictor pipeline)
# -----------------------------------------------------------------------------

def _symmetrize_zero_diagonal(Y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Symmetrize each (N, N) slice and zero its diagonal."""
    Y = (Y + Y.transpose(0, 2, 1)) / 2
    for s in range(Y.shape[0]):
        np.fill_diagonal(Y[s], 0)
    return Y


def glm_null(
    n: int = 40,
    N: int = 10,
    noise_scale: float = 1.0,
    seed: int = 99,
) -> Dict[str, Any]:
    """Pure-noise GLM data: ``Y ~ N(0, noise_scale²)``, age ~ N(0, 1).

    No planted effect. Used to verify that the pipeline produces no
    systematic inflation under H0 (type-I error control, null-dist shape).

    Returns dict: ``Y``, ``age``, ``N``, ``n``.
    """
    rng = np.random.RandomState(seed)
    Y = rng.randn(n, N, N) * noise_scale
    Y = _symmetrize_zero_diagonal(Y)
    age = rng.randn(n)
    return {"Y": Y, "age": age, "N": N, "n": n}


def glm_planted_edge(
    n: int = 40,
    N: int = 10,
    effect_edge: Tuple[int, int] = (2, 3),
    effect_beta: float = 1.0,
    noise_scale: float = 0.5,
    seed: int = 123,
) -> Dict[str, Any]:
    """Noise + a known continuous effect at one edge: ``Y[s, i, j] += β · age[s]``.

    Used for:
      - Detection tests (the planted edge should have the max t-stat)
      - Effect-size recovery (β̂ ≈ β at the planted edge)
      - Power of permutation procedures

    Parameters
    ----------
    effect_edge : (i, j)
        Which edge carries the planted effect. Symmetrized automatically.
    effect_beta : float
        Signal amplitude. Default 1.0 with ``noise_scale=0.5`` gives ~6-10 t.
    noise_scale : float
        Pre-effect noise level (applied to the whole matrix).

    Returns dict: ``Y``, ``age``, ``N``, ``n``, ``effect_edge``, ``effect_beta``.
    """
    rng = np.random.RandomState(seed)
    age = rng.randn(n)
    Y = rng.randn(n, N, N) * noise_scale
    Y = _symmetrize_zero_diagonal(Y)
    i, j = effect_edge
    for s in range(n):
        Y[s, i, j] += effect_beta * age[s]
        Y[s, j, i] += effect_beta * age[s]
    return {
        "Y": Y, "age": age, "N": N, "n": n,
        "effect_edge": effect_edge, "effect_beta": effect_beta,
    }


def glm_confound_driven(
    n: int = 40,
    N: int = 10,
    effect_edge: Tuple[int, int] = (1, 2),
    confound_beta: float = 1.5,
    confound_corr: float = 0.7,
    noise_scale: float = 0.3,
    seed: int = 456,
) -> Dict[str, Any]:
    """Signal at ``effect_edge`` driven by a *confound* (motion), not by age.

    Age and motion are correlated (``confound_corr``). This exercises
    Freedman-Lane partialling: with motion regressed out as a confound,
    the apparent age→edge association should vanish; without the
    confound in the design matrix, it spuriously surfaces.

    Returns dict: ``Y``, ``age``, ``motion``, ``N``, ``n``, ``effect_edge``.
    """
    rng = np.random.RandomState(seed)
    age = rng.randn(n)
    motion = confound_corr * age + np.sqrt(1 - confound_corr ** 2) * rng.randn(n)
    Y = rng.randn(n, N, N) * noise_scale
    Y = _symmetrize_zero_diagonal(Y)
    i, j = effect_edge
    for s in range(n):
        Y[s, i, j] += confound_beta * motion[s]
        Y[s, j, i] += confound_beta * motion[s]
    return {
        "Y": Y, "age": age, "motion": motion, "N": N, "n": n,
        "effect_edge": effect_edge,
    }
