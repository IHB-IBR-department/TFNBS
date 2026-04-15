"""
Shared fixtures for GLM extension tests.

Provides reusable synthetic data and design matrices following
the TDD test plan (dev parameters: N=10, n_subjects=20, d=1.0).
"""

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Reproducible RNG."""
    return np.random.RandomState(42)


@pytest.fixture
def null_connectivity(rng):
    """
    Pure noise connectivity matrices (n=40, N=10).
    Symmetric, zero diagonal.
    """
    n_subjects = 40
    N = 10
    Y = rng.randn(n_subjects, N, N)
    # Symmetrize
    Y = (Y + Y.transpose(0, 2, 1)) / 2
    # Zero diagonal
    for s in range(n_subjects):
        np.fill_diagonal(Y[s], 0)
    return Y


@pytest.fixture
def binary_group_design():
    """
    Binary group indicator + intercept (20 per group).
    Returns X, contrast, group_indices.
    """
    n_per_group = 20
    n_total = 2 * n_per_group
    intercept = np.ones(n_total)
    group = np.concatenate([np.zeros(n_per_group), np.ones(n_per_group)])
    X = np.column_stack([intercept, group])
    contrast = np.array([0.0, 1.0])
    return X, contrast, n_per_group


@pytest.fixture
def continuous_design(rng):
    """
    Intercept + age (continuous) + motion (confound).
    Returns X, contrast, age, motion.
    """
    n_subjects = 40
    age = rng.randn(n_subjects)
    motion = rng.randn(n_subjects)
    intercept = np.ones(n_subjects)
    X = np.column_stack([intercept, motion, age])
    contrast = np.array([0.0, 0.0, 1.0])  # test age
    return X, contrast, age, motion
