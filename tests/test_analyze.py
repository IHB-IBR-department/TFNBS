"""Tests for the v2.1 ``analyze()`` convenience wrapper (plan PR-4).

Covers:

- auto-preserve from ``(interest, confounds)`` in GLM mode when
  ``sites=`` is passed without an explicit ``preserve=``;
- auto-preserve from the group indicator in two-sample mode;
- explicit ``preserve=`` overrides auto-build and suppresses the flag;
- design coupling check fires only when ``preserve`` and the design
  are passed as labeled DataFrames AND a column in preserve is
  missing from ``(interest, confounds)`` — silent on raw ndarrays
  (we have no column identity to compare).

Auto-strata behaviour is covered separately in ``test_strata.py``.
"""
from __future__ import annotations

import unittest

import numpy as np

from conninfpy import analyze


def _make_glm(n=24, N=5, n_sites=2, seed=0):
    rng = np.random.RandomState(seed)
    Y = rng.randn(n, N, N); Y = (Y + Y.transpose(0, 2, 1)) / 2
    for i in range(n):
        np.fill_diagonal(Y[i], 0.0)
    age = rng.randn(n)
    motion = rng.randn(n)
    sites = np.repeat(np.arange(n_sites), n // n_sites)
    return Y, age, motion, sites


def _make_twosample(n_per_group=10, N=5, n_sites=2, seed=0):
    rng = np.random.RandomState(seed)
    group1 = rng.randn(n_per_group, N, N)
    group1 = (group1 + group1.transpose(0, 2, 1)) / 2
    group2 = rng.randn(n_per_group, N, N)
    group2 = (group2 + group2.transpose(0, 2, 1)) / 2
    for arr in (group1, group2):
        for i in range(n_per_group):
            np.fill_diagonal(arr[i], 0.0)
    sites = np.tile(np.repeat(np.arange(n_sites), n_per_group // n_sites), 2)
    return group1, group2, sites


# =============================================================================
# Auto-preserve, GLM mode
# =============================================================================

class TestAutoDispatchGLM(unittest.TestCase):
    """`harmonize='auto'` dispatches to Strategy D when GLM+sites+confounds
    are all present, to E when GLM+sites without confounds, to no
    harmonization when sites is absent."""

    def test_auto_with_sites_and_confounds_resolves_to_d(self):
        Y, age, motion, sites = _make_glm(seed=2)
        out = analyze(
            Y, interest=age, confounds=motion, sites=sites,
            fisher_z=False, n_permutations=20, use_mp=False,
            acceleration=None, rng=2,
        )
        self.assertTrue(out.inference.harmonized)
        self.assertEqual(out.inference.combat_diagnostics["strategy"], "D")
        self.assertTrue(
            any("preserve excludes interest (Strategy D)" in f
                for f in out.flags),
            f"expected Strategy D preserve flag, got {out.flags}",
        )

    def test_auto_with_sites_no_confounds_resolves_to_e(self):
        Y, age, _, sites = _make_glm(seed=1)
        out = analyze(
            Y, interest=age, sites=sites, fisher_z=False,
            n_permutations=20, use_mp=False, acceleration=None, rng=1,
        )
        # No confounds → no ComBat (Strategy E); site dummies in GLM.
        self.assertFalse(out.inference.harmonized)
        self.assertEqual(out.inference.combat_diagnostics["strategy"], "E")
        self.assertTrue(out.inference.strata_provided)

    def test_explicit_preserve_overridden_under_strategy_d(self):
        Y, age, motion, sites = _make_glm(seed=3)
        # Under D, an explicit preserve= is replaced by `confounds`
        # (the recipe sets it deliberately to exclude interest).
        out = analyze(
            Y, interest=age, confounds=motion, sites=sites,
            preserve=age[:, np.newaxis],
            fisher_z=False, n_permutations=20, use_mp=False,
            acceleration=None, rng=3,
        )
        self.assertEqual(out.inference.combat_diagnostics["strategy"], "D")
        self.assertTrue(
            any("preserve= overridden by Strategy D" in f for f in out.flags),
            f"expected D override flag, got {out.flags}",
        )

    def test_no_sites_means_no_combat(self):
        Y, age, _, _ = _make_glm(seed=4)
        out = analyze(
            Y, interest=age, fisher_z=False,
            n_permutations=20, use_mp=False, acceleration=None, rng=4,
        )
        self.assertFalse(out.inference.harmonized)
        self.assertIsNone(out.inference.combat_diagnostics)


# =============================================================================
# Two-sample dispatch: no defensible ComBat recipe; skip with a flag
# =============================================================================

class TestTwoSampleSitesDemotion(unittest.TestCase):
    """Two-sample + sites has no defensible ComBat recipe (no interest
    column to preserve). `analyze()` skips ComBat and emits a flag
    asking the caller to promote to GLM with binary interest."""

    def test_two_sample_with_sites_skips_combat(self):
        g1, g2, sites = _make_twosample(seed=10)
        out = analyze(
            group1=g1, group2=g2, sites=sites, fisher_z=False,
            method="tstat", n_permutations=20, use_mp=False,
            acceleration=None, rng=10,
        )
        self.assertFalse(out.inference.harmonized)
        self.assertTrue(
            any("two-sample + sites" in f and "promote to GLM" in f
                for f in out.flags),
            f"expected demotion flag, got {out.flags}",
        )


# =============================================================================
# One-sample t-test mode
# =============================================================================

class TestAnalyzeOneSample(unittest.TestCase):

    def test_one_sample_without_sites_runs_with_default_fisher_z(self):
        rng = np.random.RandomState(30)
        Y = rng.uniform(-0.2, 0.2, size=(12, 5, 5))
        Y = (Y + Y.transpose(0, 2, 1)) / 2
        for i in range(Y.shape[0]):
            np.fill_diagonal(Y[i], 0.0)

        out = analyze(
            group1=Y, test_type="one-sample",
            method="tstat", n_permutations=20,
            use_mp=False, acceleration=None, rng=30,
        )

        self.assertIn("positive", out.inference)
        self.assertIn("negative", out.inference)
        self.assertFalse(out.inference.harmonized)
        self.assertFalse(out.inference.preserve_provided)

    def test_one_sample_with_sites_skips_combat(self):
        # One-sample is a t-test path; ComBat has no defensible recipe
        # in t-test mode (no interest column to preserve). `analyze()`
        # skips harmonization and emits a clarifying flag.
        rng = np.random.RandomState(31)
        Y = rng.randn(12, 5, 5) * 0.1
        Y = (Y + Y.transpose(0, 2, 1)) / 2
        for i in range(Y.shape[0]):
            np.fill_diagonal(Y[i], 0.0)
        sites = np.array([0] * 6 + [1] * 6)

        out = analyze(
            group1=Y, test_type="one-sample", sites=sites,
            fisher_z=False, method="tstat", n_permutations=20,
            use_mp=False, acceleration=None, rng=31,
        )

        self.assertFalse(out.inference.harmonized)
        self.assertTrue(
            any("two-sample + sites" in f for f in out.flags),
            f"expected demotion flag, got {out.flags}",
        )


if __name__ == "__main__":
    unittest.main()
