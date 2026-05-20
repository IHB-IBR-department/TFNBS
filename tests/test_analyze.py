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

class TestAutoPreserveGLM(unittest.TestCase):

    def test_auto_preserve_from_interest(self):
        Y, age, _, sites = _make_glm(seed=1)
        out = analyze(
            Y, interest=age, sites=sites, fisher_z=False,
            n_permutations=20, use_mp=False, acceleration=None, rng=1,
        )
        self.assertTrue(out.inference.preserve_provided)
        self.assertTrue(
            any("preserve auto-built from (interest, confounds)" in f
                for f in out.flags),
            f"expected auto-preserve flag, got {out.flags}",
        )

    def test_auto_preserve_from_interest_plus_confounds(self):
        Y, age, motion, sites = _make_glm(seed=2)
        out = analyze(
            Y, interest=age, confounds=motion, sites=sites,
            fisher_z=False, n_permutations=20, use_mp=False,
            acceleration=None, rng=2,
        )
        self.assertTrue(out.inference.preserve_provided)
        self.assertTrue(
            any("auto-built" in f for f in out.flags),
            f"expected auto-preserve flag, got {out.flags}",
        )

    def test_explicit_preserve_overrides(self):
        Y, age, motion, sites = _make_glm(seed=3)
        # Caller provides their own preserve (just `age`, not motion)
        out = analyze(
            Y, interest=age, confounds=motion, sites=sites,
            preserve=age[:, np.newaxis],
            fisher_z=False, n_permutations=20, use_mp=False,
            acceleration=None, rng=3,
        )
        self.assertTrue(out.inference.preserve_provided)
        self.assertFalse(
            any("auto-built" in f for f in out.flags),
            f"expected no auto-build flag when preserve= passed, got {out.flags}",
        )

    def test_no_sites_no_autopreserve(self):
        Y, age, _, _ = _make_glm(seed=4)
        out = analyze(
            Y, interest=age, fisher_z=False,
            n_permutations=20, use_mp=False, acceleration=None, rng=4,
        )
        self.assertFalse(out.inference.preserve_provided)
        self.assertFalse(
            any("auto-built" in f for f in out.flags),
            f"expected no auto-preserve flag without sites=, got {out.flags}",
        )


# =============================================================================
# Auto-preserve, two-sample mode
# =============================================================================

class TestAutoPreserveTwoSample(unittest.TestCase):

    def test_auto_preserve_from_group_indicator(self):
        g1, g2, sites = _make_twosample(seed=10)
        out = analyze(
            group1=g1, group2=g2, sites=sites, fisher_z=False,
            method="tstat", n_permutations=20, use_mp=False,
            acceleration=None, rng=10,
        )
        self.assertTrue(out.inference.preserve_provided)
        self.assertTrue(
            any("group indicator" in f for f in out.flags),
            f"expected group-indicator auto-preserve flag, got {out.flags}",
        )

    def test_explicit_preserve_overrides_twosample(self):
        g1, g2, sites = _make_twosample(seed=11)
        n = g1.shape[0] + g2.shape[0]
        rng = np.random.RandomState(11)
        my_preserve = rng.randn(n, 1)
        out = analyze(
            group1=g1, group2=g2, sites=sites, preserve=my_preserve,
            fisher_z=False, method="tstat",
            n_permutations=20, use_mp=False, acceleration=None, rng=11,
        )
        self.assertTrue(out.inference.preserve_provided)
        self.assertFalse(
            any("group indicator" in f for f in out.flags),
            f"expected no group-indicator flag when preserve= given, got {out.flags}",
        )


# =============================================================================
# Design coupling check (Tier 1.3)
# =============================================================================

class TestDesignCouplingCheck(unittest.TestCase):
    """The coupling check fires only when both preserve and the design
    components are labeled (DataFrames / structured arrays). It is
    silent on raw ndarrays — documented limitation."""

    def test_dataframe_leak_flagged(self):
        import pandas as pd

        Y, age, motion, sites = _make_glm(n=24, seed=20)
        # User preserves age + sex through ComBat, but the GLM design
        # only models age. Sex's variance survives but isn't partialled.
        sex = np.random.RandomState(20).randint(0, 2, size=24).astype(float)
        preserve_df = pd.DataFrame({"age": age, "sex": sex})
        interest_df = pd.DataFrame({"age": age})  # missing sex
        out = analyze(
            Y, interest=interest_df, sites=sites, preserve=preserve_df,
            fisher_z=False, n_permutations=20, use_mp=False,
            acceleration=None, rng=20,
        )
        self.assertTrue(
            any("'sex'" in f and "not represented" in f for f in out.flags),
            f"expected coupling-check flag naming 'sex', got {out.flags}",
        )

    def test_dataframe_full_coverage_no_flag(self):
        import pandas as pd

        Y, age, motion, sites = _make_glm(n=24, seed=21)
        preserve_df = pd.DataFrame({"age": age, "motion": motion})
        interest_df = pd.DataFrame({"age": age})
        confounds_df = pd.DataFrame({"motion": motion})
        out = analyze(
            Y, interest=interest_df, confounds=confounds_df,
            sites=sites, preserve=preserve_df,
            fisher_z=False, n_permutations=20, use_mp=False,
            acceleration=None, rng=21,
        )
        self.assertFalse(
            any("not represented" in f for f in out.flags),
            f"expected no coupling-check flag when all preserve cols are "
            f"in the design, got {out.flags}",
        )

    def test_raw_ndarray_skips_silently(self):
        """No column identity → check silently skipped (documented)."""
        Y, age, motion, sites = _make_glm(n=24, seed=22)
        sex = np.random.RandomState(22).randint(0, 2, size=24).astype(float)
        preserve_arr = np.column_stack([age, sex])  # raw ndarray, no names
        out = analyze(
            Y, interest=age, sites=sites, preserve=preserve_arr,
            fisher_z=False, n_permutations=20, use_mp=False,
            acceleration=None, rng=22,
        )
        # No coupling-check flag, even though sex isn't in the GLM design
        self.assertFalse(
            any("not represented" in f for f in out.flags),
            f"raw ndarray inputs should silently skip the coupling check; "
            f"got {out.flags}",
        )


if __name__ == "__main__":
    unittest.main()
