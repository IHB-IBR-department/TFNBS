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

from conninfpy import analyze, AnalyzeResult


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
    """`harmonize='auto'` dispatches to combat_site_dummies_glm when
    GLM+sites+confounds are all present, to site_dummies_glm when GLM+sites without
    confounds, and to no harmonization when sites is absent."""

    def test_auto_with_sites_and_confounds_resolves_to_combat_site_dummies_glm(self):
        Y, age, motion, sites = _make_glm(seed=2)
        out = analyze(
            Y, interest=age, confounds=motion, sites=sites,
            fisher_z=False, n_permutations=20, use_mp=False,
            acceleration=None, rng=2,
        )
        self.assertTrue(out.inference.harmonized)
        self.assertEqual(
            out.inference.combat_diagnostics["strategy"],
            "combat_site_dummies_glm",
        )
        self.assertEqual(out.inference.combat_diagnostics["legacy_strategy"], "D")
        self.assertIn(
            "between_site_variance_ratio_after_over_before",
            out.inference.combat_diagnostics,
        )
        self.assertTrue(
            any("combat_site_dummies_glm: preserve excludes interest" in f
                for f in out.flags),
            f"expected combat_site_dummies_glm preserve flag, got {out.flags}",
        )

    def test_auto_with_sites_no_confounds_resolves_to_site_dummies_glm(self):
        Y, age, _, sites = _make_glm(seed=1)
        out = analyze(
            Y, interest=age, sites=sites, fisher_z=False,
            n_permutations=20, use_mp=False, acceleration=None, rng=1,
        )
        # No confounds → no ComBat; site dummies in GLM.
        self.assertFalse(out.inference.harmonized)
        self.assertEqual(
            out.inference.combat_diagnostics["strategy"],
            "site_dummies_glm",
        )
        self.assertEqual(out.inference.combat_diagnostics["legacy_strategy"], "E")
        self.assertTrue(out.inference.strata_provided)

    def test_explicit_preserve_overridden_under_combat_site_dummies_glm(self):
        Y, age, motion, sites = _make_glm(seed=3)
        # Under ComBat strategies, an explicit preserve= is replaced by
        # `confounds` (the recipe deliberately excludes interest).
        out = analyze(
            Y, interest=age, confounds=motion, sites=sites,
            preserve=age[:, np.newaxis],
            fisher_z=False, n_permutations=20, use_mp=False,
            acceleration=None, rng=3,
        )
        self.assertEqual(
            out.inference.combat_diagnostics["strategy"],
            "combat_site_dummies_glm",
        )
        self.assertTrue(
            any("preserve= overridden by combat_site_dummies_glm" in f
                for f in out.flags),
            f"expected combat_site_dummies_glm override flag, got {out.flags}",
        )

    def test_explicit_combat_only_runs_without_site_dummies(self):
        Y, age, motion, sites = _make_glm(seed=5)
        out = analyze(
            Y, interest=age, confounds=motion, sites=sites,
            harmonize="combat_only",
            fisher_z=False, n_permutations=20, use_mp=False,
            acceleration=None, rng=5,
        )
        self.assertTrue(out.inference.harmonized)
        self.assertEqual(
            out.inference.combat_diagnostics["strategy"],
            "combat_only",
        )
        self.assertNotIn("legacy_strategy", out.inference.combat_diagnostics)
        self.assertFalse(
            any("site dummies appended" in f for f in out.flags),
            f"combat_only should not append site dummies, got {out.flags}",
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


class TestInterestSinglePredictor(unittest.TestCase):
    """`interest=` must be a single predictor; multi-column arrays or lists
    of regressors raise with a pointer to the multi-predictor tools."""

    def test_2d_multicolumn_interest_raises(self):
        Y, age, motion, _ = _make_glm(seed=11)
        bad = np.column_stack([age, motion])          # (n, 2)
        with self.assertRaises(ValueError) as cm:
            analyze(Y, interest=bad, use_mp=False)
        self.assertIn("single predictor", str(cm.exception))

    def test_list_of_regressors_raises(self):
        Y, age, motion, _ = _make_glm(seed=12)
        with self.assertRaises(ValueError):
            analyze(Y, interest=[age, motion], use_mp=False)   # -> (2, n)

    def test_column_vector_interest_is_accepted(self):
        Y, age, motion, _ = _make_glm(seed=13)
        out = analyze(
            Y, interest=age[:, None], confounds=motion,       # (n, 1) is fine
            fisher_z=False, method="tstat", n_permutations=20,
            use_mp=False, acceleration=None, rng=13,
        )
        self.assertIn("positive", out.inference)


class TestMultiPredictorGLM(unittest.TestCase):
    """A dict ``interest={'name': vec}`` routes to compute_p_val_glm_multi
    and returns one AnalyzeResult per predictor, sharing ComBat / flags."""

    def _data(self, seed=20, n=30, N=6):
        rng = np.random.RandomState(seed)
        Y = np.tanh(rng.randn(n, N, N) * 0.3)         # in (-1, 1) for fisher_z
        Y = (Y + Y.transpose(0, 2, 1)) / 2
        for i in range(n):
            np.fill_diagonal(Y[i], 0.0)
        age = rng.randn(n)
        sex = (rng.rand(n) < 0.5).astype(float)
        motion = rng.randn(n)
        site = np.tile(np.arange(3), n // 3)
        return Y, age, sex, motion, site

    def test_dict_interest_returns_dict_of_results(self):
        Y, age, sex, motion, _ = self._data()
        out = analyze(
            Y, interest={'age': age, 'sex': sex}, confounds=motion,
            method='tstat', n_permutations=50,
            use_mp=False, acceleration=None, rng=1,
        )
        self.assertIsInstance(out, dict)
        self.assertEqual(sorted(out), ['age', 'sex'])
        for k in out:
            self.assertIn('positive', out[k].inference)
            self.assertIn('negative', out[k].inference)

    def test_single_predictor_still_returns_single_result(self):
        Y, age, _, motion, _ = self._data()
        out = analyze(
            Y, interest=age, confounds=motion, method='tstat',
            n_permutations=50, use_mp=False, acceleration=None, rng=1,
        )
        self.assertIsInstance(out, AnalyzeResult)

    def test_dict_with_sites_runs_combat_and_shares_diagnostics(self):
        Y, age, sex, motion, site = self._data()
        out = analyze(
            Y, interest={'age': age, 'sex': sex}, confounds=motion,
            sites=site, harmonize='combat_site_dummies_glm',
            method='tstat', n_permutations=50,
            use_mp=False, acceleration=None, rng=2,
        )
        for k in out:
            self.assertTrue(out[k].inference.harmonized)
            self.assertEqual(
                out[k].combat_diagnostics.get('strategy'),
                'combat_site_dummies_glm',
            )
            self.assertEqual(out[k].combat_diagnostics.get('legacy_strategy'), 'D')
            self.assertTrue(out[k].flags)        # shared flags copied to each

    def test_dict_interest_supports_eh_grid(self):
        Y, age, sex, motion, _ = self._data()
        out = analyze(
            Y, interest={'age': age, 'sex': sex}, confounds=motion,
            method='tfnbs', e=[0.4, 0.5], h=[3.0, 2.0], n=10,
            n_permutations=40, use_mp=False, acceleration=None, rng=3,
        )
        self.assertTrue(out['age'].inference.is_grid)
        self.assertEqual(out['age'].inference.positive.shape[-1], 2)

    def test_empty_dict_raises(self):
        Y, *_ = self._data()
        with self.assertRaises(ValueError):
            analyze(Y, interest={}, use_mp=False)

    def test_2d_dict_value_raises(self):
        Y, age, sex, _, _ = self._data()
        with self.assertRaises(ValueError):
            analyze(Y, interest={'bad': np.column_stack([age, sex])},
                    use_mp=False)


def _make_paired(n=20, N=6, bump=0.3, seed=0):
    """Paired conditions with a group2 > group1 signal on two edges, plus a
    condition-varying confound. A shared per-subject ``base`` cancels in the
    within-subject difference; independent condition noise gives ``Δ_Y`` real
    residual variance so the paired t / intercept GLM are well-posed (a
    perfectly deterministic difference would make the t-stat degenerate)."""
    rng = np.random.RandomState(seed)

    def _sym_noise(scale):
        a = rng.randn(n, N, N) * scale
        a = (a + a.transpose(0, 2, 1)) / 2
        for i in range(n):
            np.fill_diagonal(a[i], 0.0)
        return a

    base = _sym_noise(0.1)              # shared subject structure (cancels)
    signal = np.zeros((N, N))
    signal[1, 4] = signal[4, 1] = bump
    signal[2, 5] = signal[5, 2] = bump
    group1 = base + _sym_noise(0.05)
    group2 = base + _sym_noise(0.05) + signal
    c1 = rng.randn(n)
    c2 = c1 + rng.randn(n) * 0.3
    return group1, group2, c1, c2


# =============================================================================
# Repeated-measures GLM (paired + condition-varying confounds)
# =============================================================================

class TestRepeatedMeasuresGLM(unittest.TestCase):
    """`test_type='paired'` + `confounds_group1/2` routes to the paired-
    difference GLM while keeping the `positive = group2 > group1`
    orientation of the no-confound paired path."""

    def test_paired_glm_runs_and_returns_canonical_keys(self):
        g1, g2, c1, c2 = _make_paired(seed=1)
        out = analyze(
            group1=g1, group2=g2, test_type="paired",
            confounds_group1=c1, confounds_group2=c2,
            method="tstat", n_permutations=100,
            use_mp=False, acceleration=None, rng=1,
        )
        self.assertIn("positive", out.inference)
        self.assertIn("negative", out.inference)
        self.assertFalse(out.inference.harmonized)

    def test_orientation_matches_no_confound_paired(self):
        # The orientation contract: a group2 > group1 effect is a positive
        # signed statistic and lands in the `positive` tail on BOTH paths.
        # (Absolute significance is the underlying function's concern; the
        # GLM intercept test trades power when one edge dominates the
        # max-stat null. Here we pin the orientation, not the power.)
        g1, g2, c1, c2 = _make_paired(seed=2)
        common = dict(test_type="paired", method="tstat", n_permutations=200,
                      use_mp=False, acceleration=None, rng=3)
        out_nc = analyze(group1=g1, group2=g2, **common)
        out_rm = analyze(group1=g1, group2=g2,
                         confounds_group1=c1, confounds_group2=c2, **common)
        for out in (out_nc, out_rm):
            r = out.inference
            self.assertGreater(r.stat_signed[1, 4], 0.0)      # group2 > group1
            self.assertLessEqual(r["positive"][1, 4], r["negative"][1, 4])
        # The no-confound paired path (sign-flip null) recovers the signal.
        self.assertLess(out_nc.inference["positive"][1, 4], 0.05)

    def test_only_one_confound_raises(self):
        g1, g2, c1, _ = _make_paired(seed=3)
        with self.assertRaises(ValueError):
            analyze(group1=g1, group2=g2, test_type="paired",
                    confounds_group1=c1, use_mp=False)

    def test_condition_confounds_require_paired(self):
        g1, g2, c1, c2 = _make_paired(seed=4)
        with self.assertRaises(ValueError):
            analyze(group1=g1, group2=g2, test_type="two-sample",
                    confounds_group1=c1, confounds_group2=c2, use_mp=False)

    def test_bare_confounds_with_ttest_raises(self):
        g1, g2, c1, _ = _make_paired(seed=5)
        with self.assertRaises(ValueError):
            analyze(group1=g1, group2=g2, test_type="paired",
                    confounds=c1, use_mp=False)

    def test_paired_with_sites_skips_combat_with_note(self):
        g1, g2, c1, c2 = _make_paired(seed=6)
        sites = np.array([0, 1] * (g1.shape[0] // 2))
        out = analyze(
            group1=g1, group2=g2, test_type="paired",
            confounds_group1=c1, confounds_group2=c2, sites=sites,
            method="tstat", n_permutations=50,
            use_mp=False, acceleration=None, rng=6,
        )
        self.assertFalse(out.inference.harmonized)
        self.assertTrue(
            any("paired + sites" in f for f in out.flags),
            f"expected paired-sites note, got {out.flags}",
        )
        # The two-sample demotion advice must NOT fire for a paired design.
        self.assertFalse(any("two-sample + sites" in f for f in out.flags))


if __name__ == "__main__":
    unittest.main()
