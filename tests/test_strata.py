"""Tests for the v2.1 ``strata=`` argument (plan PR-3): within-block
exchangeability for the permutation engines, plus the auto-strata
plumbing in ``analyze()``.

Three groups:

1. **Unit tests** on the helpers (``_stratified_perm``,
   ``_stratified_choice_n1``) — fast, no external dependencies.
2. **Engine regression tests** — sign-flip paths are unchanged
   when strata are passed; two-sample/F-L paths honour per-stratum
   group totals; analyze() flags strata auto-setting.
"""
from __future__ import annotations

import unittest
import warnings

import numpy as np

from conninfpy import analyze, compute_p_val, compute_p_val_glm
from conninfpy.pairwise_stats import (
    _encode_strata,
    _stratified_choice_n1,
    _stratified_perm,
)


# =============================================================================
# Unit tests on the helpers
# =============================================================================

class TestStratifiedPerm(unittest.TestCase):

    def test_within_stratum_swaps_only(self):
        """Permuted indices stay within their original stratum."""
        rng = np.random.RandomState(0)
        # Three strata of sizes 5, 7, 4
        strata = np.array([0] * 5 + [1] * 7 + [2] * 4)
        for _ in range(50):
            perm = _stratified_perm(strata, rng)
            # For every i, the original stratum of i equals the stratum
            # of perm[i] — equivalently, strata[perm] == strata.
            np.testing.assert_array_equal(strata[perm], strata)

    def test_singleton_stratum_is_fixed(self):
        rng = np.random.RandomState(1)
        strata = np.array([0, 1, 1, 1, 2])  # stratum 0 and 2 are singletons
        perm = _stratified_perm(strata, rng)
        self.assertEqual(perm[0], 0)
        self.assertEqual(perm[4], 4)

    def test_returns_a_valid_permutation(self):
        rng = np.random.RandomState(2)
        strata = np.array([0] * 4 + [1] * 4)
        perm = _stratified_perm(strata, rng)
        # Bijection on [0, n)
        self.assertEqual(sorted(perm.tolist()), list(range(strata.size)))


class TestStratifiedChoiceN1(unittest.TestCase):

    def test_per_stratum_group_totals_held_fixed(self):
        rng = np.random.RandomState(3)
        # 3 strata, group-1 counts {2, 3, 1}
        strata = np.array([0] * 5 + [1] * 6 + [2] * 4)
        n1_per_stratum = np.array([2, 3, 1])
        for _ in range(50):
            g = _stratified_choice_n1(strata, n1_per_stratum, rng)
            for s, expected in enumerate(n1_per_stratum):
                self.assertEqual(
                    int(g[strata == s].sum()), int(expected),
                    f"per-stratum group-1 count drifted in stratum {s}",
                )


class TestEncodeStrata(unittest.TestCase):

    def test_string_labels_mapped_to_contiguous_ints(self):
        codes = _encode_strata(np.array(["NYU", "UM", "NYU", "Pitt", "UM"]))
        # Output is 0-based, contiguous, and respects sorted label order
        np.testing.assert_array_equal(codes, np.array([0, 2, 0, 1, 2]))


# =============================================================================
# Engine regression tests
# =============================================================================

class TestSignFlipUnchangedByStrata(unittest.TestCase):
    """Sign-flip paths (paired / one-sample) are stratum-invariant by
    construction. Passing strata= should not perturb the result for any
    seed."""

    def test_paired_path_unchanged(self):
        rng = np.random.RandomState(7)
        n, N = 20, 6
        Y1 = rng.randn(n, N, N); Y1 = (Y1 + Y1.transpose(0, 2, 1)) / 2
        Y2 = rng.randn(n, N, N); Y2 = (Y2 + Y2.transpose(0, 2, 1)) / 2
        for arr in (Y1, Y2):
            for i in range(n):
                np.fill_diagonal(arr[i], 0.0)
        sites = np.array([0] * 10 + [1] * 10)

        r_nostrata = compute_p_val(
            Y1, Y2, test_type="paired", method="tstat",
            n_permutations=50, use_mp=False, rng=42,
        )
        r_strata = compute_p_val(
            Y1, Y2, test_type="paired", method="tstat",
            n_permutations=50, use_mp=False, rng=42, strata=sites,
        )
        np.testing.assert_allclose(r_strata.positive, r_nostrata.positive)
        np.testing.assert_allclose(r_strata.negative, r_nostrata.negative)
        # But the provenance flag is set on the strata-passing call
        self.assertTrue(r_strata.strata_provided)
        self.assertFalse(r_nostrata.strata_provided)


class TestTwoSampleStratified(unittest.TestCase):
    """Two-sample fast path: the per-stratum group totals must be held
    fixed across permutations. Cross-check by running many perms and
    asserting the implicit group assignments respect site margins."""

    def test_perms_respect_site_margins(self):
        # Two sites: site 0 has 8 g1 + 4 g2, site 1 has 2 g1 + 6 g2.
        # The non-stratified path would routinely move g1 across sites.
        rng = np.random.RandomState(11)
        n1, n2, N = 10, 10, 4
        Y1 = rng.randn(n1, N, N); Y1 = (Y1 + Y1.transpose(0, 2, 1)) / 2
        Y2 = rng.randn(n2, N, N); Y2 = (Y2 + Y2.transpose(0, 2, 1)) / 2
        for arr in (Y1, Y2):
            for i in range(arr.shape[0]):
                np.fill_diagonal(arr[i], 0.0)
        # Sites concatenated in the order used by compute_p_val: g1 then g2
        sites = np.array([0] * 8 + [1] * 2 + [0] * 4 + [1] * 6)
        strata = _encode_strata(sites)
        n1_per_stratum = np.bincount(strata[:n1], minlength=2)
        self.assertEqual(n1_per_stratum.tolist(), [8, 2])

        # Direct margin verification on the choice helper
        for seed in range(30):
            g = _stratified_choice_n1(
                strata, n1_per_stratum, np.random.RandomState(seed),
            )
            for s in range(2):
                self.assertEqual(
                    int(g[strata == s].sum()), int(n1_per_stratum[s]),
                )


class TestAnalyzeAutoStrata(unittest.TestCase):
    """analyze() auto-sets strata=sites and emits an explanatory flag."""

    def _make_glm(self, n=24, N=5, seed=0):
        rng = np.random.RandomState(seed)
        Y = rng.randn(n, N, N); Y = (Y + Y.transpose(0, 2, 1)) / 2
        for i in range(n):
            np.fill_diagonal(Y[i], 0.0)
        age = rng.randn(n)
        return Y, age

    def test_sites_triggers_strata_flag(self):
        Y, age = self._make_glm(seed=20)
        sites = np.array([0] * 12 + [1] * 12)
        out = analyze(
            Y, interest=age, sites=sites, fisher_z=False,
            n_permutations=20, use_mp=False, acceleration=None, rng=20,
        )
        self.assertTrue(out.inference.strata_provided)
        self.assertTrue(
            any("strata= auto-set" in f for f in out.flags),
            f"expected auto-strata flag, got: {out.flags}",
        )

    def test_explicit_strata_overrides_sites(self):
        Y, age = self._make_glm(seed=21)
        sites = np.array([0] * 12 + [1] * 12)
        custom = np.array([0] * 6 + [1] * 6 + [2] * 6 + [3] * 6)
        out = analyze(
            Y, interest=age, sites=sites, fisher_z=False,
            n_permutations=20, use_mp=False, acceleration=None, rng=21,
            strata=custom,
        )
        # Explicit strata wins — no auto-set flag
        self.assertTrue(out.inference.strata_provided)
        self.assertFalse(
            any("auto-set" in f for f in out.flags),
            f"expected no auto-set flag when strata= given explicitly, "
            f"got: {out.flags}",
        )

    def test_no_sites_no_strata(self):
        Y, age = self._make_glm(seed=22)
        out = analyze(
            Y, interest=age, fisher_z=False,
            n_permutations=20, use_mp=False, acceleration=None, rng=22,
        )
        self.assertFalse(out.inference.strata_provided)


# =============================================================================
if __name__ == "__main__":
    unittest.main()
