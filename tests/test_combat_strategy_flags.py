"""Tier-1 pre-flight tests for the v2.2 ComBat-strategy convenience flags.

These tests gate every commit: they verify that the three paper-described
multi-site arms (``combat_only``, ``combat_site_dummies_glm``, and ``site_dummies_glm``) are
wired correctly and that their provenance fields and misuse-guards behave as
documented.

Six checks, each ``< 10 s`` wall:

1. ``_site_dummies`` helper produces a drop-first dummy matrix of the
   right shape and rank.
2. ``harmonize='combat_only'`` populates ComBat provenance without appending
   site dummies to the GLM nuisance.
3. ``harmonize='combat_site_dummies_glm'`` populates ComBat provenance and appends
   site dummies to the GLM nuisance.
4. ``harmonize=None`` populates site-GLM provenance:
   ``harmonized=False``, ``combat_diagnostics['strategy'] ==
   'site_dummies_glm'``, and the GLM confound design includes site dummies.
5. Misuse guards: ComBat strategies without sites= or without confounds=
   raise ``ValueError``; bad string values raise.
6. Reproducibility: two repeat calls with the same ``rng=42`` produce
   bitwise-identical p-maps and stat maps.

The dataset is a tiny ``generate_multisite_glm_dataset`` cell
(N=24, K=3, Schaefer-20) so the suite stays well under a minute total.
"""
from __future__ import annotations

import unittest
import warnings

import numpy as np

from conninfpy import analyze, generate_multisite_glm_dataset
from conninfpy._analyze import _site_dummies


# Tiny problem size — Tier-1 must stay fast.
N_SUBJECTS = 24
N_ROIS = 20
N_SITES = 3
N_PERM = 100  # GPD acceleration; small but enough to exercise the pipeline


def _tiny_h0(seed: int = 0):
    return generate_multisite_glm_dataset(
        n_subjects=N_SUBJECTS, N=N_ROIS, n_sites=N_SITES,
        effect_size=0.0, site_shift_sigma=0.2,
        corr_site_interest=0.0, seed=seed,
    )


def _analyze(data, **kwargs):
    """Single ``analyze`` call with the noisy warnings filtered."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return analyze(
            data["Y"], interest=data["interest"],
            fisher_z=False, method="tfnbs",
            n_permutations=N_PERM, acceleration="gpd",
            use_mp=False,
            **kwargs,
        )


class TestSiteDummiesHelper(unittest.TestCase):
    """Tier-1 #1 -- shared helper for both site-aware GLM strategies."""

    def test_drop_first_shape_and_rank(self):
        # 4 sites × 3 subjects each → 12 rows, 3 dummy columns, rank 3
        sites = np.repeat(np.arange(4), 3)
        dummies = _site_dummies(sites)

        self.assertEqual(dummies.shape, (12, 3))
        self.assertEqual(np.linalg.matrix_rank(dummies), 3)
        # Drop-first → reference category is sites[0]; its rows are all-zero.
        ref_rows = dummies[sites == 0]
        self.assertTrue((ref_rows == 0).all())
        # Other categories: each gets a column of ones for its rows.
        for k in range(1, 4):
            col = dummies[:, k - 1]
            self.assertTrue((col[sites == k] == 1.0).all())
            self.assertTrue((col[sites != k] == 0.0).all())

    def test_keep_all_when_drop_first_false(self):
        sites = np.repeat(np.arange(3), 2)
        dummies = _site_dummies(sites, drop_first=False)
        self.assertEqual(dummies.shape, (6, 3))
        self.assertEqual(np.linalg.matrix_rank(dummies), 3)

    def test_single_site_returns_empty(self):
        sites = np.zeros(10, dtype=int)
        dummies = _site_dummies(sites)
        self.assertEqual(dummies.shape, (10, 0))


class TestCombatOnlyProvenance(unittest.TestCase):
    """Tier-1 #2 -- harmonize='combat_only' wiring."""

    def test_combat_only_provenance(self):
        data = _tiny_h0(seed=0)
        confounds = np.random.default_rng(101).standard_normal((N_SUBJECTS, 2))

        out = _analyze(
            data, sites=data["sites"], confounds=confounds,
            harmonize="combat_only", rng=0,
        )

        self.assertIsNotNone(out.combat_diagnostics)
        self.assertEqual(out.combat_diagnostics.get("strategy"), "combat_only")
        self.assertNotIn("legacy_strategy", out.combat_diagnostics)
        self.assertEqual(
            out.combat_diagnostics.get("preserve_columns"),
            "confounds_only",
        )
        self.assertTrue(out.inference.harmonized)
        self.assertTrue(out.inference.preserve_provided)

        joined = "\n".join(out.flags)
        self.assertIn("combat_only", joined)
        self.assertIn("preserve excludes interest", joined)
        self.assertNotIn("site dummies appended", joined)


class TestCombatSiteDummiesGlmProvenance(unittest.TestCase):
    """Tier-1 #3 -- harmonize='combat_site_dummies_glm' wiring."""

    def test_combat_site_dummies_glm_provenance(self):
        data = _tiny_h0(seed=0)
        # Confounds are required for the ComBat arms.
        confounds = np.random.default_rng(101).standard_normal((N_SUBJECTS, 2))

        out = _analyze(
            data, sites=data["sites"], confounds=confounds,
            harmonize="combat_site_dummies_glm", rng=0,
        )

        # combat_diagnostics is annotated with the strategy label
        self.assertIsNotNone(out.combat_diagnostics)
        self.assertEqual(
            out.combat_diagnostics.get("strategy"),
            "combat_site_dummies_glm",
        )
        self.assertEqual(out.combat_diagnostics.get("legacy_strategy"), "D")
        self.assertEqual(
            out.combat_diagnostics.get("preserve_columns"),
            "confounds_only",
        )

        # ComBat actually ran — harmonized provenance bit is True
        self.assertTrue(out.inference.harmonized)
        self.assertTrue(out.inference.preserve_provided)

        # The combat + site-GLM flags appear in out.flags.
        joined = "\n".join(out.flags)
        self.assertIn("combat_site_dummies_glm", joined,
                      f"combat_site_dummies_glm flag missing from {out.flags!r}")
        self.assertIn("preserve excludes interest", joined)
        self.assertIn("site dummies appended", joined)


class TestSiteDummiesGlmProvenance(unittest.TestCase):
    """Tier-1 #4 -- harmonize=None wiring (site GLM)."""

    def test_site_dummies_glm_provenance(self):
        data = _tiny_h0(seed=0)
        confounds = np.random.default_rng(101).standard_normal((N_SUBJECTS, 1))

        out = _analyze(
            data, sites=data["sites"], confounds=confounds,
            harmonize=None, rng=0,
        )

        # ComBat did NOT run — harmonized is False, combat_diag carries
        # only the site-GLM label and no harmonization metrics.
        self.assertFalse(out.inference.harmonized)
        self.assertIsNotNone(out.combat_diagnostics)
        self.assertEqual(
            out.combat_diagnostics.get("strategy"),
            "site_dummies_glm",
        )
        self.assertEqual(out.combat_diagnostics.get("legacy_strategy"), "E")
        self.assertNotIn(
            "between_site_variance_ratio_after_over_before",
            out.combat_diagnostics,
            msg="site_dummies_glm should have no ComBat variance ratio",
        )

        # Site-GLM flag appears in out.flags.
        joined = "\n".join(out.flags)
        self.assertIn("site_dummies_glm", joined,
                      f"site_dummies_glm flag missing from {out.flags!r}")
        self.assertIn("no ComBat", joined)


class TestStrategyMisuseGuards(unittest.TestCase):
    """Tier-1 #5 -- strategy argument validation."""

    def setUp(self):
        self.data = _tiny_h0(seed=0)
        self.confounds = np.random.default_rng(101).standard_normal(
            (N_SUBJECTS, 1)
        )

    def test_combat_strategy_requires_sites(self):
        with self.assertRaisesRegex(ValueError, "requires sites"):
            _analyze(
                self.data, confounds=self.confounds,
                harmonize="combat_site_dummies_glm", rng=0,
            )

    def test_combat_strategy_requires_confounds(self):
        with self.assertRaisesRegex(ValueError, "requires confounds"):
            _analyze(
                self.data, sites=self.data["sites"],
                harmonize="combat_only", rng=0,
            )

    def test_unknown_harmonize_value_raises(self):
        with self.assertRaisesRegex(ValueError, "harmonize="):
            _analyze(
                self.data, sites=self.data["sites"],
                confounds=self.confounds,
                harmonize="bogus", rng=0,
            )


class TestReproducibility(unittest.TestCase):
    """Tier-1 #6 -- same seed -> bitwise-identical results."""

    def test_same_seed_identical_results(self):
        data = _tiny_h0(seed=0)
        confounds = np.random.default_rng(101).standard_normal((N_SUBJECTS, 1))

        out_a = _analyze(
            data, sites=data["sites"], confounds=confounds,
            harmonize="combat_site_dummies_glm", rng=42,
        )
        out_b = _analyze(
            data, sites=data["sites"], confounds=confounds,
            harmonize="combat_site_dummies_glm", rng=42,
        )

        # p-maps identical
        np.testing.assert_array_equal(
            out_a.inference["positive"], out_b.inference["positive"]
        )
        np.testing.assert_array_equal(
            out_a.inference["negative"], out_b.inference["negative"]
        )
        # Stat maps (PR-2) identical
        np.testing.assert_array_equal(
            out_a.inference.stat_positive, out_b.inference.stat_positive
        )
        np.testing.assert_array_equal(
            out_a.inference.stat_negative, out_b.inference.stat_negative
        )


if __name__ == "__main__":
    unittest.main()
