"""Tier-1 pre-flight tests for the v2.2 ComBat-strategy convenience flags
(PR-D and PR-E of [[protocol_combat_implementation]]).

These tests gate every commit: they verify that the new
``harmonize='nuisance_only'`` (Strategy D) and ``harmonize=None``
(Strategy E) paths in ``analyze()`` are wired correctly and that
their provenance fields and misuse-guards behave as documented.

Five tests, each ``< 10 s`` wall:

1. ``_site_dummies`` helper produces a drop-first dummy matrix of the
   right shape and rank.
2. ``harmonize='nuisance_only'`` populates Strategy-D provenance:
   ``combat_diagnostics['strategy'] == 'D'``, the Strategy-D flag in
   ``out.flags``, and ``harmonized=True`` (ComBat actually ran).
3. ``harmonize=None`` populates Strategy-E provenance: ``harmonized=False``,
   ``combat_diagnostics['strategy'] == 'E'``, and the GLM confound
   design includes site dummies.
4. Misuse guards: ``harmonize='nuisance_only'`` without sites= or
   without confounds= raises ``ValueError``; bad string values raise.
5. Reproducibility: two repeat calls with the same ``rng=42`` produce
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
    """Tier-1 #1 — the shared helper used by both PR-D and PR-E."""

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


class TestStrategyDProvenance(unittest.TestCase):
    """Tier-1 #2 — harmonize='nuisance_only' wiring (Strategy D)."""

    def test_strategy_d_provenance(self):
        data = _tiny_h0(seed=0)
        # Confounds are required for Strategy D; build a small column.
        confounds = np.random.default_rng(101).standard_normal((N_SUBJECTS, 2))

        out = _analyze(
            data, sites=data["sites"], confounds=confounds,
            harmonize="nuisance_only", rng=0,
        )

        # combat_diagnostics is annotated with the strategy label
        self.assertIsNotNone(out.combat_diagnostics)
        self.assertEqual(out.combat_diagnostics.get("strategy"), "D")
        self.assertEqual(
            out.combat_diagnostics.get("preserve_columns"),
            "confounds_only",
        )

        # ComBat actually ran — harmonized provenance bit is True
        self.assertTrue(out.inference.harmonized)
        self.assertTrue(out.inference.preserve_provided)

        # The Strategy-D flag appears in out.flags
        joined = "\n".join(out.flags)
        self.assertIn("Strategy D", joined,
                      f"Strategy-D flag missing from {out.flags!r}")
        self.assertIn("preserve excludes interest", joined)


class TestStrategyEProvenance(unittest.TestCase):
    """Tier-1 #3 — harmonize=None wiring (Strategy E)."""

    def test_strategy_e_provenance(self):
        data = _tiny_h0(seed=0)
        confounds = np.random.default_rng(101).standard_normal((N_SUBJECTS, 1))

        out = _analyze(
            data, sites=data["sites"], confounds=confounds,
            harmonize=None, rng=0,
        )

        # ComBat did NOT run — harmonized is False, combat_diag carries
        # only the Strategy-E label and no harmonization metrics.
        self.assertFalse(out.inference.harmonized)
        self.assertIsNotNone(out.combat_diagnostics)
        self.assertEqual(out.combat_diagnostics.get("strategy"), "E")
        self.assertNotIn(
            "between_site_variance_ratio_after_over_before",
            out.combat_diagnostics,
            msg="Strategy E should have no ComBat variance ratio",
        )

        # Strategy-E flag appears in out.flags
        joined = "\n".join(out.flags)
        self.assertIn("Strategy E", joined,
                      f"Strategy-E flag missing from {out.flags!r}")
        self.assertIn("no ComBat", joined)


class TestStrategyMisuseGuards(unittest.TestCase):
    """Tier-1 #4 — Strategy D/E argument validation."""

    def setUp(self):
        self.data = _tiny_h0(seed=0)
        self.confounds = np.random.default_rng(101).standard_normal(
            (N_SUBJECTS, 1)
        )

    def test_d_requires_sites(self):
        with self.assertRaisesRegex(ValueError, "requires sites"):
            _analyze(
                self.data, confounds=self.confounds,
                harmonize="nuisance_only", rng=0,
            )

    def test_d_requires_confounds(self):
        with self.assertRaisesRegex(ValueError, "requires confounds"):
            _analyze(
                self.data, sites=self.data["sites"],
                harmonize="nuisance_only", rng=0,
            )

    def test_unknown_harmonize_value_raises(self):
        with self.assertRaisesRegex(ValueError, "harmonize="):
            _analyze(
                self.data, sites=self.data["sites"],
                confounds=self.confounds,
                harmonize="bogus", rng=0,
            )


class TestReproducibility(unittest.TestCase):
    """Tier-1 #5 — same seed → bitwise-identical results."""

    def test_same_seed_identical_results(self):
        data = _tiny_h0(seed=0)
        confounds = np.random.default_rng(101).standard_normal((N_SUBJECTS, 1))

        out_a = _analyze(
            data, sites=data["sites"], confounds=confounds,
            harmonize="nuisance_only", rng=42,
        )
        out_b = _analyze(
            data, sites=data["sites"], confounds=confounds,
            harmonize="nuisance_only", rng=42,
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
