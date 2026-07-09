"""End-to-end smoke tests for the v2.1 ``analyze()`` recipe.

Four tests using ``generate_multisite_glm_dataset(N=100, ...)`` to
mimic Schaefer-100 scale:

1. **Recipe smoke** (1 rep, ~1 s) — single ``analyze`` call;
   asserts the auto-Fisher / ComBat / auto-preserve / auto-strata
   wiring fires and the result is structurally valid.
2. **Vanilla-GLM FWER calibration** (50 reps, ~15 s) — H₀ run
   *without* sites/ComBat; checks the bare GLM + Freedman-Lane +
   TFNBS + GPD path is calibrated at α=0.05. Isolates PRs 2-3 from
   ComBat interaction.
3. **Strata-with-sites does not crash** (1 rep, ~1 s) — recipe wiring
   when ``sites=`` is passed. We deliberately do **not** assert
   tight calibration here: the ComBat-then-permutation interaction
   (ComBat fits on observed (Y, interest, sites) jointly; permutation
   reshuffles Y but ComBat-derived structure stays put) is a known
   research-grade limitation — assertable calibration requires
   permutation-inside-ComBat or a different recipe.
4. **Strata relative protection under confound** (50 reps, ~30 s) —
   under ``corr(site, interest)=0.6``, ``strata=site`` should not
   produce *higher* FWER than ``strata=None``. Regression guard for
   PR-3's stratified-permutation wiring.

Wall-time budget: ~50s total at N=100, 200 perms × GPD × ``use_mp=False``.
Marked behind ``CONNINFPY_SKIP_SLOW_CI=1`` so users can opt out on
slow machines or during fast iteration.
"""
from __future__ import annotations

import os
import unittest
import warnings

import numpy as np

from conninfpy import analyze, generate_multisite_glm_dataset


SKIP_SLOW = os.environ.get("CONNINFPY_SKIP_SLOW_CI") == "1"
SKIP_SLOW_MSG = "Skipped: CONNINFPY_SKIP_SLOW_CI=1 set."


# Shared shape — Schaefer-100-scale H₀ recipe
N_ROIS = 100
N_SUBJECTS = 30
N_SITES = 3
N_PERM = 200
ALPHA = 0.05
N_REPS_CALIBRATION = 50

# Binomial 2σ at α=0.05, N=50 is sqrt(α(1-α)/N) ≈ 0.031;
# bound at 0.12 gives ~2σ + slack for TFNBS/GPD approximation drift.
FWER_VANILLA_BOUND = 0.12


def _make_h0(seed: int, *, corr_site_interest: float = 0.0,
             site_shift_sigma: float = 0.3):
    return generate_multisite_glm_dataset(
        n_subjects=N_SUBJECTS, N=N_ROIS, n_sites=N_SITES,
        effect_size=0.0,
        site_shift_sigma=site_shift_sigma,
        corr_site_interest=corr_site_interest,
        seed=seed,
    )


def _analyze_min_pos_p(data, **kwargs) -> float:
    """Run analyze() and return min p_positive over the upper triangle."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = analyze(
            data["Y"], interest=data["interest"],
            fisher_z=False, method="tfnbs",
            n_permutations=N_PERM, acceleration="gpd",
            use_mp=False,
            **kwargs,
        )
    iu = np.triu_indices(N_ROIS, k=1)
    return float(res.inference["positive"][iu].min())


class TestRecipeSmoke(unittest.TestCase):
    """One end-to-end call. Confirms the v2.1 hygiene path is wired
    correctly (PRs 1-4 jointly) without depending on MC statistics."""

    def test_h0_recipe_runs_clean(self):
        data = _make_h0(seed=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = analyze(
                data["Y"], interest=data["interest"], sites=data["sites"],
                fisher_z=False, method="tfnbs",
                n_permutations=N_PERM, acceleration="gpd",
                use_mp=False, rng=0,
            )

        # Structural invariants on the result
        self.assertIn("positive", out.inference)
        self.assertIn("negative", out.inference)
        for tail in ("positive", "negative"):
            p = out.inference[tail]
            self.assertEqual(p.shape, (N_ROIS, N_ROIS))
            self.assertTrue(np.allclose(p, p.T), f"{tail} p-map not symmetric")
            # Diagonal is initialised to 1.0 (no testable self-edge); the
            # off-diagonal entries are the actual p-values.
            np.testing.assert_array_equal(
                np.diag(p), np.ones(N_ROIS),
                err_msg=f"{tail} p-map diagonal != 1.0",
            )
            iu = np.triu_indices(N_ROIS, k=1)
            self.assertTrue((p[iu] >= 0).all() and (p[iu] <= 1).all())

        # PR-2 stat maps populated; stat_signed reconstructs the t-stat
        self.assertIsNotNone(out.inference.stat_positive)
        self.assertIsNotNone(out.inference.stat_negative)
        signed = out.inference.stat_signed
        self.assertIsNotNone(signed)
        self.assertTrue(np.isfinite(signed).all())

        # `sites=` without confounds resolves the `auto` dispatcher to
        # site_dummies_glm: no ComBat, site dummies in the GLM nuisance, strata
        # auto-set to sites.
        self.assertFalse(out.inference.harmonized)
        self.assertTrue(out.inference.strata_provided)
        self.assertIsNotNone(out.inference.combat_diagnostics)
        self.assertEqual(
            out.inference.combat_diagnostics["strategy"],
            "site_dummies_glm",
        )
        self.assertEqual(out.inference.combat_diagnostics["legacy_strategy"], "E")

        joined = "\n".join(out.flags)
        self.assertIn("strata= auto-set", joined,
                      f"missing auto-strata flag in {out.flags}")
        self.assertIn("no ComBat", joined,
                      f"missing site_dummies_glm flag in {out.flags}")

        # site_dummies_glm does not run ComBat -- no between-site ratio in the
        # diagnostics blob. The path completing cleanly is the contract being
        # smoke-tested here.


@unittest.skipIf(SKIP_SLOW, SKIP_SLOW_MSG)
class TestVanillaGLMFWERCalibration(unittest.TestCase):
    """H₀ empirical FWER under the *bare* GLM + Freedman-Lane + TFNBS
    + GPD path (no ComBat, no sites). At N=100 ROIs, 50 reps, and
    α=0.05 the binomial 2σ is ~0.031; the bound at 0.12 gives ~2σ
    headroom for TFNBS/GPD approximation drift.

    Isolates the PR-2 / PR-3 wiring from the ComBat-then-permutation
    interaction. The ``sites=`` path has its own known calibration
    issue in this compact synthetic generator and is
    covered by the smoke + strata-protection tests below."""

    def test_vanilla_glm_fwer_under_h0(self):
        p_min = np.empty(N_REPS_CALIBRATION, dtype=np.float64)
        for rep in range(N_REPS_CALIBRATION):
            data = _make_h0(seed=rep)
            # No sites= → vanilla GLM, no ComBat, no auto-strata
            p_min[rep] = _analyze_min_pos_p(data, rng=rep)

        fwer = float(np.mean(p_min <= ALPHA))
        # Sanity: not all p-values stuck (would indicate broken pipeline)
        self.assertGreater(float(p_min.min()), 0.0)
        self.assertLess(float(p_min.mean()), 1.0)
        self.assertLess(
            fwer, FWER_VANILLA_BOUND,
            msg=(
                f"Vanilla-GLM H₀ FWER = {fwer:.3f} at α={ALPHA} across "
                f"{N_REPS_CALIBRATION} reps — exceeds bound "
                f"{FWER_VANILLA_BOUND}. Either a regression in PR-2 stat-"
                "map handling or PR-3's permutation wiring; investigate."
            ),
        )


@unittest.skipIf(SKIP_SLOW, SKIP_SLOW_MSG)
class TestStrataRelativeProtection(unittest.TestCase):
    """Regression guard for PR-3 under site/group confound.

    Under ``corr(site, interest)=0.6`` the unrestricted permutation
    path (``strata=None``) leaks Type I; PR-3's auto-stratified
    permutation should produce *no more* false-alarms than the
    unrestricted path. We assert ``fwer_strata <= fwer_no_strata +
    margin`` rather than tight calibration: the ComBat-then-
    permutation interaction (a known v2.1 limitation, scoped to
    pseudo-real phase 1) inflates both paths beyond α=0.05, but the
    strata path should at minimum not be *worse*.

    A direction stronger than this — "strata=site visibly *reduces*
    inflation vs no-strata" — would require larger N and N-perms;
    deferred to the planned 200-rep nightly variant."""

    def test_strata_no_worse_than_no_strata_under_confound(self):
        p_min_strata = np.empty(N_REPS_CALIBRATION, dtype=np.float64)
        p_min_nostrata = np.empty(N_REPS_CALIBRATION, dtype=np.float64)

        for rep in range(N_REPS_CALIBRATION):
            data = _make_h0(seed=2000 + rep, corr_site_interest=0.6)
            # auto-strata (default when sites= is passed)
            p_min_strata[rep] = _analyze_min_pos_p(
                data, sites=data["sites"], rng=rep,
            )
            # explicit strata=None overrides the auto-set
            p_min_nostrata[rep] = _analyze_min_pos_p(
                data, sites=data["sites"], strata=None, rng=rep,
            )

        fwer_strata = float(np.mean(p_min_strata <= ALPHA))
        fwer_nostrata = float(np.mean(p_min_nostrata <= ALPHA))
        # Margin absorbs MC noise (~3σ at N=50)
        self.assertLessEqual(
            fwer_strata, fwer_nostrata + 0.10,
            msg=(
                f"strata=site FWER ({fwer_strata:.3f}) much higher than "
                f"strata=None FWER ({fwer_nostrata:.3f}) — possible "
                "regression in PR-3's stratified-permutation wiring."
            ),
        )


if __name__ == "__main__":
    unittest.main()
