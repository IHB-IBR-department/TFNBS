"""Tier-2 quick — site_dummies_glm calibration smoke at minimal configuration.

One pipeline (site_dummies_glm: single-stage GLM with site as nuisance,
no ComBat) exercised end-to-end at a small-but-realistic cell:

- ``N = 100`` subjects, ``K = 4`` sites, Schaefer-100 (4 950 edges)
- ``σ_site = 0.15`` (ABIDE-typical Fisher-z), ``corr(site, dx) = 0.3``
- ``n_permutations = 200`` with GPD acceleration
- 50 H₀ reps

Wall budget at the time of writing: ~50 × 0.7 s ≈ 1 min. Gated behind
``CONNINFPY_SKIP_SLOW_CI=1`` to keep the per-commit suite fast; runs
locally / on nightly CI / before kicking off the PR-cal overnight.

What this test does NOT do:

- It does **not** make a tight calibration claim. At 50 reps the
  binomial 2σ around α=0.05 is ~0.06; the assertion is the loose
  bound FWER < 0.20, which only rules out catastrophic regressions.
  The full multi-site calibration driver is where the calibrated number
  for the toolbox-paper figure lives.
- It does **not** cover the ComBat arms. Add parallel test files
  for ``combat_only`` / ``combat_site_dummies_glm`` later if the
  PR-cal driver wants a per-strategy timing assertion.

Why site_dummies_glm and not a ComBat arm for the quick test: it has no ComBat
step, so its wall time is the lower bound and a wall-time regression
caught here is a regression in the GLM/permutation core, not in the
harmonization layer.
"""
from __future__ import annotations

import os
import time
import unittest
import warnings

import numpy as np

from conninfpy import analyze, generate_multisite_glm_dataset


SKIP_SLOW = os.environ.get("CONNINFPY_SKIP_SLOW_CI") == "1"
SKIP_SLOW_MSG = "Skipped: CONNINFPY_SKIP_SLOW_CI=1 set."


# ABIDE-realistic small cell. Keep the parameters here in one block so
# the PR-cal driver can copy them verbatim when scaling up.
N_SUBJECTS = 100
N_ROIS = 100                 # Schaefer-100; 4 950 edges
N_SITES = 4
N_PERM = 200
ACCEL = "gpd"
SIGMA_SITE = 0.15            # Fisher-z, Yu 2018 ABIDE-typical
CORR_SITE_INTEREST = 0.3     # ABIDE-typical group/site imbalance
ALPHA = 0.05
N_REPS = 50                  # binomial 2σ ≈ 0.06 around α=0.05

# Wall-time regression guard. 0.7 s/call observed during writing
# at the listed parameters; ~7× headroom for CI noise.
PER_REP_WALL_BUDGET_S = 5.0

# Loose FWER bound: rules out catastrophic regressions only.
# A future PR-cal nightly will report the actual point estimate with
# ~5× tighter MC error.
FWER_LOOSE_BOUND = 0.20


def _h0_cell(seed: int):
    return generate_multisite_glm_dataset(
        n_subjects=N_SUBJECTS, N=N_ROIS, n_sites=N_SITES,
        effect_size=0.0,
        site_shift_sigma=SIGMA_SITE,
        corr_site_interest=CORR_SITE_INTEREST,
        seed=seed,
    )


def _site_dummies_glm_min_pos_p(data, rng_seed: int) -> tuple[float, float]:
    """Run one Strategy-E ``analyze`` call; return (min_pos_p, wall_s)."""
    # Independent RNG stream for confounds — generator uses
    # default_rng(seed) internally, so picking 10_000 + seed avoids
    # the same-state collinearity we hit in
    # tests/test_combat_strategy_flags.py.
    confounds = np.random.default_rng(10_000 + rng_seed).standard_normal(
        (N_SUBJECTS, 1)
    )
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = analyze(
            data["Y"], interest=data["interest"], sites=data["sites"],
            confounds=confounds,
            harmonize="site_dummies_glm",      # site dummies in GLM; no ComBat
            fisher_z=False, method="tfnbs",
            n_permutations=N_PERM, acceleration=ACCEL,
            use_mp=False, rng=rng_seed,
        )
    wall = time.time() - t0
    iu = np.triu_indices(N_ROIS, k=1)
    return float(out.inference["positive"][iu].min()), wall


@unittest.skipIf(SKIP_SLOW, SKIP_SLOW_MSG)
class TestSiteDummiesGlmQuickCalibration(unittest.TestCase):
    """Single-strategy, minimal-configuration H₀ calibration smoke."""

    def test_site_dummies_glm_h0_fwer_loose_bound(self):
        p_min = np.empty(N_REPS, dtype=np.float64)
        wall = np.empty(N_REPS, dtype=np.float64)

        for rep in range(N_REPS):
            data = _h0_cell(seed=2_000 + rep)
            p_min[rep], wall[rep] = _site_dummies_glm_min_pos_p(data, rng_seed=rep)

        # Empirical FWER point estimate + binomial 95 % CI.
        fwer = float(np.mean(p_min <= ALPHA))
        se = float(np.sqrt(ALPHA * (1 - ALPHA) / N_REPS))
        ci = (max(0.0, fwer - 1.96 * se), min(1.0, fwer + 1.96 * se))

        # Diagnostic line — printed even on pass so the operator sees
        # the actual number before running the full PR-cal overnight.
        msg = (
            f"\n[site_dummies_glm quick calibration]\n"
            f"  Cell: N={N_SUBJECTS}, K={N_SITES}, Schaefer-{N_ROIS}, "
            f"σ_site={SIGMA_SITE}, corr(site,dx)={CORR_SITE_INTEREST}\n"
            f"  Permutations: {N_PERM} ({ACCEL}); H₀ reps: {N_REPS}\n"
            f"  Empirical FWER at α={ALPHA}: {fwer:.3f}  "
            f"(95% CI [{ci[0]:.3f}, {ci[1]:.3f}], binomial SE={se:.3f})\n"
            f"  Per-rep wall: median={np.median(wall):.2f} s, "
            f"max={wall.max():.2f} s, total={wall.sum():.1f} s"
        )
        print(msg)

        # ---- Assertions ----

        # (1) All reps produced a valid p-value
        self.assertTrue(
            np.all((p_min >= 0.0) & (p_min <= 1.0)),
            "min p-values out of [0, 1] — broken pipeline",
        )

        # (2) Not all p-values stuck at 1 or 0 (catches a totally broken
        # permutation engine that returns the same answer for every call)
        self.assertGreater(p_min.std(), 0.0,
                           "min p-values identical across reps — broken engine")

        # (3) Loose calibration bound
        self.assertLess(
            fwer, FWER_LOOSE_BOUND,
            msg=(
                f"site_dummies_glm empirical FWER ({fwer:.3f}) exceeds the loose "
                f"bound ({FWER_LOOSE_BOUND}). This is a serious regression "
                f"in the GLM/permutation core — site_dummies_glm should be "
                f"approximately calibrated under H₀. The PR-cal overnight "
                f"will report the tighter number; this test exists to "
                f"catch catastrophic regressions only."
            ),
        )

        # (4) Wall-time regression guard
        self.assertLess(
            wall.max(), PER_REP_WALL_BUDGET_S,
            msg=(
                f"site_dummies_glm per-rep wall ({wall.max():.2f} s) exceeds the "
                f"budget ({PER_REP_WALL_BUDGET_S} s). Likely a regression "
                f"in the GLM precompute or the permutation pool."
            ),
        )


if __name__ == "__main__":
    unittest.main()
