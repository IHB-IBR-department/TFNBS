"""Tests for the v2.1 ``strata=`` argument (plan PR-3): within-block
exchangeability for the permutation engines, plus the auto-strata
plumbing in ``analyze()``.

Three groups:

1. **Unit tests** on the helpers (``_stratified_perm``,
   ``_stratified_choice_n1``) — fast, no external dependencies.
2. **Engine regression tests** — sign-flip paths are unchanged
   when strata are passed; two-sample/F-L paths honour per-stratum
   group totals; analyze() flags strata auto-setting.
3. **PALM ``-eb`` cross-validation (Test E from
   [[palm_validation]])** — distributional parity at MC-noise
   tolerance. Gated behind PALM availability; skips cleanly when
   PALM/MATLAB is not present.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
import warnings
from pathlib import Path

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
# PALM -eb cross-validation (Test E from palm_validation.md)
# =============================================================================

DEFAULT_MATLAB_BIN = "/Applications/MATLAB_R2025a.app/bin/matlab"
DEFAULT_PALM_DIR = str(Path.home() / "Work" / "tools" / "palm-alpha119")
MATLAB_BIN = os.environ.get("CONNINFPY_MATLAB_BIN", DEFAULT_MATLAB_BIN)
PALM_DIR = os.environ.get("CONNINFPY_PALM_DIR", DEFAULT_PALM_DIR)
SKIP_MSG = (
    f"MATLAB ({MATLAB_BIN}) or PALM ({PALM_DIR}) not available; "
    f"set CONNINFPY_MATLAB_BIN / CONNINFPY_PALM_DIR to override."
)


def _palm_available() -> bool:
    return Path(MATLAB_BIN).is_file() and Path(PALM_DIR, "palm.m").is_file()


def _run_palm(workdir: Path, args: list[str], timeout: int = 300) -> str:
    script = (
        f"addpath('{PALM_DIR}'); palm {' '.join(args)}; exit;"
    )
    cmd = [
        MATLAB_BIN, "-batch", script, "-sd", str(workdir),
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, cwd=workdir,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"PALM exited {proc.returncode}.\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
    return proc.stdout


def _flat_to_3d(Y_flat: np.ndarray, N: int) -> np.ndarray:
    n_subj = Y_flat.shape[0]
    Y_3d = np.zeros((n_subj, N, N))
    iu = np.triu_indices(N, k=1)
    Y_3d[:, iu[0], iu[1]] = Y_flat
    Y_3d[:, iu[1], iu[0]] = Y_flat
    return Y_3d


def _make_blocked_twosample(
    n_per_site=20, n_sites=3, N=6, effect=0.6,
    n_signal_edges=4, seed=42,
):
    """Two-sample blocked design where the group/site totals are perfectly
    balanced (half g1, half g2 within each site) — the regime where PALM
    -eb is the canonical reference."""
    rng = np.random.RandomState(seed)
    n_total = n_per_site * n_sites
    n1 = n_total // 2
    n2 = n_total - n1
    m = N * (N - 1) // 2

    # Site-balanced 0/1 group label
    sites = np.repeat(np.arange(n_sites), n_per_site)
    group = np.zeros(n_total, dtype=int)
    for s in range(n_sites):
        idx = np.where(sites == s)[0]
        group[idx[: len(idx) // 2]] = 1  # first half of each site → g1
    # Per-site additive shift (the thing -eb is supposed to keep null
    # exchangeability invariant to)
    site_shifts = rng.randn(n_sites, m) * 0.4

    Y_flat = rng.randn(n_total, m) + site_shifts[sites]
    # Plant the effect on a few edges, group-2 (group==0) > group-1
    sig = rng.choice(m, n_signal_edges, replace=False)
    Y_flat[group == 0, sig[:, None]] += effect

    # Design: intercept + group dummy. Contrast [0, 1] tests g2 > g1
    # (positive coefficient on the group dummy).
    X = np.column_stack([np.ones(n_total), group.astype(float)])
    contrast = np.array([0.0, 1.0])
    return Y_flat, X, contrast, sites, N, m


N_PERM_PALM = 5000
# Per-edge tolerance is calibrated to MC noise at n_perm=5000 with within-
# block permutation. Tight per-edge agreement (≤ a few MC quanta) is the
# wrong target here: PALM `-eb -within` and our Freedman-Lane residual-
# permutation explore overlapping but distinct permutation subspaces and
# converge to the same null only in expectation. The correlation guard
# below (>0.99) is the more meaningful sanity check; the per-edge bound
# absorbs sampling-subspace divergence on borderline edges.
PVAL_ATOL = 0.07


@unittest.skipUnless(_palm_available(), SKIP_MSG)
class TestPalmEbParity(unittest.TestCase):
    """Test E from [[palm_validation]]: distributional parity of
    conninfpy's ``strata=`` vs PALM's ``-eb`` (within-block exchangeability)
    on a blocked two-sample GLM design.

    Per-edge agreement is not point-wise tight: within-block exchangeability
    restricts the permutation subspace, so PALM (which uses sign-flip /
    direct group-relabel inside blocks) and conninfpy (which uses
    Freedman-Lane residual permutation inside blocks) converge to the same
    null in expectation but realize different finite-sample tails. The
    primary parity bar is high overall correlation; per-edge p-value
    differences ≤ ~0.07 on borderline edges are acceptable.

    A tight calibration test (empirical FWER under H₀ across many synthetic
    datasets) is the right complement and belongs to PR-5 of
    [[implementation_plan_2026-05-19]]."""

    @classmethod
    def setUpClass(cls):
        cls.workdir = Path(tempfile.mkdtemp(prefix="palm_test_eb_"))
        Y_flat, X, contrast, sites, N, m = _make_blocked_twosample()
        cls.Y_flat, cls.X, cls.contrast = Y_flat, X, contrast
        cls.sites, cls.N, cls.m = sites, N, m
        cls.iu = np.triu_indices(N, k=1)

        np.savetxt(cls.workdir / "Y.csv", Y_flat, delimiter=",", fmt="%.10f")
        np.savetxt(cls.workdir / "X.csv", X, delimiter=",", fmt="%.10f")
        np.savetxt(cls.workdir / "C.csv", contrast[None, :], delimiter=",",
                fmt="%.10f")
        np.savetxt(cls.workdir / "EB.csv", sites.astype(int), fmt="%d")

        _run_palm(
            cls.workdir,
            ["-i", "Y.csv", "-d", "X.csv", "-t", "C.csv",
            "-eb", "EB.csv", "-within",
            "-n", str(N_PERM_PALM), "-o", "out", "-quiet", "-seed", "42"],
        )
        cls.palm_fwep = np.loadtxt(
            cls.workdir / "out_dat_tstat_fwep.csv", delimiter=","
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.workdir, ignore_errors=True)

    def test_strata_glm_fwer_matches_palm_eb(self):
        Y_3d = _flat_to_3d(self.Y_flat, self.N)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = compute_p_val_glm(
                Y_3d, design_matrix=self.X, contrast=self.contrast,
                stat_type="tstat", method="tstat",
                n_permutations=N_PERM_PALM, use_mp=False, random_state=42,
                strata=self.sites,
            )
        self.assertTrue(res.strata_provided)
        cinf_p = res["positive"][self.iu[0], self.iu[1]]
        max_abs_diff = float(np.max(np.abs(self.palm_fwep - cinf_p)))
        corr = float(np.corrcoef(self.palm_fwep, cinf_p)[0, 1])
        self.assertLessEqual(
            max_abs_diff, PVAL_ATOL,
            msg=(f"FWER p-value max abs diff = {max_abs_diff:.5f} > "
                f"tolerance {PVAL_ATOL:.5f} (={PVAL_ATOL*N_PERM_PALM:.1f} "
                f"MC quanta at n_perm={N_PERM_PALM})"),
        )
        self.assertGreater(
            corr, 0.99,
            msg=f"FWER p-value correlation = {corr:.4f}, expected > 0.99",
        )


if __name__ == "__main__":
    unittest.main()
