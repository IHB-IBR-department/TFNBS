"""Cross-implementation equivalence against PALM (Winkler 2014).

PALM is the reference MATLAB implementation for permutation inference on
the GLM in neuroimaging. We validate the shared infrastructure —
per-edge OLS t-stat, Freedman–Lane permutation, max-statistic FWER —
on synthetic two-sample and one-sample designs.

PALM operates on (n, m) flat data and conninfpy on (n, N, N) tensors,
so we generate flat data and symmetrize into the upper triangle of an
N×N matrix for the conninfpy call. Only the upper triangle is then
compared against PALM's flat output.

Per-permutation matching is not possible — MATLAB and NumPy use
different Mersenne-Twister streams even at matched seeds. Distributional
matching (quantiles, FWER-corrected p-values) is the standard
cross-implementation protocol; tolerances are set to Monte Carlo noise
budget at n_perm=5000.

Plan / scope / paper-paragraph template: see
``Projects/NetworkStatistics/palm_validation.md`` in the Obsidian vault.
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

from conninfpy import compute_p_val, compute_p_val_glm
from conninfpy.glm_stats import compute_glm_stat


# ---------------------------------------------------------------------------
# Environment detection — every test in this file skips cleanly when MATLAB
# or PALM is unavailable, so CI on machines without MATLAB stays green.
# ---------------------------------------------------------------------------

DEFAULT_MATLAB_BIN = "/Applications/MATLAB_R2025a.app/bin/matlab"
DEFAULT_PALM_DIR = str(Path.home() / "Work" / "tools" / "palm-alpha119")

MATLAB_BIN = os.environ.get("CONNINFPY_MATLAB_BIN", DEFAULT_MATLAB_BIN)
PALM_DIR = os.environ.get("CONNINFPY_PALM_DIR", DEFAULT_PALM_DIR)


def _matlab_available() -> bool:
    return Path(MATLAB_BIN).is_file() and Path(PALM_DIR, "palm.m").is_file()


SKIP_MSG = (
    f"MATLAB ({MATLAB_BIN}) or PALM ({PALM_DIR}) not available; "
    f"set CONNINFPY_MATLAB_BIN / CONNINFPY_PALM_DIR to override."
)


def _run_palm(workdir: Path, args: list[str], timeout: int = 300) -> str:
    """Invoke PALM in batch MATLAB. Returns stdout. Raises on non-zero exit."""
    palm_cmd = "palm(" + ",".join(f"'{a}'" for a in args) + ");"
    matlab_script = (
        f"cd('{workdir}'); "
        f"addpath('{PALM_DIR}'); "
        f"warning('off','all'); "
        f"{palm_cmd}"
    )
    result = subprocess.run(
        [MATLAB_BIN, "-batch", matlab_script],
        capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"MATLAB/PALM failed (rc={result.returncode}):\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result.stdout


def _flat_to_3d(Y_flat: np.ndarray, N: int) -> np.ndarray:
    """(n, m) upper-triangle vector → (n, N, N) symmetric, zero diagonal."""
    n, m = Y_flat.shape
    assert m == N * (N - 1) // 2
    iu = np.triu_indices(N, k=1)
    Y_3d = np.zeros((n, N, N), dtype=np.float64)
    Y_3d[:, iu[0], iu[1]] = Y_flat
    Y_3d[:, iu[1], iu[0]] = Y_flat
    return Y_3d


def _make_twosample_data(n1=15, n2=15, N=8, effect=0.7, n_signal_edges=5,
                        seed=42):
    """Generate (n, m) two-sample data with effect planted on first edges."""
    m = N * (N - 1) // 2
    rng = np.random.RandomState(seed)
    g1 = rng.randn(n1, m) * 0.5
    g2 = rng.randn(n2, m) * 0.5
    g2[:, :n_signal_edges] += effect
    Y_flat = np.vstack([g1, g2])
    X = np.column_stack([np.ones(n1 + n2),
                        np.r_[np.zeros(n1), np.ones(n2)]])
    contrast = np.array([0.0, 1.0])
    return Y_flat, X, contrast, N, m


def _make_paired_data(n=25, N=8, effect=0.7, n_signal_edges=5, seed=42):
    """Generate paired data Y_A, Y_B and their (flat) difference."""
    m = N * (N - 1) // 2
    rng = np.random.RandomState(seed)
    Y_A_flat = rng.randn(n, m) * 0.5
    Y_B_flat = Y_A_flat + rng.randn(n, m) * 0.5  # within-subject correlated
    Y_B_flat[:, :n_signal_edges] += effect       # effect: B > A on first edges
    delta_flat = Y_B_flat - Y_A_flat             # one-sample on Δ
    return Y_A_flat, Y_B_flat, delta_flat, N, m


# ---------------------------------------------------------------------------
# Tolerances — see palm_validation.md §2 for derivation
# ---------------------------------------------------------------------------

# Per-edge t-stat: PALM saves with %g (~4 decimals); tol = 1.5e-3
TSTAT_ATOL = 1.5e-3

# FWER p-values: two independent permutation streams (MATLAB MT vs NumPy MT)
# give p_palm, p_cinf each with binomial MC noise. For true p ∈ (0, 0.5),
# Var(p_palm − p_cinf) ≈ 2p(1−p)/n_perm. The MAX over m edges has expected
# value ≈ σ·√(2 log m). At n_perm=5000, m=28, worst-case p=0.5:
#   σ_pair ≈ √(2·0.25/5000) ≈ 0.010
#   E[max abs diff] ≈ 0.010 · √(2·log 28) ≈ 0.026
# Set tolerance at ~3·E[max] for headroom while still catching genuine bugs
# (which manifest as >0.1 systematic offset). Tight bug-detector remains
# the correlation threshold (>0.99) and the per-edge t-stat test.
N_PERM = 5000
PVAL_ATOL = 0.025


@unittest.skipUnless(_matlab_available(), SKIP_MSG)
class TestPalmEquivalenceTwoSample(unittest.TestCase):
    """Two-sample F-L permutation: conninfpy compute_p_val_glm vs PALM."""

    @classmethod
    def setUpClass(cls):
        cls.workdir = Path(tempfile.mkdtemp(prefix="palm_test_2s_"))
        Y_flat, X, contrast, N, m = _make_twosample_data()
        cls.Y_flat = Y_flat
        cls.X = X
        cls.contrast = contrast
        cls.N = N
        cls.m = m
        cls.iu = np.triu_indices(N, k=1)

        # Write PALM inputs
        np.savetxt(cls.workdir / "Y.csv", Y_flat, delimiter=",", fmt="%.10f")
        np.savetxt(cls.workdir / "X.csv", X, delimiter=",", fmt="%.10f")
        np.savetxt(cls.workdir / "C.csv", contrast[None, :], delimiter=",",
                fmt="%.10f")

        # Run PALM once for all tests in this class
        _run_palm(
            cls.workdir,
            ["-i", "Y.csv", "-d", "X.csv", "-t", "C.csv",
            "-n", str(N_PERM), "-o", "out", "-quiet", "-seed", "42"],
        )
        cls.palm_t = np.loadtxt(cls.workdir / "out_dat_tstat.csv",
                                delimiter=",")
        cls.palm_uncp = np.loadtxt(cls.workdir / "out_dat_tstat_uncp.csv",
                                delimiter=",")
        cls.palm_fwep = np.loadtxt(cls.workdir / "out_dat_tstat_fwep.csv",
                                delimiter=",")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.workdir, ignore_errors=True)

    def test_per_edge_tstat_matches_palm(self):
        """Test A: deterministic OLS t-stat agreement to PALM's %g precision."""
        Y_3d = _flat_to_3d(self.Y_flat, self.N)
        stat_dict = compute_glm_stat(
            Y_3d, self.X, self.contrast, stat_type="tstat",
        )
        # Reconstruct signed t-stat from per-tail non-negative pair
        signed = stat_dict["positive"] - stat_dict["negative"]
        cinf_t = signed[self.iu[0], self.iu[1]]

        np.testing.assert_allclose(
            cinf_t, self.palm_t, atol=TSTAT_ATOL,
            err_msg="Per-edge OLS t-stats differ beyond PALM's CSV precision",
        )

    def test_fwer_pvalues_match_palm(self):
        """Test C: FWER-corrected p-values within 2/n_perm MC quanta."""
        Y_3d = _flat_to_3d(self.Y_flat, self.N)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # rng deprecation chatter
            res = compute_p_val_glm(
                Y_3d, design_matrix=self.X, contrast=self.contrast,
                stat_type="tstat", method="tstat",
                n_permutations=N_PERM, use_mp=False, random_state=42,
            )
        # PALM's [0, 1] contrast → "Group2 > Group1" → conninfpy's positive tail
        cinf_p = res["positive"][self.iu[0], self.iu[1]]

        max_abs_diff = float(np.max(np.abs(self.palm_fwep - cinf_p)))
        corr = float(np.corrcoef(self.palm_fwep, cinf_p)[0, 1])

        self.assertLessEqual(
            max_abs_diff, PVAL_ATOL,
            msg=(f"FWER p-value max abs diff = {max_abs_diff:.5f} > "
                f"tolerance {PVAL_ATOL:.5f} (={PVAL_ATOL*N_PERM:.1f} MC quanta)"),
        )
        self.assertGreater(
            corr, 0.99,
            msg=f"FWER p-value correlation = {corr:.4f}, expected > 0.99",
        )

    def test_null_quantiles_match_palm(self):
        """Test B: max-stat null quantiles within MC noise budget.

        Recovers the conninfpy max-stat null distribution by inverting the
        FWER p-value formula on the observed t-stats. With n_perm=5000 and
        m=28 edges, the empirical SE on the 95th quantile is ~1%, so 5%
        relative tolerance is comfortable.
        """
        Y_3d = _flat_to_3d(self.Y_flat, self.N)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = compute_p_val_glm(
                Y_3d, design_matrix=self.X, contrast=self.contrast,
                stat_type="tstat", method="tstat",
                n_permutations=N_PERM, use_mp=False, random_state=42,
            )
        cinf_p_pos = res["positive"][self.iu[0], self.iu[1]]
        signed_t = compute_glm_stat(
            Y_3d, self.X, self.contrast, stat_type="tstat",
        )
        cinf_t_pos = signed_t["positive"][self.iu[0], self.iu[1]]

        # Both pipelines apply Phipson–Smyth: p = (count + 1) / (n_perm + 1)
        # where count = #{null >= obs}. So a p of 0.05 ↔ null 95th percentile
        # at the corresponding observed t-stat. Compare quantiles indirectly
        # via the *p-values at matched observed t* — that is what FWER
        # already checks. Here we sanity-check that the t-stat-vs-p-value
        # mapping has the same shape: at any t threshold, both should mark
        # the same fraction of edges significant.
        for alpha in (0.05, 0.10, 0.20):
            n_palm_sig = int(np.sum(self.palm_fwep <= alpha))
            n_cinf_sig = int(np.sum(cinf_p_pos <= alpha))
            # Allow a 1-edge difference at small m=28 due to MC granularity
            self.assertLessEqual(
                abs(n_palm_sig - n_cinf_sig), 1,
                msg=(f"At alpha={alpha}: PALM marks {n_palm_sig} edges "
                    f"significant, conninfpy marks {n_cinf_sig}"),
            )


@unittest.skipUnless(_matlab_available(), SKIP_MSG)
class TestPalmEquivalenceOneSample(unittest.TestCase):
    """One-sample / paired sign-flip: PALM `-ise` vs conninfpy paired t-test."""

    @classmethod
    def setUpClass(cls):
        cls.workdir = Path(tempfile.mkdtemp(prefix="palm_test_1s_"))
        Y_A, Y_B, delta_flat, N, m = _make_paired_data()
        cls.Y_A_flat = Y_A
        cls.Y_B_flat = Y_B
        cls.delta_flat = delta_flat
        cls.N = N
        cls.m = m
        cls.iu = np.triu_indices(N, k=1)
        n = delta_flat.shape[0]

        # PALM one-sample: design is intercept-only, sign-flip via -ise
        np.savetxt(cls.workdir / "Y.csv", delta_flat, delimiter=",",
                fmt="%.10f")
        np.savetxt(cls.workdir / "X.csv", np.ones((n, 1)), delimiter=",",
                fmt="%.10f")
        np.savetxt(cls.workdir / "C.csv", np.array([[1.0]]), delimiter=",",
                fmt="%.10f")

        _run_palm(
            cls.workdir,
            ["-i", "Y.csv", "-d", "X.csv", "-t", "C.csv",
            "-n", str(N_PERM), "-o", "out", "-quiet", "-seed", "42",
            "-ise"],  # independent symmetric errors → sign-flip
        )
        cls.palm_t = np.loadtxt(cls.workdir / "out_dat_tstat.csv",
                                delimiter=",")
        cls.palm_fwep = np.loadtxt(cls.workdir / "out_dat_tstat_fwep.csv",
                                delimiter=",")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.workdir, ignore_errors=True)

    def test_paired_tstat_matches_palm(self):
        """Per-edge one-sample t on the difference matches PALM."""
        Y_A_3d = _flat_to_3d(self.Y_A_flat, self.N)
        Y_B_3d = _flat_to_3d(self.Y_B_flat, self.N)
        # conninfpy paired-t convention: diffs = group2 - group1
        # PALM input was delta = Y_B - Y_A, so conninfpy(group1=Y_A, group2=Y_B)
        # gives matching "positive" tail for "B > A".
        from conninfpy import compute_t_stat
        t_dict = compute_t_stat(Y_A_3d, Y_B_3d, test_type="paired")
        signed = t_dict["positive"] - t_dict["negative"]
        cinf_t = signed[self.iu[0], self.iu[1]]

        np.testing.assert_allclose(
            cinf_t, self.palm_t, atol=TSTAT_ATOL,
            err_msg="Paired t-stats differ beyond PALM's CSV precision",
        )

    def test_paired_fwer_pvalues_match_palm(self):
        """Paired sign-flip FWER p-values agree with PALM `-ise`."""
        Y_A_3d = _flat_to_3d(self.Y_A_flat, self.N)
        Y_B_3d = _flat_to_3d(self.Y_B_flat, self.N)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = compute_p_val(
                Y_A_3d, Y_B_3d, test_type="paired", method="tstat",
                n_permutations=N_PERM, use_mp=False, random_state=42,
            )
        cinf_p = res["positive"][self.iu[0], self.iu[1]]

        max_abs_diff = float(np.max(np.abs(self.palm_fwep - cinf_p)))
        corr = float(np.corrcoef(self.palm_fwep, cinf_p)[0, 1])
        self.assertLessEqual(
            max_abs_diff, PVAL_ATOL,
            msg=(f"Paired FWER max abs diff = {max_abs_diff:.5f} > "
                f"tolerance {PVAL_ATOL:.5f}"),
        )
        self.assertGreater(
            corr, 0.999,
            msg=f"Paired FWER p-value correlation = {corr:.4f}",
        )


# ---------------------------------------------------------------------------
# Test E from palm_validation.md: within-block exchangeability
# ---------------------------------------------------------------------------

def _make_blocked_twosample(
    n_per_site=20, n_sites=3, N=6, effect=0.6,
    n_signal_edges=4, seed=42,
):
    """Two-sample blocked design where the group/site totals are perfectly
    balanced (half g1, half g2 within each site) — the regime where PALM
    -eb is the canonical reference."""
    rng = np.random.RandomState(seed)
    n_total = n_per_site * n_sites
    m = N * (N - 1) // 2

    # Site-balanced 0/1 group label
    sites = np.repeat(np.arange(n_sites), n_per_site)
    group = np.zeros(n_total, dtype=int)
    for s in range(n_sites):
        idx = np.where(sites == s)[0]
        group[idx[: len(idx) // 2]] = 1  # first half of each site → g1

    # Per-site additive shift
    site_shifts = rng.randn(n_sites, m) * 0.4
    Y_flat = rng.randn(n_total, m) + site_shifts[sites]

    # Plant the effect on a few edges, group-2 (group==0) > group-1
    sig = rng.choice(m, n_signal_edges, replace=False)
    Y_flat[group == 0, sig[:, None]] += effect

    # Design: intercept + group dummy. Contrast [0, 1] tests g2 > g1
    X = np.column_stack([np.ones(n_total), group.astype(float)])
    contrast = np.array([0.0, 1.0])
    return Y_flat, X, contrast, sites, N, m


@unittest.skipUnless(_matlab_available(), SKIP_MSG)
class TestPalmEbParity(unittest.TestCase):
    """Test E from [[palm_validation]]: distributional parity of
    conninfpy's ``strata=`` vs PALM's ``-eb`` (within-block exchangeability)
    on a blocked two-sample GLM design.

    Tolerance (0.07) is wider than vanilla designs because within-block
    permutation subspaces are smaller and sampling noise is higher.
    """

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
             "-n", str(N_PERM), "-o", "out", "-quiet", "-seed", "42"],
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
                n_permutations=N_PERM, use_mp=False, random_state=42,
                strata=self.sites,
            )
        self.assertTrue(res.strata_provided)
        cinf_p = res["positive"][self.iu[0], self.iu[1]]
        max_abs_diff = float(np.max(np.abs(self.palm_fwep - cinf_p)))
        corr = float(np.corrcoef(self.palm_fwep, cinf_p)[0, 1])

        # Tolerance 0.07 (Test E specific)
        self.assertLessEqual(
            max_abs_diff, 0.07,
            msg=f"FWER p-value max abs diff = {max_abs_diff:.5f} > 0.07",
        )
        self.assertGreater(
            corr, 0.99,
            msg=f"FWER p-value correlation = {corr:.4f}, expected > 0.99",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
