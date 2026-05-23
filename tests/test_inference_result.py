"""Tests for the v2.1 InferenceResult / OmnibusInferenceResult extension
(plan PR-2): observed-statistic maps, the ``stat_signed`` property,
provenance attributes, and end-to-end population of those fields by
each of the six constructor call sites in the pipelines.
"""
import unittest

import numpy as np
import numpy.testing as npt

from conninfpy import (
    InferenceResult,
    OmnibusInferenceResult,
    analyze,
    compute_p_val,
    fisher_r_to_z,
    generate_fc_matrices,
)
from conninfpy.glm_stats import compute_p_val_glm


def _make_glm_data(n=24, N=6, seed=0):
    rng = np.random.RandomState(seed)
    Y = rng.randn(n, N, N)
    Y = (Y + Y.transpose(0, 2, 1)) / 2
    for i in range(n):
        np.fill_diagonal(Y[i], 0.0)
    age = rng.randn(n)
    return Y, age


class TestInferenceResultDefaults(unittest.TestCase):
    """Backward compatibility: pre-v2.1 kwarg surface still works, new
    fields default to None / False."""

    def test_minimal_construction(self):
        r = InferenceResult(
            np.full((4, 4), 0.5),
            np.full((4, 4), 0.6),
            method="tstat",
        )
        self.assertEqual(r["positive"].shape, (4, 4))
        npt.assert_allclose(r.positive, 0.5)
        npt.assert_allclose(r.negative, 0.6)
        # New attributes default to safe values
        self.assertIsNone(r.stat_positive)
        self.assertIsNone(r.stat_negative)
        self.assertIsNone(r.stat_signed)
        self.assertEqual(r.stat_type, "tstat")
        self.assertFalse(r.harmonized)
        self.assertFalse(r.preserve_provided)
        self.assertFalse(r.strata_provided)
        self.assertIsNone(r.combat_diagnostics)


class TestStatSignedProperty(unittest.TestCase):
    """``stat_signed = stat_positive − stat_negative`` recovers the
    original signed effect map when the inputs are tail-clipped."""

    def test_signed_recovers_original_tstat(self):
        rng = np.random.RandomState(11)
        t = rng.randn(8, 8)
        t = (t + t.T) / 2
        np.fill_diagonal(t, 0.0)
        stat_pos = np.maximum(t, 0.0)
        stat_neg = np.maximum(-t, 0.0)
        r = InferenceResult(
            np.full_like(t, 0.5), np.full_like(t, 0.5),
            stat_positive=stat_pos, stat_negative=stat_neg,
        )
        npt.assert_allclose(r.stat_signed, t)

    def test_signed_none_when_inputs_missing(self):
        r = InferenceResult(np.zeros((3, 3)), np.zeros((3, 3)))
        self.assertIsNone(r.stat_signed)


class TestOmnibusInferenceResult(unittest.TestCase):
    """Sibling class for the F-stat path: single 'omnibus' key, no
    .positive/.negative, isinstance dispatch."""

    def test_single_key_and_attribute(self):
        F = np.full((5, 5), 3.0)
        r = OmnibusInferenceResult(F, method="tstat", stat_omnibus=F)
        self.assertEqual(set(r.keys()), {"omnibus"})
        npt.assert_allclose(r["omnibus"], 3.0)
        npt.assert_allclose(r.omnibus, 3.0)
        npt.assert_allclose(r.stat_omnibus, 3.0)
        self.assertEqual(r.stat_type, "fstat")

    def test_is_not_inference_result(self):
        """LSP: the F sibling must NOT be an InferenceResult — the latter
        promises .positive/.negative which F has no meaning for."""
        r = OmnibusInferenceResult(np.zeros((4, 4)))
        self.assertNotIsInstance(r, InferenceResult)


class TestParametricPathPopulatesStats(unittest.TestCase):
    """The Bonferroni / BH-FDR parametric path populates stat maps."""

    @classmethod
    def setUpClass(cls):
        g1, g2, _ = generate_fc_matrices(
            N=12, effect_size=0.0, n_samples_group1=15,
            n_samples_group2=15, seed=1,
        )
        cls.g1z, cls.g2z = fisher_r_to_z(g1), fisher_r_to_z(g2)

    def test_bonferroni_path(self):
        r = compute_p_val(
            self.g1z, self.g2z, test_type="two-sample",
            method="bonferroni", use_mp=False,
        )
        self.assertIsInstance(r, InferenceResult)
        self.assertIsNotNone(r.stat_positive)
        self.assertIsNotNone(r.stat_negative)
        self.assertEqual(r.stat_type, "tstat")
        # stat_positive is non-negative (one-tail clipped)
        self.assertTrue(np.all(r.stat_positive >= 0))
        self.assertTrue(np.all(r.stat_negative >= 0))

    def test_bh_fdr_perm_path(self):
        r = compute_p_val(
            self.g1z, self.g2z, test_type="two-sample",
            method="bh_fdr_perm", n_permutations=30,
            use_mp=False, rng=7,
        )
        self.assertIsInstance(r, InferenceResult)
        self.assertIsNotNone(r.stat_positive)
        self.assertIsNotNone(r.stat_negative)


class TestEnhancementPathStoresRawTstat(unittest.TestCase):
    """When an enhancement (TFNBS, NBS) runs, the stat maps on the result
    must be the RAW t-statistic, not the enhanced score — otherwise
    downstream consumers (effect-matrix plot, edge export) get the wrong
    units."""

    def test_tfnbs_stores_raw_tstat(self):
        g1, g2, _ = generate_fc_matrices(
            N=10, effect_size=0.0, n_samples_group1=12,
            n_samples_group2=12, seed=2,
        )
        g1z, g2z = fisher_r_to_z(g1), fisher_r_to_z(g2)
        r = compute_p_val(
            g1z, g2z, test_type="two-sample",
            method="tfnbs", n_permutations=20,
            use_mp=False, rng=3,
        )
        self.assertIsInstance(r, InferenceResult)
        self.assertIsNotNone(r.stat_positive)
        # Reasonable t-stat magnitudes (single-digit), not TFNBS scores
        # (which can run into hundreds depending on (E, H) and density).
        self.assertLess(float(np.max(r.stat_positive)), 50.0)
        self.assertEqual(r.stat_type, "tstat")


class TestGLMTstatPath(unittest.TestCase):

    def test_glm_tstat_populates_stat_maps(self):
        Y, age = _make_glm_data(n=20, N=6, seed=4)
        r = compute_p_val_glm(
            Y, interest=age, method="tstat",
            n_permutations=20, use_mp=False, rng=4,
        )
        self.assertIsInstance(r, InferenceResult)
        self.assertIsNotNone(r.stat_positive)
        self.assertIsNotNone(r.stat_negative)
        self.assertEqual(r.stat_type, "tstat")


class TestGLMFstatPathOmnibus(unittest.TestCase):
    """F-stat path returns an OmnibusInferenceResult (no zeros-like hack
    in _analyze.py anymore)."""

    def test_fstat_returns_omnibus_inference_result(self):
        rng = np.random.RandomState(5)
        n, N = 30, 5
        Y = rng.randn(n, N, N); Y = (Y + Y.transpose(0, 2, 1)) / 2
        for i in range(n):
            np.fill_diagonal(Y[i], 0.0)
        # Two regressors → multi-row contrast → F-stat
        x1 = rng.randn(n); x2 = rng.randn(n)
        X = np.column_stack([np.ones(n), x1, x2])
        contrast = np.array([[0, 1, 0], [0, 0, 1]])
        r = compute_p_val_glm(
            Y, design_matrix=X, contrast=contrast,
            stat_type="fstat", method="tstat",
            n_permutations=20, use_mp=False, rng=5,
        )
        self.assertIsInstance(r, OmnibusInferenceResult)
        self.assertNotIsInstance(r, InferenceResult)
        self.assertEqual(set(r.keys()), {"omnibus"})
        self.assertIsNotNone(r.stat_omnibus)
        self.assertEqual(r.stat_type, "fstat")


class TestAnalyzeProvenanceThreading(unittest.TestCase):
    """`analyze()` threads ComBat provenance fields onto the result."""

    def test_provenance_on_glm_path(self):
        Y, age = _make_glm_data(n=24, N=5, seed=6)
        sites = np.array([0] * 12 + [1] * 12)
        confounds = np.linspace(-1.0, 1.0, 24).reshape(-1, 1)
        out = analyze(
            Y, interest=age, confounds=confounds, sites=sites,
            fisher_z=False, n_permutations=20, use_mp=False,
            acceleration=None, rng=6,
        )
        # GLM + sites + confounds resolves to Strategy D: ComBat runs
        # with preserve=confounds (set automatically), strata=sites.
        self.assertTrue(out.inference.harmonized)
        self.assertIsNotNone(out.inference.combat_diagnostics)
        self.assertEqual(out.inference.combat_diagnostics["strategy"], "D")
        self.assertTrue(out.inference.strata_provided)

    def test_no_sites_means_not_harmonized(self):
        Y, age = _make_glm_data(n=20, N=5, seed=7)
        out = analyze(
            Y, interest=age, fisher_z=False,
            n_permutations=20, use_mp=False, acceleration=None, rng=7,
        )
        self.assertFalse(out.inference.harmonized)
        self.assertFalse(out.inference.preserve_provided)
        self.assertIsNone(out.inference.combat_diagnostics)

    def test_analyze_result_omnibus_property_guards(self):
        Y, age = _make_glm_data(n=20, N=5, seed=8)
        out = analyze(
            Y, interest=age, fisher_z=False,
            n_permutations=20, use_mp=False, acceleration=None, rng=8,
        )
        # T-stat path: .omnibus raises, .positive/.negative work
        with self.assertRaises(AttributeError):
            _ = out.omnibus
        _ = out.positive
        _ = out.negative


class TestMultiEHGrid(unittest.TestCase):
    """Array-(E, H) flow: a single permutation pass returns a 3D
    parameter-grid InferenceResult; result-layer helpers project to 2D
    via .select() or per-call param_idx=."""

    def setUp(self):
        self.Y, self.age = _make_glm_data(n=24, N=6, seed=3)
        sex = np.array([0, 1] * 12, dtype=float)
        self.confounds = sex
        self.sites = np.array(["A"] * 8 + ["B"] * 8 + ["C"] * 8)
        self.e_grid = [0.4, 0.5, 0.75]
        self.h_grid = [3.0, 2.0, 3.0]

    def test_grid_call_returns_3d_pmaps_and_labels_axis(self):
        out = analyze(
            self.Y, interest=self.age, confounds=self.confounds,
            sites=self.sites, harmonize="auto",
            method="tfnbs",
            e=self.e_grid, h=self.h_grid, n=4,
            n_permutations=20, acceleration=None,
            use_mp=False, fisher_z=False, rng=0,
        )
        r = out.inference
        self.assertTrue(r.is_grid)
        self.assertEqual(r.positive.shape, (6, 6, 3))
        self.assertEqual(r.negative.shape, (6, 6, 3))
        npt.assert_allclose(r.e_grid, self.e_grid)
        npt.assert_allclose(r.h_grid, self.h_grid)
        # Raw t-stats stay 2D even when scores are 3D
        self.assertEqual(r.stat_positive.shape, (6, 6))
        self.assertEqual(r.stat_negative.shape, (6, 6))

    def test_n_significant_returns_per_cell_list(self):
        out = analyze(
            self.Y, interest=self.age, confounds=self.confounds,
            sites=self.sites, harmonize="auto",
            method="tfnbs",
            e=self.e_grid, h=self.h_grid, n=4,
            n_permutations=20, acceleration=None,
            use_mp=False, fisher_z=False, rng=0,
        )
        r = out.inference
        per_cell = r.n_significant(0.5)
        self.assertIsInstance(per_cell["positive"], list)
        self.assertEqual(len(per_cell["positive"]), 3)
        # With param_idx, returns a scalar count
        scalar = r.n_significant(0.5, param_idx=0)
        self.assertIsInstance(scalar["positive"], int)

    def test_select_projects_to_2d(self):
        out = analyze(
            self.Y, interest=self.age, confounds=self.confounds,
            sites=self.sites, harmonize="auto",
            method="tfnbs",
            e=self.e_grid, h=self.h_grid, n=4,
            n_permutations=20, acceleration=None,
            use_mp=False, fisher_z=False, rng=0,
        )
        r = out.inference
        sub = r.select(1)
        self.assertFalse(sub.is_grid)
        self.assertEqual(sub.positive.shape, (6, 6))
        npt.assert_allclose(sub.positive, r.positive[:, :, 1])
        npt.assert_allclose(sub.e_grid, [self.e_grid[1]])

    def test_significant_edges_requires_param_idx_on_grid(self):
        out = analyze(
            self.Y, interest=self.age, confounds=self.confounds,
            sites=self.sites, harmonize="auto",
            method="tfnbs",
            e=self.e_grid, h=self.h_grid, n=4,
            n_permutations=20, acceleration=None,
            use_mp=False, fisher_z=False, rng=0,
        )
        r = out.inference
        with self.assertRaises(ValueError):
            r.significant_edges(alpha=0.5)
        # With param_idx, returns a DataFrame for that cell
        df = r.significant_edges(alpha=0.5, param_idx=2)
        self.assertEqual(df.shape[1] > 0, True)

    def test_scalar_call_still_returns_2d(self):
        out = analyze(
            self.Y, interest=self.age, confounds=self.confounds,
            sites=self.sites, harmonize="auto",
            method="tfnbs",
            e=0.4, h=3.0, n=4,
            n_permutations=20, acceleration=None,
            use_mp=False, fisher_z=False, rng=0,
        )
        r = out.inference
        self.assertFalse(r.is_grid)
        self.assertEqual(r.positive.shape, (6, 6))
        self.assertIsNone(r.e_grid)
        self.assertIsNone(r.h_grid)


if __name__ == "__main__":
    unittest.main()
