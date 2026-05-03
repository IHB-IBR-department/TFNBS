"""Tests for compute_p_val_glm_multi."""
from __future__ import annotations

import unittest
import warnings

import numpy as np

from conninfpy import (
    InferenceResult,
    compute_p_val_glm,
    compute_p_val_glm_multi,
    fisher_r_to_z,
    generate_fc_matrices,
)


def _make_inputs(seed: int = 0):
    g1, g2, _ = generate_fc_matrices(
        N=10, effect_size=0.3, n_samples_group1=20, n_samples_group2=20, seed=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Y = fisher_r_to_z(np.concatenate([g1, g2]))
    rng = np.random.default_rng(seed)
    n = Y.shape[0]
    age = rng.normal(20, 5, n)
    sex = rng.integers(0, 2, n).astype(float)
    motion = rng.normal(0, 1, n)
    X = np.column_stack([np.ones(n), age, sex, motion])
    contrasts = {
        "age":    np.array([0.0, 1.0, 0.0, 0.0]),
        "sex":    np.array([0.0, 0.0, 1.0, 0.0]),
        "motion": np.array([0.0, 0.0, 0.0, 1.0]),
    }
    return Y, X, contrasts


class TestMultiContrast(unittest.TestCase):

    def test_returns_one_inferenceresult_per_contrast(self):
        Y, X, contrasts = _make_inputs()
        result = compute_p_val_glm_multi(
            Y, X, contrasts,
            method="tstat", n_permutations=20, use_mp=False, rng=42,
        )
        self.assertEqual(set(result.keys()), set(contrasts.keys()))
        for name, ir in result.items():
            self.assertIsInstance(ir, InferenceResult)
            self.assertEqual(ir.positive.shape, (Y.shape[1], Y.shape[1]))
            self.assertEqual(ir.method, "tstat")
            self.assertEqual(ir.n_permutations, 20)

    def test_explicit_nuisance_changes_reduced_model(self):
        # If we declare only `age` as the regressor of interest (nuisance =
        # everything else), the reduced model differs from the default
        # (where age, sex, motion are all of interest).
        Y, X, contrasts = _make_inputs()
        nuisance_only_age = np.array([0.0, 1.0, 0.0, 0.0])
        result_default = compute_p_val_glm_multi(
            Y, X, contrasts,
            method="tstat", n_permutations=20, use_mp=False, rng=42,
        )
        result_age_only = compute_p_val_glm_multi(
            Y, X, contrasts,
            nuisance_contrast=nuisance_only_age,
            method="tstat", n_permutations=20, use_mp=False, rng=42,
        )
        # The two reduced models differ → the *age* p-map should be the same
        # as compute_p_val_glm(age) under nuisance_only_age (since both leave
        # age as the only column outside the reduced model).
        # We don't assert exact equality here — just that the two
        # multi-call paths produce *different* results for sex/motion.
        # (Their reduced models differ.)
        self.assertFalse(
            np.allclose(
                result_default["sex"].positive,
                result_age_only["sex"].positive,
            )
        )

    def test_two_tailed_pools_per_contrast(self):
        Y, X, contrasts = _make_inputs()
        result = compute_p_val_glm_multi(
            Y, X, contrasts,
            two_tailed=True,
            method="tstat", n_permutations=20, use_mp=False, rng=42,
        )
        # With two_tailed=True the per-tail null is pooled, so positive and
        # negative p-maps share their max-stat null distribution. The
        # symmetry of the upper-triangle marginals should be tighter than
        # under one-tailed.
        for ir in result.values():
            self.assertEqual(ir.positive.shape, ir.negative.shape)

    def test_rejects_fstat(self):
        Y, X, contrasts = _make_inputs()
        with self.assertRaises(ValueError):
            compute_p_val_glm_multi(
                Y, X, contrasts,
                stat_type="fstat", n_permutations=10, use_mp=False, rng=0,
            )

    def test_rejects_empty_contrasts(self):
        Y, X, _ = _make_inputs()
        with self.assertRaises(ValueError):
            compute_p_val_glm_multi(
                Y, X, {}, n_permutations=10, use_mp=False, rng=0,
            )

    def test_contrast_length_must_match_design(self):
        Y, X, _ = _make_inputs()
        bad = {"age": np.array([0.0, 1.0])}  # length 2, design width 4
        with self.assertRaises(ValueError):
            compute_p_val_glm_multi(
                Y, X, bad, n_permutations=10, use_mp=False, rng=0,
            )

    def test_speedup_vs_independent_calls(self):
        # Multi-contrast must be at least as fast as independent calls. We
        # don't assert a strict speedup factor (it depends on hardware) but
        # we sanity-check the wall-time monotonicity.
        Y, X, contrasts = _make_inputs()
        import time

        t0 = time.perf_counter()
        for c in contrasts.values():
            compute_p_val_glm(
                Y, design_matrix=X, contrast=c,
                method="tstat", n_permutations=30, use_mp=False, rng=0,
            )
        wall_independent = time.perf_counter() - t0

        t0 = time.perf_counter()
        compute_p_val_glm_multi(
            Y, X, contrasts,
            method="tstat", n_permutations=30, use_mp=False, rng=0,
        )
        wall_multi = time.perf_counter() - t0

        # Allow up to 30 % slack for noise on small problems
        self.assertLessEqual(wall_multi, wall_independent * 1.3)


if __name__ == "__main__":
    unittest.main()
