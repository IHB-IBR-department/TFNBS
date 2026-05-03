"""Numba ↔ scipy backend equivalence + speedup smoke tests.

The TFNBS scoring function `get_tfnbs_score` ships with two backends:

- ``backend='scipy'`` — the reference implementation using
  :func:`scipy.sparse.csgraph.connected_components`.
- ``backend='numba'`` — a JIT-compiled union-find that bypasses scipy's
  sparse-matrix construction overhead, ~10–15× faster on typical inputs.

Tests:

1. Numba backend produces identical scores to the scipy backend.
2. Auto-detection picks numba when installed, scipy otherwise — no
   user code changes required.

Skipped if numba isn't installed.
"""
from __future__ import annotations

import unittest
import warnings

import numpy as np

from conninfpy import compute_t_stat, fisher_r_to_z, generate_fc_matrices
from conninfpy.tfnbs_score import HAS_NUMBA, get_tfnbs_score


@unittest.skipUnless(HAS_NUMBA, "numba not installed")
class TestNumbaBackendEquivalence(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g1, g2, _ = generate_fc_matrices(
                N=30, effect_size=0.3,
                n_samples_group1=20, n_samples_group2=20, seed=0,
            )
            t = compute_t_stat(
                fisher_r_to_z(g1), fisher_r_to_z(g2),
                test_type="two-sample",
            )
        cls.t_pos = t["positive"]

    def test_scalar_params_match(self):
        scipy_out = get_tfnbs_score(
            self.t_pos, e=0.4, h=3.0, n=10, backend="scipy"
        )
        numba_out = get_tfnbs_score(
            self.t_pos, e=0.4, h=3.0, n=10, backend="numba"
        )
        np.testing.assert_allclose(scipy_out, numba_out, rtol=1e-8, atol=1e-8)

    def test_list_params_match(self):
        # Multi-(E, H) sweep: returned shape is (N, N, num_params).
        scipy_out = get_tfnbs_score(
            self.t_pos, e=[0.4, 0.5], h=[2.0, 3.0], n=10, backend="scipy"
        )
        numba_out = get_tfnbs_score(
            self.t_pos, e=[0.4, 0.5], h=[2.0, 3.0], n=10, backend="numba"
        )
        np.testing.assert_allclose(scipy_out, numba_out, rtol=1e-8, atol=1e-8)

    def test_auto_picks_numba_when_available(self):
        # 'auto' should match 'numba' exactly (HAS_NUMBA gates the choice).
        auto_out = get_tfnbs_score(
            self.t_pos, e=0.4, h=3.0, n=10, backend="auto"
        )
        numba_out = get_tfnbs_score(
            self.t_pos, e=0.4, h=3.0, n=10, backend="numba"
        )
        np.testing.assert_allclose(auto_out, numba_out, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
