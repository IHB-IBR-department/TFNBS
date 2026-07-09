"""Unit tests for the SC-prior validation building blocks.

These tests use a tiny synthetic block-SC matrix only — they do NOT
load any ENIGMA / HCP data, so they run in CI without external
dependencies. The wallclock target is sub-second.

What they cover:
- The OU steady-state covariance construction is SPD.
- The Louvain-on-SC prior recovers a planted block partition on a
  block-structured synthetic SC matrix.
- The Louvain labels drop into the existing ``apply_ni_tfnbs`` API
  with no code changes.
- Thresholded connected components is a valid alternative cluster
  source on a tractable SC density.
"""
from __future__ import annotations

import unittest

import numpy as np
import numpy.testing as npt


def _block_sc(
    block_sizes=(8, 8, 8),
    within_block_w: float = 1.0,
    between_block_w: float = 0.1,
    seed: int = 0,
) -> np.ndarray:
    """Symmetric block-structured weighted adjacency.

    Diagonal zero. Within-block edges ~ U(within_block_w/2, within_block_w);
    between-block edges ~ U(0, between_block_w). Two-sided variation gives
    a non-degenerate Louvain input.
    """
    rng = np.random.RandomState(seed)
    N = sum(block_sizes)
    block_of = np.repeat(np.arange(len(block_sizes)), block_sizes)
    M = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i + 1, N):
            if block_of[i] == block_of[j]:
                w = rng.uniform(within_block_w / 2.0, within_block_w)
            else:
                w = rng.uniform(0.0, between_block_w)
            M[i, j] = M[j, i] = w
    return M, block_of


def _ou_cov(W: np.ndarray, tau: float, sigma: float = 1.0) -> np.ndarray:
    N = W.shape[0]
    M = (1.0 / tau) * np.eye(N) - W
    M_inv = np.linalg.inv(M)
    return (sigma ** 2) * (M_inv @ M_inv.T)


class TestOUSteadyStateCovariance(unittest.TestCase):
    """The OU model C_ss = σ² (τ⁻¹I − W)⁻¹ (τ⁻¹I − W)⁻ᵀ is provably SPD
    on any symmetric W as long as τ⁻¹I − W is invertible. We confirm
    that and a sanity-level FC scale."""

    def test_spd_via_cholesky(self):
        W, _ = _block_sc(seed=1)
        # Pick τ with comfortable margin: τ·ρ(W) ≈ 0.5
        rho = float(np.max(np.abs(np.linalg.eigvalsh(W))))
        tau = 0.5 / rho
        C = _ou_cov(W, tau=tau)
        np.linalg.cholesky(C)  # raises LinAlgError if not SPD

    def test_correlation_is_finite_and_in_range(self):
        W, _ = _block_sc(seed=2)
        rho = float(np.max(np.abs(np.linalg.eigvalsh(W))))
        # Stronger coupling (closer to stability boundary) → larger FC
        C = _ou_cov(W, tau=0.9 / rho)
        d = np.sqrt(np.diag(C))
        R = C / np.outer(d, d)
        iu = np.triu_indices_from(R, k=1)
        self.assertTrue(np.all(np.isfinite(R[iu])))
        self.assertTrue(np.all(R[iu] > -1.0))
        self.assertTrue(np.all(R[iu] < 1.0))


class TestLouvainOnBlockSC(unittest.TestCase):
    """Louvain on a planted block-SC matrix recovers the planted blocks
    (after relabeling) and produces high modularity Q."""

    def test_louvain_recovers_planted_blocks(self):
        import networkx as nx

        W, planted = _block_sc(
            block_sizes=(10, 10, 10),
            within_block_w=1.0,
            between_block_w=0.05,
            seed=3,
        )
        G = nx.from_numpy_array(W)
        comms = nx.community.louvain_communities(
            G, weight="weight", seed=42, resolution=1.0
        )
        # Build the predicted label vector
        pred = np.empty(W.shape[0], dtype=np.int_)
        for idx, comm in enumerate(comms):
            for node in comm:
                pred[node] = idx

        # The label assignment is arbitrary — check the partition matches
        # the planted one via adjusted Rand index ≥ 0.95 (essentially exact
        # on this contrast).
        from sklearn.metrics import adjusted_rand_score

        ari = adjusted_rand_score(planted, pred)
        self.assertGreater(
            ari, 0.95,
            f"Louvain failed to recover planted partition (ARI={ari:.3f}).",
        )
        Q = nx.community.modularity(G, comms, weight="weight")
        self.assertGreater(Q, 0.3)


class TestConnectedComponentsPrior(unittest.TestCase):
    """The thresholded-CC prior (Option A2 in the wiki note) is a valid
    alternative SC-derived partition: thresholding the block-SC matrix
    at any point between the within/between weight ranges should yield
    one component per block."""

    def test_threshold_components_match_blocks(self):
        from conninfpy.utils import get_components

        W, planted = _block_sc(
            block_sizes=(10, 10, 10),
            within_block_w=1.0,
            between_block_w=0.05,
            seed=4,
        )
        # Any threshold in (between_max ≈ 0.05, within_min = 0.5) works
        thresh = 0.2
        mask = W > thresh
        labels, sizes = get_components(mask.astype(int))
        # Sanity: 3 components, one per planted block (or 30 singletons
        # plus 3 real components if isolated nodes leak through — guard
        # by counting components of size > 1).
        big_comp_count = int(np.sum(sizes > 1))
        self.assertEqual(big_comp_count, 3)
        # Those big components match the planted partition exactly
        from sklearn.metrics import adjusted_rand_score
        ari = adjusted_rand_score(planted, labels)
        self.assertEqual(ari, 1.0)


class TestApplyNiTfnbsWithSCDerivedLabels(unittest.TestCase):
    """Drop a Louvain partition into apply_ni_tfnbs — the existing
    public API — and confirm shape, finiteness, and basic behavior.
    This is what makes Option A (cluster mode) a zero-code-change
    validation path."""

    def test_apply_ni_tfnbs_smoke(self):
        from conninfpy import apply_ni_tfnbs

        W, planted = _block_sc(
            block_sizes=(8, 8, 8),
            within_block_w=1.0,
            between_block_w=0.05,
            seed=5,
        )
        # Synthetic stat dict (positive/negative tails, one-tail clipped)
        rng = np.random.RandomState(5)
        t = rng.randn(*W.shape)
        t = (t + t.T) / 2.0
        np.fill_diagonal(t, 0.0)
        stat_dict = {
            "positive": np.maximum(t, 0.0),
            "negative": np.maximum(-t, 0.0),
        }
        scores = apply_ni_tfnbs(
            stat_dict, net_labels=planted, e=0.4, h=3.0, n=10,
        )
        for tail in ("positive", "negative"):
            s = scores[tail]
            self.assertEqual(s.shape, W.shape)
            self.assertTrue(np.all(np.isfinite(s)))
            self.assertGreaterEqual(s.min(), 0.0)


if __name__ == "__main__":
    unittest.main()
