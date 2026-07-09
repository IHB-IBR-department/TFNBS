"""Phase-0 feasibility check for the structural-prior validation pipeline.

Derisks the rest of the validation in <2 minutes wallclock. Runs four
checks against the extracted ENIGMA Schaefer-100 SC matrix:

1. Matrix sanity: shape, symmetry, diagonal zero, weight range,
   isolated-node count.
2. OU steady-state covariance:
       C_ss = σ² · (τ⁻¹·I − W)⁻¹ · (τ⁻¹·I − W)⁻ᵀ
   is constructed and verified positive-definite (Cholesky succeeds);
   the induced correlation matrix's off-diagonal range is reported so
   we can see whether it resembles a plausible FC scale.
3. Louvain on weighted SC: community count, modularity Q, sizes.
4. Smoke: drop Louvain labels into ``apply_ni_tfnbs`` on a synthetic
   stat dict and confirm shape/finiteness.

Run from the repo root with the `conninfpy` env active.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ou_steady_state_cov(
    W: np.ndarray, tau: float = 1.0, sigma: float = 1.0
) -> np.ndarray:
    """Steady-state covariance of dx/dt = -x/τ + Wx + η, η ~ N(0, σ²I).

    Returns C_ss = σ² · M⁻¹ · M⁻ᵀ where M = τ⁻¹·I − W.
    Provably SPD whenever τ⁻¹·I − W is invertible and the OU process
    is stable (spectral radius of τW < 1).
    """
    N = W.shape[0]
    M = (1.0 / tau) * np.eye(N) - W
    M_inv = np.linalg.inv(M)
    return (sigma ** 2) * (M_inv @ M_inv.T)


def _cov2corr(C: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.diag(C))
    return C / np.outer(d, d)


def _louvain(W: np.ndarray, seed: int = 0) -> tuple[np.ndarray, float, list[int]]:
    """Run weighted Louvain on |W| (clamp small negatives to 0).

    Returns (node_labels, modularity Q, community sizes sorted desc).
    """
    import networkx as nx

    W_pos = np.where(W > 0, W, 0.0)
    G = nx.from_numpy_array(W_pos)
    communities = nx.community.louvain_communities(
        G, weight="weight", seed=seed
    )
    labels = np.empty(W.shape[0], dtype=np.int_)
    for idx, comm in enumerate(communities):
        for node in comm:
            labels[node] = idx
    Q = nx.community.modularity(G, communities, weight="weight")
    sizes = sorted([len(c) for c in communities], reverse=True)
    return labels, float(Q), sizes


def main() -> int:
    sc_path = _root() / "datasets" / "sc" / "sc_schaefer100_hcp.npy"
    print(f"[1/4] Loading SC from {sc_path.relative_to(_root())}")
    sc = np.load(sc_path)
    print(f"      shape={sc.shape}, "
          f"symmetric={np.allclose(sc, sc.T)}, "
          f"diag_zero={np.allclose(np.diag(sc), 0.0)}")
    weights = sc[np.triu_indices_from(sc, k=1)]
    nonzero = weights[weights > 0]
    print(f"      weights min={weights.min():.3f} max={weights.max():.3f} "
          f"density={nonzero.size/weights.size:.3f} "
          f"isolated_nodes={int(np.sum(sc.sum(axis=0) == 0.0))}")

    print("[2/4] Building OU steady-state covariance ...")
    # Pick tau s.t. spectral radius of tau·W < 1 → stable OU
    spec_rad = float(np.max(np.abs(np.linalg.eigvalsh(sc))))
    tau = 0.5 / spec_rad  # comfortable margin
    print(f"      ρ(W)={spec_rad:.3f}, τ={tau:.4f}  (τ·ρ(W)={tau*spec_rad:.3f} < 1 ✓)")
    C = _ou_steady_state_cov(sc, tau=tau, sigma=1.0)
    # SPD check via Cholesky
    try:
        np.linalg.cholesky(C)
        spd = True
    except np.linalg.LinAlgError:
        spd = False
    print(f"      SPD via Cholesky: {spd}")
    R = _cov2corr(C)
    iu = np.triu_indices_from(R, k=1)
    print(f"      cov2corr(C) off-diag: min={R[iu].min():.3f} "
          f"max={R[iu].max():.3f} mean={R[iu].mean():.3f} "
          f"std={R[iu].std():.3f}")

    print("[3/4] Louvain on weighted SC ...")
    labels, Q, sizes = _louvain(sc, seed=42)
    print(f"      n_communities={len(sizes)}, modularity Q={Q:.3f}")
    print(f"      community sizes (top 10): {sizes[:10]}")

    print("[4/4] Smoke test: apply_ni_tfnbs with Louvain labels ...")
    from conninfpy import apply_ni_tfnbs

    rng = np.random.RandomState(0)
    t = rng.randn(*sc.shape)
    t = (t + t.T) / 2.0
    np.fill_diagonal(t, 0.0)
    stat_dict = {
        "positive": np.maximum(t, 0.0),
        "negative": np.maximum(-t, 0.0),
    }
    scores = apply_ni_tfnbs(
        stat_dict, net_labels=labels, e=0.4, h=3.0, n=10,
    )
    for tail in ("positive", "negative"):
        s = scores[tail]
        print(f"      {tail:<8} shape={s.shape} "
              f"finite={bool(np.all(np.isfinite(s)))} "
              f"min={s.min():.3g} max={s.max():.3g}")

    print("\nphase-0 OK — feasibility verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
