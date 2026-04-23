"""
ABIDE §3.9 — Method comparison on the age → FC positive control.

Runs each of the 7 methods on the same ComBat-harmonised GLM design
(interest=age, confounds=sex, mean_fd) at 5000 permutations, then writes
per-method p-maps to results/combat/age/methods/. The downstream
comparison (Spearman / Jaccard / top-k concordance / block-mass) is
produced by :mod:`audit_age_methods`.

We intentionally run on age (which has strong signal) rather than ADOS
(honest null) — the method comparison is only informative when all
methods have something to find.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
from scipy import stats as scistats

from conninfpy import compute_p_val_glm, compute_glm_stat, build_design_matrix


HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(HERE, "abide_harmonized.npz")
OUT_DIR = os.path.join(HERE, "results", "combat", "age", "methods")


def _load():
    d = np.load(DATA_FILE, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _run_method(conn, interest, confounds, net_labels, method_key, n_perm, seed):
    """Dispatch to the right compute_p_val_glm call for each method variant."""
    kwargs = dict(
        Y=conn, interest=interest, confounds=confounds,
        n_permutations=n_perm, use_mp=True, random_state=seed,
    )

    if method_key == "tstat":
        return compute_p_val_glm(**kwargs, method="tstat")
    if method_key == "tfnbs":
        return compute_p_val_glm(**kwargs, method="tfnbs", e=0.4, h=3.0, n=10)
    if method_key.startswith("nbs@"):
        thr = float(method_key.split("@")[1])
        return compute_p_val_glm(**kwargs, method="nbs", threshold=thr)
    if method_key == "cnbs":
        return compute_p_val_glm(**kwargs, method="cnbs", net_labels=net_labels)
    if method_key == "ni_tfnbs":
        return compute_p_val_glm(
            **kwargs, method="ni_tfnbs", net_labels=net_labels,
            e=0.4, h=3.0, n=10,
        )
    if method_key == "fbc_tfnbs":
        return compute_p_val_glm(
            **kwargs, method="fbc_tfnbs", net_labels=net_labels,
            e=0.4, h=3.0, n=10, min_cluster_size=3,
        )
    raise ValueError(f"Unknown method_key: {method_key}")


def _parametric_bh_fdr(conn, interest, confounds):
    """Parametric BH-FDR on GLM t-stats — added separately because
    compute_p_val_glm doesn't support bh_fdr_perm."""
    X, contrast = build_design_matrix(interest, confounds)
    t_dict = compute_glm_stat(conn, X, contrast, stat_type="tstat")
    t_signed = t_dict["positive"] - t_dict["negative"]
    df_resid = conn.shape[0] - X.shape[1]
    N = conn.shape[1]
    iu = np.triu_indices(N, k=1)
    t_vec = t_signed[iu]

    p_pos_vec = np.where(t_vec > 0, 1 - scistats.t.cdf(t_vec, df=df_resid), 1.0)
    p_neg_vec = np.where(t_vec < 0, 1 - scistats.t.cdf(-t_vec, df=df_resid), 1.0)

    def _bh(p):
        m = p.size; order = np.argsort(p)
        adj = p[order] * m / (np.arange(m) + 1)
        adj = np.minimum.accumulate(adj[::-1])[::-1]
        out = np.empty_like(p); out[order] = np.clip(adj, 0, 1)
        return out

    def _to_mat(vec):
        mat = np.ones((N, N), dtype=np.float64)
        mat[iu[0], iu[1]] = vec
        mat[iu[1], iu[0]] = vec
        return mat

    return {
        "positive": _to_mat(_bh(p_pos_vec)),
        "negative": _to_mat(_bh(p_neg_vec)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--methods", nargs="+",
        default=["tstat", "nbs@2.0", "nbs@3.0", "tfnbs", "cnbs", "ni_tfnbs", "fbc_tfnbs", "bh_fdr"],
    )
    args = ap.parse_args()
    n_perm = 500 if args.quick else 5000

    os.makedirs(OUT_DIR, exist_ok=True)
    d = _load()
    conn = d["connectivity_z_harm"]
    age = d["age"].astype(float)
    confounds = np.column_stack([d["sex"].astype(float), d["mean_fd"].astype(float)])
    net_labels = d["net_labels"]
    print(f"Age range {age.min():.1f}–{age.max():.1f}, N={len(age)}")
    print(f"Running {len(args.methods)} methods at n_perm={n_perm}")

    for m in args.methods:
        print("\n" + "=" * 72)
        print(f"METHOD: {m}")
        print("=" * 72)
        t0 = time.time()
        if m == "bh_fdr":
            p = _parametric_bh_fdr(conn, age, confounds)
            elapsed = time.time() - t0
        else:
            p = _run_method(conn, age, confounds, net_labels, m, n_perm, seed=args.seed)
            elapsed = time.time() - t0
        print(f"  time: {elapsed:.1f}s")

        # Quick report
        iu = np.triu_indices(100, k=1)
        for k, v in p.items():
            vec = v[iu]
            n_sig = int((vec < 0.05).sum())
            print(f"  [{k}]: n_sig={n_sig}/{len(vec)}, min_p={vec.min():.4f}")

        out = os.path.join(OUT_DIR, f"age_{m.replace('@', '_')}.npz")
        np.savez(out, **p)
        print(f"  saved: {out}")


if __name__ == "__main__":
    main()
