"""
ABIDE §3.11.2 — Split-half reliability figure (paper Fig 9).

Demonstrates the claim "TFNBS produces more reliable discoveries than
fixed-threshold NBS or per-edge t-tests at a realistic ABIDE sample
size".

Protocol (option A: ComBat-harmonised full cohort, then split)
--------------------------------------------------------------
On the ComBat-harmonised ABIDE features (`abide_harmonized.npz`) — the
same data backing the headline 885-edge Age claim — randomly split the
764 subjects into two halves of N≈382 a hundred times. For each split
and each method, run the same canonical Age-GLM on each half:

    compute_p_val_glm(Y_half, interest=age_half, confounds=[sex, mean_fd],
                      method=<m>, n_permutations=200, acceleration='gpd')

then compute Jaccard at α=0.05 between the two halves' thresholded
masks per direction. The 100-Jaccard distribution per method is the
reliability metric. Spearman on -log10 p (rank-based, robust to
detection-count asymmetry between halves) is reported alongside.

Methods (8) — same panel as audit_age_methods.py:
    tstat, NBS@2.0, NBS@3.0, TFNBS, cNBS, NI-TFNBS, FBC-TFNBS, BH-FDR

Headline expectation: TFNBS-family > NBS > t-stat in median Jaccard.

Output
------
results/combat/age/split_half/
  per_split.csv                  # n_splits × n_methods × tail rows
  summary.csv                    # per-method median / IQR Jaccard + Spearman
  split_half.png                 # paper Fig 9 — boxplots per method per tail

Wall-clock estimate: ~3-4 h on 18 cores at 100 splits × 8 methods ×
200-perm GPD-accelerated GLM. The bottleneck is the COUNT of
permutation tasks (1600 separate GLM runs), not perm budget per task.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
from scipy import stats as scistats

from conninfpy import compute_p_val_glm, compute_glm_stat, build_design_matrix


HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(HERE, "abide_harmonized.npz")
OUT_DIR = os.path.join(HERE, "results", "combat", "age", "split_half")


# Methods panel — matches audit_age_methods.py
METHODS = ["tstat", "nbs@2.0", "nbs@3.0", "tfnbs", "cnbs", "ni_tfnbs", "fbc_tfnbs", "bh_fdr"]


def _load():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} not found — run `python harmonize.py` first.")
    d = np.load(DATA_FILE, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _run_method_on_half(conn, age, confounds, net_labels, method_key,
                        n_perm, acceleration, seed):
    """Single method-call on a single half. Returns {'positive', 'negative'} p-maps."""
    common = dict(
        Y=conn, interest=age, confounds=confounds,
        n_permutations=n_perm, use_mp=True, random_state=seed,
    )
    accel = dict() if method_key == "bh_fdr" else dict(acceleration=acceleration)

    if method_key == "tstat":
        return compute_p_val_glm(**common, **accel, method="tstat")
    if method_key == "tfnbs":
        return compute_p_val_glm(**common, **accel, method="tfnbs", e=0.4, h=3.0, n=10)
    if method_key.startswith("nbs@"):
        thr = float(method_key.split("@")[1])
        return compute_p_val_glm(**common, **accel, method="nbs", threshold=thr)
    if method_key == "cnbs":
        return compute_p_val_glm(**common, **accel, method="cnbs", net_labels=net_labels)
    if method_key == "ni_tfnbs":
        return compute_p_val_glm(
            **common, **accel, method="ni_tfnbs", net_labels=net_labels,
            e=0.4, h=3.0, n=10,
        )
    if method_key == "fbc_tfnbs":
        return compute_p_val_glm(
            **common, **accel, method="fbc_tfnbs", net_labels=net_labels,
            e=0.4, h=3.0, n=10, min_cluster_size=3,
        )
    if method_key == "bh_fdr":
        return _parametric_bh_fdr(conn, age, confounds)
    raise ValueError(f"Unknown method_key: {method_key}")


def _parametric_bh_fdr(conn, interest, confounds):
    """Parametric BH-FDR (no permutation). Same as audit_age_methods._parametric_bh_fdr."""
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


def _jaccard_at_alpha(p1_mat, p2_mat, alpha=0.05):
    """Edge-level Jaccard between two p-maps thresholded at alpha."""
    N = p1_mat.shape[0]
    iu = np.triu_indices(N, k=1)
    m1 = p1_mat[iu] < alpha
    m2 = p2_mat[iu] < alpha
    union = (m1 | m2).sum()
    return float((m1 & m2).sum() / union) if union > 0 else 0.0


def _spearman_neglog(p1_mat, p2_mat):
    N = p1_mat.shape[0]
    iu = np.triu_indices(N, k=1)
    nlp1 = -np.log10(np.maximum(p1_mat[iu], 1e-300))
    nlp2 = -np.log10(np.maximum(p2_mat[iu], 1e-300))
    r, _ = scistats.spearmanr(nlp1, nlp2)
    return float(r)


def _make_splits(n_subjects, n_splits, seed):
    """Return list of (idx_half_a, idx_half_b) pairs."""
    rng = np.random.default_rng(seed)
    half = n_subjects // 2
    splits = []
    for _ in range(n_splits):
        perm = rng.permutation(n_subjects)
        a = np.sort(perm[:half])
        b = np.sort(perm[half:half * 2])
        splits.append((a, b))
    return splits


def _plot_split_half(summary_df, per_split_df, out_path):
    """Boxplot per method, two panels (positive / negative tail)."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    method_order = METHODS
    for ax, tail, title in [
        (axes[0], "positive", "Jaccard reliability — positive tail (age↑→FC↑)"),
        (axes[1], "negative", "Jaccard reliability — negative tail (age↑→FC↓)"),
    ]:
        data = [per_split_df[(per_split_df["method"] == m)
                              & (per_split_df["tail"] == tail)]["jaccard"].values
                for m in method_order]
        bp = ax.boxplot(data, labels=method_order, patch_artist=True,
                        medianprops=dict(color="black", linewidth=1.5))
        for patch in bp["boxes"]:
            patch.set_facecolor("#88a")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Jaccard @ α=0.05")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
        for label in ax.get_xticklabels():
            label.set_rotation(35)
            label.set_ha("right")

    fig.suptitle(
        "ABIDE Age split-half reliability  (n=764 → 100 random 50/50 splits)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="5 splits × 100 perms (smoke test, ~5 min)")
    ap.add_argument("--n-splits", type=int, default=100)
    ap.add_argument("--n-permutations", type=int, default=200)
    ap.add_argument("--no-acceleration", action="store_true",
                    help="Disable GPD; use empirical p-values")
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    if args.quick:
        n_splits = 5
        n_perm = 100
        acceleration = "gpd"
        print("** QUICK MODE: 5 splits × 100 perms with GPD **")
    else:
        n_splits = args.n_splits
        n_perm = args.n_permutations
        acceleration = None if args.no_acceleration else "gpd"

    os.makedirs(OUT_DIR, exist_ok=True)
    data = _load()
    conn_full = data["connectivity_z_harm"]
    age_full = data["age"].astype(float)
    confounds_full = np.column_stack([
        data["sex"].astype(float),
        data["mean_fd"].astype(float),
    ])
    net_labels = data["net_labels"]
    n_total = conn_full.shape[0]

    print(f"Loaded ABIDE: n_total={n_total}, N={conn_full.shape[1]}")
    print(f"Methods: {args.methods}")
    print(f"n_splits={n_splits}, n_perm={n_perm}, acceleration={acceleration}, alpha={args.alpha}")
    print(f"Each split: half_size = {n_total // 2}")

    splits = _make_splits(n_total, n_splits, seed=args.seed)

    rows = []
    t_start = time.time()
    for s_idx, (a, b) in enumerate(splits):
        print(f"\n--- split {s_idx + 1}/{n_splits} ---")
        conn_a = conn_full[a]
        conn_b = conn_full[b]
        age_a, age_b = age_full[a], age_full[b]
        c_a, c_b = confounds_full[a], confounds_full[b]

        for m in args.methods:
            t0 = time.time()
            p_a = _run_method_on_half(
                conn_a, age_a, c_a, net_labels, m,
                n_perm, acceleration, seed=args.seed + s_idx * 1000,
            )
            p_b = _run_method_on_half(
                conn_b, age_b, c_b, net_labels, m,
                n_perm, acceleration, seed=args.seed + s_idx * 1000 + 1,
            )
            elapsed = time.time() - t0

            for tail in ("positive", "negative"):
                jac = _jaccard_at_alpha(p_a[tail], p_b[tail], alpha=args.alpha)
                spr = _spearman_neglog(p_a[tail], p_b[tail])
                n_sig_a = int((p_a[tail][np.triu_indices(p_a[tail].shape[0], 1)] < args.alpha).sum())
                n_sig_b = int((p_b[tail][np.triu_indices(p_b[tail].shape[0], 1)] < args.alpha).sum())
                rows.append(dict(
                    split=s_idx, method=m, tail=tail,
                    jaccard=jac, spearman=spr,
                    n_sig_a=n_sig_a, n_sig_b=n_sig_b,
                    elapsed_s=elapsed,
                ))
            print(f"  {m:12s}  pos Jac={rows[-2]['jaccard']:.3f} "
                  f"neg Jac={rows[-1]['jaccard']:.3f}  ({elapsed:.1f}s)")

        # Periodic checkpoint (every 5 splits)
        if (s_idx + 1) % 5 == 0 or s_idx + 1 == n_splits:
            tmp = pd.DataFrame(rows)
            tmp.to_csv(os.path.join(OUT_DIR, "per_split.csv"), index=False)

    total_elapsed = time.time() - t_start
    per_split_df = pd.DataFrame(rows)
    per_split_df.to_csv(os.path.join(OUT_DIR, "per_split.csv"), index=False)

    # =========================================================================
    # Summary per (method, tail)
    # =========================================================================
    summary = (per_split_df
               .groupby(["method", "tail"])
               .agg(
                   jaccard_median=("jaccard", "median"),
                   jaccard_mean=("jaccard", "mean"),
                   jaccard_q25=("jaccard", lambda x: np.quantile(x, 0.25)),
                   jaccard_q75=("jaccard", lambda x: np.quantile(x, 0.75)),
                   spearman_median=("spearman", "median"),
                   spearman_mean=("spearman", "mean"),
                   n_sig_a_median=("n_sig_a", "median"),
                   n_sig_b_median=("n_sig_b", "median"),
                   n_splits=("split", "count"),
               )
               .reset_index())
    summary.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)

    print("\n" + "=" * 72)
    print("HEADLINE — median Jaccard per (method, tail)")
    print("=" * 72)
    for m in args.methods:
        rows_m = summary[summary["method"] == m]
        for _, r in rows_m.iterrows():
            print(f"  {m:12s}  [{r['tail']:9s}]  "
                  f"Jac med={r['jaccard_median']:.3f}  "
                  f"IQR=({r['jaccard_q25']:.3f}, {r['jaccard_q75']:.3f})  "
                  f"Spr med={r['spearman_median']:.3f}")

    plot_path = os.path.join(OUT_DIR, "split_half.png")
    _plot_split_half(summary, per_split_df, plot_path)
    print(f"\nFig saved: {plot_path}")
    print(f"Total wall-clock: {total_elapsed:.1f}s")
    print(f"All outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
