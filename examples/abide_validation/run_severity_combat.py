"""
ABIDE §3.3 — ADOS/SRS severity regression under Path-1 ComBat.

Runs the severity regression three ways as pre-registered in ValidationPlan.md §3.3:

  (1) Edge-level TFNBS FWER (expected null — honest reporting)
  (2) Edge-level BH-FDR (via bh_fdr_perm; more liberal multiple-comparison frame)
  (3) Block-level cNBS on Yeo-7 partitions (28 block tests, much higher power —
      the "positive control" alternative)

Runs on both severity measures:
  - ADOS Total (n ≈ 236 ASD)
  - SRS Total (n ≈ 162 ASD, wider dynamic range)

Outputs to ``results/combat/severity/``.
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
OUT_DIR = os.path.join(HERE, "results", "combat", "severity")


def _load():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} not found — run `python harmonize.py` first.")
    d = np.load(DATA_FILE, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _prep_asd_subset(data, score_name: str):
    """Select ASD subjects with valid `score_name` + return design arrays."""
    group = data["group"]; score = data[score_name]
    asd_mask = (group == 1) & (~np.isnan(score))
    idx = np.where(asd_mask)[0]
    print(f"  {score_name}: n_ASD with valid score = {len(idx)}, "
          f"range [{score[asd_mask].min():.1f}, {score[asd_mask].max():.1f}], "
          f"mean {score[asd_mask].mean():.1f}")

    conn_harm = data["connectivity_z_harm"][asd_mask]
    conn_raw = data["connectivity_z"][asd_mask]
    severity = score[asd_mask].astype(float)
    age = data["age"][asd_mask]
    sex = data["sex"][asd_mask]
    fd = data["mean_fd"][asd_mask]

    # Control for ADOS module (Torres 2020) only for ADOS analyses.
    extras = [age, sex, fd]
    mod = data.get("ados_module")
    if mod is not None and score_name.startswith("ados"):
        mod_asd = mod[asd_mask]
        valid_mod = ~np.isnan(mod_asd)
        if valid_mod.all() and len(np.unique(mod_asd)) > 1:
            mod_dummies = pd.get_dummies(mod_asd, drop_first=True).values.astype(float)
            extras.append(mod_dummies)
    confounds = np.column_stack(extras)
    return conn_harm, conn_raw, severity, confounds


def _summary(label, p_dict, alpha=0.05):
    iu = np.triu_indices(100, k=1)
    for k, v in p_dict.items():
        vec = v[iu]
        n_sig = int((vec < alpha).sum())
        print(f"  {label} [{k}]: n_sig={n_sig}, min_p={vec.min():.4f}, "
              f"5%-tile p={np.percentile(vec, 5):.4f}")


def run_one(data, score_name: str, n_perm: int, seed: int):
    print("\n" + "=" * 72)
    print(f"ABIDE severity — {score_name}")
    print("=" * 72)

    conn_harm, conn_raw, severity, confounds = _prep_asd_subset(data, score_name)
    results = {}

    # ---- (1) Edge-level TFNBS FWER ------------------------------------
    print(f"\n(1) Edge-level TFNBS FWER (expected null)")
    t0 = time.time()
    p_tfnbs = compute_p_val_glm(
        conn_harm, interest=severity, confounds=confounds,
        method="tfnbs", n_permutations=n_perm,
        e=0.4, h=3.0, n=10, use_mp=True, random_state=seed,
    )
    print(f"  time: {time.time() - t0:.1f}s")
    _summary("TFNBS-FWER", p_tfnbs)
    results["tfnbs"] = p_tfnbs

    # ---- (2) Edge-level BH-FDR (parametric) ------------------------
    # GLM pipeline doesn't support bh_fdr_perm directly; we compute per-edge
    # parametric p-values from the GLM t-statistic (Wald form, two-tailed)
    # and apply BH separately per tail (consistent with the FWER pipeline
    # which treats tails separately).
    print(f"\n(2) Edge-level BH-FDR (parametric GLM t-stats, two-tailed)")
    X, contrast = build_design_matrix(severity, confounds)
    t_dict = compute_glm_stat(conn_harm, X, contrast, stat_type="tstat")
    df_resid = conn_harm.shape[0] - X.shape[1]
    iu = np.triu_indices(conn_harm.shape[1], k=1)

    # Tail-wise p-values and effect-size diagnostic
    # compute_glm_stat returns non-negative pos/neg (zero-clipped). Reconstruct signed t:
    t_signed = t_dict["positive"] - t_dict["negative"]

    # Convert to per-edge Pearson-r with severity (partial, after confound regression),
    # purely for diagnostic print (not for inference): r_partial ≈ t / sqrt(t^2 + df).
    t_vec = t_signed[iu]
    r_partial = t_vec / np.sqrt(t_vec ** 2 + df_resid)
    print(f"  effect-size distribution across {t_vec.size} edges (after confound regression):")
    print(f"    |t|: max={np.abs(t_vec).max():.2f}, 95%-tile={np.percentile(np.abs(t_vec), 95):.2f}")
    print(f"    partial r: max={np.abs(r_partial).max():.3f}, 95%-tile={np.percentile(np.abs(r_partial), 95):.3f}")
    # Pre-registered context: |r|≥0.15 would be small-but-real, |r|≥0.10 is noise floor for n=236.

    # Per-edge parametric p-values (two-tailed), then split tails for BH (matching the FWER split)
    p_two = 2 * (1 - scistats.t.cdf(np.abs(t_vec), df=df_resid))
    # For positive tail: only test edges with t > 0
    pos_mask = t_vec > 0
    neg_mask = t_vec < 0
    p_pos_vec = np.full_like(p_two, 1.0)
    p_neg_vec = np.full_like(p_two, 1.0)
    p_pos_vec[pos_mask] = 1 - scistats.t.cdf(t_vec[pos_mask], df=df_resid)
    p_neg_vec[neg_mask] = 1 - scistats.t.cdf(-t_vec[neg_mask], df=df_resid)

    def _bh(p_vec_in):
        p = np.asarray(p_vec_in)
        m = p.size
        order = np.argsort(p)
        ranked = p[order]
        adj = ranked * m / (np.arange(m) + 1)
        # enforce monotonicity
        adj = np.minimum.accumulate(adj[::-1])[::-1]
        out = np.empty_like(p)
        out[order] = np.clip(adj, 0, 1)
        return out

    p_pos_bh = _bh(p_pos_vec)
    p_neg_bh = _bh(p_neg_vec)

    # Stash back into (100, 100) full matrices for saving
    N = conn_harm.shape[1]
    def _vec_to_mat(vec, iu_, N_):
        mat = np.ones((N_, N_), dtype=np.float64)
        mat[iu_[0], iu_[1]] = vec
        mat[iu_[1], iu_[0]] = vec
        return mat

    p_bh = {
        "positive": _vec_to_mat(p_pos_bh, iu, N),
        "negative": _vec_to_mat(p_neg_bh, iu, N),
    }
    _summary("BH-FDR", p_bh)
    results["bh_fdr"] = p_bh
    # Also save signed t-stat for later inspection
    results["_t_signed"] = {"t_signed": t_signed}

    # ---- (3) Block-level cNBS on Yeo-7 partitions ---------------------
    print(f"\n(3) cNBS (block-level) on Yeo-7 partitions")
    net_labels = data["net_labels"]
    t0 = time.time()
    p_cnbs = compute_p_val_glm(
        conn_harm, interest=severity, confounds=confounds,
        method="cnbs", net_labels=net_labels,
        n_permutations=n_perm, use_mp=True, random_state=seed,
    )
    print(f"  time: {time.time() - t0:.1f}s")
    _summary("cNBS", p_cnbs)
    # cNBS p-values are constant within each (block_i, block_j) pair.
    # Report per-unique block pair instead of per-edge:
    for tail, p in p_cnbs.items():
        # collect one p per block pair
        uniq = set()
        K = int(net_labels.max() + 1)
        names = list(data["network_order"])
        block_rows = []
        for bi in range(K):
            for bj in range(bi, K):
                ii = np.where(net_labels == bi)[0]
                jj = np.where(net_labels == bj)[0]
                if ii.size == 0 or jj.size == 0:
                    continue
                # Grab any in-block p
                if bi == bj:
                    if len(ii) < 2:
                        continue
                    p_val = p[ii[0], ii[1]]
                else:
                    p_val = p[ii[0], jj[0]]
                uniq.add((bi, bj, float(p_val)))
        rows = sorted(uniq, key=lambda x: x[2])
        print(f"  cNBS [{tail}] block p-values (sorted):")
        for bi, bj, pv in rows[:10]:
            marker = "*" if pv < 0.05 else " "
            print(f"    {marker} {names[bi]:>24s} × {names[bj]:<24s}  p = {pv:.4f}")
    results["cnbs"] = p_cnbs

    # ---- Save all three --------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    for m, p in results.items():
        out = os.path.join(OUT_DIR, f"severity_{score_name}_{m}.npz")
        np.savez(out, **p)
        print(f"  saved: {out}")
    # Also save the signed-t diagnostic for inspection
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--score", choices=["ados_total", "srs_total", "all"], default="all")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    n_perm = 500 if args.quick else 5000

    data = _load()
    scores = ["ados_total", "srs_total"] if args.score == "all" else [args.score]
    for s in scores:
        run_one(data, s, n_perm=n_perm, seed=args.seed)


if __name__ == "__main__":
    main()
