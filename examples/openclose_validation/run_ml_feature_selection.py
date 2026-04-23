"""
§3.8 — ML feature-selection validation: do significant edges from ConnInfPy's
stats pipeline improve out-of-sample classification?

Bidirectional (IHB → China AND China → IHB), three baselines per cell:
  1. all-edges (no selection)
  2. random-subset matched size (50 draws, averaged)
  3. univariate |t| top-k matched size — crucial: does network enhancement add value
     beyond plain thresholding?

Selectors: tstat, tfnbs, nbs@{2.0, 3.0}, cnbs, ni_tfnbs, bh_fdr.
α grid: {0.10, 0.05, 0.01, 0.005, 0.001}.

Output: CSV with one row per (direction, selector, alpha OR baseline), recording
n_edges, within-train grouped-CV AUC (for C tuning), out-of-sample AUC + balanced
accuracy on the target cohort.

Headline metric: ΔAUC (selector) minus ΔAUC (matched baseline).

Runtime budget (local Apple M5):
  - 14 selector runs (7 methods × 2 directions) at 5000 perms ≈ 1–2 min
  - Per-cell LR + CV for 7×5 selector cells + 3 baseline sets × 2 directions ≈ 5–10 min
  - Total < 15 min.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from conninfpy import compute_p_val
from examples.openclose_validation.openclose_loader import OpenCloseDataset


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "ml"

SELECTORS = {
    "tstat":     dict(method="tstat"),
    "tfnbs":     dict(method="tfnbs", e=0.4, h=3.0, n=10),
    "nbs@2.0":   dict(method="nbs", threshold=2.0),
    "nbs@3.0":   dict(method="nbs", threshold=3.0),
    "cnbs":      dict(method="cnbs"),               # net_labels filled in later
    "ni_tfnbs":  dict(method="ni_tfnbs", e=0.4, h=3.0, n=10),
    "bh_fdr":    dict(method="bh_fdr"),             # parametric, no permutations
}

ALPHA_GRID = [0.10, 0.05, 0.01, 0.005, 0.001]
C_GRID = [0.01, 0.1, 1.0, 10.0]
N_RAND_DRAWS = 50


# =============================================================================
# Feature extraction
# =============================================================================

def build_feature_matrix(ds: OpenCloseDataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack per-subject (open, close) Fisher-z upper triangles into (2N, n_edges).

    Returns
    -------
    X : (2·n_subjects, n_edges) — row order: all-opens first, then all-closes
    y : (2·n_subjects,) — 0 for open, 1 for close
    groups : (2·n_subjects,) — subject index (0..n_subjects-1), same subject repeats for open/close
    """
    o_z, c_z = ds.connectivity_z(run=0)
    n = ds.n_subjects
    iu = np.triu_indices(ds.n_rois, k=1)
    X_open = o_z[:, iu[0], iu[1]]
    X_close = c_z[:, iu[0], iu[1]]
    X = np.vstack([X_open, X_close])
    y = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    groups = np.concatenate([np.arange(n), np.arange(n)])
    return X, y, groups


# =============================================================================
# Selector p-maps
# =============================================================================

def compute_selector_pmaps(
    ds: OpenCloseDataset, n_perm: int, seed: int
) -> Dict[str, Dict[str, np.ndarray]]:
    """Run every selector on this cohort; return {method_key: {'g1>g2':.., 'g2>g1':..}}."""
    o_z, c_z = ds.connectivity_z(run=0)
    net_labels = ds.net_labels

    out = {}
    for name, base_kwargs in SELECTORS.items():
        kwargs = dict(base_kwargs)
        if kwargs["method"] == "cnbs":
            kwargs["net_labels"] = net_labels
        elif kwargs["method"] == "ni_tfnbs":
            kwargs["net_labels"] = net_labels

        t0 = time.time()
        p = compute_p_val(
            o_z, c_z,
            test_type="paired",
            n_permutations=n_perm,
            use_mp=True,
            random_state=seed,
            **kwargs,
        )
        dt = time.time() - t0
        n_sig = int((p[list(p.keys())[0]][np.triu_indices(ds.n_rois, k=1)] < 0.05).sum())
        print(f"  [{ds.experiment}] {name:10s}  {dt:5.1f}s  n_sig(0.05, g2>g1)={n_sig}")
        out[name] = p
    return out


def mask_from_pmap(p: Dict[str, np.ndarray], alpha: float, iu: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Per-edge boolean mask: significant in EITHER tail (union)."""
    # tail keys vary between 'g1>g2'/'g2>g1' and 'positive'/'negative' — handle both
    pa = p.get("g2>g1", p.get("positive"))
    pb = p.get("g1>g2", p.get("negative"))
    pu = np.minimum(pa, pb)[iu[0], iu[1]]
    return pu < alpha


# =============================================================================
# LR with grouped-CV C tuning + out-of-sample evaluation
# =============================================================================

def train_lr_grouped_cv(X_tr: np.ndarray, y_tr: np.ndarray, groups_tr: np.ndarray,
                       n_splits: int = 5, seed: int = 42) -> Tuple[LogisticRegression, StandardScaler, float, float]:
    """Fit StandardScaler + tune C via GroupKFold on training, refit on all training.

    Returns (fitted_lr, fitted_scaler, best_C, mean_cv_auc).
    """
    scaler = StandardScaler()
    X_tr_std = scaler.fit_transform(X_tr)

    gkf = GroupKFold(n_splits=n_splits)
    best_C, best_score = C_GRID[0], -np.inf
    for C in C_GRID:
        scores = []
        for tr_idx, va_idx in gkf.split(X_tr_std, y_tr, groups=groups_tr):
            lr = LogisticRegression(C=C, penalty="l2", solver="liblinear", max_iter=2000,
                                     random_state=seed)
            lr.fit(X_tr_std[tr_idx], y_tr[tr_idx])
            p = lr.predict_proba(X_tr_std[va_idx])[:, 1]
            try:
                scores.append(roc_auc_score(y_tr[va_idx], p))
            except ValueError:
                scores.append(0.5)
        if np.mean(scores) > best_score:
            best_score = np.mean(scores)
            best_C = C

    lr_final = LogisticRegression(C=best_C, penalty="l2", solver="liblinear",
                                   max_iter=2000, random_state=seed)
    lr_final.fit(X_tr_std, y_tr)
    return lr_final, scaler, best_C, best_score


def evaluate(lr: LogisticRegression, scaler: StandardScaler,
             X_te: np.ndarray, y_te: np.ndarray) -> Dict[str, float]:
    X_te_std = scaler.transform(X_te)
    p = lr.predict_proba(X_te_std)[:, 1]
    yhat = (p >= 0.5).astype(int)
    try:
        auc = roc_auc_score(y_te, p)
    except ValueError:
        auc = np.nan
    return {
        "auc": float(auc),
        "bal_acc": float(balanced_accuracy_score(y_te, yhat)),
    }


# =============================================================================
# Univariate baseline mask
# =============================================================================

def univariate_topk_mask(X_tr: np.ndarray, y_tr: np.ndarray, groups_tr: np.ndarray,
                         k: int) -> np.ndarray:
    """Two-sample paired t-stat per edge (close - open), pick top-k by |t|.

    Paired via group ids (same group id appears twice in training).
    """
    uniq = np.unique(groups_tr)
    diffs = np.empty((len(uniq), X_tr.shape[1]))
    for i, g in enumerate(uniq):
        idx = np.where(groups_tr == g)[0]
        # y_tr[idx] has 0 (open) and 1 (close); subtract open from close
        open_row = idx[y_tr[idx] == 0][0]
        close_row = idx[y_tr[idx] == 1][0]
        diffs[i] = X_tr[close_row] - X_tr[open_row]
    # one-sample t on diffs, per edge
    t = diffs.mean(axis=0) / (diffs.std(axis=0, ddof=1) / np.sqrt(len(uniq)) + 1e-12)
    abs_t = np.abs(t)
    # top-k edges
    mask = np.zeros(X_tr.shape[1], dtype=bool)
    if k > 0:
        top = np.argsort(-abs_t)[:k]
        mask[top] = True
    return mask


# =============================================================================
# Main experiment
# =============================================================================

def run_direction(
    train_ds: OpenCloseDataset, test_ds: OpenCloseDataset,
    train_pmaps: Dict[str, Dict[str, np.ndarray]],
    seed: int, rng: np.random.Generator,
) -> pd.DataFrame:
    direction = f"{train_ds.experiment}->{test_ds.experiment}"
    print(f"\n{'=' * 72}\n{direction}\n{'=' * 72}")

    X_tr, y_tr, g_tr = build_feature_matrix(train_ds)
    X_te, y_te, g_te = build_feature_matrix(test_ds)
    print(f"  train {train_ds.experiment}: X={X_tr.shape}  test {test_ds.experiment}: X={X_te.shape}")

    n_edges = X_tr.shape[1]
    iu = np.triu_indices(train_ds.n_rois, k=1)
    rows = []

    # ---- all-edges baseline (one cell) ---------------------------
    lr, sc, C, cv_auc = train_lr_grouped_cv(X_tr, y_tr, g_tr, seed=seed)
    ev = evaluate(lr, sc, X_te, y_te)
    rows.append({
        "direction": direction, "kind": "baseline", "selector": "all_edges",
        "alpha": np.nan, "n_edges_used": n_edges,
        "cv_auc": cv_auc, "best_C": C, **ev,
    })
    print(f"  all-edges baseline          n={n_edges:5d}  CV_AUC={cv_auc:.3f}  test_AUC={ev['auc']:.3f}")

    # ---- per (selector, alpha) cells -----------------------------
    for sel_name, p in train_pmaps.items():
        for alpha in ALPHA_GRID:
            mask = mask_from_pmap(p, alpha, iu)
            k = int(mask.sum())
            if k == 0:
                rows.append({
                    "direction": direction, "kind": "selector", "selector": sel_name,
                    "alpha": alpha, "n_edges_used": 0,
                    "cv_auc": np.nan, "best_C": np.nan, "auc": np.nan, "bal_acc": np.nan,
                })
                continue

            X_tr_sel = X_tr[:, mask]
            X_te_sel = X_te[:, mask]
            lr, sc, C, cv_auc = train_lr_grouped_cv(X_tr_sel, y_tr, g_tr, seed=seed)
            ev = evaluate(lr, sc, X_te_sel, y_te)
            rows.append({
                "direction": direction, "kind": "selector", "selector": sel_name,
                "alpha": alpha, "n_edges_used": k,
                "cv_auc": cv_auc, "best_C": C, **ev,
            })

            # Matched random-subset baseline (averaged over draws)
            rand_aucs, rand_baccs = [], []
            for _ in range(N_RAND_DRAWS):
                idx = rng.choice(n_edges, size=k, replace=False)
                X_tr_r = X_tr[:, idx]; X_te_r = X_te[:, idx]
                lr_r, sc_r, _, _ = train_lr_grouped_cv(X_tr_r, y_tr, g_tr, seed=seed)
                ev_r = evaluate(lr_r, sc_r, X_te_r, y_te)
                rand_aucs.append(ev_r["auc"]); rand_baccs.append(ev_r["bal_acc"])
            rows.append({
                "direction": direction, "kind": "baseline",
                "selector": f"random_matched_k{k}", "alpha": alpha, "n_edges_used": k,
                "cv_auc": np.nan, "best_C": np.nan,
                "auc": float(np.mean(rand_aucs)), "bal_acc": float(np.mean(rand_baccs)),
            })

            # Matched univariate-top-k baseline
            uni_mask = univariate_topk_mask(X_tr, y_tr, g_tr, k=k)
            X_tr_u = X_tr[:, uni_mask]; X_te_u = X_te[:, uni_mask]
            lr_u, sc_u, _, _ = train_lr_grouped_cv(X_tr_u, y_tr, g_tr, seed=seed)
            ev_u = evaluate(lr_u, sc_u, X_te_u, y_te)
            rows.append({
                "direction": direction, "kind": "baseline",
                "selector": f"uni_topk{k}", "alpha": alpha, "n_edges_used": k,
                "cv_auc": np.nan, "best_C": np.nan, **ev_u,
            })

            print(f"  {sel_name:10s} α={alpha:<6.3f} k={k:5d}  "
                  f"sel_AUC={ev['auc']:.3f}  rand_AUC={np.mean(rand_aucs):.3f}  "
                  f"uni_AUC={ev_u['auc']:.3f}")

    return pd.DataFrame(rows)


def main() -> None:
    global N_RAND_DRAWS
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-rand-draws", type=int, default=N_RAND_DRAWS)
    args = ap.parse_args()
    N_RAND_DRAWS = args.n_rand_draws

    os.makedirs(RESULTS, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("Loading cohorts...")
    ds_ihb = OpenCloseDataset.load("ihb")
    ds_china = OpenCloseDataset.load("china")

    print(f"\nComputing selector p-maps (n_perm={args.n_perm}):")
    print(f"\n-- IHB selectors --")
    pmaps_ihb = compute_selector_pmaps(ds_ihb, n_perm=args.n_perm, seed=args.seed)
    print(f"\n-- China selectors --")
    pmaps_china = compute_selector_pmaps(ds_china, n_perm=args.n_perm, seed=args.seed)

    # Save p-maps for reuse / inspection
    for name, p in pmaps_ihb.items():
        np.savez(RESULTS / f"pmap_ihb_{name.replace('@', '_').replace('.', '')}.npz", **p)
    for name, p in pmaps_china.items():
        np.savez(RESULTS / f"pmap_china_{name.replace('@', '_').replace('.', '')}.npz", **p)

    # Directions
    df_ihb_to_china = run_direction(ds_ihb, ds_china, pmaps_ihb, seed=args.seed, rng=rng)
    df_china_to_ihb = run_direction(ds_china, ds_ihb, pmaps_china, seed=args.seed, rng=rng)
    df = pd.concat([df_ihb_to_china, df_china_to_ihb], ignore_index=True)

    out = RESULTS / "ml_feature_selection.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
