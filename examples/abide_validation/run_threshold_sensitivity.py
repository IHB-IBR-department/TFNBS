"""
ABIDE §3.11.1 — Threshold-sensitivity figure (paper Fig 8).

Demonstrates the central rhetorical claim "TFNBS is stable across (E, H)
choices; NBS is sensitive to its threshold τ".

Protocol
--------
On the SAME ComBat-harmonised ABIDE Age GLM (interest=age, confounds=
[sex, mean_fd]) — i.e. the canonical pipeline producing the headline
"885 surviving edges" claim — run seven enhancement variants:

  NBS-extent at τ ∈ {2.0, 2.5, 3.0, 3.5}        (4 variants)
  TFNBS at three published-default (E, H) settings   (3 variants)
    - Hao 2024              (0.4, 3.0)   FDR-calibrated regime, our pipeline default
    - Smith & Nichols 2009  (0.5, 2.0)   original TFCE / package default
    - Baggio 2018           (0.75, 3.0)  TFNBS paper's published default

We deliberately span the three community-used (E, H) defaults rather
than synthetic corners. The rhetorical claim is "even across the
published defaults, TFNBS is stable" — the strongest sensitivity
argument for a methods paper.

Each variant produces an FWER-corrected p-map at GPD-accelerated
n_permutations=200 (validated against 5000-perm empirical to within
|Δ(-log10 p)| ≤ 0.001 on >99 % of edges, per AbideValidationResults
§4.7).

The pairwise Jaccard at α=0.05 between the four NBS-τ masks is
expected to be low/moderate; between the three TFNBS-(E,H) masks
high. We also report Spearman on -log10 p (rank-based, robust to
detection-count asymmetry).

Output
------
results/combat/age/threshold_sensitivity/
  pmaps/<variant>.npz                   # per-variant {'positive', 'negative'} p-maps
  jaccard_matrix.csv                    # 7×7 pairwise Jaccard at α=0.05 (positive tail)
  jaccard_matrix_neg.csv                # 7×7 pairwise Jaccard at α=0.05 (negative tail)
  spearman_matrix.csv                   # 7×7 Spearman on -log10 p (positive tail)
  spearman_matrix_neg.csv               # 7×7 Spearman on -log10 p (negative tail)
  detection_counts.csv                  # per-variant n_sig and min_p, both tails
  threshold_sensitivity.png             # paper Fig 8 — paired heatmaps

Wall-clock estimate: ~10 min on 18 cores with GPD-200 acceleration.
"""

from __future__ import annotations

import argparse
import os
import time
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats as scistats

from conninfpy import compute_p_val_glm


HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(HERE, "abide_harmonized.npz")
OUT_DIR = os.path.join(HERE, "results", "combat", "age", "threshold_sensitivity")
PMAPS_DIR = os.path.join(OUT_DIR, "pmaps")


# Variant specifications: (label, method_key, kwargs, family)
def _variants(tfnbs_n_thresholds: int = 10):
    """Build the 7-variant list. tfnbs_n_thresholds is `n` in compute_p_val_glm."""
    return [
        ("NBS@2.0", "nbs", dict(threshold=2.0, nbs_stat="extent"), "NBS"),
        ("NBS@2.5", "nbs", dict(threshold=2.5, nbs_stat="extent"), "NBS"),
        ("NBS@3.0", "nbs", dict(threshold=3.0, nbs_stat="extent"), "NBS"),
        ("NBS@3.5", "nbs", dict(threshold=3.5, nbs_stat="extent"), "NBS"),
        ("TFNBS_Hao2024(E0.4,H3.0)", "tfnbs",
         dict(e=0.4, h=3.0, n=tfnbs_n_thresholds), "TFNBS"),
        ("TFNBS_SmithNichols2009(E0.5,H2.0)", "tfnbs",
         dict(e=0.5, h=2.0, n=tfnbs_n_thresholds), "TFNBS"),
        ("TFNBS_Baggio2018(E0.75,H3.0)", "tfnbs",
         dict(e=0.75, h=3.0, n=tfnbs_n_thresholds), "TFNBS"),
    ]


def _load():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} not found — run `python harmonize.py` first.")
    d = np.load(DATA_FILE, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _run_one_variant(conn, age, confounds, variant, n_perm, acceleration, seed):
    """Single-variant call. Returns dict with positive/negative p-maps + timing."""
    label, method, kw, family = variant
    print(f"\n  [{label}]  method={method}  kwargs={kw}")
    t0 = time.time()
    p = compute_p_val_glm(
        conn, interest=age, confounds=confounds,
        method=method, n_permutations=n_perm,
        acceleration=acceleration,
        use_mp=True, random_state=seed,
        **kw,
    )
    elapsed = time.time() - t0
    return p, elapsed


def _summarise_pmap(p_dict, alpha=0.05):
    """Per-tail (n_sig, min_p)."""
    out = {}
    for tail, mat in p_dict.items():
        N = mat.shape[0]
        iu = np.triu_indices(N, k=1)
        vec = mat[iu]
        out[tail] = dict(n_sig=int((vec < alpha).sum()), min_p=float(vec.min()))
    return out


def _pairwise_jaccard(masks):
    """masks: dict[label -> binary upper-triangle vector]. Returns square matrix."""
    keys = list(masks.keys())
    n = len(keys)
    J = np.full((n, n), np.nan)
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            if i == j:
                J[i, j] = 1.0
                continue
            mi, mj = masks[ki], masks[kj]
            union = (mi | mj).sum()
            J[i, j] = (mi & mj).sum() / union if union > 0 else 0.0
    return pd.DataFrame(J, index=keys, columns=keys)


def _pairwise_spearman(neglog_pvecs):
    """neglog_pvecs: dict[label -> 1D vector of -log10 p]."""
    keys = list(neglog_pvecs.keys())
    n = len(keys)
    S = np.full((n, n), np.nan)
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            if i == j:
                S[i, j] = 1.0
                continue
            r, _ = scistats.spearmanr(neglog_pvecs[ki], neglog_pvecs[kj])
            S[i, j] = r
    return pd.DataFrame(S, index=keys, columns=keys)


def _plot_threshold_sensitivity(jac_pos_df, jac_neg_df, variant_family, out_path):
    """Side-by-side: NBS-block heatmap vs TFNBS-block heatmap, both tails."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, df, title in [
        (axes[0], jac_pos_df, "Pairwise Jaccard @ α=0.05  (g2>g1, age↑→FC↑)"),
        (axes[1], jac_neg_df, "Pairwise Jaccard @ α=0.05  (g1>g2, age↑→FC↓)"),
    ]:
        im = ax.imshow(df.values, vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_xticks(range(len(df.columns)))
        ax.set_yticks(range(len(df.index)))
        ax.set_xticklabels(df.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(df.index, fontsize=8)
        for i in range(len(df.index)):
            for j in range(len(df.columns)):
                v = df.iloc[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.5 else "black", fontsize=7)
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        "ABIDE Age (n=764, ComBat-harmonised, GLM with sex+motion confounds)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="50 perms, no acceleration, smoke test only")
    ap.add_argument("--n-permutations", type=int, default=200)
    ap.add_argument("--no-acceleration", action="store_true",
                    help="Disable GPD; use empirical p-values")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.quick:
        n_perm = 50
        acceleration = None
        print("** QUICK MODE: n_perm=50, no acceleration **")
    else:
        n_perm = args.n_permutations
        acceleration = None if args.no_acceleration else "gpd"

    os.makedirs(PMAPS_DIR, exist_ok=True)
    data = _load()
    conn = data["connectivity_z_harm"]
    age = data["age"].astype(float)
    confounds = np.column_stack([data["sex"].astype(float), data["mean_fd"].astype(float)])
    print(f"Loaded ABIDE: n={conn.shape[0]}, N={conn.shape[1]}")
    print(f"Age range: {age.min():.1f}–{age.max():.1f}")
    print(f"n_permutations={n_perm}, acceleration={acceleration}")

    variants = _variants()
    print(f"\nRunning {len(variants)} variants...")

    pmaps = {}              # label -> dict('positive', 'negative')
    timings = {}            # label -> seconds
    families = {}           # label -> 'NBS' or 'TFNBS'
    for v in variants:
        label, method, kw, family = v
        families[label] = family
        p, elapsed = _run_one_variant(
            conn, age, confounds, v, n_perm=n_perm,
            acceleration=acceleration, seed=args.seed,
        )
        pmaps[label] = p
        timings[label] = elapsed
        np.savez(os.path.join(PMAPS_DIR, f"{label}.npz"), **p)
        summary = _summarise_pmap(p)
        for tail, info in summary.items():
            print(f"    [{tail}] n_sig={info['n_sig']}, min_p={info['min_p']:.4f}")
        print(f"    elapsed: {elapsed:.1f}s")

    # =========================================================================
    # Build masks + p-vectors
    # =========================================================================
    N = conn.shape[1]
    iu = np.triu_indices(N, k=1)

    masks_pos, masks_neg = {}, {}
    nlp_pos, nlp_neg = {}, {}
    for label, p in pmaps.items():
        masks_pos[label] = p["positive"][iu] < 0.05
        masks_neg[label] = p["negative"][iu] < 0.05
        nlp_pos[label] = -np.log10(np.maximum(p["positive"][iu], 1e-300))
        nlp_neg[label] = -np.log10(np.maximum(p["negative"][iu], 1e-300))

    # =========================================================================
    # Pairwise Jaccard + Spearman
    # =========================================================================
    jac_pos_df = _pairwise_jaccard(masks_pos)
    jac_neg_df = _pairwise_jaccard(masks_neg)
    spr_pos_df = _pairwise_spearman(nlp_pos)
    spr_neg_df = _pairwise_spearman(nlp_neg)

    jac_pos_df.to_csv(os.path.join(OUT_DIR, "jaccard_matrix.csv"))
    jac_neg_df.to_csv(os.path.join(OUT_DIR, "jaccard_matrix_neg.csv"))
    spr_pos_df.to_csv(os.path.join(OUT_DIR, "spearman_matrix.csv"))
    spr_neg_df.to_csv(os.path.join(OUT_DIR, "spearman_matrix_neg.csv"))

    # =========================================================================
    # Detection counts
    # =========================================================================
    rows = []
    for label, p in pmaps.items():
        info = _summarise_pmap(p)
        rows.append(dict(
            variant=label, family=families[label],
            n_sig_pos=info["positive"]["n_sig"], min_p_pos=info["positive"]["min_p"],
            n_sig_neg=info["negative"]["n_sig"], min_p_neg=info["negative"]["min_p"],
            elapsed_s=timings[label],
        ))
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "detection_counts.csv"), index=False)

    # =========================================================================
    # Headline: within-family vs between-family Jaccard means
    # =========================================================================
    nbs_labels = [k for k, f in families.items() if f == "NBS"]
    tfnbs_labels = [k for k, f in families.items() if f == "TFNBS"]

    def _within_mean(labels, df):
        vals = []
        for a, b in combinations(labels, 2):
            vals.append(df.loc[a, b])
        return np.mean(vals) if vals else np.nan

    print("\n" + "=" * 72)
    print("HEADLINE — within-family pairwise Jaccard")
    print("=" * 72)
    print(f"  NBS-τ swap (4 variants → 6 pairs):")
    print(f"    pos tail Jaccard mean = {_within_mean(nbs_labels, jac_pos_df):.3f}")
    print(f"    neg tail Jaccard mean = {_within_mean(nbs_labels, jac_neg_df):.3f}")
    print(f"  TFNBS-(E,H) swap (3 variants → 3 pairs):")
    print(f"    pos tail Jaccard mean = {_within_mean(tfnbs_labels, jac_pos_df):.3f}")
    print(f"    neg tail Jaccard mean = {_within_mean(tfnbs_labels, jac_neg_df):.3f}")
    print()
    print(f"  NBS-τ swap Spearman:")
    print(f"    pos tail Spearman mean = {_within_mean(nbs_labels, spr_pos_df):.3f}")
    print(f"    neg tail Spearman mean = {_within_mean(nbs_labels, spr_neg_df):.3f}")
    print(f"  TFNBS-(E,H) swap Spearman:")
    print(f"    pos tail Spearman mean = {_within_mean(tfnbs_labels, spr_pos_df):.3f}")
    print(f"    neg tail Spearman mean = {_within_mean(tfnbs_labels, spr_neg_df):.3f}")

    # =========================================================================
    # Plot
    # =========================================================================
    plot_path = os.path.join(OUT_DIR, "threshold_sensitivity.png")
    _plot_threshold_sensitivity(jac_pos_df, jac_neg_df, families, plot_path)
    print(f"\nFig saved: {plot_path}")
    print(f"All outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
