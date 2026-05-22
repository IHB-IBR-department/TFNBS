"""
§3.2 within-site replication audit.

For each analysis type (age / naive / GLM), compute the §2.5 quartet
between NYU and UM_1 p-maps, plus a full-cohort reference. Produces a
single summary CSV and a four-panel figure.
"""

from __future__ import annotations

import os
import sys
from glob import glob
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Shared agreement helpers (examples/_audit_utils.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _audit_utils import (  # noqa: E402
    upper_triangle as upper,
    jaccard,
    jaccard_random_baseline as j_random_baseline,
    fisher_exact_2x2,
    spearman_neglog10 as spearman,
    topk_concordance as top_k,
    block_mass_correlation as block_corr,
)


HERE = os.path.dirname(os.path.abspath(__file__))
WS_ROOT = os.path.join(HERE, "results", "within_site")
AUDIT_DIR = os.path.join(WS_ROOT, "audit")
PREPARED = os.path.join(HERE, "abide_prepared.npz")

# Where the full-cohort equivalents live (for context; what the within-site should replicate)
FULL_DIR_AGE = os.path.join(HERE, "results", "combat", "age")
FULL_DIR_DX_COMBAT = os.path.join(HERE, "results", "combat")
FULL_DIR_DX_FL = os.path.join(HERE, "results")


def load_pmap(path):
    d = np.load(path, allow_pickle=True)
    out = {}
    for k in d.files:
        if k in ("positive", "negative"):
            out[k] = d[k]
        elif k == "g2>g1":
            out["positive"] = d[k]
        elif k == "g1>g2":
            out["negative"] = d[k]
    return out


def quartet(pa_full, pb_full, lab, alpha=0.05):
    """§2.5 agreement bundle keyed for this script's historical schema."""
    pa = upper(pa_full); pb = upper(pb_full)
    sa = pa < alpha; sb = pb < alpha
    ka, kb = int(sa.sum()), int(sb.sum())
    return {
        "n_sig_a": ka, "n_sig_b": kb,
        "jaccard": jaccard(sa, sb),
        "jaccard_random": j_random_baseline(ka, kb, pa.size),
        "fisher_exact_p": fisher_exact_2x2(sa, sb),
        "spearman": spearman(pa, pb),
        "top10": top_k(pa, pb, 10),
        "top50": top_k(pa, pb, 50),
        "top100": top_k(pa, pb, 100),
        "top500": top_k(pa, pb, 500),
        "block_corr": block_corr(pa_full, pb_full, lab),
    }


def main():
    os.makedirs(AUDIT_DIR, exist_ok=True)
    prep = np.load(PREPARED, allow_pickle=True)
    lab = prep["net_labels"]

    analyses = {
        "age_tfnbs":      "Age → FC TFNBS",
        "dx_naive_tfnbs": "DX naive two-sample TFNBS",
        "dx_glm_tfnbs":   "DX GLM TFNBS",
    }

    # Collect: NYU vs UM_1, plus NYU vs full-cohort (for sanity)
    rows = []
    for stem, label in analyses.items():
        nyu = os.path.join(WS_ROOT, "nyu", f"{stem}.npz")
        um = os.path.join(WS_ROOT, "um1", f"{stem}.npz")
        if not (os.path.exists(nyu) and os.path.exists(um)):
            print(f"  missing: {stem}, skipping")
            continue
        pa_all = load_pmap(nyu); pb_all = load_pmap(um)
        for tail in ("positive", "negative"):
            if tail not in pa_all or tail not in pb_all:
                continue
            q = quartet(pa_all[tail], pb_all[tail], lab)
            rows.append({
                "analysis": stem, "label": label,
                "comparison": "NYU vs UM_1",
                "tail": tail,
                **q,
            })

        # NYU vs full cohort — for context
        full_candidates = {
            "age_tfnbs": os.path.join(FULL_DIR_AGE, "age_tfnbs.npz"),
            "dx_naive_tfnbs": os.path.join(FULL_DIR_DX_COMBAT, "analysis_1_naive_combat.npz"),
            "dx_glm_tfnbs": os.path.join(FULL_DIR_DX_COMBAT, "analysis_2_glm_combat.npz"),
        }
        full_path = full_candidates.get(stem)
        if full_path and os.path.exists(full_path):
            pfull = load_pmap(full_path)
            for tail in ("positive", "negative"):
                if tail not in pa_all or tail not in pfull:
                    continue
                q = quartet(pa_all[tail], pfull[tail], lab)
                rows.append({
                    "analysis": stem, "label": label,
                    "comparison": "NYU vs full-cohort",
                    "tail": tail,
                    **q,
                })
                q = quartet(pb_all[tail], pfull[tail], lab)
                rows.append({
                    "analysis": stem, "label": label,
                    "comparison": "UM_1 vs full-cohort",
                    "tail": tail,
                    **q,
                })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(AUDIT_DIR, "within_site_agreement.csv"), index=False)
    print("\nWithin-site §2.5 agreement quartet:\n")
    key_cols = ["analysis", "comparison", "tail", "n_sig_a", "n_sig_b",
                "jaccard", "jaccard_random", "spearman", "top100", "block_corr"]
    print(df[key_cols].to_string(index=False, float_format="%.3f"))

    # ---- 4-panel figure: Spearman, block_corr, Jaccard, top100 ----
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    metrics = [("spearman", "Spearman(−log10 p)", 0, 1),
               ("block_corr", "Block-mass Pearson", -0.5, 1),
               ("top100", "Top-100 concordance", 0, 1),
               ("jaccard", "Jaccard @ 0.05", 0, 1)]

    subset = df[df["comparison"] == "NYU vs UM_1"]
    for ax, (col, title, lo, hi) in zip(axes, metrics):
        if subset.empty:
            ax.text(0.5, 0.5, "no data", ha="center"); continue
        pivot = subset.pivot_table(index="analysis", columns="tail", values=col, aggfunc="first")
        pivot = pivot.reindex(list(analyses.keys()))
        x = np.arange(len(pivot))
        w = 0.35
        ax.bar(x - w/2, pivot["positive"], w, label="positive tail", color="C0")
        ax.bar(x + w/2, pivot["negative"], w, label="negative tail", color="C1")
        ax.axhline(0, color="k", lw=0.5)
        if col == "jaccard":
            # overlay random baselines
            jr = subset.pivot_table(index="analysis", columns="tail", values="jaccard_random", aggfunc="first")
            jr = jr.reindex(list(analyses.keys()))
            ax.plot(x - w/2, jr["positive"], "o", color="C0", alpha=0.6, label="pos random")
            ax.plot(x + w/2, jr["negative"], "o", color="C1", alpha=0.6, label="neg random")
        ax.set_xticks(x)
        ax.set_xticklabels([analyses[a] for a in pivot.index], rotation=15, ha="right", fontsize=9)
        ax.set_title(title)
        ax.set_ylim(lo, hi)
        ax.legend(fontsize=7)
    fig.suptitle("§3.2 within-site replication: NYU vs UM_1 (ABIDE)")
    fig.tight_layout()
    out = os.path.join(AUDIT_DIR, "within_site_panel.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nFigure: {out}")


if __name__ == "__main__":
    main()
