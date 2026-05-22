"""
§3.7 agreement audit — IHB vs China paired TFNBS p-maps on the §2.5 quartet.

Computes the four agreement metrics (Jaccard with random baseline + Fisher's exact,
Spearman on −log10 p, top-k concordance, Yeo-7 block-mass Pearson) between:

  A. IHB paired map  vs  China (run 0) paired map — cross-cohort replication
  B. China run 0 paired map vs China run 0↔run 1 retest map — should show
     that "open vs close" differs from "close vs close" (sanity + test-retest
     characterisation)

Also reports per-tail Yeo-7 block-mass matrices side-by-side for IHB & China
as a secondary figure.

Outputs: :file:`examples/openclose_validation/results/audit/*`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from examples.openclose_validation.openclose_loader import OpenCloseDataset

# Shared agreement helpers (examples/_audit_utils.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _audit_utils import (  # noqa: E402
    upper_triangle as _upper,
    jaccard as _jaccard,
    jaccard_random_baseline as _jaccard_random,
    fisher_exact_2x2 as _fisher_exact,
    spearman_neglog10 as _spearman,
    topk_concordance as _topk,
    block_mass as _block_mass,
    block_mass_correlation as _block_mass_corr,
)


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
AUDIT = RESULTS / "audit"


def _load(name: str) -> Dict[str, np.ndarray]:
    path = RESULTS / name
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _quartet(pa_full, pb_full, lab, alpha=0.05):
    """§2.5 agreement bundle keyed for this script's historical schema."""
    pa = _upper(pa_full); pb = _upper(pb_full)
    sa = pa < alpha; sb = pb < alpha
    ka, kb = int(sa.sum()), int(sb.sum())
    return {
        "n_sig_a": ka, "n_sig_b": kb,
        "jaccard": _jaccard(sa, sb),
        "jaccard_random": _jaccard_random(ka, kb, pa.size),
        "fisher_exact_p": _fisher_exact(sa, sb),
        "spearman_neglog10p": _spearman(pa, pb),
        "top10": _topk(pa, pb, 10),
        "top50": _topk(pa, pb, 50),
        "top100": _topk(pa, pb, 100),
        "top500": _topk(pa, pb, 500),
        "block_mass_pearson": _block_mass_corr(pa_full, pb_full, lab),
    }


def _plot_block_panel(ihb_p, china_p, lab, names, outpath):
    """Side-by-side Yeo-7 block-mass heatmaps per tail."""
    K = int(lab.max() + 1)
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    for row, tail in enumerate(["g2>g1", "g1>g2"]):
        if tail not in ihb_p or tail not in china_p:
            continue
        ma = _block_mass(ihb_p[tail], lab)
        mb = _block_mass(china_p[tail], lab)
        vmax = max(ma.max(), mb.max())
        for ax, M, title in [(axes[row, 0], ma, f"IHB  {tail}"),
                             (axes[row, 1], mb, f"China run0  {tail}")]:
            im = ax.imshow(M, cmap="viridis", vmin=0, vmax=vmax)
            ax.set_xticks(range(K)); ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
            ax.set_yticks(range(K)); ax.set_yticklabels(names, fontsize=8)
            ax.set_title(title)
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("§3.7 Block-mass (Σ −log10 p) per Yeo-7 block pair — IHB vs China")
    fig.tight_layout()
    fig.savefig(outpath, dpi=140); plt.close(fig)


def main() -> None:
    os.makedirs(AUDIT, exist_ok=True)
    # Load p-maps
    ihb = _load("ihb_paired_tfnbs.npz")
    china = _load("china_paired_tfnbs_run0.npz")
    retest = _load("china_retest_tfnbs.npz")

    # Common labels (both cohorts drop same 18 ROIs → identical 182-length net_labels)
    ds_ihb = OpenCloseDataset.load("ihb")
    lab = ds_ihb.net_labels
    names = ds_ihb.get_label_names()

    # ---- Pairwise quartet ----------------------------------------
    rows = []
    pairs = [
        ("IHB vs China run0", ihb, china),
        ("China run0 paired vs China retest", china, retest),
    ]
    for label, pa_full, pb_full in pairs:
        for tail in ("g2>g1", "g1>g2"):
            if tail not in pa_full or tail not in pb_full:
                continue
            q = _quartet(pa_full[tail], pb_full[tail], lab)
            rows.append({"comparison": label, "tail": tail, **q})

    df = pd.DataFrame(rows)
    out_csv = AUDIT / "openclose_agreement.csv"
    df.to_csv(out_csv, index=False)
    key_cols = ["comparison", "tail", "n_sig_a", "n_sig_b",
                "jaccard", "jaccard_random", "fisher_exact_p",
                "spearman_neglog10p", "top100", "block_mass_pearson"]
    print("\n§3.7 agreement quartet (rounded):")
    print(df[key_cols].to_string(index=False, float_format="%.4f"))
    print(f"\nsaved: {out_csv}")

    # ---- Block-mass panel ----------------------------------------
    _plot_block_panel(ihb, china, lab, names, AUDIT / "block_mass_panel.png")
    print(f"saved: {AUDIT / 'block_mass_panel.png'}")


if __name__ == "__main__":
    main()
