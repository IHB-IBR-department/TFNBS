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
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from examples.openclose_validation.openclose_loader import OpenCloseDataset


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
AUDIT = RESULTS / "audit"


def _load(name: str) -> Dict[str, np.ndarray]:
    path = RESULTS / name
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _upper(a): return a[np.triu_indices(a.shape[0], k=1)]


def _jaccard(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u > 0 else 0.0


def _jaccard_random(k_a, k_b, n):
    if k_a + k_b == 0:
        return 0.0
    exp_inter = k_a * k_b / n
    exp_union = k_a + k_b - exp_inter
    return float(exp_inter / exp_union) if exp_union > 0 else 0.0


def _spearman(pa, pb):
    r, _ = stats.spearmanr(-np.log10(pa + 1e-12), -np.log10(pb + 1e-12))
    return float(r)


def _topk(pa, pb, k):
    if k <= 0 or k > pa.size: return float("nan")
    ia = np.argsort(pa)[:k]; ib = np.argsort(pb)[:k]
    return float(len(np.intersect1d(ia, ib)) / k)


def _block_mass(p_full, lab):
    N = p_full.shape[0]; K = int(lab.max() + 1)
    iu = np.triu_indices(N, k=1)
    neglog = -np.log10(p_full + 1e-12)
    M = np.zeros((K, K))
    for i, j in zip(*iu):
        bi, bj = sorted([lab[i], lab[j]])
        M[bi, bj] += neglog[i, j]
    M = M + M.T; np.fill_diagonal(M, M.diagonal() / 2)
    return M


def _quartet(pa_full, pb_full, lab, alpha=0.05):
    pa = _upper(pa_full); pb = _upper(pb_full)
    sa = pa < alpha; sb = pb < alpha
    ka, kb, n = int(sa.sum()), int(sb.sum()), pa.size

    # Fisher's exact "overlap better than chance"
    tp = int((sa & sb).sum()); fp = int((sa & ~sb).sum())
    fn = int((~sa & sb).sum()); tn = int((~sa & ~sb).sum())
    _, fex = stats.fisher_exact([[tp, fp], [fn, tn]], alternative="greater")

    ma = _block_mass(pa_full, lab); mb = _block_mass(pb_full, lab)
    tri = np.triu_indices_from(ma)
    bm_r, _ = stats.pearsonr(ma[tri], mb[tri])

    return {
        "n_sig_a": ka, "n_sig_b": kb,
        "jaccard": _jaccard(sa, sb),
        "jaccard_random": _jaccard_random(ka, kb, n),
        "fisher_exact_p": float(fex),
        "spearman_neglog10p": _spearman(pa, pb),
        "top10": _topk(pa, pb, 10),
        "top50": _topk(pa, pb, 50),
        "top100": _topk(pa, pb, 100),
        "top500": _topk(pa, pb, 500),
        "block_mass_pearson": float(bm_r),
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
