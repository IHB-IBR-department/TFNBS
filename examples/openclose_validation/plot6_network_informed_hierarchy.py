"""
Hierarchy of Network Constraint for the Open-Close paired contrast.

Mirrors the ABIDE Plot 9 layout, but for the paired Open-vs-Close (run 0)
contrast on IHB and China cohorts, plus the joint harmonized pooled dataset.

Three methods, ordered by how strongly the network prior constrains the
result:

  - TFNBS       (unrestricted topological enhancement)
  - NI-TFNBS    (soft per-edge network prior)
  - FBC-TFNBS   (functional-block clustering)
  - NBS 3.0     (univariate clustering baseline)

For each (cohort, method) we threshold the saved p-map at α = 0.05 on each
tail, collapse positive ∪ negative, and aggregate significant edges into a
Yeo-7 block × block count heatmap.

Input  : examples/openclose_validation/results/ml/pmap_{cohort}_{method}.npz
Output : examples/openclose_validation/results/plots/plot6_network_informed_hierarchy.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from examples.openclose_validation.openclose_loader import OpenCloseDataset


HERE = Path(__file__).resolve().parent
ML_DIR = HERE / "results" / "ml"
PLOTS = HERE / "results" / "plots"

ALPHA = 0.05

METHODS = [
    ("tfnbs",    "TFNBS (Unrestricted)",           "Greens"),
    ("ni_tfnbs", "NI-TFNBS (Soft Prior)",          "YlGn"),
    ("fbc_tfnbs","FBC-TFNBS",                      "Oranges"),
    ("nbs_30",   "NBS τ=3.0 (Fixed Threshold)",    "Blues"),
]

COHORTS = ["ihb", "china", "pooled"]


def _block_counts(pmap_path: Path, net_labels: np.ndarray, n_nets: int) -> np.ndarray:
    """Count significant edges (pos ∪ neg) per Yeo-7 block pair."""
    if not pmap_path.exists():
        print(f"  missing: {pmap_path}")
        return np.zeros((n_nets, n_nets), dtype=int)
    d = np.load(pmap_path)
    # Older runs save tails as 'g2>g1'/'g1>g2'; newer ones as 'positive'/'negative'.
    sig = None
    for key in d.files:
        arr = d[key]
        if sig is None:
            sig = np.zeros_like(arr, dtype=bool)
        sig |= arr < ALPHA
    N = sig.shape[0]
    iu = np.triu_indices(N, k=1)
    mat = np.zeros((n_nets, n_nets), dtype=int)
    for idx in np.where(sig[iu])[0]:
        i, j = iu[0][idx], iu[1][idx]
        ni, nj = int(net_labels[i]), int(net_labels[j])
        mat[ni, nj] += 1
        if ni != nj:
            mat[nj, ni] += 1
    return mat


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)

    ds = OpenCloseDataset.load("ihb")  # net_labels are identical across cohorts (182-ROI mask)
    net_labels = np.asarray(ds.net_labels, dtype=int)
    names = ds.get_label_names()
    n_nets = int(net_labels.max() + 1)

    fig, axes = plt.subplots(
        len(COHORTS), len(METHODS), figsize=(24, 18),
    )

    for r, cohort in enumerate(COHORTS):
        for c, (method, title, cmap) in enumerate(METHODS):
            pmap_path = ML_DIR / f"pmap_{cohort}_{method}.npz"
            mat = _block_counts(pmap_path, net_labels, n_nets)
            n_sig = int(mat.sum() / 2 + np.diag(mat).sum() / 2)
            ax = axes[r, c]
            sns.heatmap(
                mat, annot=True, fmt="g", cmap=cmap,
                xticklabels=names, yticklabels=names,
                ax=ax, cbar_kws={"shrink": 0.7},
            )
            pretty_cohort = cohort.upper() if cohort != "pooled" else "JOINT HARMONIZED"
            ax.set_title(f"{pretty_cohort} (OPEN vs CLOSE)\n{title} (n_sig={n_sig})",
                         fontsize=12, fontweight='bold')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right",
                               fontsize=9)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    fig.suptitle(
        "Hierarchy of Network Constraint — Open vs Close paired (α=0.05)\n"
        "Columns left → right: increasing network prior strength; Rows: discovery cohorts",
        fontsize=16, fontweight='bold'
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    out = PLOTS / "plot6_network_informed_hierarchy.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
