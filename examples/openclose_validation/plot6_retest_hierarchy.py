"""
Hierarchy of Network Constraint for the Open-Close paired contrast (Retest/Null).

Shows the specificity of network methods by applying them to the 'close-close'
empirical null contrast. Ideally, these heatmaps should be nearly empty.
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

COHORTS = ["china_retest"]


def _block_counts(pmap_path: Path, net_labels: np.ndarray, n_nets: int) -> np.ndarray:
    """Count significant edges (pos ∪ neg) per Yeo-7 block pair."""
    if not pmap_path.exists():
        print(f"  missing: {pmap_path}")
        return np.zeros((n_nets, n_nets), dtype=int)
    d = np.load(pmap_path)
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

    ds = OpenCloseDataset.load("ihb")
    net_labels = np.asarray(ds.net_labels, dtype=int)
    names = ds.get_label_names()
    n_nets = int(net_labels.max() + 1)

    fig, axes = plt.subplots(
        1, len(METHODS), figsize=(24, 7),
    )

    for c, (method, title, cmap) in enumerate(METHODS):
        pmap_path = ML_DIR / f"pmap_china_retest_{method}.npz"
        mat = _block_counts(pmap_path, net_labels, n_nets)
        n_sig = int(mat.sum() / 2 + np.diag(mat).sum() / 2)
        ax = axes[c]
        sns.heatmap(
            mat, annot=True, fmt="g", cmap=cmap,
            xticklabels=names, yticklabels=names,
            ax=ax, cbar_kws={"shrink": 0.7},
        )
        ax.set_title(f"CHINA RETEST (NULL) — {title}\n(n_sig={n_sig})",
                     fontsize=12, fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right",
                           fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    fig.suptitle(
        "Specificity of Network Constraint — Empirical Null (Close vs Close paired, α=0.05)\n"
        "Ideally these heatmaps should show near-zero edges.",
        fontsize=16, fontweight='bold'
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    out = PLOTS / "plot6_retest_hierarchy.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
