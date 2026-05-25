import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from examples.openclose_validation.openclose_loader import OpenCloseDataset

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PLOTS = RESULTS / "plots"

def _block_mass(p_map, net_labels):
    K = int(net_labels.max() + 1); mass = np.zeros((K, K)); nlp = -np.log10(np.maximum(p_map, 1e-300))
    iu = np.triu_indices(p_map.shape[0], k=1)
    for idx in range(len(iu[0])):
        i, j = iu[0][idx], iu[1][idx]; ni, nj = int(net_labels[i]), int(net_labels[j])
        mass[ni, nj] += nlp[i, j]
        if ni != nj: mass[nj, ni] += nlp[i, j]
    return mass

def main():
    ihb = np.load(RESULTS / "ihb_paired_tfnbs.npz")
    china = np.load(RESULTS / "china_paired_tfnbs_run0.npz")
    china_retest = np.load(RESULTS / "china_retest_tfnbs.npz")
    ds = OpenCloseDataset.load("ihb"); lab = ds.net_labels; names = ds.get_label_names(); K = len(names)
    fig, axes = plt.subplots(2, 3, figsize=(21, 12))
    for r, tail in enumerate(["positive", "negative"]):
        ma = _block_mass(ihb[tail], lab)
        mb = _block_mass(china[tail], lab)
        mc = _block_mass(china_retest[tail], lab)
        vmax = max(ma.max(), mb.max(), mc.max())
        for c, (M, title) in enumerate([(ma, f"IHB {tail}"), (mb, f"Beijing China State {tail}"), (mc, f"Beijing China Retest (Null) {tail}")]):
            im = axes[r, c].imshow(M, cmap="viridis", vmin=0, vmax=vmax)
            axes[r, c].set_xticks(range(K)); axes[r, c].set_xticklabels(names, rotation=60, ha="right", fontsize=8)
            axes[r, c].set_yticks(range(K)); axes[r, c].set_yticklabels(names, fontsize=8)
            axes[r, c].set_title(title); fig.colorbar(im, ax=axes[r, c], fraction=0.046)
    plt.suptitle("Block-Mass Matrix Convergence: Cross-Cohort Replication & Specificity (TF-NBS: E=0.4, H=3.0)\nComparison of topological integration patterns (Σ −log10 p)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(PLOTS / "plot2_block_mass_convergence.png", dpi=150); plt.close()

if __name__ == "__main__":
    main()
