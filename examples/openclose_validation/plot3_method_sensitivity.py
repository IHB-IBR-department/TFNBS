import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
ML_DIR = RESULTS / "ml"
PLOTS = RESULTS / "plots"

ALPHA = 0.05
METHODS = ["tstat", "tfnbs", "ni_tfnbs", "fbc_tfnbs", "nbs_20", "nbs_30", "bh_fdr"]
PRETTY = {
    "tstat": "t-stat",
    "tfnbs": "TFNBS",
    "ni_tfnbs": "NI-TFNBS",
    "fbc_tfnbs": "FBC-TFNBS",
    "nbs_20": "NBS τ=2.0",
    "nbs_30": "NBS τ=3.0",
    "bh_fdr": "BH-FDR"
}

def _load_data(cohort):
    masks = {}; stats_maps = {}
    for m in METHODS:
        path = ML_DIR / f"pmap_{cohort}_{m}.npz"
        if not path.exists():
            print(f"  missing: {path}")
            continue
        d = np.load(path); p_min = None
        for key in d.files:
            if p_min is None: p_min = d[key].copy()
            else: p_min = np.minimum(p_min, d[key])
        iu = np.triu_indices(p_min.shape[0], k=1)
        p_vec = p_min[iu]; masks[m] = p_vec < ALPHA; stats_maps[m] = -np.log10(np.maximum(p_vec, 1e-300))
    return masks, stats_maps

def _matrices(masks, stats_maps):
    n = len(METHODS); J = np.full((n, n), np.nan); S = np.full((n, n), np.nan)
    for i, mi in enumerate(METHODS):
        for j, mj in enumerate(METHODS):
            if mi in masks and mj in masks:
                union = (masks[mi] | masks[mj]).sum()
                J[i, j] = (masks[mi] & masks[mj]).sum() / union if union > 0 else 1.0
            if mi in stats_maps and mj in stats_maps:
                S[i, j], _ = stats.spearmanr(stats_maps[mi], stats_maps[mj])
    return J, S

def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), constrained_layout=True)
    for col, cohort in enumerate(["ihb", "china"]):
        masks, stats_maps = _load_data(cohort); J, S = _matrices(masks, stats_maps)
        labels = [f"{PRETTY[m]}" for m in METHODS]
        
        sns.heatmap(J, annot=True, fmt=".2f", cmap="viridis", xticklabels=labels, yticklabels=labels, vmin=0, vmax=1, ax=axes[0, col])
        axes[0, col].set_title(f"{cohort.upper()} — Jaccard Similarity (Sig. Edges)", fontsize=14, fontweight='bold')
        
        sns.heatmap(S, annot=True, fmt=".2f", cmap="magma", xticklabels=labels, yticklabels=labels, vmin=0, vmax=1, ax=axes[1, col])
        axes[1, col].set_title(f"{cohort.upper()} — Spearman Correlation (Full Gradient)", fontsize=14, fontweight='bold')
        
    plt.suptitle("Method Sensitivity & Convergence: Open vs Close Paired Contrast (TF-NBS: E=0.4, H=3.0)\n"
                 "Stability of discovery across both IHB and Beijing cohorts", fontsize=18, fontweight='bold')
    
    out_path = PLOTS / "plot3_method_sensitivity_dual.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"saved: {out_path}")

if __name__ == "__main__":
    main()
