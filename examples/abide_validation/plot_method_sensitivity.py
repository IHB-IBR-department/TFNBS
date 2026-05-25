import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True, parents=True)

def plot_method_sensitivity():
    print("Generating Method Sensitivity & Convergence (Jaccard + Spearman)...")
    methods_dir = RESULTS_DIR / "age_development" / "methods"
    csv_files = list(methods_dir.glob("age_*_edges.csv"))
    
    if len(csv_files) == 0:
        print(f"Skipping: No method CSV files found in {methods_dir}.")
        return
        
    method_names = []
    edge_sets = {}
    full_stats = {}
    
    for f in csv_files:
        name = f.stem.replace("age_", "").replace("_edges", "")
        method_names.append(name)
        
        # 1. Edge sets for Jaccard
        df = pd.read_csv(f)
        edges = set()
        for _, row in df.iterrows():
            e = tuple(sorted([int(row["roi_i"]), int(row["roi_j"])]))
            edges.add(e)
        edge_sets[name] = edges
        
        # 2. Full statistics for Spearman
        npz_path = methods_dir / f"age_{name}.npz"
        if npz_path.exists():
            with np.load(npz_path) as d:
                if "positive" in d and "negative" in d:
                    pos = 1.0 - d["positive"]
                    neg = 1.0 - d["negative"]
                    iu = np.triu_indices(pos.shape[0], k=1)
                    vec = pos[iu] - neg[iu]
                    full_stats[name] = vec

    # Custom sort to group related methods
    preferred_order = ["tstat", "nbs_2.0", "nbs_3.0", "tfnbs", "ni_tfnbs", "fbc_tfnbs", "cnbs"]
    method_names = [m for m in preferred_order if m in method_names] + sorted([m for m in method_names if m not in preferred_order])

    n_methods = len(method_names)
    jaccard_mat = np.zeros((n_methods, n_methods))
    spearman_mat = np.zeros((n_methods, n_methods))
    
    for i, m1 in enumerate(method_names):
        set1 = edge_sets[m1]
        v1 = full_stats.get(m1)
        for j, m2 in enumerate(method_names):
            set2 = edge_sets[m2]
            v2 = full_stats.get(m2)
            
            # Jaccard
            if len(set1) == 0 and len(set2) == 0:
                jaccard_mat[i, j] = 1.0
            elif len(set1) == 0 or len(set2) == 0:
                jaccard_mat[i, j] = 0.0
            else:
                intersection = len(set1.intersection(set2))
                union = len(set1.union(set2))
                jaccard_mat[i, j] = intersection / union
            
            # Spearman
            if v1 is not None and v2 is not None:
                rho, _ = stats.spearmanr(v1, v2)
                spearman_mat[i, j] = rho
            else:
                spearman_mat[i, j] = np.nan
                
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    sns.heatmap(jaccard_mat, annot=True, fmt=".2f", cmap="viridis", 
                xticklabels=method_names, yticklabels=method_names, ax=ax1)
    ax1.set_title("A: Edge-Set Jaccard Similarity (Sig. Edges)")
    
    sns.heatmap(spearman_mat, annot=True, fmt=".2f", cmap="magma", 
                xticklabels=method_names, yticklabels=method_names, ax=ax2)
    ax2.set_title("B: Spearman Correlation (Full Statistical Maps)")
    
    plt.suptitle("Method Sensitivity & Convergence (Age Control, TF-NBS: E=0.4, H=3.0)\nComBat: Strategy D (Nuisance-only)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "plot4_dual.png", dpi=150)
    plt.close()
    print(f"Method sensitivity plot saved to {PLOTS_DIR / 'plot4_dual.png'}")

if __name__ == "__main__":
    plot_method_sensitivity()
