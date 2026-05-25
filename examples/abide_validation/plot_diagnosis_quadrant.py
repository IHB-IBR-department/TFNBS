import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

def plot_dx_quadrant():
    print("Generating Diagnosis Methodology Comparison (4-Panel)...")
    data = np.load(RESULTS_DIR / "abide_prepared.npz", allow_pickle=True)
    network_order = list(data["network_order"])
    n_nets = len(network_order)
    
    def aggregate(csv_path):
        if not csv_path.exists(): return np.zeros((n_nets, n_nets))
        df = pd.read_csv(csv_path)
        net_labels = data["net_labels"]
        mat = np.zeros((n_nets, n_nets))
        for _, row in df.iterrows():
            i, j = int(row["roi_i"]), int(row["roi_j"])
            ni, nj = net_labels[i], net_labels[j]
            mat[ni, nj] += 1
            if ni != nj: mat[nj, ni] += 1
        return mat

    paths = {
        "Naive (No Site Control)": RESULTS_DIR / "diagnosis/naive/dx_naive_edges.csv",
        "Strategy E (Site Dummies ONLY)": RESULTS_DIR / "diagnosis/strategy_e/dx_strategy_e_edges.csv",
        "ComBat ONLY (No Dummies)": RESULTS_DIR / "diagnosis/combat_only/dx_combat_only_edges.csv",
        "Strategy D (ComBat + Dummies)": RESULTS_DIR / "diagnosis/dx_strategy_d_edges.csv"
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    cmaps = ["Oranges", "Purples", "Blues", "Greens"]
    
    for i, (title, path) in enumerate(paths.items()):
        mat = aggregate(path)
        n_sig = int(mat.sum()/2 + np.diag(mat).sum()/2)
        sns.heatmap(mat, annot=True, fmt="g", cmap=cmaps[i], xticklabels=network_order, yticklabels=network_order, ax=axes[i])
        axes[i].set_title(f"{title}\n(n_sig={n_sig})", fontsize=12)
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')

    plt.suptitle("The Evolution of Site-Aware Inference in ABIDE (TF-NBS: E=0.4, H=3.0)\nComBat: Strategy D (Nuisance-only)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(PLOTS_DIR / "plot6_diagnosis_methodology.png", dpi=150)
    plt.close()
    print("Methodology Comparison plot generated.")

if __name__ == "__main__":
    plot_dx_quadrant()
