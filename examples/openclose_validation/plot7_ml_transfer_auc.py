import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

HERE = Path(__file__).resolve().parent
ML_DIR = HERE / "results" / "ml"
PLOTS = HERE / "results" / "plots"

PRETTY = {
    "tstat": "t-stat",
    "tfnbs": "TFNBS",
    "ni_tfnbs": "NI-TFNBS",
    "fbc_tfnbs": "FBC-TFNBS",
    "nbs@2.0": "NBS τ=2.0",
    "nbs@3.0": "NBS τ=3.0",
    "bh_fdr": "BH-FDR",
    "cnbs": "cNBS"
}

def main():
    csv_path = ML_DIR / "ml_feature_selection.csv"
    if not csv_path.exists():
        print(f"Missing results: {csv_path}")
        return
    df = pd.read_csv(csv_path)
    all_auc_avg = df[df["selector"] == "all_edges"]["auc"].mean()
    sel_df = df[df["kind"] == "selector"].copy()
    
    # Pre-aggregate to get the best alpha per selector
    avg_df = sel_df.groupby(["selector", "alpha"]).agg({"auc": "mean", "n_edges_used": "mean"}).reset_index()
    avg_df["d_vs_all"] = avg_df["auc"] - all_auc_avg
    avg_df["pretty_name"] = avg_df["selector"].map(lambda x: PRETTY.get(x, x))
    
    best_avg = avg_df.loc[avg_df.groupby("selector")["auc"].idxmax()].sort_values("auc", ascending=False).reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    
    sns.barplot(data=best_avg, x="pretty_name", y="auc", hue="pretty_name", palette="viridis", ax=ax1, legend=False)
    ax1.axhline(all_auc_avg, color="red", ls="--", label=f"All-edges baseline ({all_auc_avg:.3f})")
    ax1.set_title("Average Cross-Site AUC (IHB <-> Beijing China)", fontsize=14, fontweight='bold')
    ax1.set_ylim(0.8, 0.95); ax1.legend()
    ax1.set_xlabel("Feature Selector", fontsize=12)
    ax1.set_ylabel("ROC-AUC", fontsize=12)
    ax1.tick_params(axis='x', rotation=30)
    
    sns.barplot(data=best_avg, x="pretty_name", y="d_vs_all", hue="pretty_name", palette="coolwarm", ax=ax2, legend=False)
    ax2.set_title("$\Delta$ AUC vs. All-Edges Baseline", fontsize=14, fontweight='bold')
    ax2.set_ylabel("$\Delta$ ROC-AUC", fontsize=12)
    ax2.set_xlabel("Feature Selector", fontsize=12)
    ax2.tick_params(axis='x', rotation=30)
    
    plt.suptitle("Machine Learning Transfer Stability: Cross-Site Predictive Core (TF-NBS: E=0.4, H=3.0)\n"
                 "Average out-of-sample performance for topologically selected features", fontsize=16, fontweight='bold')
    
    out_path = PLOTS / "plot7_ml_transfer_auc.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"ML Transfer plot saved to {out_path}")

if __name__ == "__main__":
    main()
