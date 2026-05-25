import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

HERE = Path(__file__).resolve().parent
ML_DIR = HERE / "results" / "ml"
PLOTS = HERE / "results" / "plots"

def main():
    csv_path = ML_DIR / "ml_feature_selection.csv"
    if not csv_path.exists(): return
    df = pd.read_csv(csv_path)
    all_auc_avg = df[df["selector"] == "all_edges"]["auc"].mean()
    sel_df = df[df["kind"] == "selector"].copy()
    avg_df = sel_df.groupby(["selector", "alpha"]).agg({"auc": "mean", "n_edges_used": "mean"}).reset_index()
    avg_df["d_vs_all"] = avg_df["auc"] - all_auc_avg
    best_avg = avg_df.loc[avg_df.groupby("selector")["auc"].idxmax()].sort_values("auc", ascending=False).reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    sns.barplot(data=best_avg, x="selector", y="auc", palette="viridis", ax=ax1)
    ax1.axhline(all_auc_avg, color="red", ls="--", label=f"All-edges baseline ({all_auc_avg:.3f})")
    ax1.set_title("Average Cross-Site AUC (IHB <-> Beijing China)"); ax1.set_ylim(0.8, 0.95); ax1.legend()
    
    sns.barplot(data=best_avg, x="selector", y="d_vs_all", palette="coolwarm", ax=ax2)
    ax2.set_title("Delta AUC vs. All-Edges Baseline"); ax2.set_ylabel("$\Delta$ AUC")
    
    plt.suptitle("Machine Learning Transfer Stability: Cross-Site Predictive Core (TF-NBS: E=0.4, H=3.0)\nAverage out-of-sample performance for topologically selected features", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(PLOTS / "plot7_ml_transfer_auc.png", dpi=150); plt.close()
    print(f"ML Transfer plot saved to {PLOTS / 'plot7_ml_transfer_auc.png'}")

if __name__ == "__main__":
    main()
