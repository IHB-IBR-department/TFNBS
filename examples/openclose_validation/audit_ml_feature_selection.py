"""
§3.8 audit — summarise ΔAUC vs random-subset and vs univariate-top-k baselines.

Produces:
  - `delta_auc.csv`            : one row per (direction, selector, alpha) with ΔAUCs
  - `best_cells.csv`           : per-direction best-alpha selector performance
  - `delta_auc_panel.png`      : two-panel figure, one per direction,
                                 ΔAUC(selector - random) and ΔAUC(selector - uni_topk)
                                 as grouped bars over α

Reads from: `examples/openclose_validation/results/ml/ml_feature_selection.csv`
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ML_DIR = HERE / "results" / "ml"


def main() -> None:
    df = pd.read_csv(ML_DIR / "ml_feature_selection.csv")

    # Separate selector rows from baselines
    sel = df[df["kind"] == "selector"].copy()
    base = df[df["kind"] == "baseline"].copy()

    # All-edges baseline AUC per direction
    all_auc = base[base["selector"] == "all_edges"].set_index("direction")["auc"].to_dict()

    # Build delta-AUC rows: for each (direction, selector, alpha) find matching
    # random_matched_k{k} and uni_topk{k} baselines.
    rows = []
    for _, r in sel.iterrows():
        d, s, a, k, sel_auc = r["direction"], r["selector"], r["alpha"], int(r["n_edges_used"]), r["auc"]
        if k == 0 or np.isnan(sel_auc):
            continue
        rand_key = f"random_matched_k{k}"
        uni_key = f"uni_topk{k}"
        rand_row = base[(base["direction"] == d) & (base["selector"] == rand_key) & (base["alpha"] == a)]
        uni_row = base[(base["direction"] == d) & (base["selector"] == uni_key) & (base["alpha"] == a)]
        rand_auc = rand_row["auc"].values[0] if not rand_row.empty else np.nan
        uni_auc = uni_row["auc"].values[0] if not uni_row.empty else np.nan
        rows.append({
            "direction": d, "selector": s, "alpha": a, "k": k,
            "sel_auc": sel_auc, "rand_auc": rand_auc, "uni_auc": uni_auc,
            "all_auc": all_auc.get(d, np.nan),
            "d_vs_rand": sel_auc - rand_auc,
            "d_vs_uni": sel_auc - uni_auc,
            "d_vs_all": sel_auc - all_auc.get(d, np.nan),
        })
    delta = pd.DataFrame(rows)
    delta.to_csv(ML_DIR / "delta_auc.csv", index=False)

    # Per-direction "best-alpha per selector" — by sel_auc
    best = (
        delta.loc[delta.groupby(["direction", "selector"])["sel_auc"].idxmax()]
        .sort_values(["direction", "sel_auc"], ascending=[True, False])
        .reset_index(drop=True)
    )
    best.to_csv(ML_DIR / "best_cells.csv", index=False)

    # --- Pretty print summary ---
    print("All-edges baseline per direction:")
    for d, v in all_auc.items():
        print(f"  {d:20s} test_AUC = {v:.3f}")

    for d in sorted(delta["direction"].unique()):
        sub = best[best["direction"] == d].copy()
        print(f"\n=== {d} ===")
        print(sub[["selector", "alpha", "k", "sel_auc",
                    "d_vs_all", "d_vs_rand", "d_vs_uni"]]
              .to_string(index=False, float_format="%.3f"))

    # --- Figure ---
    directions = sorted(delta["direction"].unique())
    selectors = ["tstat", "tfnbs", "nbs@2.0", "nbs@3.0", "cnbs", "ni_tfnbs", "bh_fdr"]
    alphas = sorted(delta["alpha"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for col, comparison, label in [
        (0, "d_vs_rand", "ΔAUC vs random-subset baseline"),
        (1, "d_vs_uni",  "ΔAUC vs univariate-top-k baseline"),
    ]:
        for row, d in enumerate(directions):
            ax = axes[row, col]
            sub = delta[delta["direction"] == d]
            for sel in selectors:
                s = sub[sub["selector"] == sel].sort_values("alpha")
                ax.plot(s["alpha"], s[comparison], marker="o", label=sel)
            ax.axhline(0, color="k", lw=0.5, ls="--")
            ax.set_xscale("log")
            ax.invert_xaxis()
            ax.set_xlabel("α"); ax.set_ylabel(label)
            ax.set_title(f"{d}")
            ax.legend(fontsize=7, ncol=2)
            ax.grid(alpha=0.3)
    fig.suptitle("§3.8 ML feature-selection — ΔAUC vs baselines (positive = selector wins)")
    fig.tight_layout()
    out = ML_DIR / "delta_auc_panel.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"\nFigure: {out}")
    print(f"Tables: {ML_DIR / 'delta_auc.csv'}, {ML_DIR / 'best_cells.csv'}")


if __name__ == "__main__":
    main()
