"""
ABIDE (E, H) sensitivity sweep — Multiverse Hierarchy.

Sweeps a 6×6 grid of (E, H) values for three methods:
  1. TFNBS (Unrestricted)
  2. NI-TFNBS (Soft Network Prior)
  3. FBC-TFNBS (Bayesian Control)

This demonstrates how discovery counts vary across the TF-NBS multiverse
on the ABIDE Age developmental contrast, and how the network-informed
priors constrain the result across different topological regimes.

Grid
----
E ∈ {0.20, 0.40, 0.50, 0.75, 1.00, 1.30}  (6 values)
H ∈ {1.00, 2.00, 3.00, 5.00, 7.00, 10.00} (6 values)

Output
------
results/age_development/sensitivity/eh/
  grid_summary.csv                    # per-method per-cell n_sig
  eh_sensitivity.png                  # 6-panel composite figure
"""
from __future__ import annotations

import argparse
import os
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from conninfpy import compute_p_val_glm
from conninfpy.harmonize import combat_harmonize

HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "results" / "abide_prepared.npz"
OUT_DIR = HERE / "results" / "age_development" / "sensitivity" / "eh"
PLOTS_DIR = HERE / "results" / "plots"

E_GRID = (0.20, 0.40, 0.50, 0.75, 1.00, 1.30)
H_GRID = (1.00, 2.00, 3.00, 5.00, 7.00, 10.00)

PUBLISHED_DEFAULTS = {
    (0.40, 3.00): "Hao 2024",
    (0.50, 2.00): "Smith & Nichols 2009",
    (0.75, 3.00): "Baggio 2018",
}


def _load():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"{DATA_FILE} not found — run prepare_data.py first."
        )
    d = np.load(DATA_FILE, allow_pickle=True)
    data = {k: d[k] for k in d.files}
    
    Y = data["connectivity_z"]
    sites = data["site"]
    # Preserving Sex, Mean FD (the nuisance confounds)
    confounds = np.column_stack([
        data["sex"].astype(float),
        data["mean_fd"].astype(float)
    ])
    print("Harmonizing ABIDE for sensitivity sweep (Strategy D)...")
    combat_out = combat_harmonize(Y, sites, preserve=confounds)
    data["connectivity_z_harm"] = combat_out.Y_adjusted
    return data


def _make_eh_panels(
    summary_df: pd.DataFrame,
    out_path: str,
):
    """6-panel (E, H) sensitivity figure comparing TFNBS, NI, and FBC."""
    import matplotlib.pyplot as plt

    nE, nH = len(E_GRID), len(H_GRID)

    def _pivot(method: str, tail: str):
        sub = summary_df[summary_df["method"] == method]
        col = f"n_sig_{tail}"
        return (
            sub.pivot_table(index="E", columns="H", values=col, aggfunc="first")
            .reindex(index=E_GRID, columns=H_GRID)
        )

    fig = plt.figure(figsize=(18, 10))
    panels = [
        (_pivot("tfnbs", "pos"), "YlGn", "TFNBS Detection (+)", "#"),
        (_pivot("ni_tfnbs", "pos"), "YlGn", "NI-TFNBS Detection (+)", "#"),
        (_pivot("fbc_tfnbs", "pos"), "YlGn", "FBC-TFNBS Detection (+)", "#"),

        (_pivot("tfnbs", "neg"), "YlGn", "TFNBS Detection (-)", "#"),
        (_pivot("ni_tfnbs", "neg"), "YlGn", "NI-TFNBS Detection (-)", "#"),
        (_pivot("fbc_tfnbs", "neg"), "YlGn", "FBC-TFNBS Detection (-)", "#"),
    ]
    layout = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

    for (gi, gj), (data, cmap, title, label) in zip(layout, panels):
        ax = plt.subplot2grid((2, 3), (gi, gj))
        im = ax.imshow(data.values, aspect="auto", cmap=cmap)
        ax.set_xticks(range(nH))
        ax.set_yticks(range(nE))
        ax.set_xticklabels([f"{h:.1f}" for h in H_GRID], fontsize=8)
        ax.set_yticklabels([f"{e:.2f}" for e in E_GRID], fontsize=8)
        ax.set_xlabel("H (height exponent)", fontsize=9)
        ax.set_ylabel("E (extent exponent)", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(label, fontsize=8)

        # Annotate counts
        for i in range(nE):
            for j in range(nH):
                v = data.values[i, j]
                if not np.isfinite(v):
                    continue
                txt_color = "white" if v > data.values.max() * 0.5 else "black"
                ax.text(j, i, f"{int(v)}", ha="center", va="center",
                        color=txt_color, fontsize=8)

        # Mark defaults
        for (e, h), _name in PUBLISHED_DEFAULTS.items():
            if e in E_GRID and h in H_GRID:
                ax.add_patch(plt.Rectangle(
                    (H_GRID.index(h) - 0.45, E_GRID.index(e) - 0.45),
                    0.90, 0.90,
                    fill=False, edgecolor="red", linewidth=1.5,
                ))

    fig.suptitle(
        "Multiverse Hierarchy Sensitivity Sweep — ABIDE Age (GLM)\n"
        "Comparison of discovery consistency across topological regimes and network-informed priors",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-permutations", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    data = _load()
    conn = data["connectivity_z_harm"]
    age = data["age"].astype(float)
    confounds = np.column_stack([
        data["sex"].astype(float),
        data["mean_fd"].astype(float)
    ])
    net_labels = data["net_labels"]
    print(f"Loaded ABIDE: n={conn.shape[0]}, N={conn.shape[1]}")

    cells = list(product(E_GRID, H_GRID))
    e_list = [e for e, _ in cells]
    h_list = [h for _, h in cells]
    K = len(cells)

    methods = ["tfnbs", "ni_tfnbs", "fbc_tfnbs"]
    all_rows = []

    for mname in methods:
        print(f"\nRunning {mname} sweep ({K} cells) ...")
        t0 = time.time()
        kwargs = {"net_labels": net_labels} if mname in ("ni_tfnbs", "fbc_tfnbs") else {}
        res = compute_p_val_glm(
            conn, interest=age, confounds=confounds,
            method=mname,
            n_permutations=args.n_permutations,
            acceleration="gpd",
            e=e_list, h=h_list, n=10,
            use_mp=True, rng=args.seed,
            **kwargs
        )
        print(f"  Done in {time.time()-t0:.1f}s")

        pos_edges = res["positive"][np.triu_indices(conn.shape[1], k=1)]
        neg_edges = res["negative"][np.triu_indices(conn.shape[1], k=1)]

        for k, (e, h) in enumerate(cells):
            all_rows.append(dict(
                method=mname, E=e, H=h,
                n_sig_pos=int((pos_edges[:, k] < 0.05).sum()),
                n_sig_neg=int((neg_edges[:, k] < 0.05).sum()),
            ))

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "grid_summary.csv", index=False)

    plot_path = PLOTS_DIR / "plot5_tfnbs_grid_sensitivity.png"
    _make_eh_panels(df, str(plot_path))
    print(f"\nFig saved: {plot_path}")


if __name__ == "__main__":
    main()
