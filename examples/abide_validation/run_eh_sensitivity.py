"""
ABIDE §3.11.x — TFNBS (E, H) sensitivity on real data.

Companion to ``run_threshold_sensitivity.py`` (which compares 7
published-defaults variants of NBS vs TFNBS). This script sweeps a
denser 6×6 grid of (E, H) values **inside the TFNBS family** to
characterise how stable the operator is across the parameter space —
not just at the three literature defaults.

Why this matters
----------------
Vinokur et al. (2023) report 75-fold variation in detected edge counts
across (E, H) on synthetic data inside Baggio's recommended box. We
ask: does that level of sensitivity persist on real ABIDE Age data
(n=764, ComBat-harmonised, GLM with sex+motion confounds)? Or — as
``run_threshold_sensitivity.py`` already suggests at the 3-default
level (Spearman 0.998) — does TFNBS behave like a parameter-free
operator over the published-defaults envelope?

Grid
----
E ∈ {0.20, 0.40, 0.50, 0.75, 1.00, 1.30}  (6 values)
H ∈ {1.00, 2.00, 3.00, 5.00, 7.00, 10.00} (6 values)

The grid contains all three canonical published defaults exactly:

  - Hao 2024              (0.40, 3.00)  FDR-calibrated regime
  - Smith & Nichols 2009  (0.50, 2.00)  original TFCE
  - Baggio 2018           (0.75, 3.00)  TFNBS paper

It also stretches into Vinokur's wider sensitivity box on both axes so
we can quantify decay (if any) outside the published envelope.

Output
------
results/combat/age/eh_sensitivity/
  pmaps/E{e}_H{h}.npz                 # per-cell {'positive', 'negative'}
  grid_summary.csv                    # per-cell n_sig + min_p + wall_time
  jaccard_matrix.csv  / _neg.csv      # 36×36 pairwise Jaccard at α=0.05
  spearman_matrix.csv / _neg.csv      # 36×36 pairwise Spearman on -log10 p
  eh_sensitivity.png                  # paper figure (5-panel composite)

Wall-time: ~10 min on 18 cores with numba JIT + GPD-200.
"""
from __future__ import annotations

import argparse
import os
import time
from itertools import combinations, product

import numpy as np
import pandas as pd
from scipy import stats as scistats

from conninfpy import compute_p_val_glm


HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(HERE, "abide_harmonized.npz")
OUT_DIR = os.path.join(HERE, "results", "combat", "age", "eh_sensitivity")
PMAPS_DIR = os.path.join(OUT_DIR, "pmaps")

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
            f"{DATA_FILE} not found — run `python harmonize.py` first."
        )
    d = np.load(DATA_FILE, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _label(e: float, h: float) -> str:
    return f"E{e:.2f}_H{h:.2f}"


def _summarise_pmap(p_dict, alpha=0.05):
    out = {}
    for tail, mat in p_dict.items():
        N = mat.shape[0]
        iu = np.triu_indices(N, k=1)
        vec = mat[iu]
        out[tail] = dict(n_sig=int((vec < alpha).sum()), min_p=float(vec.min()))
    return out


def _pairwise(masks_or_vecs, fn):
    keys = list(masks_or_vecs.keys())
    n = len(keys)
    M = np.full((n, n), np.nan)
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            if i == j:
                M[i, j] = 1.0
            elif j > i:
                M[i, j] = fn(masks_or_vecs[ki], masks_or_vecs[kj])
                M[j, i] = M[i, j]
    return pd.DataFrame(M, index=keys, columns=keys)


def _jac(a, b):
    union = (a | b).sum()
    return (a & b).sum() / union if union > 0 else 0.0


def _spr(a, b):
    r, _ = scistats.spearmanr(a, b)
    return float(r) if np.isfinite(r) else 0.0


def _make_eh_panels(
    grid_summary: pd.DataFrame,
    spr_pos_df: pd.DataFrame,
    spr_neg_df: pd.DataFrame,
    out_path: str,
):
    """Produce the 5-panel (E, H) sensitivity figure.

    Layout (1×3, then 1×2 below):
      (a) n_sig (positive tail)        — heatmap on (E, H) grid
      (b) n_sig (negative tail)        — heatmap on (E, H) grid
      (c) Mean Spearman vs grid (pos)  — how *central* each (E, H) is
      (d) Spearman to Hao 2024 (pos)   — distance from the FDR-calibrated regime
      (e) Spearman to Hao 2024 (neg)
    """
    import matplotlib.pyplot as plt

    nE, nH = len(E_GRID), len(H_GRID)

    # Pivot the per-cell summary into (E, H) heatmaps.
    def _pivot(col: str):
        return (
            grid_summary
            .pivot_table(index="E", columns="H", values=col, aggfunc="first")
            .reindex(index=E_GRID, columns=H_GRID)
        )

    n_sig_pos = _pivot("n_sig_pos")
    n_sig_neg = _pivot("n_sig_neg")

    # Mean Spearman per cell vs *all other* cells in the grid.
    mean_spr_pos = spr_pos_df.mean(axis=1)
    mean_spr_neg = spr_neg_df.mean(axis=1)

    def _spr_to_anchor(spr_df: pd.DataFrame, anchor_label: str) -> pd.Series:
        return spr_df[anchor_label]

    hao_label = _label(0.40, 3.00)
    spr_to_hao_pos = _spr_to_anchor(spr_pos_df, hao_label)
    spr_to_hao_neg = _spr_to_anchor(spr_neg_df, hao_label)

    def _series_to_grid(series: pd.Series) -> pd.DataFrame:
        # series index is "E{e}_H{h}" → reshape to (E, H)
        out = np.full((nE, nH), np.nan)
        for label, val in series.items():
            e, h = label.split("_")
            e_val = float(e[1:])
            h_val = float(h[1:])
            i = E_GRID.index(e_val)
            j = H_GRID.index(h_val)
            out[i, j] = val
        return pd.DataFrame(out, index=E_GRID, columns=H_GRID)

    mean_spr_pos_grid = _series_to_grid(mean_spr_pos)
    mean_spr_neg_grid = _series_to_grid(mean_spr_neg)
    spr_to_hao_pos_grid = _series_to_grid(spr_to_hao_pos)
    spr_to_hao_neg_grid = _series_to_grid(spr_to_hao_neg)

    fig = plt.figure(figsize=(15, 9))
    panels = [
        (n_sig_pos, "viridis", None, None,
         "Detection count\npositive tail (g2>g1, age↑→FC↑)", "#"),
        (n_sig_neg, "viridis", None, None,
         "Detection count\nnegative tail (g1>g2, age↑→FC↓)", "#"),
        (mean_spr_pos_grid, "magma", 0.7, 1.0,
         "Mean Spearman to all other (E, H) cells\npositive tail",
         r"$\rho$"),
        (spr_to_hao_pos_grid, "magma", 0.7, 1.0,
         "Spearman to Hao 2024 (E=0.4, H=3.0)\npositive tail",
         r"$\rho$"),
        (spr_to_hao_neg_grid, "magma", 0.7, 1.0,
         "Spearman to Hao 2024 (E=0.4, H=3.0)\nnegative tail",
         r"$\rho$"),
    ]

    grid = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    for (gi, gj), (data, cmap, vmin, vmax, title, label) in zip(grid, panels):
        ax = plt.subplot2grid((2, 3), (gi, gj))
        im = ax.imshow(
            data.values, aspect="auto", cmap=cmap,
            vmin=vmin, vmax=vmax,
        )
        ax.set_xticks(range(nH))
        ax.set_yticks(range(nE))
        ax.set_xticklabels([f"{h:.1f}" for h in H_GRID], fontsize=8)
        ax.set_yticklabels([f"{e:.2f}" for e in E_GRID], fontsize=8)
        ax.set_xlabel("H (height exponent)", fontsize=9)
        ax.set_ylabel("E (extent exponent)", fontsize=9)
        ax.set_title(title, fontsize=9)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(label, fontsize=8)

        for i in range(nE):
            for j in range(nH):
                v = data.values[i, j]
                if not np.isfinite(v):
                    continue
                fmt = f"{int(v):d}" if "Detection" in title else f"{v:.3f}"
                # Use intensity-aware text colour
                lo = data.values[np.isfinite(data.values)].min()
                hi = data.values[np.isfinite(data.values)].max()
                norm = (v - lo) / max(1e-9, hi - lo)
                txt_color = "white" if norm < 0.5 else "black"
                ax.text(j, i, fmt, ha="center", va="center",
                        color=txt_color, fontsize=7)

        # Mark the published-default cells
        for (e, h), name in PUBLISHED_DEFAULTS.items():
            if e in E_GRID and h in H_GRID:
                ax.add_patch(
                    plt.Rectangle(
                        (H_GRID.index(h) - 0.45, E_GRID.index(e) - 0.45),
                        0.90, 0.90,
                        fill=False, edgecolor="red", linewidth=1.5,
                    )
                )

    fig.suptitle(
        "TFNBS (E, H) sensitivity on ABIDE Age "
        "(n=764, ComBat-harmonised, GLM with sex+motion confounds)\n"
        "Red boxes mark published defaults: "
        "Hao 2024 (0.40, 3.00), Smith & Nichols 2009 (0.50, 2.00), "
        "Baggio 2018 (0.75, 3.00)",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    import matplotlib.pyplot as plt2
    plt2.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="2x2 grid + 50 perms (smoke test only)")
    ap.add_argument("--n-permutations", type=int, default=200)
    ap.add_argument("--no-acceleration", action="store_true",
                    help="Disable GPD; use empirical p-values")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.quick:
        n_perm = 50
        acceleration = None
        e_grid = (0.40, 0.75)
        h_grid = (2.00, 3.00)
    else:
        n_perm = args.n_permutations
        acceleration = None if args.no_acceleration else "gpd"
        e_grid = E_GRID
        h_grid = H_GRID

    os.makedirs(PMAPS_DIR, exist_ok=True)
    data = _load()
    conn = data["connectivity_z_harm"]
    age = data["age"].astype(float)
    confounds = np.column_stack(
        [data["sex"].astype(float), data["mean_fd"].astype(float)]
    )
    print(f"Loaded ABIDE: n={conn.shape[0]}, N={conn.shape[1]}")
    print(f"Age range: {age.min():.1f}–{age.max():.1f}")
    print(f"n_permutations={n_perm}, acceleration={acceleration}")
    print(f"Grid size: {len(e_grid)} E × {len(h_grid)} H = "
          f"{len(e_grid) * len(h_grid)} cells")

    pmaps = {}
    rows = []
    t_start_total = time.time()
    for k, (e, h) in enumerate(product(e_grid, h_grid), start=1):
        label = _label(e, h)
        is_default = (e, h) in PUBLISHED_DEFAULTS
        marker = "  ★" if is_default else ""
        print(f"\n[{k}/{len(e_grid)*len(h_grid)}] {label}{marker}")
        t0 = time.time()
        p = compute_p_val_glm(
            conn, interest=age, confounds=confounds,
            method="tfnbs", n_permutations=n_perm,
            acceleration=acceleration,
            e=e, h=h, n=10,
            use_mp=True, rng=args.seed,
        )
        elapsed = time.time() - t0
        pmaps[label] = p
        np.savez(os.path.join(PMAPS_DIR, f"{label}.npz"),
                 positive=p["positive"], negative=p["negative"])
        info = _summarise_pmap(p)
        rows.append(dict(
            label=label, E=e, H=h,
            published_default=PUBLISHED_DEFAULTS.get((e, h), ""),
            n_sig_pos=info["positive"]["n_sig"],
            min_p_pos=info["positive"]["min_p"],
            n_sig_neg=info["negative"]["n_sig"],
            min_p_neg=info["negative"]["min_p"],
            elapsed_s=elapsed,
        ))
        print(f"   pos: n_sig={info['positive']['n_sig']:5d}  "
              f"min_p={info['positive']['min_p']:.4f}")
        print(f"   neg: n_sig={info['negative']['n_sig']:5d}  "
              f"min_p={info['negative']['min_p']:.4f}")
        print(f"   wall: {elapsed:.1f}s")

    grid_summary = pd.DataFrame(rows)
    grid_summary.to_csv(os.path.join(OUT_DIR, "grid_summary.csv"), index=False)

    # ----- pairwise stats -----
    N = conn.shape[1]
    iu = np.triu_indices(N, k=1)
    masks_pos, masks_neg, nlp_pos, nlp_neg = {}, {}, {}, {}
    for label, p in pmaps.items():
        masks_pos[label] = p["positive"][iu] < 0.05
        masks_neg[label] = p["negative"][iu] < 0.05
        nlp_pos[label] = -np.log10(np.maximum(p["positive"][iu], 1e-300))
        nlp_neg[label] = -np.log10(np.maximum(p["negative"][iu], 1e-300))

    print("\nComputing pairwise Jaccard / Spearman matrices...")
    jac_pos_df = _pairwise(masks_pos, _jac)
    jac_neg_df = _pairwise(masks_neg, _jac)
    spr_pos_df = _pairwise(nlp_pos, _spr)
    spr_neg_df = _pairwise(nlp_neg, _spr)

    jac_pos_df.to_csv(os.path.join(OUT_DIR, "jaccard_matrix.csv"))
    jac_neg_df.to_csv(os.path.join(OUT_DIR, "jaccard_matrix_neg.csv"))
    spr_pos_df.to_csv(os.path.join(OUT_DIR, "spearman_matrix.csv"))
    spr_neg_df.to_csv(os.path.join(OUT_DIR, "spearman_matrix_neg.csv"))

    # ----- headline numbers -----
    def _offdiag_stats(df: pd.DataFrame):
        arr = df.values.astype(float)
        n = arr.shape[0]
        iu_ = np.triu_indices(n, k=1)
        vals = arr[iu_]
        vals = vals[np.isfinite(vals)]
        return float(np.median(vals)), float(np.min(vals)), float(np.max(vals))

    print("\n" + "=" * 72)
    print(f"HEADLINE — pairwise stats over {len(grid_summary)} (E, H) cells")
    print("=" * 72)
    for tail, jdf, sdf in [
        ("positive", jac_pos_df, spr_pos_df),
        ("negative", jac_neg_df, spr_neg_df),
    ]:
        jm, jlo, jhi = _offdiag_stats(jdf)
        sm, slo, shi = _offdiag_stats(sdf)
        print(f"  {tail} tail:")
        print(f"    pairwise Jaccard  median={jm:.3f}  range=({jlo:.3f}, {jhi:.3f})")
        print(f"    pairwise Spearman median={sm:.3f}  range=({slo:.3f}, {shi:.3f})")

    # Detection-count spread
    print("\nDetection-count spread (Vinokur-style):")
    for tail in ("pos", "neg"):
        col = f"n_sig_{tail}"
        nz = grid_summary[grid_summary[col] > 0][col]
        if len(nz) == 0:
            print(f"  {tail}: no cell detected anything")
            continue
        ratio = nz.max() / max(1, nz.min())
        print(f"  {tail}: n_sig range=[{int(nz.min())}, {int(nz.max())}],"
              f" max/min ratio = {ratio:.1f}×")

    # ----- plot -----
    plot_path = os.path.join(OUT_DIR, "eh_sensitivity.png")
    _make_eh_panels(grid_summary, spr_pos_df, spr_neg_df, plot_path)
    print(f"\nFig saved: {plot_path}")

    total = time.time() - t_start_total
    print(f"\nTotal wall-time: {total / 60:.1f} min")
    print(f"All outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
