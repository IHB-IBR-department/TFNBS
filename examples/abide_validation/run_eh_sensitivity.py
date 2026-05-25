"""
ABIDE TFNBS (E, H) sensitivity sweep — array-mode implementation.

Sweeps a 6×6 grid of (E, H) values inside the TFNBS family to
characterise how stable the operator is across the parameter space.

The entire grid is run in **a single permutation pass**: TFNBS's
threshold integration loop produces per-threshold cluster sizes once,
then exponentiates them per (E, H) cell. The per-cell cost is
microseconds on top of the shared loop, so a 36-cell grid takes ~the
same wall-clock as a single cell — the wall-time budget below is for
the ENTIRE grid, not per-cell.

Grid
----
E ∈ {0.20, 0.40, 0.50, 0.75, 1.00, 1.30}  (6 values)
H ∈ {1.00, 2.00, 3.00, 5.00, 7.00, 10.00} (6 values)

The grid contains all three canonical published defaults exactly:

  - Hao 2024              (0.40, 3.00)  FDR-calibrated regime
  - Smith & Nichols 2009  (0.50, 2.00)  original TFCE
  - Baggio 2018           (0.75, 3.00)  TFNBS paper

It also stretches into Vinokur's wider sensitivity box on both axes
so the decay (if any) outside the published envelope is visible.

Output
------
results/age_development/sensitivity/eh/
  pmaps/E{e}_H{h}.npz                 # per-cell {'positive', 'negative'}
  grid_summary.csv                    # per-cell n_sig + min_p
  jaccard_matrix.csv  / _neg.csv      # 36×36 pairwise Jaccard at α=0.05
  spearman_matrix.csv / _neg.csv      # 36×36 pairwise Spearman on -log10 p
  eh_sensitivity.png                  # 5-panel composite figure

Wall-time: ~20 s on 18 cores with GPD-200 (single array call replaces
the previous 36-call loop, ~36× speedup).
"""
from __future__ import annotations

import argparse
import os
import time
from itertools import product

import numpy as np
import pandas as pd
from scipy import stats as scistats

from conninfpy import compute_p_val_glm


from pathlib import Path
HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "results" / "abide_prepared.npz"
OUT_DIR = HERE / "results" / "age_development" / "sensitivity" / "eh"
PMAPS_DIR = OUT_DIR / "pmaps"

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
    
    # Perform ComBat Strategy D on the fly (Nuisance-only preservation)
    from conninfpy.harmonize import combat_harmonize
    Y = data["connectivity_z"]
    sites = data["site"]
    # Preserving Age, Sex, Mean FD (the nuisance confounds for the Age effect of interest)
    # Wait, for Age sweep, Age is the interest. Strategy D preserves nuisance.
    confounds = np.column_stack([
        data["sex"].astype(float),
        data["mean_fd"].astype(float)
    ])
    print("Harmonizing ABIDE for sensitivity sweep (Strategy D)...")
    combat_out = combat_harmonize(Y, sites, preserve=confounds)
    data["connectivity_z_harm"] = combat_out.Y_adjusted
    return data


def _label(e: float, h: float) -> str:
    return f"E{e:.2f}_H{h:.2f}"


def _pairwise(arrs_by_label, fn):
    keys = list(arrs_by_label.keys())
    n = len(keys)
    M = np.full((n, n), np.nan)
    for i, ki in enumerate(keys):
        M[i, i] = 1.0
        for j in range(i + 1, n):
            kj = keys[j]
            M[i, j] = fn(arrs_by_label[ki], arrs_by_label[kj])
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
    """6-panel (E, H) sensitivity figure.

    Layout (2x3):
      Col 1: Detection counts (Pos / Neg)
      Col 2: Mean Spearman to grid (Pos / Neg)
      Col 3: Spearman to Hao 2024 (Pos / Neg)
    """
    import matplotlib.pyplot as plt

    nE, nH = len(E_GRID), len(H_GRID)

    def _pivot(col: str):
        return (
            grid_summary
            .pivot_table(index="E", columns="H", values=col, aggfunc="first")
            .reindex(index=E_GRID, columns=H_GRID)
        )

    n_sig_pos = _pivot("n_sig_pos")
    n_sig_neg = _pivot("n_sig_neg")

    mean_spr_pos = spr_pos_df.mean(axis=1)
    mean_spr_neg = spr_neg_df.mean(axis=1)

    hao_label = _label(0.40, 3.00)
    spr_to_hao_pos = spr_pos_df[hao_label]
    spr_to_hao_neg = spr_neg_df[hao_label]

    def _series_to_grid(series: pd.Series) -> pd.DataFrame:
        out = np.full((nE, nH), np.nan)
        for label, val in series.items():
            e_part, h_part = label.split("_")
            e_val = float(e_part[1:])
            h_val = float(h_part[1:])
            i = E_GRID.index(e_val)
            j = H_GRID.index(h_val)
            out[i, j] = val
        return pd.DataFrame(out, index=E_GRID, columns=H_GRID)

    mean_spr_pos_grid = _series_to_grid(mean_spr_pos)
    mean_spr_neg_grid = _series_to_grid(mean_spr_neg)
    spr_to_hao_pos_grid = _series_to_grid(spr_to_hao_pos)
    spr_to_hao_neg_grid = _series_to_grid(spr_to_hao_neg)

    fig = plt.figure(figsize=(18, 10))
    panels = [
        (n_sig_pos, "viridis", None, None,
         "Detection count (+)", "#"),
        (mean_spr_pos_grid, "magma", 0.7, 1.0,
         "Mean Spearman to all other cells (+)", r"$\rho$"),
        (spr_to_hao_pos_grid, "magma", 0.7, 1.0,
         "Spearman to Hao 2024 (E=0.4, H=3.0) (+)", r"$\rho$"),
        
        (n_sig_neg, "viridis", None, None,
         "Detection count (-)", "#"),
        (mean_spr_neg_grid, "magma", 0.7, 1.0,
         "Mean Spearman to all other cells (-)", r"$\rho$"),
        (spr_to_hao_neg_grid, "magma", 0.7, 1.0,
         "Spearman to Hao 2024 (E=0.4, H=3.0) (-)", r"$\rho$"),
    ]
    layout = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

    for (gi, gj), (data, cmap, vmin, vmax, title, label) in zip(layout, panels):
        ax = plt.subplot2grid((2, 3), (gi, gj))
        im = ax.imshow(data.values, aspect="auto", cmap=cmap,
                       vmin=vmin, vmax=vmax)
        ax.set_xticks(range(nH))
        ax.set_yticks(range(nE))
        ax.set_xticklabels([f"{h:.1f}" for h in H_GRID], fontsize=8)
        ax.set_yticklabels([f"{e:.2f}" for e in E_GRID], fontsize=8)
        ax.set_xlabel("H (height exponent)", fontsize=9)
        ax.set_ylabel("E (extent exponent)", fontsize=9)
        ax.set_title(title, fontsize=9)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(label, fontsize=8)

        finite = data.values[np.isfinite(data.values)]
        lo, hi = (finite.min(), finite.max()) if finite.size else (0, 1)
        for i in range(nE):
            for j in range(nH):
                v = data.values[i, j]
                if not np.isfinite(v):
                    continue
                fmt = f"{int(v):d}" if "Detection" in title else f"{v:.3f}"
                norm = (v - lo) / max(1e-9, hi - lo)
                txt_color = "white" if norm < 0.5 else "black"
                ax.text(j, i, fmt, ha="center", va="center",
                        color=txt_color, fontsize=7)

        for (e, h), _name in PUBLISHED_DEFAULTS.items():
            if e in E_GRID and h in H_GRID:
                ax.add_patch(plt.Rectangle(
                    (H_GRID.index(h) - 0.45, E_GRID.index(e) - 0.45),
                    0.90, 0.90,
                    fill=False, edgecolor="red", linewidth=1.5,
                ))

    fig.suptitle(
        "TFNBS (E, H) sensitivity on ABIDE Age "
        "(ComBat-harmonised, GLM with sex+motion confounds)\n"
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

    # Flatten the 6×6 (E, H) grid into parallel arrays for the
    # single-call array API. Iteration order is row-major over E_GRID.
    cells = list(product(e_grid, h_grid))
    e_arr = np.array([e for e, _ in cells], dtype=float)
    h_arr = np.array([h for _, h in cells], dtype=float)
    labels = [_label(e, h) for e, h in cells]
    K = len(cells)
    print(f"Grid size: {len(e_grid)} E × {len(h_grid)} H = {K} cells "
          f"(single permutation pass)")

    t0 = time.time()
    result = compute_p_val_glm(
        conn, interest=age, confounds=confounds,
        method="tfnbs", n_permutations=n_perm,
        acceleration=acceleration,
        e=e_arr, h=h_arr, n=10,
        use_mp=True, rng=args.seed,
    )
    elapsed = time.time() - t0
    print(f"\nTotal permutation pass: {elapsed:.1f}s")

    pos = result["positive"]   # (N, N, K)
    neg = result["negative"]   # (N, N, K)
    N = pos.shape[0]
    iu = np.triu_indices(N, k=1)
    pos_edges = pos[iu]        # (n_edges, K)
    neg_edges = neg[iu]        # (n_edges, K)

    # ----- per-cell summary + pmap save -----
    rows = []
    for k, (label, e, h) in enumerate(zip(labels, e_arr, h_arr)):
        np.savez(
            os.path.join(PMAPS_DIR, f"{label}.npz"),
            positive=pos[:, :, k], negative=neg[:, :, k],
        )
        rows.append(dict(
            label=label, E=float(e), H=float(h),
            published_default=PUBLISHED_DEFAULTS.get((float(e), float(h)), ""),
            n_sig_pos=int((pos_edges[:, k] < 0.05).sum()),
            min_p_pos=float(pos_edges[:, k].min()),
            n_sig_neg=int((neg_edges[:, k] < 0.05).sum()),
            min_p_neg=float(neg_edges[:, k].min()),
        ))
    grid_summary = pd.DataFrame(rows)
    grid_summary.to_csv(os.path.join(OUT_DIR, "grid_summary.csv"), index=False)

    # ----- pairwise stats across cells -----
    masks_pos = {labels[k]: pos_edges[:, k] < 0.05 for k in range(K)}
    masks_neg = {labels[k]: neg_edges[:, k] < 0.05 for k in range(K)}
    nlp_pos = {labels[k]: -np.log10(np.maximum(pos_edges[:, k], 1e-300))
               for k in range(K)}
    nlp_neg = {labels[k]: -np.log10(np.maximum(neg_edges[:, k], 1e-300))
               for k in range(K)}

    print("\nComputing pairwise Jaccard / Spearman matrices...")
    jac_pos_df = _pairwise(masks_pos, _jac)
    jac_neg_df = _pairwise(masks_neg, _jac)
    spr_pos_df = _pairwise(nlp_pos, _spr)
    spr_neg_df = _pairwise(nlp_neg, _spr)

    jac_pos_df.to_csv(os.path.join(OUT_DIR, "jaccard_matrix.csv"))
    jac_neg_df.to_csv(os.path.join(OUT_DIR, "jaccard_matrix_neg.csv"))
    spr_pos_df.to_csv(os.path.join(OUT_DIR, "spearman_matrix.csv"))
    spr_neg_df.to_csv(os.path.join(OUT_DIR, "spearman_matrix_neg.csv"))

    # ----- headline -----
    def _offdiag_stats(df: pd.DataFrame):
        arr = df.values.astype(float)
        n = arr.shape[0]
        iu_ = np.triu_indices(n, k=1)
        vals = arr[iu_]
        vals = vals[np.isfinite(vals)]
        return float(np.median(vals)), float(np.min(vals)), float(np.max(vals))

    print("\n" + "=" * 72)
    print(f"HEADLINE — pairwise stats over {K} (E, H) cells")
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

    plot_path = os.path.join(HERE, "results", "plots", "plot5_tfnbs_grid_sensitivity.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    _make_eh_panels(grid_summary, spr_pos_df, spr_neg_df, plot_path)
    print(f"\nFig saved: {plot_path}")
    print(f"All outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
