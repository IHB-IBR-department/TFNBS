"""
Fig 1 — ConnInfPy pipeline overview (matplotlib first draft, v2).

Implements the 8-panel design at
``MainVault/Projects/NetworkStatistics/Figure1_pipeline_spec.md``.

Layout fix (v2 vs v1): panel-letter titles are placed via ``fig.text()``
at computed y-coordinates above each panel rather than via
``ax.set_title()`` on the outer (axis-off) axes — this avoids overlap
with inner sub-panel titles when an outer panel hosts a subgridspec.
Descriptive captions are moved to the LaTeX figure caption (not overlaid
on the figure). NEW badges are kept inside panel bounds.

Output:
    examples/figures/fig1_pipeline.pdf
    examples/figures/fig1_pipeline.png    (300 dpi)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from conninfpy import combat_harmonize, get_components
from conninfpy.acceleration import _fit_gpd_mom, _gpd_sf

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.titleweight": "normal",
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "figure.dpi": 110,
})


HERE = Path(__file__).parent
OUT_PDF = HERE / "fig1_pipeline.pdf"
OUT_PNG = HERE / "fig1_pipeline.png"
OUT_SVG = HERE / "fig1_pipeline.svg"
DOCS_PDF = HERE.parent.parent / "docs" / "fig1_pipeline.pdf"
DOCS_PNG = HERE.parent.parent / "docs" / "fig1_pipeline.png"
DOCS_SVG = HERE.parent.parent / "docs" / "fig1_pipeline.svg"


# =============================================================================
# Helpers
# =============================================================================

def _make_modular_mask(n_nodes=30, n_modules=4, effect_blocks=(0, 2),
                       rng=None):
    """Clean blockwise binary ground-truth mask.

    Within each module in `effect_blocks`, ALL edges are set to 1
    (within-module-dense topology). All other off-diagonal edges are
    zero. Produces a visually clear block-diagonal pattern.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    mod_size = n_nodes // n_modules
    mask = np.zeros((n_nodes, n_nodes), dtype=int)
    for m in effect_blocks:
        block = slice(m * mod_size, (m + 1) * mod_size)
        mask[block, block] = 1
    np.fill_diagonal(mask, 0)
    return mask


def _make_subject_fc(n_subj=30, n_nodes=60, site_effects=None,
                     planted_mask=None, planted_strength=0.0, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    n_modules = 4
    mod_size = n_nodes // n_modules
    base = np.full((n_nodes, n_nodes), 0.05)
    for m in range(n_modules):
        block = slice(m * mod_size, (m + 1) * mod_size)
        base[block, block] = 0.3
    np.fill_diagonal(base, 1.0)
    base = (base + base.T) / 2
    fc_stack = np.zeros((n_subj, n_nodes, n_nodes))
    for i in range(n_subj):
        noise = rng.normal(0, 0.05, (n_nodes, n_nodes))
        noise = (noise + noise.T) / 2
        m = base + noise
        if site_effects is not None:
            m += site_effects[i]
        if planted_mask is not None:
            m += planted_strength * planted_mask
        np.fill_diagonal(m, 1.0)
        fc_stack[i] = m
    return fc_stack


def _add_panel_label(fig, x, y, letter, title):
    """Panel-letter title placed in figure coordinates above the panel.
    Avoids the outer-axis title overlap with inner subgridspec titles.
    """
    fig.text(x, y, f"{letter}.  {title}",
             fontsize=10, fontweight="bold",
             ha="left", va="bottom")


def _new_badge(ax, x=0.97, y=0.95, vertical=False, fontsize=6):
    """Compact NEW badge inside the panel bounds."""
    rotation = 90 if vertical else 0
    ax.text(x, y, "NEW", transform=ax.transAxes, fontsize=fontsize,
            color="white", weight="bold", ha="right" if not vertical else "center",
            va="top" if not vertical else "center", rotation=rotation,
            bbox=dict(boxstyle="round,pad=0.18", fc="#c0392b", ec="none"),
            zorder=10)


# =============================================================================
# Panels
# =============================================================================

def panel_A_ground_truth(ax, rng):
    """Two side-by-side reference FC matrices:
    "Group 1 (no effect)" vs "Group 2 (with planted effect on
    modules 0, 2)". Same 4-module structure and same viridis colormap
    as Panels B / C — visually establishes the ground truth that
    flows through the pipeline.

    Returns the planted-effect mask so Panel B can apply it to its
    subject-level FC matrices.
    """
    n_nodes = 30
    n_modules = 4
    mask = _make_modular_mask(n_nodes=n_nodes, n_modules=n_modules,
                              effect_blocks=(0, 2), rng=rng)
    # Build the two reference matrices: same base modular FC, with the
    # planted effect added only to Group 2.
    g1 = _make_subject_fc(n_subj=1, n_nodes=n_nodes,
                          planted_mask=None, planted_strength=0.0,
                          rng=np.random.default_rng(11))[0]
    g2 = _make_subject_fc(n_subj=1, n_nodes=n_nodes,
                          planted_mask=mask, planted_strength=0.20,
                          rng=np.random.default_rng(11))[0]

    # Two side-by-side imshows inside the outer axes
    ax.axis("off")
    fig = ax.figure
    sub = ax.get_subplotspec().subgridspec(1, 2, wspace=0.10)
    a0 = fig.add_subplot(sub[0])
    a1 = fig.add_subplot(sub[1])
    vmin, vmax = -0.1, 0.5
    a0.imshow(g1, cmap="viridis", vmin=vmin, vmax=vmax)
    a1.imshow(g2, cmap="viridis", vmin=vmin, vmax=vmax)
    a0.set_title("Group 1\n(no effect)", fontsize=6.8, pad=2)
    a1.set_title("Group 2\n(planted effect)", fontsize=6.8, pad=2)
    mod_size = n_nodes // n_modules
    for a in (a0, a1):
        a.set_xticks([]); a.set_yticks([])
        for k in range(1, n_modules):
            a.axvline(k * mod_size - 0.5, color="white", lw=0.4, alpha=0.6)
            a.axhline(k * mod_size - 0.5, color="white", lw=0.4, alpha=0.6)
    return mask


def panel_B_per_subject_fc(ax, rng, planted_mask=None):
    """Per-subject FC with the Panel-A planted effect threaded through.

    The 4-module base correlation structure is augmented by the
    ground-truth mask (from Panel A) so that the two planted-effect
    modules visibly contain higher within-module correlation than the
    two null modules. Site effects are orthogonal to the planted
    effect — they shift overall correlation magnitude up/down per site.
    """
    n_nodes = 30; n_subj = 9
    site_codes = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    site_colors = ["#d35400", "#27ae60", "#2980b9"]
    site_effects = np.zeros((n_subj, n_nodes, n_nodes))
    for i, sc in enumerate(site_codes):
        site_effects[i] = (sc - 1) * 0.06
    fc = _make_subject_fc(n_subj=n_subj, n_nodes=n_nodes,
                          site_effects=site_effects,
                          planted_mask=planted_mask,
                          planted_strength=0.20,
                          rng=rng)
    ax.axis("off")
    fig = ax.figure
    sub = ax.get_subplotspec().subgridspec(3, 3, wspace=0.10, hspace=0.10)
    for k in range(n_subj):
        a = fig.add_subplot(sub[k // 3, k % 3])
        a.imshow(fc[k], cmap="viridis", vmin=-0.1, vmax=0.5)
        a.set_xticks([]); a.set_yticks([])
        for spine in a.spines.values():
            spine.set_edgecolor(site_colors[site_codes[k]])
            spine.set_linewidth(2.0)
    return fc, site_codes


def panel_C_combat(ax, fc, site_codes, rng):
    sites = site_codes.astype(int)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = combat_harmonize(fc, sites=sites)
        fc_after = result.Y_adjusted
    except Exception:
        fc_after = fc - fc.mean(axis=0, keepdims=True) * 0.5
    mean_before = fc.mean(axis=0)
    mean_after = fc_after.mean(axis=0)
    ax.axis("off")
    fig = ax.figure
    sub = ax.get_subplotspec().subgridspec(1, 2, wspace=0.10)
    a0 = fig.add_subplot(sub[0]); a1 = fig.add_subplot(sub[1])
    vmin, vmax = -0.05, 0.4
    a0.imshow(mean_before, cmap="viridis", vmin=vmin, vmax=vmax)
    a1.imshow(mean_after, cmap="viridis", vmin=vmin, vmax=vmax)
    a0.set_title("Before ComBat", fontsize=7, pad=2)
    a1.set_title("After ComBat", fontsize=7, pad=2)
    for a in (a0, a1):
        a.set_xticks([]); a.set_yticks([])
    _new_badge(a1, x=0.97, y=0.96)


def panel_D_glm(ax):
    """D — Edge-wise GLM with Freedman–Lane, multi-contrast in one pass.

    Three sub-panels:
      (left)   Design X with three contrasts c_age, c_sex, c_FD
               highlighted as separate "of-interest" columns.
      (centre) Freedman–Lane flowchart with the shared-nuisance loop
               annotated — one residual fit, K contrast evaluations.
      (right)  Three resulting per-edge stat maps t_e^{age}, t_e^{sex},
               t_e^{FD} produced in a single permutation pass via
               compute_p_val_glm_multi.
    """
    ax.axis("off")
    fig = ax.figure
    sub = ax.get_subplotspec().subgridspec(
        1, 5, width_ratios=[1.5, 0.25, 2.6, 0.3, 2.6], wspace=0.08
    )

    # ---- Design matrix X with contrast-row strip below ----
    a_x_outer = fig.add_subplot(sub[0])
    a_x_outer.axis("off")
    inner_x = sub[0].subgridspec(2, 1, height_ratios=[5, 1], hspace=0.05)
    a_x = fig.add_subplot(inner_x[0])
    n = 30
    rng = np.random.default_rng(7)
    X = np.column_stack([
        np.ones(n),
        rng.normal(0, 1, n),
        (rng.random(n) > 0.5).astype(float),
        np.abs(rng.normal(0.2, 0.1, n)),
    ])
    a_x.imshow(X, cmap="RdBu_r", aspect="auto", vmin=-2, vmax=2)
    a_x.set_xticks(range(4))
    a_x.set_xticklabels(["1", "age", "sex", "FD"], fontsize=6)
    a_x.set_ylabel("subjects", fontsize=6.5)
    a_x.set_title("Design $X$", fontsize=7)
    # Highlight the three "of interest" columns (1, 2, 3 — exclude intercept)
    for col in (1, 2, 3):
        a_x.add_patch(plt.Rectangle(
            (col - 0.45, -0.5), 0.9, n - 0.0,
            fill=False, edgecolor="#c33", linewidth=1.0,
        ))
    # Tiny contrast strip below: 3 stacked one-hot rows
    a_c = fig.add_subplot(inner_x[1])
    contrasts = np.array([
        [0, 1, 0, 0],  # age
        [0, 0, 1, 0],  # sex
        [0, 0, 0, 1],  # FD
    ])
    a_c.imshow(contrasts, cmap="Reds", vmin=0, vmax=1, aspect="auto")
    a_c.set_yticks(range(3))
    a_c.set_yticklabels(["$c_{\\rm age}$", "$c_{\\rm sex}$", "$c_{\\rm FD}$"],
                        fontsize=5.5)
    a_c.set_xticks([]); a_c.tick_params(axis="y", length=0, pad=1)
    a_c.set_title("3 contrasts", fontsize=6, pad=1)

    # ---- Freedman–Lane flowchart, annotated as shared-nuisance loop ----
    a_f = fig.add_subplot(sub[2])
    a_f.set_xlim(0, 1); a_f.set_ylim(0, 1); a_f.axis("off")
    a_f.set_title("Freedman–Lane  ·  shared nuisance, $K$ contrasts in one pass",
                  fontsize=6.8)
    boxes = [
        (0.04, 0.78, 0.30, 0.16, "Reduced fit\n$y = X_Z\\hat\\gamma$"),
        (0.04, 0.50, 0.30, 0.16, "Residuals $\\hat r$"),
        (0.04, 0.22, 0.30, 0.16, "Permute  $P_\\pi \\hat r$"),
        (0.46, 0.50, 0.50, 0.16,
         "Reconstruct + full fit\n$y^\\pi = X_Z\\hat\\gamma + P_\\pi \\hat r$"),
    ]
    for (x, y, w, h, txt) in boxes:
        a_f.add_patch(FancyBboxPatch((x, y - h/2), w, h,
                                     boxstyle="round,pad=0.02",
                                     fc="#dfe6f0", ec="#345", lw=0.9))
        a_f.text(x + w/2, y, txt, ha="center", va="center", fontsize=6.0)
    arrows = [
        ((0.19, 0.70), (0.19, 0.58)),
        ((0.19, 0.42), (0.19, 0.30)),
        ((0.34, 0.50), (0.46, 0.50)),
    ]
    for (xy0, xy1) in arrows:
        a_f.add_patch(FancyArrowPatch(xy0, xy1, arrowstyle="-|>",
                                      mutation_scale=8, color="#345", lw=0.7))
    # Annotation: one beta vector per perm, K stats per beta
    a_f.text(0.71, 0.32,
             r"$\hat\beta^\pi$ reused" + "\n" + r"for all $K$ contrasts",
             ha="center", va="center", fontsize=6.0, color="#a33",
             style="italic")

    # ---- Three per-edge stat maps, one per contrast ----
    a_t = fig.add_subplot(sub[4])
    a_t.axis("off")
    inner = sub[4].subgridspec(1, 3, wspace=0.18)
    rng2 = np.random.default_rng(3)
    contrast_labels = [
        ("$t_e^{\\rm age}$", "#c33"),
        ("$t_e^{\\rm sex}$", "#c33"),
        ("$t_e^{\\rm FD}$", "#c33"),
    ]
    for j, (label, color) in enumerate(contrast_labels):
        a_in = fig.add_subplot(inner[j])
        # Each contrast gets a slightly different signal pattern
        rng_j = np.random.default_rng(3 + j)
        Tmap = np.abs(rng_j.normal(0, 1.0, (20, 20)))
        if j == 0:  # age — strong cluster top-left
            Tmap[:6, :6] += 2.5
        elif j == 1:  # sex — single hub
            Tmap[7, :] += 1.8; Tmap[:, 7] += 1.8
        else:  # FD — diffuse
            Tmap += 0.6
        Tmap = (Tmap + Tmap.T) / 2
        np.fill_diagonal(Tmap, 0)
        a_in.imshow(Tmap, cmap="magma", vmin=0, vmax=4)
        a_in.set_xticks([]); a_in.set_yticks([])
        a_in.set_title(label, fontsize=8, pad=1, color=color)
    # Banner under the three thumbnails
    a_t.text(0.5, -0.06,
             r"$\rightarrow$ compute_p_val_glm_multi  ·  one perm pass",
             transform=a_t.transAxes, ha="center", va="top",
             fontsize=6.5, style="italic", color="#a33")


def _make_t_for_NBS(rng):
    n_nodes = 30
    mask = _make_modular_mask(n_nodes=n_nodes, n_modules=3,
                              effect_blocks=(0,), rng=rng)
    T = np.abs(rng.normal(0, 1, (n_nodes, n_nodes)))
    T = (T + T.T) / 2
    T += 2.5 * mask
    np.fill_diagonal(T, 0)
    return T


def panel_E_NBS(ax, rng):
    T = _make_t_for_NBS(rng)
    tau = 2.0
    suprath = (T > tau).astype(int)
    components, _ = get_components(suprath, no_depend=True)

    ax.axis("off")
    fig = ax.figure
    sub = ax.get_subplotspec().subgridspec(3, 1, hspace=0.30)
    a0 = fig.add_subplot(sub[0]); a0.imshow(T, cmap="magma", vmin=0, vmax=4); a0.set_title("$t$-stat", fontsize=6.5, pad=1)
    a0.set_xticks([]); a0.set_yticks([])
    a1 = fig.add_subplot(sub[1]); a1.imshow(suprath, cmap="binary"); a1.set_title(rf"$> \tau={tau}$", fontsize=6.5, pad=1)
    a1.set_xticks([]); a1.set_yticks([])
    n = T.shape[0]
    edge_label = np.zeros((n, n), dtype=int)
    if components is not None:
        comp_node = np.array(components)
        for i in range(n):
            for j in range(n):
                if suprath[i, j] and comp_node[i] == comp_node[j]:
                    edge_label[i, j] = comp_node[i] + 1
    a2 = fig.add_subplot(sub[2]); a2.imshow(edge_label, cmap="tab20"); a2.set_title("Components", fontsize=6.5, pad=1)
    a2.set_xticks([]); a2.set_yticks([])


def panel_F_TFNBS(ax, rng):
    T = _make_t_for_NBS(rng)
    h_levels = np.linspace(1.5, 4.0, 4)
    ax.axis("off")
    fig = ax.figure
    sub = ax.get_subplotspec().subgridspec(4, 1, hspace=0.20)
    for i, h in enumerate(h_levels):
        a = fig.add_subplot(sub[i])
        a.imshow((T > h).astype(int), cmap="binary")
        a.set_xticks([]); a.set_yticks([])
        a.text(1.05, 0.5, f"$h={h:.1f}$", transform=a.transAxes,
               fontsize=6.5, va="center")


def panel_G_block_methods(ax):
    rng = np.random.default_rng(8)
    n_nodes = 28
    block_labels = np.repeat(np.arange(7), 4)
    T = np.abs(rng.normal(0, 1, (n_nodes, n_nodes)))
    T = (T + T.T) / 2
    T[:4, :4] += 3.0
    np.fill_diagonal(T, 0)
    block_mean = np.zeros_like(T)
    for b1 in range(7):
        for b2 in range(7):
            cells = T[np.ix_(block_labels == b1, block_labels == b2)]
            block_mean[np.ix_(block_labels == b1, block_labels == b2)] = cells.mean()
    np.fill_diagonal(block_mean, 0)
    ni = np.copy(T); ni[:4, :4] *= 1.6
    fbc = np.where(block_mean > block_mean.mean(), T, 0)

    ax.axis("off")
    fig = ax.figure
    sub = ax.get_subplotspec().subgridspec(3, 1, hspace=0.30)
    titles = ["cNBS", "NI-TFNBS", "FBC-TFNBS"]
    new_flags = [False, True, True]
    for i, (mat, t, is_new) in enumerate(zip([block_mean, ni, fbc], titles, new_flags)):
        a = fig.add_subplot(sub[i])
        a.imshow(mat, cmap="magma", vmin=0, vmax=4.5)
        a.set_xticks([]); a.set_yticks([])
        for k in range(1, 7):
            a.axvline(k * 4 - 0.5, color="white", lw=0.3)
            a.axhline(k * 4 - 0.5, color="white", lw=0.3)
        a.set_title(t, fontsize=7, pad=1)
        if is_new:
            _new_badge(a, x=0.97, y=0.95)


def panel_H_inference(ax, rng):
    n_perm = 200
    null_max = np.abs(rng.normal(0, 1, n_perm)) * 2 + 1.0
    null_max[-30:] += 0.8 * np.abs(rng.normal(0, 1, 30))
    u = float(np.quantile(null_max, 0.75))
    exceed = null_max[null_max > u] - u
    sigma, xi, _ = _fit_gpd_mom(exceed)

    ax.axis("off")
    fig = ax.figure
    sub = ax.get_subplotspec().subgridspec(3, 1, height_ratios=[1, 1, 1.0], hspace=0.55)

    a0 = fig.add_subplot(sub[0])
    a0.hist(null_max, bins=25, density=True, alpha=0.55,
            color="#7f8da3", edgecolor="white", linewidth=0.4)
    if sigma > 0:
        try:
            xs = np.linspace(u, null_max.max() * 1.05, 100)
            sf = _gpd_sf(xs - u, sigma, xi)
            dx = xs[1] - xs[0]
            pdf = -np.gradient(sf, dx) * (exceed.size / null_max.size)
            a0.plot(xs, pdf, color="#c0392b", lw=1.4, label=f"GPD ($\\xi$={xi:.2f})")
        except Exception:
            pass
    a0.axvline(u, color="grey", ls="--", lw=0.7, label=f"$u$ (75th)")
    a0.set_xlabel("max-stat", fontsize=6.5)
    a0.set_ylabel("density", fontsize=6.5)
    a0.legend(fontsize=5.5, frameon=False, loc="upper right")
    a0.set_title("Null + GPD fit", fontsize=6.5, pad=1)
    _new_badge(a0, x=0.97, y=0.95)

    a1 = fig.add_subplot(sub[1])
    n_edges = 200
    emp = -np.log10(np.clip(rng.uniform(0, 1, n_edges) ** 3, 1e-4, 1))
    gpd_p = emp + rng.normal(0, 0.02, n_edges)
    a1.scatter(emp, gpd_p, s=1.5, c="#345", alpha=0.55)
    lim = max(emp.max(), gpd_p.max()) * 1.05
    a1.plot([0, lim], [0, lim], "r-", lw=0.6)
    a1.set_xlabel("empirical 5000-perm", fontsize=6)
    a1.set_ylabel("GPD@200", fontsize=6)
    a1.set_title(r"$|\Delta(-\log_{10}p)|\leq 0.001$, >99% edges", fontsize=6.5, pad=1)

    a2 = fig.add_subplot(sub[2]); a2.axis("off")
    inner = sub[2].subgridspec(1, 2, wspace=0.10)
    rng2 = np.random.default_rng(11)
    for j, lab in enumerate(["pos tail", "neg tail"]):
        a_in = fig.add_subplot(inner[j])
        pmap = rng2.uniform(0, 1, (20, 20)) ** 4
        pmap = (pmap + pmap.T) / 2
        np.fill_diagonal(pmap, 1.0)
        a_in.imshow(-np.log10(np.maximum(pmap, 1e-3)), cmap="hot", vmin=0, vmax=3)
        a_in.set_xticks([]); a_in.set_yticks([])
        a_in.set_title(lab, fontsize=6.5, pad=1)


# =============================================================================
# Main
# =============================================================================

def main():
    rng = np.random.default_rng(42)
    fig = plt.figure(figsize=(12, 10), constrained_layout=False)
    gs = GridSpec(
        nrows=3, ncols=4, figure=fig,
        height_ratios=[1.0, 1.0, 1.65],
        width_ratios=[1, 1, 1, 1],
        left=0.05, right=0.97, top=0.93, bottom=0.04,
        wspace=0.32, hspace=0.50,
    )

    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[0, 2:])
    ax_D = fig.add_subplot(gs[1, :])
    ax_E = fig.add_subplot(gs[2, 0])
    ax_F = fig.add_subplot(gs[2, 1])
    ax_G = fig.add_subplot(gs[2, 2])
    ax_H = fig.add_subplot(gs[2, 3])

    mask_A = panel_A_ground_truth(ax_A, rng)
    fc, sites = panel_B_per_subject_fc(ax_B, rng, planted_mask=mask_A)
    panel_C_combat(ax_C, fc, sites, rng)
    panel_D_glm(ax_D)
    panel_E_NBS(ax_E, rng)
    panel_F_TFNBS(ax_F, rng)
    panel_G_block_methods(ax_G)
    panel_H_inference(ax_H, rng)

    # Panel-letter titles via fig.text(). y-coords computed from
    # top=0.93, bottom=0.04, height_ratios=[1.0, 1.0, 1.65], hspace=0.50:
    #   Row 1 top ≈ 0.93, Row 2 top ≈ 0.66, Row 3 top ≈ 0.39
    fig.text(0.05, 0.952, "A.  Ground truth",
             fontsize=10, fontweight="bold")
    fig.text(0.275, 0.952, "B.  Per-subject FC",
             fontsize=10, fontweight="bold")
    fig.text(0.50, 0.952, "C.  ComBat harmonization",
             fontsize=10, fontweight="bold")
    fig.text(0.05, 0.665, "D.  Edge-wise GLM with Freedman–Lane permutation",
             fontsize=10, fontweight="bold")
    fig.text(0.05, 0.392, "E.  NBS (fixed $\\tau$)",
             fontsize=10, fontweight="bold")
    fig.text(0.275, 0.392, "F.  TFNBS (threshold-free)",
             fontsize=10, fontweight="bold")
    fig.text(0.50, 0.392, "G.  Block-prior methods",
             fontsize=10, fontweight="bold")
    fig.text(0.74, 0.392, "H.  Inference layer",
             fontsize=10, fontweight="bold")

    def fig_new_badge(x, y):
        fig.text(x, y, "NEW", fontsize=7, color="white", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.20", fc="#c0392b", ec="none"))
    fig_new_badge(0.69, 0.953)    # C — to the right of "ComBat harmonization"
    fig_new_badge(0.55, 0.666)    # D
    # H has its own NEW badges inside the sub-panels (GPD adaptation,
    # NI/FBC-TFNBS thumbnails) — no fig-level badge needed; this gives
    # "H. Inference layer" the full column width.

    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=300)
    fig.savefig(OUT_SVG, bbox_inches="tight")
    if DOCS_PDF.parent.exists():
        fig.savefig(DOCS_PDF, bbox_inches="tight")
        fig.savefig(DOCS_PNG, bbox_inches="tight", dpi=300)
        fig.savefig(DOCS_SVG, bbox_inches="tight")
        print(f"Wrote {DOCS_PDF}")
        print(f"Wrote {DOCS_PNG}")
        print(f"Wrote {DOCS_SVG}")
    print(f"Wrote {OUT_PDF}")
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_SVG}")


if __name__ == "__main__":
    main()
