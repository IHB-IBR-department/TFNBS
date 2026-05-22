"""Diagnostic plotting helpers for ConnInfPy results.

Per-panel helpers and a one-call publication figure:

- :func:`plot_block_mass` — Yeo-7 (or arbitrary partition) block-mass
  heatmap of significant edges.
- :func:`plot_p_map` — symmetric edge-level p-value heatmap with
  optional significant-edge highlight.
- :func:`plot_null_max_distribution` — max-stat null distribution
  histogram with optional GPD tail-fit overlay.
- :func:`plot_effect_matrix` — signed effect heatmap (RdBu_r, symmetric
  vmax) with significant edges outlined; optional network-block
  ordering when an :class:`AtlasInfo` is supplied.
- :func:`plot_network_summary` — P × P block heatmap of the signed
  proportion of significant edges per (network_i × network_j) cell;
  more robust than the edge-level map at high node counts.
- :func:`summary_figure` — publication-ready 2×2 layout combining the
  effect matrix, block summary, p-maps, and top-k edge table.

Each per-panel helper accepts an optional ``ax=`` argument (creating a
fresh figure if omitted) and returns the
:class:`matplotlib.axes.Axes` so the caller can compose multi-panel
figures. :func:`summary_figure` returns a :class:`matplotlib.figure.Figure`.

Importing ``conninfpy.plot`` requires :mod:`matplotlib`. The plotting
helpers are intentionally not imported at the top level of
:mod:`conninfpy` — call ``import conninfpy.plot`` only if you need
them, so headless environments (CI, batch jobs) don't pull matplotlib.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
except ImportError as exc:  # pragma: no cover (env-specific)
    raise ImportError(
        "conninfpy.plot requires matplotlib; install via `pip install matplotlib`."
    ) from exc

from .harmonize import block_mass as _block_mass

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ._result import InferenceResult, OmnibusInferenceResult
    from .atlas import AtlasInfo


__all__ = [
    "plot_block_mass",
    "plot_p_map",
    "plot_null_max_distribution",
    "plot_effect_matrix",
    "plot_network_summary",
    "summary_figure",
]


def plot_block_mass(
    p_map: npt.NDArray[np.float64],
    net_labels: npt.NDArray[np.int_],
    *,
    network_names: Optional[Sequence[str]] = None,
    alpha: float = 0.05,
    ax: Optional[Axes] = None,
    cmap: str = "viridis",
    annotate: bool = True,
    title: Optional[str] = None,
) -> Axes:
    """Yeo-style block-mass heatmap of significant edges.

    Parameters
    ----------
    p_map : ndarray of shape (N, N)
        Edge-level p-value matrix (symmetric, diagonal=1).
    net_labels : ndarray of shape (N,)
        Integer network assignment per node, ``0..K-1``.
    network_names : sequence of str, optional
        Display names for the K networks. Defaults to ``["Net 0", ...]``.
    alpha : float, default 0.05
        Significance threshold for the underlying :func:`block_mass`
        aggregation.
    ax : matplotlib Axes, optional
        Plot into this axis; otherwise a new ``(5, 4)`` figure is
        created.
    cmap : str, default "viridis"
        Matplotlib colormap.
    annotate : bool, default True
        Print edge counts inside each cell.
    title : str, optional
        Axis title.
    """
    M = _block_mass(p_map, net_labels, alpha=alpha, return_upper=True)
    K = M.shape[0]
    if network_names is None:
        network_names = [f"Net {i}" for i in range(K)]
    if ax is None:
        _fig, ax = plt.subplots(figsize=(5, 4))
    vmax = max(int(M.max()), 1)
    ax.imshow(M, cmap=cmap, vmin=0, vmax=vmax)
    if annotate:
        for i in range(K):
            for j in range(K):
                if i <= j and M[i, j] > 0:
                    color = "white" if M[i, j] < vmax / 2 else "black"
                    ax.text(j, i, str(int(M[i, j])),
                            ha="center", va="center",
                            fontsize=7, color=color)
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels(network_names, rotation=30, ha="right", fontsize=7)
    ax.set_yticklabels(network_names, fontsize=7)
    if title:
        ax.set_title(title, fontsize=9)
    return ax


def plot_p_map(
    p_map: npt.NDArray[np.float64],
    *,
    alpha: float = 0.05,
    ax: Optional[Axes] = None,
    cmap: str = "viridis_r",
    log_scale: bool = True,
    title: Optional[str] = None,
    show_significant: bool = True,
) -> Axes:
    """Edge-level p-value heatmap with optional log-scale and sig-mask.

    Parameters
    ----------
    p_map : ndarray of shape (N, N)
        Edge-level p-values.
    alpha : float, default 0.05
        Threshold for the optional significance mask.
    ax : matplotlib Axes, optional
        Plot into this axis; otherwise a new figure is created.
    cmap : str, default "viridis_r"
        Reverse viridis (so darker = more significant).
    log_scale : bool, default True
        Plot ``-log10(p)`` rather than raw ``p``.
    title : str, optional
        Axis title.
    show_significant : bool, default True
        Draw a red contour around edges with ``p <= alpha``.
    """
    if ax is None:
        _fig, ax = plt.subplots(figsize=(4.5, 4))
    if log_scale:
        with np.errstate(divide="ignore"):
            data = -np.log10(np.where(p_map <= 0, 1e-300, p_map))
        cbar_label = r"$-\log_{10}(p)$"
        ref_line = -np.log10(alpha)
    else:
        data = p_map
        cbar_label = r"$p$"
        ref_line = alpha
    im = ax.imshow(data, cmap=cmap)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(cbar_label, fontsize=8)
    if log_scale:
        cbar.ax.axhline(ref_line, color="red", linewidth=0.6,
                        linestyle="--")
    if show_significant:
        sig = (p_map <= alpha) & np.isfinite(p_map)
        np.fill_diagonal(sig, False)
        if sig.any():
            ax.contour(sig.astype(float), levels=[0.5],
                       colors="red", linewidths=0.4)
    ax.set_xlabel("node")
    ax.set_ylabel("node")
    if title:
        ax.set_title(title, fontsize=9)
    return ax


def plot_null_max_distribution(
    null_max: npt.NDArray[np.float64],
    *,
    observed_max: Optional[float] = None,
    gpd_tail: Optional[Tuple[float, float, float]] = None,
    threshold_quantile: float = 0.75,
    ax: Optional[Axes] = None,
    bins: int = 40,
    title: Optional[str] = None,
) -> Axes:
    """Max-statistic null histogram with optional GPD-tail overlay.

    Parameters
    ----------
    null_max : ndarray of shape (J,)
        Per-permutation maximum statistic.
    observed_max : float, optional
        Observed max statistic, drawn as a red vertical line for
        eyeballing FWER significance.
    gpd_tail : tuple ``(u, sigma, xi)``, optional
        GPD parameters from :func:`conninfpy.fit_gpd_tail`. If supplied,
        the GPD survival function is overlaid as a smooth red curve on
        the upper tail.
    threshold_quantile : float, default 0.75
        Empirical quantile at which the GPD tail is anchored — only
        used to position the dashed grey reference line; should match
        the value passed to ``fit_gpd_tail``.
    ax : matplotlib Axes, optional
        Plot into this axis; otherwise a new figure is created.
    bins : int, default 40
        Histogram bin count.
    title : str, optional
        Axis title.
    """
    if ax is None:
        _fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.hist(null_max, bins=bins, density=True, alpha=0.55,
            edgecolor="black", linewidth=0.3,
            label=f"empirical null (J={len(null_max)})")
    u_emp = float(np.quantile(null_max, threshold_quantile))
    ax.axvline(u_emp, color="grey", linestyle="--", linewidth=0.6,
               label=f"u = {threshold_quantile:.0%}-tile")
    if gpd_tail is not None:
        u, sigma, xi = gpd_tail
        xs = np.linspace(u, null_max.max() * 1.02, 200)
        z = (xs - u) / sigma
        # Differentiate the survival function: f(x) = (1/σ) (1 + ξ z)^{-1 - 1/ξ}
        if abs(xi) < 1e-8:
            pdf = (1.0 / sigma) * np.exp(-z)
        else:
            base = np.maximum(0, 1 + xi * z)
            pdf = (1.0 / sigma) * np.power(base, -1.0 - 1.0 / xi)
        ax.plot(xs, pdf, color="red", linewidth=1.3,
                label=f"GPD tail (ξ={xi:.2f})")
    if observed_max is not None:
        ax.axvline(observed_max, color="black", linewidth=0.9,
                   label=f"observed max = {observed_max:.2f}")
    ax.set_xlabel("max statistic across edges")
    ax.set_ylabel("density")
    ax.legend(fontsize=7, frameon=False)
    if title:
        ax.set_title(title, fontsize=9)
    return ax


# =============================================================================
# Result-driven helpers (consume InferenceResult / OmnibusInferenceResult)
# =============================================================================

def _effect_and_pmin_from_result(
    result: Any,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], str]:
    """Extract a signed effect map and per-edge p-min from a result.

    Returns ``(effect, p_min, label)`` where ``label`` is the colorbar
    label (``"t"``, ``"β"``, or ``"F"`` for the omnibus path).
    """
    from ._result import InferenceResult, OmnibusInferenceResult

    if isinstance(result, OmnibusInferenceResult):
        if result.stat_omnibus is None:
            raise ValueError(
                "stat_omnibus is None — v2.0 pickled results cannot be "
                "plotted with plot_effect_matrix; rerun the pipeline."
            )
        return (
            np.asarray(result.stat_omnibus, dtype=np.float64),
            np.asarray(result["omnibus"], dtype=np.float64),
            "F",
        )
    if isinstance(result, InferenceResult):
        signed = result.stat_signed
        if signed is None:
            raise ValueError(
                "stat_positive / stat_negative are None — v2.0 pickled "
                "results cannot be plotted with plot_effect_matrix; "
                "rerun the pipeline."
            )
        p_min = np.minimum(result["positive"], result["negative"])
        label = "β" if result.stat_type == "beta" else "t"
        return (
            np.asarray(signed, dtype=np.float64),
            np.asarray(p_min, dtype=np.float64),
            label,
        )
    raise TypeError(
        f"Unsupported result type {type(result).__name__}; expected "
        "InferenceResult or OmnibusInferenceResult."
    )


def _network_order(
    atlas: "AtlasInfo",
) -> Tuple[npt.NDArray[np.int_], list, list]:
    """Return ``(perm, block_starts, block_names)`` for atlas-ordered plots.

    ``perm`` is the row/column permutation that groups ROIs by network
    in their first-appearance order. ``block_starts`` lists the first
    index of each network block (length K) and ``block_names`` the
    network labels in the same order.
    """
    networks = np.asarray(atlas.networks, dtype=object)
    seen_order: list = []
    for n in networks:
        if n not in seen_order:
            seen_order.append(n)
    perm_parts = [np.where(networks == name)[0] for name in seen_order]
    perm = np.concatenate(perm_parts).astype(int)
    block_starts = [0]
    for part in perm_parts[:-1]:
        block_starts.append(block_starts[-1] + len(part))
    return perm, block_starts, seen_order


def plot_effect_matrix(
    result: Any,
    *,
    alpha: float = 0.05,
    vmax: Optional[float] = None,
    atlas: Optional["AtlasInfo"] = None,
    ax: Optional[Axes] = None,
    order_by_network: bool = True,
    show_network_lines: bool = True,
    title: Optional[str] = None,
) -> Axes:
    """Signed effect heatmap with significant edges outlined.

    Parameters
    ----------
    result : InferenceResult or OmnibusInferenceResult
        Inference output. The signed observed statistic is read from
        ``result.stat_signed`` (t / β path) or ``result.stat_omnibus``
        (F path, unsigned).
    alpha : float, default 0.05
        Significance threshold for the edge-outline overlay
        (``min(p_positive, p_negative) <= alpha`` on the tail path;
        ``p_omnibus <= alpha`` on the F path).
    vmax : float, optional
        Symmetric color bound. Defaults to the 98th percentile of the
        absolute observed statistic.
    atlas : :class:`AtlasInfo`, optional
        When supplied (and ``order_by_network=True``), rows and columns
        are reordered so that same-network ROIs are contiguous; thin
        lines mark the network blocks if ``show_network_lines=True``.
    ax : matplotlib Axes, optional
        Plot into this axis; otherwise a new figure is created.
    order_by_network : bool, default True
        Permute rows/columns to group by network. Ignored without an
        atlas.
    show_network_lines : bool, default True
        Draw light separator lines between network blocks. Ignored
        without an atlas or when ``order_by_network=False``.
    title : str, optional
        Axis title.
    """
    effect, p_min, stat_label = _effect_and_pmin_from_result(result)
    if effect.ndim != 2 or effect.shape[0] != effect.shape[1]:
        raise ValueError(
            f"effect matrix must be square; got {effect.shape}."
        )
    N = effect.shape[0]

    perm: Optional[np.ndarray] = None
    block_starts: list = []
    if atlas is not None and order_by_network:
        if len(atlas) != N:
            raise ValueError(
                f"atlas has {len(atlas)} ROIs but the result is on "
                f"{N} nodes; cannot reorder."
            )
        perm, block_starts, _ = _network_order(atlas)
        effect = effect[np.ix_(perm, perm)]
        p_min = p_min[np.ix_(perm, perm)]

    if ax is None:
        _fig, ax = plt.subplots(figsize=(5, 4.5))

    # Symmetric color bound.
    if vmax is None:
        finite = effect[np.isfinite(effect)]
        if finite.size:
            vmax = float(np.quantile(np.abs(finite), 0.98))
        if not vmax or vmax <= 0:
            vmax = 1.0
    is_unsigned = stat_label == "F"
    if is_unsigned:
        im = ax.imshow(effect, cmap="magma", vmin=0, vmax=vmax)
    else:
        im = ax.imshow(effect, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(stat_label, fontsize=8)

    # Significant-edge contour
    sig = (p_min <= alpha) & np.isfinite(p_min)
    np.fill_diagonal(sig, False)
    if sig.any():
        ax.contour(sig.astype(float), levels=[0.5],
                   colors="black", linewidths=0.5)

    # Network block separators
    if atlas is not None and order_by_network and show_network_lines:
        for s in block_starts[1:]:
            ax.axvline(s - 0.5, color="grey", linewidth=0.4, alpha=0.7)
            ax.axhline(s - 0.5, color="grey", linewidth=0.4, alpha=0.7)

    ax.set_xlabel("node")
    ax.set_ylabel("node")
    if title:
        ax.set_title(title, fontsize=9)
    elif atlas is not None and order_by_network:
        ax.set_title("effect matrix (network-ordered)", fontsize=9)
    return ax


def plot_network_summary(
    result: Any,
    *,
    atlas: "AtlasInfo",
    alpha: float = 0.05,
    ax: Optional[Axes] = None,
    cmap: str = "RdBu_r",
    annotate: bool = True,
    title: Optional[str] = None,
) -> Axes:
    """K × K block heatmap of the signed proportion of significant edges.

    For each pair of networks ``(i, j)`` (canonical, unordered), the
    cell value is

    .. math::

       s_{ij} = \\frac{n_{ij}^{+} - n_{ij}^{-}}{n_{ij}^{\\text{total}}}

    where ``n_{ij}^{+}`` and ``n_{ij}^{-}`` are counts of significant
    positive- and negative-tail edges with one endpoint in network
    ``i`` and the other in network ``j``, and the denominator is the
    total number of edges in that block. The ratio is bounded in
    ``[-1, 1]``; ``+1`` means every block edge is significantly
    positive, ``-1`` significantly negative, ``0`` no net direction.

    For the F-stat omnibus path the cell holds the *unsigned*
    proportion ``n_{ij} / n_{ij}^{total}`` and the colormap defaults
    to a sequential scheme.

    Parameters
    ----------
    result : InferenceResult or OmnibusInferenceResult
        Inference output (uses ``result['positive']`` / ``['negative']``
        or ``['omnibus']`` p-value maps).
    atlas : :class:`AtlasInfo`
        Required — supplies the per-ROI network labels.
    alpha : float, default 0.05
        Significance threshold.
    ax : matplotlib Axes, optional
        Plot into this axis; otherwise a new figure is created.
    cmap : str, default "RdBu_r"
        Matplotlib colormap (sequential for the F-stat path).
    annotate : bool, default True
        Print the cell value inside each block.
    title : str, optional
        Axis title.
    """
    from ._result import InferenceResult, OmnibusInferenceResult

    if not isinstance(result, (InferenceResult, OmnibusInferenceResult)):
        raise TypeError(
            f"Unsupported result type {type(result).__name__}; expected "
            "InferenceResult or OmnibusInferenceResult."
        )
    networks = np.asarray(atlas.networks, dtype=object)
    if isinstance(result, OmnibusInferenceResult):
        p_arr = np.asarray(result["omnibus"], dtype=np.float64)
        N = p_arr.shape[0]
    else:
        p_pos = np.asarray(result["positive"], dtype=np.float64)
        p_neg = np.asarray(result["negative"], dtype=np.float64)
        N = p_pos.shape[0]
    if len(atlas) != N:
        raise ValueError(
            f"atlas has {len(atlas)} ROIs but the result is on "
            f"{N} nodes; cannot summarize."
        )

    # Map each unique network name to an integer code in first-appearance
    # order, so the heatmap rows align with _network_order().
    seen: dict = {}
    for n in networks:
        if n not in seen:
            seen[n] = len(seen)
    codes = np.array([seen[n] for n in networks], dtype=int)
    K = len(seen)
    block_names = list(seen.keys())

    iu, ju = np.triu_indices(N, k=1)
    bi = codes[iu]
    bj = codes[ju]
    lo = np.minimum(bi, bj)
    hi = np.maximum(bi, bj)

    total = np.zeros((K, K), dtype=np.float64)
    np.add.at(total, (lo, hi), 1.0)

    if isinstance(result, OmnibusInferenceResult):
        sig_pos = (p_arr[iu, ju] <= alpha)
        count_pos = np.zeros((K, K), dtype=np.float64)
        np.add.at(count_pos, (lo[sig_pos], hi[sig_pos]), 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            cell = np.where(total > 0, count_pos / total, 0.0)
        vmin, vmax = 0.0, 1.0
        cmap_eff = "magma" if cmap == "RdBu_r" else cmap
    else:
        sig_p = (p_pos[iu, ju] <= alpha)
        sig_n = (p_neg[iu, ju] <= alpha)
        count_p = np.zeros((K, K), dtype=np.float64)
        count_n = np.zeros((K, K), dtype=np.float64)
        np.add.at(count_p, (lo[sig_p], hi[sig_p]), 1.0)
        np.add.at(count_n, (lo[sig_n], hi[sig_n]), 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            cell = np.where(total > 0, (count_p - count_n) / total, 0.0)
        vmax = float(np.max(np.abs(cell))) if cell.size else 1.0
        if vmax <= 0:
            vmax = 1.0
        vmin = -vmax
        cmap_eff = cmap

    # Mirror upper to lower for a symmetric display.
    cell_full = cell + np.triu(cell, 1).T

    if ax is None:
        _fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cell_full, cmap=cmap_eff, vmin=vmin, vmax=vmax)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    if isinstance(result, OmnibusInferenceResult):
        cbar.set_label("sig fraction", fontsize=8)
    else:
        cbar.set_label("signed sig fraction", fontsize=8)
    if annotate:
        for i in range(K):
            for j in range(K):
                if total[min(i, j), max(i, j)] <= 0:
                    continue
                ax.text(j, i, f"{cell_full[i, j]:+.2f}",
                        ha="center", va="center",
                        fontsize=6, color="black")
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels(block_names, rotation=30, ha="right", fontsize=7)
    ax.set_yticklabels(block_names, fontsize=7)
    if title:
        ax.set_title(title, fontsize=9)
    return ax


def summary_figure(
    result: Any,
    *,
    atlas: Optional["AtlasInfo"] = None,
    alpha: float = 0.05,
    top_k: int = 10,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12.0, 9.0),
) -> Figure:
    """Publication-ready 2 × 2 summary of a ConnInfPy inference run.

    Panels:

    - **top-left**: :func:`plot_effect_matrix` (network-ordered if an
      atlas is supplied).
    - **top-right**: :func:`plot_network_summary` if an atlas is
      supplied, otherwise :func:`plot_block_mass` with a fallback
      single-block partition.
    - **bottom-left**: :func:`plot_p_map` of the positive tail
      (``-log10 p`` heatmap; for the F-stat path the single
      ``omnibus`` map is shown).
    - **bottom-right**: top-``top_k`` row table from
      ``result.significant_edges(atlas, sort='p')``.

    Parameters
    ----------
    result : InferenceResult or OmnibusInferenceResult
        Inference output.
    atlas : :class:`AtlasInfo`, optional
        Atlas for the network-ordered effect matrix, network summary
        block, and the edge-table annotation.
    alpha : float, default 0.05
        Significance threshold used by every panel.
    top_k : int, default 10
        Number of rows in the edge table.
    title : str, optional
        Figure-level title.
    figsize : tuple of float, default ``(12, 9)``
        Inches.
    """
    from ._result import InferenceResult, OmnibusInferenceResult

    if not isinstance(result, (InferenceResult, OmnibusInferenceResult)):
        raise TypeError(
            f"Unsupported result type {type(result).__name__}; expected "
            "InferenceResult or OmnibusInferenceResult."
        )

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Top-left: effect matrix
    plot_effect_matrix(result, alpha=alpha, atlas=atlas, ax=axes[0, 0])

    # Top-right: network / block summary
    if atlas is not None:
        plot_network_summary(result, atlas=atlas, alpha=alpha,
                             ax=axes[0, 1])
    else:
        # Fall back to the existing block_mass helper with a single-block
        # partition; not very informative but keeps the layout consistent.
        if isinstance(result, InferenceResult):
            p_for_block = np.minimum(result["positive"], result["negative"])
        else:
            p_for_block = result["omnibus"]
        N = p_for_block.shape[0]
        plot_block_mass(p_for_block, np.zeros(N, dtype=int),
                        network_names=["all"], alpha=alpha,
                        ax=axes[0, 1],
                        title="block mass (no atlas)")

    # Bottom-left: p-map
    if isinstance(result, InferenceResult):
        plot_p_map(result["positive"], alpha=alpha, ax=axes[1, 0],
                   title="positive-tail p-map")
    else:
        plot_p_map(result["omnibus"], alpha=alpha, ax=axes[1, 0],
                   title="omnibus p-map")

    # Bottom-right: top-k edge table
    ax_t = axes[1, 1]
    ax_t.axis("off")
    try:
        df = result.significant_edges(atlas=atlas, alpha=alpha,
                                      sort="p", top_k=top_k)
        if df.empty:
            ax_t.text(0.5, 0.5, "No significant edges at α = "
                      f"{alpha:.3g}", ha="center", va="center",
                      fontsize=10)
        else:
            # Pick a small, readable subset of columns.
            if atlas is not None:
                if isinstance(result, OmnibusInferenceResult):
                    cols = ["roi_i_name", "roi_j_name",
                            "network_pair", "F", "p_omnibus"]
                else:
                    cols = ["roi_i_name", "roi_j_name",
                            "network_pair", "t_signed", "p_min", "tail"]
            else:
                if isinstance(result, OmnibusInferenceResult):
                    cols = ["roi_i", "roi_j", "F", "p_omnibus"]
                else:
                    cols = ["roi_i", "roi_j", "t_signed", "p_min", "tail"]
            cols = [c for c in cols if c in df.columns]
            disp = df[cols].copy()
            # Round numeric columns for compact display.
            for c in disp.columns:
                if disp[c].dtype.kind in "fc":
                    disp[c] = disp[c].map(lambda v: f"{v:.3g}")
            table = ax_t.table(
                cellText=disp.values.tolist(),
                colLabels=list(disp.columns),
                loc="center",
                cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(7)
            table.scale(1.0, 1.2)
            ax_t.set_title(f"top {len(disp)} significant edges",
                           fontsize=9, loc="left")
    except (ValueError, TypeError) as exc:
        # Stat maps missing or other export-time error — show the message
        # so the figure still composes cleanly.
        ax_t.text(0.5, 0.5, f"(edge table unavailable: {exc})",
                  ha="center", va="center", fontsize=8, wrap=True)

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig
