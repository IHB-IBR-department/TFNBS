"""Diagnostic plotting helpers for ConnInfPy results.

Three minimal helpers that cover the common post-inference figures
without forcing every user to reinvent them:

- :func:`plot_block_mass` — Yeo-7 (or arbitrary partition) block-mass
  heatmap of significant edges.
- :func:`plot_p_map` — symmetric edge-level p-value heatmap with optional
  significant-edge highlight.
- :func:`plot_null_max_distribution` — max-stat null distribution
  histogram with optional GPD tail-fit overlay.

Each helper accepts an optional ``ax=`` argument (creating a fresh
figure if omitted) and returns the :class:`matplotlib.axes.Axes` so
the caller can compose multi-panel figures.

Importing ``conninfpy.plot`` requires :mod:`matplotlib`. The plotting
helpers are intentionally not imported at the top level of
:mod:`conninfpy` — call ``import conninfpy.plot`` only if you need
them, so headless environments (CI, batch jobs) don't pull matplotlib.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
except ImportError as exc:  # pragma: no cover (env-specific)
    raise ImportError(
        "conninfpy.plot requires matplotlib; install via `pip install matplotlib`."
    ) from exc

from .harmonize import block_mass as _block_mass


__all__ = [
    "plot_block_mass",
    "plot_p_map",
    "plot_null_max_distribution",
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
