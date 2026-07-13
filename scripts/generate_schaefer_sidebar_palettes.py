#!/usr/bin/env python3
"""Create compact, centroid-based Yeo-7 sidebar palettes for Schaefer atlases."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from nilearn.plotting import plot_connectome


ROOT = Path(__file__).resolve().parents[1]
ATLAS_DIR = ROOT / "conninfpy" / "atlas_data"
ASSET_DIR = ROOT / "apps" / "assets"

NETWORK_COLORS = {
    "Visual": "#6D28D9",
    "Somatomotor": "#2563EB",
    "Dorsal attention": "#06B6D4",
    "Ventral attention": "#14B8A6",
    "Limbic": "#FACC15",
    "Frontoparietal": "#F97316",
    "Default": "#DC2626",
}
NETWORK_ORDER = list(NETWORK_COLORS)
NETWORK_ALIASES = {
    "SomMot": "Somatomotor",
    "DorsAttn": "Dorsal attention",
    "SalVentAttn": "Ventral attention",
    "Cont": "Frontoparietal",
}
PROJECTIONS = [
    ("Sagittal", "x"),
    ("Coronal", "y"),
    ("Axial", "z"),
]


def _draw_projection(fig, rect, atlas: pd.DataFrame, display_mode: str, marker_size: float) -> None:
    ax = fig.add_axes(rect)
    coords = atlas[["x", "y", "z"]].to_numpy()
    node_colors = [NETWORK_COLORS[network] for network in atlas["network"]]
    plot_connectome(
        np.zeros((len(atlas), len(atlas))),
        coords,
        node_color=node_colors,
        node_size=marker_size,
        display_mode=display_mode,
        figure=fig,
        axes=ax,
        title=None,
        annotate=False,
        black_bg=False,
        alpha=0.95,
        colorbar=False,
        node_kwargs={"edgecolors": "white", "linewidths": 0.25},
    )


def create_palette(n_rois: int) -> Path:
    atlas = pd.read_csv(ATLAS_DIR / f"schaefer{n_rois}_yeo7.csv")
    atlas["network"] = atlas["network"].replace(NETWORK_ALIASES)
    marker_size = {100: 55, 200: 34, 400: 18}[n_rois]

    fig = plt.figure(figsize=(4.77, 15.62), dpi=100, facecolor="white")

    fig.text(0.08, 0.972, "Yeo-7", fontsize=29, fontweight="bold", color="#1F2937", va="top")
    fig.text(0.08, 0.936, f"Schaefer-{n_rois} network palette", fontsize=15, color="#64748B", va="top")

    for rect, (title, display_mode) in zip(
        ([0.08, 0.66, 0.84, 0.22], [0.08, 0.39, 0.84, 0.22], [0.08, 0.12, 0.84, 0.22]),
        PROJECTIONS,
    ):
        _draw_projection(fig, rect, atlas, display_mode, marker_size)
        fig.text(rect[0], rect[1] + rect[3] + 0.008, title, fontsize=13, color="#64748B", va="bottom")

    legend_ax = fig.add_axes([0.08, 0.01, 0.84, 0.12])
    legend_ax.axis("off")
    for index, network in enumerate(NETWORK_ORDER):
        y = 0.92 - index * 0.125
        legend_ax.scatter(0.04, y, s=150, marker="s", color=NETWORK_COLORS[network], edgecolors="none")
        legend_ax.text(0.10, y, network, fontsize=13, color="#334155", va="center")
    legend_ax.text(0.04, 0.01, "Generated from packaged ROI centroids", fontsize=9.5, color="#94A3B8", va="bottom")
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)

    output = ASSET_DIR / f"yeo7_schaefer{n_rois}_sidebar_palette.png"
    fig.savefig(output, facecolor="white")
    plt.close(fig)
    return output


if __name__ == "__main__":
    for n_rois in (100, 200, 400):
        print(create_palette(n_rois))
