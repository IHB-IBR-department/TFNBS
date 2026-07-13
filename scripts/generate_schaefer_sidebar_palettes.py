#!/usr/bin/env python3
"""Create compact, centroid-based Yeo-7 sidebar palettes for Schaefer atlases."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull


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
    ("Sagittal", "y", "z"),
    ("Coronal", "x", "z"),
    ("Axial", "x", "y"),
]


def _draw_projection(ax, atlas: pd.DataFrame, title: str, horizontal: str, vertical: str, marker_size: float) -> None:
    points = atlas[[horizontal, vertical]].to_numpy()
    hull = ConvexHull(points)
    outline = points[hull.vertices]
    ax.add_patch(
        Polygon(outline, closed=True, facecolor="#F8FAFC", edgecolor="#94A3B8", linewidth=1.2, zorder=0)
    )

    for network in NETWORK_ORDER:
        subset = atlas.loc[atlas["network"] == network]
        ax.scatter(
            subset[horizontal],
            subset[vertical],
            s=marker_size,
            color=NETWORK_COLORS[network],
            edgecolors="white",
            linewidths=0.35,
            alpha=0.94,
            zorder=2,
        )

    x_span = np.ptp(points[:, 0])
    y_span = np.ptp(points[:, 1])
    ax.set_xlim(points[:, 0].min() - x_span * 0.09, points[:, 0].max() + x_span * 0.09)
    ax.set_ylim(points[:, 1].min() - y_span * 0.09, points[:, 1].max() + y_span * 0.09)
    ax.set_aspect("equal")
    ax.set_title(title, loc="left", fontsize=13, color="#475569", pad=7)
    ax.axis("off")


def create_palette(n_rois: int) -> Path:
    atlas = pd.read_csv(ATLAS_DIR / f"schaefer{n_rois}_yeo7.csv")
    atlas["network"] = atlas["network"].replace(NETWORK_ALIASES)
    marker_size = {100: 52, 200: 30, 400: 15}[n_rois]

    fig = plt.figure(figsize=(4.77, 15.62), dpi=100, facecolor="white")
    fig.subplots_adjust(left=0.08, right=0.92, top=0.91, bottom=0.05, hspace=0.26)
    grid = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 0.7])

    fig.text(0.08, 0.972, "Yeo-7", fontsize=29, fontweight="bold", color="#1F2937", va="top")
    fig.text(0.08, 0.936, f"Schaefer-{n_rois} network palette", fontsize=15, color="#64748B", va="top")

    for index, (title, horizontal, vertical) in enumerate(PROJECTIONS):
        _draw_projection(fig.add_subplot(grid[index]), atlas, title, horizontal, vertical, marker_size)

    legend_ax = fig.add_subplot(grid[3])
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
