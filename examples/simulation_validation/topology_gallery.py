"""
Topology gallery for the 10 representative scenarios used in power/FPR analysis.

For each scenario, displays:
  - Ground-truth effect mask (binary/weighted)
  - Uncorrected signed t-statistic from simulated two-sample data

Generation parameters match fpr_config_local.yaml / sweep_config_power_local.yaml:
  n_nodes=60, n_modules=4, intra_corr=0.3, inter_corr=0.05, noise_level=0.05

Usage:
    python examples/simulation_validation/topology_gallery.py
    python examples/simulation_validation/topology_gallery.py --output-dir examples/simulation_validation/results/topologies
    python examples/simulation_validation/topology_gallery.py --effect-size 0.5 --n-samples 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))

from conninfpy.topologies import TopologyDatasetGenerator
try:
    from ._plotting import draw_module_boundaries as _draw_module_boundaries
except ImportError:  # Support direct execution: python examples/.../topology_gallery.py
    from _plotting import draw_module_boundaries as _draw_module_boundaries
from conninfpy.pairwise_stats import compute_t_stat

# The 10 scenarios used in power analysis and FPR calibration configs.
SCENARIOS: List[str] = [
    "within_module_dense",
    "between_modules_dense",
    "partial_bipartite_between_modules",
    "gradient_core_periphery_within_module",
    "scattered_cross_block",
    "hub",
    "rich_club",
    "cross_block_connected_chain",
    "chain",
    "fragmented_within_module",
]

# Short display names for plot titles.
DISPLAY_NAMES = {
    "within_module_dense": "Within-module dense",
    "between_modules_dense": "Between-modules dense",
    "partial_bipartite_between_modules": "Partial bipartite",
    "gradient_core_periphery_within_module": "Core-periphery gradient",
    "scattered_cross_block": "Scattered cross-block",
    "hub": "Hub (star)",
    "rich_club": "Rich club",
    "cross_block_connected_chain": "Cross-block chain",
    "chain": "Chain",
    "fragmented_within_module": "Fragmented within-module",
}


def plot_topology_gallery(
    n_nodes: int = 60,
    n_modules: int = 4,
    intra_corr: float = 0.3,
    inter_corr: float = 0.05,
    noise_level: float = 0.05,
    effect_size: float = 0.3,
    n_samples: int = 30,
    seed: int = 42,
    output_dir: Path = Path("examples/output"),
) -> Path:
    """Generate a 10x2 gallery figure: ground truth + t-stat for each scenario."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gen = TopologyDatasetGenerator(
        n_nodes=n_nodes,
        n_modules=n_modules,
        intra_corr=intra_corr,
        inter_corr=inter_corr,
        noise_level=noise_level,
        seed=seed,
    )

    nrows = 5
    ncols = 4  # 2 scenarios per row × 2 panels (GT + t-stat)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 20))

    for idx, scenario_name in enumerate(SCENARIOS):
        row = idx // 2
        col_base = (idx % 2) * 2  # 0 or 2

        ds = gen.generate(
            scenario_name,
            effect_size=effect_size,
            n_samples=n_samples,
            time_points=30,
        )

        g1_z, g2_z = ds.fisher_z()
        t_dict = compute_t_stat(g1_z, g2_z, test_type="two-sample")
        t_signed = t_dict["g2>g1"] - t_dict["g1>g2"]

        effect_gt = ds.effect_mask * ds.effect_size
        n_edges = int(np.sum(ds.effect_mask != 0) // 2)
        display_name = DISPLAY_NAMES.get(scenario_name, scenario_name)

        # Ground truth panel
        ax_gt = axes[row, col_base]
        max_gt = float(np.max(np.abs(effect_gt))) if np.any(effect_gt) else 1.0
        im_gt = ax_gt.imshow(effect_gt, cmap="seismic", vmin=-max_gt, vmax=max_gt)
        ax_gt.set_title(f"{display_name}\nGT ({n_edges} edges)", fontsize=9)
        _draw_module_boundaries(ax_gt, ds.net_labels)
        fig.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)

        # T-stat panel
        ax_t = axes[row, col_base + 1]
        max_t = float(np.max(np.abs(t_signed))) if np.any(t_signed) else 1.0
        im_t = ax_t.imshow(t_signed, cmap="seismic", vmin=-max_t, vmax=max_t)
        ax_t.set_title(f"{display_name}\nt-stat (uncorrected)", fontsize=9)
        _draw_module_boundaries(ax_t, ds.net_labels)
        fig.colorbar(im_t, ax=ax_t, fraction=0.046, pad=0.04)

        for ax in (ax_gt, ax_t):
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        f"Topology scenarios  |  effect_size={effect_size}  |  "
        f"n_samples={n_samples}  |  {n_nodes} nodes, {n_modules} modules",
        fontsize=13,
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"topologies_gallery_es{effect_size:g}_n{n_samples}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot ground-truth masks and t-stats for the 10 representative topologies."
    )
    parser.add_argument("--n-nodes", type=int, default=60)
    parser.add_argument("--n-modules", type=int, default=4)
    parser.add_argument("--intra-corr", type=float, default=0.3)
    parser.add_argument("--inter-corr", type=float, default=0.05)
    parser.add_argument("--noise-level", type=float, default=0.05)
    parser.add_argument("--effect-size", type=float, default=0.3)
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "results" / "topologies",
    )
    args = parser.parse_args(argv)

    out_path = plot_topology_gallery(
        n_nodes=args.n_nodes,
        n_modules=args.n_modules,
        intra_corr=args.intra_corr,
        inter_corr=args.inter_corr,
        noise_level=args.noise_level,
        effect_size=args.effect_size,
        n_samples=args.n_samples,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
