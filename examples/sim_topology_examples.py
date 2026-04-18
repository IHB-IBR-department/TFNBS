"""
Topology simulation gallery (symmetric connectomes).

Visualization demo for the topology scenarios in
:mod:`conninfpy.topologies`. For each scenario, renders:

1. the ground-truth signed effect matrix (expected Group2 − Group1 difference),
2. a simple signed t-statistic map from simulated noisy samples (two-sample Welch).

Meant for quick qualitative inspection of how different topological patterns
look in matrix space, before running full permutation-based inference.

Outputs are saved into ``examples/output/`` by default (no images are committed).

Note on effect size vs t-stat magnitude
---------------------------------------
The observed t-stat magnitude depends strongly on the *between-subject*
variability of edge values. If subject noise is too small, even modest
absolute shifts (e.g. 0.2) produce unrealistically large t-stats. Defaults
are chosen so that ``effect_size=0.2`` typically yields |t| in the ~2–5
range on masked edges.

Usage
-----

    python examples/sim_topology_examples.py --scenario chain
    python examples/sim_topology_examples.py --all-scenarios
    python examples/sim_topology_examples.py --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

# Add repo root to path so the example works without editable install.
sys.path.append(str(Path(__file__).parent.parent))

from conninfpy.pairwise_stats import compute_t_stat
from conninfpy.topologies import (
    TopologyDataset,
    TopologyDatasetGenerator,
    TopologyScenario,
    get_scenario,
    list_scenarios,
)


ArrayF = np.ndarray


# -----------------------------------------------------------------------------
# Plotting helpers (public — MICCAI scripts import draw_module_boundaries)
# -----------------------------------------------------------------------------

def draw_module_boundaries(ax, labels: np.ndarray) -> None:
    """Overlay dashed lines at module boundaries on an imshow axis."""
    boundaries = np.where(labels[:-1] != labels[1:])[0] + 0.5
    for b in boundaries:
        ax.axhline(b, color="black", linestyle="--", linewidth=0.5, alpha=0.25)
        ax.axvline(b, color="black", linestyle="--", linewidth=0.5, alpha=0.25)


def _slug(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def _effect_size_tag(effect_size: float) -> str:
    """Make a filename-safe tag like ``es0p3`` for ``effect_size=0.3``."""
    text = f"{effect_size:g}"
    text = text.replace("-", "m").replace(".", "p")
    return f"es{text}"


def _plot_gt_and_tstat(
    gt_effect: ArrayF,
    t_stat_signed: ArrayF,
    labels: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Symmetrize + zero diagonal for consistent display.
    gt_effect = (gt_effect + gt_effect.T) / 2.0
    np.fill_diagonal(gt_effect, 0.0)
    t_stat_signed = (t_stat_signed + t_stat_signed.T) / 2.0
    np.fill_diagonal(t_stat_signed, 0.0)

    max_gt = float(np.max(np.abs(gt_effect))) if np.any(gt_effect) else 1.0
    max_t = float(np.max(np.abs(t_stat_signed))) if np.any(t_stat_signed) else 1.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=14)

    im0 = axes[0].imshow(gt_effect, cmap="seismic", vmin=-max_gt, vmax=max_gt)
    axes[0].set_title("Ground Truth Effect (Group2 − Group1)")
    draw_module_boundaries(axes[0], labels)
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(t_stat_signed, cmap="seismic", vmin=-max_t, vmax=max_t)
    axes[1].set_title("Signed t-stat (two-sample)")
    draw_module_boundaries(axes[1], labels)
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Demo runner
# -----------------------------------------------------------------------------

def render_scenario(
    scenario: TopologyScenario,
    gen: TopologyDatasetGenerator,
    output_dir: Path,
    effect_size: float,
    n_samples: int,
    time_points: int,
) -> Path:
    """Generate one scenario + signed-t-stat and save a comparison PNG."""
    ds: TopologyDataset = gen.generate(
        scenario,
        effect_size=effect_size,
        n_samples=n_samples,
        time_points=time_points,
    )

    g1_z, g2_z = ds.fisher_z()
    t_dict = compute_t_stat(g1_z, g2_z, test_type="two-sample")
    t_signed = t_dict["g2>g1"] - t_dict["g1>g2"]

    out_path = (
        output_dir
        / f"topology_{_slug(scenario.name)}_{_effect_size_tag(effect_size)}_n{n_samples}.png"
    )
    _plot_gt_and_tstat(
        ds.effect_mask * ds.effect_size,
        t_signed,
        ds.net_labels,
        title=f"{scenario.name} (effect_size={effect_size:g}, n={n_samples})",
        out_path=out_path,
    )
    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--scenario", type=str, help="Scenario name to render.")
    parser.add_argument("--all-scenarios", action="store_true",
                        help="Render every scenario in the registry.")
    parser.add_argument("--list", action="store_true",
                        help="List scenario names and exit.")
    parser.add_argument("--n-nodes", type=int, default=60)
    parser.add_argument("--n-modules", type=int, default=4)
    parser.add_argument("--effect-size", type=float, default=0.25)
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--time-points", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).parent / "output")
    args = parser.parse_args(argv)

    if args.list:
        for name in list_scenarios():
            print(name)
        return 0

    gen = TopologyDatasetGenerator(
        n_nodes=args.n_nodes,
        n_modules=args.n_modules,
        seed=args.seed,
    )

    if args.all_scenarios:
        scenarios = [get_scenario(name) for name in list_scenarios()]
    elif args.scenario:
        scenarios = [get_scenario(args.scenario)]
    else:
        parser.error("Provide --scenario NAME, --all-scenarios, or --list.")
        return 2  # unreachable; parser.error exits

    for s in scenarios:
        out = render_scenario(
            s, gen,
            output_dir=args.output_dir,
            effect_size=args.effect_size,
            n_samples=args.n_samples,
            time_points=args.time_points,
        )
        print(f"wrote {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
