"""
Generate synthetic topology datasets for TFNBS analysis.

This script generates synthetic functional connectivity data with specific
topological effect patterns and saves them for later analysis.

Usage:
    python syntetic_experiments/generate_topology_example.py --config syntetic_experiments/topology_config.yaml

The script generates:
- Group 1 and Group 2 connectivity matrices (Fisher z-transformed)
- Effect mask (ground truth)
- Network labels
- Metadata JSON with all parameters
- Visualization images (PNG): effect mask and uncorrected t-statistics
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

# Add repo root to path so script works without installation
sys.path.append(str(Path(__file__).parent.parent))

from syntetic_experiments.sim_topology_examples import (
    TopologyDatasetGenerator,
    get_scenario,
    list_scenarios,
)
from tfnbs.utils import fisher_r_to_z
from tfnbs.pairwise_stats import compute_t_stat


def save_visualization(
    effect_mask: np.ndarray,
    t_stats_pos: np.ndarray,
    t_stats_neg: np.ndarray,
    net_labels: np.ndarray,
    output_path: str,
    topology: str,
    effect_size: float,
) -> None:
    """
    Save visualization of effect mask and uncorrected t-statistics.

    Parameters
    ----------
    effect_mask : np.ndarray
        Ground truth effect mask (N x N).
    t_stats_pos : np.ndarray
        Positive t-statistics (g2 > g1).
    t_stats_neg : np.ndarray
        Negative t-statistics (g1 > g2).
    net_labels : np.ndarray
        Network module labels for each node.
    output_path : str
        Path to save the PNG image.
    topology : str
        Name of the topology.
    effect_size : float
        Effect size used.
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Effect mask (ground truth)
    ax = axes[0]
    mask_display = effect_mask.copy()
    im1 = ax.imshow(mask_display, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    ax.set_title(f'Ground Truth Effect Mask\n({topology})', fontsize=11)
    ax.set_xlabel('Node')
    ax.set_ylabel('Node')
    _draw_module_boundaries(ax, net_labels)
    plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04, label='Effect weight')

    # 2. Uncorrected t-statistics (positive: g2 > g1)
    ax = axes[1]
    # Combine into signed t-map for display
    t_combined = t_stats_pos - t_stats_neg
    vmax = max(np.abs(t_combined).max(), 0.1)
    im2 = ax.imshow(t_combined, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
    ax.set_title(f'Uncorrected T-statistics\n(ES={effect_size})', fontsize=11)
    ax.set_xlabel('Node')
    ax.set_ylabel('Node')
    _draw_module_boundaries(ax, net_labels)
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04, label='t-value')

    # 3. Thresholded t-statistics (|t| > 2)
    ax = axes[2]
    t_thresh = t_combined.copy()
    t_thresh[np.abs(t_thresh) < 2.0] = 0
    im3 = ax.imshow(t_thresh, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
    ax.set_title('T-statistics (|t| > 2)\n(uncorrected)', fontsize=11)
    ax.set_xlabel('Node')
    ax.set_ylabel('Node')
    _draw_module_boundaries(ax, net_labels)
    plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04, label='t-value')

    plt.tight_layout()
    plt.savefig(output_path, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)




def _draw_module_boundaries(ax, labels: np.ndarray) -> None:
    """Draw dashed lines at module boundaries."""
    boundaries = np.where(labels[:-1] != labels[1:])[0] + 0.5
    for b in boundaries:
        ax.axhline(b, color="black", linestyle="--", linewidth=0.5, alpha=0.3)
        ax.axvline(b, color="black", linestyle="--", linewidth=0.5, alpha=0.3)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_and_save_dataset(
    output_dir: str,
    topology: str,
    effect_size: float,
    n_samples: int,
    n_nodes: int = 60,
    n_modules: int = 4,
    time_points: int = 50,
    intra_corr: float = 0.3,
    inter_corr: float = 0.05,
    noise_level: float = 0.05,
    seed: int = 42,
    save_raw: bool = True,
    scenario_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate a synthetic topology dataset and save to disk.

    Parameters
    ----------
    output_dir : str
        Directory to save output files.
    topology : str
        Name of the topology scenario (e.g., 'within_module_dense', 'hub').
    effect_size : float
        Magnitude of the effect (Cohen's d-like).
    n_samples : int
        Number of subjects per group.
    n_nodes : int
        Number of nodes (ROIs).
    n_modules : int
        Number of functional modules.
    time_points : int
        Length of simulated time series per subject.
    intra_corr : float
        Base correlation within modules.
    inter_corr : float
        Base correlation between modules.
    noise_level : float
        Noise added to covariance structure.
    seed : int
        Random seed for reproducibility.
    save_raw : bool
        If True, also save raw correlation matrices (not Fisher z-transformed).
    scenario_params : dict, optional
        Additional parameters for the topology scenario.

    Returns
    -------
    dict
        Metadata about the generated dataset.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize generator
    gen = TopologyDatasetGenerator(
        n_nodes=n_nodes,
        n_modules=n_modules,
        intra_corr=intra_corr,
        inter_corr=inter_corr,
        noise_level=noise_level,
        seed=seed,
    )

    # Generate dataset
    dataset = gen.generate(
        scenario=topology,
        effect_size=effect_size,
        n_samples=n_samples,
        time_points=time_points,
        scenario_params=scenario_params or {},
    )

    # Fisher z-transform
    group1_z, group2_z = dataset.fisher_z()

    # Prepare metadata
    metadata = {
        "topology": topology,
        "effect_size": effect_size,
        "n_samples_g1": dataset.group1.shape[0],
        "n_samples_g2": dataset.group2.shape[0],
        "n_nodes": n_nodes,
        "n_modules": n_modules,
        "time_points": time_points,
        "intra_corr": intra_corr,
        "inter_corr": inter_corr,
        "noise_level": noise_level,
        "seed": seed,
        "scenario_params": scenario_params or {},
        "n_effect_edges": int(np.sum(dataset.effect_mask != 0) // 2),  # Upper triangle only
    }
    metadata.update(dataset.meta)

    # Save files
    prefix = f"{topology}_es{effect_size}_n{n_samples}"

    # Fisher z-transformed data (primary output)
    np.save(os.path.join(output_dir, f"{prefix}_group1_z.npy"), group1_z)
    np.save(os.path.join(output_dir, f"{prefix}_group2_z.npy"), group2_z)

    # Raw correlation matrices (optional)
    if save_raw:
        np.save(os.path.join(output_dir, f"{prefix}_group1_r.npy"), dataset.group1)
        np.save(os.path.join(output_dir, f"{prefix}_group2_r.npy"), dataset.group2)

    # Effect mask and labels
    np.save(os.path.join(output_dir, f"{prefix}_effect_mask.npy"), dataset.effect_mask)
    np.save(os.path.join(output_dir, f"{prefix}_net_labels.npy"), dataset.net_labels)

    # Compute uncorrected t-statistics for visualization
    t_stats = compute_t_stat(group1_z, group2_z, test_type="two-sample")
    t_stats_pos = t_stats['g2>g1']  # Positive effects (group2 > group1)
    t_stats_neg = t_stats['g1>g2']  # Negative effects (group1 > group2)

    # Save t-statistics
    np.save(os.path.join(output_dir, f"{prefix}_tstat_pos.npy"), t_stats_pos)
    np.save(os.path.join(output_dir, f"{prefix}_tstat_neg.npy"), t_stats_neg)

    # Save visualization image (PNG)
    viz_path = os.path.join(output_dir, f"{prefix}_visualization.png")
    save_visualization(
        effect_mask=dataset.effect_mask,
        t_stats_pos=t_stats_pos,
        t_stats_neg=t_stats_neg,
        net_labels=dataset.net_labels,
        output_path=viz_path,
        topology=topology,
        effect_size=effect_size,
    )

    # Metadata
    with open(os.path.join(output_dir, f"{prefix}_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved dataset: {prefix}")
    print(f"  - Group 1: {dataset.group1.shape}")
    print(f"  - Group 2: {dataset.group2.shape}")
    print(f"  - Effect edges: {metadata['n_effect_edges']}")
    print(f"  - Visualization: {viz_path}")

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic topology datasets for TFNBS analysis."
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--topology",
        type=str,
        help="Topology scenario name (overrides config).",
    )
    parser.add_argument(
        "--effect-size",
        type=float,
        help="Effect size (overrides config).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        help="Number of samples per group (overrides config).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (overrides config).",
    )
    parser.add_argument(
        "--list-topologies",
        action="store_true",
        help="List available topology scenarios and exit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed (overrides config).",
    )

    args = parser.parse_args()

    # List topologies if requested
    if args.list_topologies:
        print("Available topology scenarios:")
        for name in list_scenarios():
            print(f"  - {name}")
        return

    # Load config or use defaults
    if args.config:
        config = load_config(args.config)
    else:
        config = {}

    # Override with command-line arguments
    if args.topology:
        config["topology"] = args.topology
    if args.effect_size is not None:
        config["effect_size"] = args.effect_size
    if args.n_samples is not None:
        config["n_samples"] = args.n_samples
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.seed is not None:
        config["seed"] = args.seed

    # Set defaults
    config.setdefault("topology", "within_module_dense")
    config.setdefault("effect_size", 0.25)
    config.setdefault("n_samples", 20)
    config.setdefault("n_nodes", 60)
    config.setdefault("n_modules", 4)
    config.setdefault("time_points", 50)
    config.setdefault("intra_corr", 0.3)
    config.setdefault("inter_corr", 0.05)
    config.setdefault("noise_level", 0.05)
    config.setdefault("seed", 42)
    config.setdefault("output_dir", "syntetic_experiments/output/generated_data")
    config.setdefault("save_raw", True)
    config.setdefault("scenario_params", {})

    # Handle multiple topologies
    topologies = config.get("topologies", [config["topology"]])
    if isinstance(topologies, str):
        topologies = [topologies]

    # Handle multiple effect sizes
    effect_sizes = config.get("effect_sizes", [config["effect_size"]])
    if isinstance(effect_sizes, (int, float)):
        effect_sizes = [effect_sizes]

    # Generate datasets
    all_metadata = []
    for topology in topologies:
        for effect_size in effect_sizes:
            metadata = generate_and_save_dataset(
                output_dir=config["output_dir"],
                topology=topology,
                effect_size=effect_size,
                n_samples=config["n_samples"],
                n_nodes=config["n_nodes"],
                n_modules=config["n_modules"],
                time_points=config["time_points"],
                intra_corr=config["intra_corr"],
                inter_corr=config["inter_corr"],
                noise_level=config["noise_level"],
                seed=config["seed"],
                save_raw=config["save_raw"],
                scenario_params=config.get("scenario_params", {}).get(topology, {}),
            )
            all_metadata.append(metadata)

    # Save combined metadata
    summary_path = os.path.join(config["output_dir"], "generation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump({
            "config": config,
            "datasets": all_metadata,
        }, f, indent=2)

    print(f"\nGenerated {len(all_metadata)} dataset(s)")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
