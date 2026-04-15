"""
Method comparison demo on synthetic symmetric connectomes.

This script:
1) builds a ground-truth topology mask (one of the predefined scenarios),
2) generates two groups of correlation matrices using ModularDatasetGenerator,
3) computes the original signed t-stat map (two-sample, Fisher-z domain),
4) computes permutation-based p-values via compute_p_val() for multiple methods,
5) saves a figure with Ground Truth, t-stat, and (1 - p) maps per method,
   including TP/FN counts on the ground-truth edges.

Methods (from methods_description.md):
- tstat (baseline: max-stat correction on raw t)
- tfnbs
- nbs (extent)
- nbs (intensity)
- cnbs
- ni_tfnbs
- fbc_tfnbs

Notes
-----
- This is an examples script; defaults are chosen for runtime, not publication.
- Uses max-statistic permutation correction across the enhanced score map.
- TP/FN are computed only on ground-truth edges (upper triangle, k=1).
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Running from repo root without installation.
sys.path.append(str(Path(__file__).parent.parent))

import sim_topology_examples as topo
from conninfpy.pairwise_stats import compute_p_val, compute_t_stat


ArrayF = np.ndarray


def _effect_size_tag(effect_size: float) -> str:
    return topo._effect_size_tag(effect_size)


def _slug(text: str) -> str:
    return topo._slug(text)


def _draw_module_boundaries(ax, labels: np.ndarray) -> None:
    topo._draw_module_boundaries(ax, labels)


def _ensure_symmetric_zero_diag(matrix: ArrayF) -> ArrayF:
    return topo._ensure_symmetric_zero_diag(matrix)


def _tp_fn_fp_from_p_map(
    p_map: ArrayF,
    gt_mask: ArrayF,
    alpha: float,
) -> Tuple[int, int, int, int]:
    n_nodes = p_map.shape[0]
    tri = np.triu_indices(n_nodes, k=1)
    mask_ut = gt_mask[tri] != 0
    sig_ut = p_map[tri] < alpha

    tp = int(np.sum(sig_ut & mask_ut))
    fn = int(np.sum((~sig_ut) & mask_ut))
    fp = int(np.sum(sig_ut & (~mask_ut)))
    n_true = int(np.sum(mask_ut))
    return tp, fn, fp, n_true


def _compute_p_map(
    group1_z: ArrayF,
    group2_z: ArrayF,
    *,
    method: str,
    method_kwargs: Dict[str, object],
    n_permutations: int,
    random_state: int,
    use_mp: bool,
    n_processes: Optional[int],
) -> ArrayF:
    p_vals = compute_p_val(
        group1_z,
        group2_z,
        n_permutations=n_permutations,
        test_type="two-sample",
        method=method,
        use_mp=use_mp,
        random_state=random_state,
        n_processes=n_processes,
        **method_kwargs,
    )
    return np.asarray(p_vals["g2>g1"], dtype=np.float64)


def _plot_comparison(
    gt_effect_r: ArrayF,
    t_signed: ArrayF,
    p_maps: Dict[str, ArrayF],
    gt_mask: ArrayF,
    labels: np.ndarray,
    alpha: float,
    title: str,
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("Ground Truth (Δr)", gt_effect_r, "seismic", None),
        ("Signed t-stat", t_signed, "seismic", None),
        ("tstat (1-p)", 1.0 - p_maps["tstat"], "magma", (0.0, 1.0)),
        ("tfnbs (1-p)", 1.0 - p_maps["tfnbs"], "magma", (0.0, 1.0)),
        ("nbs extent (1-p)", 1.0 - p_maps["nbs_extent"], "magma", (0.0, 1.0)),
        ("nbs intensity (1-p)", 1.0 - p_maps["nbs_intensity"], "magma", (0.0, 1.0)),
        ("cnbs (1-p)", 1.0 - p_maps["cnbs"], "magma", (0.0, 1.0)),
        ("ni_tfnbs (1-p)", 1.0 - p_maps["ni_tfnbs"], "magma", (0.0, 1.0)),
        ("fbc_tfnbs (1-p)", 1.0 - p_maps["fbc_tfnbs"], "magma", (0.0, 1.0)),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    fig.suptitle(title, fontsize=14)

    # Common limits.
    gt_lim = float(np.max(np.abs(gt_effect_r))) if np.any(gt_effect_r) else 1.0
    t_lim = float(np.max(np.abs(t_signed))) if np.any(t_signed) else 1.0

    for ax, (name, mat, cmap, fixed_lim) in zip(axes.flat, panels):
        mat = _ensure_symmetric_zero_diag(mat)
        if fixed_lim is None:
            if name.startswith("Ground Truth"):
                vmin, vmax = -gt_lim, gt_lim
            elif name.startswith("Signed t-stat"):
                vmin, vmax = -t_lim, t_lim
            else:
                vmin, vmax = float(np.min(mat)), float(np.max(mat))
        else:
            vmin, vmax = fixed_lim

        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
        _draw_module_boundaries(ax, labels)

        if "(1-p)" in name:
            # TP/FN on ground-truth edges (positive tail).
            method_key = name.split(" (")[0].replace(" ", "_").replace("-", "_")
            method_key = {
                "tstat": "tstat",
                "tfnbs": "tfnbs",
                "nbs_extent": "nbs_extent",
                "nbs_intensity": "nbs_intensity",
                "cnbs": "cnbs",
                "ni_tfnbs": "ni_tfnbs",
                "fbc_tfnbs": "fbc_tfnbs",
            }[method_key]
            tp, fn, fp, n_true = _tp_fn_fp_from_p_map(p_maps[method_key], gt_mask, alpha=alpha)
            ax.set_title(f"{name}\nTP={tp}/{n_true}  FN={fn}  FP={fp}", fontsize=10)
        else:
            ax.set_title(name, fontsize=10)

        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compare multiple NBS-family methods on synthetic data.")
    parser.add_argument("--effect-size", type=float, default=0.25)
    parser.add_argument("--n-nodes", type=int, default=60)
    parser.add_argument("--n-modules", type=int, default=4)
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--time-points", type=int, default=30)
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-mp", action="store_true")
    parser.add_argument("--n-processes", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "output")
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=None,
        help="Scenario names to run. Default: a small representative set.",
    )
    parser.add_argument(
        "--all-scenarios",
        action="store_true",
        help="Run all available topology scenarios (can be slow).",
    )

    # Data generation params.
    parser.add_argument("--noise-level", type=float, default=0.05)
    parser.add_argument("--intra-corr", type=float, default=0.3)
    parser.add_argument("--inter-corr", type=float, default=0.05)
    parser.add_argument("--uniform-corr", type=float, default=0.15)

    # TFNBS-family params.
    parser.add_argument("--e", type=float, default=0.5)
    parser.add_argument("--h", type=float, default=2.0)
    parser.add_argument("--n-thresholds", type=int, default=50)
    parser.add_argument("--start-thres", type=float, default=1.65)
    parser.add_argument("--fbc-min-cluster", type=int, default=3)

    # NBS params.
    parser.add_argument("--nbs-threshold", type=float, default=2.0)

    args = parser.parse_args(argv)

    if args.effect_size <= 0:
        raise SystemExit("--effect-size must be > 0 (positive-effect demo).")
    if args.n_permutations < 1:
        raise SystemExit("--n-permutations must be >= 1.")

    # Matplotlib cache in sandbox.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runtime_cache_dir = args.output_dir / "_runtime_cache"
    mplconfig_dir = runtime_cache_dir / "mplconfig"
    xdg_cache_dir = runtime_cache_dir / "cache"
    mplconfig_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(mplconfig_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir))

    scenarios = topo.get_scenarios()
    if args.all_scenarios:
        pass
    elif args.scenarios is None:
        default_names = {
            "within_module_dense",
            "perfect_matching_within_module",
            "cross_block_connected_chain",
        }
        scenarios = [s for s in scenarios if s.name in default_names]
    else:
        wanted = set(args.scenarios)
        scenarios = [s for s in scenarios if s.name in wanted]
        missing = sorted(wanted - {s.name for s in scenarios})
        if missing:
            raise SystemExit(f"Unknown scenario(s): {missing}")

    ds_gen = topo.TopologyDatasetGenerator(
        n_nodes=args.n_nodes,
        n_modules=args.n_modules,
        intra_corr=args.intra_corr,
        inter_corr=args.inter_corr,
        uniform_corr=args.uniform_corr,
        noise_level=args.noise_level,
        seed=args.seed,
    )

    for idx, scenario in enumerate(scenarios, start=1):
        ds = ds_gen.generate(
            scenario,
            effect_size=args.effect_size,
            n_samples=args.n_samples,
            time_points=args.time_points,
        )
        g1_z, g2_z = ds.fisher_z()

        # Original signed t-stat map for display.
        t_stat = compute_t_stat(g1_z, g2_z, test_type="two-sample")
        t_signed = t_stat["g2>g1"] - t_stat["g1>g2"]

        p_maps: Dict[str, ArrayF] = {}
        p_maps["tstat"] = _compute_p_map(
            g1_z,
            g2_z,
            method="tstat",
            method_kwargs={},
            n_permutations=args.n_permutations,
            random_state=args.seed,
            use_mp=args.use_mp,
            n_processes=args.n_processes,
        )
        p_maps["tfnbs"] = _compute_p_map(
            g1_z,
            g2_z,
            method="tfnbs",
            method_kwargs={"e": args.e, "h": args.h, "n": args.n_thresholds, "start_thres": args.start_thres},
            n_permutations=args.n_permutations,
            random_state=args.seed,
            use_mp=args.use_mp,
            n_processes=args.n_processes,
        )
        p_maps["nbs_extent"] = _compute_p_map(
            g1_z,
            g2_z,
            method="nbs",
            method_kwargs={"threshold": args.nbs_threshold, "nbs_stat": "extent"},
            n_permutations=args.n_permutations,
            random_state=args.seed,
            use_mp=args.use_mp,
            n_processes=args.n_processes,
        )
        p_maps["nbs_intensity"] = _compute_p_map(
            g1_z,
            g2_z,
            method="nbs",
            method_kwargs={"threshold": args.nbs_threshold, "nbs_stat": "intensity"},
            n_permutations=args.n_permutations,
            random_state=args.seed,
            use_mp=args.use_mp,
            n_processes=args.n_processes,
        )
        p_maps["cnbs"] = _compute_p_map(
            g1_z,
            g2_z,
            method="cnbs",
            method_kwargs={"net_labels": ds.net_labels},
            n_permutations=args.n_permutations,
            random_state=args.seed,
            use_mp=args.use_mp,
            n_processes=args.n_processes,
        )
        p_maps["ni_tfnbs"] = _compute_p_map(
            g1_z,
            g2_z,
            method="ni_tfnbs",
            method_kwargs={
                "net_labels": ds.net_labels,
                "e": args.e,
                "h": args.h,
                "n": args.n_thresholds,
                "start_thres": args.start_thres,
            },
            n_permutations=args.n_permutations,
            random_state=args.seed,
            use_mp=args.use_mp,
            n_processes=args.n_processes,
        )
        p_maps["fbc_tfnbs"] = _compute_p_map(
            g1_z,
            g2_z,
            method="fbc_tfnbs",
            method_kwargs={
                "net_labels": ds.net_labels,
                "e": args.e,
                "h": args.h,
                "n": args.n_thresholds,
                "start_thres": args.start_thres,
                "min_cluster_size": args.fbc_min_cluster,
            },
            n_permutations=args.n_permutations,
            random_state=args.seed,
            use_mp=args.use_mp,
            n_processes=args.n_processes,
        )

        gt_effect_r = ds.effect_mask * ds.effect_size
        eff_tag = _effect_size_tag(args.effect_size)
        out_file = args.output_dir / (
            f"method_compare_{idx:02d}_{_slug(scenario.name)}_pos_{eff_tag}_T{args.time_points}_perm{args.n_permutations}.png"
        )
        fig_title = (
            f"{scenario.name} | base={scenario.base_kind} | effect_size={args.effect_size:g} | "
            f"n={args.n_samples} | T={args.time_points} | perms={args.n_permutations}"
        )

        _plot_comparison(
            gt_effect_r=gt_effect_r,
            t_signed=t_signed,
            p_maps=p_maps,
            gt_mask=ds.effect_mask,
            labels=ds.net_labels,
            alpha=args.alpha,
            title=fig_title,
            out_path=out_file,
        )
        print(f"[OK] {out_file}")

        # Save results dictionary to pickle file (includes metadata for aggregation).
        pkl_file = out_file.with_suffix(".pkl")
        results_data = {
            "p_maps": p_maps,
            "gt_mask": ds.effect_mask,
            "gt_effect_r": gt_effect_r,
            "t_signed": t_signed,
            "net_labels": ds.net_labels,
            "scenario_name": scenario.name,
            "base_kind": scenario.base_kind,
            "params": {
                "effect_size": args.effect_size,
                "n_nodes": args.n_nodes,
                "n_modules": args.n_modules,
                "n_samples": args.n_samples,
                "time_points": args.time_points,
                "n_permutations": args.n_permutations,
                "alpha": args.alpha,
                "seed": args.seed,
                "e": args.e,
                "h": args.h,
                "n_thresholds": args.n_thresholds,
                "start_thres": args.start_thres,
                "nbs_threshold": args.nbs_threshold,
                "fbc_min_cluster": args.fbc_min_cluster,
            },
        }
        with open(pkl_file, "wb") as f:
            pickle.dump(results_data, f)
        print(f"[OK] {pkl_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
