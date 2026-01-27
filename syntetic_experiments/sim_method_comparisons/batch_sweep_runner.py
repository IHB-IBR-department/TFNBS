"""
Batch parameter sweep runner for sim_method_comparisons.py.

This script automates running multiple parameter combinations for systematic
method comparison experiments. Supports sweeping over:
- effect_size
- time_points
- n_samples
- n_permutations
- TFNBS parameters (e, h, n_thresholds, start_thres)
- NBS threshold
- FBC min_cluster_size

Usage examples:

1) Effect-size sweep on default scenarios:
   python batch_sweep_runner.py --effect-size 0.10 0.15 0.20 0.25 0.30

2) Full grid sweep (effect_size x time_points):
   python batch_sweep_runner.py \
     --effect-size 0.15 0.25 \
     --time-points 30 50 80 \
     --scenarios within_module_dense chain

3) NBS threshold sweep:
   python batch_sweep_runner.py \
     --nbs-threshold 1.5 2.0 2.5 3.0 \
     --scenarios within_module_dense

4) Load from YAML config:
   python batch_sweep_runner.py --config sweep_config.yaml

5) BATCH MODE - Faster TFNBS parameter sweep:
   python batch_sweep_runner.py \
     --e 0.3 0.4 0.5 \
     --h 2.0 2.5 3.0 \
     --batch-tfnbs-params \
     --scenarios within_module_dense

   IMPORTANT: e and h must have EQUAL LENGTH - they are paired:
   - (e=0.3, h=2.0), (e=0.4, h=2.5), (e=0.5, h=3.0)

   This mode:
   - Passes e/h as paired lists to compute_p_val (computed in single pass)
   - Much faster than running separate processes for each (e, h) pair
   - Produces 3D p_maps: shape (N, N, num_params) - cannot be plotted
   - Output pickle contains 'p_maps_3d' dict with e_values/h_values metadata

Output is organized into subdirectories by sweep parameters.
"""

from __future__ import annotations

import argparse
import itertools
import pickle
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Optional YAML support.
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# Add parent to path for imports when running from syntetic_experiments/.
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class SweepConfig:
    """Configuration for a parameter sweep."""

    # Sweep parameters (lists to iterate over).
    effect_size: List[float] = field(default_factory=lambda: [0.25])
    time_points: List[int] = field(default_factory=lambda: [50])
    n_samples: List[int] = field(default_factory=lambda: [20])
    n_permutations: List[int] = field(default_factory=lambda: [200])
    scenarios: List[str] = field(default_factory=list)  # Empty = use script default.

    # TFNBS parameters (lists for sweep, or single value).
    e: List[float] = field(default_factory=lambda: [0.5])
    h: List[float] = field(default_factory=lambda: [2.0])
    n_thresholds: List[int] = field(default_factory=lambda: [50])
    start_thres: List[float] = field(default_factory=lambda: [1.65])

    # NBS parameters.
    nbs_threshold: List[float] = field(default_factory=lambda: [2.0])

    # FBC parameters.
    fbc_min_cluster: List[int] = field(default_factory=lambda: [3])

    # Batch mode: pass TFNBS param lists to single run (faster, 3D output).
    batch_tfnbs_params: bool = False

    # Fixed parameters (not swept).
    n_nodes: int = 60
    n_modules: int = 4
    alpha: float = 0.05
    seed: int = 42
    noise_level: float = 0.05
    intra_corr: float = 0.3
    inter_corr: float = 0.05
    uniform_corr: float = 0.15

    # Execution options.
    use_mp: bool = False
    n_processes: Optional[int] = None
    output_dir: Path = field(default_factory=lambda: Path("output/sweeps"))
    dry_run: bool = False
    all_scenarios: bool = False

    @classmethod
    def from_yaml(cls, path: Path) -> "SweepConfig":
        """Load configuration from YAML file."""
        if not HAS_YAML:
            raise ImportError("PyYAML is required for --config. Install with: pip install pyyaml")
        with open(path) as f:
            data = yaml.safe_load(f)

        # Convert single values to lists for sweep parameters.
        sweep_keys = [
            "effect_size", "time_points", "n_samples", "n_permutations",
            "e", "h", "n_thresholds", "start_thres", "nbs_threshold", "fbc_min_cluster",
        ]
        for key in sweep_keys:
            if key in data and not isinstance(data[key], list):
                data[key] = [data[key]]

        if "output_dir" in data:
            data["output_dir"] = Path(data["output_dir"])

        return cls(**data)


def generate_sweep_combinations(config: SweepConfig) -> List[Dict[str, Any]]:
    """Generate all parameter combinations from sweep config.

    In batch_tfnbs_params mode, TFNBS parameters (e, h) are kept as lists
    and passed together to a single run, rather than creating separate
    combinations. This is faster but produces 3D p_maps.
    """
    if config.batch_tfnbs_params:
        # Batch mode: keep TFNBS params (e, h) as paired lists, only iterate over data params.
        # NOTE: e and h must have equal length - they are paired, not a grid!
        if len(config.e) != len(config.h):
            raise ValueError(
                f"In batch mode, 'e' and 'h' must have equal length (paired parameters). "
                f"Got e={config.e} (len={len(config.e)}), h={config.h} (len={len(config.h)})"
            )

        data_params = {
            "effect_size": config.effect_size,
            "time_points": config.time_points,
            "n_samples": config.n_samples,
            "n_permutations": config.n_permutations,
            "nbs_threshold": config.nbs_threshold,
            "fbc_min_cluster": config.fbc_min_cluster,
        }
        # TFNBS params passed as paired lists (e[i], h[i] are used together).
        tfnbs_params = {
            "e": config.e,  # Paired list with h
            "h": config.h,  # Paired list with e
            "n_thresholds": config.n_thresholds[0] if len(config.n_thresholds) == 1 else config.n_thresholds,
            "start_thres": config.start_thres[0] if len(config.start_thres) == 1 else config.start_thres,
        }

        keys = list(data_params.keys())
        values = [data_params[k] for k in keys]

        combinations = []
        for combo in itertools.product(*values):
            combo_dict = dict(zip(keys, combo))
            combo_dict.update(tfnbs_params)  # Add TFNBS params as paired lists
            combo_dict["_batch_tfnbs"] = True  # Mark as batch mode
            combinations.append(combo_dict)

        return combinations
    else:
        # Standard mode: full cartesian product of all params.
        sweep_params = {
            "effect_size": config.effect_size,
            "time_points": config.time_points,
            "n_samples": config.n_samples,
            "n_permutations": config.n_permutations,
            "e": config.e,
            "h": config.h,
            "n_thresholds": config.n_thresholds,
            "start_thres": config.start_thres,
            "nbs_threshold": config.nbs_threshold,
            "fbc_min_cluster": config.fbc_min_cluster,
        }

        keys = list(sweep_params.keys())
        values = [sweep_params[k] for k in keys]

        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))

        return combinations


def build_command(
    combo: Dict[str, Any],
    config: SweepConfig,
    output_subdir: Path,
) -> List[str]:
    """Build command-line arguments for sim_method_comparisons.py."""
    script_path = Path(__file__).parent / "sim_method_comparisons.py"

    cmd = [sys.executable, str(script_path)]

    # Sweep parameters.
    cmd.extend(["--effect-size", str(combo["effect_size"])])
    cmd.extend(["--time-points", str(combo["time_points"])])
    cmd.extend(["--n-samples", str(combo["n_samples"])])
    cmd.extend(["--n-permutations", str(combo["n_permutations"])])
    cmd.extend(["--e", str(combo["e"])])
    cmd.extend(["--h", str(combo["h"])])
    cmd.extend(["--n-thresholds", str(combo["n_thresholds"])])
    cmd.extend(["--start-thres", str(combo["start_thres"])])
    cmd.extend(["--nbs-threshold", str(combo["nbs_threshold"])])
    cmd.extend(["--fbc-min-cluster", str(combo["fbc_min_cluster"])])

    # Fixed parameters.
    cmd.extend(["--n-nodes", str(config.n_nodes)])
    cmd.extend(["--n-modules", str(config.n_modules)])
    cmd.extend(["--alpha", str(config.alpha)])
    cmd.extend(["--seed", str(config.seed)])
    cmd.extend(["--noise-level", str(config.noise_level)])
    cmd.extend(["--intra-corr", str(config.intra_corr)])
    cmd.extend(["--inter-corr", str(config.inter_corr)])
    cmd.extend(["--uniform-corr", str(config.uniform_corr)])

    # Scenarios.
    if config.all_scenarios:
        cmd.append("--all-scenarios")
    elif config.scenarios:
        cmd.extend(["--scenarios"] + config.scenarios)

    # Execution options.
    if config.use_mp:
        cmd.append("--use-mp")
    if config.n_processes is not None:
        cmd.extend(["--n-processes", str(config.n_processes)])

    cmd.extend(["--output-dir", str(output_subdir)])

    return cmd


def combo_to_dirname(combo: Dict[str, Any], sweep_keys: List[str]) -> str:
    """Generate directory name from parameter combination."""
    parts = []
    key_abbrev = {
        "effect_size": "es",
        "time_points": "T",
        "n_samples": "n",
        "n_permutations": "perm",
        "e": "e",
        "h": "h",
        "n_thresholds": "nth",
        "start_thres": "st",
        "nbs_threshold": "nbst",
        "fbc_min_cluster": "fbc",
    }
    for key in sweep_keys:
        val = combo[key]
        abbr = key_abbrev.get(key, key[:3])
        if isinstance(val, float):
            val_str = f"{val:.2f}".replace(".", "p")
        else:
            val_str = str(val)
        parts.append(f"{abbr}{val_str}")
    return "_".join(parts)


def run_batch_tfnbs_direct(
    combo: Dict[str, Any],
    config: SweepConfig,
    output_dir: Path,
) -> Tuple[bool, str, float]:
    """Run a single combination with batched TFNBS params using direct Python calls.

    This is faster than subprocess for TFNBS parameter sweeps because:
    1. No process spawn overhead
    2. TFNBS params (e, h) are passed as lists, computed in single pass
    3. Produces 3D p_maps (n_e, n_h, n_nodes, n_nodes) for stats aggregation

    Returns:
        (success, message, elapsed_time)
    """
    start_time = time.time()

    try:
        # Late imports to avoid circular dependencies.
        import sim_topology_examples as topo
        from tfnbs.pairwise_stats import compute_p_val, compute_t_stat

        # Get scenarios.
        scenarios = topo.get_scenarios()
        if config.all_scenarios:
            pass
        elif config.scenarios:
            wanted = set(config.scenarios)
            scenarios = [s for s in scenarios if s.name in wanted]
        else:
            default_names = {
                "within_module_dense",
                "perfect_matching_within_module",
                "cross_block_connected_chain",
            }
            scenarios = [s for s in scenarios if s.name in default_names]

        # Setup data generator.
        ds_gen = topo.TopologyDatasetGenerator(
            n_nodes=config.n_nodes,
            n_modules=config.n_modules,
            intra_corr=config.intra_corr,
            inter_corr=config.inter_corr,
            uniform_corr=config.uniform_corr,
            noise_level=config.noise_level,
            seed=config.seed,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        n_files = 0

        for idx, scenario in enumerate(scenarios, start=1):
            ds = ds_gen.generate(
                scenario,
                effect_size=combo["effect_size"],
                n_samples=combo["n_samples"],
                time_points=combo["time_points"],
            )
            g1_z, g2_z = ds.fisher_z()

            # Compute t-stat for reference.
            t_stat = compute_t_stat(g1_z, g2_z, test_type="two-sample")
            t_signed = t_stat["g2>g1"] - t_stat["g1>g2"]

            # Batched TFNBS computation with e/h as lists.
            e_list = combo["e"] if isinstance(combo["e"], list) else [combo["e"]]
            h_list = combo["h"] if isinstance(combo["h"], list) else [combo["h"]]

            # p_maps will be 3D: (n_e * n_h, n_nodes, n_nodes) or indexed by (e, h).
            p_maps_3d: Dict[str, Any] = {}

            # Compute TFNBS-family methods with batched params.
            for method in ["tfnbs", "ni_tfnbs", "fbc_tfnbs"]:
                method_kwargs: Dict[str, Any] = {
                    "e": e_list,
                    "h": h_list,
                    "n": combo.get("n_thresholds", 50),
                    "start_thres": combo.get("start_thres", 1.65),
                }
                if method in ["ni_tfnbs", "fbc_tfnbs"]:
                    method_kwargs["net_labels"] = ds.net_labels
                if method == "fbc_tfnbs":
                    method_kwargs["min_cluster_size"] = combo.get("fbc_min_cluster", 3)

                p_vals = compute_p_val(
                    g1_z, g2_z,
                    n_permutations=combo["n_permutations"],
                    test_type="two-sample",
                    method=method,
                    use_mp=config.use_mp,
                    random_state=config.seed,
                    n_processes=config.n_processes,
                    **method_kwargs,
                )
                p_maps_3d[method] = {
                    "p_vals": p_vals,
                    "e_values": e_list,
                    "h_values": h_list,
                }

            # Also compute non-TFNBS methods (2D output).
            p_maps_2d: Dict[str, np.ndarray] = {}
            for method, kwargs in [
                ("tstat", {}),
                ("nbs", {"threshold": combo["nbs_threshold"], "nbs_stat": "extent"}),
                ("cnbs", {"net_labels": ds.net_labels}),
            ]:
                p_vals = compute_p_val(
                    g1_z, g2_z,
                    n_permutations=combo["n_permutations"],
                    test_type="two-sample",
                    method=method,
                    use_mp=config.use_mp,
                    random_state=config.seed,
                    n_processes=config.n_processes,
                    **kwargs,
                )
                method_key = method if method != "nbs" else "nbs_extent"
                p_maps_2d[method_key] = np.asarray(p_vals["g2>g1"], dtype=np.float64)

            # Save results.
            eff_tag = f"es{combo['effect_size']:.2f}".replace(".", "p")
            n_params = len(e_list)
            pkl_file = output_dir / (
                f"batch_{idx:02d}_{scenario.name}_{eff_tag}_"
                f"params{n_params}.pkl"
            )
            results_data = {
                "p_maps_3d": p_maps_3d,  # TFNBS methods with batched params
                "p_maps_2d": p_maps_2d,  # Other methods (2D)
                "gt_mask": ds.effect_mask,
                "gt_effect_r": ds.effect_mask * combo["effect_size"],
                "t_signed": t_signed,
                "net_labels": ds.net_labels,
                "scenario_name": scenario.name,
                "base_kind": scenario.base_kind,
                "params": {
                    "effect_size": combo["effect_size"],
                    "n_samples": combo["n_samples"],
                    "time_points": combo["time_points"],
                    "n_permutations": combo["n_permutations"],
                    # Paired TFNBS parameters: (e_values[i], h_values[i]) are used together.
                    "e_values": e_list,
                    "h_values": h_list,
                    "param_pairs": list(zip(e_list, h_list)),  # [(e0,h0), (e1,h1), ...]
                    "n_thresholds": combo.get("n_thresholds", 50),
                    "start_thres": combo.get("start_thres", 1.65),
                    "nbs_threshold": combo["nbs_threshold"],
                    "fbc_min_cluster": combo.get("fbc_min_cluster", 3),
                    "alpha": config.alpha,
                    "seed": config.seed,
                },
                "_batch_mode": True,
            }
            with open(pkl_file, "wb") as f:
                pickle.dump(results_data, f)
            n_files += 1

        elapsed = time.time() - start_time
        return True, f"Created {n_files} result file(s)", elapsed

    except Exception as e:
        elapsed = time.time() - start_time
        return False, str(e), elapsed


def run_sweep(config: SweepConfig) -> int:
    """Run all parameter combinations."""
    combinations = generate_sweep_combinations(config)

    # Identify which parameters are actually being swept (have multiple values).
    sweep_params = {
        "effect_size": config.effect_size,
        "time_points": config.time_points,
        "n_samples": config.n_samples,
        "n_permutations": config.n_permutations,
        "e": config.e,
        "h": config.h,
        "n_thresholds": config.n_thresholds,
        "start_thres": config.start_thres,
        "nbs_threshold": config.nbs_threshold,
        "fbc_min_cluster": config.fbc_min_cluster,
    }
    active_sweep_keys = [k for k, v in sweep_params.items() if len(v) > 1]

    # In batch mode, e/h are not individual sweep keys.
    if config.batch_tfnbs_params:
        active_sweep_keys = [k for k in active_sweep_keys if k not in ("e", "h")]

    print(f"=== Batch Sweep Runner ===")
    print(f"Mode: {'BATCH TFNBS (direct call, 3D output)' if config.batch_tfnbs_params else 'subprocess'}")
    print(f"Total combinations: {len(combinations)}")
    if active_sweep_keys:
        print(f"Sweeping over: {', '.join(active_sweep_keys)}")
    if config.batch_tfnbs_params and len(config.e) > 1:
        pairs = list(zip(config.e, config.h))
        print(f"Batched TFNBS param pairs (e, h): {pairs}")
    print(f"Output directory: {config.output_dir}")
    print()

    config.output_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    timings = []

    for i, combo in enumerate(combinations, start=1):
        # Create subdirectory for this combination.
        if active_sweep_keys:
            subdir_name = combo_to_dirname(combo, active_sweep_keys)
        else:
            subdir_name = "default"
        output_subdir = config.output_dir / subdir_name

        print(f"[{i}/{len(combinations)}] Running: {subdir_name}")

        # Use batch mode (direct call) or subprocess mode.
        if config.batch_tfnbs_params:
            if config.dry_run:
                print(f"  [BATCH] e={combo.get('e')}, h={combo.get('h')}")
                continue

            success, msg, elapsed = run_batch_tfnbs_direct(combo, config, output_subdir)
            timings.append(elapsed)

            if success:
                print(f"  OK ({elapsed:.1f}s) - {msg}")
            else:
                print(f"  FAILED: {msg}")
                failed.append((subdir_name, -1, msg))
        else:
            # Subprocess mode.
            cmd = build_command(combo, config, output_subdir)

            if config.dry_run:
                print(f"  CMD: {' '.join(cmd)}")
                continue

            start_time = time.time()
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=Path(__file__).parent,
                )
                elapsed = time.time() - start_time
                timings.append(elapsed)

                if result.returncode != 0:
                    print(f"  FAILED (exit code {result.returncode})")
                    print(f"  stderr: {result.stderr[:500]}")
                    failed.append((subdir_name, result.returncode, result.stderr))
                else:
                    print(f"  OK ({elapsed:.1f}s)")
                    # Show output files created.
                    if output_subdir.exists():
                        pkl_files = list(output_subdir.glob("*.pkl"))
                        print(f"  Created {len(pkl_files)} result file(s)")

            except Exception as e:
                elapsed = time.time() - start_time
                print(f"  EXCEPTION: {e}")
                failed.append((subdir_name, -1, str(e)))

    # Summary.
    print()
    print("=== Summary ===")
    print(f"Completed: {len(combinations) - len(failed)}/{len(combinations)}")
    if timings:
        print(f"Total time: {sum(timings):.1f}s")
        print(f"Avg per run: {sum(timings)/len(timings):.1f}s")
    if failed:
        print(f"Failed runs:")
        for name, code, _err in failed:
            print(f"  - {name}: exit {code}")

    return 1 if failed else 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Batch parameter sweep runner for method comparisons.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Config file option.
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Load sweep configuration from YAML file.",
    )

    # Sweep parameters (lists).
    parser.add_argument("--effect-size", type=float, nargs="+", default=None)
    parser.add_argument("--time-points", type=int, nargs="+", default=None)
    parser.add_argument("--n-samples", type=int, nargs="+", default=None)
    parser.add_argument("--n-permutations", type=int, nargs="+", default=None)
    parser.add_argument("--e", type=float, nargs="+", default=None)
    parser.add_argument("--h", type=float, nargs="+", default=None)
    parser.add_argument("--n-thresholds", type=int, nargs="+", default=None)
    parser.add_argument("--start-thres", type=float, nargs="+", default=None)
    parser.add_argument("--nbs-threshold", type=float, nargs="+", default=None)
    parser.add_argument("--fbc-min-cluster", type=int, nargs="+", default=None)

    # Scenarios.
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--all-scenarios", action="store_true")

    # Fixed parameters.
    parser.add_argument("--n-nodes", type=int, default=60)
    parser.add_argument("--n-modules", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-level", type=float, default=0.05)
    parser.add_argument("--intra-corr", type=float, default=0.3)
    parser.add_argument("--inter-corr", type=float, default=0.05)
    parser.add_argument("--uniform-corr", type=float, default=0.15)

    # Execution options.
    parser.add_argument("--use-mp", action="store_true")
    parser.add_argument("--n-processes", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "output" / "sweeps")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument(
        "--batch-tfnbs-params", action="store_true",
        help="Batch TFNBS params (e, h) in single runs. Faster but produces 3D p_maps (no plots).",
    )

    args = parser.parse_args(argv)

    # Load config from YAML or build from CLI args.
    if args.config is not None:
        config = SweepConfig.from_yaml(args.config)
        # CLI args override YAML.
        if args.effect_size is not None:
            config.effect_size = args.effect_size
        if args.time_points is not None:
            config.time_points = args.time_points
        if args.n_samples is not None:
            config.n_samples = args.n_samples
        if args.n_permutations is not None:
            config.n_permutations = args.n_permutations
        if args.e is not None:
            config.e = args.e
        if args.h is not None:
            config.h = args.h
        if args.n_thresholds is not None:
            config.n_thresholds = args.n_thresholds
        if args.start_thres is not None:
            config.start_thres = args.start_thres
        if args.nbs_threshold is not None:
            config.nbs_threshold = args.nbs_threshold
        if args.fbc_min_cluster is not None:
            config.fbc_min_cluster = args.fbc_min_cluster
    else:
        config = SweepConfig(
            effect_size=args.effect_size or [0.25],
            time_points=args.time_points or [50],
            n_samples=args.n_samples or [20],
            n_permutations=args.n_permutations or [200],
            e=args.e or [0.5],
            h=args.h or [2.0],
            n_thresholds=args.n_thresholds or [50],
            start_thres=args.start_thres or [1.65],
            nbs_threshold=args.nbs_threshold or [2.0],
            fbc_min_cluster=args.fbc_min_cluster or [3],
            scenarios=args.scenarios or [],
            all_scenarios=args.all_scenarios,
            n_nodes=args.n_nodes,
            n_modules=args.n_modules,
            alpha=args.alpha,
            seed=args.seed,
            noise_level=args.noise_level,
            intra_corr=args.intra_corr,
            inter_corr=args.inter_corr,
            uniform_corr=args.uniform_corr,
            use_mp=args.use_mp,
            n_processes=args.n_processes,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            batch_tfnbs_params=args.batch_tfnbs_params,
        )

    # Apply remaining CLI overrides (only if explicitly provided, not defaults).
    if args.scenarios is not None:
        config.scenarios = args.scenarios
    if args.all_scenarios:
        config.all_scenarios = True
    # Only override output_dir if not using config file or if explicitly provided.
    default_output_dir = Path(__file__).parent / "output" / "sweeps"
    if args.config is None or args.output_dir != default_output_dir:
        config.output_dir = args.output_dir
    if args.dry_run:
        config.dry_run = True
    if args.use_mp:
        config.use_mp = True
    if args.n_processes is not None:
        config.n_processes = args.n_processes
    if args.batch_tfnbs_params:
        config.batch_tfnbs_params = True

    return run_sweep(config)


if __name__ == "__main__":
    raise SystemExit(main())
