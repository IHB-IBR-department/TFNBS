"""
Result aggregator for method comparison experiments.

This script collects .pkl result files from sim_method_comparisons.py runs,
computes TP/FN/FP/TN statistics for each method, and produces:
- CSV summary table
- Optional markdown report
- Optional power curve plots

Usage examples:

1) Aggregate results from a directory:
   python aggregate_results.py --input-dir output/sweeps

2) Aggregate with plots:
   python aggregate_results.py --input-dir output/sweeps --plot

3) Filter by scenario:
   python aggregate_results.py --input-dir output/sweeps --scenarios within_module_dense

4) Custom alpha threshold:
   python aggregate_results.py --input-dir output/sweeps --alpha 0.01

Output files:
- results_summary.csv: Full table of metrics per (scenario, method, params)
- results_summary.md: Markdown-formatted report (if --markdown)
- power_curves_*.png: Power curve plots (if --plot)
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Methods expected in p_maps.
METHODS = ["tstat", "tfnbs", "nbs_extent", "nbs_intensity", "cnbs", "ni_tfnbs", "fbc_tfnbs"]


@dataclass
class ResultMetrics:
    """Metrics computed from a single result file."""

    file_path: Path
    scenario_name: str
    base_kind: str
    method: str
    effect_size: float
    time_points: int
    n_samples: int
    n_permutations: int
    alpha: float

    # Detection metrics (on upper triangle).
    tp: int  # True positives.
    fn: int  # False negatives.
    fp: int  # False positives.
    tn: int  # True negatives.
    n_true: int  # Total ground-truth edges.
    n_null: int  # Total null edges.

    # Derived metrics.
    tpr: float  # True positive rate (sensitivity/recall).
    fpr: float  # False positive rate.
    precision: float  # Precision (PPV).
    f1: float  # F1 score.

    # Additional params.
    e: float = 0.5
    h: float = 2.0
    n_thresholds: int = 50
    start_thres: float = 1.65
    nbs_threshold: float = 2.0
    fbc_min_cluster: int = 3


def compute_metrics(
    p_map: np.ndarray,
    gt_mask: np.ndarray,
    alpha: float,
) -> Tuple[int, int, int, int, int, int]:
    """Compute TP, FN, FP, TN from p-value map and ground truth mask."""
    n_nodes = p_map.shape[0]
    tri = np.triu_indices(n_nodes, k=1)

    mask_ut = gt_mask[tri] != 0  # Ground truth edges.
    sig_ut = p_map[tri] < alpha  # Significant edges.

    tp = int(np.sum(sig_ut & mask_ut))
    fn = int(np.sum((~sig_ut) & mask_ut))
    fp = int(np.sum(sig_ut & (~mask_ut)))
    tn = int(np.sum((~sig_ut) & (~mask_ut)))

    n_true = int(np.sum(mask_ut))
    n_null = int(np.sum(~mask_ut))

    return tp, fn, fp, tn, n_true, n_null


def _create_result_metrics(
    path: Path,
    scenario_name: str,
    base_kind: str,
    method: str,
    p_map: np.ndarray,
    gt_mask: np.ndarray,
    params: Dict[str, Any],
    alpha: float,
    e_val: float,
    h_val: float,
) -> ResultMetrics:
    """Helper to create ResultMetrics from a 2D p_map."""
    tp, fn, fp, tn, n_true, n_null = compute_metrics(p_map, gt_mask, alpha)

    # Compute derived metrics.
    tpr = tp / n_true if n_true > 0 else 0.0
    fpr = fp / n_null if n_null > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    return ResultMetrics(
        file_path=path,
        scenario_name=scenario_name,
        base_kind=base_kind,
        method=method,
        effect_size=params.get("effect_size", 0.0),
        time_points=params.get("time_points", 0),
        n_samples=params.get("n_samples", 0),
        n_permutations=params.get("n_permutations", 0),
        alpha=alpha,
        tp=tp,
        fn=fn,
        fp=fp,
        tn=tn,
        n_true=n_true,
        n_null=n_null,
        tpr=tpr,
        fpr=fpr,
        precision=precision,
        f1=f1,
        e=e_val,
        h=h_val,
        n_thresholds=params.get("n_thresholds", 50),
        start_thres=params.get("start_thres", 1.65),
        nbs_threshold=params.get("nbs_threshold", 2.0),
        fbc_min_cluster=params.get("fbc_min_cluster", 3),
    )


def load_result_file(path: Path, alpha: float) -> List[ResultMetrics]:
    """Load a .pkl result file and compute metrics for all methods.

    Handles both standard format (2D p_maps) and batch mode format (3D p_maps
    for TFNBS methods with multiple (e, h) parameter pairs).
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    # Check for batch mode (from batch_sweep_runner.py with --batch-tfnbs-params).
    is_batch_mode = data.get("_batch_mode", False)

    if is_batch_mode:
        return _load_batch_result_file(path, data, alpha)

    # Standard format handling.
    if "p_maps" in data:
        p_maps = data["p_maps"]
        gt_mask = data.get("gt_mask")
        scenario_name = data.get("scenario_name", "unknown")
        base_kind = data.get("base_kind", "unknown")
        params = data.get("params", {})
    else:
        # Old format: data is just p_maps dict.
        p_maps = data
        gt_mask = None
        scenario_name = "unknown"
        base_kind = "unknown"
        params = {}

    if gt_mask is None:
        # Cannot compute metrics without ground truth.
        return []

    results = []
    for method in METHODS:
        if method not in p_maps:
            continue

        p_map = np.asarray(p_maps[method], dtype=np.float64)
        result = _create_result_metrics(
            path=path,
            scenario_name=scenario_name,
            base_kind=base_kind,
            method=method,
            p_map=p_map,
            gt_mask=gt_mask,
            params=params,
            alpha=alpha,
            e_val=params.get("e", 0.5),
            h_val=params.get("h", 2.0),
        )
        results.append(result)

    return results


def _load_batch_result_file(path: Path, data: Dict[str, Any], alpha: float) -> List[ResultMetrics]:
    """Load batch mode result file with 3D p_maps for TFNBS methods.

    Batch mode output structure:
    - p_maps_3d: {method: {p_vals: {g2>g1: 3D, g1>g2: 3D}, e_values, h_values}}
    - p_maps_2d: {method: 2D array}
    - params.param_pairs: [(e0, h0), (e1, h1), ...]
    """
    gt_mask = data.get("gt_mask")
    if gt_mask is None:
        return []

    scenario_name = data.get("scenario_name", "unknown")
    base_kind = data.get("base_kind", "unknown")
    params = data.get("params", {})

    # Get parameter pairs.
    param_pairs = params.get("param_pairs", [])
    e_values = params.get("e_values", [])
    h_values = params.get("h_values", [])

    if not param_pairs and e_values and h_values:
        param_pairs = list(zip(e_values, h_values))

    results = []

    # Process 2D methods (tstat, nbs_extent, cnbs, etc.).
    p_maps_2d = data.get("p_maps_2d", {})
    for method, p_map in p_maps_2d.items():
        if method not in METHODS:
            continue
        p_map = np.asarray(p_map, dtype=np.float64)
        # 2D methods don't have e/h params, use defaults.
        result = _create_result_metrics(
            path=path,
            scenario_name=scenario_name,
            base_kind=base_kind,
            method=method,
            p_map=p_map,
            gt_mask=gt_mask,
            params=params,
            alpha=alpha,
            e_val=0.0,  # Not applicable for non-TFNBS methods.
            h_val=0.0,
        )
        results.append(result)

    # Process 3D methods (tfnbs, ni_tfnbs, fbc_tfnbs).
    p_maps_3d = data.get("p_maps_3d", {})
    for method, method_data in p_maps_3d.items():
        if method not in METHODS:
            continue

        p_vals = method_data.get("p_vals", {})
        # Get the 3D array for g2>g1 direction.
        p_map_3d = p_vals.get("g2>g1")
        if p_map_3d is None:
            continue

        p_map_3d = np.asarray(p_map_3d, dtype=np.float64)

        # Iterate over each (e, h) parameter pair.
        n_params = p_map_3d.shape[2] if p_map_3d.ndim == 3 else 1

        for i in range(n_params):
            # Extract 2D slice for this parameter pair.
            if p_map_3d.ndim == 3:
                p_map_2d = p_map_3d[:, :, i]
            else:
                p_map_2d = p_map_3d

            # Get (e, h) for this index.
            if i < len(param_pairs):
                e_val, h_val = param_pairs[i]
            else:
                e_val = e_values[i] if i < len(e_values) else 0.5
                h_val = h_values[i] if i < len(h_values) else 2.0

            result = _create_result_metrics(
                path=path,
                scenario_name=scenario_name,
                base_kind=base_kind,
                method=method,
                p_map=p_map_2d,
                gt_mask=gt_mask,
                params=params,
                alpha=alpha,
                e_val=e_val,
                h_val=h_val,
            )
            results.append(result)

    return results


def collect_results(
    input_dir: Path,
    alpha: float,
    scenarios: Optional[List[str]] = None,
    recursive: bool = True,
) -> List[ResultMetrics]:
    """Collect all results from .pkl files in directory."""
    pattern = "**/*.pkl" if recursive else "*.pkl"
    pkl_files = sorted(input_dir.glob(pattern))

    print(f"Found {len(pkl_files)} .pkl files in {input_dir}")

    all_results = []
    for path in pkl_files:
        # Skip runtime cache files.
        if "_runtime_cache" in str(path):
            continue

        try:
            results = load_result_file(path, alpha)
            if scenarios:
                results = [r for r in results if r.scenario_name in scenarios]
            all_results.extend(results)
        except Exception as e:
            print(f"  Warning: Failed to load {path}: {e}")

    print(f"Loaded {len(all_results)} method results")
    return all_results


def results_to_csv(results: List[ResultMetrics], output_path: Path) -> None:
    """Write results to CSV file."""
    headers = [
        "scenario", "base_kind", "method",
        "effect_size", "time_points", "n_samples", "n_permutations",
        "alpha", "TP", "FN", "FP", "TN", "n_true", "n_null",
        "TPR", "FPR", "precision", "F1",
        "e", "h", "n_thresholds", "start_thres", "nbs_threshold", "fbc_min_cluster",
        "file",
    ]

    with open(output_path, "w") as f:
        f.write(",".join(headers) + "\n")
        for r in results:
            row = [
                r.scenario_name,
                r.base_kind,
                r.method,
                f"{r.effect_size:.3f}",
                str(r.time_points),
                str(r.n_samples),
                str(r.n_permutations),
                f"{r.alpha:.3f}",
                str(r.tp),
                str(r.fn),
                str(r.fp),
                str(r.tn),
                str(r.n_true),
                str(r.n_null),
                f"{r.tpr:.4f}",
                f"{r.fpr:.4f}",
                f"{r.precision:.4f}",
                f"{r.f1:.4f}",
                f"{r.e:.2f}",
                f"{r.h:.2f}",
                str(r.n_thresholds),
                f"{r.start_thres:.2f}",
                f"{r.nbs_threshold:.2f}",
                str(r.fbc_min_cluster),
                str(r.file_path.name),
            ]
            f.write(",".join(row) + "\n")

    print(f"Wrote CSV: {output_path}")


def results_to_markdown(results: List[ResultMetrics], output_path: Path) -> None:
    """Write results to markdown report."""
    # Group by scenario.
    by_scenario: Dict[str, List[ResultMetrics]] = {}
    for r in results:
        by_scenario.setdefault(r.scenario_name, []).append(r)

    lines = ["# Method Comparison Results\n"]

    for scenario_name in sorted(by_scenario.keys()):
        scenario_results = by_scenario[scenario_name]
        lines.append(f"\n## Scenario: {scenario_name}\n")

        # Group by effect_size within scenario.
        by_effect: Dict[float, List[ResultMetrics]] = {}
        for r in scenario_results:
            by_effect.setdefault(r.effect_size, []).append(r)

        for effect_size in sorted(by_effect.keys()):
            effect_results = by_effect[effect_size]
            lines.append(f"\n### Effect Size: {effect_size:.2f}\n")
            lines.append("| Method | TP | FN | FP | TPR | FPR | F1 |")
            lines.append("|--------|----|----|----|----|-----|-----|")

            # Sort by method name.
            for r in sorted(effect_results, key=lambda x: x.method):
                lines.append(
                    f"| {r.method} | {r.tp}/{r.n_true} | {r.fn} | {r.fp} | "
                    f"{r.tpr:.3f} | {r.fpr:.4f} | {r.f1:.3f} |"
                )

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Wrote Markdown: {output_path}")


def plot_power_curves(
    results: List[ResultMetrics],
    output_dir: Path,
    group_by: str = "effect_size",
) -> None:
    """Plot power curves (TPR vs parameter) for each method."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping plots")
        return

    # Group by scenario.
    by_scenario: Dict[str, List[ResultMetrics]] = {}
    for r in results:
        by_scenario.setdefault(r.scenario_name, []).append(r)

    for scenario_name, scenario_results in by_scenario.items():
        # Collect data by method.
        method_data: Dict[str, Dict[float, List[float]]] = {}
        for r in scenario_results:
            if r.method not in method_data:
                method_data[r.method] = {}
            x_val = getattr(r, group_by)
            method_data[r.method].setdefault(x_val, []).append(r.tpr)

        if not method_data:
            continue

        # Create plot.
        fig, ax = plt.subplots(figsize=(10, 6))

        colors = plt.cm.tab10.colors
        for i, (method, data) in enumerate(sorted(method_data.items())):
            x_vals = sorted(data.keys())
            y_vals = [np.mean(data[x]) for x in x_vals]
            y_stds = [np.std(data[x]) for x in x_vals] if any(len(data[x]) > 1 for x in x_vals) else None

            color = colors[i % len(colors)]
            ax.plot(x_vals, y_vals, "o-", label=method, color=color, linewidth=2, markersize=8)
            if y_stds and max(y_stds) > 0:
                ax.fill_between(
                    x_vals,
                    [y - s for y, s in zip(y_vals, y_stds)],
                    [y + s for y, s in zip(y_vals, y_stds)],
                    alpha=0.2,
                    color=color,
                )

        ax.set_xlabel(group_by.replace("_", " ").title(), fontsize=12)
        ax.set_ylabel("True Positive Rate (Power)", fontsize=12)
        ax.set_title(f"Power Curves: {scenario_name}", fontsize=14)
        ax.legend(loc="lower right", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        out_path = output_dir / f"power_curves_{scenario_name}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote plot: {out_path}")

    # Also create FPR plot (False Positive Rate).
    for scenario_name, scenario_results in by_scenario.items():
        method_data: Dict[str, Dict[float, List[float]]] = {}
        for r in scenario_results:
            if r.method not in method_data:
                method_data[r.method] = {}
            x_val = getattr(r, group_by)
            method_data[r.method].setdefault(x_val, []).append(r.fpr)

        if not method_data:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))

        for i, (method, data) in enumerate(sorted(method_data.items())):
            x_vals = sorted(data.keys())
            y_vals = [np.mean(data[x]) for x in x_vals]

            color = colors[i % len(colors)]
            ax.plot(x_vals, y_vals, "o-", label=method, color=color, linewidth=2, markersize=8)

        ax.axhline(y=0.05, color="red", linestyle="--", alpha=0.5, label="alpha=0.05")
        ax.set_xlabel(group_by.replace("_", " ").title(), fontsize=12)
        ax.set_ylabel("False Positive Rate", fontsize=12)
        ax.set_title(f"FPR Control: {scenario_name}", fontsize=14)
        ax.legend(loc="upper right", fontsize=10)
        ax.set_ylim(0, max(0.15, ax.get_ylim()[1]))
        ax.grid(True, alpha=0.3)

        out_path = output_dir / f"fpr_curves_{scenario_name}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote plot: {out_path}")


def print_summary_table(results: List[ResultMetrics]) -> None:
    """Print a summary table to console."""
    # Group by scenario and method, averaging across params.
    summary: Dict[Tuple[str, str], List[ResultMetrics]] = {}
    for r in results:
        key = (r.scenario_name, r.method)
        summary.setdefault(key, []).append(r)

    print("\n=== Summary Table ===")
    print(f"{'Scenario':<35} {'Method':<15} {'TPR':>8} {'FPR':>8} {'F1':>8} {'n':>4}")
    print("-" * 80)

    current_scenario = None
    for (scenario, method), rlist in sorted(summary.items()):
        if scenario != current_scenario:
            if current_scenario is not None:
                print()
            current_scenario = scenario

        avg_tpr = np.mean([r.tpr for r in rlist])
        avg_fpr = np.mean([r.fpr for r in rlist])
        avg_f1 = np.mean([r.f1 for r in rlist])
        n = len(rlist)

        print(f"{scenario:<35} {method:<15} {avg_tpr:>8.3f} {avg_fpr:>8.4f} {avg_f1:>8.3f} {n:>4}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate method comparison results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input-dir", type=Path, required=True,
        help="Directory containing .pkl result files (searched recursively).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for CSV/plots. Default: same as input-dir.",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Significance threshold for computing TP/FP.",
    )
    parser.add_argument(
        "--scenarios", nargs="+", default=None,
        help="Filter to specific scenarios.",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate power curve plots.",
    )
    parser.add_argument(
        "--plot-by", type=str, default="effect_size",
        choices=["effect_size", "time_points", "n_samples", "n_permutations"],
        help="X-axis variable for power curves.",
    )
    parser.add_argument(
        "--markdown", action="store_true",
        help="Generate markdown report.",
    )
    parser.add_argument(
        "--no-recursive", action="store_true",
        help="Don't search subdirectories.",
    )

    args = parser.parse_args(argv)

    if not args.input_dir.exists():
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return 1

    output_dir = args.output_dir or args.input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect results.
    results = collect_results(
        args.input_dir,
        alpha=args.alpha,
        scenarios=args.scenarios,
        recursive=not args.no_recursive,
    )

    if not results:
        print("No results found.")
        return 1

    # Print summary.
    print_summary_table(results)

    # Write CSV.
    csv_path = output_dir / "results_summary.csv"
    results_to_csv(results, csv_path)

    # Write markdown if requested.
    if args.markdown:
        md_path = output_dir / "results_summary.md"
        results_to_markdown(results, md_path)

    # Generate plots if requested.
    if args.plot:
        plot_power_curves(results, output_dir, group_by=args.plot_by)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
