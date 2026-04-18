#!/usr/bin/env python3
"""
Computational cost estimation for ConnInfPy methods.

Times ``compute_p_val`` across different network sizes and permutation counts.
Generates a summary table and optional CSV for inclusion in papers.

Usage
-----
Quick local test (30 nodes, 100 permutations)::

    python examples/benchmarks/timing_benchmark.py --n_nodes 30 --n_perms 100

Typical atlas sizes::

    python examples/benchmarks/timing_benchmark.py --n_nodes 90 200 400 --n_perms 1000

All parameters::

    python examples/benchmarks/timing_benchmark.py \\
        --n_nodes 90 200 400 \\
        --n_perms 500 1000 \\
        --n_samples 30 \\
        --methods tstat tfnbs ni_tfnbs fbc_tfnbs nbs_extent nbs_intensity cnbs bonferroni bh_fdr \\
        --repeats 3 \\
        --output examples/results/timing_benchmark.csv
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── ensure the package is importable ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (2 levels up from examples/benchmarks/)

from conninfpy import (
    compute_p_val,
    ModularDatasetGenerator,
    fisher_r_to_z,
    StatMethod,
)


# ── helpers ───────────────────────────────────────────────────────────────────

ALL_METHODS = [
    "tstat",
    "tfnbs",
    "ni_tfnbs",
    "fbc_tfnbs",
    "nbs_extent",
    "nbs_intensity",
    "cnbs",
    "bonferroni",
    "bh_fdr",
]

PARAMETRIC_METHODS = {"bonferroni", "bh_fdr"}


def _generate_data(n_nodes: int, n_samples: int, seed: int = 42):
    """Generate a pair of FC matrices with a within-module effect."""
    gen = ModularDatasetGenerator(
        N=n_nodes, n_modules=4, intra_corr=0.3, inter_corr=0.05,
        noise_level=0.05, seed=seed,
    )
    mask = gen.get_mask_within_module(0)
    g1, g2, _ = gen.generate_data(
        mask, effect_size=0.3, n_samples_g1=n_samples, n_samples_g2=n_samples,
    )
    return fisher_r_to_z(g1), fisher_r_to_z(g2)


def _make_net_labels(n_nodes: int) -> np.ndarray:
    """Generate module labels for n_nodes split into 4 modules."""
    mod_size = n_nodes // 4
    labels = np.repeat(np.arange(4), mod_size)
    if n_nodes % 4 != 0:
        labels = np.concatenate([labels, np.full(n_nodes - len(labels), 3)])
    return labels


# Map user-facing method names to compute_p_val method + extra kwargs
METHOD_TO_API = {
    "tstat": "tstat",
    "tfnbs": "tfnbs",
    "ni_tfnbs": "ni_tfnbs",
    "fbc_tfnbs": "fbc_tfnbs",
    "nbs_extent": "nbs",
    "nbs_intensity": "nbs",
    "cnbs": "cnbs",
    "bonferroni": "bonferroni",
    "bh_fdr": "bh_fdr",
}


def _get_method_kwargs(method: str, n_nodes: int):
    """Return (api_method, extra_kwargs) for compute_p_val."""
    kw = {}
    if method in ("tfnbs", "ni_tfnbs", "fbc_tfnbs"):
        kw.update(e=0.5, h=2.0, n=30, start_thres=1.25)
    if method == "fbc_tfnbs":
        kw["min_cluster_size"] = 3
    if method in ("nbs_extent", "nbs_intensity"):
        kw["threshold"] = 2.0
    if method == "nbs_extent":
        kw["nbs_stat"] = "extent"
    elif method == "nbs_intensity":
        kw["nbs_stat"] = "intensity"
    # cnbs, ni_tfnbs, and fbc_tfnbs all require net_labels
    if method in ("cnbs", "ni_tfnbs", "fbc_tfnbs"):
        kw["net_labels"] = _make_net_labels(n_nodes)
    return kw


def time_method(
    method: str,
    group1: np.ndarray,
    group2: np.ndarray,
    n_perms: int,
    seed: int = 42,
) -> float:
    """Time a single compute_p_val call. Returns elapsed seconds."""
    n_nodes = group1.shape[1]
    api_method = METHOD_TO_API[method]
    kw = _get_method_kwargs(method, n_nodes)

    t0 = time.perf_counter()
    compute_p_val(
        group1, group2,
        n_permutations=n_perms,
        test_type="two-sample",
        method=api_method,
        use_mp=False,       # single-core for reproducible timing
        random_state=seed,
        **kw,
    )
    return time.perf_counter() - t0


# ── main ──────────────────────────────────────────────────────────────────────

def run_benchmark(
    n_nodes_list: list,
    n_perms_list: list,
    methods: list,
    n_samples: int = 30,
    repeats: int = 3,
    seed: int = 42,
):
    """Run the full timing benchmark and return a DataFrame."""
    rows = []
    total_tasks = len(n_nodes_list) * len(n_perms_list) * len(methods) * repeats
    done = 0

    for n_nodes in n_nodes_list:
        n_edges = n_nodes * (n_nodes - 1) // 2
        print(f"\n{'='*70}")
        print(f"  n_nodes={n_nodes}  (n_edges={n_edges}), n_samples={n_samples}")
        print(f"{'='*70}")

        # Generate data once per network size
        g1, g2 = _generate_data(n_nodes, n_samples, seed)

        for n_perms in n_perms_list:
            print(f"\n  n_permutations={n_perms}")
            print(f"  {'Method':<18} {'Time (s)':>10}  {'per-perm (ms)':>14}  {'per-edge-perm (us)':>18}")
            print(f"  {'-'*64}")

            for method in methods:
                times = []
                for r in range(repeats):
                    t = time_method(method, g1, g2, n_perms, seed=seed + r)
                    times.append(t)
                    done += 1

                mean_t = np.mean(times)
                std_t = np.std(times) if repeats > 1 else 0.0

                # Parametric methods don't scale with n_perms
                if method in PARAMETRIC_METHODS:
                    per_perm_ms = 0.0
                    per_edge_perm_us = 0.0
                else:
                    per_perm_ms = mean_t / n_perms * 1000
                    per_edge_perm_us = mean_t / n_perms / n_edges * 1e6

                print(
                    f"  {method:<18} {mean_t:>10.3f}  {per_perm_ms:>14.3f}  "
                    f"{per_edge_perm_us:>18.3f}  "
                    f"(+/- {std_t:.3f}s, {done}/{total_tasks})"
                )

                rows.append({
                    "n_nodes": n_nodes,
                    "n_edges": n_edges,
                    "n_samples": n_samples,
                    "n_permutations": n_perms,
                    "method": method,
                    "time_mean_s": round(mean_t, 4),
                    "time_std_s": round(std_t, 4),
                    "per_perm_ms": round(per_perm_ms, 4),
                    "per_edge_perm_us": round(per_edge_perm_us, 4),
                    "repeats": repeats,
                })

    return pd.DataFrame(rows)


def _format_time_cell(t: float, width: int = 10) -> str:
    """Format a time value as a right-aligned table cell."""
    if t < 0.1:
        s = f"{t*1000:.0f} ms"
    elif t < 1:
        s = f"{t*1000:.0f} ms"
    elif t < 60:
        s = f"{t:.1f} s"
    elif t < 3600:
        s = f"{t/60:.1f} min"
    else:
        s = f"{t/3600:.1f} hr"
    return f" {s:>{width}} |"


def print_summary_table(df: pd.DataFrame):
    """Print a markdown summary table grouped by n_nodes."""
    print(f"\n{'='*70}")
    print("  SUMMARY: Estimated wall-clock time (single core)")
    print(f"{'='*70}\n")

    for n_nodes in sorted(df["n_nodes"].unique()):
        sub = df[df["n_nodes"] == n_nodes]
        n_edges = sub.iloc[0]["n_edges"]
        n_perms_list = sorted(sub["n_permutations"].unique())

        print(f"### n_nodes={n_nodes} ({n_edges} edges)\n")
        header = f"| {'Method':<18} |"
        sep = f"|{'-'*20}|"
        for np_ in n_perms_list:
            header += f" {np_} perms |"
            sep += f"{'-'*12}|"
        print(header)
        print(sep)

        for method in sub["method"].unique():
            line = f"| {method:<18} |"
            for np_ in n_perms_list:
                row = sub[(sub["method"] == method) & (sub["n_permutations"] == np_)]
                if len(row) > 0:
                    line += _format_time_cell(row.iloc[0]["time_mean_s"])
                else:
                    line += f" {'—':>10} |"
            print(line)
        print()

    # Extrapolation: estimated time for 1000 permutations
    n_nodes_tested = sorted(df["n_nodes"].unique())
    print("### Estimated time for 1000 permutations (single core)\n")
    header = f"| {'Method':<18} |"
    sep = f"|{'-'*20}|"
    for nn in n_nodes_tested:
        ne = nn * (nn - 1) // 2
        header += f" {nn} nodes ({ne} edges) |"
        sep += f"{'-'*22}|"
    print(header)
    print(sep)

    for method in df["method"].unique():
        line = f"| {method:<18} |"
        for nn in n_nodes_tested:
            row = df[(df["method"] == method) & (df["n_nodes"] == nn)]
            if len(row) > 0:
                if method in PARAMETRIC_METHODS:
                    # Parametric: constant time regardless of n_perms
                    t = row.iloc[0]["time_mean_s"]
                else:
                    # Scale to 1000 perms using measured per-perm cost
                    t = row.iloc[0]["per_perm_ms"] * 1000 / 1000  # per_perm_ms * 1000_perms / 1000
                line += _format_time_cell(t, width=20)
            else:
                line += f" {'—':>20} |"
        print(line)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark compute_p_val timing across network sizes and permutation counts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--n_nodes", type=int, nargs="+", default=[90, 200, 400],
        help="Network sizes to test (default: 90 200 400)",
    )
    parser.add_argument(
        "--n_perms", type=int, nargs="+", default=[1000],
        help="Permutation counts to test (default: 1000)",
    )
    parser.add_argument(
        "--n_samples", type=int, default=30,
        help="Samples per group (default: 30)",
    )
    parser.add_argument(
        "--methods", nargs="+", default=ALL_METHODS,
        help=f"Methods to benchmark (default: all). Choices: {ALL_METHODS}",
    )
    parser.add_argument(
        "--repeats", type=int, default=3,
        help="Timing repeats per condition (default: 3)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save CSV results (optional)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    print("ConnInfPy Timing Benchmark")
    print(f"  n_nodes:  {args.n_nodes}")
    print(f"  n_perms:  {args.n_perms}")
    print(f"  methods:  {args.methods}")
    print(f"  n_samples: {args.n_samples}")
    print(f"  repeats:  {args.repeats}")

    df = run_benchmark(
        n_nodes_list=args.n_nodes,
        n_perms_list=args.n_perms,
        methods=args.methods,
        n_samples=args.n_samples,
        repeats=args.repeats,
        seed=args.seed,
    )

    print_summary_table(df)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
