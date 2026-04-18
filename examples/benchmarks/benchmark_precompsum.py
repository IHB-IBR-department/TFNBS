"""
Benchmark: pre-computed sums fast path vs slow path.

Measures per-permutation timing across (N, n_sub, n_perm) grids for three
scenarios:

    paired       — sign-flip permutation, paired/one-sample t-test
    two-sample   — group-label permutation, Welch's two-sample t-test
    enhancement  — NBS and TFNBS enhancement on top of the fast t-stat step
    composed     — combining pre-computation with GPD acceleration

Prints a comparison table. At the end of each section a soft regression
check runs: if any grid cell shows `fast >= slow`, a warning is printed and
the script exits non-zero. This lets the file double as a local performance
tripwire (``python examples/benchmarks/benchmark_precompsum.py && echo OK``).

Usage
-----
Default (all scenarios on the full grid):

    python examples/benchmarks/benchmark_precompsum.py

Pick one scenario:

    python examples/benchmarks/benchmark_precompsum.py --case paired
    python examples/benchmarks/benchmark_precompsum.py --case enhancement

Smaller grid for a quick run:

    python examples/benchmarks/benchmark_precompsum.py --quick

Customize the grid:

    python examples/benchmarks/benchmark_precompsum.py \\
        --n-nodes 100 200 \\
        --n-sub 30 60 \\
        --n-perm 100 500
"""

from __future__ import annotations

import argparse
import sys
import time
from functools import partial
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from conninfpy._enhancement import apply_nbs, apply_tfnbs
from conninfpy.pairwise_stats import (
    _extract_max_stats,
    _fast_perm_task_enhanced_paired,
    _fast_permutation_task_ind,
    _fast_permutation_task_paired,
    _permutation_task_ind,
    _permutation_task_paired,
    _precompute_edge_sums,
    _precompute_twosample_sums,
    compute_t_stat,
    compute_t_stat_diff,
)

# Ensure sibling benchmark helpers are importable when run directly.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from _common import PAPER_TOPOLOGIES, make_two_sample  # noqa: E402


DEFAULT_N_GRID = (60, 100, 200, 400)
DEFAULT_NSUB_GRID = (30, 60, 100)
DEFAULT_NPERM_GRID = (100, 500, 1000)

QUICK_N_GRID = (100, 200)
QUICK_NSUB_GRID = (30,)
QUICK_NPERM_GRID = (100,)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _make_diffs(n_sub: int, N: int, topology: str, seed: int) -> np.ndarray:
    """Biologically-realistic paired-differences array from a topology scenario.

    Builds a two-sample pair via the shared factory and returns ``g2 - g1``,
    simulating paired differences with a topology-defined effect on top of
    modular-covariance backbone.
    """
    g1, g2 = make_two_sample(N=N, n_sub=n_sub, topology=topology, seed=seed)
    return g2 - g1


def _make_two_groups(n1: int, n2: int, N: int, topology: str, seed: int):
    """Biologically-realistic two-sample arrays (equal nsub per group by default).

    Uses make_two_sample for the majority group, then generates the second
    via an additional draw if sizes differ.
    """
    nsub_eq = max(n1, n2)
    g1_full, g2_full = make_two_sample(
        N=N, n_sub=nsub_eq, topology=topology, seed=seed,
    )
    return g1_full[:n1], g2_full[:n2]


def _time_loop(task_fn: Callable, seeds: Iterable[int]) -> float:
    t0 = time.perf_counter()
    for seed in seeds:
        task_fn(seed)
    return time.perf_counter() - t0


def _print_header(title: str) -> None:
    print()
    print(title)
    print(f'{"N":>4} {"n_sub":>6} {"n_perm":>7} {"topology":<24}  '
          f'{"slow (s)":>10} {"fast (s)":>10} {"speedup":>9}')
    print('-' * 82)


def _print_row(
    N: int, n_sub: int, n_perm: int, t_slow: float, t_fast: float,
    topology: str = "",
) -> None:
    speedup = t_slow / t_fast if t_fast > 0 else float('inf')
    print(f'{N:>4} {n_sub:>6} {n_perm:>7} {topology:<24}  '
          f'{t_slow:>10.3f} {t_fast:>10.3f} {speedup:>8.1f}x')


def _check_speedup(
    regressions: List[str],
    label: str,
    N: int,
    n_sub: int,
    n_perm: int,
    t_slow: float,
    t_fast: float,
) -> None:
    """Soft regression check. Appends a message to `regressions` if fast ≥ slow."""
    if t_fast >= t_slow:
        regressions.append(
            f'  [{label}] N={N}, n_sub={n_sub}, n_perm={n_perm}: '
            f'slow={t_slow:.3f}s, fast={t_fast:.3f}s — fast path is NOT faster'
        )


# -----------------------------------------------------------------------------
# Scenarios
# -----------------------------------------------------------------------------

def run_paired(
    n_grid: Sequence[int],
    nsub_grid: Sequence[int],
    nperm_grid: Sequence[int],
    topologies: Sequence[str],
    regressions: List[str],
    records: Optional[List[dict]] = None,
) -> None:
    """Paired/one-sample: slow vs fast over full permutation loop."""
    rng = np.random.RandomState(0)
    _print_header("Paired / one-sample (sign-flip permutation)")
    for N in n_grid:
        for n_sub in nsub_grid:
            for topo in topologies:
                diffs = _make_diffs(n_sub, N, topology=topo, seed=rng.randint(1_000_000))
                X, sumsq = _precompute_edge_sums(diffs)
                slow_task = partial(_permutation_task_paired, diffs, compute_t_stat_diff)
                fast_task = partial(_fast_permutation_task_paired, X, sumsq)
                for n_perm in nperm_grid:
                    seeds = rng.randint(0, 2**31 - 1, size=n_perm)
                    t_slow = _time_loop(slow_task, seeds)
                    t_fast = _time_loop(fast_task, seeds)
                    _print_row(N, n_sub, n_perm, t_slow, t_fast, topology=topo)
                    _check_speedup(regressions, f"paired[{topo}]", N, n_sub, n_perm, t_slow, t_fast)
                    if records is not None:
                        records.append(dict(
                            case="paired", N=N, n_sub=n_sub, n_perm=n_perm,
                            topology=topo, method="", t_slow=t_slow, t_fast=t_fast,
                            speedup=t_slow / t_fast if t_fast > 0 else float("inf"),
                        ))


def run_two_sample(
    n_grid: Sequence[int],
    nsub_grid: Sequence[int],
    nperm_grid: Sequence[int],
    topologies: Sequence[str],
    regressions: List[str],
    records: Optional[List[dict]] = None,
) -> None:
    """Two-sample (Welch): slow vs fast over full permutation loop."""
    rng = np.random.RandomState(1)
    _print_header("Two-sample (Welch, group-label permutation)")
    for N in n_grid:
        for n_sub in nsub_grid:
            n1 = n_sub // 2
            n2 = n_sub - n1
            for topo in topologies:
                g1, g2 = _make_two_groups(n1, n2, N, topology=topo, seed=rng.randint(1_000_000))
                full = np.concatenate([g1, g2], axis=0)
                Xall, Xall2, sum_all, sumsq_all = _precompute_twosample_sums(full)
                slow_task = partial(_permutation_task_ind, full, compute_t_stat, n1)
                fast_task = partial(
                    _fast_permutation_task_ind, Xall, Xall2, sum_all, sumsq_all, n1
                )
                for n_perm in nperm_grid:
                    seeds = rng.randint(0, 2**31 - 1, size=n_perm)
                    t_slow = _time_loop(slow_task, seeds)
                    t_fast = _time_loop(fast_task, seeds)
                    _print_row(N, n_sub, n_perm, t_slow, t_fast, topology=topo)
                    _check_speedup(regressions, f"two-sample[{topo}]", N, n_sub, n_perm, t_slow, t_fast)
                    if records is not None:
                        records.append(dict(
                            case="two-sample", N=N, n_sub=n_sub, n_perm=n_perm,
                            topology=topo, method="", t_slow=t_slow, t_fast=t_fast,
                            speedup=t_slow / t_fast if t_fast > 0 else float("inf"),
                        ))


def run_enhancement(
    n_grid: Sequence[int],
    nsub_grid: Sequence[int],
    nperm_grid: Sequence[int],
    topologies: Sequence[str],
    regressions: List[str],
    records: Optional[List[dict]] = None,
) -> None:
    """Enhancement methods (NBS, TFNBS): slow scorer vs fast enhanced perm task.

    The slow path here reproduces the pre-refactor behaviour: inside each
    permutation, ``compute_t_stat_diff`` is called (full 3D mean+std pass)
    and then the shared ``apply_*`` enhancement wrapper runs on the (N, N)
    t-stat matrices. The fast path uses sums + reconstruction + enhancement.
    """
    rng = np.random.RandomState(3)
    enhancers: Tuple[Tuple[str, Callable, dict], ...] = (
        ("nbs", apply_nbs, {"threshold": 2.0, "nbs_stat": "extent"}),
        ("tfnbs", apply_tfnbs, {"e": 0.4, "h": 3.0, "n": 10}),
    )

    def slow_task(diffs, enhancer, enh_kwargs, seed):
        rng_ = np.random.RandomState(seed)
        signs = rng_.choice([1, -1], diffs.shape[0]).reshape(-1, 1, 1)
        t_stat_dict = compute_t_stat_diff(signs * diffs)
        stat_dict = enhancer(t_stat_dict, **enh_kwargs)
        return _extract_max_stats(stat_dict, diffs[0].shape)

    for method_name, enhancer, enh_kwargs in enhancers:
        _print_header(f"Enhancement: {method_name}")
        for N in n_grid:
            for n_sub in nsub_grid:
                for topo in topologies:
                    diffs = _make_diffs(n_sub, N, topology=topo, seed=rng.randint(1_000_000))
                    X, sumsq = _precompute_edge_sums(diffs)
                    triu_idx = np.triu_indices(N, k=1)
                    slow_fn = partial(slow_task, diffs, enhancer, enh_kwargs)
                    fast_fn = partial(
                        _fast_perm_task_enhanced_paired,
                        X, sumsq, triu_idx, N, enhancer, enh_kwargs,
                    )
                    for n_perm in nperm_grid:
                        seeds = rng.randint(0, 2**31 - 1, size=n_perm)
                        t_slow = _time_loop(slow_fn, seeds)
                        t_fast = _time_loop(fast_fn, seeds)
                        _print_row(N, n_sub, n_perm, t_slow, t_fast, topology=topo)
                        _check_speedup(
                            regressions, f"{method_name}[{topo}]", N, n_sub, n_perm, t_slow, t_fast,
                        )
                        if records is not None:
                            records.append(dict(
                                case="enhancement", N=N, n_sub=n_sub, n_perm=n_perm,
                                topology=topo, method=method_name,
                                t_slow=t_slow, t_fast=t_fast,
                                speedup=t_slow / t_fast if t_fast > 0 else float("inf"),
                            ))


def run_composed(
    regressions: List[str],
    composed_records: Optional[List[dict]] = None,
) -> None:
    """Pre-computation × GPD acceleration: end-to-end wall-clock.

    The two optimizations are orthogonal:
      - pre-computation: each permutation faster (method='tstat')
      - acceleration='gpd': fewer permutations needed (~200 vs ~5000)
    When combined, both stack.
    """
    rng = np.random.RandomState(2)
    print()
    print("Composed: pre-computation × GPD acceleration (two-sample)")
    print(f'{"N":>4} {"n_sub":>6}   '
          f'{"baseline (s)":>14} {"+gpd only (s)":>16} '
          f'{"+precomp only (s)":>20} {"both (s)":>10}')
    print('-' * 80)

    n_perm_full = 5000
    n_perm_gpd = 200

    for N in (100, 200, 400):
        for n_sub in (30, 60):
            n1 = n_sub // 2
            n2 = n_sub - n1
            # Composed benchmark uses a fixed topology (within_module_dense)
            # — its goal is the orthogonal-optimizations demonstration, not
            # a topology sweep.
            g1, g2 = _make_two_groups(n1, n2, N, topology="within_module_dense",
                                      seed=rng.randint(1_000_000))

            full = np.concatenate([g1, g2], axis=0)
            Xall, Xall2, sum_all, sumsq_all = _precompute_twosample_sums(full)
            slow_task = partial(_permutation_task_ind, full, compute_t_stat, n1)
            fast_task = partial(
                _fast_permutation_task_ind, Xall, Xall2, sum_all, sumsq_all, n1
            )

            seeds_full = rng.randint(0, 2**31 - 1, size=n_perm_full)
            t_baseline = _time_loop(slow_task, seeds_full)

            seeds_gpd = rng.randint(0, 2**31 - 1, size=n_perm_gpd)
            t_gpd_only = _time_loop(slow_task, seeds_gpd)

            t_precomp_only = _time_loop(fast_task, seeds_full)
            t_both = _time_loop(fast_task, seeds_gpd)

            print(f'{N:>4} {n_sub:>6}   '
                  f'{t_baseline:>14.2f} {t_gpd_only:>16.2f} '
                  f'{t_precomp_only:>20.2f} {t_both:>10.3f}')

            if not (t_both < t_gpd_only and t_both < t_precomp_only and t_both < t_baseline):
                regressions.append(
                    f'  [composed] N={N}, n_sub={n_sub}: combined should be fastest '
                    f'(baseline={t_baseline:.2f}s, gpd={t_gpd_only:.2f}s, '
                    f'precomp={t_precomp_only:.2f}s, both={t_both:.3f}s)'
                )
            if composed_records is not None:
                composed_records.append(dict(
                    N=N, n_sub=n_sub,
                    n_perm_full=n_perm_full, n_perm_gpd=n_perm_gpd,
                    t_baseline=t_baseline, t_gpd_only=t_gpd_only,
                    t_precomp_only=t_precomp_only, t_both=t_both,
                ))


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--case",
        choices=("paired", "two-sample", "enhancement", "composed", "all"),
        default="all",
        help="Which scenario(s) to run (default: all)",
    )
    parser.add_argument("--n-nodes", type=int, nargs="+", default=None,
                        help=f"Network sizes N (default: {DEFAULT_N_GRID})")
    parser.add_argument("--n-sub", type=int, nargs="+", default=None,
                        help=f"Subjects per group (default: {DEFAULT_NSUB_GRID})")
    parser.add_argument("--n-perm", type=int, nargs="+", default=None,
                        help=f"Permutation counts (default: {DEFAULT_NPERM_GRID})")
    parser.add_argument("--quick", action="store_true",
                        help=f"Small grid for a fast run "
                             f"(N={QUICK_N_GRID}, n_sub={QUICK_NSUB_GRID}, "
                             f"n_perm={QUICK_NPERM_GRID})")
    parser.add_argument("--topologies", nargs="+", default=["within_module_dense"],
                        help=f"Topology scenarios to sweep (default: within_module_dense). "
                             f"The 4 paper topologies: {list(PAPER_TOPOLOGIES)}")
    parser.add_argument(
        "--output", type=str,
        default="examples/benchmarks/res_benchmarks/precompsum.csv",
        help="Path to save CSV results (default: res_benchmarks/precompsum.csv). "
             "Composed case saved to <output>_composed.csv. Pass empty to skip.",
    )
    args = parser.parse_args(argv)

    if args.quick:
        n_grid = args.n_nodes or QUICK_N_GRID
        nsub_grid = args.n_sub or QUICK_NSUB_GRID
        nperm_grid = args.n_perm or QUICK_NPERM_GRID
    else:
        n_grid = args.n_nodes or DEFAULT_N_GRID
        nsub_grid = args.n_sub or DEFAULT_NSUB_GRID
        nperm_grid = args.n_perm or DEFAULT_NPERM_GRID

    regressions: List[str] = []
    records: List[dict] = []
    composed_records: List[dict] = []

    if args.case in ("paired", "all"):
        run_paired(n_grid, nsub_grid, nperm_grid, args.topologies, regressions, records)
    if args.case in ("two-sample", "all"):
        run_two_sample(n_grid, nsub_grid, nperm_grid, args.topologies, regressions, records)
    if args.case in ("enhancement", "all"):
        run_enhancement(n_grid, nsub_grid, nperm_grid, args.topologies, regressions, records)
    if args.case in ("composed", "all"):
        run_composed(regressions, composed_records)

    if args.output:
        import pandas as pd
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        if records:
            pd.DataFrame(records).to_csv(out, index=False)
            print(f"\nResults saved to {out}")
        if composed_records:
            composed_out = out.with_name(out.stem + "_composed.csv")
            pd.DataFrame(composed_records).to_csv(composed_out, index=False)
            print(f"Composed results saved to {composed_out}")

    print()
    if regressions:
        print(f"REGRESSION: {len(regressions)} grid cell(s) where fast path is not faster:")
        for msg in regressions:
            print(msg)
        return 1
    print("OK: fast path beats slow path at every grid cell.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
