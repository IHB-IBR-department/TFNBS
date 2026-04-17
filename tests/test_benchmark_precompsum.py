"""
Benchmark: pre-computed sums fast path vs slow path across (N, n_sub, n_perm).

Opt-in — set CONNINFPY_TEST_BENCHMARK=1 to run. Prints a comparison table to
stdout and asserts the fast path beats the slow path at every grid point.

Invoke with:
    CONNINFPY_TEST_BENCHMARK=1 pytest tests/test_benchmark_precompsum.py -s
"""

import os
import time
import unittest
from functools import partial
from unittest import TestCase

import numpy as np

from conninfpy.pairwise_stats import (
    _fast_permutation_task_ind,
    _fast_permutation_task_paired,
    _permutation_task_ind,
    _permutation_task_paired,
    _precompute_edge_sums,
    _precompute_twosample_sums,
    compute_p_val,
    compute_t_stat,
    compute_t_stat_diff,
)


N_GRID = (60, 100, 200, 400)
NSUB_GRID = (30, 60, 100)
NPERM_GRID = (100, 500, 1000)


def _make_symmetric_data(n_sub, N, rng):
    x = rng.randn(n_sub, N, N)
    x = (x + x.transpose(0, 2, 1)) / 2
    for s in range(n_sub):
        np.fill_diagonal(x[s], 0)
    return x


def _time_loop(task_fn, seeds):
    t0 = time.perf_counter()
    for seed in seeds:
        task_fn(seed)
    return time.perf_counter() - t0


def _print_header():
    print()
    print(f'{"N":>4} {"n_sub":>6} {"n_perm":>7}   '
          f'{"slow (s)":>10} {"fast (s)":>10} {"speedup":>9}')
    print('-' * 60)


def _print_row(N, n_sub, n_perm, t_slow, t_fast):
    speedup = t_slow / t_fast if t_fast > 0 else float('inf')
    print(f'{N:>4} {n_sub:>6} {n_perm:>7}   '
          f'{t_slow:>10.3f} {t_fast:>10.3f} {speedup:>8.1f}x')


@unittest.skipUnless(
    os.getenv("CONNINFPY_TEST_BENCHMARK") == "1",
    "Opt-in benchmark — set CONNINFPY_TEST_BENCHMARK=1 to run",
)
class TestPrecomputedSumsBenchmark(TestCase):
    """Benchmark the sign-flip and group-label permutation fast paths.

    Grid: N ∈ {60, 100, 200, 400}, n_sub ∈ {30, 60, 100}, n_perm ∈ {100, 500, 1000}.
    """

    def test_paired(self):
        """Paired / one-sample: slow vs fast over full permutation loop."""
        rng = np.random.RandomState(0)
        _print_header()
        for N in N_GRID:
            for n_sub in NSUB_GRID:
                diffs = _make_symmetric_data(n_sub, N, rng)
                X, sumsq = _precompute_edge_sums(diffs)
                slow_task = partial(_permutation_task_paired, diffs, compute_t_stat_diff)
                fast_task = partial(_fast_permutation_task_paired, X, sumsq)
                for n_perm in NPERM_GRID:
                    seeds = rng.randint(0, 2**31 - 1, size=n_perm)
                    t_slow = _time_loop(slow_task, seeds)
                    t_fast = _time_loop(fast_task, seeds)
                    _print_row(N, n_sub, n_perm, t_slow, t_fast)
                    self.assertLess(
                        t_fast, t_slow,
                        f'fast path should beat slow at N={N}, n_sub={n_sub}, n_perm={n_perm}'
                    )

    def test_two_sample(self):
        """Two-sample (Welch): slow vs fast over full permutation loop."""
        rng = np.random.RandomState(1)
        _print_header()
        for N in N_GRID:
            for n_sub in NSUB_GRID:
                n1 = n_sub // 2
                n2 = n_sub - n1
                g1 = _make_symmetric_data(n1, N, rng)
                g2 = _make_symmetric_data(n2, N, rng)
                full = np.concatenate([g1, g2], axis=0)
                Xall, Xall2, sum_all, sumsq_all = _precompute_twosample_sums(full)
                slow_task = partial(_permutation_task_ind, full, compute_t_stat, n1)
                fast_task = partial(
                    _fast_permutation_task_ind, Xall, Xall2, sum_all, sumsq_all, n1
                )
                for n_perm in NPERM_GRID:
                    seeds = rng.randint(0, 2**31 - 1, size=n_perm)
                    t_slow = _time_loop(slow_task, seeds)
                    t_fast = _time_loop(fast_task, seeds)
                    _print_row(N, n_sub, n_perm, t_slow, t_fast)
                    self.assertLess(
                        t_fast, t_slow,
                        f'fast path should beat slow at N={N}, n_sub={n_sub}, n_perm={n_perm}'
                    )

    def test_composed_with_acceleration(self):
        """Pre-computation × GPD acceleration: end-to-end wall-clock.

        The two optimizations are orthogonal:
          - pre-computation: each permutation faster (applies to method='tstat')
          - acceleration='gpd': fewer permutations needed (~200 vs ~5000)
        When combined, both stack.
        """
        rng = np.random.RandomState(2)
        print()
        print(f'{"N":>4} {"n_sub":>6}   '
              f'{"baseline (s)":>14} {"+gpd only (s)":>16} '
              f'{"+precomp only (s)":>20} {"both (s)":>10}')
        print('-' * 80)

        # Configuration: baseline uses n_perm=5000, GPD uses n_perm=200.
        # Baseline is simulated by running the slow-path loop directly at 5000
        # perms (we can't easily disable the fast path in compute_p_val without
        # monkey-patching, so we time the slow task loop directly as a proxy).
        n_perm_full = 5000
        n_perm_gpd = 200

        for N in (100, 200, 400):
            for n_sub in (30, 60):
                n1 = n_sub // 2
                g1 = _make_symmetric_data(n1, N, rng)
                g2 = _make_symmetric_data(n_sub - n1, N, rng)

                full = np.concatenate([g1, g2], axis=0)
                Xall, Xall2, sum_all, sumsq_all = _precompute_twosample_sums(full)
                slow_task = partial(_permutation_task_ind, full, compute_t_stat, n1)
                fast_task = partial(
                    _fast_permutation_task_ind, Xall, Xall2, sum_all, sumsq_all, n1
                )

                # Baseline: slow loop × 5000 perms (emulated cost; acceleration off)
                seeds_full = rng.randint(0, 2**31 - 1, size=n_perm_full)
                t_baseline = _time_loop(slow_task, seeds_full)

                # GPD only: slow loop × 200 perms (25× fewer perms, same per-perm cost)
                seeds_gpd = rng.randint(0, 2**31 - 1, size=n_perm_gpd)
                t_gpd_only = _time_loop(slow_task, seeds_gpd)

                # Pre-computation only: fast loop × 5000 perms
                t_precomp_only = _time_loop(fast_task, seeds_full)

                # Both: fast loop × 200 perms (what compute_p_val does by default
                # with method='tstat' + acceleration='gpd' in the updated code)
                t_both = _time_loop(fast_task, seeds_gpd)

                print(f'{N:>4} {n_sub:>6}   '
                      f'{t_baseline:>14.2f} {t_gpd_only:>16.2f} '
                      f'{t_precomp_only:>20.2f} {t_both:>10.3f}')

                # Sanity: combined should beat either alone
                self.assertLess(t_both, t_gpd_only)
                self.assertLess(t_both, t_precomp_only)
                self.assertLess(t_both, t_baseline)
