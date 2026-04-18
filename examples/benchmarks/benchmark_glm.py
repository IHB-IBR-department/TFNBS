"""
Benchmark: GLM pipeline performance across network sizes and methods.

Measures timing for the full GLM pipeline (compute_p_val_glm) including
Freedman-Lane permutation with various enhancement methods.

Usage:
    python examples/benchmarks/benchmark_glm.py
    python examples/benchmarks/benchmark_glm.py --quick
    python examples/benchmarks/benchmark_glm.py --full
    python examples/benchmarks/benchmark_glm.py --mp      # with multiprocessing
"""

import argparse
import time

import numpy as np

from conninfpy.glm_stats import compute_p_val_glm, compute_glm_stat, build_design_matrix


def generate_data(N, n_subjects, seed=42):
    """Generate synthetic connectivity with continuous + confound."""
    rng = np.random.RandomState(seed)
    Y = rng.randn(n_subjects, N, N) * 0.5
    age = rng.randn(n_subjects)
    motion = rng.randn(n_subjects)

    for s in range(n_subjects):
        Y[s] = (Y[s] + Y[s].T) / 2
        np.fill_diagonal(Y[s], 0)

    # Plant effects: 5 edges correlated with age
    effect_edges = [(1, 2), (2, 3), (3, 4), (5, 6), (7, 8)]
    for i, j in effect_edges:
        if i < N and j < N:
            for s in range(n_subjects):
                Y[s, i, j] += 1.0 * age[s]
                Y[s, j, i] += 1.0 * age[s]

    return Y, age, motion


def benchmark_compute_glm_stat(N_values, n_subjects, n_repeats=5):
    """Benchmark just the GLM stat computation (no permutation)."""
    print("\n--- compute_glm_stat (no permutation) ---")
    print(f"{'N':>5} {'edges':>7} {'time':>10} {'edges/s':>10}")
    print("-" * 35)

    for N in N_values:
        Y, age, motion = generate_data(N, n_subjects)
        X, contrast = build_design_matrix(age, confounds=motion)

        times = []
        for _ in range(n_repeats):
            t0 = time.time()
            compute_glm_stat(Y, X, contrast, stat_type='tstat')
            times.append(time.time() - t0)

        avg_t = np.mean(times)
        n_edges = N * (N - 1) // 2
        print(f"{N:>5} {n_edges:>7} {avg_t*1000:>8.1f}ms {n_edges/avg_t:>10.0f}")


def benchmark_full_pipeline(N_values, n_subjects, n_perms, use_mp, methods):
    """Benchmark full pipeline with permutation testing."""
    print(f"\n--- Full pipeline ({n_perms} perms, mp={use_mp}) ---")
    header = f"{'N':>5} {'edges':>7} {'method':<10} {'time':>8} {'p_planted':>10}"
    print(header)
    print("-" * len(header))

    for N in N_values:
        Y, age, motion = generate_data(N, n_subjects)
        n_edges = N * (N - 1) // 2

        for method_name, method_kwargs in methods:
            net_labels = None
            if method_name in ("cnbs", "ni_tfnbs", "fbc_tfnbs"):
                # Simple 2-community partition
                net_labels = np.array([i % 2 for i in range(N)])
                method_kwargs["net_labels"] = net_labels

            t0 = time.time()
            p = compute_p_val_glm(
                Y, interest=age, confounds=motion,
                method=method_name,
                n_permutations=n_perms,
                use_mp=use_mp,
                random_state=42,
                **method_kwargs,
            )
            elapsed = time.time() - t0

            p_planted = p["positive"][1, 2] if N > 2 else float("nan")
            print(f"{N:>5} {n_edges:>7} {method_name:<10} {elapsed:>7.2f}s {p_planted:>10.4f}")

        print()


def main():
    parser = argparse.ArgumentParser(description="GLM pipeline benchmarks")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--mp", action="store_true", help="Enable multiprocessing")
    args = parser.parse_args()

    if args.quick:
        N_values = [10, 30]
        n_perms = 100
    elif args.full:
        N_values = [10, 30, 50, 90]
        n_perms = 1000
    else:
        N_values = [10, 30, 50]
        n_perms = 500

    n_subjects = 30

    methods = [
        ("tstat", {}),
        ("tfnbs", {"e": 0.4, "h": 3.0, "n": 10}),
        ("nbs", {"threshold": 2.0}),
        ("cnbs", {}),
    ]

    print("GLM Pipeline Benchmark")
    print(f"Network sizes: {N_values}")
    print(f"Subjects: {n_subjects}")
    print(f"Permutations: {n_perms}")
    print(f"Multiprocessing: {args.mp}")

    benchmark_compute_glm_stat(N_values, n_subjects)
    benchmark_full_pipeline(N_values, n_subjects, n_perms, args.mp, methods)


if __name__ == "__main__":
    main()
