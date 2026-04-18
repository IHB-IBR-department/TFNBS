"""
Benchmark: GPD/gamma permutation acceleration vs full empirical.

Compares timing and p-value accuracy across:
- Network sizes: N=10, 30, 50, 90
- Enhancement methods: tstat, tfnbs, nbs
- Acceleration: none, gpd, gamma
- Permutation counts: 200 (accelerated) vs 5000 (reference)

Usage:
    python examples/benchmarks/benchmark_acceleration.py
    python examples/benchmarks/benchmark_acceleration.py --quick     # N=10,30 only
    python examples/benchmarks/benchmark_acceleration.py --full      # includes N=90
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from conninfpy.glm_stats import compute_p_val_glm

# Sibling benchmark helpers
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import PAPER_TOPOLOGIES, make_glm_data  # noqa: E402


def run_benchmark(N, n_subjects, methods, n_perm_ref, n_perm_accel,
                  topology="within_module_dense", seed=42, records=None):
    """Run benchmark for one network size × topology."""
    effect_edge = (1, 2)
    Y, age, _ = make_glm_data(
        N=N, n_sub=n_subjects, topology=topology,
        effect_edge=effect_edge, effect_beta=1.5,
        seed=seed,
    )
    n_edges = N * (N - 1) // 2
    triu = np.triu_indices(N, k=1)

    print(f"\n{'='*70}")
    print(f"Network: {N}x{N} ({n_edges} edges), {n_subjects} subjects, topology={topology}")
    print(f"Reference: {n_perm_ref} perms | Accelerated: {n_perm_accel} perms")
    print(f"{'='*70}")

    header = f"{'Method':<10} {'Accel':<8} {'Perms':>6} {'Time':>8} {'Speedup':>8} {'r(pos)':>8} {'MAE':>8} {'p_plant':>8}"
    print(header)
    print("-" * len(header))

    for method_name, method_kwargs in methods:
        # Reference: full permutations, no acceleration
        t0 = time.time()
        p_ref = compute_p_val_glm(
            Y, interest=age, method=method_name,
            n_permutations=n_perm_ref, use_mp=False, random_state=seed,
            **method_kwargs,
        )
        t_ref = time.time() - t0
        p_planted_ref = p_ref["positive"][effect_edge]

        print(f"{method_name:<10} {'none':<8} {n_perm_ref:>6} {t_ref:>7.2f}s {'1.0x':>8} {'ref':>8} {'ref':>8} {p_planted_ref:>8.4f}")
        if records is not None:
            records.append(dict(
                N=N, n_subjects=n_subjects, topology=topology, method=method_name,
                accel="none", n_perms=n_perm_ref, time_s=t_ref, speedup=1.0,
                r_pos=float("nan"), mae=float("nan"), p_planted=p_planted_ref,
            ))

        # Accelerated variants
        for accel_name in ["gpd", "gamma"]:
            t0 = time.time()
            p_accel = compute_p_val_glm(
                Y, interest=age, method=method_name,
                n_permutations=n_perm_accel, use_mp=False, random_state=seed,
                acceleration=accel_name, **method_kwargs,
            )
            t_accel = time.time() - t0

            r = np.corrcoef(
                p_ref["positive"][triu], p_accel["positive"][triu]
            )[0, 1]
            mae = np.mean(
                np.abs(p_ref["positive"][triu] - p_accel["positive"][triu])
            )
            p_planted = p_accel["positive"][effect_edge]
            speedup = t_ref / t_accel

            print(f"{'':<10} {accel_name:<8} {n_perm_accel:>6} {t_accel:>7.2f}s {speedup:>7.1f}x {r:>8.4f} {mae:>8.4f} {p_planted:>8.4f}")
            if records is not None:
                records.append(dict(
                    N=N, n_subjects=n_subjects, topology=topology, method=method_name,
                    accel=accel_name, n_perms=n_perm_accel, time_s=t_accel, speedup=speedup,
                    r_pos=r, mae=mae, p_planted=p_planted,
                ))

        # Empirical with same few perms (to compare: is acceleration better?)
        t0 = time.time()
        p_emp = compute_p_val_glm(
            Y, interest=age, method=method_name,
            n_permutations=n_perm_accel, use_mp=False, random_state=seed,
            **method_kwargs,
        )
        t_emp = time.time() - t0

        r = np.corrcoef(
            p_ref["positive"][triu], p_emp["positive"][triu]
        )[0, 1]
        mae = np.mean(
            np.abs(p_ref["positive"][triu] - p_emp["positive"][triu])
        )
        p_planted = p_emp["positive"][effect_edge]
        speedup = t_ref / t_emp

        print(f"{'':<10} {'emp':<8} {n_perm_accel:>6} {t_emp:>7.2f}s {speedup:>7.1f}x {r:>8.4f} {mae:>8.4f} {p_planted:>8.4f}")
        if records is not None:
            records.append(dict(
                N=N, n_subjects=n_subjects, topology=topology, method=method_name,
                accel="emp", n_perms=n_perm_accel, time_s=t_emp, speedup=speedup,
                r_pos=r, mae=mae, p_planted=p_planted,
            ))

        print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark acceleration methods")
    parser.add_argument("--quick", action="store_true", help="Quick run (N=10,30)")
    parser.add_argument("--full", action="store_true", help="Full run (includes N=90)")
    parser.add_argument(
        "--topologies", nargs="+", default=["within_module_dense"],
        help=f"Topology scenarios to sweep (default: within_module_dense). "
             f"The 4 paper topologies: {list(PAPER_TOPOLOGIES)}",
    )
    parser.add_argument(
        "--output", type=str,
        default="examples/benchmarks/res_benchmarks/acceleration.csv",
        help="Path to save CSV results (default: res_benchmarks/acceleration.csv). "
             "Pass empty to skip.",
    )
    args = parser.parse_args()

    if args.quick:
        network_sizes = [10, 30]
    elif args.full:
        network_sizes = [10, 30, 50, 90]
    else:
        network_sizes = [10, 30, 50]

    methods = [
        ("tstat", {}),
        ("tfnbs", {"e": 0.4, "h": 3.0, "n": 10}),
        ("nbs", {"threshold": 2.0}),
    ]

    n_subjects = 30
    n_perm_ref = 5000
    n_perm_accel = 200

    print("Permutation Acceleration Benchmark")
    print(f"Methods: {[m[0] for m in methods]}")
    print(f"Network sizes: {network_sizes}")
    print(f"Topologies: {args.topologies}")
    print(f"Reference perms: {n_perm_ref}, Accelerated perms: {n_perm_accel}")

    records: list = []
    for N in network_sizes:
        for topo in args.topologies:
            run_benchmark(N, n_subjects, methods, n_perm_ref, n_perm_accel,
                          topology=topo, records=records)

    if args.output and records:
        import pandas as pd
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(out, index=False)
        print(f"\nResults saved to {out}")

    print("\n" + "=" * 70)
    print("KEY: r(pos) = correlation of positive p-values with reference")
    print("     MAE = mean absolute error vs reference p-values")
    print("     p_plant = p-value at planted edge (1,2)")
    print("     Speedup = time(reference) / time(accelerated)")
    print()
    print("NOTE: Speedup is ~linear (n_perm_ref / n_perm_accel).")
    print("GPD/gamma advantage over empirical is in p-value PRECISION,")
    print("not speed — they resolve small p-values better with few perms.")


if __name__ == "__main__":
    main()
