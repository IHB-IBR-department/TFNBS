"""
Exp 1: Compute 95% confidence intervals for TPR/FDR from existing results.

Reads per-scenario CSVs from examples/simulation_validation/results/power_analysis_1000_30/per_scenario/
and outputs summary tables with mean, SE, 95% CI (normal + bootstrap).

Usage:
    python examples/simulation_validation/sensitivity/compute_confidence_intervals.py \
        --results-dir examples/simulation_validation/results/power_analysis_1000_30 \
        --output-dir examples/simulation_validation/results/confidence_intervals
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def compute_ci_normal(values, confidence=0.95):
    """Compute normal-approximation CI: mean +/- z * SE."""
    n = len(values)
    mean = np.mean(values)
    se = np.std(values, ddof=1) / np.sqrt(n) if n > 1 else 0.0
    z = 1.96 if confidence == 0.95 else 2.576
    return mean, se, mean - z * se, mean + z * se


def compute_ci_bootstrap(values, n_bootstrap=10000, confidence=0.95, seed=42):
    """Compute bootstrap percentile CI."""
    rng = np.random.RandomState(seed)
    n = len(values)
    if n < 2:
        m = np.mean(values)
        return m, 0.0, m, m
    boot_means = np.array([
        np.mean(rng.choice(values, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = (1 - confidence) / 2
    lo = np.percentile(boot_means, 100 * alpha)
    hi = np.percentile(boot_means, 100 * (1 - alpha))
    return np.mean(values), np.std(boot_means), lo, hi


def process_csv(df, groupby_cols, metrics=("TPR", "FDR")):
    """Group by columns and compute CIs for each metric."""
    rows = []
    for group_keys, grp in df.groupby(groupby_cols):
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)
        row = dict(zip(groupby_cols, group_keys))
        row["n_repeats"] = len(grp)

        for metric in metrics:
            vals = grp[metric].values
            mean, se, ci_lo, ci_hi = compute_ci_normal(vals)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_se"] = se
            row[f"{metric}_ci95_lo"] = ci_lo
            row[f"{metric}_ci95_hi"] = ci_hi

            _, _, boot_lo, boot_hi = compute_ci_bootstrap(vals)
            row[f"{metric}_boot_ci95_lo"] = boot_lo
            row[f"{metric}_boot_ci95_hi"] = boot_hi

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Compute CIs for power analysis results")
    parser.add_argument(
        "--results-dir", type=Path,
        default=Path("examples/simulation_validation/results/power_analysis_1000_30"),
        help="Directory containing per_scenario/ CSVs",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("examples/simulation_validation/results/confidence_intervals"),
        help="Output directory for CI tables",
    )
    args = parser.parse_args()

    scenario_dir = args.results_dir / "per_scenario"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Process effect-size CSVs
    es_csvs = sorted(scenario_dir.glob("power_vs_effect_size_*.csv"))
    if es_csvs:
        dfs = [pd.read_csv(p) for p in es_csvs]
        df_all = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(dfs)} effect-size CSVs ({len(df_all)} rows)")

        ci_df = process_csv(
            df_all,
            groupby_cols=["method", "method_params", "scenario", "effect_size"],
        )
        out_path = args.output_dir / "ci_effect_size.csv"
        ci_df.to_csv(out_path, index=False)
        print(f"Saved {out_path} ({len(ci_df)} rows)")

    # Process sample-size CSVs
    ss_csvs = sorted(scenario_dir.glob("power_vs_sample_size_*.csv"))
    if ss_csvs:
        dfs = [pd.read_csv(p) for p in ss_csvs]
        df_all = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(dfs)} sample-size CSVs ({len(df_all)} rows)")

        ci_df = process_csv(
            df_all,
            groupby_cols=["method", "method_params", "scenario", "n_samples"],
        )
        out_path = args.output_dir / "ci_sample_size.csv"
        ci_df.to_csv(out_path, index=False)
        print(f"Saved {out_path} ({len(ci_df)} rows)")

    print("Done.")


if __name__ == "__main__":
    main()
