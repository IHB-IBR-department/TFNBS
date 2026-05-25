"""
Exp 2: Analyze prior misspecification results.

Compares TPR/FDR of cNBS, NI-TFNBS, FBC-TFNBS under:
  - true labels (ground truth, 4 blocks)
  - 7blocks (misspecified, too many)
  - 2blocks (misspecified, too few)
with TFNBS and BH-FDR as label-independent controls.

Usage:
    # After running the 3 prior misspecification sweeps:
    python examples/simulation_validation/sensitivity/analyze_prior_misspec.py \
        --true-dir examples/simulation_validation/results/prior_misspec_true \
        --seven-dir examples/simulation_validation/results/prior_misspec_7blocks \
        --two-dir examples/simulation_validation/results/prior_misspec_2blocks \
        --output-dir examples/simulation_validation/results/prior_misspec_comparison

Run the sweeps with:
    python examples/simulation_validation/power/power_analysis.py \
        --config examples/simulation_validation/configs/sweep_config_prior_misspec.yaml \
        --label-override true --output-dir examples/simulation_validation/results/prior_misspec_true

    python examples/simulation_validation/power/power_analysis.py \
        --config examples/simulation_validation/configs/sweep_config_prior_misspec.yaml \
        --label-override 7blocks --output-dir examples/simulation_validation/results/prior_misspec_7blocks

    python examples/simulation_validation/power/power_analysis.py \
        --config examples/simulation_validation/configs/sweep_config_prior_misspec.yaml \
        --label-override 2blocks --output-dir examples/simulation_validation/results/prior_misspec_2blocks
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CONSTRAINED_METHODS = ["cnbs", "ni_tfnbs", "fbc_tfnbs"]
CONTROL_METHODS = ["tfnbs", "bh_fdr"]
ALL_METHODS = CONSTRAINED_METHODS + CONTROL_METHODS


def load_results(results_dir: Path) -> pd.DataFrame:
    """Load power_vs_effect_size.csv from a results directory."""
    path = results_dir / "power_vs_effect_size.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    return pd.read_csv(path)


def build_comparison_table(
    df_true: pd.DataFrame,
    df_7blocks: pd.DataFrame,
    df_2blocks: pd.DataFrame,
) -> pd.DataFrame:
    """Build a comparison table showing TPR/FDR across label conditions."""
    rows = []

    for label_name, df in [("true (4 blocks)", df_true),
                            ("7 blocks", df_7blocks),
                            ("2 blocks", df_2blocks)]:
        for method in ALL_METHODS:
            sub = df[df["method"] == method]
            if sub.empty:
                continue
            for scenario in sub["scenario"].unique():
                for es in sub["effect_size"].unique():
                    cond = sub[(sub["scenario"] == scenario) &
                               (sub["effect_size"] == es)]
                    if cond.empty:
                        continue
                    rows.append({
                        "label_condition": label_name,
                        "method": method,
                        "scenario": scenario,
                        "effect_size": es,
                        "TPR_mean": cond["TPR"].mean(),
                        "TPR_se": cond["TPR"].std() / np.sqrt(len(cond)),
                        "FDR_mean": cond["FDR"].mean(),
                        "FDR_se": cond["FDR"].std() / np.sqrt(len(cond)),
                        "n_repeats": len(cond),
                    })

    return pd.DataFrame(rows)


def print_degradation_summary(df_comp: pd.DataFrame):
    """Print summary of TPR degradation for constrained methods."""
    print("\n=== Prior Misspecification: TPR Degradation ===\n")

    for method in CONSTRAINED_METHODS:
        sub = df_comp[df_comp["method"] == method]
        if sub.empty:
            continue

        true_rows = sub[sub["label_condition"] == "true (4 blocks)"]
        for _, row7 in sub[sub["label_condition"] == "7 blocks"].iterrows():
            true_match = true_rows[
                (true_rows["scenario"] == row7["scenario"]) &
                (true_rows["effect_size"] == row7["effect_size"])
            ]
            if true_match.empty:
                continue
            true_tpr = true_match.iloc[0]["TPR_mean"]
            delta = row7["TPR_mean"] - true_tpr
            print(f"  {method:12s} | {row7['scenario']:30s} | d={row7['effect_size']:.2f} | "
                  f"TPR: {true_tpr:.3f} -> {row7['TPR_mean']:.3f} (7blk, delta={delta:+.3f})")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze prior misspecification results"
    )
    parser.add_argument("--true-dir", type=Path, required=True)
    parser.add_argument("--seven-dir", type=Path, required=True)
    parser.add_argument("--two-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("examples/simulation_validation/results/prior_misspec_comparison"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df_true = load_results(args.true_dir)
    df_7blocks = load_results(args.seven_dir)
    df_2blocks = load_results(args.two_dir)

    print(f"True: {len(df_true)} rows, 7blocks: {len(df_7blocks)}, 2blocks: {len(df_2blocks)}")

    df_comp = build_comparison_table(df_true, df_7blocks, df_2blocks)
    out_path = args.output_dir / "prior_misspec_comparison.csv"
    df_comp.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(df_comp)} rows)")

    print_degradation_summary(df_comp)

    # Verify controls are stable
    print("\n=== Control Methods (should be stable across conditions) ===\n")
    for method in CONTROL_METHODS:
        sub = df_comp[df_comp["method"] == method]
        if sub.empty:
            continue
        for scenario in sub["scenario"].unique():
            s = sub[sub["scenario"] == scenario]
            tpr_range = s["TPR_mean"].max() - s["TPR_mean"].min()
            fdr_range = s["FDR_mean"].max() - s["FDR_mean"].min()
            print(f"  {method:12s} | {scenario:30s} | "
                  f"TPR range: {tpr_range:.4f}, FDR range: {fdr_range:.4f}")


if __name__ == "__main__":
    main()
