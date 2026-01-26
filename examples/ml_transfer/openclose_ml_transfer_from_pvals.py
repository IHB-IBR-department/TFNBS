#!/usr/bin/env python3
"""
ML transfer experiment using saved p-value maps.

Training on IHB, testing on RMET. Edge masks are derived from saved p-maps.

Usage:
  python -m examples.ml_transfer.openclose_ml_transfer_from_pvals \\
    --pvals-dir results/openclose_hcp/pvals \\
    --output-csv results/openclose_hcp/ml_transfer_results.csv

Optional:
  --methods tstat,tfnbs,nbs
  --alphas 0.15,0.1,0.05
  --tune-per-mask
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from examples.ml_transfer.openclose_loader import OpenCloseDataset
from examples.ml_transfer.ml_validation import build_ml_dataset, mask_to_feature_indices


def _parse_list(values: Optional[str], cast=float) -> List:
    if values is None:
        return []
    return [cast(v.strip()) for v in values.split(",") if v.strip()]


def _load_pvals_file(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def _collect_pvals_files(pvals_dir: Path, dataset: str, experiment: str) -> List[Path]:
    pattern = f"pvals_{dataset}_{experiment}_*.pkl"
    return sorted(pvals_dir.glob(pattern))


def _extract_param_list(param_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not param_meta:
        return [{}]
    if "param_triplets" in param_meta:
        return param_meta["param_triplets"]
    if "param_pairs" in param_meta:
        return param_meta["param_pairs"]
    if "e_values" in param_meta and "h_values" in param_meta:
        return [
            {"e": e_val, "h": h_val}
            for e_val, h_val in zip(param_meta["e_values"], param_meta["h_values"])
        ]
    if "threshold_values" in param_meta and "nbs_stats" in param_meta:
        param_list = []
        for nbs_stat in param_meta["nbs_stats"]:
            for threshold in param_meta["threshold_values"]:
                param_list.append({"threshold": threshold, "nbs_stat": nbs_stat})
        return param_list
    return [{}]


def _mask_from_pvals(p_vals: Dict[str, np.ndarray], alpha: float) -> np.ndarray:
    p_pos = p_vals["g2>g1"]
    p_neg = p_vals["g1>g2"]
    p_union = np.minimum(p_pos, p_neg)
    mask = p_union < alpha
    mask = mask | mask.T
    np.fill_diagonal(mask, False)
    return mask


def _cv_metrics(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_indices: np.ndarray,
    C: float,
    n_splits: int,
) -> Tuple[float, float]:
    if len(feature_indices) == 0:
        return 0.5, 0.5

    gkf = GroupKFold(n_splits=n_splits)
    acc_scores = []
    auc_scores = []

    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx][:, feature_indices])
        X_val = scaler.transform(X[val_idx][:, feature_indices])

        lr = LogisticRegression(C=C, solver="liblinear", max_iter=1000)
        lr.fit(X_train, y[train_idx])

        y_pred = lr.predict(X_val)
        y_prob = lr.predict_proba(X_val)[:, 1]

        acc_scores.append(accuracy_score(y[val_idx], y_pred))
        auc_scores.append(roc_auc_score(y[val_idx], y_prob))

    return (
        float(np.mean(acc_scores)),
        float(np.mean(auc_scores)),
    )


def _fit_test_metrics(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_indices: np.ndarray,
    C: float,
) -> Tuple[float, float]:
    if len(feature_indices) == 0:
        return 0.5, 0.5

    scaler = StandardScaler()
    X_train_sel = scaler.fit_transform(X_train[:, feature_indices])
    X_test_sel = scaler.transform(X_test[:, feature_indices])

    lr = LogisticRegression(C=C, solver="liblinear", max_iter=1000)
    lr.fit(X_train_sel, y_train)

    y_pred = lr.predict(X_test_sel)
    y_prob = lr.predict_proba(X_test_sel)[:, 1]

    return (
        accuracy_score(y_test, y_pred),
        roc_auc_score(y_test, y_prob),
    )


def _tune_C(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    C_values: List[float],
    n_splits: int,
) -> float:
    gkf = GroupKFold(n_splits=n_splits)
    best_C = C_values[0]
    best_score = -1.0

    for C in C_values:
        scores = []
        for train_idx, val_idx in gkf.split(X, y, groups=groups):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_val = scaler.transform(X[val_idx])

            lr = LogisticRegression(C=C, solver="liblinear", max_iter=1000)
            lr.fit(X_train, y[train_idx])

            scores.append(accuracy_score(y[val_idx], lr.predict(X_val)))

        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score = mean_score
            best_C = C

    return best_C


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pvals-dir", type=Path, default=Path("results/openclose_hcp/pvals"))
    parser.add_argument("--dataset", type=str, default="hcp")
    parser.add_argument("--source-experiment", type=str, default="ihb")
    parser.add_argument("--target-experiment", type=str, default="rmet")
    parser.add_argument("--methods", type=str, default=None,
                        help="Comma-separated method keys to include (e.g., tstat,tfnbs,nbs).")
    parser.add_argument("--alphas", type=str, default="0.15, 0.1,0.05")
    parser.add_argument("--c-values", type=str, default="1e-4,1e-3,1e-2,1e-1,1.0")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--tune-per-mask", action="store_true")
    parser.add_argument("--output-csv", type=Path, default=Path("results/openclose_hcp/ml_transfer_results.csv"))
    args = parser.parse_args()

    methods_filter = None
    if args.methods:
        methods_filter = {m.strip() for m in args.methods.split(",") if m.strip()}

    alpha_values = _parse_list(args.alphas, float)
    C_values = _parse_list(args.c_values, float)

    if args.dataset == "sch200":
        loader = OpenCloseDataset.schaefer200
    else:
        loader = OpenCloseDataset.hcp

    ds_source = loader(args.source_experiment)
    ds_target = loader(args.target_experiment)

    open_source, close_source = ds_source.fisher_z()
    open_target, close_target = ds_target.fisher_z()

    X_source, y_source, groups_source, tri = build_ml_dataset(open_source, close_source)
    X_target, y_target, _, _ = build_ml_dataset(open_target, close_target)

    output_rows = []

    # Baseline on all edges
    baseline_C = _tune_C(X_source, y_source, groups_source, C_values, args.n_splits)
    base_acc, base_auc = _cv_metrics(
        X_source, y_source, groups_source, np.arange(X_source.shape[1]), baseline_C, args.n_splits
    )
    test_acc, test_auc = _fit_test_metrics(
        X_source, y_source, X_target, y_target, np.arange(X_source.shape[1]), baseline_C
    )

    output_rows.append({
        "method": "baseline_all_edges",
        "method_key": "baseline_all_edges",
        "param_idx": 0,
        "param_json": "{}",
        "alpha": None,
        "n_edges": int(X_source.shape[1]),
        "ihb_cv_accuracy": base_acc,
        "ihb_cv_roc_auc": base_auc,
        "rmet_accuracy": test_acc,
        "rmet_roc_auc": test_auc,
        "C": baseline_C,
        "pmap_file": "baseline",
    })

    pvals_dir = args.pvals_dir
    pvals_files = _collect_pvals_files(pvals_dir, args.dataset, args.source_experiment)

    for pvals_path in pvals_files:
        data = _load_pvals_file(pvals_path)
        method_key = data.get("method_key") or data.get("method") or pvals_path.stem
        method = data.get("method", method_key)

        if methods_filter and method_key not in methods_filter and method not in methods_filter:
            continue

        p_vals = data.get("p_vals")
        if p_vals is None:
            continue

        param_meta = data.get("param_meta", {})
        param_list = _extract_param_list(param_meta)

        p_pos = p_vals["g2>g1"]
        p_neg = p_vals["g1>g2"]

        if p_pos.ndim == 2:
            p_pos = p_pos[..., np.newaxis]
            p_neg = p_neg[..., np.newaxis]

        for param_idx in range(p_pos.shape[-1]):
            p_slice = {
                "g2>g1": p_pos[..., param_idx],
                "g1>g2": p_neg[..., param_idx],
            }
            param_info = param_list[param_idx] if param_idx < len(param_list) else {}
            param_json = json.dumps(param_info, sort_keys=True)

            for alpha in alpha_values:
                mask = _mask_from_pvals(p_slice, alpha)
                feature_indices = mask_to_feature_indices(mask, tri)

                if args.tune_per_mask:
                    if len(feature_indices) == 0:
                        C = baseline_C
                    else:
                        C = _tune_C(
                            X_source[:, feature_indices],
                            y_source,
                            groups_source,
                            C_values,
                            args.n_splits,
                        )
                else:
                    C = baseline_C

                cv_acc, cv_auc = _cv_metrics(
                    X_source, y_source, groups_source, feature_indices, C, args.n_splits
                )
                test_acc, test_auc = _fit_test_metrics(
                    X_source, y_source, X_target, y_target, feature_indices, C
                )

                output_rows.append({
                    "method": method,
                    "method_key": method_key,
                    "param_idx": param_idx,
                    "param_json": param_json,
                    "alpha": alpha,
                    "n_edges": int(len(feature_indices)),
                    "ihb_cv_accuracy": cv_acc,
                    "ihb_cv_roc_auc": cv_auc,
                    "rmet_accuracy": test_acc,
                    "rmet_roc_auc": test_auc,
                    "C": C,
                    "pmap_file": str(pvals_path),
                })

    df = pd.DataFrame(output_rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved: {args.output_csv}")


if __name__ == "__main__":
    main()
