"""
ML validation of statistical edge-detection methods.

Goal: Validate whether edges selected by stats methods (compute_p_val)
provide measurable benefit to a simple ML classifier.

Key principle: ML does NOT generate edge p-values. Edge subsets come
from statistical p-value maps produced by compute_p_val.

Usage:
    from ml_validation import (
        build_edge_mask_from_pvals,
        grouped_cv_with_edge_selection,
        ml_transfer_experiment,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Edge mask construction from stats p-maps
# =============================================================================


def build_edge_mask_from_pvals(
    p_vals: Dict[str, np.ndarray],
    alpha: float = 0.05,
    mode: Literal["union", "pos", "neg"] = "union",
) -> np.ndarray:
    """
    Build edge selection mask from compute_p_val output.

    Parameters
    ----------
    p_vals : dict
        Output from compute_p_val with keys "g2>g1" (close>open) and "g1>g2" (open>close)
    alpha : float
        Significance threshold
    mode : str
        "union": select edges significant in either direction (p_union = min(p_pos, p_neg))
        "pos": only close > open edges
        "neg": only open > close edges

    Returns
    -------
    mask : (n_nodes, n_nodes) boolean array
    """
    p_pos = p_vals["g2>g1"]  # close > open
    p_neg = p_vals["g1>g2"]  # open > close

    if mode == "union":
        p_union = np.minimum(p_pos, p_neg)
        mask = p_union < alpha
    elif mode == "pos":
        mask = p_pos < alpha
    elif mode == "neg":
        mask = p_neg < alpha
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Ensure symmetric and zero diagonal
    mask = mask | mask.T
    np.fill_diagonal(mask, False)

    return mask


def mask_to_feature_indices(
    mask: np.ndarray,
    tri: Tuple[np.ndarray, np.ndarray] = None,
) -> np.ndarray:
    """
    Convert 2D mask to indices into upper-triangle feature vector.

    Parameters
    ----------
    mask : (n_nodes, n_nodes) boolean
    tri : tuple, optional
        Pre-computed triu_indices

    Returns
    -------
    indices : 1D array of feature indices to keep
    """
    n_nodes = mask.shape[0]
    if tri is None:
        tri = np.triu_indices(n_nodes, k=1)

    mask_upper = mask[tri]
    return np.where(mask_upper)[0]


# =============================================================================
# Data preparation
# =============================================================================


def build_ml_dataset(
    open_data: np.ndarray,
    close_data: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple]:
    """
    Build ML dataset from open/close connectivity matrices.

    Parameters
    ----------
    open_data : (n_subjects, n_nodes, n_nodes)
    close_data : (n_subjects, n_nodes, n_nodes)

    Returns
    -------
    X : (2*n_subjects, n_features) feature matrix
    y : (2*n_subjects,) labels (0=open, 1=close)
    subject_ids : (2*n_subjects,) subject grouping for GroupKFold
    tri : tuple of indices for reconstructing matrices
    """
    n_subjects = open_data.shape[0]
    n_nodes = open_data.shape[1]
    tri = np.triu_indices(n_nodes, k=1)

    # Vectorize upper triangle
    X_open = open_data[:, tri[0], tri[1]]
    X_close = close_data[:, tri[0], tri[1]]

    # Stack: first all open, then all close
    X = np.vstack([X_open, X_close])
    y = np.array([0] * n_subjects + [1] * n_subjects)

    # Subject IDs for grouping (same subject appears twice: once open, once close)
    subject_ids = np.concatenate([np.arange(n_subjects), np.arange(n_subjects)])

    return X, y, subject_ids, tri


# =============================================================================
# Grouped cross-validation utilities
# =============================================================================


def tune_regularization(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    C_values: List[float] = None,
    n_splits: int = 5,
) -> Tuple[float, Dict[float, float]]:
    """
    Tune LR regularization parameter C via GroupKFold CV.

    Parameters
    ----------
    X, y, subject_ids : arrays
    C_values : list of float
        C values to try (default: [1e-4, 1e-3, 1e-2, 1e-1, 1.0])
    n_splits : int

    Returns
    -------
    best_C : float
    results : dict mapping C -> mean balanced accuracy
    """
    if C_values is None:
        C_values = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]

    gkf = GroupKFold(n_splits=n_splits)
    results = {}

    for C in C_values:
        scores = []
        for train_idx, val_idx in gkf.split(X, y, groups=subject_ids):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_val = scaler.transform(X[val_idx])

            lr = LogisticRegression(C=C, solver="liblinear", max_iter=1000)
            lr.fit(X_train, y[train_idx])
            y_pred = lr.predict(X_val)
            scores.append(balanced_accuracy_score(y[val_idx], y_pred))

        results[C] = float(np.mean(scores))

    best_C = max(results, key=results.get)
    return best_C, results


@dataclass
class CVFoldResult:
    """Results from a single CV fold."""

    n_edges_selected: int
    accuracy: float
    balanced_accuracy: float
    roc_auc: float


@dataclass
class CVResult:
    """Aggregated CV results."""

    mean_n_edges: float
    std_n_edges: float
    mean_accuracy: float
    std_accuracy: float
    mean_balanced_accuracy: float
    std_balanced_accuracy: float
    mean_roc_auc: float
    std_roc_auc: float
    fold_results: List[CVFoldResult]


def grouped_cv_with_edge_selection(
    open_data: np.ndarray,
    close_data: np.ndarray,
    compute_pvals_fn: Callable,
    alpha: float = 0.05,
    C: float = 1e-3,
    n_splits: int = 5,
    mask_mode: Literal["union", "pos", "neg"] = "union",
) -> CVResult:
    """
    GroupKFold CV with edge selection inside each fold.

    IMPORTANT: p-maps are computed on fold-train only to avoid leakage.

    Parameters
    ----------
    open_data : (n_subjects, n_nodes, n_nodes)
    close_data : (n_subjects, n_nodes, n_nodes)
    compute_pvals_fn : callable
        Function (open_train, close_train) -> p_vals dict
        This should call compute_p_val with the desired method
    alpha : float
        Significance threshold for edge selection
    C : float
        LR regularization
    n_splits : int
    mask_mode : str
        "union", "pos", or "neg"

    Returns
    -------
    CVResult
    """
    n_subjects = open_data.shape[0]
    n_nodes = open_data.shape[1]
    tri = np.triu_indices(n_nodes, k=1)

    # Build full feature matrix
    X, y, subject_ids, _ = build_ml_dataset(open_data, close_data)

    gkf = GroupKFold(n_splits=n_splits)
    fold_results = []

    for train_idx, val_idx in gkf.split(X, y, groups=subject_ids):
        # Get subject indices for train fold
        train_subjects = np.unique(subject_ids[train_idx])

        # Extract train data in matrix form for compute_p_val
        open_train = open_data[train_subjects]
        close_train = close_data[train_subjects]

        # Compute p-maps on train fold only
        p_vals = compute_pvals_fn(open_train, close_train)

        # Build edge mask
        mask = build_edge_mask_from_pvals(p_vals, alpha=alpha, mode=mask_mode)
        feature_indices = mask_to_feature_indices(mask, tri)
        n_edges = len(feature_indices)

        if n_edges == 0:
            # No edges selected - use chance performance
            fold_results.append(CVFoldResult(
                n_edges_selected=0,
                accuracy=0.5,
                balanced_accuracy=0.5,
                roc_auc=0.5,
            ))
            continue

        # Select features
        X_train_selected = X[train_idx][:, feature_indices]
        X_val_selected = X[val_idx][:, feature_indices]

        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_val_scaled = scaler.transform(X_val_selected)

        # Train and evaluate
        lr = LogisticRegression(C=C, solver="liblinear", max_iter=1000)
        lr.fit(X_train_scaled, y[train_idx])

        y_pred = lr.predict(X_val_scaled)
        y_prob = lr.predict_proba(X_val_scaled)[:, 1]

        fold_results.append(CVFoldResult(
            n_edges_selected=n_edges,
            accuracy=accuracy_score(y[val_idx], y_pred),
            balanced_accuracy=balanced_accuracy_score(y[val_idx], y_pred),
            roc_auc=roc_auc_score(y[val_idx], y_prob),
        ))

    # Aggregate
    n_edges_list = [r.n_edges_selected for r in fold_results]
    acc_list = [r.accuracy for r in fold_results]
    bacc_list = [r.balanced_accuracy for r in fold_results]
    auc_list = [r.roc_auc for r in fold_results]

    return CVResult(
        mean_n_edges=float(np.mean(n_edges_list)),
        std_n_edges=float(np.std(n_edges_list)),
        mean_accuracy=float(np.mean(acc_list)),
        std_accuracy=float(np.std(acc_list)),
        mean_balanced_accuracy=float(np.mean(bacc_list)),
        std_balanced_accuracy=float(np.std(bacc_list)),
        mean_roc_auc=float(np.mean(auc_list)),
        std_roc_auc=float(np.std(auc_list)),
        fold_results=fold_results,
    )


# =============================================================================
# Full transfer experiment
# =============================================================================


@dataclass
class TransferResult:
    """Results from IHB -> RMET transfer experiment."""

    method: str
    alpha: float
    n_edges_selected: int
    ihb_cv_balanced_accuracy: float
    ihb_cv_roc_auc: float
    rmet_accuracy: float
    rmet_balanced_accuracy: float
    rmet_roc_auc: float
    feature_mask: np.ndarray  # (n_nodes, n_nodes) boolean


def ml_transfer_experiment(
    open_ihb: np.ndarray,
    close_ihb: np.ndarray,
    open_rmet: np.ndarray,
    close_rmet: np.ndarray,
    compute_pvals_fn: Callable,
    method_name: str = "method",
    alpha: float = 0.05,
    C: float = 1e-3,
    mask_mode: Literal["union", "pos", "neg"] = "union",
    n_cv_splits: int = 5,
    verbose: bool = True,
) -> TransferResult:
    """
    Full ML transfer experiment: train on IHB, test on RMET.

    Protocol:
    1. Compute p-maps on full IHB
    2. Select edges at alpha threshold
    3. Train LR on IHB using selected edges
    4. Evaluate on RMET

    Also runs CV on IHB for internal validation.

    Parameters
    ----------
    open_ihb, close_ihb : (n_ihb, n_nodes, n_nodes)
    open_rmet, close_rmet : (n_rmet, n_nodes, n_nodes)
    compute_pvals_fn : callable
        (open, close) -> p_vals dict
    method_name : str
        Name for reporting
    alpha : float
    C : float
    mask_mode : str
    n_cv_splits : int
    verbose : bool

    Returns
    -------
    TransferResult
    """
    n_nodes = open_ihb.shape[1]
    tri = np.triu_indices(n_nodes, k=1)

    if verbose:
        print(f"=== {method_name} @ alpha={alpha} ===")

    # 1. Run CV on IHB for internal validation
    if verbose:
        print("  Running IHB cross-validation...")

    cv_result = grouped_cv_with_edge_selection(
        open_ihb, close_ihb,
        compute_pvals_fn=compute_pvals_fn,
        alpha=alpha, C=C, n_splits=n_cv_splits, mask_mode=mask_mode,
    )

    if verbose:
        print(f"  IHB CV: bal_acc={cv_result.mean_balanced_accuracy:.3f} "
              f"(±{cv_result.std_balanced_accuracy:.3f}), "
              f"edges={cv_result.mean_n_edges:.0f}")

    # 2. Compute p-maps on full IHB
    if verbose:
        print("  Computing p-maps on full IHB...")

    p_vals = compute_pvals_fn(open_ihb, close_ihb)

    # 3. Build mask
    mask = build_edge_mask_from_pvals(p_vals, alpha=alpha, mode=mask_mode)
    feature_indices = mask_to_feature_indices(mask, tri)
    n_edges = len(feature_indices)

    if verbose:
        print(f"  Selected {n_edges} edges")

    if n_edges == 0:
        if verbose:
            print("  WARNING: No edges selected, returning chance performance")
        return TransferResult(
            method=method_name,
            alpha=alpha,
            n_edges_selected=0,
            ihb_cv_balanced_accuracy=cv_result.mean_balanced_accuracy,
            ihb_cv_roc_auc=cv_result.mean_roc_auc,
            rmet_accuracy=0.5,
            rmet_balanced_accuracy=0.5,
            rmet_roc_auc=0.5,
            feature_mask=mask,
        )

    # 4. Build datasets
    X_ihb, y_ihb, _, _ = build_ml_dataset(open_ihb, close_ihb)
    X_rmet, y_rmet, _, _ = build_ml_dataset(open_rmet, close_rmet)

    # Select features
    X_ihb_selected = X_ihb[:, feature_indices]
    X_rmet_selected = X_rmet[:, feature_indices]

    # 5. Standardize (fit on IHB only)
    scaler = StandardScaler()
    X_ihb_scaled = scaler.fit_transform(X_ihb_selected)
    X_rmet_scaled = scaler.transform(X_rmet_selected)

    # 6. Train on IHB
    lr = LogisticRegression(C=C, solver="liblinear", max_iter=1000)
    lr.fit(X_ihb_scaled, y_ihb)

    # 7. Evaluate on RMET
    y_pred = lr.predict(X_rmet_scaled)
    y_prob = lr.predict_proba(X_rmet_scaled)[:, 1]

    rmet_acc = accuracy_score(y_rmet, y_pred)
    rmet_bacc = balanced_accuracy_score(y_rmet, y_pred)
    rmet_auc = roc_auc_score(y_rmet, y_prob)

    if verbose:
        print(f"  RMET: acc={rmet_acc:.3f}, bal_acc={rmet_bacc:.3f}, AUC={rmet_auc:.3f}")

    return TransferResult(
        method=method_name,
        alpha=alpha,
        n_edges_selected=n_edges,
        ihb_cv_balanced_accuracy=cv_result.mean_balanced_accuracy,
        ihb_cv_roc_auc=cv_result.mean_roc_auc,
        rmet_accuracy=rmet_acc,
        rmet_balanced_accuracy=rmet_bacc,
        rmet_roc_auc=rmet_auc,
        feature_mask=mask,
    )


def baseline_all_edges(
    open_ihb: np.ndarray,
    close_ihb: np.ndarray,
    open_rmet: np.ndarray,
    close_rmet: np.ndarray,
    C: float = 1e-3,
    n_cv_splits: int = 5,
    verbose: bool = True,
) -> Dict:
    """
    Baseline: LR on all edges (no selection).

    Returns
    -------
    dict with IHB CV and RMET test metrics
    """
    if verbose:
        print("=== Baseline (all edges) ===")

    X_ihb, y_ihb, subj_ihb, tri = build_ml_dataset(open_ihb, close_ihb)
    X_rmet, y_rmet, _, _ = build_ml_dataset(open_rmet, close_rmet)

    n_features = X_ihb.shape[1]

    # CV on IHB
    gkf = GroupKFold(n_splits=n_cv_splits)
    cv_scores = []

    for train_idx, val_idx in gkf.split(X_ihb, y_ihb, groups=subj_ihb):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_ihb[train_idx])
        X_val = scaler.transform(X_ihb[val_idx])

        lr = LogisticRegression(C=C, solver="liblinear", max_iter=1000)
        lr.fit(X_train, y_ihb[train_idx])
        cv_scores.append(balanced_accuracy_score(y_ihb[val_idx], lr.predict(X_val)))

    if verbose:
        print(f"  IHB CV: bal_acc={np.mean(cv_scores):.3f} (±{np.std(cv_scores):.3f})")

    # Train on full IHB, test on RMET
    scaler = StandardScaler()
    X_ihb_scaled = scaler.fit_transform(X_ihb)
    X_rmet_scaled = scaler.transform(X_rmet)

    lr = LogisticRegression(C=C, solver="liblinear", max_iter=1000)
    lr.fit(X_ihb_scaled, y_ihb)

    y_pred = lr.predict(X_rmet_scaled)
    y_prob = lr.predict_proba(X_rmet_scaled)[:, 1]

    rmet_bacc = balanced_accuracy_score(y_rmet, y_pred)
    rmet_auc = roc_auc_score(y_rmet, y_prob)

    if verbose:
        print(f"  RMET: bal_acc={rmet_bacc:.3f}, AUC={rmet_auc:.3f}")
        print(f"  n_features={n_features}")

    return {
        "n_features": n_features,
        "ihb_cv_balanced_accuracy": float(np.mean(cv_scores)),
        "ihb_cv_std": float(np.std(cv_scores)),
        "rmet_balanced_accuracy": rmet_bacc,
        "rmet_roc_auc": rmet_auc,
    }


def sweep_alpha(
    open_ihb: np.ndarray,
    close_ihb: np.ndarray,
    open_rmet: np.ndarray,
    close_rmet: np.ndarray,
    compute_pvals_fn: Callable,
    method_name: str,
    alpha_values: List[float] = None,
    C: float = 1e-3,
    verbose: bool = True,
) -> List[TransferResult]:
    """
    Sweep over alpha values for a single method.

    Parameters
    ----------
    alpha_values : list
        Default: [0.1, 0.05, 0.01, 0.005, 0.001]

    Returns
    -------
    list of TransferResult
    """
    if alpha_values is None:
        alpha_values = [0.1, 0.05, 0.01, 0.005, 0.001]

    results = []
    for alpha in alpha_values:
        result = ml_transfer_experiment(
            open_ihb, close_ihb, open_rmet, close_rmet,
            compute_pvals_fn=compute_pvals_fn,
            method_name=method_name,
            alpha=alpha, C=C, verbose=verbose,
        )
        results.append(result)

    return results


if __name__ == "__main__":
    # Quick test with synthetic data
    import sys
    sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

    rng = np.random.default_rng(42)

    n_ihb, n_rmet, n_nodes = 30, 20, 50

    # Synthetic data with weak signal
    open_ihb = rng.standard_normal((n_ihb, n_nodes, n_nodes))
    close_ihb = open_ihb + rng.standard_normal((n_ihb, n_nodes, n_nodes)) * 0.1
    close_ihb[:, :5, :5] += 0.3  # Add signal in one block

    open_rmet = rng.standard_normal((n_rmet, n_nodes, n_nodes))
    close_rmet = open_rmet + rng.standard_normal((n_rmet, n_nodes, n_nodes)) * 0.1
    close_rmet[:, :5, :5] += 0.3

    # Symmetrize
    for data in [open_ihb, close_ihb, open_rmet, close_rmet]:
        for i in range(data.shape[0]):
            data[i] = (data[i] + data[i].T) / 2
            np.fill_diagonal(data[i], 0)

    # Mock compute_pvals_fn (returns random p-values)
    def mock_compute_pvals(open_data, close_data):
        n = open_data.shape[1]
        p = rng.random((n, n))
        p = (p + p.T) / 2
        return {"g2>g1": p, "g1>g2": p}

    print("Testing baseline...")
    baseline = baseline_all_edges(open_ihb, close_ihb, open_rmet, close_rmet)

    print("\nTesting transfer experiment...")
    result = ml_transfer_experiment(
        open_ihb, close_ihb, open_rmet, close_rmet,
        compute_pvals_fn=mock_compute_pvals,
        method_name="mock",
        alpha=0.1,
    )
    print(f"Result: {result.n_edges_selected} edges, RMET acc={result.rmet_balanced_accuracy:.3f}")
