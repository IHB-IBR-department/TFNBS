from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any

from conninfpy.loaders.base import LoadedDataset, DataValidationReport

def validate_loaded_dataset(dataset: LoadedDataset, check_inference_ready: bool = False) -> DataValidationReport:
    """Validate a LoadedDataset object for structural coherence and mathematical validity.
    
    Parameters
    ----------
    dataset : LoadedDataset
        The loaded dataset to validate.
    check_inference_ready : bool
        If True, validates that the dataset is fully preprocessed and ready for statistical inference
        (e.g., matrices are square, symmetric, data_kind is correlation/fisher_z).
        If False, validates the raw loaded format (which might be raw timeseries).
        
    Returns
    -------
    DataValidationReport
        Contains the pass status (ok), list of error strings, list of warning strings, and summary info.
    """
    errors: list[str] = []
    warnings: list[str] = []
    summary: dict[str, Any] = {}

    data = dataset.data
    pheno = dataset.pheno
    data_kind = dataset.data_kind

    # Basic type checks
    if not isinstance(data, np.ndarray):
        errors.append("dataset.data must be a numpy ndarray.")
        return DataValidationReport(ok=False, errors=errors, warnings=warnings, summary={})

    if not isinstance(pheno, pd.DataFrame):
        errors.append("dataset.pheno must be a pandas DataFrame.")
        return DataValidationReport(ok=False, errors=errors, warnings=warnings, summary={})

    if data.ndim != 3:
        errors.append(f"dataset.data must be 3D (got ndim={data.ndim}, shape={data.shape}).")
        return DataValidationReport(ok=False, errors=errors, warnings=warnings, summary={})

    n_observations = data.shape[0]
    n_pheno_rows = len(pheno)

    if n_observations != n_pheno_rows:
        errors.append(f"Subject count mismatch: data has {n_observations} observations, but pheno has {n_pheno_rows} rows.")

    # Check numeric type before calling np.isfinite(), whose error for object
    # arrays hides the real ingestion problem (usually a CSV header mismatch).
    if not np.issubdtype(data.dtype, np.number):
        errors.append(
            f"Data must be numeric, got dtype {data.dtype}. Check CSV header and separator settings."
        )
    elif not np.isfinite(data).all():
        errors.append("Data contains missing, infinite, or NaN values.")

    # Retrieve subject IDs if present
    subj_ids = dataset.subject_ids
    if subj_ids is not None:
        if len(subj_ids) != n_observations:
            errors.append(f"Length of subject_ids ({len(subj_ids)}) does not match data shape ({n_observations}).")
        
        # Check uniqueness of subject IDs
        # Allow repeats ONLY if there is a condition_column declared and valid
        if len(set(subj_ids)) < len(subj_ids):
            if not dataset.condition_column or dataset.condition_column not in pheno.columns:
                warnings.append("Duplicate subject IDs detected, but no valid condition_column is specified.")
            else:
                # Validate that combinations of subject_id + condition are unique
                combos = list(zip(subj_ids, pheno[dataset.condition_column]))
                if len(set(combos)) < len(combos):
                    errors.append("Duplicate subject IDs found within the same condition.")

    # Validation based on kind
    if data_kind == "timeseries":
        n_timepoints = data.shape[1]
        n_nodes = data.shape[2]
        summary["n_timepoints"] = n_timepoints
        summary["n_rois"] = n_nodes
        
        if n_timepoints < 10:
            warnings.append(f"Very short timeseries detected ({n_timepoints} timepoints). Correlation estimation may be unstable.")
            
        # Check for constant timeseries (zero variance)
        for i in range(n_observations):
            var = np.var(data[i], axis=0)
            if np.any(var == 0):
                zero_nodes = np.where(var == 0)[0]
                warnings.append(f"Observation {i} has constant timeseries (zero variance) in ROIs: {zero_nodes.tolist()}.")
                
    elif data_kind in {"correlation", "fisher_z"}:
        n_nodes_i = data.shape[1]
        n_nodes_j = data.shape[2]
        summary["n_rois"] = n_nodes_i

        if n_nodes_i != n_nodes_j:
            errors.append(f"Connectivity matrices are not square: got shape ({n_nodes_i}, {n_nodes_j}).")
        else:
            # Check symmetry (within numerical precision)
            for i in range(n_observations):
                matrix = data[i]
                if not np.allclose(matrix, matrix.T, atol=1e-5):
                    errors.append(f"Connectivity matrix for observation {i} is not symmetric.")
                    break
    else:
        errors.append(f"Unknown data_kind: {data_kind}. Must be 'timeseries', 'correlation', or 'fisher_z'.")

    # If checking inference readiness, enforce that it is connectivity and contains no NaNs
    if check_inference_ready:
        if data_kind not in {"correlation", "fisher_z"}:
            errors.append(f"Inference-ready dataset must be 'correlation' or 'fisher_z', not {data_kind}.")
        if data.shape[1] != data.shape[2]:
            errors.append("Inference-ready connectivity matrices must be square.")

    # Check atlas matching if atlas is attached
    if dataset.atlas is not None:
        expected_nodes = len(dataset.atlas)
        actual_nodes = summary.get("n_rois")
        if actual_nodes is not None and actual_nodes != expected_nodes:
            errors.append(f"Atlas/Node count mismatch: atlas has {expected_nodes} labels, but data has {actual_nodes} nodes.")

    summary["n_observations"] = n_observations
    summary["data_kind"] = data_kind
    summary["ok"] = len(errors) == 0

    return DataValidationReport(
        ok=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        summary=summary
    )
