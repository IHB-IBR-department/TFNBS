import os
import yaml
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Union
from conninfpy.loaders.base import BaseDataLoader, LoadedDataset, DatasetPreview

class DatasetManifest:
    def __init__(self, data: Dict[str, Any], manifest_path: str):
        self.data = data
        self.manifest_path = manifest_path
        self.schema_version = data.get("schema_version", 1)
        self.name = data.get("name", "Unnamed Dataset")
        self.loader = data.get("loader")
        self.paths = data.get("paths", {})
        self.metadata = data.get("metadata", {})
        self.params = data.get("params", {})
        self.preprocessing = data.get("preprocessing", {})
        self.checks = data.get("checks", {})

def resolve_path(p: Any, manifest_dir: Path) -> Any:
    if isinstance(p, dict):
        return {k: resolve_path(v, manifest_dir) for k, v in p.items()}
    if isinstance(p, list):
        return [resolve_path(x, manifest_dir) for x in p]
    if isinstance(p, str):
        if p.startswith("BUILTIN:"):
            # Resolve to builtin package directory
            pkg_dir = Path(__file__).resolve().parent.parent
            atlas_dir = pkg_dir / "atlas_data"
            token = p.split(":", 1)[1]
            if token == "bna_246_2mm":
                return str(atlas_dir / "BN_Atlas_246_2mm.nii.gz")
            elif token == "bna_246_1mm":
                return str(atlas_dir / "BN_Atlas_246_1mm.nii.gz")
            elif token == "bna246.csv":
                return str(atlas_dir / "bna246.csv")
            elif token == "schaefer100":
                return str(atlas_dir / "schaefer100_yeo7.csv")
            elif token == "schaefer200":
                return str(atlas_dir / "schaefer200_yeo7.csv")
            elif token == "schaefer400":
                return str(atlas_dir / "schaefer400_yeo7.csv")
            else:
                cand = atlas_dir / token
                if cand.exists():
                    return str(cand)
                # Fallback to check datasets/atlases
                datasets_atlases = pkg_dir / "datasets" / "atlases"
                if (datasets_atlases / token).exists():
                    return str(datasets_atlases / token)
                return p
        
        path_obj = Path(p)
        if path_obj.is_absolute():
            return p
        else:
            return str((manifest_dir / path_obj).resolve())
    return p

def load_manifest(path: str) -> DatasetManifest:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Manifest file not found: {path}")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return DatasetManifest(data, path)

class ManifestLoader(BaseDataLoader):
    """Loader wrapper that parses data.yaml manifests and delegates to real loaders."""
    name = "ManifestLoader"
    
    def __init__(self, manifest_path: str):
        self.manifest_path = manifest_path
        self.manifest = load_manifest(manifest_path)
        
        manifest_dir = Path(manifest_path).resolve().parent
        self.resolved_paths = resolve_path(self.manifest.paths, manifest_dir)
        
        from conninfpy.loaders.builtins import (
            NumpyLoader,
            CSVDirectoryLoader,
            NiftiDirectoryLoader,
            AbideSchaeferLoader,
            OpenCloseLoader,
            StressTimeseriesLoader,
            ZerssenNiftiLoader,
            FmriprepDerivativesLoader,
            TimeseriesDirectoryLoader,
            ConnectivityMatrixLoader,
            ConditionTimeseriesArrayLoader,
            CustomPreparedZerssenLoader,
            ChinaCloseCloseLoader
        )
        
        loader_map = {
            "NumpyLoader": NumpyLoader,
            "CSVDirectoryLoader": CSVDirectoryLoader,
            "NiftiDirectoryLoader": NiftiDirectoryLoader,
            "AbideSchaeferLoader": AbideSchaeferLoader,
            "OpenCloseLoader": OpenCloseLoader,
            "StressTimeseriesLoader": StressTimeseriesLoader,
            "ZerssenNiftiLoader": ZerssenNiftiLoader,
            "FmriprepDerivativesLoader": FmriprepDerivativesLoader,
            "TimeseriesDirectoryLoader": TimeseriesDirectoryLoader,
            "ConnectivityMatrixLoader": ConnectivityMatrixLoader,
            "ConditionTimeseriesArrayLoader": ConditionTimeseriesArrayLoader,
            "CustomPreparedZerssenLoader": CustomPreparedZerssenLoader,
            "ChinaCloseCloseLoader": ChinaCloseCloseLoader
        }
        
        loader_name = self.manifest.loader
        if not loader_name:
            raise ValueError("Manifest does not specify a loader.")
        if loader_name not in loader_map:
            raise ValueError(f"Unknown loader: {loader_name}")
            
        loader_class = loader_map[loader_name]
        
        # Build kwargs: merge resolved_paths and params
        kwargs = {}
        kwargs.update(self.resolved_paths)
        kwargs.update(self.manifest.params)
        
        # Filter kwargs to match constructor signature
        import inspect
        sig = inspect.signature(loader_class.__init__)
        valid_args = list(sig.parameters.keys())
        
        filtered_kwargs = {}
        for k, v in kwargs.items():
            if k in valid_args or 'kwargs' in valid_args:
                filtered_kwargs[k] = v
                
        self.target_loader = loader_class(**filtered_kwargs)
        
    def preview(self) -> DatasetPreview:
        preview = self.target_loader.preview()
        keep_rois = self.resolved_paths.get("keep_rois") or self.manifest.params.get("keep_rois")
        if keep_rois is not None:
            try:
                keep_ids = []
                if isinstance(keep_rois, str) and os.path.exists(keep_rois):
                    df = pd.read_csv(keep_rois)
                    if "id" in df.columns:
                        keep_ids = list(df["id"].values)
                    elif "roi_id" in df.columns:
                        keep_ids = list(df["roi_id"].values)
                    else:
                        keep_ids = list(df.iloc[:, 0].values)
                elif isinstance(keep_rois, (list, tuple, np.ndarray)):
                    keep_ids = list(keep_rois)
                if keep_ids:
                    preview.n_rois = len(keep_ids)
            except Exception:
                pass
        return preview
        
    def load(self) -> LoadedDataset:
        dataset = self.target_loader.load()
        
        # Atlas Auto-Loading
        if not dataset.atlas:
            atlas_path = self.resolved_paths.get("atlas") or self.resolved_paths.get("atlas_metadata") or self.resolved_paths.get("atlas_image_path")
            if atlas_path and os.path.exists(atlas_path) and atlas_path.endswith(".csv"):
                from conninfpy.atlas import AtlasInfo
                try:
                    dataset.atlas = AtlasInfo.from_csv(atlas_path)
                    dataset.roi_labels = dataset.atlas.labels
                except Exception:
                    pass
                    
        # Apply ROI filtering if keep_rois is specified
        keep_rois = self.resolved_paths.get("keep_rois") or self.manifest.params.get("keep_rois")
        if keep_rois is not None:
            dataset = filter_dataset_rois(dataset, keep_rois)
            
        # Apply checks/validations from manifest
        validate_manifest_dataset(dataset, self.manifest)
        
        return dataset

def filter_dataset_rois(dataset: LoadedDataset, keep_rois: Union[List[int], str]) -> LoadedDataset:
    """Filter the loaded dataset time-series or correlation matrices to keep only specified ROIs."""
    keep_ids = []
    if isinstance(keep_rois, str):
        if os.path.exists(keep_rois):
            try:
                df = pd.read_csv(keep_rois)
                if "id" in df.columns:
                    keep_ids = list(df["id"].values)
                elif "roi_id" in df.columns:
                    keep_ids = list(df["roi_id"].values)
                else:
                    keep_ids = list(df.iloc[:, 0].values)
            except Exception as e:
                dataset.warnings.append(f"Failed to read keep_rois CSV: {e}")
                return dataset
        else:
            dataset.warnings.append(f"keep_rois path not found: {keep_rois}")
            return dataset
    elif isinstance(keep_rois, (list, tuple, np.ndarray)):
        keep_ids = list(keep_rois)
    else:
        return dataset
        
    if not keep_ids:
        return dataset
        
    # Convert 1-indexed to 0-indexed
    keep_idx = np.array([int(x) - 1 for x in keep_ids if x > 0], dtype=int)
    n_rois = dataset.data.shape[-1]
    keep_idx = np.array([idx for idx in keep_idx if 0 <= idx < n_rois], dtype=int)
    
    if len(keep_idx) == 0:
        dataset.warnings.append("No valid ROI indices found to keep; skipping ROI filtering.")
        return dataset
        
    # Filter data tensor
    if dataset.data.ndim == 3:
        if dataset.data.shape[1] == dataset.data.shape[2]:
            dataset.data = dataset.data[:, keep_idx, :][:, :, keep_idx]
        else:
            dataset.data = dataset.data[:, :, keep_idx]
            
    # Filter atlas metadata
    if dataset.atlas is not None:
        from conninfpy.atlas import AtlasInfo
        labels = [dataset.atlas.labels[i] for i in keep_idx if i < len(dataset.atlas.labels)]
        networks = [dataset.atlas.networks[i] for i in keep_idx if i < len(dataset.atlas.networks)]
        coords = dataset.atlas.coords[keep_idx] if dataset.atlas.coords is not None else None
        hemisphere = [dataset.atlas.hemisphere[i] for i in keep_idx if i < len(dataset.atlas.hemisphere)] if dataset.atlas.hemisphere is not None else None
        
        dataset.atlas = AtlasInfo(
            labels=labels,
            networks=networks,
            coords=coords,
            hemisphere=hemisphere,
            source=f"{dataset.atlas.source} (filtered to {len(keep_idx)} ROIs)"
        )
        
    if dataset.roi_labels is not None:
        dataset.roi_labels = [dataset.roi_labels[i] for i in keep_idx if i < len(dataset.roi_labels)]
        
    return dataset

def validate_manifest_dataset(dataset: LoadedDataset, manifest: DatasetManifest):
    checks = manifest.checks
    if not checks:
        return
        
    errors = []
    
    # expected_rois
    if "expected_rois" in checks:
        expected_rois = checks["expected_rois"]
        n_rois = dataset.data.shape[-1]
        if n_rois != expected_rois:
            errors.append(f"ROI count mismatch: expected {expected_rois}, found {n_rois}")
            
    # min_subjects
    if "min_subjects" in checks:
        min_subjects = checks["min_subjects"]
        n_subs = dataset.data.shape[0]
        if n_subs < min_subjects:
            errors.append(f"Subject count below minimum: expected at least {min_subjects}, found {n_subs}")
            
    # min_timepoints
    if "min_timepoints" in checks:
        min_timepoints = checks["min_timepoints"]
        if dataset.data.ndim == 3:
            n_tps = dataset.data.shape[1]
            if n_tps < min_timepoints:
                errors.append(f"Timepoints below minimum: expected at least {min_timepoints}, found {n_tps}")
                
    # require_pheno_columns
    if "require_pheno_columns" in checks:
        req_cols = checks["require_pheno_columns"]
        for col in req_cols:
            if col not in dataset.pheno.columns:
                errors.append(f"Missing required phenotype column: {col}")
                
    # allowed_conditions
    if "allowed_conditions" in checks:
        allowed = checks["allowed_conditions"]
        if isinstance(allowed, list) and "condition" in dataset.pheno.columns:
            invalid = dataset.pheno[~dataset.pheno["condition"].isin(allowed)]
            if not invalid.empty:
                errors.append(f"Found invalid conditions in pheno: {invalid['condition'].unique()}")
        elif isinstance(allowed, dict):
            for col, vals in allowed.items():
                if col in dataset.pheno.columns:
                    invalid = dataset.pheno[~dataset.pheno[col].isin(vals)]
                    if not invalid.empty:
                        errors.append(f"Found invalid values in '{col}': {invalid[col].unique()}")
                        
    # require_symmetric
    if checks.get("require_symmetric", False):
        if dataset.data.ndim == 3:
            for idx in range(dataset.data.shape[0]):
                mat = dataset.data[idx]
                if not np.allclose(mat, mat.T, atol=1e-5, equal_nan=True):
                    errors.append(f"Matrix at index {idx} is not symmetric.")
                    break
                    
    # require_zero_diagonal
    if checks.get("require_zero_diagonal", False):
        if dataset.data.ndim == 3:
            for idx in range(dataset.data.shape[0]):
                mat = dataset.data[idx]
                diag = np.diagonal(mat)
                if not np.allclose(diag, 0, atol=1e-5, equal_nan=True):
                    errors.append(f"Matrix at index {idx} has non-zero diagonal.")
                    break

    # expected_subjects
    if "expected_subjects" in checks:
        expected_subs = checks["expected_subjects"]
        if isinstance(expected_subs, dict) and "group" in dataset.pheno.columns:
            for grp, count in expected_subs.items():
                actual_count = dataset.pheno[dataset.pheno["group"].str.lower() == grp.lower()].shape[0]
                if actual_count != count:
                    errors.append(f"Subject count mismatch for group '{grp}': expected {count}, found {actual_count}")
        elif isinstance(expected_subs, int):
            if dataset.data.shape[0] != expected_subs:
                errors.append(f"Subject count mismatch: expected {expected_subs}, found {dataset.data.shape[0]}")
                
    if errors:
        raise ValueError("Manifest validation failed:\n" + "\n".join(errors))
