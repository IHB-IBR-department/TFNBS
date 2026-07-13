from __future__ import annotations

import os
import glob
import pickle
import hashlib
import re
import numpy as np
import pandas as pd
from typing import Any, Literal
from pathlib import Path

from conninfpy.atlas import AtlasInfo
from conninfpy.loaders.base import BaseDataLoader, LoadedDataset, DatasetPreview
from conninfpy.loaders.validation import validate_loaded_dataset

def compute_dir_hash(files: list[str], extra_params: dict) -> str:
    """Compute a deterministic hash for a list of files and parameters to check cache validity."""
    h = hashlib.sha256()
    for f in sorted(files):
        if os.path.exists(f):
            stat = os.stat(f)
            h.update(f.encode('utf-8'))
            h.update(str(stat.st_mtime).encode('utf-8'))
            h.update(str(stat.st_size).encode('utf-8'))
    for k, v in sorted(extra_params.items()):
        h.update(f"{k}:{v}".encode('utf-8'))
    return h.hexdigest()


class NumpyLoader(BaseDataLoader):
    """Loader for standard numpy files (.npy or .npz) containing 3D arrays."""
    name = "NumpyLoader"

    def __init__(self, data_path: str, pheno_path: str | None = None, data_kind: Literal["timeseries", "correlation", "fisher_z"] = "correlation"):
        self.data_path = data_path
        self.pheno_path = pheno_path
        self.data_kind = data_kind

    def preview(self) -> DatasetPreview:
        warnings = []
        if not os.path.exists(self.data_path):
            return DatasetPreview(None, None, None, None, "unknown", None, None, [], None, None, [f"File not found: {self.data_path}"])
        
        try:
            # Check if it is a pickled dict (like ABIDE)
            d = np.load(self.data_path, allow_pickle=True)
            if d.ndim == 0:  # object array containing dict
                return DatasetPreview(
                    n_subjects=None, n_observations=None, n_rois=None, n_timepoints=None,
                    data_kind_guess="unknown", atlas_guess=None, conditions=None,
                    subject_ids_sample=[], file_count=1,
                    file_sizes_mb=os.path.getsize(self.data_path) / (1024 * 1024),
                    warnings=["This file appears to contain a pickled dictionary. Please use AbideSchaeferLoader instead."]
                )
            
            shape = d.shape
            n_obs = shape[0]
            n_rois = shape[2] if len(shape) == 3 else None
            n_tp = shape[1] if len(shape) == 3 and self.data_kind == "timeseries" else None
            
            return DatasetPreview(
                n_subjects=n_obs,
                n_observations=n_obs,
                n_rois=n_rois,
                n_timepoints=n_tp,
                data_kind_guess=self.data_kind,
                atlas_guess=None,
                conditions=None,
                subject_ids_sample=[f"sub_{i:02d}" for i in range(min(5, n_obs))],
                file_count=1,
                file_sizes_mb=os.path.getsize(self.data_path) / (1024 * 1024)
            )
        except Exception as e:
            return DatasetPreview(None, None, None, None, "unknown", None, None, [], 1, None, [f"Error reading file: {e}"])

    def load(self) -> LoadedDataset:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"File not found: {self.data_path}")
            
        data = np.load(self.data_path, allow_pickle=True)
        if data.ndim == 0:
            raise ValueError("This numpy file contains a pickled dictionary. Use AbideSchaeferLoader.")
            
        n_obs = data.shape[0]
        
        # Load pheno or generate a default one
        if self.pheno_path and os.path.exists(self.pheno_path):
            pheno = pd.read_csv(self.pheno_path)
        else:
            pheno = pd.DataFrame({
                "subject_id": [f"sub_{i:02d}" for i in range(n_obs)]
            })
            
        subject_ids = list(pheno["subject_id"].values) if "subject_id" in pheno.columns else [f"sub_{i:02d}" for i in range(n_obs)]
        
        return LoadedDataset(
            data=data,
            pheno=pheno,
            data_kind=self.data_kind,
            subject_ids=subject_ids,
            provenance={"data_path": self.data_path, "pheno_path": self.pheno_path}
        )


class CSVDirectoryLoader(BaseDataLoader):
    """Loader for a directory containing individual subject CSV/TSV timeseries files."""
    name = "CSVDirectoryLoader"

    def __init__(self, dir_path: str, header: bool | None = None, sep: str = ","):
        self.dir_path = dir_path
        self.header = header
        self.sep = sep

    def _get_files(self) -> list[str]:
        if not os.path.exists(self.dir_path):
            return []
        files = glob.glob(os.path.join(self.dir_path, "*.csv")) + glob.glob(os.path.join(self.dir_path, "*.tsv"))
        return sorted(files)

    def preview(self) -> DatasetPreview:
        files = self._get_files()
        if not files:
            return DatasetPreview(None, None, None, None, "unknown", None, None, [], 0, 0.0, ["No CSV or TSV files found in directory."])
            
        total_size = sum(os.path.getsize(f) for f in files) / (1024 * 1024)
        
        # Read a sample file to guess shapes
        try:
            sample_file = files[0]
            # Detect header: if first line is numeric, it is headerless
            header_choice = self.header
            if header_choice is None:
                with open(sample_file, 'r') as f:
                    first_line = f.readline().strip()
                try:
                    [float(x) for x in first_line.split(self.sep)]
                    header_choice = None  # No header
                except ValueError:
                    header_choice = 0  # First row is header
            
            df = pd.read_csv(sample_file, sep=self.sep, header=header_choice)
            n_timepoints, n_rois = df.shape
            
            # Extract sample subject IDs
            subject_ids_sample = [os.path.basename(f).split(".")[0] for f in files[:5]]
            
            return DatasetPreview(
                n_subjects=len(files),
                n_observations=len(files),
                n_rois=n_rois,
                n_timepoints=n_timepoints,
                data_kind_guess="timeseries",
                atlas_guess=None,
                conditions=None,
                subject_ids_sample=subject_ids_sample,
                file_count=len(files),
                file_sizes_mb=total_size
            )
        except Exception as e:
            return DatasetPreview(len(files), len(files), None, None, "timeseries", None, None, [], len(files), total_size, [f"Error previewing sample file: {e}"])

    def load(self) -> LoadedDataset:
        files = self._get_files()
        if not files:
            raise FileNotFoundError(f"No CSV/TSV files found in {self.dir_path}")
            
        # Determine header
        header_choice = self.header
        if header_choice is None:
            with open(files[0], 'r') as f:
                first_line = f.readline().strip()
            try:
                [float(x) for x in first_line.split(self.sep)]
                header_choice = None
            except ValueError:
                header_choice = 0

        data_list = []
        subject_ids = []
        
        for f in files:
            df = pd.read_csv(f, sep=self.sep, header=header_choice)
            data_list.append(df.values)
            # Infer subject ID from filename
            sub_id = os.path.basename(f).split(".")[0]
            # Strip common timeseries suffixes if present
            sub_id = re.sub(r'(_ts|_timeseries|_Brainnetome|_Schaefer\d+)', '', sub_id)
            subject_ids.append(sub_id)
            
        data = np.array(data_list) # shape: (n_subjects, n_timepoints, n_rois)
        pheno = pd.DataFrame({"subject_id": subject_ids})
        
        return LoadedDataset(
            data=data,
            pheno=pheno,
            data_kind="timeseries",
            subject_ids=subject_ids,
            provenance={"dir_path": self.dir_path, "file_count": len(files)}
        )


class NiftiDirectoryLoader(BaseDataLoader):
    """Loader for directories containing preprocessed functional NIfTIs (.nii/.nii.gz)."""
    name = "NiftiDirectoryLoader"

    def __init__(
        self,
        dir_path: str | None = None,
        atlas_image_path: str | None = None,
        cache_dir: str | None = None,
        confounds_pattern: str | None = None,
        file_pattern: str | None = None,
        subject_regex: str | None = None,
        group_from_folder: bool = False,
        extraction_strategy: str = "mean_signal",
        standardize: str | bool = "zscore",
        detrend: bool = False,
        subjects: dict[str, str] | None = None,
        pheno_path: str | None = None,
        atlas_metadata: str | None = None,
        atlas_lut: str | None = None
    ):
        self.dir_path = dir_path
        self.atlas_image_path = atlas_image_path
        self.cache_dir = cache_dir or os.path.expanduser("~/.conninfpy/loaders")
        self.confounds_pattern = confounds_pattern
        self.file_pattern = file_pattern
        self.subject_regex = subject_regex
        self.group_from_folder = group_from_folder
        self.extraction_strategy = extraction_strategy
        self.standardize = standardize
        self.detrend = detrend
        self.subjects = subjects
        self.pheno_path = pheno_path
        self.atlas_metadata = atlas_metadata
        self.atlas_lut = atlas_lut

    def _get_files(self) -> list[str]:
        if not self.dir_path or not os.path.exists(self.dir_path):
            return []
        if self.file_pattern:
            if "*" in self.file_pattern or "?" in self.file_pattern:
                files = glob.glob(os.path.join(self.dir_path, self.file_pattern))
                if not files:
                    files = glob.glob(os.path.join(self.dir_path, "**", self.file_pattern), recursive=True)
            else:
                files = [os.path.join(self.dir_path, self.file_pattern)]
        else:
            files = glob.glob(os.path.join(self.dir_path, "*.nii")) + glob.glob(os.path.join(self.dir_path, "*.nii.gz"))
        return sorted([f for f in files if os.path.exists(f)])

    def preview(self) -> DatasetPreview:
        if self.subjects:
            total_files = 0
            total_size = 0.0
            warnings = []
            subject_ids = []
            for grp, sdir in self.subjects.items():
                temp_loader = NiftiDirectoryLoader(
                    dir_path=sdir,
                    atlas_image_path=self.atlas_image_path,
                    file_pattern=self.file_pattern,
                    subject_regex=self.subject_regex
                )
                files = temp_loader._get_files()
                total_files += len(files)
                total_size += sum(os.path.getsize(f) for f in files) / (1024 * 1024)
                for f in files[:3]:
                    sub_id = os.path.basename(f).split(".")[0]
                    if self.subject_regex:
                        m = re.search(self.subject_regex, os.path.basename(f))
                        if m:
                            sub_id = m.group(1)
                    subject_ids.append(f"{grp}/{sub_id}")
            
            n_rois = None
            if self.atlas_image_path and os.path.isdir(self.atlas_image_path):
                n_rois = len(glob.glob(os.path.join(self.atlas_image_path, "*.nii*")))
                
            return DatasetPreview(
                n_subjects=total_files,
                n_observations=total_files,
                n_rois=n_rois,
                n_timepoints=None,
                data_kind_guess="timeseries",
                atlas_guess=os.path.basename(self.atlas_image_path) if self.atlas_image_path else None,
                conditions=list(self.subjects.keys()),
                subject_ids_sample=subject_ids[:5],
                file_count=total_files,
                file_sizes_mb=total_size,
                warnings=warnings
            )

        files = self._get_files()
        if not files:
            return DatasetPreview(None, None, None, None, "unknown", None, None, [], 0, 0.0, ["No NIfTI files found in directory."])
            
        total_size = sum(os.path.getsize(f) for f in files) / (1024 * 1024)
        
        # Check cache status
        cache_key = compute_dir_hash(files, {
            "atlas": self.atlas_image_path,
            "confounds": self.confounds_pattern,
            "file_pattern": self.file_pattern,
            "subject_regex": self.subject_regex,
            "standardize": self.standardize,
            "detrend": self.detrend,
            "extraction_strategy": self.extraction_strategy
        })
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        cached_msg = " [Cached]" if os.path.exists(cache_file) else " [Extraction Needed]"
        
        n_rois = None
        if self.atlas_image_path and os.path.isdir(self.atlas_image_path):
            n_rois = len(glob.glob(os.path.join(self.atlas_image_path, "*.nii*")))
        
        subject_ids_sample = []
        for f in files[:5]:
            sub_id = os.path.basename(f).split(".")[0]
            if self.subject_regex:
                m = re.search(self.subject_regex, os.path.basename(f))
                if m:
                    sub_id = m.group(1)
            subject_ids_sample.append(sub_id)
        
        return DatasetPreview(
            n_subjects=len(files),
            n_observations=len(files),
            n_rois=n_rois,
            n_timepoints=None,
            data_kind_guess="timeseries",
            atlas_guess=os.path.basename(self.atlas_image_path) if self.atlas_image_path else None,
            conditions=None,
            subject_ids_sample=subject_ids_sample,
            file_count=len(files),
            file_sizes_mb=total_size,
            warnings=[f"Cache Status: {cached_msg}"]
        )

    def load(self) -> LoadedDataset:
        import nibabel as nib
        from nilearn.maskers import NiftiLabelsMasker, NiftiMasker

        if self.subjects:
            all_data = []
            all_subject_ids = []
            all_groups = []
            for grp, sdir in self.subjects.items():
                sub_loader = NiftiDirectoryLoader(
                    dir_path=sdir,
                    atlas_image_path=self.atlas_image_path,
                    cache_dir=self.cache_dir,
                    confounds_pattern=self.confounds_pattern,
                    file_pattern=self.file_pattern,
                    subject_regex=self.subject_regex,
                    group_from_folder=self.group_from_folder,
                    extraction_strategy=self.extraction_strategy,
                    standardize=self.standardize,
                    detrend=self.detrend,
                    atlas_metadata=self.atlas_metadata,
                    atlas_lut=self.atlas_lut
                )
                sub_ds = sub_loader.load()
                all_data.append(sub_ds.data)
                all_subject_ids.extend(sub_ds.subject_ids)
                all_groups.extend([grp] * len(sub_ds.subject_ids))
            
            # Stack and match shape lengths
            min_len = min(d.shape[1] for d in all_data)
            data_list_clean = [d[:, :min_len, :] for d in all_data]
            data = np.vstack(data_list_clean)
            
            pheno = pd.DataFrame({
                "subject_id": all_subject_ids,
                "group": all_groups
            })
            
            if self.pheno_path and os.path.exists(self.pheno_path):
                if self.pheno_path.endswith(".xlsx"):
                    ext_pheno = pd.read_excel(self.pheno_path)
                else:
                    ext_pheno = pd.read_csv(self.pheno_path)
                id_col = next((c for c in ext_pheno.columns if c.lower() in ("subject_id", "subject", "sub_id", "id")), None)
                if id_col:
                    ext_pheno = ext_pheno.rename(columns={id_col: "subject_id"})
                    ext_pheno["subject_id"] = ext_pheno["subject_id"].astype(str)
                    pheno["subject_id"] = pheno["subject_id"].astype(str)
                    pheno = pd.merge(pheno, ext_pheno, on="subject_id", how="left")
            
            from conninfpy.atlas import AtlasInfo
            atlas = None
            if self.atlas_metadata and os.path.exists(self.atlas_metadata):
                try:
                    atlas = AtlasInfo.from_csv(self.atlas_metadata)
                except Exception:
                    pass
            
            return LoadedDataset(
                data=data,
                pheno=pheno,
                data_kind="timeseries",
                subject_ids=all_subject_ids,
                atlas=atlas,
                provenance={
                    "subjects": self.subjects,
                    "atlas_image_path": self.atlas_image_path,
                    "pheno_path": self.pheno_path
                }
            )

        files = self._get_files()
        if not files:
            raise FileNotFoundError(f"No NIfTI files found in {self.dir_path}")
            
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_key = compute_dir_hash(files, {
            "atlas": self.atlas_image_path,
            "confounds": self.confounds_pattern,
            "file_pattern": self.file_pattern,
            "subject_regex": self.subject_regex,
            "standardize": self.standardize,
            "detrend": self.detrend,
            "extraction_strategy": self.extraction_strategy
        })
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        data_list = []
        subject_ids = []
        groups = []

        is_mask_dir = self.atlas_image_path and os.path.isdir(self.atlas_image_path)
        if is_mask_dir:
            mask_files = sorted(glob.glob(os.path.join(self.atlas_image_path, "*.nii*")))
            if not mask_files:
                raise ValueError(f"No mask files found in atlas directory: {self.atlas_image_path}")

        std_val = self.standardize
        if std_val == "zscore_sample":
            std_val = "zscore"

        for idx, f in enumerate(files):
            sub_id = os.path.basename(f).split(".")[0]
            group_val = None
            if self.subject_regex:
                m = re.search(self.subject_regex, os.path.basename(f))
                if m:
                    sub_id = m.group(1)
                    if len(m.groups()) >= 2:
                        group_val = m.group(2)
            subject_ids.append(sub_id)
            
            if self.group_from_folder:
                group_val = os.path.basename(os.path.dirname(f))
            if group_val:
                groups.append(group_val)
                
            confounds = None
            if self.confounds_pattern:
                parent_dir = os.path.dirname(f)
                confound_files = glob.glob(os.path.join(parent_dir, f"*{sub_id}*{self.confounds_pattern}*"))
                if not confound_files:
                    confound_files = glob.glob(os.path.join(self.dir_path, f"*{sub_id}*{self.confounds_pattern}*"))
                if confound_files:
                    confounds = confound_files[0]

            if is_mask_dir:
                ts_cols = []
                for mask_file in mask_files:
                    masker = NiftiMasker(mask_img=mask_file, standardize=std_val, detrend=self.detrend)
                    ts = masker.fit_transform(f, confounds=confounds)
                    ts_cols.append(ts.mean(axis=1))
                timeseries = np.column_stack(ts_cols)
            else:
                masker = NiftiLabelsMasker(labels_img=self.atlas_image_path, standardize=std_val, detrend=self.detrend)
                timeseries = masker.fit_transform(f, confounds=confounds)
                
            data_list.append(timeseries)

        min_len = min(ts.shape[0] for ts in data_list)
        data_list_clean = [ts[:min_len] for ts in data_list]
        data = np.array(data_list_clean)

        pheno = pd.DataFrame({"subject_id": subject_ids})
        if groups:
            pheno["group"] = groups
            
        if self.pheno_path and os.path.exists(self.pheno_path):
            if self.pheno_path.endswith(".xlsx"):
                ext_pheno = pd.read_excel(self.pheno_path)
            else:
                ext_pheno = pd.read_csv(self.pheno_path)
            id_col = next((c for c in ext_pheno.columns if c.lower() in ("subject_id", "subject", "sub_id", "id")), None)
            if id_col:
                ext_pheno = ext_pheno.rename(columns={id_col: "subject_id"})
                ext_pheno["subject_id"] = ext_pheno["subject_id"].astype(str)
                pheno["subject_id"] = pheno["subject_id"].astype(str)
                pheno = pd.merge(pheno, ext_pheno, on="subject_id", how="left")

        from conninfpy.atlas import AtlasInfo
        atlas = None
        if self.atlas_metadata and os.path.exists(self.atlas_metadata):
            try:
                atlas = AtlasInfo.from_csv(self.atlas_metadata)
            except Exception:
                pass

        result = LoadedDataset(
            data=data,
            pheno=pheno,
            data_kind="timeseries",
            subject_ids=subject_ids,
            atlas=atlas,
            provenance={
                "dir_path": self.dir_path,
                "atlas_image_path": self.atlas_image_path,
                "confounds_pattern": self.confounds_pattern,
                "hash": cache_key
            }
        )

        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)

        return result


class AbideSchaeferLoader(BaseDataLoader):
    """Loader for the pickled dictionary format of the ABIDE Schaefer-100 dataset."""
    name = "AbideSchaeferLoader"

    def __init__(self, data_path: str, pheno_csv_path: str | None = None):
        self.data_path = data_path
        self.pheno_csv_path = pheno_csv_path

    def preview(self) -> DatasetPreview:
        if not os.path.exists(self.data_path):
            return DatasetPreview(None, None, None, None, "unknown", None, None, [], 0, 0.0, ["File not found."])
            
        try:
            d = np.load(self.data_path, allow_pickle=True).item()
            keys = list(d.keys())
            sample_key = keys[0]
            ts = d[sample_key]["time_series"]
            # ts is shape (timepoints, rois)
            n_tp, n_rois = ts.shape
            
            return DatasetPreview(
                n_subjects=len(keys),
                n_observations=len(keys),
                n_rois=n_rois,
                n_timepoints=n_tp,
                data_kind_guess="timeseries",
                atlas_guess="Schaefer-100",
                conditions=None,
                subject_ids_sample=keys[:5],
                file_count=1,
                file_sizes_mb=os.path.getsize(self.data_path) / (1024 * 1024)
            )
        except Exception as e:
            return DatasetPreview(None, None, None, None, "unknown", None, None, [], 1, None, [f"Error reading ABIDE dict: {e}"])

    def load(self) -> LoadedDataset:
        d = np.load(self.data_path, allow_pickle=True).item()
        
        subject_ids = list(d.keys())
        labels = []
        sites = []
        corr_list = []
        
        # Since ABIDE timepoints might slightly differ by site, we compute correlation
        # matrices for each subject directly in the loader.
        for sub_id in subject_ids:
            ts = d[sub_id]["time_series"] # shape: (timepoints, 100)
            # Compute Pearson correlation matrix
            corr = np.corrcoef(ts.T)
            corr = np.nan_to_num(corr)
            corr_list.append(corr)
            
            labels.append(d[sub_id].get("label", "unknown"))
            sites.append(d[sub_id].get("site", "unknown"))
            
        data = np.array(corr_list) # shape: (n_subjects, 100, 100)
        
        pheno = pd.DataFrame({
            "subject_id": subject_ids,
            "group_interest": labels,
            "site": sites
        })
        
        # Merge with external phenotypic CSV if provided
        if self.pheno_csv_path and os.path.exists(self.pheno_csv_path):
            try:
                ext_pheno = pd.read_csv(self.pheno_csv_path)
                
                # The .npy keys correspond to FILE_ID in the ABIDE dataset.
                if "FILE_ID" in ext_pheno.columns:
                    ext_pheno["FILE_ID"] = ext_pheno["FILE_ID"].astype(str)
                    pheno = pheno.merge(ext_pheno, left_on="subject_id", right_on="FILE_ID", how="left")
                else:
                    sub_col = None
                    for col in ext_pheno.columns:
                        if col.lower() in {"subject_id", "sub_id", "sub_id_str"}:
                            sub_col = col
                            break
                    if sub_col:
                        ext_pheno[sub_col] = ext_pheno[sub_col].astype(str)
                        pheno = pheno.merge(ext_pheno, left_on="subject_id", right_on=sub_col, how="left")
                
                # Filter down to core ABIDE columns to prevent UI clutter
                core_columns = [
                    "subject_id", "site", "group_interest", "DX_GROUP", 
                    "AGE_AT_SCAN", "SEX", "FIQ", "func_mean_fd", 
                    "ADOS_TOTAL", "SRS_RAW_TOTAL"
                ]
                keep_cols = [c for c in core_columns if c in pheno.columns]
                pheno = pheno[keep_cols]
                
                # Coerce target columns to numeric (except DX_GROUP which we map to categorical strings)
                numeric_cols = ["AGE_AT_SCAN", "SEX", "FIQ", "func_mean_fd", "ADOS_TOTAL", "SRS_RAW_TOTAL"]
                for nc in numeric_cols:
                    if nc in pheno.columns:
                        pheno[nc] = pd.to_numeric(pheno[nc], errors='coerce')
                
                # Rename columns for clarity in the UI
                rename_map = {
                    "DX_GROUP": "Diagnosis",
                    "AGE_AT_SCAN": "Age",
                    "SEX": "Sex",
                    "func_mean_fd": "Motion_FD"
                }
                pheno = pheno.rename(columns=rename_map)
                
                # Map Diagnosis to match group_interest string values (1 -> autism, 2 -> healthy_control)
                if "Diagnosis" in pheno.columns:
                    pheno["Diagnosis"] = pheno["Diagnosis"].map({1: "autism", 2: "healthy_control"})
                
            except Exception as e:
                # Add warning but continue
                pass
                
        return LoadedDataset(
            data=data,
            pheno=pheno,
            data_kind="correlation",
            subject_ids=subject_ids,
            provenance={"data_path": self.data_path, "pheno_csv_path": self.pheno_csv_path}
        )


class OpenCloseLoader(BaseDataLoader):
    """Loader for the open_close dataset containing paired timeseries arrays for different conditions."""
    name = "OpenCloseLoader"

    def __init__(self, open_path: str, close_path: str, subject_list_path: str | None = None, atlas: str | None = None, drop_missing_rois: bool = False):
        self.open_path = open_path
        self.close_path = close_path
        self.subject_list_path = subject_list_path
        self.atlas = atlas
        self.drop_missing_rois = drop_missing_rois

    def _get_kept_indices(self) -> np.ndarray | None:
        if self.drop_missing_rois and self.atlas and os.path.exists(self.atlas):
            try:
                df = pd.read_csv(self.atlas)
                if "id" in df.columns:
                    return df["id"].values - 1
            except Exception:
                pass
        return None

    def preview(self) -> DatasetPreview:
        if not os.path.exists(self.open_path) or not os.path.exists(self.close_path):
            return DatasetPreview(None, None, None, None, "unknown", None, None, [], 0, 0.0, ["Open or close file path not found."])
            
        try:
            d_open = np.load(self.open_path)
            shape = d_open.shape # (subjects, timepoints, rois)
            n_sub, n_tp, n_rois = shape
            
            kept_idx = self._get_kept_indices()
            if kept_idx is not None:
                n_rois = len(kept_idx)
            
            subject_ids = []
            if self.subject_list_path and os.path.exists(self.subject_list_path):
                with open(self.subject_list_path, 'r') as f:
                    subject_ids = [line.strip() for line in f if line.strip()]
            else:
                subject_ids = [f"sub_{i:02d}" for i in range(n_sub)]
                
            return DatasetPreview(
                n_subjects=n_sub,
                n_observations=n_sub * 2,
                n_rois=n_rois,
                n_timepoints=n_tp,
                data_kind_guess="timeseries",
                atlas_guess=os.path.basename(self.atlas) if self.atlas else f"Schaefer-{n_rois}",
                conditions=["open", "close"],
                subject_ids_sample=subject_ids[:5],
                file_count=2,
                file_sizes_mb=(os.path.getsize(self.open_path) + os.path.getsize(self.close_path)) / (1024 * 1024)
            )
        except Exception as e:
            return DatasetPreview(None, None, None, None, "unknown", None, None, [], 2, None, [f"Error previewing open_close: {e}"])

    def load(self) -> LoadedDataset:
        open_data = np.load(self.open_path)
        close_data = np.load(self.close_path)
        
        if close_data.ndim == 4:
            close_data = close_data[..., 0]
            
        n_sub, n_tp, n_rois = open_data.shape
        
        kept_idx = self._get_kept_indices()
        if kept_idx is not None:
            open_data = open_data[:, :, kept_idx]
            close_data = close_data[:, :, kept_idx]
            n_rois = len(kept_idx)
            
        subject_ids = []
        if self.subject_list_path and os.path.exists(self.subject_list_path):
            with open(self.subject_list_path, 'r') as f:
                subject_ids = [line.strip() for line in f if line.strip()]
        else:
            subject_ids = [f"sub_{i:02d}" for i in range(n_sub)]
            
        if len(subject_ids) != n_sub:
            # Fallback if subject list length mismatch
            subject_ids = [f"sub_{i:02d}" for i in range(n_sub)]

        # Convert both to connectivity first
        open_corr = np.array([np.nan_to_num(np.corrcoef(ts.T)) for ts in open_data])
        close_corr = np.array([np.nan_to_num(np.corrcoef(ts.T)) for ts in close_data])
        
        # Combine into long format
        data = np.vstack([open_corr, close_corr]) # shape: (n_sub * 2, n_rois, n_rois)
        
        pheno = pd.DataFrame({
            "subject_id": subject_ids * 2,
            "condition": ["open"] * n_sub + ["close"] * n_sub
        })
        
        # Load atlas
        atlas = None
        if self.atlas and os.path.exists(self.atlas):
            from conninfpy.atlas import AtlasInfo
            try:
                atlas = AtlasInfo.from_csv(self.atlas)
            except Exception:
                pass
        
        return LoadedDataset(
            data=data,
            pheno=pheno,
            data_kind="correlation",
            subject_ids=subject_ids * 2,
            conditions=["open", "close"],
            condition_column="condition",
            atlas=atlas,
            provenance={"open_path": self.open_path, "close_path": self.close_path}
        )


class MultiSiteOpenCloseLoader(BaseDataLoader):
    """Combine paired Open/Close cohorts while preserving their site labels.

    Every subject contributes one open and one close observation at the same
    acquisition site. Subject IDs are prefixed with that site to guarantee
    uniqueness when paired inference is configured from the combined table.
    """
    name = "MultiSiteOpenCloseLoader"

    def __init__(
        self,
        site_configs: dict[str, dict[str, str]],
        atlas: str | None = None,
        drop_missing_rois: bool = True,
    ):
        if not site_configs:
            raise ValueError("site_configs must contain at least one site.")
        self.site_configs = site_configs
        self.atlas = atlas
        self.drop_missing_rois = drop_missing_rois

    def _site_loaders(self) -> list[tuple[str, OpenCloseLoader]]:
        loaders = []
        for site, config in self.site_configs.items():
            loaders.append((
                site,
                OpenCloseLoader(
                    config["open_path"],
                    config["close_path"],
                    subject_list_path=config.get("subject_list_path"),
                    atlas=self.atlas,
                    drop_missing_rois=self.drop_missing_rois,
                ),
            ))
        return loaders

    def preview(self) -> DatasetPreview:
        previews = [(site, loader.preview()) for site, loader in self._site_loaders()]
        warnings = [
            f"{site}: {warning}"
            for site, preview in previews
            for warning in preview.warnings
        ]
        valid = [preview for _, preview in previews if preview.n_rois is not None]
        if not valid:
            return DatasetPreview(
                None, None, None, None, "unknown", None, None, [], 0, 0.0, warnings
            )

        roi_counts = {preview.n_rois for preview in valid}
        if len(roi_counts) != 1:
            warnings.append(f"Site ROI counts differ: {sorted(roi_counts)}.")

        timepoints = [preview.n_timepoints for preview in valid if preview.n_timepoints is not None]
        n_timepoints = (
            timepoints[0] if len(set(timepoints)) == 1
            else (min(timepoints), max(timepoints))
        )
        return DatasetPreview(
            n_subjects=sum(preview.n_subjects or 0 for preview in valid),
            n_observations=sum(preview.n_observations or 0 for preview in valid),
            n_rois=valid[0].n_rois,
            n_timepoints=n_timepoints,
            data_kind_guess="correlation",
            atlas_guess=valid[0].atlas_guess,
            conditions=["open", "close"],
            subject_ids_sample=[
                f"{site}:{subject_id}"
                for site, preview in previews
                for subject_id in preview.subject_ids_sample[:2]
            ][:5],
            file_count=sum(preview.file_count or 0 for preview in valid),
            file_sizes_mb=sum(preview.file_sizes_mb or 0.0 for preview in valid),
            warnings=warnings,
        )

    def load(self) -> LoadedDataset:
        datasets = [(site, loader.load()) for site, loader in self._site_loaders()]
        roi_counts = {dataset.data.shape[1] for _, dataset in datasets}
        if len(roi_counts) != 1:
            raise ValueError(f"All sites must have the same ROI count; got {sorted(roi_counts)}.")

        data_parts = []
        pheno_parts = []
        subject_ids = []
        for site, dataset in datasets:
            site_pheno = dataset.pheno.copy()
            original_ids = site_pheno["subject_id"].astype(str)
            site_pheno["site"] = site
            site_pheno["subject_id"] = site + ":" + original_ids
            data_parts.append(dataset.data)
            pheno_parts.append(site_pheno)
            subject_ids.extend(site_pheno["subject_id"].tolist())

        return LoadedDataset(
            data=np.concatenate(data_parts, axis=0),
            pheno=pd.concat(pheno_parts, ignore_index=True),
            data_kind="correlation",
            subject_ids=subject_ids,
            conditions=["open", "close"],
            condition_column="condition",
            atlas=datasets[0][1].atlas,
            provenance={
                "sites": list(self.site_configs),
                "paired_within_site": True,
                "site_configs": self.site_configs,
            },
        )


class StressTimeseriesLoader(BaseDataLoader):
    """Loader for the stress dataset containing individual headerless CSV timeseries files."""
    name = "StressTimeseriesLoader"

    def __init__(self, dir_path: str):
        self.dir_path = dir_path

    def preview(self) -> DatasetPreview:
        # Thin wrapper over CSVDirectoryLoader, customized for Brainnetome-246
        loader = CSVDirectoryLoader(self.dir_path, header=None)
        prev = loader.preview()
        prev.atlas_guess = "Brainnetome-246"
        return prev

    def load(self) -> LoadedDataset:
        # Load timeseries using standard CSV loader
        loader = CSVDirectoryLoader(self.dir_path, header=None)
        ds = loader.load()
        
        # Infer Brainnetome atlas
        from conninfpy.atlas import AtlasInfo
        try:
            ds.atlas = AtlasInfo.bna_246()
            ds.roi_labels = ds.atlas.labels
        except Exception:
            pass
            
        return ds


class ZerssenNiftiLoader(BaseDataLoader):
    """Loader for the Zerssen dataset containing preprocessed fMRI files separated by HC and Patients folders."""
    name = "ZerssenNiftiLoader"

    def __init__(self, subjects_dir: str, atlas_image_path: str, cache_dir: str | None = None):
        self.subjects_dir = subjects_dir
        self.atlas_image_path = atlas_image_path
        self.cache_dir = cache_dir

    def _get_folders_and_files(self) -> tuple[list[str], list[str]]:
        hc_dir = os.path.join(self.subjects_dir, "HC")
        pat_dir = os.path.join(self.subjects_dir, "Patients")
        
        hc_files = sorted(glob.glob(os.path.join(hc_dir, "*.nii*")))
        pat_files = sorted(glob.glob(os.path.join(pat_dir, "*.nii*")))
        return hc_files, pat_files

    def preview(self) -> DatasetPreview:
        hc_files, pat_files = self._get_folders_and_files()
        all_files = hc_files + pat_files
        
        if not all_files:
            return DatasetPreview(None, None, None, None, "unknown", None, None, [], 0, 0.0, ["No HC or Patients functional NIfTI files found."])
            
        total_size = sum(os.path.getsize(f) for f in all_files) / (1024 * 1024)
        
        n_rois = None
        if os.path.isdir(self.atlas_image_path):
            n_rois = len(glob.glob(os.path.join(self.atlas_image_path, "*.nii*")))
            
        subject_ids_sample = [os.path.basename(f).split("_")[0] for f in all_files[:5]]
        
        return DatasetPreview(
            n_subjects=len(all_files),
            n_observations=len(all_files),
            n_rois=n_rois,
            n_timepoints=None,
            data_kind_guess="timeseries",
            atlas_guess="Brainnetome-246",
            conditions=["HC", "PAT"],
            subject_ids_sample=subject_ids_sample,
            file_count=len(all_files),
            file_sizes_mb=total_size
        )

    def load(self) -> LoadedDataset:
        hc_files, pat_files = self._get_folders_and_files()
        
        # Load HC group using NiftiDirectoryLoader
        hc_loader = NiftiDirectoryLoader(os.path.join(self.subjects_dir, "HC"), self.atlas_image_path, self.cache_dir)
        hc_ds = hc_loader.load()
        
        # Load Patients group
        pat_loader = NiftiDirectoryLoader(os.path.join(self.subjects_dir, "Patients"), self.atlas_image_path, self.cache_dir)
        pat_ds = pat_loader.load()
        
        # Combine data
        data = np.vstack([hc_ds.data, pat_ds.data])
        
        # Generate Pheno DataFrame
        subject_ids = hc_ds.subject_ids + pat_ds.subject_ids
        groups = ["HC"] * len(hc_ds.subject_ids) + ["PAT"] * len(pat_ds.subject_ids)
        
        pheno = pd.DataFrame({
            "subject_id": subject_ids,
            "group": groups
        })
        from conninfpy.atlas import AtlasInfo
        atlas = None
        try:
            atlas = AtlasInfo.bna_246()
        except Exception:
            pass

        return LoadedDataset(
            data=data,
            pheno=pheno,
            data_kind="timeseries",
            subject_ids=subject_ids,
            atlas=atlas,
            provenance={"subjects_dir": self.subjects_dir, "atlas_image_path": self.atlas_image_path}
        )


def build_fmriprep_confounds(img_path: str, strategy: int, n_compcor: int | None = 10, use_GSR: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
    from nilearn.interfaces.fmriprep import load_confounds
    
    assert strategy in [1, 2, 3, 4, 5, 6], "Strategy must be 1-6"
    
    strategy_1 = {'strategy': ['motion'], 'motion': 'full'}
    strategy_2 = {'strategy': ['motion', 'compcor', 'high_pass'], 'motion': 'derivatives', 'compcor': 'anat_combined', 'n_compcor': n_compcor}
    strategy_3 = {'strategy': ['motion', 'compcor', 'high_pass'], 'motion': 'derivatives', 'compcor': 'anat_combined', 'n_compcor': 'all'}
    strategy_4 = {'strategy': ['motion', 'compcor', 'high_pass'], 'motion': 'full', 'compcor': 'anat_combined', 'n_compcor': n_compcor}
    strategy_5 = {'strategy': ['motion', 'compcor', 'high_pass'], 'motion': 'full', 'compcor': 'anat_combined', 'n_compcor': 'all'}
    strategy_6 = {'strategy': ['motion', 'compcor', 'high_pass'], 'motion': 'full', 'compcor': 'temporal_anat_combined', 'n_compcor': 'all'}
    
    strategy_dict = [strategy_1, strategy_2, strategy_3, strategy_4, strategy_5, strategy_6][strategy - 1].copy()
    
    if use_GSR:
        strategy_dict['strategy'] = list(strategy_dict['strategy']) + ['global_signal']
        strategy_dict['global_signal'] = 'full'
        
    return load_confounds(img_path, **strategy_dict)


class FmriprepDerivativesLoader(BaseDataLoader):
    """Loader for fMRIPrep derivatives, performing atlas extraction and confound regression."""
    name = "FmriprepDerivativesLoader"
    
    def __init__(
        self,
        derivatives: dict[str, str] | None = None,
        derivatives_dir: str | None = None,
        pheno_path: str | None = None,
        atlas_image_path: str | None = None,
        atlas_metadata: str | None = None,
        bold_pattern: str = "sub-*/func/*desc-preproc_bold.nii.gz",
        confounds_pattern: str = "sub-*/func/*desc-confounds_timeseries.tsv",
        subject_regex: str = "sub-([^_/]+)",
        group_from_folder: bool = False,
        denoising: dict | None = None,
        cache_dir: str | None = None
    ):
        self.derivatives = derivatives
        self.derivatives_dir = derivatives_dir
        self.pheno_path = pheno_path
        self.atlas_image_path = atlas_image_path
        self.atlas_metadata = atlas_metadata
        self.bold_pattern = bold_pattern
        self.confounds_pattern = confounds_pattern
        self.subject_regex = subject_regex
        self.group_from_folder = group_from_folder
        self.denoising = denoising or {}
        self.cache_dir = cache_dir or os.path.expanduser("~/.conninfpy/loaders")

    def _get_files(self, base_dir: str) -> list[tuple[str, str]]:
        if not os.path.exists(base_dir):
            return []
        
        bold_files = glob.glob(os.path.join(base_dir, self.bold_pattern))
        if not bold_files:
            bold_files = glob.glob(os.path.join(base_dir, "**", self.bold_pattern), recursive=True)
            
        results = []
        for bf in sorted(bold_files):
            sub_id = "sub-.*"
            m = re.search(self.subject_regex, os.path.basename(bf))
            if m:
                sub_id = m.group(1)
            
            parent_dir = os.path.dirname(bf)
            confound_files = glob.glob(os.path.join(parent_dir, f"*confounds*.tsv"))
            if not confound_files:
                confound_files = glob.glob(os.path.join(base_dir, "**", f"sub-{sub_id}*confounds*.tsv"), recursive=True)
            
            cf = confound_files[0] if confound_files else None
            results.append((bf, cf))
        return results

    def preview(self) -> DatasetPreview:
        dirs_to_scan = {}
        if self.derivatives:
            dirs_to_scan = self.derivatives
        elif self.derivatives_dir:
            dirs_to_scan = {"all": self.derivatives_dir}
            
        total_files = 0
        total_size = 0.0
        sample_subs = []
        for grp, sdir in dirs_to_scan.items():
            pairs = self._get_files(sdir)
            total_files += len(pairs)
            for bf, _ in pairs:
                total_size += os.path.getsize(bf) / (1024 * 1024)
            for bf, _ in pairs[:3]:
                sub_id = os.path.basename(bf).split("_")[0]
                sample_subs.append(f"{grp}/{sub_id}")
                
        n_rois = None
        if self.atlas_image_path and os.path.isdir(self.atlas_image_path):
            n_rois = len(glob.glob(os.path.join(self.atlas_image_path, "*.nii*")))
            
        return DatasetPreview(
            n_subjects=total_files,
            n_observations=total_files,
            n_rois=n_rois,
            n_timepoints=None,
            data_kind_guess="timeseries",
            atlas_guess=os.path.basename(self.atlas_image_path) if self.atlas_image_path else None,
            conditions=list(dirs_to_scan.keys()) if self.derivatives else None,
            subject_ids_sample=sample_subs[:5],
            file_count=total_files,
            file_sizes_mb=total_size
        )

    def load(self) -> LoadedDataset:
        import nibabel as nib
        from nilearn.maskers import NiftiLabelsMasker, NiftiMasker

        dirs_to_scan = {}
        if self.derivatives:
            dirs_to_scan = self.derivatives
        elif self.derivatives_dir:
            dirs_to_scan = {"all": self.derivatives_dir}
            
        all_data = []
        all_subject_ids = []
        all_groups = []
        
        strategy = self.denoising.get("strategy", 4)
        n_compcor = self.denoising.get("n_compcor", 10)
        use_GSR = self.denoising.get("use_GSR", False)
        smoothing = self.denoising.get("smoothing", None)
        
        is_mask_dir = self.atlas_image_path and os.path.isdir(self.atlas_image_path)
        if is_mask_dir:
            mask_files = sorted(glob.glob(os.path.join(self.atlas_image_path, "*.nii*")))
            if not mask_files:
                raise ValueError(f"No mask files found in atlas directory: {self.atlas_image_path}")

        for grp, sdir in dirs_to_scan.items():
            pairs = self._get_files(sdir)
            for bf, cf in pairs:
                sub_id = os.path.basename(bf).split(".")[0]
                m = re.search(self.subject_regex, os.path.basename(bf))
                if m:
                    sub_id = m.group(1)
                
                cache_key = compute_dir_hash([bf, cf] if cf else [bf], {
                    "atlas": self.atlas_image_path,
                    "strategy": strategy,
                    "n_compcor": n_compcor,
                    "use_GSR": use_GSR,
                    "smoothing": smoothing
                })
                os.makedirs(self.cache_dir, exist_ok=True)
                cache_file = os.path.join(self.cache_dir, f"fmriprep_{cache_key}.npy")
                
                if os.path.exists(cache_file):
                    timeseries = np.load(cache_file)
                else:
                    confounds = None
                    if cf:
                        confounds, _ = build_fmriprep_confounds(bf, strategy, n_compcor, use_GSR)
                    
                    if is_mask_dir:
                        ts_cols = []
                        for mask_file in mask_files:
                            masker = NiftiMasker(mask_img=mask_file, standardize="zscore")
                            if smoothing is not None:
                                masker.set_params(smoothing_fwhm=smoothing)
                            ts = masker.fit_transform(bf, confounds=confounds)
                            ts_cols.append(ts.mean(axis=1))
                        timeseries = np.column_stack(ts_cols)
                    else:
                        masker = NiftiLabelsMasker(labels_img=self.atlas_image_path, standardize="zscore")
                        if smoothing is not None:
                            masker.set_params(smoothing_fwhm=smoothing)
                        timeseries = masker.fit_transform(bf, confounds=confounds)
                    
                    np.save(cache_file, timeseries)
                    
                all_data.append(timeseries)
                all_subject_ids.append(sub_id)
                if self.derivatives:
                    all_groups.append(grp)
                elif self.group_from_folder:
                    all_groups.append(os.path.basename(os.path.dirname(os.path.dirname(bf))))
                else:
                    all_groups.append("control")
                    
        min_len = min(ts.shape[0] for ts in all_data)
        data_list_clean = [ts[:min_len] for ts in all_data]
        data = np.array(data_list_clean)
        
        pheno = pd.DataFrame({
            "subject_id": all_subject_ids,
            "group": all_groups
        })
        
        if self.pheno_path and os.path.exists(self.pheno_path):
            if self.pheno_path.endswith(".xlsx"):
                ext_pheno = pd.read_excel(self.pheno_path)
            else:
                ext_pheno = pd.read_csv(self.pheno_path)
            id_col = next((c for c in ext_pheno.columns if c.lower() in ("subject_id", "subject", "sub_id", "id")), None)
            if id_col:
                ext_pheno = ext_pheno.rename(columns={id_col: "subject_id"})
                ext_pheno["subject_id"] = ext_pheno["subject_id"].astype(str)
                pheno["subject_id"] = pheno["subject_id"].astype(str)
                pheno = pd.merge(pheno, ext_pheno, on="subject_id", how="left")
                
        from conninfpy.atlas import AtlasInfo
        atlas = None
        if self.atlas_metadata and os.path.exists(self.atlas_metadata):
            try:
                atlas = AtlasInfo.from_csv(self.atlas_metadata)
            except Exception:
                pass
                
        return LoadedDataset(
            data=data,
            pheno=pheno,
            data_kind="timeseries",
            subject_ids=all_subject_ids,
            atlas=atlas,
            provenance={
                "bold_pattern": self.bold_pattern,
                "strategy": strategy,
                "use_GSR": use_GSR
            }
        )


class TimeseriesDirectoryLoader(BaseDataLoader):
    """Loader for directories containing already extracted time-series CSV/NPY files."""
    name = "TimeseriesDirectoryLoader"
    
    def __init__(
        self,
        dir_path: str | None = None,
        dir_paths: dict[str, str] | None = None,
        pheno_path: str | None = None,
        atlas: str | None = None,
        pattern: str = "*.csv",
        sep: str = ",",
        header: str | bool | None = "infer",
        subject_regex: str = "sub-([^_/]+)",
        run_regex: str | None = None
    ):
        self.dir_path = dir_path
        self.dir_paths = dir_paths
        self.pheno_path = pheno_path
        self.atlas = atlas
        self.pattern = pattern
        self.sep = sep
        self.header = header
        self.subject_regex = subject_regex
        self.run_regex = run_regex

    def _get_files_from_dir(self, directory: str) -> list[str]:
        if not os.path.exists(directory):
            return []
        files = glob.glob(os.path.join(directory, self.pattern))
        if not files:
            files = glob.glob(os.path.join(directory, "**", self.pattern), recursive=True)
        return sorted(files)

    def _get_all_files(self) -> dict[str, list[str]]:
        if self.dir_paths:
            return {grp: self._get_files_from_dir(d) for grp, d in self.dir_paths.items()}
        elif self.dir_path:
            return {"control": self._get_files_from_dir(self.dir_path)}
        return {}

    def preview(self) -> DatasetPreview:
        all_groups_files = self._get_all_files()
        total_files = sum(len(f) for f in all_groups_files.values())
        if total_files == 0:
            return DatasetPreview(None, None, None, None, "unknown", None, None, [], 0, 0.0, ["No files matched pattern."])
            
        first_grp = list(all_groups_files.keys())[0]
        first_file = all_groups_files[first_grp][0]
        n_tp, n_rois = None, None
        try:
            if first_file.endswith(".npy"):
                arr = np.load(first_file)
                n_tp, n_rois = arr.shape
            else:
                hdr = 0 if self.header == "infer" or self.header is True else None
                df = pd.read_csv(first_file, sep=self.sep, header=hdr)
                n_tp, n_rois = df.shape
        except Exception:
            pass
            
        subject_ids = []
        for grp, files in all_groups_files.items():
            for f in files[:3]:
                sub_id = os.path.basename(f).split(".")[0]
                m = re.search(self.subject_regex, os.path.basename(f))
                if m:
                    sub_id = m.group(1)
                subject_ids.append(f"{grp}/{sub_id}")
                
        total_sizes = 0.0
        for grp, files in all_groups_files.items():
            total_sizes += sum(os.path.getsize(f) for f in files) / (1024 * 1024)
            
        return DatasetPreview(
            n_subjects=total_files,
            n_observations=total_files,
            n_rois=n_rois,
            n_timepoints=n_tp,
            data_kind_guess="timeseries",
            atlas_guess=os.path.basename(self.atlas) if self.atlas else None,
            conditions=list(all_groups_files.keys()) if self.dir_paths else None,
            subject_ids_sample=subject_ids[:5],
            file_count=total_files,
            file_sizes_mb=total_sizes
        )

    def load(self) -> LoadedDataset:
        all_groups_files = self._get_all_files()
        total_files = sum(len(f) for f in all_groups_files.values())
        if total_files == 0:
            raise FileNotFoundError(f"No files matched pattern: {self.pattern}")
            
        data_list = []
        subject_ids = []
        groups = []
        
        hdr = 0 if self.header == "infer" or self.header is True else None
        
        for grp, files in all_groups_files.items():
            for f in files:
                sub_id = os.path.basename(f).split(".")[0]
                m = re.search(self.subject_regex, os.path.basename(f))
                if m:
                    sub_id = m.group(1)
                subject_ids.append(sub_id)
                groups.append(grp)
                
                if f.endswith(".npy"):
                    arr = np.asarray(np.load(f), dtype=np.float64)
                else:
                    df = pd.read_csv(f, sep=self.sep, header=hdr)
                    try:
                        arr = df.apply(pd.to_numeric, errors="raise").to_numpy(
                            dtype=np.float64,
                        )
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"Time-series file contains non-numeric values: {f}. "
                            "Check the manifest 'header' and 'sep' settings."
                        ) from exc
                data_list.append(arr)
            
        min_len = min(arr.shape[0] for arr in data_list)
        data_list_clean = [arr[:min_len] for arr in data_list]
        data = np.array(data_list_clean)
        
        pheno = pd.DataFrame({
            "subject_id": subject_ids,
            "group": groups
        })
        
        if self.pheno_path and os.path.exists(self.pheno_path):
            if self.pheno_path.endswith(".xlsx"):
                ext_pheno = pd.read_excel(self.pheno_path)
            else:
                ext_pheno = pd.read_csv(self.pheno_path)
            id_col = next((c for c in ext_pheno.columns if c.lower() in ("subject_id", "subject", "sub_id", "id")), None)
            if id_col:
                ext_pheno = ext_pheno.rename(columns={id_col: "subject_id"})
                ext_pheno["subject_id"] = ext_pheno["subject_id"].astype(str)
                pheno["subject_id"] = pheno["subject_id"].astype(str)
                pheno = pd.merge(pheno, ext_pheno, on="subject_id", how="left")
                
        from conninfpy.atlas import AtlasInfo
        atlas = None
        if self.atlas and os.path.exists(self.atlas):
            try:
                atlas = AtlasInfo.from_csv(self.atlas)
            except Exception:
                pass
                
        return LoadedDataset(
            data=data,
            pheno=pheno,
            data_kind="timeseries",
            subject_ids=subject_ids,
            atlas=atlas,
            provenance={
                "dir_path": self.dir_path,
                "dir_paths": self.dir_paths,
                "pattern": self.pattern
            }
        )


class ConditionTimeseriesArrayLoader(BaseDataLoader):
    """Loader for paired-condition arrays (e.g. Open-Close) in .npy format."""
    name = "ConditionTimeseriesArrayLoader"
    
    def __init__(
        self,
        data_dir: str,
        subject_order_path: str,
        atlas: str | None = None,
        site: str = "ihb",
        conditions: list[str] | None = None,
        atlas_token: str = "Schaefer200",
        strategy: str = "4",
        gsr: str = "GSR",
        file_template: str = "{site}_{condition}_{atlas_token}_strategy-{strategy}_{gsr}.npy",
        array_shape: str = "subjects_timepoints_rois",
        close_session_policy: str = "first"
    ):
        self.data_dir = data_dir
        self.subject_order_path = subject_order_path
        self.atlas = atlas
        self.site = site
        self.conditions = conditions or ["open", "close"]
        self.atlas_token = atlas_token
        self.strategy = strategy
        self.gsr = gsr
        self.file_template = file_template
        self.array_shape = array_shape
        self.close_session_policy = close_session_policy

    def _get_subject_order(self) -> list[str]:
        if os.path.exists(self.subject_order_path):
            with open(self.subject_order_path, "r") as f:
                return [line.strip() for line in f if line.strip()]
        return []

    def preview(self) -> DatasetPreview:
        sub_list = self._get_subject_order()
        if not sub_list:
            return DatasetPreview(None, None, None, None, "unknown", None, None, [], 0, 0.0, ["Subject order list empty or missing."])
            
        cond = self.conditions[0]
        fname = self.file_template.format(
            site=self.site, condition=cond, atlas_token=self.atlas_token,
            strategy=self.strategy, gsr=self.gsr
        )
        fpath = os.path.join(self.data_dir, fname)
        if not os.path.exists(fpath):
            return DatasetPreview(None, None, None, None, "unknown", None, None, [], 0, 0.0, [f"File not found: {fname}"])
            
        try:
            arr = np.load(fpath)
            n_sub = arr.shape[0]
            n_tp = arr.shape[1]
            n_rois = arr.shape[2]
            
            return DatasetPreview(
                n_subjects=len(sub_list),
                n_observations=len(sub_list) * len(self.conditions),
                n_rois=n_rois,
                n_timepoints=n_tp,
                data_kind_guess="timeseries",
                atlas_guess=os.path.basename(self.atlas) if self.atlas else self.atlas_token,
                conditions=self.conditions,
                subject_ids_sample=sub_list[:5],
                file_count=len(self.conditions),
                file_sizes_mb=os.path.getsize(fpath) * len(self.conditions) / (1024 * 1024)
            )
        except Exception as e:
            return DatasetPreview(None, None, None, None, "unknown", None, None, [], 0, 0.0, [f"Error: {e}"])

    def load(self) -> LoadedDataset:
        sub_list = self._get_subject_order()
        if not sub_list:
            raise FileNotFoundError(f"Subject list not found: {self.subject_order_path}")
            
        all_corr = []
        all_conditions = []
        all_pheno_subs = []
        
        for cond in self.conditions:
            fname = self.file_template.format(
                site=self.site, condition=cond, atlas_token=self.atlas_token,
                strategy=self.strategy, gsr=self.gsr
            )
            fpath = os.path.join(self.data_dir, fname)
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"Missing file for condition '{cond}': {fpath}")
                
            arr = np.load(fpath)
            
            if arr.ndim == 4:
                if self.close_session_policy == "first":
                    arr = arr[:, :, :, 0]
                elif self.close_session_policy == "mean":
                    arr = arr.mean(axis=-1)
                else:
                    raise ValueError(f"Unsupported session policy: {self.close_session_policy}")
            
            corr_mats = np.array([np.nan_to_num(np.corrcoef(ts.T)) for ts in arr])
            
            all_corr.append(corr_mats)
            all_conditions.extend([cond] * len(sub_list))
            all_pheno_subs.extend(sub_list)
            
        data = np.vstack(all_corr)
        pheno = pd.DataFrame({
            "subject_id": all_pheno_subs,
            "condition": all_conditions
        })
        
        from conninfpy.atlas import AtlasInfo
        atlas = None
        if self.atlas and os.path.exists(self.atlas):
            try:
                atlas = AtlasInfo.from_csv(self.atlas)
            except Exception:
                pass
                
        return LoadedDataset(
            data=data,
            pheno=pheno,
            data_kind="correlation",
            subject_ids=all_pheno_subs,
            conditions=self.conditions,
            condition_column="condition",
            atlas=atlas,
            provenance={"data_dir": self.data_dir, "strategy": self.strategy}
        )


class ConnectivityMatrixLoader(BaseDataLoader):
    """Loader for inference-ready connectivity matrices (.npy or .npz)."""
    name = "ConnectivityMatrixLoader"
    
    def __init__(
        self,
        data_path: str | None = None,
        data_paths: dict[str, str] | None = None,
        pheno_path: str | None = None,
        atlas: str | None = None,
        data_kind: Literal["correlation", "fisher_z"] = "fisher_z"
    ):
        self.data_path = data_path
        self.data_paths = data_paths
        self.pheno_path = pheno_path
        self.atlas = atlas
        self.data_kind = data_kind

    def preview(self) -> DatasetPreview:
        if self.data_paths:
            first_grp = list(self.data_paths.keys())[0]
            nl = NumpyLoader(data_path=self.data_paths[first_grp], pheno_path=self.pheno_path, data_kind=self.data_kind)
            prev = nl.preview()
            prev.atlas_guess = os.path.basename(self.atlas) if self.atlas else None
            
            total_observations = 0
            total_sizes = 0.0
            for grp, pth in self.data_paths.items():
                if os.path.exists(pth):
                    try:
                        arr = np.load(pth)
                        total_observations += arr.shape[0]
                        total_sizes += os.path.getsize(pth) / (1024 * 1024)
                    except Exception:
                        pass
            prev.n_observations = total_observations
            prev.n_subjects = total_observations
            prev.conditions = list(self.data_paths.keys())
            prev.file_count = len(self.data_paths)
            prev.file_sizes_mb = total_sizes
            return prev
        else:
            nl = NumpyLoader(data_path=self.data_path, pheno_path=self.pheno_path, data_kind=self.data_kind)
            prev = nl.preview()
            prev.atlas_guess = os.path.basename(self.atlas) if self.atlas else None
            return prev

    def load(self) -> LoadedDataset:
        if self.data_paths:
            all_data = []
            all_groups = []
            all_subject_ids = []
            for grp, pth in self.data_paths.items():
                nl = NumpyLoader(data_path=pth, pheno_path=self.pheno_path, data_kind=self.data_kind)
                ds = nl.load()
                all_data.append(ds.data)
                all_groups.extend([grp] * len(ds.data))
                all_subject_ids.extend([f"{grp}_{sid}" for sid in ds.subject_ids])
                
            data = np.vstack(all_data)
            pheno = pd.DataFrame({
                "subject_id": all_subject_ids,
                "group": all_groups
            })
            
            from conninfpy.atlas import AtlasInfo
            atlas = None
            if self.atlas and os.path.exists(self.atlas):
                try:
                    atlas = AtlasInfo.from_csv(self.atlas)
                except Exception:
                    pass
                    
            return LoadedDataset(
                data=data,
                pheno=pheno,
                data_kind=self.data_kind,
                subject_ids=all_subject_ids,
                atlas=atlas,
                provenance={"data_paths": self.data_paths}
            )
        else:
            nl = NumpyLoader(data_path=self.data_path, pheno_path=self.pheno_path, data_kind=self.data_kind)
            dataset = nl.load()
            
            from conninfpy.atlas import AtlasInfo
            if self.atlas and os.path.exists(self.atlas):
                try:
                    dataset.atlas = AtlasInfo.from_csv(self.atlas)
                    dataset.roi_labels = dataset.atlas.labels
                except Exception:
                    pass
                    
            return dataset


class CustomPreparedZerssenLoader(BaseDataLoader):
    """Loader for precomputed Zerssen Prepared set84 NPZ datasets."""
    name = "CustomPreparedZerssenLoader"
    
    def __init__(
        self,
        prepared_npz: str,
        set84_bnt_ids: str,
        atlas_metadata: str,
        matrix_key: str = "set84m_pearson_z",
        data_kind: str = "fisher_z",
    ):
        self.prepared_npz = prepared_npz
        self.set84_bnt_ids = set84_bnt_ids
        self.atlas_metadata = atlas_metadata
        self.matrix_key = matrix_key
        self.data_kind = data_kind

    @staticmethod
    def _available_connectivity_keys(archive) -> list[str]:
        """Return square subject-by-ROI-by-ROI arrays stored in the archive."""
        return [
            key for key in archive.files
            if archive[key].ndim == 3 and archive[key].shape[1] == archive[key].shape[2]
        ]

    def _selected_matrix(self, archive) -> tuple[str, np.ndarray]:
        available = self._available_connectivity_keys(archive)
        if self.matrix_key not in available:
            raise ValueError(
                f"Zerssen matrix_key {self.matrix_key!r} is unavailable or not a square "
                f"connectivity tensor. Available matrix keys: {available}"
            )
        return self.matrix_key, np.asarray(archive[self.matrix_key])

    def matrix_options(self) -> list[dict[str, object]]:
        """Describe selectable matrices for an optional manifest UI control."""
        if not os.path.exists(self.prepared_npz):
            return []
        with np.load(self.prepared_npz) as archive:
            options = []
            for key in self._available_connectivity_keys(archive):
                n_rois = int(archive[key].shape[1])
                if n_rois not in {84, 246}:
                    continue
                data_kind = "fisher_z" if key.endswith("pearson_z") else "correlation"
                atlas_label = "BNA-246 set84 subset" if n_rois == 84 else "BNA-246 full atlas"
                options.append({
                    "key": key,
                    "n_rois": n_rois,
                    "data_kind": data_kind,
                    "label": f"{key} ({n_rois} ROIs; {atlas_label})",
                })
            return options

    def set_matrix_key(self, matrix_key: str) -> dict[str, object]:
        """Apply a UI-selected matrix and its corresponding input scale."""
        options = {option["key"]: option for option in self.matrix_options()}
        if matrix_key not in options:
            raise ValueError(f"Unsupported Zerssen matrix selection: {matrix_key!r}")
        selected = options[matrix_key]
        self.matrix_key = matrix_key
        self.data_kind = str(selected["data_kind"])
        return selected

    def _set84_roi_ids(self, archive) -> list[int]:
        """Read the 1-based BNA IDs in the exact prepared-matrix order."""
        if "set84_bnt_ids" in archive:
            return [int(value) for value in np.asarray(archive["set84_bnt_ids"]).ravel()]
        if not os.path.exists(self.set84_bnt_ids):
            return []
        try:
            import json
            with open(self.set84_bnt_ids, "r") as file_handle:
                metadata = json.load(file_handle)
            values = metadata.get("bnt_ids", []) if isinstance(metadata, dict) else metadata
            return [int(value) for value in values]
        except (OSError, TypeError, ValueError):
            return []

    def preview(self) -> DatasetPreview:
        if not os.path.exists(self.prepared_npz):
            return DatasetPreview(None, None, None, None, "unknown", None, None, [], 0, 0.0, ["Prepared NPZ not found."])
        try:
            d = np.load(self.prepared_npz)
            matrix_key, data = self._selected_matrix(d)
            keys = self._available_connectivity_keys(d)
            subject_ids = [f"sub_{i:02d}" for i in range(len(data))]
            
            return DatasetPreview(
                n_subjects=len(subject_ids),
                n_observations=len(subject_ids),
                n_rois=data.shape[1],
                n_timepoints=None,
                data_kind_guess=self.data_kind,
                atlas_guess=(
                    "BNA-246" if data.shape[1] == 246
                    else "BNA-246 set84 subset" if data.shape[1] == 84
                    else os.path.basename(self.atlas_metadata)
                ),
                conditions=None,
                subject_ids_sample=subject_ids[:5],
                file_count=1,
                file_sizes_mb=os.path.getsize(self.prepared_npz) / (1024 * 1024),
                warnings=[
                    f"Selected matrix: {matrix_key} ({data.shape[1]} ROIs).",
                    f"Available connectivity matrices: {keys}",
                ]
            )
        except Exception as e:
            return DatasetPreview(None, None, None, None, "unknown", None, None, [], 0, 0.0, [f"Error: {e}"])

    def load(self) -> LoadedDataset:
        if not os.path.exists(self.prepared_npz):
            raise FileNotFoundError(f"Prepared NPZ not found: {self.prepared_npz}")
            
        d = np.load(self.prepared_npz)
        matrix_key, data = self._selected_matrix(d)
        
        subject_ids = [f"sub_{i:02d}" for i in range(len(data))]
        groups = d["group"].tolist() if "group" in d and len(d["group"]) == len(data) else []
        if not groups:
            groups = ["PAT"] * 20 + ["HC"] * 41
            
        pheno = pd.DataFrame({
            "subject_id": subject_ids,
            "group": groups
        })
        # The prepared archive includes subject-aligned clinical, demographic,
        # and behavioral scores that can serve as GLM predictors or covariates.
        for column in ("age", "sex", "bdi", "hads_d", "hads_a", "zerssen", "ds_rt", "dn_rt"):
            if column not in d.files:
                continue
            values = np.asarray(d[column])
            if values.ndim == 1 and len(values) == len(data):
                pheno[column] = pd.to_numeric(values, errors="coerce")
        
        from conninfpy.atlas import AtlasInfo
        atlas = None
        if os.path.exists(self.atlas_metadata):
            try:
                roi_ids = self._set84_roi_ids(d)
                atlas_df = pd.read_csv(self.atlas_metadata)
                if data.shape[1] == 84:
                    if len(roi_ids) != data.shape[1]:
                        raise ValueError(
                            "Prepared set84 matrix requires 84 ROI IDs; "
                            f"found {len(roi_ids)}."
                        )
                    id_column = "roi_id" if "roi_id" in atlas_df.columns else "id"
                    if id_column not in atlas_df.columns:
                        raise ValueError(
                            "BNA atlas metadata must include an 'roi_id' or 'id' column."
                        )
                    atlas_df = atlas_df.set_index(id_column).reindex(roi_ids)
                    if atlas_df.isna().all(axis=1).any():
                        missing = [
                            roi_id for roi_id, row in atlas_df.iterrows()
                            if row.isna().all()
                        ]
                        raise ValueError(f"BNA atlas is missing set84 ROI IDs: {missing}")
                    atlas = AtlasInfo(
                        labels=atlas_df["name"].astype(str).tolist(),
                        networks=atlas_df["network"].astype(str).tolist(),
                        coords=atlas_df[["x", "y", "z"]].to_numpy(dtype=float),
                        hemisphere=atlas_df["hemisphere"].astype(str).tolist(),
                        source="Zerssen prepared set84 BNA subset",
                    )
                elif data.shape[1] == 246:
                    atlas = AtlasInfo.from_csv(self.atlas_metadata)
                else:
                    raise ValueError(
                        f"Matrix {matrix_key!r} has {data.shape[1]} ROIs. "
                        "This manifest supplies BNA metadata only for the 84-ROI subset "
                        "or the full 246-ROI atlas."
                    )
            except Exception as exc:
                raise ValueError(
                    "Failed to construct atlas metadata for the prepared Zerssen matrix: "
                    f"{exc}"
                ) from exc
                
        return LoadedDataset(
            data=data,
            pheno=pheno,
            data_kind=self.data_kind,
            subject_ids=subject_ids,
            atlas=atlas,
            provenance={
                "prepared_npz": self.prepared_npz,
                "matrix_key": matrix_key,
                "phenotype_columns": list(pheno.columns),
            }
        )


class ChinaCloseCloseLoader(BaseDataLoader):
    """Loader for China Eyes-Closed Run 1 vs Run 2 (test-retest) comparison."""
    name = "ChinaCloseCloseLoader"
    
    def __init__(self, close_path: str, subject_list_path: str | None = None, atlas: str | None = None, drop_missing_rois: bool = True):
        self.close_path = close_path
        self.subject_list_path = subject_list_path
        self.atlas = atlas
        self.drop_missing_rois = drop_missing_rois
        
    def _get_kept_indices(self) -> np.ndarray | None:
        if self.drop_missing_rois and self.atlas and os.path.exists(self.atlas):
            try:
                df = pd.read_csv(self.atlas)
                if "id" in df.columns:
                    return df["id"].values - 1
            except Exception:
                pass
        return None

    def preview(self) -> DatasetPreview:
        if not os.path.exists(self.close_path):
            return DatasetPreview(None, None, None, None, "unknown", None, None, [], 0, 0.0, ["Close file path not found."])
        try:
            close_raw = np.load(self.close_path)
            if close_raw.ndim != 4 or close_raw.shape[-1] != 2:
                return DatasetPreview(None, None, None, None, "unknown", None, None, [], 0, 0.0, ["Expected 4D array with 2 runs."])
            n_sub, n_tp, n_rois, n_runs = close_raw.shape
            
            kept_idx = self._get_kept_indices()
            if kept_idx is not None:
                n_rois = len(kept_idx)
                
            subject_ids = []
            if self.subject_list_path and os.path.exists(self.subject_list_path):
                with open(self.subject_list_path, 'r') as f:
                    subject_ids = [line.strip() for line in f if line.strip()]
            else:
                subject_ids = [f"sub_{i:02d}" for i in range(n_sub)]
                
            return DatasetPreview(
                n_subjects=n_sub,
                n_observations=n_sub * 2,
                n_rois=n_rois,
                n_timepoints=n_tp,
                data_kind_guess="timeseries",
                atlas_guess=os.path.basename(self.atlas) if self.atlas else f"Schaefer-{n_rois}",
                conditions=["close_run1", "close_run2"],
                subject_ids_sample=subject_ids[:5],
                file_count=1,
                file_sizes_mb=os.path.getsize(self.close_path) / (1024 * 1024)
            )
        except Exception as e:
            return DatasetPreview(None, None, None, None, "unknown", None, None, [], 1, None, [f"Error previewing close_close: {e}"])

    def load(self) -> LoadedDataset:
        close_raw = np.load(self.close_path)
        if close_raw.ndim != 4 or close_raw.shape[-1] != 2:
            raise ValueError(f"Expected 4D array with 2 runs; got shape {close_raw.shape}")
            
        run0_data = close_raw[..., 0]
        run1_data = close_raw[..., 1]
        
        n_sub, n_tp, n_rois = run0_data.shape
        
        kept_idx = self._get_kept_indices()
        if kept_idx is not None:
            run0_data = run0_data[:, :, kept_idx]
            run1_data = run1_data[:, :, kept_idx]
            n_rois = len(kept_idx)
            
        subject_ids = []
        if self.subject_list_path and os.path.exists(self.subject_list_path):
            with open(self.subject_list_path, 'r') as f:
                subject_ids = [line.strip() for line in f if line.strip()]
        else:
            subject_ids = [f"sub_{i:02d}" for i in range(n_sub)]
            
        if len(subject_ids) != n_sub:
            subject_ids = [f"sub_{i:02d}" for i in range(n_sub)]
            
        keep_mask = np.ones(n_sub, dtype=bool)
        for idx, sid in enumerate(subject_ids):
            if sid == "sub-3258811":
                keep_mask[idx] = False
                
        run0_data = run0_data[keep_mask]
        run1_data = run1_data[keep_mask]
        subject_ids = [sid for idx, sid in enumerate(subject_ids) if keep_mask[idx]]
        n_sub = len(subject_ids)
        
        run0_corr = np.array([np.nan_to_num(np.corrcoef(ts.T)) for ts in run0_data])
        run1_corr = np.array([np.nan_to_num(np.corrcoef(ts.T)) for ts in run1_data])
        
        data = np.vstack([run0_corr, run1_corr])
        pheno = pd.DataFrame({
            "subject_id": subject_ids * 2,
            "condition": ["close_run1"] * n_sub + ["close_run2"] * n_sub
        })
        
        atlas = None
        if self.atlas and os.path.exists(self.atlas):
            from conninfpy.atlas import AtlasInfo
            try:
                atlas = AtlasInfo.from_csv(self.atlas)
            except Exception:
                pass
                
        return LoadedDataset(
            data=data,
            pheno=pheno,
            data_kind="correlation",
            subject_ids=subject_ids * 2,
            conditions=["close_run1", "close_run2"],
            condition_column="condition",
            atlas=atlas,
            provenance={"close_path": self.close_path}
        )
