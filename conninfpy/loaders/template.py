"""Template file illustrating how to write a custom DataLoader for ConnInfPy.

Use this file as a template when writing your own dataset loaders or
when instructing an AI agent to write one for you.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from typing import Literal

from conninfpy.atlas import AtlasInfo
from conninfpy.loaders.base import BaseDataLoader, LoadedDataset, DatasetPreview
from conninfpy.loaders.validation import validate_loaded_dataset

class CustomTemplateLoader(BaseDataLoader):
    """A heavily commented template class subclassing BaseDataLoader.
    
    Implementations must implement:
      1. preview(): Fast inspection of data files to extract shapes, counts, and guess format.
      2. load(): The full loading process returning a LoadedDataset.
    """
    name = "CustomTemplateLoader"

    def __init__(self, data_path: str, pheno_path: str | None = None):
        """Initialize your loader with paths or extraction parameters.
        
        CRITICAL: Never hard-code absolute user paths or API keys in the class code.
        Always pass them as constructor arguments.
        """
        self.data_path = data_path
        self.pheno_path = pheno_path

    def preview(self) -> DatasetPreview:
        """Lightweight preview of the data.
        
        This method must be fast! Do not load large arrays or run slow extraction.
        Simply read metadata, file headers, or scan directories.
        """
        warnings = []
        if not os.path.exists(self.data_path):
            warnings.append(f"Source path not found: {self.data_path}")
            return DatasetPreview(
                n_subjects=None, n_observations=None, n_rois=None, n_timepoints=None,
                data_kind_guess="unknown", atlas_guess=None, conditions=None,
                subject_ids_sample=[], file_count=0, file_sizes_mb=0.0, warnings=warnings
            )
            
        # Example: Guess sizes/subjects based on a single file or directory scanning
        # Let's assume we scan a directory or file size
        file_size_mb = os.path.getsize(self.data_path) / (1024 * 1024)
        
        return DatasetPreview(
            n_subjects=1,  # Guess or parse from files
            n_observations=1,
            n_rois=100,  # Guess or parse from header
            n_timepoints=200,
            data_kind_guess="timeseries",  # 'timeseries', 'correlation', or 'fisher_z'
            atlas_guess="Schaefer-100",  # Guess from file names/sizes
            conditions=None,  # list of strings if multi-condition
            subject_ids_sample=["sample_sub_01"],
            file_count=1,
            file_sizes_mb=file_size_mb,
            warnings=warnings
        )

    def load(self) -> LoadedDataset:
        """Load the full dataset into memory.
        
        Perform the full data extraction, load files, align subjects, and construct
        the phenotypic dataframe.
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Source path does not exist: {self.data_path}")
            
        # 1. Load or compute the connectivity/timeseries array (must be 3D numpy array)
        # Shape: (subjects, timepoints, rois) for timeseries
        # Shape: (subjects, rois, rois) for correlation/fisher_z
        data = np.zeros((1, 200, 100)) # Placeholder: replace with actual load
        
        # 2. Build or load the phenotypic metadata DataFrame
        # Pheno must contain a 'subject_id' column
        subject_ids = ["sample_sub_01"]
        pheno = pd.DataFrame({
            "subject_id": subject_ids,
            "group_interest": [0],  # Example variable of interest
            "age": [45]
        })
        
        # 3. Create the LoadedDataset container
        dataset = LoadedDataset(
            data=data,
            pheno=pheno,
            data_kind="timeseries",  # Must match data representation
            subject_ids=subject_ids,
            atlas=AtlasInfo.schaefer_100_yeo7(),  # Optional: attach parcellation info
            provenance={
                "loader_name": self.name,
                "data_path": self.data_path,
                "pheno_path": self.pheno_path
            }
        )
        
        # 4. CRITICAL: Run validation before returning to check for structural bugs
        report = validate_loaded_dataset(dataset)
        if not report.ok:
            raise ValueError(f"Loaded dataset failed validation: {report.errors}")
            
        return dataset
