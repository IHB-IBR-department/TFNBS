from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal
import numpy as np
import pandas as pd

from conninfpy.atlas import AtlasInfo

@dataclass
class LoadedDataset:
    """Standardized return container for all ConnInfPy data loaders.
    
    Attributes
    ----------
    data : np.ndarray
        Either a 3D timeseries tensor of shape (n_subjects, n_timepoints, n_rois)
        or a 3D connectivity tensor of shape (n_subjects, n_rois, n_rois).
    pheno : pd.DataFrame
        Phenotypic metadata. Must have one row per subject/observation matching `data`.
    data_kind : Literal["timeseries", "correlation", "fisher_z"]
        The type of data stored in `data`.
    subject_ids : list[str] | None
        Explicit list of subject identifiers corresponding to axis 0 of `data`.
    atlas : AtlasInfo | None
        Optional parcellation metadata associated with the dataset.
    roi_labels : list[str] | None
        Optional list of ROI name strings.
    conditions : list[str] | None
        List of unique condition names (e.g. ['open', 'close']) present in the dataset.
    condition_column : str | None
        The column in `pheno` containing the condition variable.
    provenance : dict[str, Any]
        Metadata describing source files, loader version, and runtime options.
    warnings : list[str]
        Warnings accumulated during data loading.
    """
    data: np.ndarray
    pheno: pd.DataFrame
    data_kind: Literal["timeseries", "correlation", "fisher_z"]
    subject_ids: list[str] | None = None
    atlas: AtlasInfo | None = None
    roi_labels: list[str] | None = None
    conditions: list[str] | None = None
    condition_column: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class DatasetPreview:
    """Lightweight metadata summary shown before executing a full load."""
    n_subjects: int | None
    n_observations: int | None
    n_rois: int | None
    n_timepoints: int | tuple[int, int] | None  # Single int or (min, max) for variable length
    data_kind_guess: Literal["timeseries", "correlation", "fisher_z", "unknown"]
    atlas_guess: str | None
    conditions: list[str] | None
    subject_ids_sample: list[str]
    file_count: int | None
    file_sizes_mb: float | None
    warnings: list[str] = field(default_factory=list)


@dataclass
class DataValidationReport:
    """Report detailing output validation checks."""
    ok: bool
    errors: list[str]
    warnings: list[str]
    summary: dict[str, Any]


class BaseDataLoader(ABC):
    """Abstract Base Class defining the contract for all dataset loaders.
    
    All custom user or built-in data loaders should subclass this.
    """
    name: str = "BaseDataLoader"

    @abstractmethod
    def preview(self) -> DatasetPreview:
        """Cheaply inspect files or metadata to return a lightweight summary."""
        pass

    @abstractmethod
    def load(self) -> LoadedDataset | dict[str, LoadedDataset]:
        """Perform the actual loading and return the standardized LoadedDataset."""
        pass
