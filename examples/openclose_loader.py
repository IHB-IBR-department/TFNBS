"""
OpenClose data loader for HCP and Schaefer200 atlases.

Provides unified loading interface for Open/Close resting-state data
with network labels and validation.

Usage:
    from openclose_loader import OpenCloseDataset

    ds = OpenCloseDataset.hcp("ihb")
    open_z, close_z = ds.fisher_z()
    net_labels = ds.get_net_labels("cole")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd

# Base path for datasets (relative to project root)
DATASETS_DIR = Path(__file__).parent.parent / "datasets" / "OpenClose"


@dataclass
class OpenCloseDataset:
    """Container for Open/Close connectivity data with labels."""

    open_data: np.ndarray  # (n_subjects, n_nodes, n_nodes)
    close_data: np.ndarray  # (n_subjects, n_nodes, n_nodes)
    atlas_df: pd.DataFrame  # Atlas description with labels
    atlas_type: str  # "hcp" or "sch200"
    experiment: str  # "ihb" or "rmet"

    @classmethod
    def hcp(cls, experiment: Literal["ihb", "rmet"] = "ihb") -> "OpenCloseDataset":
        """
        Load HCP atlas data (373 ROIs).

        Parameters
        ----------
        experiment : str
            "ihb" (n=84) or "rmet" (n=63)

        Returns
        -------
        OpenCloseDataset
        """
        data_dir = DATASETS_DIR / "HCP" / "corr"
        atlas_path = DATASETS_DIR / "HCP" / "HCPex_atlas_description.xlsx"

        open_data = np.load(data_dir / f"open_{experiment}.npy")
        close_data = np.load(data_dir / f"close_{experiment}.npy")

        atlas_df = pd.read_excel(atlas_path)
        atlas_df = atlas_df.sort_values("NEW_ID").reset_index(drop=True)

        return cls(
            open_data=open_data,
            close_data=close_data,
            atlas_df=atlas_df,
            atlas_type="hcp",
            experiment=experiment,
        )

    @classmethod
    def schaefer200(cls, experiment: Literal["ihb", "rmet"] = "ihb") -> "OpenCloseDataset":
        """
        Load Schaefer200 atlas data (200 ROIs, Yeo 7 Networks).

        Parameters
        ----------
        experiment : str
            "ihb" or "rmet"

        Returns
        -------
        OpenCloseDataset
        """
        data_dir = DATASETS_DIR / "data_sch200"
        labels_path = data_dir / "labels.csv"

        open_data = np.load(data_dir / f"opened_{experiment}.npy")
        close_data = np.load(data_dir / f"closed_{experiment}.npy")

        atlas_df = pd.read_csv(labels_path)

        return cls(
            open_data=open_data,
            close_data=close_data,
            atlas_df=atlas_df,
            atlas_type="sch200",
            experiment=experiment,
        )

    @property
    def n_subjects(self) -> int:
        return self.open_data.shape[0]

    @property
    def n_nodes(self) -> int:
        return self.open_data.shape[1]

    @property
    def n_edges(self) -> int:
        """Number of unique edges (upper triangle)."""
        n = self.n_nodes
        return n * (n - 1) // 2

    def get_net_labels(
        self, scheme: Literal["cortical", "cole", "yeo7"] = "cole"
    ) -> np.ndarray:
        """
        Get network labels as contiguous integers 0..K-1.

        Parameters
        ----------
        scheme : str
            For HCP: "cortical" (24 divisions) or "cole" (14 networks)
            For Schaefer200: "yeo7" (7 networks)

        Returns
        -------
        net_labels : (n_nodes,) array of int
        """
        if self.atlas_type == "hcp":
            if scheme == "cortical":
                raw_labels = self.atlas_df["Cortical_Division_Number"].values
            elif scheme == "cole":
                raw_labels = self.atlas_df["ColeAnticevic_functional_network"].values
            else:
                raise ValueError(f"Unknown scheme for HCP: {scheme}")
        else:  # sch200
            if scheme != "yeo7":
                raise ValueError(f"Schaefer200 only supports 'yeo7', got: {scheme}")
            raw_labels = self.atlas_df["network"].values

        # Remap to contiguous 0..K-1
        _, net_labels = np.unique(raw_labels, return_inverse=True)
        return net_labels

    def get_label_names(
        self, scheme: Literal["cortical", "cole", "yeo7"] = "cole"
    ) -> list[str]:
        """Get unique label names in order."""
        if self.atlas_type == "hcp":
            if scheme == "cortical":
                col = "Cortical Division"
            else:
                col = "ColeAnticevic_functional_network_label"
            # Get unique values preserving order
            seen = {}
            for val in self.atlas_df[col]:
                if val not in seen:
                    seen[val] = len(seen)
            return list(seen.keys())
        else:
            return self.atlas_df["network"].unique().tolist()

    def fisher_z(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return Fisher z-transformed data.

        Note: HCP data is already Fisher z-transformed per corr_code.txt.
        This method is provided for API consistency.

        Returns
        -------
        open_z, close_z : arrays of shape (n_subjects, n_nodes, n_nodes)
        """
        # Data is already z-transformed for HCP
        return self.open_data.copy(), self.close_data.copy()

    def validate(self) -> None:
        """
        Validate connectivity matrices.

        Raises
        ------
        AssertionError
            If validation fails
        """
        for name, data in [("open", self.open_data), ("close", self.close_data)]:
            # Symmetry
            if not np.allclose(data, data.transpose(0, 2, 1)):
                raise AssertionError(f"{name} data is not symmetric")

            # Zero diagonal
            for i in range(data.shape[0]):
                if not np.allclose(np.diag(data[i]), 0):
                    raise AssertionError(f"{name} data has non-zero diagonal")

            # Finite values
            if not np.isfinite(data).all():
                raise AssertionError(f"{name} data contains NaN/Inf")

        # Shape consistency
        if self.open_data.shape != self.close_data.shape:
            raise AssertionError("open and close data have different shapes")

        # Labels match matrix size
        if len(self.atlas_df) != self.n_nodes:
            raise AssertionError(
                f"Atlas has {len(self.atlas_df)} rows but matrix has {self.n_nodes} nodes"
            )

        print(f"Validation passed:")
        print(f"  Subjects: {self.n_subjects}")
        print(f"  Nodes: {self.n_nodes}")
        print(f"  Edges: {self.n_edges}")
        print(f"  Value range: [{self.open_data.min():.3f}, {self.open_data.max():.3f}]")

    def to_features(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert to ML feature matrix format.

        Returns
        -------
        X : (2*n_subjects, n_edges) feature matrix
        y : (2*n_subjects,) labels (0=open, 1=close)
        subject_ids : (2*n_subjects,) subject identifiers
        """
        tri = np.triu_indices(self.n_nodes, k=1)

        # Vectorize upper triangle
        X_open = self.open_data[:, tri[0], tri[1]]
        X_close = self.close_data[:, tri[0], tri[1]]

        X = np.vstack([X_open, X_close])
        y = np.array([0] * self.n_subjects + [1] * self.n_subjects)
        subject_ids = np.array(list(range(self.n_subjects)) * 2)

        return X, y, subject_ids


def load_both_experiments(
    atlas: Literal["hcp", "sch200"] = "hcp"
) -> Tuple[OpenCloseDataset, OpenCloseDataset]:
    """
    Load both IHB and RMET datasets.

    Returns
    -------
    ds_ihb, ds_rmet : OpenCloseDataset instances
    """
    loader = OpenCloseDataset.hcp if atlas == "hcp" else OpenCloseDataset.schaefer200
    return loader("ihb"), loader("rmet")


if __name__ == "__main__":
    # Quick validation
    print("=== HCP IHB ===")
    ds = OpenCloseDataset.hcp("ihb")
    ds.validate()
    print(f"Cole networks: {len(np.unique(ds.get_net_labels('cole')))} unique")

    print("\n=== HCP RMET ===")
    ds = OpenCloseDataset.hcp("rmet")
    ds.validate()

    print("\n=== Schaefer200 IHB ===")
    ds = OpenCloseDataset.schaefer200("ihb")
    ds.validate()
    print(f"Yeo7 networks: {len(np.unique(ds.get_net_labels('yeo7')))} unique")
