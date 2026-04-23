"""
Open-Close data loader — Schaefer-200 with 182-ROI unified mask.

This loader provides the IHB and China (Beijing) Eyes-Open / Eyes-Closed
resting-state time series pre-processed via pipeline ``strategy-4_GSR``
from the BenchmarkingEOEC project (Medvedeva et al. 2025,
bioRxiv 10.1101/2025.10.20.683049).

Data layout on disk
-------------------
``datasets/open_close/``:

- ``ihb_{open,close}_Schaefer200_strategy-4_GSR.npy`` — shape (84, 120, 200) each
- ``china_open_Schaefer200_strategy-4_GSR.npy``       — shape (48, 240, 200)
- ``china_close_Schaefer200_strategy-4_GSR.npy``      — shape (48, 240, 200, 2)
  where the last axis is two independent closed-eyes runs (test-retest).
- ``schaefer_labels.csv`` — 182 valid ROIs with Yeo-7 network labels
- ``subject_order_{ihb,china}.txt`` — subject IDs aligned to row index

The Schaefer-200 atlas is applied at a 10 % voxel-coverage threshold; 18
ROIs (IDs 5, 55–59, 64, 75, 86, 159–164, 168, 186, 191) fail coverage in
IHB and are dropped from both cohorts so edges are index-aligned across
sites. See ``schaefer_labels.csv`` for the 182 retained ROIs.

Known data-quality issue
------------------------
China subject ``sub-3258811`` has run-3 (close second run) incomplete —
only 53/240 TRs (77.9 % zeros). Exclude when analysing both close runs.
The helper ``bad_retest_subjects()`` reports this subject ID.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd

from conninfpy import fisher_r_to_z


__all__ = [
    "OpenCloseDataset",
    "DATA_DIR",
    "MISSED_ROI_IDS",
    "bad_retest_subjects",
]


# Absolute path on disk (relative to repo root)
DATA_DIR = Path(__file__).resolve().parents[2] / "datasets" / "open_close"

# 1-indexed Schaefer-200 ROI IDs that fail the 10 % coverage threshold in IHB.
# Complement is the 182 IDs present in schaefer_labels.csv.
MISSED_ROI_IDS: Tuple[int, ...] = (
    5, 55, 56, 57, 58, 59, 64, 75, 86,
    159, 160, 161, 162, 163, 164, 168, 186, 191,
)


def _missed_col_indices() -> np.ndarray:
    """0-indexed columns of the on-disk 200-col arrays that correspond to MISSED_ROI_IDS."""
    return np.array([i - 1 for i in MISSED_ROI_IDS], dtype=int)


def _keep_col_mask(n_total: int = 200) -> np.ndarray:
    """Boolean length-200 mask: True for columns to keep (182 True, 18 False)."""
    mask = np.ones(n_total, dtype=bool)
    mask[_missed_col_indices()] = False
    return mask


def bad_retest_subjects() -> Tuple[str, ...]:
    """China subjects whose second close run is incomplete. Exclude from §3.6 test-retest."""
    return ("sub-3258811",)


@dataclass
class OpenCloseDataset:
    """Container for Open/Close time-series data with labels, post-coverage-drop.

    Attributes
    ----------
    experiment : "ihb" or "china"
    open_ts : (n_subjects, T, n_rois) time series (eyes open)
    close_ts : (n_subjects, T, n_rois) time series (eyes close, primary run)
    close_ts_run1 : (n_subjects, T, n_rois) or None — China only; IHB is None
    atlas_df : pd.DataFrame with columns (id, name, network), 182 rows
    net_labels : (n_rois,) int array 0..K-1 (contiguous Yeo-7 assignments)
    subject_ids : list[str] of length n_subjects (row-aligned)
    n_rois : effective ROI count (182 after coverage drop, or 200 if not dropped)
    """

    experiment: str
    open_ts: np.ndarray
    close_ts: np.ndarray
    close_ts_run1: Optional[np.ndarray]
    atlas_df: pd.DataFrame
    net_labels: np.ndarray
    subject_ids: list
    n_rois: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_rois = int(self.open_ts.shape[-1])

    # -----------------------------------------------------------------
    # Loading
    # -----------------------------------------------------------------

    @classmethod
    def load(
        cls,
        experiment: Literal["ihb", "china"],
        drop_missing_rois: bool = True,
        data_dir: Optional[Path] = None,
    ) -> "OpenCloseDataset":
        """Load time-series data + labels + subject IDs for one experiment.

        Parameters
        ----------
        experiment : "ihb" or "china"
        drop_missing_rois : if True (default), drop the 18 low-coverage ROIs
            so both cohorts are index-aligned at 182 ROIs.
        data_dir : override the default ``datasets/open_close/`` path.
        """
        base = Path(data_dir) if data_dir is not None else DATA_DIR
        if experiment not in ("ihb", "china"):
            raise ValueError(f"experiment must be 'ihb' or 'china', got {experiment!r}")

        # -- time series ---------------------------------------------
        open_path = base / f"{experiment}_open_Schaefer200_strategy-4_GSR.npy"
        close_path = base / f"{experiment}_close_Schaefer200_strategy-4_GSR.npy"
        open_ts = np.load(open_path).astype(np.float64, copy=False)
        close_raw = np.load(close_path).astype(np.float64, copy=False)

        if experiment == "china":
            # close_raw shape: (48, 240, 200, 2) — two runs
            if close_raw.ndim != 4 or close_raw.shape[-1] != 2:
                raise RuntimeError(
                    f"Expected china close shape (..., 2); got {close_raw.shape}"
                )
            close_ts = close_raw[..., 0]
            close_ts_run1 = close_raw[..., 1]
        else:
            close_ts = close_raw
            close_ts_run1 = None

        # -- coverage drop -------------------------------------------
        if drop_missing_rois:
            keep = _keep_col_mask(open_ts.shape[-1])
            open_ts = open_ts[..., keep]
            close_ts = close_ts[..., keep]
            if close_ts_run1 is not None:
                close_ts_run1 = close_ts_run1[..., keep]

        # -- labels --------------------------------------------------
        # schaefer_labels.csv on disk lists the 182 retained ROIs (the 18
        # low-coverage IDs aren't in it). When drop_missing_rois=False, the
        # TS arrays have 200 columns but labels are still only for 182 —
        # net_labels length will mismatch n_rois. That mode is for diagnostic
        # inspection only; normal use is drop_missing_rois=True.
        atlas_df = pd.read_csv(base / "schaefer_labels.csv")
        if drop_missing_rois and len(atlas_df) != 182:
            raise RuntimeError(
                f"schaefer_labels.csv has {len(atlas_df)} rows, expected 182. "
                f"If the file was updated, re-verify MISSED_ROI_IDS."
            )
        _, inverse = np.unique(atlas_df["network"].values, return_inverse=True)
        net_labels = inverse.astype(int)

        # -- subject IDs ---------------------------------------------
        with (base / f"subject_order_{experiment}.txt").open() as f:
            subject_ids = [line.strip() for line in f if line.strip()]
        if len(subject_ids) != open_ts.shape[0]:
            raise RuntimeError(
                f"subject_order_{experiment}.txt has {len(subject_ids)} IDs but data "
                f"has {open_ts.shape[0]} subjects"
            )

        return cls(
            experiment=experiment,
            open_ts=open_ts,
            close_ts=close_ts,
            close_ts_run1=close_ts_run1,
            atlas_df=atlas_df,
            net_labels=net_labels,
            subject_ids=subject_ids,
        )

    # -----------------------------------------------------------------
    # Per-subject connectivity
    # -----------------------------------------------------------------

    @staticmethod
    def _ts_to_fisher_z(ts: np.ndarray) -> np.ndarray:
        """(n_subjects, T, n_rois) → (n_subjects, n_rois, n_rois) Fisher-z.

        Per-subject Pearson correlation (zeroes the diagonal) then Fisher z.
        Clips r to [-0.9999, 0.9999] to avoid atanh infinities on degenerate
        time series.
        """
        n_subjects, T, N = ts.shape
        out = np.empty((n_subjects, N, N), dtype=np.float64)
        for s in range(n_subjects):
            c = np.corrcoef(ts[s].T)  # (N, N)
            np.fill_diagonal(c, 0.0)
            out[s] = np.clip(c, -0.9999, 0.9999)
        return fisher_r_to_z(out)

    def connectivity_z(
        self, run: Literal[0, 1] = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Per-subject Fisher-z correlation matrices for (open, close).

        Parameters
        ----------
        run : 0 or 1 — selects the China close run. IHB only has run=0.

        Returns
        -------
        (open_z, close_z) each of shape (n_subjects, n_rois, n_rois).
        """
        if run not in (0, 1):
            raise ValueError(f"run must be 0 or 1, got {run}")
        open_z = self._ts_to_fisher_z(self.open_ts)
        if run == 0:
            close_ts = self.close_ts
        else:
            if self.close_ts_run1 is None:
                raise ValueError(
                    f"{self.experiment} has no close run 1 (IHB has only one close run)."
                )
            close_ts = self.close_ts_run1
        close_z = self._ts_to_fisher_z(close_ts)
        return open_z, close_z

    def fisher_z(self) -> Tuple[np.ndarray, np.ndarray]:
        """Backward-compatible alias for ``connectivity_z(run=0)``."""
        return self.connectivity_z(run=0)

    # -----------------------------------------------------------------
    # Accessors
    # -----------------------------------------------------------------

    @property
    def n_subjects(self) -> int:
        return int(self.open_ts.shape[0])

    @property
    def n_timepoints(self) -> int:
        return int(self.open_ts.shape[1])

    @property
    def n_edges(self) -> int:
        return self.n_rois * (self.n_rois - 1) // 2

    def get_net_labels(self) -> np.ndarray:
        """Contiguous Yeo-7 integer labels of length n_rois."""
        return self.net_labels

    def get_label_names(self) -> list:
        """Yeo-7 network names in the order assigned by net_labels (0..K-1)."""
        uniq, idx = np.unique(self.atlas_df["network"].values, return_index=True)
        # np.unique returns sorted; inverse mapping in __init__ used same sort
        order = np.argsort(idx)  # would restore first-appearance order, but not needed
        return list(uniq)

    def retest_subject_mask(self) -> np.ndarray:
        """Boolean (n_subjects,) mask for subjects valid for §3.6 test-retest.

        False for subjects in :func:`bad_retest_subjects`; True otherwise.
        """
        bad = set(bad_retest_subjects())
        return np.array([sid not in bad for sid in self.subject_ids], dtype=bool)


# Convenience: retain the old class name as an alias so existing imports
# (``from openclose_loader import OpenCloseDataset``) continue to work.
