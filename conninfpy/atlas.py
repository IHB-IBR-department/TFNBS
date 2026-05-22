"""Atlas / parcellation metadata for connectivity inference.

:class:`AtlasInfo` carries per-ROI labels, resting-state-network (or lobe)
assignments, MNI coordinates, hemisphere, and a free-text provenance
string. It is consumed by the export and plotting helpers
(``InferenceResult.significant_edges``, ``plot_effect_matrix``,
``summary_figure``) so significant edges can be reported with ROI names
and network context rather than bare indices.

Two bundled atlases ship with v2.1:

* :meth:`AtlasInfo.schaefer_100_yeo7`,
  :meth:`AtlasInfo.schaefer_200_yeo7`,
  :meth:`AtlasInfo.schaefer_400_yeo7` — Schaefer 2018 cortical
  parcellations at three resolutions with Yeo 7-network labels and
  MNI152 centroid coordinates.
* :meth:`AtlasInfo.bna_246` — Brainnetome Atlas (Fan et al. 2016) 246
  cortical + subcortical regions, with the BNA seven-lobe grouping in
  the ``network`` column and MNI152-1mm centroids derived from
  ``BN_Atlas_246_1mm.nii.gz``.

Custom CSVs can be loaded via :meth:`AtlasInfo.from_csv`.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt


_NetworkIndex = npt.NDArray[np.int_]


@dataclass
class AtlasInfo:
    """Per-ROI metadata for a brain parcellation.

    Attributes
    ----------
    labels : list[str]
        ROI names, length ``N``. Order is significant — index ``i`` in
        an ``(n_subjects, N, N)`` connectivity tensor corresponds to
        ``labels[i]``.
    networks : list[str]
        Per-ROI resting-state-network (or lobe) label.
    coords : ndarray of shape (N, 3) or None
        MNI millimeter centroids ``(x, y, z)``. ``None`` when the
        bundled atlas did not ship coordinates.
    hemisphere : list[str] or None
        Per-ROI hemisphere code (``'L'`` / ``'R'`` / ``'M'`` for medial
        or subcortical structures spanning the midline). Optional.
    source : str or None
        Provenance string. For bundled atlases this points to the
        original publication and the resource the CSV was generated
        from.
    """

    labels: List[str]
    networks: List[str]
    coords: Optional[npt.NDArray[np.float64]] = None
    hemisphere: Optional[List[str]] = None
    source: Optional[str] = None

    def __post_init__(self) -> None:
        n = len(self.labels)
        if len(self.networks) != n:
            raise ValueError(
                f"labels and networks must be the same length; "
                f"got {n} labels and {len(self.networks)} networks."
            )
        if self.coords is not None:
            self.coords = np.asarray(self.coords, dtype=np.float64)
            if self.coords.shape != (n, 3):
                raise ValueError(
                    f"coords must have shape ({n}, 3); got {self.coords.shape}."
                )
        if self.hemisphere is not None and len(self.hemisphere) != n:
            raise ValueError(
                f"hemisphere must have length {n}; "
                f"got {len(self.hemisphere)}."
            )

    def __len__(self) -> int:
        return len(self.labels)

    def network_index(self) -> _NetworkIndex:
        """Integer-coded networks for ``block_mass`` / cnbs / ni_tfnbs.

        Two ROIs share an integer iff they share a network label.
        Codes are assigned in order of first appearance.
        """
        seen: dict = {}
        out = np.empty(len(self.networks), dtype=np.int_)
        for i, n in enumerate(self.networks):
            if n not in seen:
                seen[n] = len(seen)
            out[i] = seen[n]
        return out

    @classmethod
    def from_csv(
        cls,
        path: Union[str, Path],
        *,
        label_col: str = "name",
        network_col: str = "network",
        coord_cols: Optional[Sequence[str]] = ("x", "y", "z"),
        hemisphere_col: Optional[str] = "hemisphere",
        source: Optional[str] = None,
    ) -> "AtlasInfo":
        """Load an atlas from a CSV file.

        Required columns: ``label_col`` (default ``'name'``) and
        ``network_col`` (default ``'network'``). Coordinate columns
        (default ``('x', 'y', 'z')``) and ``hemisphere_col`` (default
        ``'hemisphere'``) are loaded when present and non-empty;
        otherwise ``coords`` / ``hemisphere`` are set to ``None``.
        Rows are taken in file order.
        """
        path = Path(path)
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            raise ValueError(f"{path}: no rows found.")
        if label_col not in rows[0]:
            raise ValueError(
                f"{path}: missing required column {label_col!r}."
            )
        if network_col not in rows[0]:
            raise ValueError(
                f"{path}: missing required column {network_col!r}."
            )

        labels = [r[label_col] for r in rows]
        networks = [r[network_col] for r in rows]

        coords: Optional[np.ndarray] = None
        if coord_cols is not None and all(c in rows[0] for c in coord_cols):
            raw = np.array(
                [[r[c].strip() for c in coord_cols] for r in rows]
            )
            if (raw == "").all():
                coords = None
            else:
                # NaN-fill any blank cells; numpy will fail otherwise.
                clean = np.where(raw == "", "nan", raw)
                coords = clean.astype(np.float64)

        hemisphere: Optional[List[str]] = None
        if hemisphere_col is not None and hemisphere_col in rows[0]:
            hemisphere = [r[hemisphere_col] for r in rows]
            if all(h == "" for h in hemisphere):
                hemisphere = None

        return cls(
            labels=labels,
            networks=networks,
            coords=coords,
            hemisphere=hemisphere,
            source=source or str(path),
        )

    # ------------------------------------------------------------------
    # Bundled atlases
    # ------------------------------------------------------------------
    @classmethod
    def schaefer_100_yeo7(cls) -> "AtlasInfo":
        """Schaefer-2018 100-parcel atlas with Yeo-7 networks.

        Source: ThomasYeoLab/CBIG Schaefer 2018 release. Labels and
        Yeo-7 network assignments follow the upstream naming
        (``7Networks_LH_Vis_1`` etc.); ``coords`` are MNI152 2mm
        centroids derived from the published parcellation image.
        """
        return _load_bundled("schaefer100_yeo7.csv", source=_SCHAEFER_SRC)

    @classmethod
    def schaefer_200_yeo7(cls) -> "AtlasInfo":
        """Schaefer-2018 200-parcel atlas with Yeo-7 networks.

        Same source and conventions as :meth:`schaefer_100_yeo7`, at
        the 200-parcel resolution.
        """
        return _load_bundled("schaefer200_yeo7.csv", source=_SCHAEFER_SRC)

    @classmethod
    def schaefer_400_yeo7(cls) -> "AtlasInfo":
        """Schaefer-2018 400-parcel atlas with Yeo-7 networks.

        Same source and conventions as :meth:`schaefer_100_yeo7`, at
        the 400-parcel resolution.
        """
        return _load_bundled("schaefer400_yeo7.csv", source=_SCHAEFER_SRC)

    @classmethod
    def bna_246(cls) -> "AtlasInfo":
        """Brainnetome Atlas (Fan et al. 2016) 246-region parcellation.

        Names follow ``BN_Atlas_246_LUT.txt``. The ``network`` column
        carries the BNA seven-lobe grouping (Frontal / Temporal /
        Parietal / Insular / Limbic / Occipital / Subcortical). MNI
        coordinates are voxel-mass centroids computed from the
        ``BN_Atlas_246_1mm.nii.gz`` parcellation (atlas.brainnetome.org).
        """
        return _load_bundled(
            "bna246.csv",
            source=(
                "Fan et al. 2016 Brainnetome Atlas; networks = "
                "BNA seven-lobe grouping by ROI-id range; centroids "
                "from BN_Atlas_246_1mm.nii.gz."
            ),
        )


_SCHAEFER_SRC = (
    "Schaefer et al. 2018 (ThomasYeoLab/CBIG release); "
    "centroids from FSLMNI152 2mm parcellation."
)


def _load_bundled(filename: str, *, source: str) -> AtlasInfo:
    """Load a packaged CSV via ``importlib.resources``."""
    data_pkg = resources.files("conninfpy.data")
    csv_path = data_pkg / filename
    with resources.as_file(csv_path) as path:
        return AtlasInfo.from_csv(path, source=source)


__all__ = ["AtlasInfo"]
