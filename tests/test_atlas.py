"""Tests for :mod:`conninfpy.atlas`."""
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from conninfpy import AtlasInfo


class TestAtlasInfoBasic(unittest.TestCase):
    """Basic construction, validation, and indexing."""

    def test_len_and_attrs(self):
        atlas = AtlasInfo(
            labels=["A", "B", "C"],
            networks=["Vis", "Vis", "DMN"],
            coords=np.array([[0.0, 0, 0], [1, 1, 1], [2, 2, 2]]),
            hemisphere=["L", "R", "L"],
            source="unit-test",
        )
        self.assertEqual(len(atlas), 3)
        self.assertEqual(atlas.labels[1], "B")
        self.assertEqual(atlas.networks[2], "DMN")
        self.assertEqual(atlas.hemisphere[0], "L")
        self.assertEqual(atlas.coords.shape, (3, 3))

    def test_network_index(self):
        atlas = AtlasInfo(labels=list("abcd"), networks=["X", "Y", "X", "Z"])
        idx = atlas.network_index()
        # First-appearance encoding: X=0, Y=1, Z=2
        np.testing.assert_array_equal(idx, [0, 1, 0, 2])

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            AtlasInfo(labels=["A", "B"], networks=["X"])

    def test_bad_coord_shape_raises(self):
        with self.assertRaises(ValueError):
            AtlasInfo(
                labels=["A", "B"],
                networks=["X", "Y"],
                coords=np.zeros((2, 2)),  # wrong second dim
            )

    def test_hemisphere_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            AtlasInfo(
                labels=["A", "B"], networks=["X", "Y"], hemisphere=["L"]
            )


class TestFromCsv(unittest.TestCase):
    """Round-trip a synthetic CSV through :meth:`AtlasInfo.from_csv`."""

    def _write_csv(self, tmp: Path, header, rows):
        path = tmp / "atlas.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        return path

    def test_round_trip_with_coords_and_hemisphere(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            path = self._write_csv(
                tmp,
                ["name", "network", "hemisphere", "x", "y", "z"],
                [
                    ["roi1", "DMN", "L", "1.0", "2.0", "3.0"],
                    ["roi2", "DMN", "R", "4.0", "5.0", "6.0"],
                    ["roi3", "Vis", "L", "-1.0", "0.0", "1.0"],
                ],
            )
            atlas = AtlasInfo.from_csv(path)
            self.assertEqual(atlas.labels, ["roi1", "roi2", "roi3"])
            self.assertEqual(atlas.networks, ["DMN", "DMN", "Vis"])
            self.assertEqual(atlas.hemisphere, ["L", "R", "L"])
            np.testing.assert_allclose(
                atlas.coords,
                [[1, 2, 3], [4, 5, 6], [-1, 0, 1]],
            )

    def test_empty_coord_columns_become_none(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            path = self._write_csv(
                tmp,
                ["name", "network", "x", "y", "z"],
                [["roi1", "DMN", "", "", ""], ["roi2", "Vis", "", "", ""]],
            )
            atlas = AtlasInfo.from_csv(path)
            self.assertIsNone(atlas.coords)

    def test_partial_coord_blanks_become_nan(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            path = self._write_csv(
                tmp,
                ["name", "network", "x", "y", "z"],
                [["roi1", "DMN", "1", "2", "3"], ["roi2", "Vis", "", "", ""]],
            )
            atlas = AtlasInfo.from_csv(path)
            self.assertEqual(atlas.coords.shape, (2, 3))
            np.testing.assert_allclose(atlas.coords[0], [1, 2, 3])
            self.assertTrue(np.all(np.isnan(atlas.coords[1])))

    def test_missing_required_column_raises(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            path = self._write_csv(tmp, ["roi", "net"], [["a", "x"]])
            with self.assertRaises(ValueError):
                AtlasInfo.from_csv(path)


class TestBundledAtlases(unittest.TestCase):
    """Bundled atlas resources load and have the expected shape."""

    def test_schaefer_100_yeo7(self):
        atlas = AtlasInfo.schaefer_100_yeo7()
        self.assertEqual(len(atlas), 100)
        # All seven Yeo networks should appear among the 100 ROIs.
        unique_networks = set(atlas.networks)
        self.assertEqual(
            unique_networks,
            {
                "Visual", "SomMot", "DorsAttn",
                "SalVentAttn", "Limbic", "Cont", "Default",
            },
        )
        # Hemisphere column populated.
        self.assertIsNotNone(atlas.hemisphere)
        self.assertTrue(set(atlas.hemisphere) <= {"LH", "RH"})
        # Coordinates populated and finite.
        self.assertIsNotNone(atlas.coords)
        self.assertEqual(atlas.coords.shape, (100, 3))
        self.assertTrue(np.isfinite(atlas.coords).all())

    def test_schaefer_100_network_index_round_trip(self):
        atlas = AtlasInfo.schaefer_100_yeo7()
        idx = atlas.network_index()
        self.assertEqual(idx.shape, (100,))
        # Seven distinct integer codes for seven Yeo networks.
        self.assertEqual(len(np.unique(idx)), 7)

    def test_schaefer_200_yeo7(self):
        atlas = AtlasInfo.schaefer_200_yeo7()
        self.assertEqual(len(atlas), 200)
        self.assertEqual(len(set(atlas.networks)), 7)
        self.assertTrue(set(atlas.hemisphere) <= {"LH", "RH"})
        self.assertEqual(atlas.coords.shape, (200, 3))
        self.assertTrue(np.isfinite(atlas.coords).all())

    def test_schaefer_400_yeo7(self):
        atlas = AtlasInfo.schaefer_400_yeo7()
        self.assertEqual(len(atlas), 400)
        self.assertEqual(len(set(atlas.networks)), 7)
        self.assertTrue(set(atlas.hemisphere) <= {"LH", "RH"})
        self.assertEqual(atlas.coords.shape, (400, 3))
        self.assertTrue(np.isfinite(atlas.coords).all())

    def test_bna_246(self):
        atlas = AtlasInfo.bna_246()
        self.assertEqual(len(atlas), 246)
        # BNA seven-lobe grouping.
        unique_networks = set(atlas.networks)
        expected_lobes = {
            "Frontal", "Temporal", "Parietal", "Insular",
            "Limbic", "Occipital", "Subcortical",
        }
        self.assertEqual(unique_networks, expected_lobes)
        # Hemisphere column populated.
        self.assertIsNotNone(atlas.hemisphere)
        self.assertTrue(set(atlas.hemisphere) <= {"L", "R", "M"})
        # Spot-check known anchor IDs.
        self.assertEqual(atlas.labels[0], "A8m_L")
        self.assertEqual(atlas.labels[245], "lPFtha_R")
        self.assertEqual(atlas.networks[0], "Frontal")
        self.assertEqual(atlas.networks[245], "Subcortical")

    def test_bna_246_has_coords(self):
        # Coordinates are MNI152-1mm voxel-mass centroids from
        # BN_Atlas_246_1mm.nii.gz. All 246 ROIs should be finite and
        # roughly bracketed by the standard MNI bounding box.
        atlas = AtlasInfo.bna_246()
        self.assertIsNotNone(atlas.coords)
        self.assertEqual(atlas.coords.shape, (246, 3))
        self.assertTrue(np.isfinite(atlas.coords).all())
        # MNI152 brain extent is roughly x∈[-90,90], y∈[-126,90], z∈[-72,108].
        self.assertGreater(atlas.coords[:, 0].min(), -90)
        self.assertLess(atlas.coords[:, 0].max(), 90)
        # Left-hemisphere ROIs (suffix '_L') should have x ≤ 0
        # (allowing a small midline tolerance for medial structures).
        for i, (name, x) in enumerate(
            zip(atlas.labels, atlas.coords[:, 0])
        ):
            if name.endswith("_L"):
                self.assertLess(x, 5, f"L hemi ROI {name} at x={x:.2f}")
            elif name.endswith("_R"):
                self.assertGreater(x, -5, f"R hemi ROI {name} at x={x:.2f}")


class TestNetworkIndexForBlockMass(unittest.TestCase):
    """``network_index`` plays nicely with the ``block_mass`` API."""

    def test_block_mass_round_trip(self):
        # Synthetic p-value map with two networks and a fake signal block.
        from conninfpy import block_mass

        atlas = AtlasInfo(
            labels=[f"r{i}" for i in range(6)],
            networks=["A"] * 3 + ["B"] * 3,
        )
        idx = atlas.network_index()
        p = np.full((6, 6), 0.5, dtype=np.float64)
        # Significant within-A block (upper-tri only matters for block_mass).
        p[:3, :3] = 0.001
        np.fill_diagonal(p, 1.0)
        counts = block_mass(p, idx, alpha=0.05)
        # block_mass returns an upper-triangular (K, K) matrix of
        # within/between-network significant-edge counts. With A=0, B=1
        # the only signal is in the (A, A) cell.
        self.assertEqual(counts.shape, (2, 2))
        self.assertGreater(counts[0, 0], 0)
        self.assertEqual(counts[0, 1], 0)
        self.assertEqual(counts[1, 1], 0)


if __name__ == "__main__":
    unittest.main()
