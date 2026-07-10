import os
import tempfile
import unittest
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import matplotlib.pyplot as plt

from conninfpy.loaders.manifest import load_manifest, resolve_path, ManifestLoader, validate_manifest_dataset
from conninfpy.loaders.base import LoadedDataset
from conninfpy.loaders.builtins import NumpyLoader, NiftiDirectoryLoader

class TestManifestLoader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.manifest_dir = Path(self.temp_dir.name)
        
    def tearDown(self):
        self.temp_dir.cleanup()
        
    def test_resolve_path_builtin(self):
        # BUILTIN:bna246.csv should resolve to package resources
        res = resolve_path("BUILTIN:bna246.csv", self.manifest_dir)
        self.assertTrue(res.endswith("bna246.csv"))
        self.assertTrue(os.path.exists(res))
        
    def test_resolve_path_relative(self):
        # Relative path should resolve relative to manifest_dir
        res = resolve_path("subfolder/data.npy", self.manifest_dir)
        expected = str((self.manifest_dir / "subfolder/data.npy").resolve())
        self.assertEqual(res, expected)
        
    def test_validate_manifest_dataset_rois(self):
        # Create a dummy LoadedDataset
        data = np.zeros((5, 10, 10))
        pheno = pd.DataFrame({"subject_id": [f"sub_{i}" for i in range(5)]})
        dataset = LoadedDataset(data, pheno, "correlation")
        
        # Manifest expecting 10 ROIs should pass
        manifest_data = {
            "loader": "NumpyLoader",
            "paths": {},
            "checks": {
                "expected_rois": 10
            }
        }
        from conninfpy.loaders.manifest import DatasetManifest
        manifest = DatasetManifest(manifest_data, "data.yaml")
        
        # Should not raise error
        validate_manifest_dataset(dataset, manifest)
        
        # Manifest expecting 12 ROIs should fail
        manifest.checks["expected_rois"] = 12
        with self.assertRaises(ValueError):
            validate_manifest_dataset(dataset, manifest)

    def test_validate_manifest_dataset_subjects(self):
        data = np.zeros((5, 10, 10))
        pheno = pd.DataFrame({"subject_id": [f"sub_{i}" for i in range(5)]})
        dataset = LoadedDataset(data, pheno, "correlation")
        
        manifest_data = {
            "loader": "NumpyLoader",
            "paths": {},
            "checks": {
                "min_subjects": 10
            }
        }
        from conninfpy.loaders.manifest import DatasetManifest
        manifest = DatasetManifest(manifest_data, "data.yaml")
        
        # 5 < 10, should raise ValueError
        with self.assertRaises(ValueError):
            validate_manifest_dataset(dataset, manifest)
            
    def test_manifest_loader_instantiation(self):
        # Create a temporary manifest for NumpyLoader
        npy_path = self.manifest_dir / "test_data.npy"
        np.save(npy_path, np.zeros((5, 10, 10)))
        
        yaml_content = {
            "schema_version": 1,
            "name": "Test Numpy Dataset",
            "loader": "NumpyLoader",
            "paths": {
                "data_path": "test_data.npy"
            },
            "params": {
                "data_kind": "correlation"
            },
            "checks": {
                "expected_rois": 10,
                "min_subjects": 3
            }
        }
        
        manifest_path = self.manifest_dir / "data.yaml"
        with open(manifest_path, "w") as f:
            yaml.dump(yaml_content, f)
            
        # Instantiate ManifestLoader
        m_loader = ManifestLoader(str(manifest_path))
        self.assertIsInstance(m_loader.target_loader, NumpyLoader)
        self.assertEqual(os.path.realpath(m_loader.target_loader.data_path), os.path.realpath(str(npy_path)))
        
        # Load dataset
        dataset = m_loader.load()
        self.assertEqual(dataset.data.shape, (5, 10, 10))
        self.assertEqual(len(dataset.subject_ids), 5)

    def test_decoding_evidence_summarization_and_scoring(self) -> None:
        # Create a mock decoded dataframe
        decoded_rois = pd.DataFrame([
            {"roi_id": 0, "roi_name": "ROI_0", "network": "Visual", "rank": 1, "term": "social", "score": 4.5},
            {"roi_id": 0, "roi_name": "ROI_0", "network": "Visual", "rank": 2, "term": "fmri", "score": 3.0}, # method word
            {"roi_id": 1, "roi_name": "ROI_1", "network": "Default", "rank": 1, "term": "social", "score": 5.0},
            {"roi_id": 1, "roi_name": "ROI_1", "network": "Default", "rank": 2, "term": "memory", "score": 4.0},
        ])
        
        # Create mock edges
        edges = pd.DataFrame([
            {"roi_i": 0, "roi_j": 1}
        ])
        
        # Create dummy AtlasInfo
        from conninfpy.atlas import AtlasInfo
        atlas = AtlasInfo(
            labels=["ROI_0", "ROI_1"],
            networks=["Visual", "Default"],
            coords=np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
        )
        
        from conninfpy.interpret.evidence import summarize_decoded_terms, score_decoding_evidence
        summary = summarize_decoded_terms(decoded_rois, edges, atlas)
        
        # 'fmri' should be filtered out by default stop words list
        terms = [t["term"] for t in summary["aggregated_terms"]]
        self.assertNotIn("fmri", terms)
        self.assertIn("social", terms)
        self.assertIn("memory", terms)
        
        # 'social' appears in 2 ROIs, so weighted count is 2.0 (since burden is 1 for each ROI)
        social_stats = [t for t in summary["aggregated_terms"] if t["term"] == "social"][0]
        self.assertEqual(social_stats["roi_count"], 2)
        self.assertEqual(social_stats["weighted_count"], 2.0)
        self.assertEqual(social_stats["best_rank"], 1)
        
        score = score_decoding_evidence(summary)
        self.assertEqual(score["evidence_quality"], "weak") # recurrence count is 1 ('social'), len(terms) is 2, max_weighted is 2.0

    def test_manifest_loader_roi_filtering(self) -> None:
        # Create a temporary manifest for NumpyLoader with keep_rois
        npy_path = self.manifest_dir / "test_data.npy"
        np.save(npy_path, np.zeros((5, 10, 10))) # 10 ROIs initially
        
        yaml_content = {
            "schema_version": 1,
            "name": "Test Numpy Dataset ROI Filtered",
            "loader": "NumpyLoader",
            "paths": {
                "data_path": "test_data.npy"
            },
            "params": {
                "data_kind": "correlation",
                "keep_rois": [1, 3, 5]  # keep 1-indexed ROIs 1, 3, 5 -> 3 ROIs total
            },
            "checks": {
                "expected_rois": 3
            }
        }
        
        manifest_path = self.manifest_dir / "data.yaml"
        with open(manifest_path, "w") as f:
            yaml.dump(yaml_content, f)
            
        m_loader = ManifestLoader(str(manifest_path))
        
        # Preview should show 3 ROIs
        preview = m_loader.preview()
        self.assertEqual(preview.n_rois, 3)
        
        # Load should filter data to shape (5, 3, 3)
        dataset = m_loader.load()
        self.assertEqual(dataset.data.shape, (5, 3, 3))

    def test_align_atlas_coordinates(self) -> None:
        from conninfpy.atlas import AtlasInfo
        from apps.utils.helpers import align_atlas_coordinates
        
        ref_atlas = AtlasInfo(
            labels=["ROI_1", "ROI_2", "ROI_3"],
            networks=["Vis", "Mot", "Aud"],
            coords=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        )
        
        target_atlas = AtlasInfo(
            labels=["ROI_3", "ROI_1"],
            networks=["Aud", "Vis"],
            coords=None
        )
        
        aligned = align_atlas_coordinates(target_atlas, ref_atlas)
        self.assertIsNotNone(aligned.coords)
        np.testing.assert_array_equal(
            aligned.coords,
            np.array([[7.0, 8.0, 9.0], [1.0, 2.0, 3.0]])
        )

    def test_plot_connectome_graph(self) -> None:
        from conninfpy.atlas import AtlasInfo
        from conninfpy.plot import plot_connectome_graph
        import pandas as pd
        import matplotlib
        
        # Avoid showing window during unit tests
        matplotlib.use("Agg")
        
        atlas = AtlasInfo(
            labels=["ROI_1", "ROI_2", "ROI_3"],
            networks=["Vis", "Mot", "Aud"],
            coords=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        )
        
        edges_df = pd.DataFrame([
            {"roi_i": 0, "roi_j": 1, "t_signed": 3.5, "p_positive": 0.01, "p_negative": 0.99},
            {"roi_i": 1, "roi_j": 2, "t_signed": -2.1, "p_positive": 0.95, "p_negative": 0.02}
        ])
        
        # Test positive connectome plot
        fig_pos = plot_connectome_graph(edges_df, atlas, "positive", alpha=0.05)
        self.assertIsNotNone(fig_pos)
        plt.close(fig_pos)
        
        # Test negative connectome plot
        fig_neg = plot_connectome_graph(edges_df, atlas, "negative", alpha=0.05)
        self.assertIsNotNone(fig_neg)
        plt.close(fig_neg)

if __name__ == "__main__":
    unittest.main()
