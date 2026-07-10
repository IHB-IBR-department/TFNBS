import unittest
import tempfile
import shutil
import os
import numpy as np
import pandas as pd

from conninfpy.loaders import (
    LoadedDataset,
    validate_loaded_dataset,
    NumpyLoader,
    CSVDirectoryLoader,
    BaseDataLoader
)
from conninfpy.atlas import AtlasInfo

class TestLoaders(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    def test_validation_coherence(self):
        # 1. Successful timeseries validation
        data_ts = np.random.randn(5, 100, 10)  # (subjects, timepoints, rois)
        pheno_ts = pd.DataFrame({"subject_id": [f"sub_{i}" for i in range(5)]})
        ds_ts = LoadedDataset(data=data_ts, pheno=pheno_ts, data_kind="timeseries")
        report = validate_loaded_dataset(ds_ts)
        self.assertTrue(report.ok)
        self.assertEqual(report.summary["n_timepoints"], 100)
        self.assertEqual(report.summary["n_rois"], 10)
        
        # 2. Successful correlation validation
        data_corr = np.array([np.eye(10) for _ in range(5)])
        pheno_corr = pd.DataFrame({"subject_id": [f"sub_{i}" for i in range(5)]})
        ds_corr = LoadedDataset(data=data_corr, pheno=pheno_corr, data_kind="correlation")
        report = validate_loaded_dataset(ds_corr)
        self.assertTrue(report.ok)
        self.assertEqual(report.summary["n_rois"], 10)
        
        # 3. Failed validation: non-square correlation matrices
        data_bad = np.random.randn(5, 10, 8)
        ds_bad = LoadedDataset(data=data_bad, pheno=pheno_corr, data_kind="correlation")
        report = validate_loaded_dataset(ds_bad)
        self.assertFalse(report.ok)
        self.assertTrue(any("not square" in err for err in report.errors))
        
        # 4. Failed validation: asymmetric correlation matrices
        data_asym = np.array([np.random.randn(10, 10) for _ in range(5)])
        ds_asym = LoadedDataset(data=data_asym, pheno=pheno_corr, data_kind="correlation")
        report = validate_loaded_dataset(ds_asym)
        self.assertFalse(report.ok)
        self.assertTrue(any("not symmetric" in err for err in report.errors))

        # 5. Failed validation: mismatched subject counts
        pheno_mismatch = pd.DataFrame({"subject_id": [f"sub_{i}" for i in range(4)]})
        ds_mismatch = LoadedDataset(data=data_corr, pheno=pheno_mismatch, data_kind="correlation")
        report = validate_loaded_dataset(ds_mismatch)
        self.assertFalse(report.ok)
        self.assertTrue(any("Subject count mismatch" in err for err in report.errors))

    def test_numpy_loader(self):
        arr = np.random.randn(10, 50, 50)
        # Symmetrize matrices for correlation
        for i in range(10):
            arr[i] = (arr[i] + arr[i].T) / 2
            
        npy_path = os.path.join(self.temp_dir, "test_data.npy")
        np.save(npy_path, arr)
        
        loader = NumpyLoader(npy_path, data_kind="correlation")
        preview = loader.preview()
        self.assertEqual(preview.n_subjects, 10)
        self.assertEqual(preview.n_rois, 50)
        
        ds = loader.load()
        self.assertEqual(ds.data.shape, (10, 50, 50))
        self.assertEqual(len(ds.pheno), 10)
        self.assertEqual(ds.data_kind, "correlation")

    def test_csv_directory_loader(self):
        # Create a mock directory with subject timeseries CSVs
        for i in range(3):
            df = pd.DataFrame(np.random.randn(100, 5))
            df.to_csv(os.path.join(self.temp_dir, f"sub_{i}_ts_Schaefer100.csv"), index=False, header=False)
            
        loader = CSVDirectoryLoader(self.temp_dir, header=None)
        preview = loader.preview()
        self.assertEqual(preview.n_subjects, 3)
        self.assertEqual(preview.n_timepoints, 100)
        self.assertEqual(preview.n_rois, 5)
        
        ds = loader.load()
        self.assertEqual(ds.data.shape, (3, 100, 5))
        self.assertEqual(ds.subject_ids, ["sub_0", "sub_1", "sub_2"])
