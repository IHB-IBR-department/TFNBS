import sys
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

# Mock nimare before importing decode module
mock_nimare = MagicMock()
mock_decode_discrete = MagicMock()
mock_extract = MagicMock()
mock_io = MagicMock()

# Setup NeurosynthDecoder Mock
class MockNeurosynthDecoder:
    def __init__(self, frequency_threshold=0.001, prior=0.5):
        pass
    def fit(self, dataset):
        pass
    def transform(self, active_ids):
        # Return a simple mock dataframe
        data = {
            'z_assoc': [5.0, 4.0, 3.0, 2.0, 1.0, 0.5],
            'p_assoc': [0.0001, 0.0002, 0.0003, 0.004, 0.05, 0.1]
        }
        terms = ['memory', 'attention', 'executive', 'motor', 'visual', 'auditory']
        return pd.DataFrame(data, index=terms)

mock_decode_discrete.NeurosynthDecoder = MockNeurosynthDecoder

# Apply mock patches
sys.modules['nimare'] = mock_nimare
sys.modules['nimare.decode'] = mock_nimare.decode
sys.modules['nimare.decode.discrete'] = mock_decode_discrete
sys.modules['nimare.extract'] = mock_extract
sys.modules['nimare.io'] = mock_io

from conninfpy.atlas import AtlasInfo
from conninfpy.decode import decode_rois, annotate_edge_table
from conninfpy._result import InferenceResult

@pytest.fixture
def sample_atlas():
    labels = ['ROI_A', 'ROI_B', 'ROI_C']
    networks = ['Default', 'Control', 'Default']
    coords = np.array([
        [10.0, 10.0, 10.0],
        [20.0, 20.0, 20.0],
        [100.0, 100.0, 100.0]
    ])
    return AtlasInfo(labels=labels, networks=networks, coords=coords)

@pytest.fixture
def mock_dataset():
    coords_df = pd.DataFrame({
        'id': ['study1', 'study1', 'study2', 'study3'],
        'x': [10.0, 15.0, 20.0, 100.0],
        'y': [10.0, 15.0, 20.0, 100.0],
        'z': [10.0, 15.0, 20.0, 100.0]
    })
    dataset = MagicMock()
    dataset.coordinates = coords_df
    return dataset

def test_decode_rois_shape_and_values(sample_atlas, mock_dataset):
    # Decode 2 ROIs (ROI_A and ROI_B) with top_n = 3
    res = decode_rois(
        sample_atlas,
        roi_ids=[0, 1],
        top_n=3,
        radius_mm=6.0,
        dataset=mock_dataset
    )
    
    # Rows should be: 2 ROIs * 3 top_n = 6 rows
    assert len(res) == 6
    assert list(res.columns) == ['roi_id', 'roi_name', 'network', 'rank', 'term', 'score']
    
    # Verify values for ROI_A (id=0)
    roi_a_df = res[res['roi_id'] == 0]
    assert len(roi_a_df) == 3
    assert list(roi_a_df['rank']) == [1, 2, 3]
    assert list(roi_a_df['term']) == ['memory', 'attention', 'executive']
    assert list(roi_a_df['score']) == [5.0, 4.0, 3.0]

def test_decode_rois_empty_studies(sample_atlas, mock_dataset):
    # ROI_C is at (100, 100, 100). If we set radius_mm=1.0, only study3 is matched.
    # If we set radius_mm=0.1, ROI_A (at 10,10,10) has no studies within 0.1 mm (study1 is at 10,10,10, but distance check works).
    # Wait, study1 is exactly at 10,10,10 so it matches.
    # Let's set coordinates of ROI_B to something far away, e.g. 500,500,500.
    sample_atlas.coords[1] = [500.0, 500.0, 500.0]
    res = decode_rois(
        sample_atlas,
        roi_ids=[1],
        top_n=3,
        radius_mm=6.0,
        dataset=mock_dataset
    )
    assert len(res) == 1
    assert res.iloc[0]['term'] == 'inconclusive'
    assert res.iloc[0]['score'] == 0.0

def test_decode_rois_missing_coords(sample_atlas):
    # Remove coordinates
    sample_atlas.coords = None
    with pytest.raises(ValueError, match="Atlas coordinates are None"):
        decode_rois(sample_atlas, roi_ids=[0])

def test_annotate_edge_table(sample_atlas, mock_dataset):
    edges = pd.DataFrame({
        'roi_i': [0, 1],
        'roi_j': [1, 2],
        'stat': [2.5, 3.1]
    })
    
    annotated = annotate_edge_table(
        edges,
        sample_atlas,
        top_n=2,
        dataset=mock_dataset,
        radius_mm=15.0  # Large enough to match studies for all
    )
    
    assert len(annotated) == 2
    for col in ['roi_i_terms', 'roi_j_terms', 'roi_i_top_term', 'roi_j_top_term']:
        assert col in annotated.columns
        
    # Check that roi_i_terms contains 'memory; attention' (joined top 2 terms)
    assert annotated.iloc[0]['roi_i_terms'] == 'memory; attention'
    assert annotated.iloc[0]['roi_i_top_term'] == 'memory'

def test_annotate_edge_table_empty_short_circuit(sample_atlas):
    edges = pd.DataFrame(columns=['roi_i', 'roi_j'])
    
    annotated = annotate_edge_table(
        edges,
        sample_atlas,
        top_n=2
    )
    
    assert annotated.empty
    for col in ['roi_i_terms', 'roi_j_terms', 'roi_i_top_term', 'roi_j_top_term']:
        assert col in annotated.columns

def test_inference_result_decoded_edges(sample_atlas, mock_dataset):
    # Construct an InferenceResult and test decoded_edges method
    # InferenceResult needs positive, negative, and other attributes
    # We can mock InferenceResult's significant_edges method
    res = InferenceResult(
        np.zeros((3, 3)),
        np.zeros((3, 3)),
        method='tfnbs',
        n_permutations=10
    )
    
    mock_edges = pd.DataFrame({
        'roi_i': [0],
        'roi_j': [1]
    })
    
    with patch.object(res, 'significant_edges', return_value=mock_edges) as mock_sig:
        annotated = res.decoded_edges(
            sample_atlas,
            top_n=2,
            dataset=mock_dataset,
            radius_mm=15.0
        )
        mock_sig.assert_called_once_with(atlas=sample_atlas)
        assert len(annotated) == 1
        assert annotated.iloc[0]['roi_i_terms'] == 'memory; attention'
