import pytest
import os
from unittest.mock import patch
import pandas as pd
import numpy as np
from conninfpy.atlas import AtlasInfo
from conninfpy.interpret.evidence import build_decoding_evidence, validate_evidence
from conninfpy.interpret.llm_narrative import LLMNarrator, check_narrative_terms, load_dotenv_manually

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
def mock_decoded_rois():
    return pd.DataFrame([
        {"roi_id": 0, "roi_name": "ROI_A", "network": "Default", "rank": 1, "term": "memory", "score": 4.5},
        {"roi_id": 0, "roi_name": "ROI_A", "network": "Default", "rank": 2, "term": "attention", "score": 3.2},
        {"roi_id": 1, "roi_name": "ROI_B", "network": "Control", "rank": 1, "term": "control", "score": 5.1}
    ])

@pytest.fixture
def mock_edges():
    return pd.DataFrame({
        'roi_i': [0, 1],
        'roi_j': [1, 2],
        'network_pair': ['Control—Default', 'Control—Default']
    })

def test_build_decoding_evidence(mock_edges, sample_atlas, mock_decoded_rois):
    evidence = build_decoding_evidence(
        mock_edges,
        sample_atlas,
        mock_decoded_rois,
        contrast_name="ABIDE age",
        radius_mm=6.0,
        scoring="chi2",
        top_n=5
    )
    
    assert evidence["query"]["source"] == "conninfpy_edges"
    assert evidence["query"]["contrast"] == "ABIDE age"
    assert evidence["query"]["edge_count"] == 2
    assert set(evidence["query"]["roi_ids"]) == {0, 1, 2}
    
    # Check network pairs aggregation
    net_pairs = evidence["query"]["network_pairs"]
    assert len(net_pairs) == 1
    assert net_pairs[0]["source"] == "Control"
    assert net_pairs[0]["target"] == "Default"
    assert net_pairs[0]["n_edges"] == 2
    
    # Check decoder metadata
    assert evidence["decoder"]["backend"] == "NiMARE"
    assert evidence["decoder"]["radius_mm"] == 6.0
    assert evidence["decoder"]["top_n"] == 5
    
    # Check terms listing
    assert len(evidence["terms"]) == 3
    assert evidence["terms"][0]["term"] == "memory"
    assert evidence["terms"][0]["roi_id"] == 0

def test_validate_evidence(mock_edges, sample_atlas, mock_decoded_rois):
    evidence = build_decoding_evidence(
        mock_edges,
        sample_atlas,
        mock_decoded_rois,
        contrast_name="Test"
    )
    
    # Should validate without error
    validate_evidence(evidence)
    
    # Malformed: missing query key
    bad_evidence = evidence.copy()
    del bad_evidence["query"]
    with pytest.raises(ValueError, match="Missing required top-level evidence fields"):
        validate_evidence(bad_evidence)
        
    # Malformed: empty terms
    bad_evidence = evidence.copy()
    bad_evidence["terms"] = []
    with pytest.raises(ValueError, match="Field 'terms' list is empty"):
        validate_evidence(bad_evidence)

def test_check_narrative_terms(mock_edges, sample_atlas, mock_decoded_rois):
    evidence = build_decoding_evidence(
        mock_edges,
        sample_atlas,
        mock_decoded_rois,
        contrast_name="Test"
    )
    
    # Narrative that only uses terms in the evidence / structure
    safe_narrative = (
        "NiMARE decoding of Default and Control networks showed associations with memory. "
        "These memory and control findings reflect spatial literature frequency."
    )
    flagged = check_narrative_terms(safe_narrative, evidence)
    assert len(flagged) == 0
    
    # Narrative introducing outside cognitive term like 'pain' or 'vision'
    hallucinated_narrative = (
        "This contrast activates pain and vision networks, showing executive dysfunction."
    )
    flagged = check_narrative_terms(hallucinated_narrative, evidence)
    assert "pain" in flagged
    assert "executive" in flagged

def test_llm_narrator_mock(mock_edges, sample_atlas, mock_decoded_rois):
    evidence = build_decoding_evidence(
        mock_edges,
        sample_atlas,
        mock_decoded_rois,
        contrast_name="ABIDE"
    )
    
    narrator = LLMNarrator(provider="mock")
    res = narrator.generate(evidence)
    
    assert "### Decoding summary" in res
    assert "### Main associated terms" in res
    assert "'memory'" in res
    assert "'control'" in res
    assert "Default" in res
    assert "Control" in res

def test_llm_narrator_missing_api_keys():
    # Use patch to guarantee that environment variables are completely empty
    with patch.dict(os.environ, {}, clear=True):
        narrator_openai = LLMNarrator(provider="openai", api_key="")
        narrator_openai.api_key = None
        with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
            narrator_openai.generate({"query": {"source": "a", "atlas": "b"}, "decoder": {"backend": "a", "dataset": "b", "method": "c", "scoring": "d"}, "terms": [{"term": "a", "rank": 1, "score": 1.0}], "caveats": []})
            
        narrator_gemini = LLMNarrator(provider="gemini", api_key="")
        narrator_gemini.api_key = None
        with pytest.raises(ValueError, match="GEMINI_API_KEY or GOOGLE_API_KEY is not set"):
            narrator_gemini.generate({"query": {"source": "a", "atlas": "b"}, "decoder": {"backend": "a", "dataset": "b", "method": "c", "scoring": "d"}, "terms": [{"term": "a", "rank": 1, "score": 1.0}], "caveats": []})

        narrator_openrouter = LLMNarrator(provider="openrouter", api_key="")
        narrator_openrouter.api_key = None
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY is not set"):
            narrator_openrouter.generate({"query": {"source": "a", "atlas": "b"}, "decoder": {"backend": "a", "dataset": "b", "method": "c", "scoring": "d"}, "terms": [{"term": "a", "rank": 1, "score": 1.0}], "caveats": []})

def test_llm_narrator_openrouter_setup():
    with patch.dict(os.environ, {}, clear=True):
        narrator = LLMNarrator(provider="openrouter", api_key="sk-or-v1-testkey", model="meta-llama/test")
        assert narrator.provider == "openrouter"
        assert narrator.api_key == "sk-or-v1-testkey"
        assert narrator.model == "meta-llama/test"

def test_load_dotenv_manually():
    with patch.dict(os.environ, {}, clear=True):
        load_dotenv_manually()
        assert "OPENROUTER_API_KEY" in os.environ
        assert os.environ["OPENROUTER_API_KEY"].startswith("sk-or-")
