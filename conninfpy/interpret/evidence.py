import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from ..atlas import AtlasInfo

DEFAULT_CAVEATS = [
    "Decoding is an association with the neuroimaging literature, not a mechanistic claim.",
    "Term scores summarize spatial co-occurrence; they do not prove task engagement.",
    "Network-level summaries are safer than single-edge interpretations."
]

def build_decoding_evidence(
    edges: pd.DataFrame,
    atlas: AtlasInfo,
    decoded_rois: pd.DataFrame,
    *,
    contrast_name: str = "Unknown",
    radius_mm: float = 6.0,
    scoring: str = "chi2",
    top_n: int = 10,
    source: str = "conninfpy_edges",
    backend: str = "NiMARE",
    dataset_name: str = "Neurosynth",
    decoder_method: str = "NeurosynthDecoder",
    caveats: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Compile raw results and metadata into a structured evidence dictionary.
    
    Parameters
    ----------
    edges : pandas.DataFrame
        Significant edges DataFrame.
    atlas : AtlasInfo
        Atlas metadata.
    decoded_rois : pandas.DataFrame
        DataFrame from ``decode_rois()``.
    contrast_name : str, default 'Unknown'
        Name of the statistical contrast.
    radius_mm : float, default 6.0
        Sphere radius used.
    scoring : str, default 'chi2'
        Scoring/decoder metric.
    top_n : int, default 10
        Max terms per ROI.
    source : str, default 'conninfpy_edges'
        Evidence source label.
    backend : str, default 'NiMARE'
        Decoding library used.
    dataset_name : str, default 'Neurosynth'
        Underlying dataset name.
    decoder_method : str, default 'NeurosynthDecoder'
        Decoder class used.
    caveats : list of str, optional
        Custom warnings/limitations. If None, uses default caveats.
        
    Returns
    -------
    dict
        A JSON-compatible dictionary conforming to the evidence schema.
    """
    # 1. Query info
    if not edges.empty and 'roi_i' in edges.columns:
        roi_ids = sorted(list(pd.concat([edges['roi_i'], edges['roi_j']]).unique()))
        roi_ids = [int(x) for x in roi_ids]
    else:
        roi_ids = sorted(list(decoded_rois['roi_id'].unique()))
        roi_ids = [int(x) for x in roi_ids]

    network_pairs = []
    if not edges.empty:
        if 'network_pair' not in edges.columns:
            networks = np.asarray(atlas.networks)
            i_idx = edges['roi_i'].to_numpy()
            j_idx = edges['roi_j'].to_numpy()
            i_nets = networks[i_idx]
            j_nets = networks[j_idx]
            pair_col = np.array(["—".join(sorted((a, b))) for a, b in zip(i_nets, j_nets)])
        else:
            pair_col = edges['network_pair'].to_numpy()
            
        unique_pairs, counts = np.unique(pair_col, return_counts=True)
        for pair, count in zip(unique_pairs, counts):
            parts = pair.split("—")
            if len(parts) == 2:
                src, tgt = parts
            else:
                src = tgt = parts[0]
            network_pairs.append({
                "source": str(src),
                "target": str(tgt),
                "n_edges": int(count)
            })

    # Find atlas source or label
    atlas_name = atlas.source or "Unknown Atlas"
    if "Schaefer" in atlas_name:
        # short name
        pass
    else:
        # Check lengths to guess standard bundled atlases
        n_rois = len(atlas)
        if n_rois == 100:
            atlas_name = "Schaefer-100 Yeo-7"
        elif n_rois == 200:
            atlas_name = "Schaefer-200 Yeo-7"
        elif n_rois == 400:
            atlas_name = "Schaefer-400 Yeo-7"
        elif n_rois == 246:
            atlas_name = "BNA-246"

    query_info = {
        "source": source,
        "contrast": contrast_name,
        "atlas": atlas_name,
        "roi_ids": roi_ids,
        "edge_count": len(edges),
        "network_pairs": network_pairs
    }

    # 2. Decoder config
    decoder_config = {
        "backend": backend,
        "dataset": dataset_name,
        "method": decoder_method,
        "scoring": scoring,
        "radius_mm": float(radius_mm),
        "top_n": int(top_n)
    }

    # 3. Terms list
    terms_list = []
    for _, row in decoded_rois.iterrows():
        terms_list.append({
            "roi_id": int(row["roi_id"]),
            "roi_name": str(row["roi_name"]),
            "network": str(row["network"]),
            "rank": int(row["rank"]),
            "term": str(row["term"]),
            "score": float(row["score"])
        })

    # 4. Caveats
    if caveats is None:
        caveats = DEFAULT_CAVEATS

    return {
        "query": query_info,
        "decoder": decoder_config,
        "terms": terms_list,
        "caveats": list(caveats)
    }

def validate_evidence(evidence: Dict[str, Any]) -> None:
    """Enforce structural correctness and sanity rules on the evidence packet.
    
    Raises
    ------
    ValueError
        If any required fields or metadata are missing or malformed.
    """
    required_keys = {"query", "decoder", "terms", "caveats"}
    missing = required_keys - set(evidence.keys())
    if missing:
        raise ValueError(f"Missing required top-level evidence fields: {missing}")

    # Query validation
    query = evidence["query"]
    if not isinstance(query, dict):
        raise ValueError("Field 'query' must be a dictionary.")
    for k in ("source", "atlas"):
        if k not in query or not query[k]:
            raise ValueError(f"Field 'query.{k}' is required and cannot be empty.")

    # Decoder validation
    decoder = evidence["decoder"]
    if not isinstance(decoder, dict):
        raise ValueError("Field 'decoder' must be a dictionary.")
    for k in ("backend", "dataset", "method", "scoring"):
        if k not in decoder or not decoder[k]:
            raise ValueError(f"Field 'decoder.{k}' is required and cannot be empty.")

    # Terms validation
    terms = evidence["terms"]
    if not isinstance(terms, list):
        raise ValueError("Field 'terms' must be a list.")
    if not terms:
        raise ValueError("Field 'terms' list is empty. No decoding evidence available.")
        
    for idx, term_entry in enumerate(terms):
        if not isinstance(term_entry, dict):
            raise ValueError(f"Term entry at index {idx} is not a dictionary.")
        for k in ("term", "rank", "score"):
            if k not in term_entry:
                raise ValueError(f"Term entry at index {idx} is missing required field '{k}'.")
            if k == "term" and not term_entry[k]:
                raise ValueError(f"Term entry at index {idx} has an empty 'term' value.")

    # Caveats validation
    caveats = evidence["caveats"]
    if not isinstance(caveats, list):
        raise ValueError("Field 'caveats' must be a list of warning strings.")
