import numpy as np
import pandas as pd
from typing import Sequence, Callable, Optional, Any
from ._decode_cache import fetch_neurosynth_dataset
from .atlas import AtlasInfo

def decode_rois(atlas: AtlasInfo,
                roi_ids: Optional[Sequence[int]] = None,
                *,
                top_n: int = 10,
                radius_mm: float = 6.0,
                scoring: str = 'chi2',
                dataset: Optional[Any] = None,
                term_filter: Optional[Callable[[str], bool]] = None,
            ) -> pd.DataFrame:
    """Decode a set of ROI coordinates against the Neurosynth database.
    
    Parameters
    ----------
    atlas : AtlasInfo
        Atlas metadata containing centroids in ``atlas.coords``.
    roi_ids : sequence of int, optional
        ROI indices to decode. If None, decodes all regions in the atlas.
    top_n : int, default 10
        Number of top terms to return per ROI.
    radius_mm : float, default 6.0
        Sphere radius around MNI centroids for study lookup.
    scoring : str, default 'chi2'
        Scoring/association method. Currently mapped to NeurosynthDecoder.
    dataset : nimare.Dataset, optional
        Pre-loaded NiMARE dataset. If None, loads from cache.
    term_filter : callable, optional
        Function taking a term string and returning True to keep it.
        
    Returns
    -------
    pandas.DataFrame
        Columns: ``roi_id``, ``roi_name``, ``network``, ``rank``, ``term``, ``score``.
    """
    if atlas.coords is None:
        raise ValueError("Atlas coordinates are None; cannot decode without spatial centroids.")
        
    if roi_ids is None:
        roi_ids = list(range(len(atlas)))
        
    if dataset is None:
        dataset = fetch_neurosynth_dataset()
        
    try:
        from nimare.decode.discrete import NeurosynthDecoder
    except ImportError:
        raise ImportError("conninfpy.decode requires 'pip install conninfpy[decode]'")
        
    # Fit decoder on full dataset
    decoder = NeurosynthDecoder(frequency_threshold=0.001, prior=0.5)
    decoder.fit(dataset)
    
    rows = []
    
    for roi_id in roi_ids:
        roi_name = atlas.labels[roi_id]
        network_name = atlas.networks[roi_id]
        coord = atlas.coords[roi_id]
        x, y, z = coord
        
        # Distance calculation
        dists = np.sqrt(
            (dataset.coordinates['x'] - x) ** 2 +
            (dataset.coordinates['y'] - y) ** 2 +
            (dataset.coordinates['z'] - z) ** 2
        )
        active_ids = dataset.coordinates.loc[dists <= radius_mm, 'id'].unique().tolist()
        
        if not active_ids:
            rows.append({
                "roi_id": roi_id,
                "roi_name": roi_name,
                "network": network_name,
                "rank": 1,
                "term": "inconclusive",
                "score": 0.0
            })
            continue
            
        decoded_df = decoder.transform(active_ids)
        
        z_col = None
        for candidate in ['z_assoc', 'z', 'z_association', 'est']:
            if candidate in decoded_df.columns:
                z_col = candidate
                break
        if z_col is None:
            z_col = decoded_df.columns[0]
            
        sorted_df = decoded_df.sort_values(by=z_col, ascending=False)
        
        rank = 1
        for term, row_data in sorted_df.iterrows():
            term_str = str(term)
            score_val = float(row_data[z_col])
            
            if term_filter is not None and not term_filter(term_str):
                continue
                
            rows.append({
                "roi_id": roi_id,
                "roi_name": roi_name,
                "network": network_name,
                "rank": rank,
                "term": term_str,
                "score": score_val
            })
            rank += 1
            if rank > top_n:
                break
                
        if rank == 1:
            rows.append({
                "roi_id": roi_id,
                "roi_name": roi_name,
                "network": network_name,
                "rank": 1,
                "term": "inconclusive",
                "score": 0.0
            })
            
    return pd.DataFrame(rows)

def annotate_edge_table(edges: pd.DataFrame,
                        atlas: AtlasInfo,
                        *,
                        top_n: int = 5,
                        sep: str = '; ',
                        dataset: Optional[Any] = None,
                        **decode_kwargs) -> pd.DataFrame:
    """Add term annotations to an edges DataFrame.
    
    Parameters
    ----------
    edges : pandas.DataFrame
        Significant edges DataFrame. Must contain ``roi_i`` and ``roi_j`` columns.
    atlas : AtlasInfo
        Atlas info containing centroid coordinates.
    top_n : int, default 5
        Number of top terms to join.
    sep : str, default '; '
        Separator for joining terms.
    dataset : nimare.Dataset, optional
        Pre-loaded NiMARE dataset.
    **decode_kwargs
        Forwarded to ``decode_rois``.
        
    Returns
    -------
    pandas.DataFrame
        Input DataFrame annotated with ``roi_i_terms``, ``roi_j_terms``,
        ``roi_i_top_term``, ``roi_j_top_term``.
    """
    if edges.empty:
        df = edges.copy()
        for col in ['roi_i_terms', 'roi_j_terms', 'roi_i_top_term', 'roi_j_top_term']:
            df[col] = pd.Series(dtype=object)
        return df
        
    roi_ids = pd.concat([edges['roi_i'], edges['roi_j']]).unique().tolist()
    
    decoded = decode_rois(
        atlas,
        roi_ids,
        top_n=top_n,
        dataset=dataset,
        **decode_kwargs
    )
    
    roi_terms = {}
    roi_top_term = {}
    
    for roi_id, group in decoded.groupby('roi_id'):
        sorted_group = group.sort_values('rank')
        terms_list = sorted_group['term'].tolist()
        
        roi_terms[roi_id] = sep.join(terms_list)
        roi_top_term[roi_id] = terms_list[0] if terms_list else "inconclusive"
        
    df = edges.copy()
    df['roi_i_terms'] = df['roi_i'].map(roi_terms).fillna("inconclusive")
    df['roi_j_terms'] = df['roi_j'].map(roi_terms).fillna("inconclusive")
    df['roi_i_top_term'] = df['roi_i'].map(roi_top_term).fillna("inconclusive")
    df['roi_j_top_term'] = df['roi_j'].map(roi_top_term).fillna("inconclusive")
    
    return df
