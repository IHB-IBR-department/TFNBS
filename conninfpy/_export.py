"""Edge-level export helpers for :class:`InferenceResult` /
:class:`OmnibusInferenceResult`.

Implements the shared logic behind :meth:`significant_edges` and
:meth:`to_csv` on both result classes. Kept in a single module so the
tail and omnibus paths share the same column-naming and atlas-merging
conventions.

.. note::

   *Significance is a statistical, not mechanistic, statement.* The
   tables this module produces are FWER/FDR-controlled claims about
   edge-wise null rejection; they do not by themselves establish that
   an edge corresponds to a specific cognitive or biological
   mechanism. Block-level patterns
   (``roi_i_network`` × ``roi_j_network``) carry more interpretive
   weight than individual edges; prefer ``sort='network_pair'`` for
   reporting.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import numpy.typing as npt

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .atlas import AtlasInfo


_VALID_TAILS = ("both", "positive", "negative")
_VALID_SORTS = ("p", "effect_size", "network_pair")


def _check_square(arr: npt.NDArray, name: str) -> None:
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(
            f"{name} must be a square (N, N) matrix; got shape {arr.shape}."
        )


def _row_tail_label(
    p_pos: npt.NDArray[np.float64],
    p_neg: npt.NDArray[np.float64],
    alpha: float,
) -> npt.NDArray[np.object_]:
    """Per-edge significance label: 'positive', 'negative', 'both', or ''."""
    pos_sig = p_pos <= alpha
    neg_sig = p_neg <= alpha
    out = np.full(p_pos.shape, "", dtype=object)
    out[pos_sig & ~neg_sig] = "positive"
    out[~pos_sig & neg_sig] = "negative"
    out[pos_sig & neg_sig] = "both"
    return out


def _annotate_with_atlas(
    df: pd.DataFrame,
    atlas: "AtlasInfo",
    *,
    n_nodes: int,
) -> pd.DataFrame:
    """Merge ROI names / networks / hemisphere / network_pair into ``df``."""
    if len(atlas) != n_nodes:
        raise ValueError(
            f"atlas has {len(atlas)} ROIs but the result is on "
            f"{n_nodes} nodes; cannot annotate."
        )
    labels = np.asarray(atlas.labels, dtype=object)
    networks = np.asarray(atlas.networks, dtype=object)
    hemi = (
        np.asarray(atlas.hemisphere, dtype=object)
        if atlas.hemisphere is not None
        else None
    )

    i_idx = df["roi_i"].to_numpy()
    j_idx = df["roi_j"].to_numpy()

    df = df.copy()
    df["roi_i_name"] = labels[i_idx]
    df["roi_j_name"] = labels[j_idx]
    df["roi_i_network"] = networks[i_idx]
    df["roi_j_network"] = networks[j_idx]
    # Canonicalize network_pair as a sorted tuple-string so that a
    # within-block (X, Y) is the same key as (Y, X).
    pair = np.array(
        [
            "—".join(sorted((a, b)))
            for a, b in zip(df["roi_i_network"], df["roi_j_network"])
        ],
        dtype=object,
    )
    df["network_pair"] = pair
    if hemi is not None:
        df["hemisphere_i"] = hemi[i_idx]
        df["hemisphere_j"] = hemi[j_idx]
    return df


def _column_order_tailed(with_atlas: bool) -> list:
    if with_atlas:
        cols = [
            "edge_id",
            "roi_i",
            "roi_i_name",
            "roi_i_network",
            "roi_j",
            "roi_j_name",
            "roi_j_network",
            "network_pair",
            "t_signed",
            "p_positive",
            "p_negative",
            "p_min",
            "tail",
            "hemisphere_i",
            "hemisphere_j",
        ]
    else:
        cols = [
            "edge_id",
            "roi_i",
            "roi_j",
            "t_signed",
            "p_positive",
            "p_negative",
            "p_min",
            "tail",
        ]
    return cols


def _column_order_omnibus(with_atlas: bool) -> list:
    if with_atlas:
        return [
            "edge_id",
            "roi_i",
            "roi_i_name",
            "roi_i_network",
            "roi_j",
            "roi_j_name",
            "roi_j_network",
            "network_pair",
            "F",
            "p_omnibus",
            "hemisphere_i",
            "hemisphere_j",
        ]
    return ["edge_id", "roi_i", "roi_j", "F", "p_omnibus"]


def build_tailed_dataframe(
    p_positive: npt.NDArray[np.float64],
    p_negative: npt.NDArray[np.float64],
    *,
    stat_signed: Optional[npt.NDArray[np.float64]] = None,
    atlas: Optional["AtlasInfo"] = None,
    alpha: float = 0.05,
    tail: str = "both",
    sort: str = "p",
    include_nonsig: bool = False,
    top_k: Optional[int] = None,
) -> pd.DataFrame:
    """Build the significant-edges table for a t / β inference result."""
    if not HAS_PANDAS:
        raise ImportError(
            "pandas is required for build_tailed_dataframe(). "
            "Install it with `pip install conninfpy[full]`."
        )
    if tail not in _VALID_TAILS:
        raise ValueError(
            f"tail must be one of {_VALID_TAILS}; got {tail!r}."
        )
    if sort not in _VALID_SORTS:
        raise ValueError(
            f"sort must be one of {_VALID_SORTS}; got {sort!r}."
        )
    if sort == "network_pair" and atlas is None:
        raise ValueError(
            "sort='network_pair' requires an atlas (no network labels "
            "available without one)."
        )
    if stat_signed is None:
        raise ValueError(
            "stat maps not populated; v2.0 pickled results are not "
            "exportable — rerun with v2.1+ pipeline."
        )

    _check_square(p_positive, "p_positive")
    _check_square(p_negative, "p_negative")
    if p_negative.shape != p_positive.shape:
        raise ValueError("p_positive and p_negative must have the same shape.")
    _check_square(stat_signed, "stat_signed")
    if stat_signed.shape != p_positive.shape:
        raise ValueError("stat_signed must match p_positive shape.")

    N = p_positive.shape[0]
    iu, ju = np.triu_indices(N, k=1)
    p_pos_v = p_positive[iu, ju]
    p_neg_v = p_negative[iu, ju]
    t_v = stat_signed[iu, ju]
    p_min_v = np.minimum(p_pos_v, p_neg_v)

    # Filter by `tail` parameter.
    if tail == "both":
        sig_mask = p_min_v <= alpha
    elif tail == "positive":
        sig_mask = p_pos_v <= alpha
    else:  # negative
        sig_mask = p_neg_v <= alpha

    keep = np.ones_like(sig_mask) if include_nonsig else sig_mask

    tail_label_full = _row_tail_label(
        p_positive, p_negative, alpha
    )[iu, ju]

    edge_id = np.arange(len(iu))[keep]
    df = pd.DataFrame(
        {
            "edge_id": edge_id,
            "roi_i": iu[keep],
            "roi_j": ju[keep],
            "t_signed": t_v[keep],
            "p_positive": p_pos_v[keep],
            "p_negative": p_neg_v[keep],
            "p_min": p_min_v[keep],
            "tail": tail_label_full[keep],
        }
    )

    with_atlas = atlas is not None
    if with_atlas:
        df = _annotate_with_atlas(df, atlas, n_nodes=N)

    df = _sort_dataframe(df, sort, by="p_min")

    if top_k is not None:
        df = df.head(int(top_k))

    cols = [c for c in _column_order_tailed(with_atlas) if c in df.columns]
    return df[cols].reset_index(drop=True)


def build_omnibus_dataframe(
    p_omnibus: npt.NDArray[np.float64],
    *,
    stat_omnibus: Optional[npt.NDArray[np.float64]] = None,
    atlas: Optional["AtlasInfo"] = None,
    alpha: float = 0.05,
    sort: str = "p",
    include_nonsig: bool = False,
    top_k: Optional[int] = None,
) -> pd.DataFrame:
    """Build the significant-edges table for an F-stat omnibus result."""
    if not HAS_PANDAS:
        raise ImportError(
            "pandas is required for build_omnibus_dataframe(). "
            "Install it with `pip install conninfpy[full]`."
        )
    if sort not in _VALID_SORTS:
        raise ValueError(
            f"sort must be one of {_VALID_SORTS}; got {sort!r}."
        )
    if sort == "network_pair" and atlas is None:
        raise ValueError(
            "sort='network_pair' requires an atlas."
        )
    if stat_omnibus is None:
        raise ValueError(
            "stat map not populated; v2.0 pickled F-stat results are "
            "not exportable — rerun with v2.1+ pipeline."
        )

    _check_square(p_omnibus, "p_omnibus")
    _check_square(stat_omnibus, "stat_omnibus")
    if stat_omnibus.shape != p_omnibus.shape:
        raise ValueError("stat_omnibus must match p_omnibus shape.")

    N = p_omnibus.shape[0]
    iu, ju = np.triu_indices(N, k=1)
    p_v = p_omnibus[iu, ju]
    f_v = stat_omnibus[iu, ju]

    keep = np.ones_like(p_v, dtype=bool) if include_nonsig else (p_v <= alpha)

    df = pd.DataFrame(
        {
            "edge_id": np.arange(len(iu))[keep],
            "roi_i": iu[keep],
            "roi_j": ju[keep],
            "F": f_v[keep],
            "p_omnibus": p_v[keep],
        }
    )

    with_atlas = atlas is not None
    if with_atlas:
        df = _annotate_with_atlas(df, atlas, n_nodes=N)

    df = _sort_dataframe(df, sort, by="p_omnibus", effect_col="F")

    if top_k is not None:
        df = df.head(int(top_k))

    cols = [c for c in _column_order_omnibus(with_atlas) if c in df.columns]
    return df[cols].reset_index(drop=True)


def _sort_dataframe(
    df: pd.DataFrame,
    sort: str,
    *,
    by: str,
    effect_col: str = "t_signed",
) -> pd.DataFrame:
    if df.empty:
        return df
    if sort == "p":
        return df.sort_values(by=by, ascending=True, kind="mergesort")
    if sort == "effect_size":
        if effect_col == "t_signed":
            key = df[effect_col].abs()
        else:
            key = df[effect_col]
        return df.assign(_k=key).sort_values(
            by="_k", ascending=False, kind="mergesort"
        ).drop(columns="_k")
    # network_pair: group by (sorted) network pair then by p within group
    return df.sort_values(
        by=["network_pair", by], ascending=[True, True], kind="mergesort"
    )


__all__ = ["build_tailed_dataframe", "build_omnibus_dataframe"]
