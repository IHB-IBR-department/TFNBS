"""Structured return types for the permutation pipelines.

:class:`InferenceResult` is a thin enrichment of
:class:`~conninfpy._compat.TailResult` that carries run-time metadata
(method, n_permutations, acceleration, wall-time, optional null
distribution), observed statistic maps, and provenance about any
upstream harmonization step alongside the per-tail p-value matrices.

:class:`OmnibusInferenceResult` is the sibling type returned by the
F-statistic omnibus path (multi-row contrast). It carries the same
metadata + provenance attributes but only a single ``'omnibus'``
p-value map — F has no sign, so the positive/negative split is not
defined for it.

Backward-compatible by design: both classes are ``dict`` subclasses
(via :class:`TailResult`), so existing code that accesses
``result['positive']`` / ``result['negative']`` (or
``result['omnibus']`` for the F path) continues to work. The attribute
APIs (``result.positive``, ``result.stat_signed``, ``result.method``,
``print(result)``) are additive.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import numpy.typing as npt

from ._compat import TailResult, make_tail_result


class InferenceResult(TailResult):
    """Per-tail p-value maps + metadata from a single permutation run.

    Behaves as a two-key dict (``'positive'`` / ``'negative'``) and also
    exposes named attributes.

    Attributes
    ----------
    positive, negative : ndarray
        Per-tail p-value (or score) matrices.
    method : str
        Enhancement method label, e.g. ``'tfnbs'``, ``'cnbs'``, ``'nbs'``,
        ``'tstat'``, ``'bh_fdr'``.
    n_permutations : int
        Number of permutations used (0 for parametric methods).
    acceleration : str or None
        ``'gpd'``, ``'gamma'``, or ``None`` for empirical p-values.
    wall_time_s : float or None
        Wall-clock seconds for the inference call (when measured).
    null_max_dist : dict[str, ndarray] or None
        Optional max-stat null distribution per tail (set when
        ``store_null=True`` is passed to the pipeline).
    stat_positive, stat_negative : ndarray or None
        Observed one-tail-clipped (non-negative) test statistic maps.
        ``stat_positive[i, j]`` equals ``max(t_ij, 0)``; ``stat_negative``
        equals ``max(-t_ij, 0)``. The original signed effect map is
        recoverable as ``stat_signed`` (= positive − negative). Populated
        by the pipeline since v2.1; ``None`` on older pickled results.
    stat_type : {'tstat', 'beta'}
        Which statistic the stat maps hold. ``'tstat'`` for the default
        t-statistic path, ``'beta'`` when the GLM is run with
        ``stat_type='beta'``.
    harmonized : bool
        True iff ComBat ran upstream of inference (set by
        :func:`~conninfpy.analyze`).
    preserve_provided : bool
        True iff a non-None ``preserve`` was supplied to ComBat.
    strata_provided : bool
        True iff inference used stratified permutation.
    combat_diagnostics : dict or None
        ComBat diagnostics blob when available.
        ``strategy='combat_only'`` and ``strategy='combat_site_dummies_glm'``
        diagnostics include between-site variance before/after and the explicit
        ``between_site_variance_ratio_after_over_before`` key used by
        :func:`conninfpy.analyze` for residual-site flags.
        ``strategy='site_dummies_glm'`` records provenance but does not run ComBat.
        ``legacy_strategy`` may contain the old ``'D'`` / ``'E'`` label from
        ``[[protocol_combat_implementation]]`` for compatibility.
    """

    method: str
    n_permutations: int
    acceleration: Optional[str]
    wall_time_s: Optional[float]
    null_max_dist: Optional[Dict[str, npt.NDArray[np.float64]]]
    stat_positive: Optional[npt.NDArray[np.float64]]
    stat_negative: Optional[npt.NDArray[np.float64]]
    stat_type: str
    harmonized: bool
    preserve_provided: bool
    strata_provided: bool
    combat_diagnostics: Optional[Dict[str, Any]]
    e_grid: Optional[npt.NDArray[np.float64]]
    h_grid: Optional[npt.NDArray[np.float64]]

    def __new__(
        cls,
        positive: npt.NDArray[np.float64],
        negative: npt.NDArray[np.float64],
        *,
        method: str = "tstat",
        n_permutations: int = 0,
        acceleration: Optional[str] = None,
        wall_time_s: Optional[float] = None,
        null_max_dist: Optional[Dict[str, npt.NDArray[np.float64]]] = None,
        stat_positive: Optional[npt.NDArray[np.float64]] = None,
        stat_negative: Optional[npt.NDArray[np.float64]] = None,
        stat_type: str = "tstat",
        harmonized: bool = False,
        preserve_provided: bool = False,
        strata_provided: bool = False,
        combat_diagnostics: Optional[Dict[str, Any]] = None,
        e_grid: Optional[npt.NDArray[np.float64]] = None,
        h_grid: Optional[npt.NDArray[np.float64]] = None,
    ) -> "InferenceResult":
        instance = super().__new__(cls)
        TailResult.__init__(
            instance, [("positive", positive), ("negative", negative)]
        )
        instance.method = method
        instance.n_permutations = n_permutations
        instance.acceleration = acceleration
        instance.wall_time_s = wall_time_s
        instance.null_max_dist = null_max_dist
        instance.stat_positive = stat_positive
        instance.stat_negative = stat_negative
        instance.stat_type = stat_type
        instance.harmonized = harmonized
        instance.preserve_provided = preserve_provided
        instance.strata_provided = strata_provided
        instance.combat_diagnostics = combat_diagnostics
        instance.e_grid = e_grid
        instance.h_grid = h_grid
        return instance

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
        # __new__ already populated the dict and attributes; suppress the
        # default dict.__init__(*args, **kwargs) which would otherwise wipe
        # the canonical keys when called with kwargs like method="tfnbs".
        return

    @property
    def positive(self) -> npt.NDArray[np.float64]:
        return self["positive"]

    @property
    def negative(self) -> npt.NDArray[np.float64]:
        return self["negative"]

    @property
    def stat_signed(self) -> Optional[npt.NDArray[np.float64]]:
        """Recovered signed effect map (``stat_positive − stat_negative``).

        Both tail arrays are non-negative one-tail clips of the original
        signed statistic; for any edge exactly one of them is zero, so
        their difference reconstructs the original t/β map.

        Returns ``None`` when stat maps are unavailable (older pickled
        results from before v2.1).
        """
        if self.stat_positive is None or self.stat_negative is None:
            return None
        return self.stat_positive - self.stat_negative

    @property
    def is_grid(self) -> bool:
        """True iff this result holds a multi-(E, H) parameter grid.

        When True, ``self['positive']`` and ``self['negative']`` are
        ``(N, N, K)`` tensors and ``self.e_grid`` / ``self.h_grid``
        give the parameter cells. Use :meth:`select` to project to a
        single 2D cell.
        """
        return self["positive"].ndim == 3

    def _project_2d(
        self,
        arr: npt.NDArray[np.float64],
        param_idx: Optional[int],
    ) -> npt.NDArray[np.float64]:
        """Project ``arr`` to a 2D ``(N, N)`` slice.

        For 2D inputs, returns the array unchanged (and rejects
        ``param_idx`` ≠ ``None``). For 3D inputs, slices at
        ``param_idx`` along the last axis.
        """
        if arr.ndim == 2:
            if param_idx is not None and param_idx != 0:
                raise ValueError(
                    f"param_idx={param_idx} given, but this result is "
                    f"not a parameter grid (2D p-maps)."
                )
            return arr
        if param_idx is None:
            raise ValueError(
                f"Result holds a (N, N, K={arr.shape[-1]}) parameter "
                f"grid; pass param_idx to select a cell (see .e_grid "
                f"/ .h_grid), or call .select(param_idx) first."
            )
        return arr[:, :, param_idx]

    def select(self, param_idx: int) -> "InferenceResult":
        """Project a multi-(E, H) result to the single cell ``param_idx``.

        Returns a new :class:`InferenceResult` with 2D p-maps and the
        same metadata; the returned object has ``is_grid is False``
        and ``e_grid`` / ``h_grid`` reduced to single-element arrays
        recording which cell was selected.
        """
        if not self.is_grid:
            raise ValueError(
                "select() requires a multi-(E, H) result; this one is 2D."
            )
        e_sel = self.e_grid[param_idx:param_idx + 1] if self.e_grid is not None else None
        h_sel = self.h_grid[param_idx:param_idx + 1] if self.h_grid is not None else None
        return InferenceResult(
            self["positive"][:, :, param_idx].copy(),
            self["negative"][:, :, param_idx].copy(),
            method=self.method,
            n_permutations=self.n_permutations,
            acceleration=self.acceleration,
            wall_time_s=self.wall_time_s,
            null_max_dist=None,
            stat_positive=self.stat_positive,
            stat_negative=self.stat_negative,
            stat_type=self.stat_type,
            harmonized=self.harmonized,
            preserve_provided=self.preserve_provided,
            strata_provided=self.strata_provided,
            combat_diagnostics=self.combat_diagnostics,
            e_grid=e_sel,
            h_grid=h_sel,
        )

    def n_significant(
        self,
        alpha: float = 0.05,
        *,
        param_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Count significant edges per tail at threshold ``alpha``.

        For grid results, pass ``param_idx`` to count one cell;
        omit to get a list of counts per cell along the parameter axis.
        """
        out: Dict[str, Any] = {}
        for tail in ("positive", "negative"):
            arr = self[tail]
            if arr.ndim == 2:
                if arr.shape[0] == arr.shape[1]:
                    iu = np.triu_indices_from(arr, k=1)
                    out[tail] = int(np.sum(arr[iu] <= alpha))
                else:
                    out[tail] = int(np.sum(arr <= alpha))
            else:
                # (N, N, K) grid
                if param_idx is not None:
                    sub = arr[:, :, param_idx]
                    iu = np.triu_indices_from(sub, k=1)
                    out[tail] = int(np.sum(sub[iu] <= alpha))
                else:
                    iu = np.triu_indices(arr.shape[0], k=1)
                    sub = arr[iu[0], iu[1], :]  # (n_edges, K)
                    out[tail] = [int(np.sum(sub[:, k] <= alpha))
                                 for k in range(arr.shape[-1])]
        return out

    def significant_edges(
        self,
        atlas: Optional[Any] = None,
        *,
        alpha: float = 0.05,
        tail: str = "both",
        sort: str = "p",
        include_nonsig: bool = False,
        top_k: Optional[int] = None,
        param_idx: Optional[int] = None,
    ) -> "Any":  # pd.DataFrame, kept loose to avoid hard typing import
        """Sorted table of significant edges with optional atlas annotation.

        Parameters
        ----------
        atlas : :class:`~conninfpy.AtlasInfo`, optional
            When supplied, adds ROI names, RS-network labels,
            hemisphere, and a canonical ``network_pair`` column. Length
            must match ``self['positive'].shape[0]``.
        alpha : float, default 0.05
            Edge-wise significance threshold.
        tail : {'both', 'positive', 'negative'}, default 'both'
            Which tail(s) to filter on. ``'both'`` keeps edges where
            either ``p_positive`` or ``p_negative`` falls below
            ``alpha``; the per-row ``tail`` column reports which
            tail(s) actually hit significance for that edge.
        sort : {'p', 'effect_size', 'network_pair'}, default 'p'
            Sort key. ``'p'`` — ascending ``min(p_positive, p_negative)``.
            ``'effect_size'`` — descending ``|t_signed|``.
            ``'network_pair'`` — group by canonical network pair,
            then by ``p`` ascending within group; requires ``atlas``.
        include_nonsig : bool, default False
            If True, include all edges (not just significant ones).
        top_k : int, optional
            Keep only the top-K rows after sorting.

        Returns
        -------
        pandas.DataFrame
            Columns (with atlas): ``edge_id, roi_i, roi_i_name,
            roi_i_network, roi_j, roi_j_name, roi_j_network,
            network_pair, t_signed, p_positive, p_negative, p_min,
            tail, hemisphere_i, hemisphere_j``. Without an atlas the
            ``roi_*_name``, ``roi_*_network``, ``hemisphere_*`` and
            ``network_pair`` columns are omitted.

        Notes
        -----
        Significance is an FWER/FDR-controlled claim about edge-wise
        null rejection; it does not by itself establish that an edge
        corresponds to a specific cognitive or biological mechanism.
        For reporting, prefer block-level patterns
        (``roi_i_network`` × ``roi_j_network``) over individual edges
        — use ``sort='network_pair'`` to group them.

        Raises
        ------
        ValueError
            If ``stat_positive`` / ``stat_negative`` are missing (e.g.
            pickled pre-v2.1 results), or if ``sort='network_pair'``
            without an atlas.
        """
        from ._export import build_tailed_dataframe

        return build_tailed_dataframe(
            self._project_2d(self["positive"], param_idx),
            self._project_2d(self["negative"], param_idx),
            stat_signed=self.stat_signed,
            atlas=atlas,
            alpha=alpha,
            tail=tail,
            sort=sort,
            include_nonsig=include_nonsig,
            top_k=top_k,
        )

    def to_csv(
        self,
        path: "Any",
        *,
        atlas: Optional[Any] = None,
        alpha: float = 0.05,
        tail: str = "both",
        sort: str = "p",
        include_nonsig: bool = False,
        top_k: Optional[int] = None,
        index: bool = False,
        param_idx: Optional[int] = None,
        **to_csv_kwargs: Any,
    ) -> None:
        """Convenience: ``self.significant_edges(...).to_csv(path, ...)``.

        Same filtering / sorting / atlas keywords as
        :meth:`significant_edges`. Extra keyword arguments are
        forwarded to :meth:`pandas.DataFrame.to_csv`.
        """
        df = self.significant_edges(
            atlas=atlas,
            alpha=alpha,
            tail=tail,
            sort=sort,
            include_nonsig=include_nonsig,
            top_k=top_k,
            param_idx=param_idx,
        )
        df.to_csv(path, index=index, **to_csv_kwargs)

    def decoded_edges(self, atlas: Any, *, top_n: int = 5, **kwargs: Any) -> pd.DataFrame:
        """Decode significant edges using Neurosynth/NiMARE meta-analytic decoding.
        
        Convenience wrapper that calls ``significant_edges`` first, and then annotates
        it with decoded terms using ``annotate_edge_table``.
        """
        from .decode import annotate_edge_table
        
        # Split kwargs between significant_edges and annotate_edge_table
        sig_keys = {'alpha', 'tail', 'sort', 'include_nonsig', 'top_k', 'param_idx'}
        sig_kwargs = {k: v for k, v in kwargs.items() if k in sig_keys}
        decode_kwargs = {k: v for k, v in kwargs.items() if k not in sig_keys}
        
        edges = self.significant_edges(atlas=atlas, **sig_kwargs)
        return annotate_edge_table(edges, atlas, top_n=top_n, **decode_kwargs)

    def __repr__(self) -> str:  # pragma: no cover (cosmetic)
        nsig = self.n_significant(0.05)
        wall = (
            f"{self.wall_time_s:.2f}s" if self.wall_time_s is not None else "—"
        )
        accel = self.acceleration if self.acceleration else "empirical"
        prov = []
        if self.harmonized:
            prov.append("harmonized")
        if self.strata_provided:
            prov.append("strata")
        prov_str = (" prov=" + ",".join(prov)) if prov else ""
        return (
            f"InferenceResult(method={self.method!r}, "
            f"n_perms={self.n_permutations}, accel={accel}, "
            f"sig@α=0.05: pos={nsig['positive']}, neg={nsig['negative']}, "
            f"wall_time={wall}{prov_str})"
        )


class OmnibusInferenceResult(TailResult):
    """Single-tail p-value map + metadata for the F-stat omnibus path.

    Behaves as a one-key dict (``'omnibus'``) and exposes the same
    metadata / provenance attributes as :class:`InferenceResult` plus
    the observed F-statistic map (``stat_omnibus``). Sibling — not
    subclass — of :class:`InferenceResult` because F has no sign and
    a subclass that raised on ``.positive``/``.negative`` would
    violate LSP.

    Use ``isinstance(r, OmnibusInferenceResult)`` to dispatch on the
    F-stat path; ``isinstance(r, InferenceResult)`` will be ``False``
    for this type.
    """

    method: str
    n_permutations: int
    acceleration: Optional[str]
    wall_time_s: Optional[float]
    null_max_dist: Optional[Dict[str, npt.NDArray[np.float64]]]
    stat_omnibus: Optional[npt.NDArray[np.float64]]
    stat_type: str
    harmonized: bool
    preserve_provided: bool
    strata_provided: bool
    combat_diagnostics: Optional[Dict[str, Any]]

    def __new__(
        cls,
        omnibus: npt.NDArray[np.float64],
        *,
        method: str = "tstat",
        n_permutations: int = 0,
        acceleration: Optional[str] = None,
        wall_time_s: Optional[float] = None,
        null_max_dist: Optional[Dict[str, npt.NDArray[np.float64]]] = None,
        stat_omnibus: Optional[npt.NDArray[np.float64]] = None,
        stat_type: str = "fstat",
        harmonized: bool = False,
        preserve_provided: bool = False,
        strata_provided: bool = False,
        combat_diagnostics: Optional[Dict[str, Any]] = None,
    ) -> "OmnibusInferenceResult":
        instance = super().__new__(cls)
        # Don't go through TailResult's positive/negative initializer —
        # this type uses a single 'omnibus' key.
        dict.__init__(instance, [("omnibus", omnibus)])
        instance.method = method
        instance.n_permutations = n_permutations
        instance.acceleration = acceleration
        instance.wall_time_s = wall_time_s
        instance.null_max_dist = null_max_dist
        instance.stat_omnibus = stat_omnibus
        instance.stat_type = stat_type
        instance.harmonized = harmonized
        instance.preserve_provided = preserve_provided
        instance.strata_provided = strata_provided
        instance.combat_diagnostics = combat_diagnostics
        return instance

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
        return

    @property
    def omnibus(self) -> npt.NDArray[np.float64]:
        return self["omnibus"]

    def n_significant(self, alpha: float = 0.05) -> Dict[str, int]:
        arr = self["omnibus"]
        if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
            iu = np.triu_indices_from(arr, k=1)
            return {"omnibus": int(np.sum(arr[iu] <= alpha))}
        return {"omnibus": int(np.sum(arr <= alpha))}

    def significant_edges(
        self,
        atlas: Optional[Any] = None,
        *,
        alpha: float = 0.05,
        sort: str = "p",
        include_nonsig: bool = False,
        top_k: Optional[int] = None,
    ) -> "Any":
        """Sorted table of significant edges (F-stat omnibus path).

        Same conventions as
        :meth:`InferenceResult.significant_edges` but with a single
        unsigned ``F`` / ``p_omnibus`` pair instead of the tail split.
        ``sort='effect_size'`` ranks by ``F`` descending.

        Notes
        -----
        Significance is an FWER/FDR-controlled claim about edge-wise
        null rejection; it does not by itself establish that an edge
        corresponds to a specific cognitive or biological mechanism.
        """
        from ._export import build_omnibus_dataframe

        return build_omnibus_dataframe(
            self["omnibus"],
            stat_omnibus=self.stat_omnibus,
            atlas=atlas,
            alpha=alpha,
            sort=sort,
            include_nonsig=include_nonsig,
            top_k=top_k,
        )

    def to_csv(
        self,
        path: "Any",
        *,
        atlas: Optional[Any] = None,
        alpha: float = 0.05,
        sort: str = "p",
        include_nonsig: bool = False,
        top_k: Optional[int] = None,
        index: bool = False,
        **to_csv_kwargs: Any,
    ) -> None:
        """Convenience: ``self.significant_edges(...).to_csv(path, ...)``."""
        df = self.significant_edges(
            atlas=atlas,
            alpha=alpha,
            sort=sort,
            include_nonsig=include_nonsig,
            top_k=top_k,
        )
        df.to_csv(path, index=index, **to_csv_kwargs)

    def __repr__(self) -> str:  # pragma: no cover (cosmetic)
        nsig = self.n_significant(0.05)["omnibus"]
        wall = (
            f"{self.wall_time_s:.2f}s" if self.wall_time_s is not None else "—"
        )
        accel = self.acceleration if self.acceleration else "empirical"
        prov = []
        if self.harmonized:
            prov.append("harmonized")
        if self.strata_provided:
            prov.append("strata")
        prov_str = (" prov=" + ",".join(prov)) if prov else ""
        return (
            f"OmnibusInferenceResult(method={self.method!r}, "
            f"n_perms={self.n_permutations}, accel={accel}, "
            f"sig@α=0.05: omnibus={nsig}, wall_time={wall}{prov_str})"
        )


def make_inference_result(
    tail_result: TailResult,
    *,
    method: str,
    n_permutations: int = 0,
    acceleration: Optional[str] = None,
    wall_time_s: Optional[float] = None,
    null_max_dist: Optional[Dict[str, npt.NDArray[np.float64]]] = None,
    stat_positive: Optional[npt.NDArray[np.float64]] = None,
    stat_negative: Optional[npt.NDArray[np.float64]] = None,
    stat_type: str = "tstat",
    harmonized: bool = False,
    preserve_provided: bool = False,
    strata_provided: bool = False,
    combat_diagnostics: Optional[Dict[str, Any]] = None,
    e_grid: Optional[npt.NDArray[np.float64]] = None,
    h_grid: Optional[npt.NDArray[np.float64]] = None,
) -> InferenceResult:
    """Promote a :class:`TailResult` into a metadata-enriched
    :class:`InferenceResult` without copying the p-value arrays."""
    return InferenceResult(
        tail_result["positive"],
        tail_result["negative"],
        method=method,
        n_permutations=n_permutations,
        acceleration=acceleration,
        wall_time_s=wall_time_s,
        null_max_dist=null_max_dist,
        stat_positive=stat_positive,
        stat_negative=stat_negative,
        stat_type=stat_type,
        harmonized=harmonized,
        preserve_provided=preserve_provided,
        strata_provided=strata_provided,
        combat_diagnostics=combat_diagnostics,
        e_grid=e_grid,
        h_grid=h_grid,
    )


__all__ = [
    "InferenceResult",
    "OmnibusInferenceResult",
    "make_inference_result",
]
