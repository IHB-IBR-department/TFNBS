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
        ComBat diagnostics blob (between-site variance ratio, etc.) when
        ``harmonized=True``.
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

    def n_significant(self, alpha: float = 0.05) -> Dict[str, int]:
        """Count significant edges per tail at threshold ``alpha``."""
        out = {}
        for tail in ("positive", "negative"):
            arr = self[tail]
            if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
                # Symmetric matrix: count upper triangle only
                iu = np.triu_indices_from(arr, k=1)
                out[tail] = int(np.sum(arr[iu] <= alpha))
            else:
                out[tail] = int(np.sum(arr <= alpha))
        return out

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
    )


__all__ = [
    "InferenceResult",
    "OmnibusInferenceResult",
    "make_inference_result",
]
