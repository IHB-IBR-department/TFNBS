"""Structured return type for permutation pipelines.

:class:`InferenceResult` is a thin enrichment of
:class:`~conninfpy._compat.TailResult` that carries run-time metadata
(method, n_permutations, acceleration, wall-time, optional null
distribution) alongside the per-tail p-value matrices.

Backward-compatible by design: an :class:`InferenceResult` *is* a
``dict`` (via :class:`TailResult`), so all existing code that accesses
``result['positive']`` / ``result['negative']`` continues to work. The
new attribute-style API (``result.positive``, ``result.method``,
``print(result)``) is additive.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import numpy.typing as npt

from ._compat import TailResult, make_tail_result


class InferenceResult(TailResult):
    """Per-tail p-value maps + metadata from a single permutation run.

    Behaves as a two-key dict (``'positive'`` / ``'negative'``) and also
    exposes named attributes:

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
    """

    # Type hints for the metadata attributes set in __new__ (mypy needs
    # explicit class-level declarations because dict subclasses with
    # extra attributes confuse it otherwise).
    method: str
    n_permutations: int
    acceleration: Optional[str]
    wall_time_s: Optional[float]
    null_max_dist: Optional[Dict[str, npt.NDArray[np.float64]]]

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
    ) -> "InferenceResult":
        instance = super().__new__(cls)
        super().__init__(
            instance, [("positive", positive), ("negative", negative)]
        )
        instance.method = method
        instance.n_permutations = n_permutations
        instance.acceleration = acceleration
        instance.wall_time_s = wall_time_s
        instance.null_max_dist = null_max_dist
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
        return (
            f"InferenceResult(method={self.method!r}, "
            f"n_perms={self.n_permutations}, accel={accel}, "
            f"sig@α=0.05: pos={nsig['positive']}, neg={nsig['negative']}, "
            f"wall_time={wall})"
        )


def make_inference_result(
    tail_result: TailResult,
    *,
    method: str,
    n_permutations: int = 0,
    acceleration: Optional[str] = None,
    wall_time_s: Optional[float] = None,
    null_max_dist: Optional[Dict[str, npt.NDArray[np.float64]]] = None,
) -> InferenceResult:
    """Promote a :class:`TailResult` into a metadata-enriched
    :class:`InferenceResult` without copying the arrays."""
    return InferenceResult(
        tail_result["positive"],
        tail_result["negative"],
        method=method,
        n_permutations=n_permutations,
        acceleration=acceleration,
        wall_time_s=wall_time_s,
        null_max_dist=null_max_dist,
    )


__all__ = ["InferenceResult", "make_inference_result"]
