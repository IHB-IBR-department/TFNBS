"""Unified random-state handling.

The public pipelines accept ``rng: int | numpy.random.Generator | None`` —
an int seed, a pre-built numpy ``Generator``, or ``None`` for
non-deterministic behaviour. Internally the pipelines still use integer
seeds (because :class:`multiprocessing.Pool` workers need picklable seeds
per task), so this module provides a single conversion point.
"""
from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np

RngLike = Union[None, int, np.random.Generator, np.random.RandomState]


def resolve_seed(
    rng: RngLike = None,
    legacy_random_state: Optional[int] = None,
) -> Optional[int]:
    """Resolve an ``rng`` argument to a single integer seed.

    Parameters
    ----------
    rng : None, int, ``numpy.random.Generator``, or ``numpy.random.RandomState``
        The user-facing random-state argument.
    legacy_random_state : int, optional
        Value of the v1.x ``random_state`` argument. Used when ``rng`` is
        ``None``; emits a :class:`DeprecationWarning` to nudge callers
        toward the new ``rng=`` argument.

    Returns
    -------
    int or None
        Integer seed suitable for :func:`numpy.random.seed`,
        :class:`numpy.random.RandomState`, or as a per-task seed in
        :class:`multiprocessing.Pool`. Returns ``None`` for
        non-deterministic behaviour.
    """
    if rng is None:
        if legacy_random_state is not None:
            return int(legacy_random_state)
        return None
    if isinstance(rng, (int, np.integer)):
        return int(rng)
    if isinstance(rng, np.random.Generator):
        # Draw a single int from the generator so subsequent calls remain
        # reproducible from the user's seed.
        return int(rng.integers(0, 2**31 - 1))
    if isinstance(rng, np.random.RandomState):
        return int(rng.randint(0, 2**31 - 1))
    raise TypeError(
        f"rng must be None, int, numpy.random.Generator, or "
        f"numpy.random.RandomState; got {type(rng).__name__}."
    )


def warn_legacy_random_state(callers_kwarg: str = "random_state") -> None:
    """Emit a one-time DeprecationWarning for the legacy ``random_state``."""
    warnings.warn(
        f"The {callers_kwarg!r} keyword is deprecated; use ``rng`` instead. "
        "It accepts an int seed, a numpy.random.Generator, or None. "
        f"{callers_kwarg!r} will be removed in conninfpy v2.1.",
        DeprecationWarning,
        stacklevel=3,
    )


__all__ = ["resolve_seed", "warn_legacy_random_state", "RngLike"]
