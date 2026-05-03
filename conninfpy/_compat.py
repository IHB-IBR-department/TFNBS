"""Backwards-compatible dict-key aliases.

Both pipelines (t-test via :mod:`pairwise_stats` and GLM via :mod:`glm_stats`)
return per-tail dictionaries. Until v2.0 the t-test pipeline used the keys
``'g2>g1'`` / ``'g1>g2'`` while the GLM pipeline used ``'positive'`` /
``'negative'``. From v2.0 onwards both pipelines use the canonical keys
``'positive'`` / ``'negative'``; the legacy keys remain accessible via this
shim and emit a :class:`DeprecationWarning` on access.

Removal target: v2.1.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, Iterator

LEGACY_TO_CANONICAL: Dict[str, str] = {
    "g2>g1": "positive",
    "g1>g2": "negative",
}
CANONICAL_KEYS = ("positive", "negative")


class TailResult(dict):
    """A two-key dict (``'positive'`` / ``'negative'``) with legacy aliases.

    Reading the legacy keys ``'g2>g1'`` / ``'g1>g2'`` works but emits a
    :class:`DeprecationWarning`. The legacy keys are *not* materialised in
    iteration order or :meth:`keys`, so consumers that iterate the dict
    see only the canonical keys.
    """

    def __getitem__(self, key: str) -> Any:
        if key in LEGACY_TO_CANONICAL:
            canonical = LEGACY_TO_CANONICAL[key]
            warnings.warn(
                f"Dict key {key!r} is deprecated; use {canonical!r} instead. "
                "Legacy keys will be removed in conninfpy v2.1.",
                DeprecationWarning,
                stacklevel=2,
            )
            return super().__getitem__(canonical)
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        if key in LEGACY_TO_CANONICAL:
            return self[key]  # triggers warning + returns canonical
        return super().get(key, default)

    def __contains__(self, key: object) -> bool:
        if super().__contains__(key):
            return True
        if isinstance(key, str) and key in LEGACY_TO_CANONICAL:
            return super().__contains__(LEGACY_TO_CANONICAL[key])
        return False


def make_tail_result(positive: Any, negative: Any) -> TailResult:
    """Build a :class:`TailResult` with canonical keys."""
    return TailResult([("positive", positive), ("negative", negative)])


def normalize_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of ``d`` with any legacy keys remapped to canonical.

    Used by enhancement wrappers and acceleration helpers so they accept
    either key naming during the deprecation cycle. Does *not* warn — the
    warning fires when the user reads the legacy key from a result dict,
    not when an internal helper sees it.
    """
    out: Dict[str, Any] = {}
    for k, v in d.items():
        out[LEGACY_TO_CANONICAL.get(k, k)] = v
    return out


def iter_tails(d: Dict[str, Any]) -> Iterator[tuple[str, Any]]:
    """Iterate over canonical (key, value) pairs after normalisation."""
    yield from normalize_keys(d).items()


__all__ = [
    "TailResult",
    "make_tail_result",
    "normalize_keys",
    "iter_tails",
    "LEGACY_TO_CANONICAL",
    "CANONICAL_KEYS",
]
