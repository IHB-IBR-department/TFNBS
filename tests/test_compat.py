"""Tests for the v1→v2 dict-key deprecation shim (`conninfpy._compat`)."""
from __future__ import annotations

import unittest
import warnings

import numpy as np

from conninfpy._compat import (
    LEGACY_TO_CANONICAL,
    TailResult,
    make_tail_result,
    normalize_keys,
)


class TestTailResult(unittest.TestCase):
    def setUp(self):
        self.pos = np.ones((3, 3))
        self.neg = np.zeros((3, 3))
        self.r = make_tail_result(self.pos, self.neg)

    def test_canonical_keys_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning fails the test
            np.testing.assert_array_equal(self.r["positive"], self.pos)
            np.testing.assert_array_equal(self.r["negative"], self.neg)

    def test_legacy_keys_emit_deprecation_warning(self):
        for legacy_key, canonical in LEGACY_TO_CANONICAL.items():
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                value = self.r[legacy_key]
                self.assertEqual(len(w), 1)
                self.assertTrue(issubclass(w[0].category, DeprecationWarning))
                self.assertIn(canonical, str(w[0].message))
            np.testing.assert_array_equal(
                value,
                self.r[canonical] if canonical == "positive" else self.r["negative"],
            )

    def test_iteration_yields_only_canonical_keys(self):
        # `keys()` must not list legacy aliases — they exist only via __getitem__
        self.assertEqual(set(self.r.keys()), {"positive", "negative"})
        self.assertEqual(set(iter(self.r)), {"positive", "negative"})

    def test_contains_works_for_both_namings(self):
        self.assertIn("positive", self.r)
        self.assertIn("g2>g1", self.r)
        self.assertNotIn("nonexistent", self.r)

    def test_get_with_legacy_key_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.r.get("g2>g1")
            self.assertTrue(any(
                issubclass(warning.category, DeprecationWarning) for warning in w
            ))


class TestNormalizeKeys(unittest.TestCase):
    def test_remap_legacy(self):
        d = {"g2>g1": 1, "g1>g2": 2}
        out = normalize_keys(d)
        self.assertEqual(out, {"positive": 1, "negative": 2})

    def test_passthrough_canonical(self):
        d = {"positive": 1, "negative": 2}
        out = normalize_keys(d)
        self.assertEqual(out, d)

    def test_mixed_keys(self):
        d = {"g2>g1": 1, "negative": 2}  # last-write-wins is undefined, just check no crash
        out = normalize_keys(d)
        self.assertIn("positive", out)
        self.assertIn("negative", out)

    def test_does_not_warn_internally(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            normalize_keys({"g2>g1": 1, "g1>g2": 2})


class TestPipelineReturnsTailResult(unittest.TestCase):
    """The public pipelines must return TailResult so legacy keys keep working."""

    def test_compute_t_stat_returns_tail_result(self):
        from conninfpy import compute_t_stat
        rng = np.random.default_rng(0)
        g1 = rng.standard_normal((10, 5, 5))
        g2 = rng.standard_normal((10, 5, 5))
        result = compute_t_stat(g1, g2, test_type="two-sample")
        self.assertIsInstance(result, TailResult)
        # legacy key works (with warning)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self.assertTrue(np.array_equal(result["g2>g1"], result["positive"]))


if __name__ == "__main__":
    unittest.main()
