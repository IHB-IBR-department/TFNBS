"""Tests for InferenceResult / OmnibusInferenceResult export helpers."""
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from conninfpy import (
    AtlasInfo,
    InferenceResult,
    OmnibusInferenceResult,
)


def _make_tailed_result(N: int = 6) -> InferenceResult:
    """Construct a small InferenceResult with non-trivial stat maps.

    Plant two signals: a strong positive edge (0, 1) and a moderate
    negative edge (2, 3). All other edges are non-significant.
    """
    rng = np.random.default_rng(0)
    p_pos = rng.uniform(0.1, 0.9, size=(N, N))
    p_neg = rng.uniform(0.1, 0.9, size=(N, N))
    # Symmetric
    p_pos = (p_pos + p_pos.T) / 2
    p_neg = (p_neg + p_neg.T) / 2
    np.fill_diagonal(p_pos, 1.0)
    np.fill_diagonal(p_neg, 1.0)

    # Plant signals
    p_pos[0, 1] = p_pos[1, 0] = 0.001
    p_neg[2, 3] = p_neg[3, 2] = 0.002

    # Stat maps: one-tail clipped non-negative; signed = pos - neg
    stat_pos = np.zeros((N, N))
    stat_neg = np.zeros((N, N))
    stat_pos[0, 1] = stat_pos[1, 0] = 4.2     # positive effect
    stat_neg[2, 3] = stat_neg[3, 2] = 3.7     # negative effect

    return InferenceResult(
        p_pos,
        p_neg,
        method="tfnbs",
        n_permutations=100,
        stat_positive=stat_pos,
        stat_negative=stat_neg,
        stat_type="tstat",
    )


def _make_omnibus_result(N: int = 6) -> OmnibusInferenceResult:
    rng = np.random.default_rng(1)
    p = rng.uniform(0.1, 0.9, size=(N, N))
    p = (p + p.T) / 2
    np.fill_diagonal(p, 1.0)
    p[0, 1] = p[1, 0] = 0.001
    p[2, 4] = p[4, 2] = 0.01

    F = np.zeros((N, N))
    F[0, 1] = F[1, 0] = 12.5
    F[2, 4] = F[4, 2] = 7.1

    return OmnibusInferenceResult(
        p,
        method="tfnbs",
        n_permutations=100,
        stat_omnibus=F,
        stat_type="fstat",
    )


def _toy_atlas(N: int) -> AtlasInfo:
    networks = ["A"] * (N // 2) + ["B"] * (N - N // 2)
    hemisphere = ["L"] * (N // 2) + ["R"] * (N - N // 2)
    coords = np.column_stack([
        np.arange(N, dtype=float), np.zeros(N), np.zeros(N)
    ])
    return AtlasInfo(
        labels=[f"roi{i}" for i in range(N)],
        networks=networks,
        hemisphere=hemisphere,
        coords=coords,
        source="unit-test-toy",
    )


class TestTailedSignificantEdges(unittest.TestCase):

    def test_basic_no_atlas(self):
        res = _make_tailed_result()
        df = res.significant_edges(alpha=0.05)
        self.assertIsInstance(df, pd.DataFrame)
        # Two planted significant edges (0-1 positive, 2-3 negative)
        self.assertEqual(len(df), 2)
        expected_cols = {
            "edge_id", "roi_i", "roi_j", "t_signed",
            "p_positive", "p_negative", "p_min", "tail",
        }
        self.assertEqual(set(df.columns), expected_cols)
        # tail labels populated
        tails = set(df["tail"].tolist())
        self.assertEqual(tails, {"positive", "negative"})

    def test_with_atlas_columns_and_network_pair(self):
        res = _make_tailed_result(N=6)
        atlas = _toy_atlas(6)
        df = res.significant_edges(atlas, alpha=0.05)
        for col in (
            "roi_i_name", "roi_j_name",
            "roi_i_network", "roi_j_network",
            "network_pair", "hemisphere_i", "hemisphere_j",
        ):
            self.assertIn(col, df.columns)
        # ROIs 0,1 in network A; ROIs 2,3 in network A as well (toy split).
        # network_pair should be canonical "A—A" / "B—B" / "A—B"
        for val in df["network_pair"]:
            self.assertIn(val, {"A—A", "A—B", "B—B"})

    def test_t_signed_recovers_from_stat_maps(self):
        res = _make_tailed_result()
        df = res.significant_edges(alpha=0.05, sort="effect_size")
        # Strongest by |t_signed| should be the positive 0-1 edge (4.2)
        row0 = df.iloc[0]
        self.assertEqual(int(row0["roi_i"]), 0)
        self.assertEqual(int(row0["roi_j"]), 1)
        self.assertAlmostEqual(row0["t_signed"], 4.2)
        # Second strongest is the negative 2-3 edge (|−3.7|)
        row1 = df.iloc[1]
        self.assertEqual(int(row1["roi_i"]), 2)
        self.assertEqual(int(row1["roi_j"]), 3)
        self.assertAlmostEqual(row1["t_signed"], -3.7)

    def test_tail_filter_positive(self):
        res = _make_tailed_result()
        df = res.significant_edges(alpha=0.05, tail="positive")
        self.assertEqual(len(df), 1)
        self.assertEqual(int(df.iloc[0]["roi_i"]), 0)
        self.assertEqual(df.iloc[0]["tail"], "positive")

    def test_tail_filter_negative(self):
        res = _make_tailed_result()
        df = res.significant_edges(alpha=0.05, tail="negative")
        self.assertEqual(len(df), 1)
        self.assertEqual(int(df.iloc[0]["roi_i"]), 2)
        self.assertEqual(df.iloc[0]["tail"], "negative")

    def test_include_nonsig(self):
        res = _make_tailed_result(N=6)
        df = res.significant_edges(alpha=0.05, include_nonsig=True)
        # Upper triangle of 6×6 has 15 edges
        self.assertEqual(len(df), 15)
        # Non-significant rows have an empty tail label
        sig = (df["p_min"] <= 0.05).sum()
        self.assertEqual(sig, 2)

    def test_top_k(self):
        res = _make_tailed_result(N=6)
        df = res.significant_edges(alpha=1.0, top_k=3, include_nonsig=True)
        self.assertEqual(len(df), 3)

    def test_sort_by_p(self):
        res = _make_tailed_result()
        df = res.significant_edges(alpha=0.05, sort="p")
        # First row should be the smallest min(p)
        self.assertLessEqual(df.iloc[0]["p_min"], df.iloc[1]["p_min"])

    def test_sort_network_pair_requires_atlas(self):
        res = _make_tailed_result()
        with self.assertRaises(ValueError):
            res.significant_edges(sort="network_pair")

    def test_sort_network_pair_with_atlas(self):
        res = _make_tailed_result(N=6)
        atlas = _toy_atlas(6)
        df = res.significant_edges(atlas, alpha=0.05, sort="network_pair")
        # Within each network_pair group, p_min should be ascending
        for _, group in df.groupby("network_pair", sort=False):
            self.assertTrue(group["p_min"].is_monotonic_increasing)

    def test_missing_stat_maps_raises(self):
        # Construct a result without stat maps (legacy pickled v2.0 shape)
        N = 4
        p = np.full((N, N), 0.001)
        res = InferenceResult(p, p, method="tfnbs", n_permutations=100)
        # stat_positive and stat_negative default to None → stat_signed is None
        with self.assertRaises(ValueError):
            res.significant_edges()

    def test_invalid_tail_raises(self):
        res = _make_tailed_result()
        with self.assertRaises(ValueError):
            res.significant_edges(tail="bogus")

    def test_invalid_sort_raises(self):
        res = _make_tailed_result()
        with self.assertRaises(ValueError):
            res.significant_edges(sort="bogus")

    def test_atlas_size_mismatch_raises(self):
        res = _make_tailed_result(N=6)
        atlas = _toy_atlas(5)  # off-by-one
        with self.assertRaises(ValueError):
            res.significant_edges(atlas)


class TestOmnibusSignificantEdges(unittest.TestCase):

    def test_columns_no_atlas(self):
        res = _make_omnibus_result()
        df = res.significant_edges(alpha=0.05)
        self.assertEqual(
            set(df.columns),
            {"edge_id", "roi_i", "roi_j", "F", "p_omnibus"},
        )
        self.assertNotIn("tail", df.columns)
        # 2 planted significant edges
        self.assertEqual(len(df), 2)

    def test_columns_with_atlas(self):
        res = _make_omnibus_result(N=6)
        atlas = _toy_atlas(6)
        df = res.significant_edges(atlas, alpha=0.05)
        for col in (
            "roi_i_name", "roi_j_name", "roi_i_network", "roi_j_network",
            "network_pair", "F", "p_omnibus",
        ):
            self.assertIn(col, df.columns)

    def test_sort_effect_size_ranks_by_F(self):
        res = _make_omnibus_result()
        df = res.significant_edges(alpha=0.05, sort="effect_size")
        # Largest F first
        self.assertGreaterEqual(df.iloc[0]["F"], df.iloc[1]["F"])

    def test_missing_stat_omnibus_raises(self):
        N = 4
        p = np.full((N, N), 0.001)
        res = OmnibusInferenceResult(p, method="tfnbs", n_permutations=100)
        with self.assertRaises(ValueError):
            res.significant_edges()


class TestToCsvRoundTrip(unittest.TestCase):

    def test_tailed_to_csv(self):
        res = _make_tailed_result(N=6)
        atlas = _toy_atlas(6)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "edges.csv"
            res.to_csv(out, atlas=atlas, alpha=0.05)
            self.assertTrue(out.exists())
            df = pd.read_csv(out)
            self.assertGreater(len(df), 0)
            self.assertIn("network_pair", df.columns)
            self.assertIn("t_signed", df.columns)

    def test_omnibus_to_csv(self):
        res = _make_omnibus_result(N=6)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "omnibus.csv"
            res.to_csv(out, alpha=0.05)
            self.assertTrue(out.exists())
            df = pd.read_csv(out)
            self.assertGreater(len(df), 0)
            self.assertIn("F", df.columns)
            self.assertIn("p_omnibus", df.columns)


class TestAnalyzeResultDelegation(unittest.TestCase):

    def test_significant_edges_delegates(self):
        from conninfpy._analyze import AnalyzeResult

        res = _make_tailed_result(N=6)
        ar = AnalyzeResult(inference=res)
        df = ar.significant_edges(alpha=0.05)
        self.assertEqual(len(df), 2)

    def test_to_csv_delegates(self):
        from conninfpy._analyze import AnalyzeResult

        res = _make_tailed_result(N=6)
        atlas = _toy_atlas(6)
        ar = AnalyzeResult(inference=res)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "edges.csv"
            ar.to_csv(out, atlas=atlas)
            self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()
