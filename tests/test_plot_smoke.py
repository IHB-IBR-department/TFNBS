"""Smoke tests for the v2.1 plotting helpers.

These tests verify that each helper composes a figure without raising,
that the returned types match what the docstrings promise, and that
optional knobs (atlas vs no atlas, F-stat path, missing stat maps)
behave gracefully. Pixel-level correctness is not checked — figure
inspection is handled visually in the demo notebook.
"""
import unittest

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    HAS_MPL = True
except ImportError:  # pragma: no cover
    HAS_MPL = False

if HAS_MPL:
    from conninfpy import AtlasInfo, InferenceResult, OmnibusInferenceResult
    from conninfpy.plot import (
        plot_effect_matrix,
        plot_network_summary,
        summary_figure,
    )


def _tailed_result(N: int = 8) -> "InferenceResult":
    rng = np.random.default_rng(0)
    p_pos = rng.uniform(0.1, 0.9, size=(N, N))
    p_neg = rng.uniform(0.1, 0.9, size=(N, N))
    p_pos = (p_pos + p_pos.T) / 2
    p_neg = (p_neg + p_neg.T) / 2
    np.fill_diagonal(p_pos, 1.0)
    np.fill_diagonal(p_neg, 1.0)
    p_pos[0, 1] = p_pos[1, 0] = 0.001
    p_neg[2, 3] = p_neg[3, 2] = 0.005

    stat_pos = np.zeros((N, N))
    stat_neg = np.zeros((N, N))
    stat_pos[0, 1] = stat_pos[1, 0] = 4.5
    stat_neg[2, 3] = stat_neg[3, 2] = 3.7

    return InferenceResult(
        p_pos, p_neg,
        method="tfnbs", n_permutations=100,
        stat_positive=stat_pos, stat_negative=stat_neg,
        stat_type="tstat",
    )


def _omnibus_result(N: int = 8) -> "OmnibusInferenceResult":
    rng = np.random.default_rng(1)
    p = rng.uniform(0.1, 0.9, size=(N, N))
    p = (p + p.T) / 2
    np.fill_diagonal(p, 1.0)
    p[0, 1] = p[1, 0] = 0.001
    F = np.zeros((N, N))
    F[0, 1] = F[1, 0] = 12.0
    return OmnibusInferenceResult(
        p, method="tfnbs", n_permutations=100,
        stat_omnibus=F, stat_type="fstat",
    )


def _toy_atlas(N: int) -> "AtlasInfo":
    half = N // 2
    return AtlasInfo(
        labels=[f"roi{i}" for i in range(N)],
        networks=["A"] * half + ["B"] * (N - half),
        hemisphere=["L"] * half + ["R"] * (N - half),
        source="unit-test-toy",
    )


@unittest.skipUnless(HAS_MPL, "matplotlib not available")
class TestPlotEffectMatrix(unittest.TestCase):

    def setUp(self):
        plt.close("all")

    def test_returns_axes_no_atlas(self):
        res = _tailed_result()
        ax = plot_effect_matrix(res, alpha=0.05)
        self.assertIsInstance(ax, Axes)

    def test_returns_axes_with_atlas(self):
        res = _tailed_result(N=8)
        atlas = _toy_atlas(8)
        ax = plot_effect_matrix(res, atlas=atlas)
        self.assertIsInstance(ax, Axes)
        # An RdBu_r colormap was applied for the t-statistic.
        self.assertEqual(ax.images[0].get_cmap().name, "RdBu_r")

    def test_returns_axes_omnibus(self):
        res = _omnibus_result()
        ax = plot_effect_matrix(res)
        self.assertIsInstance(ax, Axes)
        # F-stat is unsigned → uses a sequential colormap.
        self.assertEqual(ax.images[0].get_cmap().name, "magma")

    def test_missing_stat_maps_raises(self):
        N = 4
        p = np.full((N, N), 0.001)
        bad = InferenceResult(p, p, method="tfnbs", n_permutations=10)
        with self.assertRaises(ValueError):
            plot_effect_matrix(bad)

    def test_atlas_size_mismatch_raises(self):
        res = _tailed_result(N=8)
        atlas = _toy_atlas(6)
        with self.assertRaises(ValueError):
            plot_effect_matrix(res, atlas=atlas)


@unittest.skipUnless(HAS_MPL, "matplotlib not available")
class TestPlotNetworkSummary(unittest.TestCase):

    def setUp(self):
        plt.close("all")

    def test_returns_axes_tailed(self):
        res = _tailed_result(N=8)
        atlas = _toy_atlas(8)
        ax = plot_network_summary(res, atlas=atlas, alpha=0.05)
        self.assertIsInstance(ax, Axes)
        # K=2 networks (A, B) → 2 ticks per axis
        self.assertEqual(len(ax.get_xticklabels()), 2)
        self.assertEqual(len(ax.get_yticklabels()), 2)

    def test_returns_axes_omnibus(self):
        res = _omnibus_result(N=8)
        atlas = _toy_atlas(8)
        ax = plot_network_summary(res, atlas=atlas, alpha=0.05)
        self.assertIsInstance(ax, Axes)

    def test_atlas_required(self):
        res = _tailed_result()
        with self.assertRaises(TypeError):
            # missing atlas (positional/keyword)
            plot_network_summary(res, alpha=0.05)


@unittest.skipUnless(HAS_MPL, "matplotlib not available")
class TestSummaryFigure(unittest.TestCase):

    def setUp(self):
        plt.close("all")

    def test_returns_figure_with_atlas(self):
        res = _tailed_result(N=8)
        atlas = _toy_atlas(8)
        fig = summary_figure(res, atlas=atlas, alpha=0.05, top_k=5)
        self.assertIsInstance(fig, Figure)
        # 4 panel axes + colorbar axes for the heatmap panels.
        self.assertGreaterEqual(len(fig.axes), 4)
        plt.close(fig)

    def test_returns_figure_without_atlas(self):
        res = _tailed_result(N=8)
        fig = summary_figure(res, alpha=0.05, top_k=3)
        self.assertIsInstance(fig, Figure)
        plt.close(fig)

    def test_returns_figure_omnibus(self):
        res = _omnibus_result(N=8)
        atlas = _toy_atlas(8)
        fig = summary_figure(res, atlas=atlas, alpha=0.05, top_k=3)
        self.assertIsInstance(fig, Figure)
        plt.close(fig)

    def test_handles_missing_stat_maps_gracefully(self):
        N = 8
        p = np.full((N, N), 0.5)
        np.fill_diagonal(p, 1.0)
        # No stat maps → effect matrix panel will raise; summary_figure
        # should propagate the ValueError so users notice rather than
        # silently producing a blank figure.
        bad = InferenceResult(p, p, method="tfnbs", n_permutations=10)
        with self.assertRaises(ValueError):
            summary_figure(bad)


if __name__ == "__main__":
    unittest.main()
