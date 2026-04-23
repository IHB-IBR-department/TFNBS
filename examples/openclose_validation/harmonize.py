"""
Cohort-as-site ComBat harmonization for the pooled IHB + China Open-Close data.

Each cohort comes from a different scanner (IHB: Philips 3T, TR=2.5s; China:
Siemens 3T, TR=2.0s) with different acquisition windows (120 vs 240 TRs). If
you want to pool the two cohorts — for §3.7 direct cross-cohort comparison or
the §3.8 pooled-cohort ML sanity check — scanner-aligned variance confounds the
Open-vs-Close contrast. ComBat with ``cohort`` as the batch variable removes
that nuisance while preserving the task effect.

Uses :func:`conninfpy.combat_harmonize` — the in-package parametric empirical-
Bayes ComBat implementation, no external dependency.

Model
-----
We pool ``(n_ihb + n_china) × 2`` subject-condition samples (stacked open on top
of close), tag ``cohort ∈ {IHB, China}`` as the batch, and preserve the
open/close indicator. The post-ComBat matrices carry the task contrast but
have the cohort-specific location/scale removed.

Output
------
An ``.npz`` file with:
  ``Y_ihb_open_harm``, ``Y_ihb_close_harm``  (84, 182, 182)
  ``Y_china_open_harm``, ``Y_china_close_harm``  (48, 182, 182)
  diagnostics dict (between-cohort variance before / after, per-cohort n)

Downstream analyses (paired TFNBS per cohort, IHB↔China agreement, ML transfer)
can read this file or call :func:`harmonize_openclose` directly to get the
same matrices in memory.

Command-line
------------
    python -m examples.openclose_validation.harmonize
    python -m examples.openclose_validation.harmonize --no-drop-rois \\
        --output examples/openclose_validation/results/openclose_harmonized.npz
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from conninfpy import combat_harmonize, fisher_r_to_z
from examples.openclose_validation.openclose_loader import OpenCloseDataset


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "openclose_harmonized.npz"


@dataclass
class HarmonizedOpenClose:
    """Harmonized connectivity for IHB + China, per-condition split."""

    Y_ihb_open: np.ndarray
    Y_ihb_close: np.ndarray
    Y_china_open: np.ndarray
    Y_china_close: np.ndarray
    diagnostics: Dict[str, object]


def _ts_to_fisher_z(ts: np.ndarray) -> np.ndarray:
    """(n_subjects, T, N) → (n_subjects, N, N) Fisher-z correlation matrices."""
    n, _, N = ts.shape
    out = np.empty((n, N, N), dtype=np.float64)
    for s in range(n):
        c = np.corrcoef(ts[s].T)
        np.fill_diagonal(c, 0.0)
        out[s] = np.clip(c, -0.9999, 0.9999)
    return fisher_r_to_z(out)


def _stack_for_combat(
    Y_ihb_open: np.ndarray, Y_ihb_close: np.ndarray,
    Y_china_open: np.ndarray, Y_china_close: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack (n_subj_cond × 4 blocks) and build cohort + condition vectors."""
    blocks = [Y_ihb_open, Y_ihb_close, Y_china_open, Y_china_close]
    cohorts = ["IHB", "IHB", "China", "China"]
    conditions = [1.0, 0.0, 1.0, 0.0]  # 1 = open, 0 = close
    Y_stack = np.concatenate(blocks, axis=0)
    cohort = np.concatenate(
        [np.array([c] * b.shape[0]) for c, b in zip(cohorts, blocks)]
    )
    cond = np.concatenate(
        [np.full(b.shape[0], v) for v, b in zip(conditions, blocks)]
    )
    return Y_stack, cohort, cond


def _unstack(
    Y_harm: np.ndarray,
    shapes: Tuple[int, int, int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reverse of :func:`_stack_for_combat`."""
    n_io, n_ic, n_co, n_cc = shapes
    bounds = np.cumsum([0, n_io, n_ic, n_co, n_cc])
    return (
        Y_harm[bounds[0]:bounds[1]],
        Y_harm[bounds[1]:bounds[2]],
        Y_harm[bounds[2]:bounds[3]],
        Y_harm[bounds[3]:bounds[4]],
    )


def harmonize_openclose(
    ihb: OpenCloseDataset,
    china: OpenCloseDataset,
) -> HarmonizedOpenClose:
    """
    Cohort-as-site ComBat on pooled IHB + China Fisher-z correlations.

    The open/close task indicator is preserved (its variance in Y survives
    harmonization). Cohort-aligned (scanner/population) variance orthogonal
    to the task is removed.

    Returns per-condition harmonized matrices at the original shapes:
    ``(84, 182, 182)`` for IHB, ``(48, 182, 182)`` for China (when
    ``drop_missing_rois=True`` was used in the loader).
    """
    if ihb.n_rois != china.n_rois:
        raise ValueError(
            f"IHB has {ihb.n_rois} ROIs, China has {china.n_rois} — the "
            f"coverage mask (drop_missing_rois=True) must be applied to both."
        )

    # Per-subject Fisher-z correlation matrices per condition
    Y_ihb_open = _ts_to_fisher_z(ihb.open_ts)
    Y_ihb_close = _ts_to_fisher_z(ihb.close_ts)
    Y_china_open = _ts_to_fisher_z(china.open_ts)
    Y_china_close = _ts_to_fisher_z(china.close_ts)

    Y_stack, cohort, cond = _stack_for_combat(
        Y_ihb_open, Y_ihb_close, Y_china_open, Y_china_close,
    )
    shapes = (
        Y_ihb_open.shape[0], Y_ihb_close.shape[0],
        Y_china_open.shape[0], Y_china_close.shape[0],
    )

    result = combat_harmonize(
        Y_stack,
        sites=cohort,
        preserve=cond,
        return_diagnostics=True,
    )

    Y_harm = result.Y_adjusted
    Y_ihb_open_h, Y_ihb_close_h, Y_china_open_h, Y_china_close_h = _unstack(
        Y_harm, shapes,
    )

    diag = dict(result.diagnostics)
    diag["cohort_labels"] = ["IHB", "China"]
    diag["per_cohort_n_open"] = [Y_ihb_open.shape[0], Y_china_open.shape[0]]
    diag["per_cohort_n_close"] = [Y_ihb_close.shape[0], Y_china_close.shape[0]]

    return HarmonizedOpenClose(
        Y_ihb_open=Y_ihb_open_h,
        Y_ihb_close=Y_ihb_close_h,
        Y_china_open=Y_china_open_h,
        Y_china_close=Y_china_close_h,
        diagnostics=diag,
    )


def _save(result: HarmonizedOpenClose, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        Y_ihb_open_harm=result.Y_ihb_open,
        Y_ihb_close_harm=result.Y_ihb_close,
        Y_china_open_harm=result.Y_china_open,
        Y_china_close_harm=result.Y_china_close,
        between_cohort_variance_ratio_before=np.float64(
            result.diagnostics["between_site_variance_ratio_before"]
        ),
        between_cohort_variance_ratio_after=np.float64(
            result.diagnostics["between_site_variance_ratio_after"]
        ),
        ratio_reduction=np.float64(result.diagnostics["ratio_reduction"]),
        cohort_labels=np.array(result.diagnostics["cohort_labels"]),
        per_cohort_n_open=np.array(result.diagnostics["per_cohort_n_open"]),
        per_cohort_n_close=np.array(result.diagnostics["per_cohort_n_close"]),
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Cohort-as-site ComBat on pooled IHB + China Open-Close data.",
    )
    ap.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Output .npz path (created with its parents).",
    )
    ap.add_argument(
        "--no-drop-rois", action="store_true",
        help="Keep all 200 Schaefer ROIs (default: drop 18 low-coverage → 182).",
    )
    args = ap.parse_args()

    print("Loading IHB ...")
    ihb = OpenCloseDataset.load("ihb", drop_missing_rois=not args.no_drop_rois)
    print(f"  shape: open={ihb.open_ts.shape}, close={ihb.close_ts.shape}")

    print("Loading China ...")
    china = OpenCloseDataset.load("china", drop_missing_rois=not args.no_drop_rois)
    print(f"  shape: open={china.open_ts.shape}, close={china.close_ts.shape}")

    print("Running cohort-as-site ComBat (preserving open/close) ...")
    result = harmonize_openclose(ihb, china)

    ratio_before = result.diagnostics["between_site_variance_ratio_before"]
    ratio_after = result.diagnostics["between_site_variance_ratio_after"]
    reduction = result.diagnostics["ratio_reduction"] * 100
    print(f"Between-cohort variance ratio: before={ratio_before:.4f}  "
          f"after={ratio_after:.4f}  (reduction: {reduction:.1f}%)")

    print(f"Saving to {args.output}")
    _save(result, args.output)
    print("done.")


if __name__ == "__main__":
    main()
