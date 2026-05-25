"""
State-vs-empirical-null specificity per method (China cohort).

The China cohort has two close-eyes runs collected with the open-eyes
condition counterbalanced across subjects. Under this design the
close-run0 vs close-run1 paired contrast is a matched-design empirical
null for the open-vs-close state contrast: same paired structure, no
state manipulation, time-ordered drift averaged out by counterbalancing.

Per method (TFNBS, NI-TFNBS, FBC-TFNBS, NBS τ=2.0, NBS τ=3.0, BH-FDR, t-stat)
we compare n_sig at α=0.05, state vs empirical-null, grouped bars.
Lower retest bar ⇒ better FWER specificity.

Inputs : results/ml/pmap_china_{method}.npz           (state contrast)
         results/ml/pmap_china_retest_{method}.npz    (empirical null)
Output : results/plots/plot5_retest_specificity.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ML_DIR = HERE / "results" / "ml"
PLOTS = HERE / "results" / "plots"

ALPHA = 0.05

METHODS = ["tstat", "tfnbs", "ni_tfnbs", "fbc_tfnbs", "nbs_20", "nbs_30", "bh_fdr"]
PRETTY = {
    "tstat":    "t-stat",
    "tfnbs":    "TFNBS",
    "ni_tfnbs": "NI-TFNBS",
    "fbc_tfnbs":"FBC-TFNBS",
    "nbs_20":   "NBS τ=2.0",
    "nbs_30":   "NBS τ=3.0",
    "bh_fdr":   "BH-FDR",
}


def _load_pmap(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (p_min over tails, n) flattened over upper triangle. None if missing."""
    if not path.exists():
        return None
    d = np.load(path)
    arrs = [d[k] for k in d.files]
    p_min = arrs[0].copy()
    for a in arrs[1:]:
        p_min = np.minimum(p_min, a)
    iu = np.triu_indices(p_min.shape[0], k=1)
    return p_min[iu]


def _n_sig(path: Path, alpha: float) -> int:
    """Count edges significant in EITHER tail at α."""
    p = _load_pmap(path)
    return 0 if p is None else int((p < alpha).sum())


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)

    # ---- counts for left panel -----------------------------------
    state_n = []
    null_n = []
    for m in METHODS:
        state_n.append(_n_sig(ML_DIR / f"pmap_china_{m}.npz", ALPHA))
        null_n.append(_n_sig(ML_DIR / f"pmap_china_retest_{m}.npz", ALPHA))

    n_edges_total = 182 * 181 // 2  # for axis annotation
    null_floor = ALPHA * n_edges_total  # expected n_sig under perfect uniform null at α=0.05

    fig, ax_bar = plt.subplots(figsize=(10, 7), constrained_layout=True)

    # ---- Panel A — bar chart -------------------------------------
    x = np.arange(len(METHODS))
    w = 0.4
    ax_bar.bar(x - w/2, state_n, w, label="State (Open vs Close)",
               color="#2a8a4a", edgecolor="black")
    ax_bar.bar(x + w/2, null_n, w, label="Empirical null (Close vs Close)",
               color="#aa3030", edgecolor="black")
    ax_bar.axhline(null_floor, ls="--", lw=1, color="gray",
                   label=f"Uniform-null expected (α·E ≈ {null_floor:.0f})")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([PRETTY[m] for m in METHODS], rotation=30, ha="right")
    ax_bar.set_ylabel(f"# edges significant at α={ALPHA} (pos ∪ neg)")
    ax_bar.set_title(
        "Edge counts: state vs empirical null (China paired)\n"
        "Specificity ratio = state / null  ↑ better"
    )
    ax_bar.legend(loc="upper left")
    for i, (s, n_) in enumerate(zip(state_n, null_n)):
        ratio = s / max(1, n_)
        ax_bar.text(i, max(s, n_) * 1.02, f"{ratio:.1f}×",
                    ha="center", fontsize=9, color="black")
    ax_bar.set_ylim(0, max(max(state_n), max(null_n)) * 1.18)

    fig.suptitle(
        "Specificity under Matched-Design Empirical Null (TF-NBS: E=0.4, H=3.0)\n"
        "(China close-run0 vs close-run1, counterbalanced)",
        fontsize=16, fontweight='bold'
    )
    out = PLOTS / "plot5_retest_specificity.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved: {out}")

    # Headline table
    print("\nSpecificity headline (state / empirical-null ratio):")
    print(f"  {'method':10s}  {'state':>8s}  {'null':>8s}  {'ratio':>8s}")
    for m, s, n_ in zip(METHODS, state_n, null_n):
        ratio = s / max(1, n_)
        print(f"  {PRETTY[m]:10s}  {s:8d}  {n_:8d}  {ratio:8.2f}")


if __name__ == "__main__":
    main()
