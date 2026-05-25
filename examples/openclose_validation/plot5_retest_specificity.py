"""
Plot 6 — State-vs-empirical-null specificity per method (China cohort).

The China cohort has two close-eyes runs collected with the open-eyes
condition counterbalanced across subjects. Under this design the
close-run0 vs close-run1 paired contrast is a matched-design empirical
null for the open-vs-close state contrast: same paired structure, no
state manipulation, time-ordered drift averaged out by counterbalancing.

Per method (TFNBS, NI-TFNBS, cNBS, NBS τ=2.0, NBS τ=3.0, BH-FDR, t-stat)
we compare:

  - Left panel  — n_sig at α=0.05, state vs empirical-null, grouped bars.
                  Lower retest bar ⇒ better FWER specificity.
  - Right panel — QQ plot of empirical-null −log10 p vs uniform.
                  Closer to the identity line ⇒ closer to nominal calibration.

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

METHODS = ["tstat", "tfnbs", "ni_tfnbs", "cnbs", "nbs_20", "nbs_30", "bh_fdr"]
PRETTY = {
    "tstat":    "t-stat",
    "tfnbs":    "TFNBS",
    "ni_tfnbs": "NI-TFNBS",
    "cnbs":     "cNBS",
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

    fig = plt.figure(figsize=(17, 7))

    # ---- Panel A — bar chart -------------------------------------
    ax_bar = plt.subplot(1, 2, 1)
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
        "Panel A — Edge counts: state vs empirical null (China paired)\n"
        "Specificity ratio = state / null  ↑ better"
    )
    ax_bar.legend(loc="upper left")
    for i, (s, n_) in enumerate(zip(state_n, null_n)):
        ratio = s / max(1, n_)
        ax_bar.text(i, max(s, n_) * 1.02, f"{ratio:.1f}×",
                    ha="center", fontsize=9, color="black")
    ax_bar.set_ylim(0, max(max(state_n), max(null_n)) * 1.18)

    # ---- Panel B — QQ plot of null p-values ----------------------
    ax_qq = plt.subplot(1, 2, 2)
    cmap = plt.cm.tab10
    for k, m in enumerate(METHODS):
        p = _load_pmap(ML_DIR / f"pmap_china_retest_{m}.npz")
        if p is None:
            continue
        p_sorted = np.sort(p)
        n = p_sorted.size
        # Empirical CDF expectation under uniform: i/n
        expected = (np.arange(1, n + 1)) / (n + 1)
        # Plot -log10 expected vs -log10 observed (max 5 to keep axis readable)
        ax_qq.plot(
            -np.log10(expected),
            -np.log10(np.maximum(p_sorted, 1e-300)),
            label=PRETTY[m], color=cmap(k), lw=1.4,
        )
    lim = 4.0
    ax_qq.plot([0, lim], [0, lim], ls="--", color="black", lw=1, label="uniform null (y=x)")
    ax_qq.set_xlim(0, lim)
    ax_qq.set_ylim(0, lim)
    ax_qq.set_xlabel("Expected −log10(p) under uniform null")
    ax_qq.set_ylabel("Observed −log10(p) on empirical null")
    ax_qq.set_title("Panel B — Empirical-null QQ per method\n"
                    "Curves above y=x ⇒ method over-calls on the null")
    ax_qq.legend(loc="upper left", fontsize=9)
    ax_qq.grid(alpha=0.3)

    fig.suptitle(
        "Plot 6 — Specificity under Matched-Design Empirical Null (TF-NBS: E=0.4, H=3.0) "
        "(China close-run0 vs close-run1, counterbalanced)",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
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
