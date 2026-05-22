"""
§3.9 method-comparison audit for the ABIDE age → FC positive-control sweep.

Reads per-method p-maps from results/combat/age/methods/ and produces the
pairwise agreement panel described in ValidationPlan.md §2.5 + §3.9:

- Spearman rank correlation of -log10 p maps
- Jaccard at α=0.05 (with random-baseline and Fisher's exact p)
- Top-k concordance for k ∈ {10, 50, 100, 500}
- Block-mass Pearson on Yeo-7 partitions
- Per-method n_sig + network breakdown
- Runtime + file-size summary (method cost)

Outputs CSV + PNG figures under results/combat/age/methods/audit/.
"""

from __future__ import annotations

import os
import sys
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Shared agreement helpers (examples/_audit_utils.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _audit_utils import (  # noqa: E402
    upper_triangle as upper,
    jaccard,
    jaccard_random_baseline,
    fisher_exact_2x2,
    spearman_neglog10,
    topk_concordance as topk,
    block_mass,
    block_mass_correlation as block_mass_corr,
)


HERE = os.path.dirname(os.path.abspath(__file__))
METHODS_DIR = os.path.join(HERE, "results", "combat", "age", "methods")
AUDIT_DIR = os.path.join(METHODS_DIR, "audit")
PREPARED = os.path.join(HERE, "abide_prepared.npz")

# Ordered method display names (controls row/column order in the panels)
METHOD_ORDER = [
    "tstat",
    "bh_fdr",
    "nbs_2.0",
    "nbs_3.0",
    "tfnbs",
    "cnbs",
    "ni_tfnbs",
    "fbc_tfnbs",
]


def load_pmap(path: str) -> Dict[str, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files if k in ("positive", "negative")}


def agreement(pa_full, pb_full, labels, alpha=0.05):
    """§2.5 agreement bundle. Thin wrapper over the shared helper to
    preserve this script's historical output schema."""
    pa = upper(pa_full); pb = upper(pb_full)
    sa = pa < alpha; sb = pb < alpha
    ka, kb = int(sa.sum()), int(sb.sum())
    return {
        "jaccard": jaccard(sa, sb),
        "jaccard_random": jaccard_random_baseline(ka, kb, pa.size),
        "fisher_exact_p": fisher_exact_2x2(sa, sb),
        "spearman": spearman_neglog10(pa, pb),
        "top10": topk(pa, pb, 10),
        "top50": topk(pa, pb, 50),
        "top100": topk(pa, pb, 100),
        "top500": topk(pa, pb, 500),
        "block_corr": block_mass_corr(pa_full, pb_full, labels),
        "ka": ka, "kb": kb,
    }


def _build_matrix(pairwise_df, col, methods):
    M = pd.pivot_table(pairwise_df, index="a", columns="b", values=col,
                       aggfunc="first").reindex(index=methods, columns=methods)
    # The pairwise loop produced each pair once (a < b); mirror it.
    M_sym = M.copy()
    for i, a in enumerate(methods):
        for j, b in enumerate(methods):
            if pd.isna(M_sym.loc[a, b]):
                M_sym.loc[a, b] = M.loc[b, a]
        M_sym.loc[a, a] = 1.0 if col in ("spearman", "block_corr", "jaccard",
                                         "top10", "top50", "top100", "top500") else 0.0
    return M_sym


def plot_method_panel(df_long, methods, net_labels, outpath):
    metrics = [
        ("spearman", "Spearman(−log10 p)"),
        ("block_corr", "Block-mass Pearson"),
        ("top100", "Top-100 concordance"),
        ("jaccard", "Jaccard @ α=0.05"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4.0))
    for ax, (col, title) in zip(axes, metrics):
        M = _build_matrix(df_long, col, methods)
        im = ax.imshow(M.values, cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods, fontsize=8)
        ax.set_title(title, fontsize=10)
        # annotate cells
        for i in range(len(methods)):
            for j in range(len(methods)):
                v = M.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=6, color="white" if v < 0.5 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"§3.9 method comparison — ABIDE age→FC (averaged across tails)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


def plot_network_profile(per_method_df, outpath, network_order, net_labels):
    """Stacked bar chart of surviving-edge network distribution per method."""
    # Per-method, per-block-pair count of significant edges, positive tail
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for tail, ax in zip(["positive", "negative"], axes):
        sub = per_method_df[per_method_df["tail"] == tail]
        if sub.empty:
            continue
        methods = sub["method"].unique()
        # network_order contains the block names in block-id order (0..K-1)
        block_pair_names = []
        K = len(network_order)
        for i in range(K):
            for j in range(i, K):
                name = network_order[i] if i == j else f"{network_order[i]}×{network_order[j]}"
                block_pair_names.append(name)

        # Get counts per method for each block pair
        all_counts = {m: np.zeros(len(block_pair_names)) for m in methods}
        for _, row in sub.iterrows():
            m = row["method"]; idx = row["block_pair_idx"]
            if idx is not None and 0 <= idx < len(block_pair_names):
                all_counts[m][idx] += row["n_edges"]

        # Keep only top 8 block pairs by total across methods
        totals = np.sum([all_counts[m] for m in methods], axis=0)
        top_idx = np.argsort(-totals)[:8]
        labels = [block_pair_names[i] for i in top_idx]
        bottoms = np.zeros(len(methods))
        for bp_i in top_idx:
            vals = np.array([all_counts[m][bp_i] for m in methods])
            ax.bar(range(len(methods)), vals, bottom=bottoms, label=block_pair_names[bp_i])
            bottoms += vals
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=60, ha="right", fontsize=8)
        ax.set_title(f"{tail} tail: surviving-edge network distribution")
        ax.set_ylabel("n_sig edges")
        ax.legend(fontsize=6, loc="upper right", ncol=1)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


def summarize_networks(p_full, alpha, net_labels, network_order):
    """Return {(bi, bj): n_sig_edges} for the upper-triangle at alpha."""
    N = p_full.shape[0]
    iu = np.triu_indices(N, k=1)
    sig = p_full[iu] < alpha
    counts = {}
    K = int(net_labels.max() + 1)
    # build a deterministic block_pair_idx mapping
    block_pair_idx = {}
    k = 0
    for i in range(K):
        for j in range(i, K):
            block_pair_idx[(i, j)] = k
            k += 1
    for ii_e, jj_e in zip(iu[0][sig], iu[1][sig]):
        bi, bj = sorted([net_labels[ii_e], net_labels[jj_e]])
        counts[(bi, bj)] = counts.get((bi, bj), 0) + 1
    return counts, block_pair_idx


def main():
    os.makedirs(AUDIT_DIR, exist_ok=True)
    prep = np.load(PREPARED, allow_pickle=True)
    net_labels = prep["net_labels"]
    network_order = list(prep["network_order"])

    # Discover method files
    files = {}
    for path in sorted(glob(os.path.join(METHODS_DIR, "age_*.npz"))):
        key = os.path.basename(path).removesuffix(".npz").removeprefix("age_")
        files[key] = path
    print(f"Found {len(files)} method files: {list(files.keys())}")

    # Per-method summary: n_sig per tail + top-5 block pairs
    per_method_rows = []
    pmaps = {k: load_pmap(p) for k, p in files.items()}

    for method, p in pmaps.items():
        for tail, full in p.items():
            n_sig = int((upper(full) < 0.05).sum())
            counts, pair_idx = summarize_networks(full, 0.05, net_labels, network_order)
            # Per-block entries
            for (bi, bj), c in counts.items():
                pref = "within" if bi == bj else "between"
                per_method_rows.append({
                    "method": method,
                    "tail": tail,
                    "n_sig": n_sig,
                    "block_pair_idx": pair_idx[(bi, bj)],
                    "block_a": network_order[bi],
                    "block_b": network_order[bj],
                    "n_edges": c,
                    "kind": pref,
                })

    per_df = pd.DataFrame(per_method_rows)
    per_df.to_csv(os.path.join(AUDIT_DIR, "per_method_networks.csv"), index=False)
    print(f"  per-method network breakdown → per_method_networks.csv  ({len(per_df)} rows)")

    # n_sig summary
    nsig = (
        per_df.groupby(["method", "tail"])
        .agg(n_sig=("n_sig", "first"))
        .reset_index()
        .pivot(index="method", columns="tail", values="n_sig")
        .fillna(0).astype(int)
    )
    nsig.to_csv(os.path.join(AUDIT_DIR, "n_sig_summary.csv"))
    print(f"\nn_sig per method (α=0.05):\n{nsig}")

    # Pairwise §2.5 quartet (per tail)
    methods = [m for m in METHOD_ORDER if m in files]
    pair_rows = []
    for i, a in enumerate(methods):
        for b in methods[i+1:]:
            for tail in ["positive", "negative"]:
                pa = pmaps[a].get(tail); pb = pmaps[b].get(tail)
                if pa is None or pb is None:
                    continue
                agr = agreement(pa, pb, net_labels)
                pair_rows.append({"a": a, "b": b, "tail": tail, **agr})
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(os.path.join(AUDIT_DIR, "pairwise_quartet.csv"), index=False)
    print(f"\n{len(pair_df)} pairwise comparisons → pairwise_quartet.csv")

    # Aggregate across tails for the panel plot
    agg = pair_df.groupby(["a", "b"]).agg({
        "spearman": "mean", "block_corr": "mean",
        "top10": "mean", "top50": "mean", "top100": "mean", "top500": "mean",
        "jaccard": "mean", "jaccard_random": "mean",
    }).reset_index()
    agg.to_csv(os.path.join(AUDIT_DIR, "pairwise_quartet_avg_tails.csv"), index=False)

    plot_method_panel(agg, methods, net_labels,
                      outpath=os.path.join(AUDIT_DIR, "method_agreement_panel.png"))
    print(f"\nFigure: method_agreement_panel.png")

    plot_network_profile(per_df, outpath=os.path.join(AUDIT_DIR, "network_profile.png"),
                         network_order=network_order, net_labels=net_labels)
    print(f"Figure: network_profile.png")

    # Print a concise table
    print(f"\nPairwise (averaged across tails) highlights:")
    for col in ["spearman", "block_corr", "jaccard"]:
        print(f"\n  {col}:")
        M = _build_matrix(agg, col, methods)
        print(M.round(2))


if __name__ == "__main__":
    main()
