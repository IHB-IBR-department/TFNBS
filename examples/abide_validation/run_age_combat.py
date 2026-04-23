"""
ABIDE — Age → FC regression as GLM positive control.

Age effects on FC are among the most replicated continuous-predictor findings
in neuroimaging: within-network integration decreases and between-network
integration increases across development (segregation → integration trade-off).
With n=764 subjects and a 6-58y range, this is the cleanest "does the GLM
pipeline actually work on real data" check available in ABIDE.

Runs three variants on ComBat-harmonised features:
  (1) age → FC, confounds = sex, FD (full cohort, diagnosis-agnostic)
  (2) age → FC, confounds = sex, FD, diagnosis (controls for possible age × DX difference)
  (3) mean-FD → FC, confounds = age, sex, diagnosis (motion positive control —
      FD should drive widespread "connectivity inflation" in known long-range edges)

All three use TFNBS FWER at n_permutations=5000.

Output: results/combat/age/
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from conninfpy import compute_p_val_glm


HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(HERE, "abide_harmonized.npz")
OUT_DIR = os.path.join(HERE, "results", "combat", "age")


def _load():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} not found — run `python harmonize.py` first.")
    d = np.load(DATA_FILE, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _summary(label, p_dict, alpha=0.05, net_labels=None, network_order=None):
    N = p_dict[list(p_dict.keys())[0]].shape[0]
    iu = np.triu_indices(N, k=1)
    for k, v in p_dict.items():
        vec = v[iu]
        n_sig = int((vec < alpha).sum())
        print(f"  {label} [{k}]: n_sig={n_sig} / {len(vec)}, min_p={vec.min():.4f}")
        if n_sig > 0 and net_labels is not None and network_order is not None:
            sig = vec < alpha
            sig_i, sig_j = iu[0][sig], iu[1][sig]
            counts = {}
            for ii, jj in zip(sig_i, sig_j):
                pair = tuple(sorted([network_order[net_labels[ii]], network_order[net_labels[jj]]]))
                counts[pair] = counts.get(pair, 0) + 1
            for (n1, n2), c in sorted(counts.items(), key=lambda x: -x[1])[:8]:
                pref = "within" if n1 == n2 else "between"
                print(f"    {pref} {n1:>24s} × {n2:<24s}  {c} edges")


def run_variant(data, interest_name, confound_names, label, n_perm, seed):
    print("\n" + "=" * 72)
    print(f"{label}: interest={interest_name}, confounds={confound_names}")
    print("=" * 72)

    conn = data["connectivity_z_harm"]
    interest = data[interest_name].astype(float)
    confounds = np.column_stack([data[c].astype(float) for c in confound_names])
    print(f"  N={conn.shape[0]}, confounds shape {confounds.shape}")

    t0 = time.time()
    p = compute_p_val_glm(
        conn, interest=interest, confounds=confounds,
        method="tfnbs", n_permutations=n_perm,
        e=0.4, h=3.0, n=10, use_mp=True, random_state=seed,
    )
    print(f"  time: {time.time() - t0:.1f}s")
    _summary(f"TFNBS-FWER {label}", p, net_labels=data["net_labels"],
             network_order=data["network_order"])

    out = os.path.join(OUT_DIR, f"{label}.npz")
    np.savez(out, **p)
    print(f"  saved: {out}")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--variant", choices=["age", "age_dx", "fd", "all"], default="all")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    n_perm = 500 if args.quick else 5000

    os.makedirs(OUT_DIR, exist_ok=True)
    data = _load()

    if args.variant in ("age", "all"):
        run_variant(data, "age", ["sex", "mean_fd"],
                    label="age_tfnbs", n_perm=n_perm, seed=args.seed)
    if args.variant in ("age_dx", "all"):
        run_variant(data, "age", ["sex", "mean_fd", "group"],
                    label="age_controlling_dx_tfnbs", n_perm=n_perm, seed=args.seed)
    if args.variant in ("fd", "all"):
        run_variant(data, "mean_fd", ["age", "sex", "group"],
                    label="fd_tfnbs", n_perm=n_perm, seed=args.seed)


if __name__ == "__main__":
    main()
