"""
Run all selectors on the pooled harmonized Open-Close dataset.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from conninfpy import compute_p_val
from examples.openclose_validation.openclose_loader import OpenCloseDataset

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
ML_DIR = RESULTS / "ml"
HARM_PATH = RESULTS / "openclose_harmonized.npz"

SELECTORS = {
    "tstat":     dict(method="tstat"),
    "tfnbs":     dict(method="tfnbs", e=0.4, h=3.0, n=10),
    "ni_tfnbs":  dict(method="ni_tfnbs", e=0.4, h=3.0, n=10),
    "fbc_tfnbs": dict(method="fbc_tfnbs"),
    "nbs_30":    dict(method="nbs", threshold=3.0),
}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ML_DIR.mkdir(parents=True, exist_ok=True)

    if not HARM_PATH.exists():
        print(f"Error: {HARM_PATH} not found. Run harmonize_pooled_cohorts.py first.")
        return

    d = np.load(HARM_PATH)
    # Stack across cohorts
    o_z = np.concatenate([d["Y_ihb_open_harm"], d["Y_china_open_harm"]], axis=0)
    c_z = np.concatenate([d["Y_ihb_close_harm"], d["Y_china_close_harm"]], axis=0)
    
    # Load labels
    ds = OpenCloseDataset.load("ihb")
    net_labels = ds.net_labels
    
    # Stratum labels for exchangeability blocking (cohorts)
    n_ihb = d["Y_ihb_open_harm"].shape[0]
    n_china = d["Y_china_open_harm"].shape[0]
    strata = np.concatenate([np.zeros(n_ihb), np.ones(n_china)])

    print(f"Pooled harmonized N={o_z.shape[0]} (IHB={n_ihb}, China={n_china})")
    print(f"n_perm={args.n_perm}")

    for name, kwargs in SELECTORS.items():
        base_kwargs = dict(kwargs)
        if base_kwargs["method"] in ("cnbs", "ni_tfnbs", "fbc_tfnbs"):
            base_kwargs["net_labels"] = net_labels
        
        t0 = time.time()
        p = compute_p_val(
            o_z, c_z,
            test_type="paired",
            n_permutations=args.n_perm,
            use_mp=True,
            random_state=args.seed,
            strata=strata,
            **base_kwargs,
        )
        dt = time.time() - t0
        
        out_path = ML_DIR / f"pmap_pooled_{name}.npz"
        np.savez(out_path, **{k: p[k] for k in p})
        print(f"  {name:10s} {dt:5.1f}s saved: {out_path}")

if __name__ == "__main__":
    main()
