"""
ABIDE §3.2 — Within-site replication (NYU vs UM_1).

The real replication test. Run the same analyses on NYU (n ~ 172 after QC)
and UM_1 (n ~ 85), each as a single-site analysis — no site confound
because there is no between-site variance to adjust for.

We deliberately run on BOTH:
  (A) Age → FC  (strong signal — expected to replicate strongly)
  (B) ASD vs HC two-sample naive
  (C) ASD vs HC GLM with confounds

Comparing the within-site §2.5 quartet on (A) vs (B/C) is the scientifically
interesting contrast: does the method reproduce the strong continuous-predictor
signal while (appropriately) failing to reproduce the weak DX contrast?

Uses the raw Fisher-z connectivity (not the ComBat-harmonised version) —
within a single site, ComBat has no batch variance to remove, and using
raw features keeps the within-site analysis independent of global ComBat
fit choices.

Output: results/within_site/{nyu,um1}/*.npz
Audit separately with audit_within_site.py.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from conninfpy import compute_p_val, compute_p_val_glm


HERE = os.path.dirname(os.path.abspath(__file__))
PREPARED = os.path.join(HERE, "abide_prepared.npz")
OUT_ROOT = os.path.join(HERE, "results", "within_site")

SITES = {
    "nyu": ["NYU"],
    "um1": ["UM_1"],          # keep UM_1 alone — pooling with UM_2 reintroduces a site effect
}


def _subset(data, site_names):
    site = data["site"]
    mask = np.isin(site, site_names)
    print(f"  site filter {site_names}: {mask.sum()} subjects")
    return {k: (v[mask] if hasattr(v, "shape") and v.shape and v.shape[0] == len(site) else v)
            for k, v in data.items()}


def _report(label, p_dict, net_labels, network_order, alpha=0.05):
    iu = np.triu_indices(100, k=1)
    for k, v in p_dict.items():
        vec = v[iu]
        n_sig = int((vec < alpha).sum())
        print(f"  {label} [{k}]: n_sig={n_sig}, min_p={vec.min():.4f}")
        if n_sig > 0:
            sig = vec < alpha
            counts = {}
            for ii, jj in zip(iu[0][sig], iu[1][sig]):
                pair = tuple(sorted([network_order[net_labels[ii]], network_order[net_labels[jj]]]))
                counts[pair] = counts.get(pair, 0) + 1
            for (n1, n2), c in sorted(counts.items(), key=lambda x: -x[1])[:5]:
                pref = "within" if n1 == n2 else "between"
                print(f"    {pref} {n1}-{n2}: {c}")


def run_site(data_full, site_key, site_names, n_perm, seed):
    print("\n" + "=" * 72)
    print(f"SITE: {site_key}  (names={site_names})")
    print("=" * 72)
    ds = _subset(data_full, site_names)
    n = int(ds["group"].sum()) + int((ds["group"] == 0).sum())
    if len(ds["group"]) < 40:
        print(f"  {len(ds['group'])} subjects — too small, skipping")
        return
    net_labels = data_full["net_labels"]
    network_order = list(data_full["network_order"])
    out_dir = os.path.join(OUT_ROOT, site_key)
    os.makedirs(out_dir, exist_ok=True)

    # ---- (A) Age → FC GLM ------------------------------------------
    print(f"\n(A) Age → FC GLM (interest=age, confounds=[sex, mean_fd])")
    age = ds["age"].astype(float)
    sex = ds["sex"].astype(float)
    fd = ds["mean_fd"].astype(float)
    confounds = np.column_stack([sex, fd])
    print(f"  age range {age.min():.1f}–{age.max():.1f}")
    t0 = time.time()
    p_age = compute_p_val_glm(
        ds["connectivity_z"], interest=age, confounds=confounds,
        method="tfnbs", n_permutations=n_perm,
        e=0.4, h=3.0, n=10, use_mp=True, random_state=seed,
    )
    print(f"  time: {time.time() - t0:.1f}s")
    _report("age TFNBS", p_age, net_labels, network_order)
    np.savez(os.path.join(out_dir, "age_tfnbs.npz"), **p_age)

    # ---- (B) ASD vs HC — naive two-sample ---------------------------
    print(f"\n(B) Naive two-sample (ASD vs HC, no confounds)")
    hc = ds["connectivity_z"][ds["group"] == 0]
    asd = ds["connectivity_z"][ds["group"] == 1]
    print(f"  HC={hc.shape[0]}, ASD={asd.shape[0]}")
    if hc.shape[0] < 5 or asd.shape[0] < 5:
        print("  skipping — group too small for two-sample test")
    else:
        t0 = time.time()
        p_naive = compute_p_val(
            hc, asd,
            test_type="two-sample", method="tfnbs",
            n_permutations=n_perm, e=0.4, h=3.0, n=10,
            use_mp=True, random_state=seed,
        )
        print(f"  time: {time.time() - t0:.1f}s")
        _report("naive TFNBS", p_naive, net_labels, network_order)
        np.savez(os.path.join(out_dir, "dx_naive_tfnbs.npz"), **p_naive)

    # ---- (C) ASD vs HC — GLM with confounds -------------------------
    print(f"\n(C) GLM DX (interest=group, confounds=[age, sex, mean_fd])")
    group = ds["group"].astype(float)
    glm_confounds = np.column_stack([age, sex, fd])
    t0 = time.time()
    p_glm = compute_p_val_glm(
        ds["connectivity_z"], interest=group, confounds=glm_confounds,
        method="tfnbs", n_permutations=n_perm,
        e=0.4, h=3.0, n=10, use_mp=True, random_state=seed,
    )
    print(f"  time: {time.time() - t0:.1f}s")
    _report("GLM DX", p_glm, net_labels, network_order)
    np.savez(os.path.join(out_dir, "dx_glm_tfnbs.npz"), **p_glm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sites", nargs="+", choices=list(SITES.keys()) + ["all"], default=["all"])
    args = ap.parse_args()
    n_perm = 500 if args.quick else 5000

    data = np.load(PREPARED, allow_pickle=True)
    data = {k: data[k] for k in data.files}
    targets = list(SITES.keys()) if args.sites == ["all"] else args.sites
    for site_key in targets:
        run_site(data, site_key, SITES[site_key], n_perm, args.seed)


if __name__ == "__main__":
    main()
