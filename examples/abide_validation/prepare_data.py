"""
ABIDE data preparation for ConnInfPy validation.

Loads time series from abide_ts.npy, computes Pearson correlation matrices,
applies Fisher z-transform, merges with phenotypic data, applies QC
exclusions, and saves prepared dataset as .npz.

Usage:
    python examples/abide_validation/prepare_data.py
"""

import os
import re
import sys

import numpy as np
import pandas as pd

from conninfpy import fisher_r_to_z

# =============================================================================
# Paths
# =============================================================================

DATA_DIR = os.path.expanduser(
    "~/Yandex.Disk.localized/IHB/ABIDE_TS_Schaefer100"
)
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "abide_prepared.npz")

TS_FILE = os.path.join(DATA_DIR, "abide_ts.npy")
PHENO_FILE = os.path.join(DATA_DIR,
    "ABIDE1_Phenotypic_V1_0b_preprocessed1_from_nilearn.csv")
LABELS_FILE = os.path.join(DATA_DIR, "sch100_labels.csv")

# =============================================================================
# QC thresholds
# =============================================================================

MIN_TS_LENGTH = 100        # minimum time points
MAX_MEAN_FD = 0.5          # maximum mean framewise displacement (mm)
MIN_SUBJECTS_PER_SITE = 10  # minimum subjects per site per group
MISSING_VALUE = -9999       # ABIDE missing value code


def load_and_prepare():
    print("Loading time series...")
    ts_data = np.load(TS_FILE, allow_pickle=True).item()
    print(f"  Total subjects in ts: {len(ts_data)}")

    print("Loading phenotypic data...")
    pheno = pd.read_csv(PHENO_FILE)
    pheno = pheno.replace(MISSING_VALUE, np.nan)
    print(f"  Phenotypic rows: {len(pheno)}")

    print("Loading ROI labels...")
    roi_labels = pd.read_csv(LABELS_FILE)
    n_rois = len(roi_labels)
    print(f"  ROIs: {n_rois}")

    # Network labels for constrained methods
    network_names = roi_labels["Network"].values
    unique_networks = sorted(set(network_names))
    net_label_map = {name: i for i, name in enumerate(unique_networks)}
    net_labels = np.array([net_label_map[n] for n in network_names])
    print(f"  Networks: {unique_networks}")

    # =========================================================================
    # Match subjects between ts and phenotypic
    # =========================================================================
    print("\nMatching subjects...")

    # Extract numeric IDs from ts keys (e.g., 'CMU_a_0050642' → 50642)
    ts_subjects = {}
    for key in ts_data:
        nums = re.findall(r"(\d{7})", key)
        if nums:
            ts_subjects[int(nums[0])] = key

    # Match with phenotypic
    pheno_matched = pheno[pheno["SUB_ID"].isin(ts_subjects.keys())].copy()
    pheno_matched = pheno_matched.sort_values("SUB_ID").reset_index(drop=True)
    print(f"  Matched subjects: {len(pheno_matched)}")

    # =========================================================================
    # Compute connectivity matrices
    # =========================================================================
    print("\nComputing connectivity matrices...")

    sub_ids = pheno_matched["SUB_ID"].values
    connectivity = []
    ts_lengths = []
    valid_mask = np.ones(len(sub_ids), dtype=bool)

    for i, sub_id in enumerate(sub_ids):
        ts_key = ts_subjects[sub_id]
        ts = ts_data[ts_key]["time_series"]  # (T, 100)
        ts_lengths.append(ts.shape[0])

        # QC: minimum time series length
        if ts.shape[0] < MIN_TS_LENGTH:
            valid_mask[i] = False
            connectivity.append(np.zeros((n_rois, n_rois)))
            continue

        # Standardize time series per ROI
        ts_std = (ts - ts.mean(axis=0)) / (ts.std(axis=0) + 1e-10)

        # Pearson correlation
        corr = np.corrcoef(ts_std.T)  # (100, 100)
        np.fill_diagonal(corr, 0)

        # Clip to [-1, 1] (numerical safety)
        corr = np.clip(corr, -1, 1)

        connectivity.append(corr)

    connectivity = np.array(connectivity)  # (n_subjects, 100, 100)
    ts_lengths = np.array(ts_lengths)
    print(f"  Excluded for short TS (<{MIN_TS_LENGTH}): {np.sum(~valid_mask)}")

    # =========================================================================
    # QC: motion exclusion
    # =========================================================================
    print("\nApplying motion QC...")
    mean_fd = pheno_matched["func_mean_fd"].values.astype(float)
    motion_valid = (mean_fd <= MAX_MEAN_FD) & (~np.isnan(mean_fd))
    print(f"  Excluded for high motion (>{MAX_MEAN_FD}mm): "
          f"{np.sum(valid_mask & ~motion_valid)}")
    valid_mask &= motion_valid

    # =========================================================================
    # QC: site size filter
    # =========================================================================
    print("\nApplying site size filter...")
    dx = pheno_matched["DX_GROUP"].values  # 1=ASD, 2=control
    sites = pheno_matched["SITE_ID"].values

    # Count per site per group among valid subjects
    site_counts = {}
    for i in range(len(sub_ids)):
        if not valid_mask[i]:
            continue
        site = sites[i]
        grp = dx[i]
        key = (site, grp)
        site_counts[key] = site_counts.get(key, 0) + 1

    # Find sites with enough subjects in both groups
    valid_sites = set()
    all_sites = set(s for s, _ in site_counts.keys())
    for site in all_sites:
        n_asd = site_counts.get((site, 1), 0)
        n_hc = site_counts.get((site, 2), 0)
        if n_asd >= MIN_SUBJECTS_PER_SITE and n_hc >= MIN_SUBJECTS_PER_SITE:
            valid_sites.add(site)

    excluded_sites = all_sites - valid_sites
    if excluded_sites:
        print(f"  Excluded sites (<{MIN_SUBJECTS_PER_SITE}/group): {excluded_sites}")

    site_valid = np.array([s in valid_sites for s in sites])
    valid_mask &= site_valid

    # =========================================================================
    # Apply mask and Fisher transform
    # =========================================================================
    print("\nFinalizing dataset...")

    conn_final = connectivity[valid_mask]
    conn_z = fisher_r_to_z(conn_final)

    pheno_final = pheno_matched[valid_mask].reset_index(drop=True)

    # Extract arrays
    dx_final = pheno_final["DX_GROUP"].values  # 1=ASD, 2=control
    group = (dx_final == 1).astype(float)  # 1=ASD, 0=control
    age = pheno_final["AGE_AT_SCAN"].values.astype(float)
    sex = pheno_final["SEX"].values.astype(float)  # 1=male, 2=female
    mean_fd_final = pheno_final["func_mean_fd"].values.astype(float)
    site_final = pheno_final["SITE_ID"].values

    # Site dummy variables
    site_dummies = pd.get_dummies(site_final, drop_first=True).values.astype(float)
    site_names = pd.get_dummies(site_final, drop_first=True).columns.tolist()

    # ADOS scores (for severity regression, ASD only)
    ados_total = pheno_final["ADOS_TOTAL"].values.astype(float)
    ados_css = pheno_final["ADOS_GOTHAM_SEVERITY"].values.astype(float)
    ados_module = pheno_final["ADOS_MODULE"].values.astype(float)
    srs_total = pheno_final["SRS_RAW_TOTAL"].values.astype(float)

    # =========================================================================
    # Report demographics
    # =========================================================================
    n_total = len(pheno_final)
    n_asd = np.sum(group == 1)
    n_hc = np.sum(group == 0)

    print(f"\n{'='*60}")
    print(f"FINAL DATASET")
    print(f"{'='*60}")
    print(f"Total subjects: {n_total}")
    print(f"  ASD: {n_asd}")
    print(f"  HC:  {n_hc}")
    print(f"  Sites: {len(valid_sites)}")
    print(f"  ROIs: {n_rois}")
    print(f"  Edges: {n_rois * (n_rois - 1) // 2}")
    print(f"\nAge: ASD={age[group==1].mean():.1f}±{age[group==1].std():.1f}, "
          f"HC={age[group==0].mean():.1f}±{age[group==0].std():.1f}")
    print(f"Sex (M%): ASD={100*np.mean(sex[group==1]==1):.0f}%, "
          f"HC={100*np.mean(sex[group==0]==1):.0f}%")
    print(f"Mean FD: ASD={mean_fd_final[group==1].mean():.3f}±{mean_fd_final[group==1].std():.3f}, "
          f"HC={mean_fd_final[group==0].mean():.3f}±{mean_fd_final[group==0].std():.3f}")

    # ADOS coverage (ASD only)
    asd_mask = group == 1
    n_ados_total = np.sum(~np.isnan(ados_total[asd_mask]))
    n_ados_css = np.sum(~np.isnan(ados_css[asd_mask]))
    n_srs = np.sum(~np.isnan(srs_total[asd_mask]))
    print(f"\nSeverity data (ASD only):")
    print(f"  ADOS Total: {n_ados_total}/{n_asd}")
    print(f"  ADOS Gotham CSS: {n_ados_css}/{n_asd}")
    print(f"  SRS Raw Total: {n_srs}/{n_asd}")

    # =========================================================================
    # Save
    # =========================================================================
    print(f"\nSaving to {OUTPUT_FILE}...")
    np.savez(
        OUTPUT_FILE,
        # Connectivity
        connectivity_z=conn_z,
        # Group variables
        group=group,
        dx_group=dx_final,
        # Demographics
        age=age,
        sex=sex,
        mean_fd=mean_fd_final,
        site=site_final,
        site_dummies=site_dummies,
        site_names=site_names,
        # Severity (ASD only, NaN for HC)
        ados_total=ados_total,
        ados_css=ados_css,
        ados_module=ados_module,
        srs_total=srs_total,
        # ROI info
        roi_names=roi_labels["ROI"].values,
        network_names=network_names,
        net_labels=net_labels,
        network_order=unique_networks,
        # Subject IDs
        sub_ids=pheno_final["SUB_ID"].values,
    )
    print(f"Done. File size: {os.path.getsize(OUTPUT_FILE) / 1e6:.1f} MB")


if __name__ == "__main__":
    load_and_prepare()
