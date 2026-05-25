# Open-Close Validation Plan: The Causal-Adjacent Replication Story

**Status:** Active Validation Protocol (Updated with Results 2026-05-25)
**Goal:** Demonstrate the biological specificity, test-retest separability, and predictive utility of `conninfpy` on a paired, within-subject multi-site dataset comparing eyes-open vs. eyes-closed states.

---

## 1. The Core Narrative

While the ABIDE dataset is our test case for observational cross-sectional inference and harmonization, the Open-Close dataset provides the strongest causal-adjacent design: a paired within-subject contrast.

By comparing eyes-open to eyes-closed states, we expect canonical fMRI shifts (visual / somatomotor coupling vs. default mode upregulation). The narrative relies on a strict two-cohort repeated-measures design:

1.  **Causal-Adjacent Replication.** Canonical brain-state shifts replicate at the network level across independent scanners (IHB St. Petersburg and Beijing China) and populations.
2.  **Test-Retest Separability.** The pipeline isolates state-dependent signal rather than vigilance drift or scanner drift, as proven by the near-zero agreement with a matched-design empirical null.
3.  **Machine Learning Transfer.** Significant edges define the stable biological core by maintaining cross-site classification performance with a fraction of the features (>95% reduction).

---

## 2. Summary of Results

### 2.1 Cross-Cohort Replication (IHB vs. China)
- **Edge level:** Jaccard ≈ **0.148** (pos) and **0.159** (neg) — significantly above random baselines (0.042 / 0.024), Fisher's exact p < 10⁻¹¹¹.
- **Rank level:** Spearman correlation of −log10 p gradient is ≈ **0.25** on both tails.
- **System level:** Yeo-7 block-mass Pearson is **0.79** (pos) and **0.61** (neg).
- **Insight**: Strong convergence at the network-block level confirms canonical visual upregulation (EO) and default-mode upregulation (EC) across both scanners.

### 2.2 Test-Retest Separability (China)
- **Contrast**: "Open vs. Close (run 0)" vs. "Close run 0 vs. Close run 1".
- **Specificity**: Jaccard ≈ **0.006 / 0.0** against random baselines (0.023 / 0.015), Spearman ≈ **−0.10**.
- **Insight**: The paired TFNBS pipeline cleanly separates biological signal from matched-design noise or scanner drift.

### 2.3 Machine Learning Transfer AUC (IHB <-> China)
- **Baseline**: All-edges average cross-site AUC = **0.880**.
- **TFNBS Selection**: Average AUC = **0.885** (with **2,376 edges** vs. 16,471 total).
- **Conclusion**: Topological selection successfully isolates the predictively useful biological core, capturing the full signal with >85% fewer features.

---

## 3. Publication Figures

**Output Directory:** `examples/openclose_validation/results/plots/`

### Site-Effect Neutralization Diagnostic
- **Claim**: ComBat aligns pooled scanner variance without erasing state-dependent signal.
- **Filename**: `plot1_combat_impact.png`

### Block-Mass Matrix Convergence
- **Claim**: Visualizes strong cross-cohort replication of the canonical integration patterns.
- **Filename**: `plot2_block_mass_convergence.png`

### Method Sensitivity & Convergence
- **Claim**: Dual Jaccard/Spearman analysis proves TFNBS robustness compared to brittle fixed thresholds.
- **Filename**: `plot3_method_sensitivity_dual.png`

### TFNBS Grid Multiverse Sensitivity
- **Claim**: Proves that the TFNBS core is insensitive to specific (E, H) parameter choices.
- **Filename**: `plot4_tfnbs_grid_sensitivity.png`

### Test-Retest Specificity
- **Claim**: Directly contrasts biological discovery against a matched matched-design baseline.
- **Filename**: `plot5_retest_specificity.png`

### Hierarchy of Network Constraint
- **Claim**: Demonstrates pruning of topological leakage via network-informed priors.
- **Filename**: `plot6_network_informed_hierarchy.png`

### Machine Learning Transfer Stability
- **Claim**: Confirms that discovery edges define a predictively stable and transferable core.
- **Filename**: `plot7_ml_transfer_auc.png`
