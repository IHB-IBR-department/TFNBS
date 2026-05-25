# ABIDE Validation Plan: The "Reality Check" Story

**Status:** Active Validation Protocol (Updated with Results 2026-05-25)
**Goal:** Demonstrate the applicability and usability of `conninfpy` on the ABIDE I dataset, specifically catering to the ML biomarker literature while maintaining absolute statistical rigor.

---

## 1. The Core Narrative

ABIDE is the obvious test case because the autism biomarker literature has used it heavily. That makes it scientifically useful and risky at the same time: site effects, leakage, and optimistic ML validation are common failure modes.

The validation story is ordered by evidential strength:
1.  **Positive Controls (ABIDE Age)**: Strong, known signals.
2.  **Parameter Sensitivity**: Prove results aren't arbitrary.
3.  **Observational Case-Control (ABIDE Diagnosis)**: The final stress test involving multi-site harmonization (Strategy D).

---

## 2. Summary of Results

### 2.1 Positive Control (Age)
- **Contrast**: Older vs. Younger adults (15 sites, ComBat Strategy D).
- **Discovery**: Found **256 significant edges** (FWER < 0.05).
- **Interpretation**: Successfully captured the canonical integration-segregation trade-off in the maturing brain.

### 2.2 Parameter Sensitivity (E, H)
- **Grid**: 36-cell grid sweep (E ∈ {0.2–1.3}, H ∈ {1–10}).
- **Stability**: **Spearman Median = 0.997**, **Jaccard Median = 0.699**.
- **Conclusion**: TFNBS provides a remarkably stable topological core across the entire literature-standard parameter space.

### 2.3 Diagnosis Discovery (ASD vs. HC)
- **Naive Baseline**: **520 edges** (Extremely high site/scanner leakage).
- **Strategy D (Corrected)**: **187 edges**.
- **The "Reality Check"**: Removing scanner-specific mean shifts and scaling differences reduces discovery to a robust, biological core, exposing the over-optimism of site-unaware ML models.

### 2.4 Hierarchy of Network Constraint
- **TFNBS (Unrestricted)**: **187 edges**.
- **FBC-TFNBS (Boundary)**: **157 edges**.
- **NI-TFNBS (Prior)**: **85 edges**.
- **cNBS (Block)**: **675 edges**.
- **Insight**: Moving from unrestricted TFNBS to NI-TFNBS prunes "topological leakage," isolating the unassailable biological core (85 edges).

---

## 3. Publication Figures

**Output Directory:** `examples/abide_validation/results/plots/`

### Site-Variance Reduction Diagnostic
- **Claim**: Strategy D successfully crushed scanner/site variance down to the null floor without leaking the ASD label.
- **Filename**: `plot1_combat_site_variance.png`

### Diagnosis Effect-Size Distribution
- **Claim**: True univariate effects for ASD vs. HC are small (r ≈ 0.10 - 0.15), exposing "Optimism Bias" in ML.
- **Filename**: `plot2_dx_effect_size_distribution.png`

### Network-Level Block-Mass Matrix (Age Control)
- **Claim**: Translates edge-lists into interpretable systems neuroscience.
- **Filename**: `plot3_network_block_mass.png`

### Method Sensitivity & Convergence (Age Control)
- **Claim**: TFNBS variants form a stable topological core compared to brittle fixed thresholds.
- **Filename**: `plot4_dual.png`

### TFNBS Grid Multiverse Sensitivity
- **Claim**: Characterizes the absolute stability of TFNBS across the full (E, H) literature box.
- **Filename**: `plot5_tfnbs_grid_sensitivity.png`

### Evolution of Site-Aware Inference in ABIDE (2x2)
- **Claim**: Contrasts Naive vs Site-Aware tiers to prove Strategy D necessity.
- **Filename**: `plot6_diagnosis_methodology.png`

### Hierarchy of Network Constraint in ASD Discovery
- **Claim**: Shows how network-informed priors reveal the unassailable biological core.
- **Filename**: `plot7_diagnosis_hierarchy.png`
