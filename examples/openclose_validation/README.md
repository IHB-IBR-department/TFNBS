# Open-Close Paired Validation Suite

This directory contains the production-ready validation suite for `conninfpy` using the IHB and Beijing (China) Open-Close datasets. It demonstrates biological replication, specificity under matched-design nulls, and predictive stability.

## 1. Scientific Claims

1.  **Causal-Adjacent Replication**: Canonical brain-state shifts (Visual vs. DMN) replicate at the network level across independent scanners and populations.
2.  **Test-Retest Specificity**: The pipeline cleanly separates state-dependent signal from matched-design vigilance or scanner drift.
3.  **Predictive Parsimony**: Topologically selected significant edges capture the predictively stable biological core, maintaining cross-site classification performance with >95% fewer features.

## 2. Core Execution Scripts

| Script | Purpose | Output |
|---|---|---|
| `run_paired_tfnbs.py` | Canonical paired sign-flip inference on both cohorts and retest runs. | `results/{ihb,china}_paired_tfnbs.npz` |
| `harmonize_pooled_cohorts.py` | Cross-cohort ComBat (Cohort = Batch, State = Preserved) for the pooled analysis. | `results/openclose_harmonized.npz` |
| `run_eh_sensitivity.py` | Grid sweep of TFNBS (E, H) parameters on the paired China cohort. | `results/plots/plot4_tfnbs_grid_sensitivity.png` |
| `ml/run_ml_feature_selection.py` | Cross-site biomarker transfer evaluation (IHB <-> China). | `results/ml/ml_feature_selection.csv` |

## 3. Publication Figures

**Output Directory:** `results/plots/`

| Plot # | Filename | Script | Claim / Description |
|---|---|---|---|
| **1** | `plot1_combat_impact.png` | `plot1_combat_impact.py` | **Site-Effect Neutralization**: Shows how ComBat aligns pooled scanner variance without erasing signal. |
| **2** | `plot2_block_mass_convergence.png`| `plot2_block_mass_convergence.py` | **Cohort Replication**: Strong systems-level convergence between independent cohorts. |
| **3** | `plot3_method_sensitivity_dual.png`| `plot3_method_sensitivity.py` | **Stability & Convergence**: Dual Jaccard/Spearman analysis proving TFNBS robustness. |
| **4** | `plot4_tfnbs_grid_sensitivity.png` | `run_eh_sensitivity.py` | **Parameter Multiverse**: Proves TFNBS core is insensitive to specific (E, H) choices. |
| **5** | `plot5_retest_specificity.png` | `plot5_retest_specificity.py` | **Retest Specificity**: Contrasts biological discovery against a matched matched-design baseline. |
| **6** | `plot6_network_informed_hierarchy.png` | `plot6_network_informed_hierarchy.py` | **Hierarchy of Constraint**: Demonstrates pruning of topological leakage via soft/hard priors. |
| **7** | `plot7_ml_transfer_auc.png` | `plot7_ml_transfer_auc.py` | **Predictive Utility**: Confirms that discovery edges define a predictively transferable core. |

## 4. Quick Start

To regenerate the entire validation story:
```bash
# Activate environment
conda activate conninfpy

# 1. Run inference and sweeps
python run_paired_tfnbs.py
python harmonize_pooled_cohorts.py
python run_eh_sensitivity.py
python ml/run_ml_feature_selection.py

# 2. Run audits
python audit_openclose_agreement.py
python audit_ml_feature_selection.py

# 3. Generate all plots
python plot1_combat_impact.py
python plot2_block_mass_convergence.py
python plot3_method_sensitivity.py
python plot5_retest_specificity.py
python plot6_network_informed_hierarchy.py
python plot7_ml_transfer_auc.py
```
