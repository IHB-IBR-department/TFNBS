# Simulation & Calibration Validation

This directory implements the "Multiverse Audit" axis: family-wise error (FWE) control under the null, and statistical power under known topologies.

## Scientific Claims

1.  **Null Calibration (FPR)**: Demonstration that α=0.05 yields exactly ~5% False Positive Rate across all enhancement methods.
2.  **Topological Power**: TPR, FDR, and Precision curves across different spatial effect patterns (rich clubs, hubs, chains).
3.  **Multiverse Sensitivity**: Audit of hyperparameters (TFNBS exponents, NBS thresholds, cluster sizes).

## Execution Order

- **Calibration**: `python fpr/fpr_calibration.py` (uses `configs/fpr_config_quick.yaml` for testing).
- **Power**: `python power/power_analysis.py` (uses `configs/sweep_config_power_quick.yaml`).
- **Audit**: `python sensitivity/analyze_mmin_sensitivity.py` (etc).

## Directory Structure

- `fpr/`: False Positive Rate calibration scripts.
- `power/`: Topology-dependent power analyses.
- `sensitivity/`: Hyperparameter sensitivity sweeps (Specification Curve analysis).
- `topology_gallery.py`: Interactive visualization of simulated effect patterns.
