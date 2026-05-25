# ConnInfPy Examples & Validation

This directory contains the tutorials, paper validation pipelines, and performance benchmarks for the `conninfpy` package.

## Audience Routing

- **New to ConnInfPy?** Start with the **[notebooks/](./notebooks/)** for interactive tutorials on t-tests, GLM inference, and enhancement methods.
- **Evaluating Paper Claims?** See the validation folders below. We recommend following the **Evidence Hierarchy** order.
- **Performance Tuning?** See **[benchmarks/](./benchmarks/)** for acceleration and GLM timing tests.

## Evidence Hierarchy (Validation Folders)

The validation of `conninfpy` is organized by the strength of the statistical design, moving from controlled simulations to observational multi-site data:

1.  **[simulation_validation/](./simulation_validation/)**: **Null Calibration (FPR)** and Power. The foundation of belief; demonstrates that the package controls Type I error under null data.
2.  **[abide_validation/](./abide_validation/)**: **Positive Controls (Age, Motion)**. Demonstrates that the pipeline reliably detects strong, well-replicated developmental and artifactual signals in real fMRI data.
3.  **[openclose_validation/](./openclose_validation/)**: **Paired Within-Subject (Open vs. Closed Eyes)**. The strongest causal-adjacent real-data design, utilizing exact sign-flip permutations.
4.  **[abide_validation/](./abide_validation/)**: **Observational Case-Control (ASD vs. HC)**. The final stress test involving multi-site harmonization (ComBat Strategy D) and stratified permutations.

---

## Directory Overview

- `abide_validation/`: Multi-site GLM validation on the ABIDE I cohort.
- `openclose_validation/`: Paired-design validation on IHB and Beijing cohorts.
- `simulation_validation/`: FPR calibration, power analysis, and sensitivity audits.
- `notebooks/`: Jupyter tutorials for getting started.
- `benchmarks/`: Scalability and timing benchmarks.
