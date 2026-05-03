# ConnInfPy — Connectivity Inference in Python

[![CI - tests](https://img.shields.io/github/actions/workflow/status/IHB-IBR-department/ConnInfPy/documentation.yml?label=docs)](https://ihb-ibr-department.github.io/ConnInfPy/)
![license](https://img.shields.io/github/license/IHB-IBR-department/ConnInfPy)
![release](https://img.shields.io/github/v/release/IHB-IBR-department/ConnInfPy)
![last-commit](https://img.shields.io/github/last-commit/IHB-IBR-department/ConnInfPy)
![size](https://img.shields.io/github/repo-size/IHB-IBR-department/ConnInfPy)

**A unified Python framework for permutation-based statistical inference on brain connectivity networks (fMRI, EEG).**

ConnInfPy implements a single permutation engine shared across nine inference
methods, with edge-wise GLM (Freedman–Lane), confound-aware cross-site
harmonization (parametric empirical-Bayes ComBat), and tail-approximation
acceleration — all under one Python API.

What you get out of one `pip install`:

- **Six topology-aware enhancement operators** — NBS-extent, NBS-intensity,
  TFNBS, cNBS, **NI-TFNBS** (novel — network-informed soft block-density
  prior), **FBC-TFNBS** (novel — hard block-prior with minimum cluster
  size) — plus three baselines (per-edge $t$, Bonferroni, BH-FDR), all
  sharing one permutation engine and the +1 Phipson–Smyth correction.
- **Edge-wise GLM with Freedman–Lane permutation** — continuous predictors,
  confound regression, $t$/$\beta$/F-contrast statistics, paired
  within-subject designs with Δ-level confounds, two-tailed FWER with
  positive/negative directional split.
- **GPD/gamma tail acceleration** — 200-perm runs reproduce 5000-perm
  empirical FWER p-values on real data to within $|\Delta(-\log_{10}p)| \le 0.001$
  on >99% of edges (≈25× wall-clock saving), with Anderson–Darling
  goodness-of-fit guard and empirical fallback.
- **In-package multi-site harmonization** — parametric empirical-Bayes
  ComBat (Johnson 2007; Fortin 2017/2018) reimplemented in NumPy, with
  separate `combat_fit`/`combat_apply` for cross-site ML transfer and a
  `design_diagnostics` layer (VIF + condition number + plain-English flags).
- **19-scenario topology benchmark library** — controlled effect topologies
  (hub, rich-club, chain, scattered, gradient, fragmented within-module,
  …) used in the paper's no-method-dominates-across-topologies finding.

![Overview](https://github.com/IHB-IBR-department/ConnInfPy/blob/main/docs/Figure_Overview.png)

> **History.** Originally developed as a TFNBS-only implementation
> (`tfnbs`); renamed and substantially expanded in 2026-04 to the
> unified framework presented here. The old GitHub URL
> `https://github.com/IHB-IBR-department/TFNBS` redirects automatically.

---

## Installation

```bash
# Create the conda env (Python 3.11)
conda create -n conninfpy python=3.11 -y
conda activate conninfpy

# Install the package in development mode
pip install -e .

# With development dependencies (sphinx, notebook tools, optional pytest runner)
pip install -e ".[dev]"
```

## Running the tests

```bash
python -m unittest discover -s tests -t .
```

The suite uses Python's standard `unittest` (no pytest required). Per-module or
per-class runs:

```bash
python -m unittest tests.test_glm_stats
python -m unittest tests.test_glm_stats.TestFStatCompute
python -m unittest tests.test_glm_stats.TestFStatCompute.test_fstat_single_row_equals_tstat_squared
```

## Building the docs

```bash
cd docs
sphinx-build source _build
# Docs auto-build on push to main via GitHub Actions → gh-pages.
```

---

## What's in the package

### Statistical inference

| Function | Purpose | Design |
|---|---|---|
| `compute_p_val` | Permutation p-values for group comparisons | Two-sample, paired, or one-sample |
| `compute_p_val_glm` | GLM with confound regression | Continuous predictors + nuisance, Freedman-Lane permutation; supports 1D contrast (t-stat/beta) and 2D contrast (omnibus F-test) |
| `compute_p_val_paired_glm` | Paired A vs B with Δ-level confounds | Convenience wrapper — routes to paired-t when no confounds, else one-sample GLM on Δ |
| `compute_null_dist` | Generate null distribution only | For custom workflows |
| `compute_t_stat` / `compute_t_stat_diff` | Edge-wise t-statistics | Paired / one-sample / two-sample |
| `compute_glm_stat` | Edge-wise GLM statistic | `stat_type ∈ {tstat, beta, fstat}` |
| `build_design_matrix` | Convenience builder for `X` + contrast | Interest + confounds → `[1, C, interest]` |

### Enhancement methods (shared between t-test and GLM pipelines)

| Method string | Description | Required args |
|---|---|---|
| `tstat` | Raw t-statistics, max-stat correction | — |
| `tfnbs` | Threshold-free cluster enhancement for networks | `e, h, n, start_thres` |
| `nbs` | Classical NBS with fixed threshold | `threshold, nbs_stat` |
| `cnbs` | Constrained NBS (block-level aggregation) | `net_labels` |
| `ni_tfnbs` | Network-informed TFNBS (spatial priors) | `net_labels` |
| `fbc_tfnbs` | Functional-block clustering TFNBS | `net_labels, min_cluster_size` |
| `bonferroni` / `bh_fdr` | Parametric baselines (no permutation) | — |
| `bh_fdr_perm` | Permutation-based BH-FDR | — |

Enhancement can also be applied standalone (no permutation) via
`apply_tfnbs`, `apply_nbs`, `apply_cnbs`, `apply_ni_tfnbs`, `apply_fbc_tfnbs`.

### Permutation acceleration (Winkler et al. 2016)

| Function | Purpose |
|---|---|
| `fit_gpd_tail` | GPD tail approximation — 10–25× speedup |
| `fit_gamma_tail` | Gamma (Pearson type III) approximation |
| `compute_p_values_accelerated` | Drop-in replacement for empirical p-values |

Integrated via `acceleration='gpd'|'gamma'` in both `compute_p_val` and
`compute_p_val_glm` (~200 permutations instead of ~5000).

### Multi-site harmonization & design diagnostics (`conninfpy.harmonize`)

Parametric empirical-Bayes ComBat (Johnson 2007) implemented in pure numpy —
no `neuroHarmonize` or `neurocombat` dependency.

| Function | Purpose |
|---|---|
| `combat_harmonize(Y, sites, preserve=None)` | Fit + transform in one call; returns `CombatResult` with `Y_adjusted`, fitted `model`, and between-site variance diagnostics |
| `combat_fit` / `combat_apply` | Separate fit and apply — for cross-site ML transfer (fit on training sites, apply to held-out subjects) |
| `compute_vif(X)` | Variance inflation factor per design-matrix column |
| `design_diagnostics(X, names=None)` | Condition number, VIF, pairwise correlation, plain-English flags |
| `flatten_upper` / `unflatten_upper` | Vectorise / un-vectorise the upper triangle of `(n, N, N)` connectivity |

Accepts either `(n, N, N)` connectivity matrices or pre-flattened `(n, p)`
features.

### Synthetic data + topology library

| Function / class | Purpose |
|---|---|
| `generate_fc_matrices` | Modular network with controlled effect size |
| `ModularDatasetGenerator` | Class-based generator for modular network structures |
| `TopologyDatasetGenerator`, `get_scenario`, `list_scenarios` | 19+ canonical scenarios (hub, chain, rich_club, within/between-module, gradient, core-periphery, …) for methods benchmarking |

### Core primitives & utilities

- `tfnbs_score.py`: `get_tfnbs_score`, `get_network_informed_tfnbs_score`, `get_fbc_tfnbs_score` — the underlying scoring functions.
- `nbs_score.py`: `nbs_bct` — classical NBS reference.
- `utils.py`: `fisher_r_to_z`, `fisher_z_to_r`, `get_components`, `binarize`.
- `eeg_utils.py`: EEG-specific data structures (`EEGData`, `Electrodes`, `Bands`, `PairsElectrodes1020`) and helpers.

---

## Minimal usage

### Two-sample permutation (t-test pipeline)

```python
from conninfpy import compute_p_val, fisher_r_to_z

group1_z = fisher_r_to_z(group1)   # (n1, N, N), symmetric, zero diagonal
group2_z = fisher_r_to_z(group2)

p_vals = compute_p_val(
    group1_z, group2_z,
    test_type="two-sample",
    method="tfnbs",
    n_permutations=1000,
    e=0.4, h=3.0, n=10,
    use_mp=True,
)
# → {'g2>g1': (N, N), 'g1>g2': (N, N)}
```

### GLM with continuous predictor and confounds

```python
from conninfpy import compute_p_val_glm, fisher_r_to_z

Y = fisher_r_to_z(connectivity_matrices)     # (n_subjects, N, N)
p_vals = compute_p_val_glm(
    Y, interest=age, confounds=motion,
    method="tfnbs", n_permutations=5000,
)
# → {'positive': (N, N), 'negative': (N, N)}
```

### Paired design with difference-level confound (new)

```python
from conninfpy import compute_p_val_paired_glm

# Y_A, Y_B: aligned (n_subjects, N, N); fd_A, fd_B: per-condition motion
p_vals = compute_p_val_paired_glm(
    Y_A, Y_B,
    confounds_A=fd_A, confounds_B=fd_B,  # None to skip → delegates to paired t-test
    method="tfnbs", n_permutations=5000,
)
# Tests A vs B within-subject, partialling out Δmotion = fd_A − fd_B.
```

### Omnibus F-contrast for ≥3 conditions (new)

```python
import numpy as np
from conninfpy import compute_p_val_glm

# 3-group dummy coding: intercept + group_B + group_C (group_A = reference)
X = np.column_stack([np.ones(n), group_B, group_C])
# Joint test β_B = β_C = 0
C = np.array([[0, 1, 0], [0, 0, 1]])

p_vals = compute_p_val_glm(
    Y, design_matrix=X, contrast=C,
    stat_type="fstat",
    method="tfnbs", n_permutations=5000,
)
# → {'omnibus': (N, N)}   -- F is non-negative, no sign to split
```

### Multi-site harmonization (new)

```python
from conninfpy import combat_harmonize, design_diagnostics
import numpy as np

# Harmonize site-aligned variance while preserving biological covariates
result = combat_harmonize(
    Y, sites=site_labels,
    preserve=np.column_stack([age, sex, diagnosis]),
)
Y_adjusted = result.Y_adjusted
print(result.diagnostics)   # between-site variance reduction, per-site n, ...

# Audit your design matrix before running the GLM
X = np.column_stack([np.ones(n), age, motion, *site_dummies])
report = design_diagnostics(X, names=["intercept", "age", "fd", "site_1", "site_2"])
for flag in report["flags"]:
    print("⚠️ ", flag)
```

### Acceleration (fewer permutations, same FWER)

```python
p_vals = compute_p_val_glm(
    Y, interest=age, confounds=motion,
    method="tfnbs", n_permutations=200,
    acceleration="gpd",   # ~25× speedup; also 'gamma'
)
```

---

## Tutorials and examples

| Path | What |
|---|---|
| [examples/notebooks/](examples/notebooks/) | 8-notebook tutorial series: quickstart, enhancement methods, GLM, acceleration, multi-param sweep, topology gallery, two-task example, EEG |
| [examples/benchmarks/](examples/benchmarks/) | Timing / GLM / acceleration / precompsum benchmarks with CSV output and `plot_results.py` |
| [examples/miccai_paper_reproducing/](examples/miccai_paper_reproducing/) | Simulation validation (FPR + power) backing the MICCAI 2026 submission |
| [examples/abide_validation/](examples/abide_validation/) | Real-data validation on ABIDE (age, diagnosis, motion, within-site replication) |
| [examples/ml_transfer/](examples/ml_transfer/) | Open-Close bidirectional ML transfer (IHB ↔ China) |
| [examples/sim_method_comparisons/](examples/sim_method_comparisons/) | Side-by-side method comparison on synthetic data |

Single-command method-comparison example:

```bash
python examples/sim_method_comparisons/sim_method_comparisons.py \
    --all-scenarios --effect-size 0.25 --time-points 50 --n-permutations 50
```

---

## Documentation

Full reference at [IHB-IBR-department.github.io/ConnInfPy](https://IHB-IBR-department.github.io/ConnInfPy/).
The docs auto-build on push to `main`.

## Citing the toolbox

To cite the toolbox: [doi]() and refer to the paper [paper_doi]()

```
[doi]
```

For further discussions or to report bugs, please contact [knyazeva@ihb.spb.ru](mailto:knyazeva@ihb.spb.ru) or open an issue at https://github.com/IHB-IBR-department/ConnInfPy/issues.
