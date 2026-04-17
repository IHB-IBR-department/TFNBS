
# ConnInfPy — Connectivity Inference in Python

``conninfpy`` (formerly ``tfnbs``)

<!--- [pypi version] -->
![size](https://img.shields.io/github/repo-size/IHB-IBR-department/TFNBS)
![license](https://img.shields.io/github/license/IHB-IBR-department/TFNBS)
![release](https://img.shields.io/github/v/release/IHB-IBR-department/TFNBS)
![last-commit](https://img.shields.io/github/last-commit/IHB-IBR-department/TFNBS)

ConnInfPy is a Python package for statistical inference on brain connectivity networks (fMRI, EEG). It provides
permutation-based tests for group comparisons (t-test) and continuous predictors with confound regression (GLM with
Freedman-Lane), plus a family of enhancement methods that include classical NBS, TFNBS (threshold-free
cluster enhancement adapted for networks), cNBS, network-informed TFNBS, and functional-block clustering TFNBS. The
core idea behind TFNBS is to eliminate the need for an arbitrary statistical threshold by integrating cluster
statistics across a range of thresholds. Implementations are vectorised, support multiprocessing, and can be
accelerated with GPD/gamma tail approximation.


![Overview](https://github.com/IHB-IBR-department/TFNBS/blob/main/docs/Figure_Overview.png)

## Installation

```bash
pip install -e .
```

For development dependencies:

```bash
pip install -e ".[dev]"
```

## Documentation

For more information on features, modules and usage, see the [official documentation](https://IHB-IBR-department.github.io/TFNBS/).

## Permutation p-values (`compute_p_val`)

The main entry point for permutation-based inference is `conninfpy.pairwise_stats.compute_p_val`.

Supported `method` values:
- `tstat`: raw t-statistics (max-stat correction)
- `tfnbs`: threshold-free NBS / TFCE-style enhancement
- `nbs`: classical NBS with fixed threshold (`nbs_stat="extent"` or `"intensity"`)
- `cnbs`: constrained NBS (requires `net_labels`)
- `ni_tfnbs`: network-informed TFNBS (requires `net_labels`)
- `fbc_tfnbs`: functional-block clustering TFNBS (requires `net_labels`)

Minimal example (two-sample):
```python
from conninfpy.pairwise_stats import compute_p_val
from conninfpy.utils import fisher_r_to_z

# group1, group2: arrays of shape (n_subjects, N, N), symmetric, diagonal=0
group1_z = fisher_r_to_z(group1)
group2_z = fisher_r_to_z(group2)

p_vals = compute_p_val(
    group1_z,
    group2_z,
    n_permutations=1000,
    test_type="two-sample",
    method="tfnbs",
    use_mp=True,
)
```

Notes:
- Constrained methods (`cnbs`, `ni_tfnbs`, `fbc_tfnbs`) require `net_labels: ndarray[int]` of shape `(N,)`.
- NBS uses `threshold` and `nbs_stat`; TFNBS-family uses `e`, `h`, `n`, and `start_thres` (plus `min_cluster_size` for `fbc_tfnbs`).

## GLM inference (continuous predictors + confounds)

```python
from conninfpy import compute_p_val_glm, fisher_r_to_z

Y = fisher_r_to_z(connectivity_matrices)   # (n_subjects, N, N)
p_vals = compute_p_val_glm(
    Y, interest=age, confounds=motion,
    method="tfnbs", n_permutations=1000,
)
# p_vals['positive'] / p_vals['negative']
```

## Synthetic method-comparison example

See `examples/sim_method_comparisons/sim_method_comparisons.py`, which compares all methods via `compute_p_val` and
saves GT/t-stat/`1-p` heatmaps:

```bash
python examples/sim_method_comparisons/sim_method_comparisons.py --all-scenarios --effect-size 0.25 --time-points 50 --n-permutations 50
```


## Citing the toolbox
To cite the toolbox: [doi]() and refer to the paper [paper_doi]()
```base
    [doi]
```

For further discussions or reports on bugs, please contact [ashish@ireddy.ru]()
