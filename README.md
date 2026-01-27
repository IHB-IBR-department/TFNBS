
# Threshold-Free Network-Based Statistics in Neuroscience

``TFNBS`` 

<!--- [pypi version] -->
![size](https://img.shields.io/github/repo-size/IHB-IBR-department/TFNBS)
![license](https://img.shields.io/github/license/IHB-IBR-department/TFNBS)
![release](https://img.shields.io/github/v/release/IHB-IBR-department/TFNBS)
![last-commit](https://img.shields.io/github/last-commit/IHB-IBR-department/TFNBS)
![TFNBS](https://img.shields.io/github/downloads/IHB-IBR-department/TFNBS/total)

TFNBS Toolbox is a Python package for computation and generation of network-based statistics for neuroscience data 
(i.e. fMRI, EEG data). The core concept is based on eliminating the use of a hardcoded threshold using threshold-free 
cluster enhancement (TFCE) scores to assess statistical significance. It works on the principle of networks, where TFCE 
statistical values are computed across a range of thresholds over n cycles of permutations to uncover possible significance in the data. 
Our implementation of TFNBS follows efficient computing and allows for computations to be performed on parallel cores therefore 
massively reducing computation time and resources. 


![Overview of TFNOS](https://github.com/IHB-IBR-department/TFNBS/blob/main/docs/Figure_Overview.png)

## Installation 
TFNOS toolbox can be installed using: 

```bash
    !pip install tfnbs
```

## Documentation

For more information on TFNBS's features, modules and usage, please refer to the [official documentation](https://IHB-IBR-department.github.io/TFNBS/). 

Examples of usage on fMRI and EEG data are avaialble in [notebooks]() and [data]().

## Project structure

```
TFNBS/
  tfnbs/                         # core library modules (scores, stats, utils)
  tests/                         # unit tests (unittest)
  syntetic_experiments/          # synthetic data experiments and scripts
    generate_topology_example.py
    sim_topology_examples.py
    sim_method_comparisons/
    power_test/
    fpr/
    output/
  datasets/                      # sample data files
  docs/                          # Sphinx docs source
```

## Synthetic experiments

Generate a single topology dataset (saves z-matrices, effect mask, t-stats, and a PNG visualization):
```bash
python syntetic_experiments/generate_topology_example.py \
  --topology within_module_dense \
  --effect-size 0.25 \
  --n-samples 20 \
  --output-dir syntetic_experiments/output/my_data
```

Generate multiple datasets from a config:
```bash
python syntetic_experiments/generate_topology_example.py \
  --config syntetic_experiments/topology_config.yaml
```

Method comparison sweep (quick config available in the folder):
```bash
python syntetic_experiments/sim_method_comparisons/sim_method_comparisons.py \
  --all-scenarios --effect-size 0.25 --time-points 50 --n-permutations 50
```

Power analysis sweep:
```bash
python syntetic_experiments/power_test/power_analysis.py \
  --config syntetic_experiments/power_test/sweep_config_power_quick.yaml
```

FPR calibration:
```bash
python syntetic_experiments/fpr/fpr_calibration.py \
  --config syntetic_experiments/fpr/fpr_config_quick.yaml
```

## Permutation p-values (`compute_p_val`)

The main entry point for permutation-based inference is `tfnbs.pairwise_stats.compute_p_val`.

Supported `method` values:
- `tstat`: raw t-statistics (max-stat correction)
- `tfnbs`: threshold-free NBS / TFCE-style enhancement
- `nbs`: classical NBS with fixed threshold (`nbs_stat="extent"` or `"intensity"`)
- `cnbs`: constrained NBS (requires `net_labels`)
- `ni_tfnbs`: network-informed TFNBS (requires `net_labels`)
- `fbc_tfnbs`: functional-block clustering TFNBS (requires `net_labels`)

Minimal example (two-sample):
```python
from tfnbs.pairwise_stats import compute_p_val
from tfnbs.utils import fisher_r_to_z

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

## Synthetic method-comparison example

See `syntetic_experiments/sim_method_comparisons/sim_method_comparisons.py` which compares all methods via `compute_p_val` and saves GT/t-stat/`1-p` heatmaps:
```bash
python syntetic_experiments/sim_method_comparisons/sim_method_comparisons.py --all-scenarios --effect-size 0.25 --time-points 50 --n-permutations 50
```


## Citing the toolbox 
To cite the toolbox, you can use: [doi]() and refer to the paper [paper_doi]()
```base
    [doi]
```

For further discussions or reports on bugs, please contact [ashish@ireddy.ru]()



