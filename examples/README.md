# Examples

New to ConnInfPy? Start here → [`notebooks/EEG_example.ipynb`](notebooks/EEG_example.ipynb) (real paired-EEG comparison) or [`notebooks/TFNBS_Example.ipynb`](notebooks/TFNBS_Example.ipynb) (TMFC synthetic dataset with MATLAB reference outputs).

## Layout

| Directory | What's in it |
|---|---|
| [`notebooks/`](notebooks/) | Narrated Jupyter notebooks — recommended starting point |
| [`benchmarks/`](benchmarks/) | Performance characterization scripts (per-permutation timing, GPD acceleration, GLM pipeline, sums fast path) |
| [`abide_validation/`](abide_validation/) | Real-data validation on ABIDE I (871 subjects, Schaefer 100): naive / GLM / severity regression / method comparison / acceleration check |
| [`ml_transfer/`](ml_transfer/) | ML transfer workflow using saved p-value maps (HCP IHB / RMET Open-Close) |
| [`reproducibility_exp/`](reproducibility_exp/) | Cross-experiment + split-half reproducibility metrics (Jaccard, Dice, t-map correlation) |
| [`sim_method_comparisons/`](sim_method_comparisons/) | Side-by-side synthetic comparison of all 9 methods on a given topology |
| [`miccai_paper_reproducing/`](miccai_paper_reproducing/) | Scripts + YAML configs to reproduce the MICCAI 2026 paper results |
| `sim_topology_examples.py` | Topology gallery demo (visualizes the ground-truth masks from the `conninfpy.topologies` library) |

## Running notebooks

Install the notebooks extras once:

```bash
pip install -e ".[notebooks]"
python -m ipykernel install --user --name conninfpy --display-name "Python (conninfpy)"
```

Then open any `.ipynb` in VS Code (or JupyterLab) and select the **Python (conninfpy)** kernel.

## Running benchmarks

Each benchmark script is standalone with `--quick` / `--full` flags:

```bash
python examples/benchmarks/benchmark_precompsum.py --quick   # ~30s
python examples/benchmarks/timing_benchmark.py --help
python examples/benchmarks/benchmark_acceleration.py
python examples/benchmarks/benchmark_glm.py
```

Benchmarks exit non-zero when a speed regression is detected, so they're usable as local CI checks (`python examples/benchmarks/benchmark_precompsum.py && echo OK`).

## Workflow scripts (real / realistic data)

The subfolders above the benchmarks contain end-to-end analyses rather than tutorials — take one, adapt it to your dataset. Each has its own README or header docstring explaining inputs/outputs.

## See also

- [`CLAUDE.md`](../CLAUDE.md) — architecture + dev commands
- [`conninfpy/`](../conninfpy/) — package source
- [`tests/`](../tests/) — unit tests (with `tests/fixtures.py` providing named synthetic-data scenarios)
