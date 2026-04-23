# Examples

New to ConnInfPy? Start with the numbered tutorial series in [`notebooks/`](notebooks/) — open [`01_quickstart_ttest.ipynb`](notebooks/01_quickstart_ttest.ipynb) first.

## Tutorial notebooks

Each notebook is self-contained, runs on synthetic data (or bundled demo data for 07/EEG), and targets ≲ 1 minute runtime.

| # | Notebook | Teaches |
|---|---|---|
| 01 | [`01_quickstart_ttest.ipynb`](notebooks/01_quickstart_ttest.ipynb) | Minimum viable TFNBS pipeline end-to-end |
| 02 | [`02_enhancement_methods.ipynb`](notebooks/02_enhancement_methods.ipynb) | `tstat` / `nbs` / `tfnbs` / `cnbs` / `ni_tfnbs` / `fbc_tfnbs` side-by-side |
| 03 | [`03_glm_inference.ipynb`](notebooks/03_glm_inference.ipynb) | Continuous predictor + confound partialling via Freedman-Lane |
| 04 | [`04_acceleration.ipynb`](notebooks/04_acceleration.ipynb) | `acceleration='gpd'` — 10× fewer perms, same answer |
| 05 | [`05_multi_param_sweep.ipynb`](notebooks/05_multi_param_sweep.ipynb) | List-valued `e`/`h` TFCE exponent sensitivity |
| 06 | [`06_topology_gallery.ipynb`](notebooks/06_topology_gallery.ipynb) | Visual atlas of `conninfpy.topologies` scenarios |
| 07 | [`07_two_task_example.ipynb`](notebooks/07_two_task_example.ipynb) | Real TMFC dataset; TFNBS vs MATLAB NBS/FDR references |
| 08 | [`08_eeg_example.ipynb`](notebooks/08_eeg_example.ipynb) | Real paired EEG comparison (177 subjects) |

## Layout

| Directory | What's in it |
|---|---|
| [`notebooks/`](notebooks/) | Narrated Jupyter notebooks — recommended starting point |
| [`benchmarks/`](benchmarks/) | Performance characterization scripts (per-permutation timing, GPD acceleration, GLM pipeline, sums fast path) |
| [`abide_validation/`](abide_validation/) | Real-data validation on ABIDE I (871 subjects, Schaefer 100): ComBat harmonization + naive / GLM / severity / method comparison / acceleration / within-site replication |
| [`openclose_validation/`](openclose_validation/) | Open-Close validation (IHB + China Schaefer-200, 182 ROIs): paired TFNBS, cross-cohort agreement, cohort-as-site ComBat, bidirectional ML transfer from p-value maps |
| [`reproducibility_exp/`](reproducibility_exp/) | **Legacy** (pre-rename / pre-dataset-migration). References `ml_transfer.openclose_loader.OpenCloseDataset.hcp("rmet")`, an API that no longer exists. Kept for history; ports to §3.11.2 split-half should live under `abide_validation/` or `openclose_validation/`. |
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
