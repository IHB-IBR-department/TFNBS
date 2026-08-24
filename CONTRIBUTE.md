For developers information.


## Installation

```bash
# Until the PyPI release is out, install from a source checkout instead:
git clone https://github.com/IHB-IBR-department/ConnInfPy.git
cd ConnInfPy

# Create the conda env (Python 3.11)
conda create -n conninfpy python=3.11 -y
conda activate conninfpy
python -m pip install .
python -m pip install -r requirements-dev.txt
```

---

## Running the tests

The test suite requires the development dependencies (pyyaml, scikit-learn,
pytest, …), which are not pulled in by the base install:

Then run the suite:

```bash
python -m unittest discover -s tests -t .
```

> **Note:** a few tests depend on local-only fixtures that are not distributed
> with the repo (both gitignored): `datasets/eeg_dataframe_nansfilled.csv`
> (EEG reshape round-trip) and a `.env` with an `OPENROUTER_API_KEY`
> (dotenv loading). On a fresh clone those specific tests error out —
> the rest of the suite runs green.

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
