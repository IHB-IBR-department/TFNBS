# TODO: Synthetic Data Testing (Topology + Parameter Sweeps)

## Goal
Provide a repeatable procedure to compare `compute_p_val(..., method=...)` methods on **synthetic symmetric connectomes**, with controlled topology and tunable difficulty.

Main things to stress-test:
- Effect size sensitivity (`--effect-size`)
- Sampling noise sensitivity (`--time-points`, `--n-samples`, `--noise-level`)
- Method parameter sensitivity:
  - TFNBS-family: `--e`, `--h`, `--n-thresholds`, `--start-thres`, `--fbc-min-cluster`
  - NBS: `--nbs-threshold` (script reports both extent+intensity)
- Runtime scaling: `--n-permutations`, `--use-mp`, `--n-processes`

Outputs are saved as images in `examples/output/` (or custom `--output-dir`), one image per topology scenario.

---

## Scripts used

### 1) Topology gallery (GT + t-stat only)
`examples/sim_topology_examples.py`
- Purpose: quick qualitative check that effect size and sampling noise produce realistic t-stat magnitudes.
- Output: ground-truth Δr map and signed t-stat map.

### 2) Method comparison (GT + t-stat + 1-p for each method)
`examples/sim_method_comparisons.py`
- Purpose: compare methods via `compute_p_val` and show TP/FN/FP vs ground-truth topology mask.
- Output: ground-truth Δr, signed t, and `1 - p` maps for:
  - `tstat`, `tfnbs`, `nbs extent`, `nbs intensity`, `cnbs`, `ni_tfnbs`, `fbc_tfnbs`

---

## Quick smoke test (single scenario, fast perms)
Run a single scenario with small permutations to catch obvious regressions:

```bash
python examples/sim_method_comparisons.py \
  --scenarios within_module_dense \
  --effect-size 0.25 \
  --time-points 50 \
  --n-samples 20 \
  --n-permutations 20 \
  --output-dir examples/output
```

---

## Full topology sweep (all scenarios)
Run all built-in topologies (this is the closest to a “regression suite”):

```bash
python examples/sim_method_comparisons.py \
  --all-scenarios \
  --effect-size 0.25 \
  --time-points 50 \
  --n-samples 20 \
  --n-permutations 50 \
  --output-dir examples/output
```

If you want to avoid mixing outputs from different experiments, use a dedicated directory:

```bash
python examples/sim_method_comparisons.py \
  --all-scenarios \
  --effect-size 0.25 \
  --time-points 50 \
  --n-samples 20 \
  --n-permutations 50 \
  --output-dir examples/output/method_compare_es0p25_T50_perm50
```

---

## Effect-size sweep (difficulty sweep)
Typical sweep values (weak → strong):
- `0.10`, `0.15`, `0.20`, `0.25`, `0.30`

Example (same scenario, multiple runs):

```bash
python examples/sim_method_comparisons.py --scenarios within_module_dense --effect-size 0.15 --time-points 50 --n-permutations 50
python examples/sim_method_comparisons.py --scenarios within_module_dense --effect-size 0.20 --time-points 50 --n-permutations 50
python examples/sim_method_comparisons.py --scenarios within_module_dense --effect-size 0.25 --time-points 50 --n-permutations 50
python examples/sim_method_comparisons.py --scenarios within_module_dense --effect-size 0.30 --time-points 50 --n-permutations 50
```

Recommended interpretation:
- As `effect_size` decreases, methods should lose power differently depending on topology.
- FP should stay controlled (but expect some “leakage” because the generator enforces SPD).

---

## Sampling-noise sweep (time-points and n-samples)
Two knobs strongly control t-stat magnitude:
- `--time-points` (per-subject time series length)
- `--n-samples` (subjects per group)

Example:

```bash
python examples/sim_method_comparisons.py --scenarios chain --effect-size 0.25 --time-points 30 --n-samples 20 --n-permutations 50
python examples/sim_method_comparisons.py --scenarios chain --effect-size 0.25 --time-points 50 --n-samples 20 --n-permutations 50
python examples/sim_method_comparisons.py --scenarios chain --effect-size 0.25 --time-points 80 --n-samples 20 --n-permutations 50
```

Use `examples/sim_topology_examples.py` if you only want to inspect GT vs t-stat quickly:

```bash
python examples/sim_topology_examples.py --scenarios chain --effect-size 0.25 --time-points 30 --n-samples 20 --report
```

---

## TFNBS parameter sweep (e/h/n/start_thres)
TFNBS-family parameters in `examples/sim_method_comparisons.py`:
- `--e` (extent exponent)
- `--h` (height exponent)
- `--n-thresholds` (integration steps)
- `--start-thres` (minimum threshold)

Example: compare two TFNBS settings on a topology where TFNBS should help (gradient effect):

```bash
python examples/sim_method_comparisons.py \
  --scenarios gradient_effect_chain \
  --effect-size 0.25 --time-points 50 --n-samples 20 --n-permutations 50 \
  --e 0.5 --h 2.0 --n-thresholds 50 --start-thres 1.65

python examples/sim_method_comparisons.py \
  --scenarios gradient_effect_chain \
  --effect-size 0.25 --time-points 50 --n-samples 20 --n-permutations 50 \
  --e 1.0 --h 2.0 --n-thresholds 50 --start-thres 2.0
```

Guidance:
- Higher `start_thres` makes TFNBS-family behave more “NBS-like” (only strong edges contribute).
- Increasing `n-thresholds` improves integration fidelity but increases runtime (slightly).

---

## NBS threshold sweep
NBS is very sensitive to `--nbs-threshold`:

```bash
python examples/sim_method_comparisons.py --scenarios within_module_dense --effect-size 0.25 --time-points 50 --n-permutations 50 --nbs-threshold 1.5
python examples/sim_method_comparisons.py --scenarios within_module_dense --effect-size 0.25 --time-points 50 --n-permutations 50 --nbs-threshold 2.0
python examples/sim_method_comparisons.py --scenarios within_module_dense --effect-size 0.25 --time-points 50 --n-permutations 50 --nbs-threshold 3.0
```

Interpretation:
- Too low threshold: giant components, higher FP.
- Too high threshold: fragmented components, higher FN.

---

## FBC-TFNBS sensitivity (`min_cluster_size`)
FBC is meant to boost “many edges in the same functional block” even if not topologically connected.

Use a topology where this matters:
- `perfect_matching_within_module` (many edges in a block, no shared nodes)

```bash
python examples/sim_method_comparisons.py \
  --scenarios perfect_matching_within_module \
  --effect-size 0.25 --time-points 50 --n-samples 20 --n-permutations 50 \
  --fbc-min-cluster 2

python examples/sim_method_comparisons.py \
  --scenarios perfect_matching_within_module \
  --effect-size 0.25 --time-points 50 --n-samples 20 --n-permutations 50 \
  --fbc-min-cluster 6
```

---

## Runtime testing (permutations + multiprocessing)
For faster runs on large settings:
- increase `--n-permutations` only when needed
- enable multiprocessing:
  - `--use-mp`
  - `--n-processes <k>`

Example:

```bash
python examples/sim_method_comparisons.py \
  --scenarios within_module_dense chain cross_block_connected_chain \
  --effect-size 0.25 --time-points 50 --n-samples 20 \
  --n-permutations 200 \
  --use-mp --n-processes 8
```

If you want a simple wall-time measurement:

```bash
time python examples/sim_method_comparisons.py --scenarios within_module_dense --effect-size 0.25 --time-points 50 --n-permutations 200
```

---

## Suggested “method-discriminative” scenario set
If you don’t want to run all scenarios, run these first:
- `within_module_dense` (easy block effect)
- `perfect_matching_within_module` (block-coherent, topologically disconnected)
- `cross_block_connected_chain` (topologically connected, block-fragmented)
- `gradient_effect_chain` (heterogeneous effect strengths)
- `scattered_cross_block` (FP stress test)

Example:

```bash
python examples/sim_method_comparisons.py \
  --scenarios within_module_dense perfect_matching_within_module cross_block_connected_chain gradient_effect_chain scattered_cross_block \
  --effect-size 0.25 --time-points 50 --n-samples 20 --n-permutations 50
```

---

## Notes / interpretation caveats
- Ground truth is defined by the **effect mask** used in generation, but the generator enforces SPD, so some off-mask edges may change slightly (expect some FP even for “good” methods).
- Reported TP/FN/FP in figures are computed on the **upper triangle** only (then visualized symmetrically).
- `cnbs/ni_tfnbs/fbc_tfnbs` depend on `net_labels`; changing module granularity changes their behavior (especially FBC `min_cluster_size`).

