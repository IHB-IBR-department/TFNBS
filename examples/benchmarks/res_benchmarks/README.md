# Benchmark results

CSV outputs and PNG plots from `examples/benchmarks/*.py`. Regeneratable — gitignored (only this README is tracked).

## Files produced by each benchmark

| CSV | Plot | Source |
|---|---|---|
| `timing_benchmark.csv` | `timing_benchmark.png` | `benchmarks/timing_benchmark.py` |
| `precompsum.csv` | `precompsum.png` | `benchmarks/benchmark_precompsum.py` (paired / two-sample / enhancement) |
| `precompsum_composed.csv` | `precompsum_composed.png` | same, composed (pre-comp × GPD) case |
| `acceleration.csv` | `acceleration.png` | `benchmarks/benchmark_acceleration.py` |
| `glm_stat.csv` | `glm_stat.png` | `benchmarks/benchmark_glm.py` — GLM stat computation |
| `glm_pipeline.csv` | `glm_pipeline.png` | `benchmarks/benchmark_glm.py` — full pipeline with permutation |

## Workflow

```bash
# 1. Run benchmarks (CSVs drop here)
python examples/benchmarks/benchmark_precompsum.py --topologies within_module_dense hub rich_club scattered_cross_block
python examples/benchmarks/timing_benchmark.py
python examples/benchmarks/benchmark_acceleration.py --topologies within_module_dense hub rich_club scattered_cross_block
python examples/benchmarks/benchmark_glm.py --topologies within_module_dense hub rich_club scattered_cross_block

# 2. Generate plots from CSVs
python examples/benchmarks/plot_results.py

# 3. View plots (open in VS Code, preview, or your image viewer)
```

`plot_results.py` accepts `--csv-dir`, `--output-dir`, and `--only NAME [NAME ...]` for subset plotting. Missing CSVs are skipped.

## What each plot shows

- **timing_benchmark**: left — wall-clock bar chart per method × N; right — per-permutation cost log-log (scaling).
- **precompsum**: left — speedup by case × topology; right — speedup vs N averaged over topologies.
- **precompsum_composed**: grouped bar comparing baseline vs +GPD only vs +pre-compute only vs both, per (N, n_sub).
- **acceleration**: left — scatter of accelerated planted-edge p-values against reference (y=x line); right — p-value MAE by method × accel.
- **glm_stat**: log-log throughput (edges/s) vs N.
- **glm_pipeline**: left — wall-clock per method × N; right — heatmap of planted-edge p-values per (topology × method) at the largest N.
