# Benchmark results

CSV outputs from `examples/benchmarks/*.py`. Regeneratable — gitignored.

Files produced when you run a benchmark with `--output` (or default):

| File | Source |
|---|---|
| `timing_benchmark.csv` | `examples/benchmarks/timing_benchmark.py` |
| `precompsum.csv` | `examples/benchmarks/benchmark_precompsum.py` (paired / two-sample / enhancement cases) |
| `precompsum_composed.csv` | same, composed (pre-comp × GPD) case |
| `acceleration.csv` | `examples/benchmarks/benchmark_acceleration.py` |
| `glm_stat.csv` + `glm_pipeline.csv` | `examples/benchmarks/benchmark_glm.py` |

Run:
```bash
python examples/benchmarks/benchmark_precompsum.py --quick
python examples/benchmarks/benchmark_precompsum.py --output examples/benchmarks/res_benchmarks/precompsum.csv
```

Each CSV has columns for the grid (N, n_sub, n_perm, topology, method, etc.) and timing metrics. Can be loaded with `pandas.read_csv()` for analysis or plotting.
