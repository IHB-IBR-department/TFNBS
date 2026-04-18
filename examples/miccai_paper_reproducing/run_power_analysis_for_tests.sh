#!/bin/bash
# run_power_analysis_for_tests.sh — local smoke test of power analysis
#
# Uses the full server YAML config but overrides the expensive parameters
# via CLI flags. CLI flags take priority over YAML values.
#
# Full server config (sweep_config_power_server_short.yaml):
#   n_permutations=1000, n_repeats=100, 10 scenarios, 8 ES, 8 SS
#   → ~325 hours on 32-core server, impossible locally
#
# This script overrides to a quick local test:
#   n_permutations=200, n_repeats=5, 3 scenarios, 4 ES, 4 SS
#   Measured: ~1.4 min per (scenario, ES, repeat) on 16 cores (macOS)
#   effect-size only: 3×4×5 = 60 units → ~84 min (1.4 hours)
#   both modes: ~168 min (2.8 hours)
#
# The (e,h) grid (16×11=176 combos) is FREE thanks to batched 3D mode —
# no need to reduce it.
#
# Cost ranking of parameters (most expensive first):
#   1. n_permutations (1000→200 = 5× faster) — dominates total runtime
#   2. n_repeats      (100→5   = 20× faster) — linear multiplier
#   3. scenarios      (10→3    = 3× faster)  — linear multiplier
#   4. effect_sizes   (8→4     = 2× faster)  — linear multiplier
#   5. sample_sizes   (8→4     = 2× faster)  — linear multiplier
#   6. e/h grid       (176 combos)           — FREE (batched, not a multiplier)
#   7. n_thresholds   (30)                   — minor, inside scoring loop

set -euo pipefail
cd "$(dirname "$0")"

LOGDIR=.log
mkdir -p "$LOGDIR"
LOG="$LOGDIR/TEST_power_$(date +%d%m_%H%M).log"

echo "=== Local Power Analysis Test ==="
echo "Config: sweep_config_power_server_short.yaml + local overrides"
echo "Log: $LOG"
echo ""

# Run in foreground (use nohup ... & for background)
python -u examples/miccai_paper_reproducing/power_test/power_analysis.py \
  --config examples/miccai_paper_reproducing/power_test/sweep_config_power_server_short.yaml \
  --mode effect-size \
  --n-permutations 200 \
  --n-repeats 5 \
  --scenarios within_module_dense chain hub \
  --effect-sizes 0.15 0.25 0.40 0.60 \
  --output-dir examples/miccai_paper_reproducing/results/power_analysis_test \
  --checkpoint-every 5 \
  2>&1 | tee "$LOG"

# To also run sample-size mode, uncomment:
# python -u examples/miccai_paper_reproducing/power_test/power_analysis.py \
#   --config examples/miccai_paper_reproducing/power_test/sweep_config_power_server_short.yaml \
#   --mode sample-size \
#   --n-permutations 200 \
#   --n-repeats 5 \
#   --scenarios within_module_dense chain hub \
#   --sample-sizes 15 25 50 100 \
#   --output-dir examples/miccai_paper_reproducing/results/power_analysis_test \
#   --checkpoint-every 5 \
#   2>&1 | tee -a "$LOG"
