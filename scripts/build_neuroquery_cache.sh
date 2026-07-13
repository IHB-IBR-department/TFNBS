#!/usr/bin/env bash
# Build the reusable compact NeuroQuery NiMARE cache for ConnInfPy.

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda_env="${CONNINFPY_ENV:-conninfpy}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/conninfpy-mpl}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/conninfpy-numba}"
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"

if ! command -v conda >/dev/null 2>&1; then
    echo "Conda was not found. Activate the conninfpy environment and rerun."
    exit 1
fi

cd "$project_root"

conda run --no-capture-output -n "$conda_env" python -c '
from conninfpy._decode_cache import fetch_neuroquery_dataset, get_cache_dir

cache_path = get_cache_dir() / "neuroquery_dataset.pkl"
if cache_path.exists():
    print(f"NeuroQuery cache already exists: {cache_path}")
    print(f"Size: {cache_path.stat().st_size / 1024**2:.0f} MB")
else:
    print("Building compact NeuroQuery cache from local raw files...", flush=True)
    fetch_neuroquery_dataset()
    print(f"Created: {cache_path}")
    print(f"Size: {cache_path.stat().st_size / 1024**2:.0f} MB")
'
