
"""Extract ENIGMA group-mean structural-connectivity matrices into the
repository for the structural-prior validation pipeline.

Run this script ONCE under an environment with `enigmatoolbox` installed
(e.g. the `fmri-fitting` conda env on this machine); the resulting
``.npy`` + ``.json`` sidecar pair is committed to the repo so that
downstream validation does not require enigmatoolbox at runtime.

Usage
-----
    conda activate fmri-fitting
    python examples/simulation_validation/sc_prior/extract_enigma_sc.py

Outputs (under ``datasets/atlases/sc/``):
    sc_schaefer100_hcp.npy        # (100, 100) cortical SC, symmetric
    sc_schaefer100_hcp.labels.txt # (100,) ROI labels, one per line
    sc_schaefer100_hcp.json       # provenance sidecar
    sc_schaefer200_hcp.npy        # secondary target (resolution check)
    sc_schaefer200_hcp.labels.txt
    sc_schaefer200_hcp.json
    sc_schaefer400_hcp.npy        # secondary target
    sc_schaefer400_hcp.labels.txt
    sc_schaefer400_hcp.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _out_dir() -> Path:
    return _root() / "datasets" / "atlases" / "sc"


def _symmetrize_zero_diag(M: np.ndarray) -> np.ndarray:
    M = (M + M.T) / 2.0
    np.fill_diagonal(M, 0.0)
    return M


def extract_one(parcellation: str) -> dict:
    """Pull one Schaefer-N SC matrix from ENIGMA and write {.npy, .labels.txt, .json}.

    Returns a small report dict for the master extraction log.
    """
    from enigmatoolbox import __version__ as enigma_version
    from enigmatoolbox.datasets import load_sc

    sc, labels, _sctx, _sctx_labels = load_sc(parcellation=parcellation)
    sc = np.asarray(sc, dtype=np.float64)
    sc = _symmetrize_zero_diag(sc)
    labels = [str(x) for x in labels]

    tag = parcellation.replace("schaefer_", "schaefer")  # 'schaefer100'
    base = _out_dir() / f"sc_{tag}_hcp"

    base.parent.mkdir(parents=True, exist_ok=True)
    np.save(base.with_suffix(".npy"), sc)
    base.with_suffix(".labels.txt").write_text("\n".join(labels) + "\n")

    weights = sc[np.triu_indices_from(sc, k=1)]
    nonzero = weights[weights > 0]
    report = {
        "parcellation": parcellation,
        "n_nodes": int(sc.shape[0]),
        "enigmatoolbox_version": enigma_version,
        "source": "ENIGMA Toolbox load_sc() — HCP-derived group mean",
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "weight_mean": float(weights.mean()),
        "edge_density_nonzero": float((nonzero.size) / max(1, weights.size)),
        "isolated_nodes": int(np.sum(sc.sum(axis=0) == 0.0)),
        "files": {
            "matrix": str(base.with_suffix(".npy").relative_to(_root())),
            "labels": str(base.with_suffix(".labels.txt").relative_to(_root())),
            "sidecar": str(base.with_suffix(".json").relative_to(_root())),
        },
    }
    base.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    reports = []
    for parc in ("schaefer_100", "schaefer_200", "schaefer_400"):
        print(f"[extract] {parc} ...")
        report = extract_one(parc)
        print(
            f"  → N={report['n_nodes']}, "
            f"weight∈[{report['weight_min']:.3g},{report['weight_max']:.3g}], "
            f"density={report['edge_density_nonzero']:.3f}, "
            f"isolated={report['isolated_nodes']}"
        )
        reports.append(report)

    summary_path = _out_dir() / "extraction_summary.json"
    summary_path.write_text(json.dumps(reports, indent=2) + "\n")
    print(f"\nSummary written to {summary_path.relative_to(_root())}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
