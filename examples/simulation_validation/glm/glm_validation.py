"""
GLM validation on synthetic connectivity data.

This script adds two GLM-specific validation arms for the toolbox paper:

1. Null FWER calibration for a continuous tested regressor, optionally with a
   nuisance covariate effect included in the data and adjusted in the GLM.
2. Power analysis for a continuous covariate with effects planted in known
   topological edge sets.

The goal is methodological validation of the GLM API, not a new neuroscience
claim. Outputs mirror the existing simulation validation CSV style so the same
calibration logic and downstream plotting conventions can be reused.

Usage:
    python examples/simulation_validation/glm/glm_validation.py \
        --config examples/simulation_validation/configs/glm_config_quick.yaml

    python examples/simulation_validation/glm/glm_validation.py \
        --config examples/simulation_validation/configs/glm_config_1000.yaml \
        --mode calibration
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_SINGLE_THREAD = "1"
os.environ.setdefault("OMP_NUM_THREADS", _SINGLE_THREAD)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _SINGLE_THREAD)
os.environ.setdefault("MKL_NUM_THREADS", _SINGLE_THREAD)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", _SINGLE_THREAD)
os.environ.setdefault("NUMBA_NUM_THREADS", _SINGLE_THREAD)

import numpy as np
import pandas as pd

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from tqdm import tqdm

from conninfpy import topologies as topo
from conninfpy.glm_stats import compute_p_val_glm
from conninfpy.pairwise_stats import get_available_cores

SIM_ROOT = Path(__file__).resolve().parents[1]
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))
from fpr.fpr_calibration import compute_fpr_summary  # noqa: E402


GLM_METHODS = [
    "tstat",
    "tfnbs",
    "ni_tfnbs",
    "fbc_tfnbs",
    "nbs_extent_2.0",
    "nbs_intensity_2.0",
    "nbs_extent_3.0",
    "nbs_intensity_3.0",
    "cnbs",
]


@dataclass
class GLMConfig:
    """Configuration for GLM validation."""

    methods: List[str]
    mode: str = "all"

    # Null calibration
    n_null: int = 100
    sample_sizes: List[int] = None

    # Power
    n_repeats: int = 20
    n_subjects: int = 60
    effect_sizes: List[float] = None
    scenarios: List[str] = None

    # Shared inference
    n_permutations: int = 500
    alpha: float = 0.05
    n_jobs: int = 1
    use_mp: bool = False
    seed: int = 42

    # Synthetic network
    n_nodes: int = 60
    n_modules: int = 4
    time_points: int = 30
    intra_corr: float = 0.3
    inter_corr: float = 0.05
    uniform_corr: float = 0.15
    noise_level: float = 0.05

    # GLM covariates
    interest_confound_corr: float = 0.3
    confound_effect_size: float = 0.15
    confound_scenario: str = "between_modules_dense"

    # Method parameters
    tfnbs_e: float = 0.3
    tfnbs_h: float = 3.0
    tfnbs_n: int = 50
    tfnbs_start_thres: float = 1.65
    nbs_threshold: float = 2.0
    fbc_min_cluster: int = 3

    output_dir: Path = Path("examples/simulation_validation/results/glm/glm_validation")

    def __post_init__(self) -> None:
        if self.sample_sizes is None:
            self.sample_sizes = [30, 60, 100]
        if self.effect_sizes is None:
            self.effect_sizes = [0.10, 0.20, 0.30]
        if self.scenarios is None:
            self.scenarios = ["within_module_dense", "hub", "chain"]


@dataclass
class GLMCalibrationResult:
    method: str
    null_run_id: int
    n_samples: int
    n_edges: int
    any_significant_pos: bool
    n_significant_pos: int
    fpr_pos: float
    min_p_pos: float
    any_significant_neg: bool
    n_significant_neg: int
    fpr_neg: float
    min_p_neg: float
    any_significant: bool
    n_significant_total: int
    fpr_total: float
    elapsed_time: float


@dataclass
class GLMPowerResult:
    method: str
    method_params: str
    scenario: str
    effect_size: float
    n_subjects: int
    repeat_id: int
    n_true: int
    n_null: int
    TP: int
    FP: int
    FN: int
    TN: int
    TPR: float
    FPR: float
    precision: float
    FDR: float
    elapsed_time: float


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / max(x.std(ddof=0), 1e-12)


def _stable_int_token(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _load_config(path: Optional[Path]) -> GLMConfig:
    if path is None:
        return GLMConfig(methods=list(GLM_METHODS))
    if not HAS_YAML:
        raise RuntimeError("PyYAML is required to read YAML configs.")
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if "methods" not in data:
        data["methods"] = list(GLM_METHODS)
    if "output_dir" in data:
        data["output_dir"] = Path(data["output_dir"])
    return GLMConfig(**data)


def _parse_nbs_method(method: str, fallback_threshold: float) -> Optional[dict]:
    prefixes = {
        "nbs_extent": "extent",
        "nbs_intensity": "intensity",
    }
    for prefix, stat in prefixes.items():
        if method == prefix:
            return {"threshold": fallback_threshold, "nbs_stat": stat}
        marker = prefix + "_"
        if method.startswith(marker):
            return {"threshold": float(method[len(marker):]), "nbs_stat": stat}
    return None


def _method_call(method: str, cfg: GLMConfig, net_labels: np.ndarray) -> Tuple[str, Dict[str, Any]]:
    nbs_kwargs = _parse_nbs_method(method, cfg.nbs_threshold)
    if nbs_kwargs is not None:
        return "nbs", nbs_kwargs

    kwargs: Dict[str, Any] = {}
    if method in {"tfnbs", "ni_tfnbs", "fbc_tfnbs"}:
        kwargs.update({
            "e": cfg.tfnbs_e,
            "h": cfg.tfnbs_h,
            "n": cfg.tfnbs_n,
            "start_thres": cfg.tfnbs_start_thres,
        })
    if method in {"cnbs", "ni_tfnbs", "fbc_tfnbs"}:
        kwargs["net_labels"] = net_labels
    if method == "fbc_tfnbs":
        kwargs["min_cluster_size"] = cfg.fbc_min_cluster
    return method, kwargs


def _method_params_label(method: str, kwargs: Dict[str, Any]) -> str:
    if not kwargs:
        return ""
    keys = {
        "tfnbs": ("e", "h", "n", "start_thres"),
        "ni_tfnbs": ("e", "h", "n", "start_thres"),
        "fbc_tfnbs": ("e", "h", "n", "start_thres", "min_cluster_size"),
        "nbs_extent_2.0": ("threshold",),
        "nbs_intensity_2.0": ("threshold",),
        "nbs_extent_3.0": ("threshold",),
        "nbs_intensity_3.0": ("threshold",),
    }.get(method, tuple(kwargs.keys()))
    return ",".join(f"{key}={kwargs[key]}" for key in keys if key in kwargs)


def _make_topology_generator(cfg: GLMConfig, seed: int) -> topo.TopologyDatasetGenerator:
    return topo.TopologyDatasetGenerator(
        n_nodes=cfg.n_nodes,
        n_modules=cfg.n_modules,
        intra_corr=cfg.intra_corr,
        inter_corr=cfg.inter_corr,
        uniform_corr=cfg.uniform_corr,
        noise_level=cfg.noise_level,
        seed=seed,
    )


def generate_glm_dataset(
    cfg: GLMConfig,
    *,
    scenario: str,
    effect_size: float,
    n_subjects: int,
    seed: int,
) -> dict:
    """Generate Fisher-z connectivity data for a continuous-covariate GLM."""
    rng = np.random.default_rng(seed)
    generator = _make_topology_generator(cfg, seed)

    base = generator.generate(
        scenario,
        effect_size=0.0,
        n_samples=n_subjects,
        time_points=cfg.time_points,
    )
    Y, _ = base.fisher_z()
    signal_weights = np.abs(base.effect_mask)
    signal_mask = signal_weights > 0

    if cfg.confound_scenario == scenario:
        confound_weights = signal_weights.copy()
    else:
        confound_ds = generator.generate(
            cfg.confound_scenario,
            effect_size=0.0,
            n_samples=n_subjects,
            time_points=cfg.time_points,
        )
        confound_weights = np.abs(confound_ds.effect_mask)

    interest = _zscore(rng.standard_normal(n_subjects))
    confound_noise = _zscore(rng.standard_normal(n_subjects))
    rho = float(np.clip(cfg.interest_confound_corr, -0.99, 0.99))
    confound = _zscore(rho * interest + np.sqrt(1.0 - rho**2) * confound_noise)
    confounds = confound.reshape(-1, 1)

    if effect_size != 0.0:
        Y += effect_size * interest[:, None, None] * signal_weights[None, :, :]
    if cfg.confound_effect_size != 0.0:
        Y += cfg.confound_effect_size * confound[:, None, None] * confound_weights[None, :, :]

    diag = np.arange(cfg.n_nodes)
    Y[:, diag, diag] = 0.0
    return {
        "Y": Y,
        "interest": interest,
        "confounds": confounds,
        "net_labels": base.net_labels,
        "signal_mask": signal_mask,
    }


def _run_glm(
    method: str,
    cfg: GLMConfig,
    data: dict,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, float, str]:
    method_key, kwargs = _method_call(method, cfg, data["net_labels"])
    method_params = _method_params_label(method, kwargs)
    start = time.time()
    result = compute_p_val_glm(
        data["Y"],
        interest=data["interest"],
        confounds=data["confounds"],
        method=method_key,
        n_permutations=cfg.n_permutations,
        use_mp=cfg.use_mp,
        rng=seed,
        **kwargs,
    )
    elapsed = time.time() - start
    return np.asarray(result["positive"]), np.asarray(result["negative"]), elapsed, method_params


def run_single_calibration_task(task: tuple) -> GLMCalibrationResult:
    method, n_subjects, run_id, cfg = task
    seed = cfg.seed + 10_000 * run_id + 137 * n_subjects
    data = generate_glm_dataset(
        cfg,
        scenario=cfg.scenarios[0],
        effect_size=0.0,
        n_subjects=n_subjects,
        seed=seed,
    )
    n_edges = cfg.n_nodes * (cfg.n_nodes - 1) // 2
    triu_idx = np.triu_indices(cfg.n_nodes, k=1)

    try:
        p_pos, p_neg, elapsed, _ = _run_glm(method, cfg, data, seed)
        p_pos_upper = p_pos[triu_idx]
        p_neg_upper = p_neg[triu_idx]

        sig_pos = p_pos_upper <= cfg.alpha
        sig_neg = p_neg_upper <= cfg.alpha
        n_sig_pos = int(sig_pos.sum())
        n_sig_neg = int(sig_neg.sum())
        min_p_pos = float(np.min(p_pos_upper))
        min_p_neg = float(np.min(p_neg_upper))
    except Exception as exc:
        print(f"  ERROR in GLM calibration {method} run {run_id}: {exc}")
        elapsed = 0.0
        n_sig_pos = n_sig_neg = 0
        min_p_pos = min_p_neg = 1.0

    n_sig_total = n_sig_pos + n_sig_neg
    return GLMCalibrationResult(
        method=method,
        null_run_id=run_id,
        n_samples=n_subjects,
        n_edges=n_edges,
        any_significant_pos=bool(n_sig_pos > 0),
        n_significant_pos=n_sig_pos,
        fpr_pos=n_sig_pos / n_edges,
        min_p_pos=min_p_pos,
        any_significant_neg=bool(n_sig_neg > 0),
        n_significant_neg=n_sig_neg,
        fpr_neg=n_sig_neg / n_edges,
        min_p_neg=min_p_neg,
        any_significant=bool(n_sig_total > 0),
        n_significant_total=n_sig_total,
        fpr_total=n_sig_total / (2 * n_edges),
        elapsed_time=elapsed,
    )


def run_single_power_task(task: tuple) -> GLMPowerResult:
    method, scenario, effect_size, repeat_id, cfg = task
    seed = cfg.seed + 50_000 * repeat_id + _stable_int_token(scenario) % 10_000
    data = generate_glm_dataset(
        cfg,
        scenario=scenario,
        effect_size=effect_size,
        n_subjects=cfg.n_subjects,
        seed=seed,
    )
    triu_idx = np.triu_indices(cfg.n_nodes, k=1)
    signal_upper = data["signal_mask"][triu_idx]
    n_true = int(signal_upper.sum())
    n_null = int((~signal_upper).sum())

    try:
        p_pos, _, elapsed, method_params = _run_glm(method, cfg, data, seed)
        sig_upper = p_pos[triu_idx] <= cfg.alpha
        TP = int(np.sum(sig_upper & signal_upper))
        FP = int(np.sum(sig_upper & ~signal_upper))
        FN = int(np.sum(~sig_upper & signal_upper))
        TN = int(np.sum(~sig_upper & ~signal_upper))
    except Exception as exc:
        print(f"  ERROR in GLM power {method} {scenario} repeat {repeat_id}: {exc}")
        method_params = ""
        elapsed = 0.0
        TP, FP, FN, TN = 0, 0, n_true, n_null

    TPR = TP / n_true if n_true else 0.0
    FPR = FP / n_null if n_null else 0.0
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    FDR = FP / (TP + FP) if (TP + FP) else 0.0

    return GLMPowerResult(
        method=method,
        method_params=method_params,
        scenario=scenario,
        effect_size=effect_size,
        n_subjects=cfg.n_subjects,
        repeat_id=repeat_id,
        n_true=n_true,
        n_null=n_null,
        TP=TP,
        FP=FP,
        FN=FN,
        TN=TN,
        TPR=TPR,
        FPR=FPR,
        precision=precision,
        FDR=FDR,
        elapsed_time=elapsed,
    )


def _run_tasks(tasks: List[tuple], worker, cfg: GLMConfig, desc: str) -> list:
    if cfg.n_jobs == 1 or len(tasks) <= 1:
        return [worker(task) for task in tqdm(tasks, desc=desc, unit="run")]

    n_jobs = get_available_cores() if cfg.n_jobs < 0 else cfg.n_jobs
    n_jobs = min(n_jobs, len(tasks))
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_jobs) as pool:
        return list(tqdm(pool.imap_unordered(worker, tasks), total=len(tasks), desc=desc, unit="run"))


def run_calibration(cfg: GLMConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tasks = [
        (method, n_subjects, run_id, cfg)
        for method, n_subjects, run_id in itertools.product(
            cfg.methods, cfg.sample_sizes, range(cfg.n_null)
        )
    ]
    print("=== GLM Null FWER Calibration ===")
    print(f"Methods: {cfg.methods}")
    print(f"Subject counts: {cfg.sample_sizes}")
    print(f"Null repeats per cell: {cfg.n_null}")
    print(f"Permutations: {cfg.n_permutations}")
    results = _run_tasks(tasks, run_single_calibration_task, cfg, "glm-fwer")

    df = pd.DataFrame([vars(result) for result in results])
    summary = compute_fpr_summary(df, alpha=cfg.alpha)

    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "glm_fwer_results.csv", index=False)
    summary.to_csv(out_dir / "glm_fwer_summary.csv", index=False)
    return df, summary


def run_power(cfg: GLMConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tasks = [
        (method, scenario, effect_size, repeat_id, cfg)
        for method, scenario, effect_size, repeat_id in itertools.product(
            cfg.methods, cfg.scenarios, cfg.effect_sizes, range(cfg.n_repeats)
        )
    ]
    print("=== GLM Continuous-Covariate Power ===")
    print(f"Methods: {cfg.methods}")
    print(f"Scenarios: {cfg.scenarios}")
    print(f"Effect sizes: {cfg.effect_sizes}")
    print(f"Repeats per cell: {cfg.n_repeats}")
    print(f"Subjects: {cfg.n_subjects}")
    print(f"Permutations: {cfg.n_permutations}")
    results = _run_tasks(tasks, run_single_power_task, cfg, "glm-power")

    df = pd.DataFrame([vars(result) for result in results])
    group_cols = ["method", "method_params", "scenario", "effect_size", "n_subjects"]
    summary = (
        df.groupby(group_cols)
        .agg(
            TPR_mean=("TPR", "mean"),
            TPR_std=("TPR", "std"),
            FDR_mean=("FDR", "mean"),
            FDR_std=("FDR", "std"),
            FPR_mean=("FPR", "mean"),
            precision_mean=("precision", "mean"),
            elapsed_time_mean=("elapsed_time", "mean"),
            n_repeats=("repeat_id", "count"),
        )
        .reset_index()
    )

    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "glm_power_results.csv", index=False)
    summary.to_csv(out_dir / "glm_power_summary.csv", index=False)
    per_scenario = out_dir / "per_scenario"
    per_scenario.mkdir(exist_ok=True)
    for scenario, sdf in df.groupby("scenario"):
        sdf.to_csv(per_scenario / f"glm_power_results_{scenario}.csv", index=False)
    return df, summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run GLM null calibration and continuous-covariate power validation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", type=Path, default=None, help="YAML config path.")
    parser.add_argument(
        "--mode",
        choices=["calibration", "power", "all"],
        default=None,
        help="Override config mode.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output directory.")
    args = parser.parse_args(argv)

    cfg = _load_config(args.config)
    if args.mode is not None:
        cfg.mode = args.mode
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    if cfg.mode in {"calibration", "all"}:
        _, summary = run_calibration(cfg)
        print(summary[["method", "n_samples", "fwer_max_tail", "fwer_two_sided_bonf"]])
    if cfg.mode in {"power", "all"}:
        _, summary = run_power(cfg)
        print(summary.head(20))

    print(f"\nGLM validation outputs saved to {cfg.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
