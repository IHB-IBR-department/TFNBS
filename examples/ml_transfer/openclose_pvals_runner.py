#!/usr/bin/env python3
"""
Code to compute OpenClose paired p-value maps from a YAML config.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    import yaml
except ImportError as exc:  
    raise ImportError(
        "PyYAML is required. Install with: pip install pyyaml"
    ) from exc

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "examples"))

from openclose_loader import OpenCloseDataset
from tfnbs.pairwise_stats import compute_p_val, compute_t_stat


CONSTRAINED_METHODS = {"cnbs", "ni_tfnbs", "fbc_tfnbs"}
TFNBS_FAMILY = {"tfnbs", "ni_tfnbs", "fbc_tfnbs"}


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    return [value]


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)


def _validate_tfnbs_params(tfnbs_cfg: Dict[str, Any]) -> None:
    e_list = _as_list(tfnbs_cfg.get("e", 0.4))
    h_list = _as_list(tfnbs_cfg.get("h", 3.0))
    if len(e_list) != len(h_list):
        raise ValueError("tfnbs.e and tfnbs.h must have the same length.")


def _build_method_kwargs(method: str, cfg: Dict[str, Any], net_labels: np.ndarray | None) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}

    if method == "nbs":
        nbs_cfg = cfg.get("nbs", {})
        threshold = nbs_cfg.get("threshold", 2.0)
        if isinstance(threshold, list):
            threshold = threshold[0]
        kwargs["threshold"] = threshold
        nbs_stat = nbs_cfg.get("nbs_stat", "extent")
        if isinstance(nbs_stat, list):
            nbs_stat = nbs_stat[0]
        kwargs["nbs_stat"] = nbs_stat

    if method in TFNBS_FAMILY:
        tfnbs_cfg = cfg.get("tfnbs", {})
        kwargs["e"] = tfnbs_cfg.get("e", 0.4)
        kwargs["h"] = tfnbs_cfg.get("h", 3.0)
        kwargs["n"] = tfnbs_cfg.get("n_thresholds", 50)
        kwargs["start_thres"] = tfnbs_cfg.get("start_thres", 1.65)

    if method in CONSTRAINED_METHODS:
        if net_labels is None:
            raise ValueError(f"Method '{method}' requires net_labels.")
        kwargs["net_labels"] = net_labels

    if method == "fbc_tfnbs":
        fbc_cfg = cfg.get("fbc_tfnbs", {})
        min_cluster_size = fbc_cfg.get("min_cluster_size", 3)
        if isinstance(min_cluster_size, list):
            min_cluster_size = min_cluster_size[0]
        kwargs["min_cluster_size"] = min_cluster_size

    return kwargs


def _method_key(method: str, cfg: Dict[str, Any]) -> str:
    if method != "nbs":
        return method
    nbs_stat = cfg.get("nbs", {}).get("nbs_stat", "extent")
    if isinstance(nbs_stat, list):
        return "nbs"
    return f"nbs_{nbs_stat}"


def _concat_p_vals(p_vals_list: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    keys = list(p_vals_list[0].keys())
    stacked: Dict[str, np.ndarray] = {}
    for key in keys:
        arrays = []
        for p_vals in p_vals_list:
            arr = np.asarray(p_vals[key])
            if arr.ndim == 2:
                arr = arr[..., np.newaxis]
            arrays.append(arr)
        stacked[key] = np.concatenate(arrays, axis=-1)
    return stacked


def _save_results(output_path: Path, results: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(results, f)


def _init_logger(output_dir: Path, dataset: str) -> callable:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"pvals_run_{dataset}_{timestamp}.log"

    def _log(message: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
        print(line)
        with log_path.open("a") as f:
            f.write(line + "\n")

    return _log


def run(config_path: Path) -> None:
    cfg = _load_config(config_path)

    dataset = cfg.get("dataset", "hcp")
    experiments = cfg.get("experiments", ["ihb"])
    test_type = cfg.get("test_type", "paired")
    label_scheme = cfg.get("label_scheme", None)
    methods = cfg.get("methods", ["tfnbs"])

    n_permutations = int(cfg.get("n_permutations", 200))
    use_mp = bool(cfg.get("use_mp", True))
    random_state = cfg.get("random_state", None)
    n_processes = cfg.get("n_processes", None)

    save_t_signed = bool(cfg.get("save_t_signed", True))
    save_net_labels = bool(cfg.get("save_net_labels", True))
    output_dir = Path(cfg.get("output_dir", "results/openclose_hcp/pvals"))
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    _validate_tfnbs_params(cfg.get("tfnbs", {}))
    log = _init_logger(output_dir, dataset)
    run_start = time.perf_counter()

    if dataset == "sch200":
        if label_scheme is None:
            label_scheme = "yeo7"
        elif label_scheme != "yeo7":
            raise ValueError("Schaefer200 only supports label_scheme='yeo7'.")
        loader = OpenCloseDataset.schaefer200
    else:
        if label_scheme is None:
            label_scheme = "cole"
        elif label_scheme not in {"cole", "cortical"}:
            raise ValueError("HCP label_scheme must be 'cole' or 'cortical'.")
        loader = OpenCloseDataset.hcp

    log(f"Run start | dataset={dataset} experiments={experiments} methods={methods}")

    for experiment in experiments:
        ds = loader(experiment)
        open_z, close_z = ds.fisher_z()

        needs_labels = any(m in CONSTRAINED_METHODS for m in methods)
        net_labels = ds.get_net_labels(label_scheme) if (needs_labels or save_net_labels) else None

        t_signed = None
        if save_t_signed:
            t_stat = compute_t_stat(open_z, close_z, test_type=test_type)
            t_signed = t_stat["g2>g1"] - t_stat["g1>g2"]

        p_maps_2d: Dict[str, Any] = {}
        p_maps_3d: Dict[str, Any] = {}

        for method in methods:
            method_start = time.perf_counter()
            if method == "nbs":
                nbs_cfg = cfg.get("nbs", {})
                thresholds = _as_list(nbs_cfg.get("threshold", 2.0))
                nbs_stats = _as_list(nbs_cfg.get("nbs_stat", "extent"))
                if len(thresholds) > 1 or len(nbs_stats) > 1:
                    p_vals_list = []
                    param_pairs = []
                    for nbs_stat in nbs_stats:
                        for threshold in thresholds:
                            method_kwargs = _build_method_kwargs(method, cfg, net_labels)
                            method_kwargs["threshold"] = threshold
                            method_kwargs["nbs_stat"] = nbs_stat
                            p_vals_list.append(
                                compute_p_val(
                                    open_z,
                                    close_z,
                                    n_permutations=n_permutations,
                                    test_type=test_type,
                                    method=method,
                                    use_mp=use_mp,
                                    random_state=random_state,
                                    n_processes=n_processes,
                                    **method_kwargs,
                                )
                            )
                            param_pairs.append(
                                {"threshold": threshold, "nbs_stat": nbs_stat}
                            )
                    key = _method_key(method, cfg)
                    p_vals_concat = _concat_p_vals(p_vals_list)
                    p_maps_3d[key] = {
                        "p_vals": _concat_p_vals(p_vals_list),
                        "threshold_values": thresholds,
                        "nbs_stats": nbs_stats,
                        "param_pairs": param_pairs,
                    }
                    method_result = {
                        "dataset": dataset,
                        "experiment": experiment,
                        "method": method,
                        "method_key": key,
                        "test_type": test_type,
                        "label_scheme": label_scheme,
                        "net_labels": net_labels if save_net_labels else None,
                        "p_vals": p_vals_concat,
                        "param_meta": {
                            "threshold_values": thresholds,
                            "nbs_stats": nbs_stats,
                            "param_pairs": param_pairs,
                        },
                        "params": {
                            "n_permutations": n_permutations,
                            "use_mp": use_mp,
                            "random_state": random_state,
                            "n_processes": n_processes,
                        },
                        "t_signed": t_signed if save_t_signed else None,
                    }
                    method_path = output_dir / f"pvals_{dataset}_{experiment}_{key}.pkl"
                    _save_results(method_path, method_result)
                    elapsed = time.perf_counter() - method_start
                    log(f"{experiment} | {key} saved={method_path} elapsed_s={elapsed:.2f}")
                    continue

            if method == "fbc_tfnbs":
                fbc_cfg = cfg.get("fbc_tfnbs", {})
                min_sizes = _as_list(fbc_cfg.get("min_cluster_size", 3))
                if len(min_sizes) > 1:
                    tfnbs_cfg = cfg.get("tfnbs", {})
                    e_vals = _as_list(tfnbs_cfg.get("e", 0.4))
                    h_vals = _as_list(tfnbs_cfg.get("h", 3.0))
                    p_vals_list = []
                    param_triplets = []
                    for min_cluster in min_sizes:
                        method_kwargs = _build_method_kwargs(method, cfg, net_labels)
                        method_kwargs["min_cluster_size"] = min_cluster
                        p_vals_list.append(
                            compute_p_val(
                                open_z,
                                close_z,
                                n_permutations=n_permutations,
                                test_type=test_type,
                                method=method,
                                use_mp=use_mp,
                                random_state=random_state,
                                n_processes=n_processes,
                                **method_kwargs,
                            )
                        )
                        for e_val, h_val in zip(e_vals, h_vals):
                            param_triplets.append(
                                {"min_cluster_size": min_cluster, "e": e_val, "h": h_val}
                            )
                    key = _method_key(method, cfg)
                    p_vals_concat = _concat_p_vals(p_vals_list)
                    p_maps_3d[key] = {
                        "p_vals": p_vals_concat,
                        "min_cluster_size_values": min_sizes,
                        "e_values": e_vals,
                        "h_values": h_vals,
                        "param_triplets": param_triplets,
                    }
                    method_result = {
                        "dataset": dataset,
                        "experiment": experiment,
                        "method": method,
                        "method_key": key,
                        "test_type": test_type,
                        "label_scheme": label_scheme,
                        "net_labels": net_labels if save_net_labels else None,
                        "p_vals": p_vals_concat,
                        "param_meta": {
                            "min_cluster_size_values": min_sizes,
                            "e_values": e_vals,
                            "h_values": h_vals,
                            "param_triplets": param_triplets,
                        },
                        "params": {
                            "n_permutations": n_permutations,
                            "use_mp": use_mp,
                            "random_state": random_state,
                            "n_processes": n_processes,
                        },
                        "t_signed": t_signed if save_t_signed else None,
                    }
                    method_path = output_dir / f"pvals_{dataset}_{experiment}_{key}.pkl"
                    _save_results(method_path, method_result)
                    elapsed = time.perf_counter() - method_start
                    log(f"{experiment} | {key} saved={method_path} elapsed_s={elapsed:.2f}")
                    continue

            method_kwargs = _build_method_kwargs(method, cfg, net_labels)
            p_vals = compute_p_val(
                open_z,
                close_z,
                n_permutations=n_permutations,
                test_type=test_type,
                method=method,
                use_mp=use_mp,
                random_state=random_state,
                n_processes=n_processes,
                **method_kwargs,
            )

            key = _method_key(method, cfg)
            map_shape = p_vals["g2>g1"].ndim

            if map_shape == 3:
                tfnbs_cfg = cfg.get("tfnbs", {})
                e_vals = _as_list(tfnbs_cfg.get("e", 0.4))
                h_vals = _as_list(tfnbs_cfg.get("h", 3.0))
                p_maps_3d[key] = {
                    "p_vals": p_vals,
                    "e_values": e_vals,
                    "h_values": h_vals,
                    "param_pairs": list(zip(e_vals, h_vals)),
                }
            else:
                p_maps_2d[key] = p_vals

            method_result = {
                "dataset": dataset,
                "experiment": experiment,
                "method": method,
                "method_key": key,
                "test_type": test_type,
                "label_scheme": label_scheme,
                "net_labels": net_labels if save_net_labels else None,
                "p_vals": p_vals,
                "param_meta": p_maps_3d.get(key, {}),
                "params": {
                    "n_permutations": n_permutations,
                    "use_mp": use_mp,
                    "random_state": random_state,
                    "n_processes": n_processes,
                },
                "t_signed": t_signed if save_t_signed else None,
            }
            method_path = output_dir / f"pvals_{dataset}_{experiment}_{key}.pkl"
            _save_results(method_path, method_result)
            elapsed = time.perf_counter() - method_start
            log(f"{experiment} | {key} saved={method_path} elapsed_s={elapsed:.2f}")

        results = {
            "dataset": dataset,
            "experiment": experiment,
            "test_type": test_type,
            "label_scheme": label_scheme,
            "net_labels": net_labels,
            "p_maps_2d": p_maps_2d,
            "p_maps_3d": p_maps_3d,
            "t_signed": t_signed,
            "params": {
                "n_permutations": n_permutations,
                "use_mp": use_mp,
                "random_state": random_state,
                "n_processes": n_processes,
                "tfnbs": cfg.get("tfnbs", {}),
                "nbs": cfg.get("nbs", {}),
                "fbc_tfnbs": cfg.get("fbc_tfnbs", {}),
                "methods": methods,
            },
        }

        output_path = output_dir / f"pvals_{dataset}_{experiment}.pkl"
        _save_results(output_path, results)
        log(f"{experiment} | combined saved={output_path}")

    total_elapsed = time.perf_counter() - run_start
    log(f"JOB DONE | total_elapsed_s={total_elapsed:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML config file.",
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
