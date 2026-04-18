"""
Topology-based synthetic connectivity scenarios.

Built on top of :class:`conninfpy.synth_datasets.ModularDatasetGenerator`,
this module adds a library of named topological effect patterns (hub,
chain, rich-club, checkerboard, gradient, etc.) plus a scenario registry
so callers can request a scenario by name.

Public API
----------
- :class:`TopologyScenario` — scenario specification
- :class:`TopologyDataset`  — generated output container
- :class:`TopologyDatasetGenerator` — the main entry point
- :func:`list_scenarios`, :func:`get_scenario`, :func:`get_scenarios` — registry

Mask generator functions (``_mask_*``, ``_scenario_mask_*``) are private;
access them indirectly via the scenario registry.

Example
-------
>>> from conninfpy.topologies import TopologyDatasetGenerator, list_scenarios
>>> list_scenarios()[:3]
['within_module_dense', 'between_modules_dense', 'hub']
>>> gen = TopologyDatasetGenerator(n_nodes=60, n_modules=4, seed=1)
>>> ds = gen.generate("chain", effect_size=0.3, n_samples=20, time_points=30)
>>> ds.group1.shape
(20, 60, 60)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np

from .synth_datasets import ModularDatasetGenerator
from .utils import fisher_r_to_z


__all__ = [
    "TopologyScenario",
    "TopologyDataset",
    "TopologyDatasetGenerator",
    "list_scenarios",
    "get_scenarios",
    "get_scenario",
]


ArrayF = np.ndarray


def _stable_seed(*parts: object) -> int:
    """
    Deterministic 32-bit seed from arbitrary parts (order-independent of Python hash).

    This makes per-scenario generation reproducible regardless of scenario iteration order.
    """
    h = hashlib.sha256()
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"|")
    return int.from_bytes(h.digest()[:4], "little", signed=False)


def _params_token(params: Mapping[str, object]) -> str:
    items = sorted(params.items(), key=lambda kv: kv[0])
    return ",".join(f"{k}={v}" for k, v in items)


@dataclass(frozen=True)
class TopologyScenario:
    """
    Scenario specification: how to build labels + an effect mask topology.

    - `mask_fn(labels, rng=..., **mask_params)` returns a symmetric (N, N) matrix where:
      0 means "no effect" and non-zero values scale the effect magnitude per edge.
    - `labels_fn(n_nodes, n_modules)` returns `net_labels` (shape (N,)).
    """

    name: str
    base_kind: str  # "modular" or "uniform"
    mask_fn: Callable[..., ArrayF]
    mask_params: Dict[str, object] = field(default_factory=dict)
    labels_fn: Callable[[int, int], np.ndarray] = lambda n_nodes, n_modules: np.sort(
        np.arange(n_nodes) % n_modules
    )

    def with_mask_params(self, **overrides: object) -> "TopologyScenario":
        merged = dict(self.mask_params)
        merged.update(overrides)
        return replace(self, mask_params=merged)


@dataclass(frozen=True)
class TopologyDataset:
    """
    Output of a topology simulation, ready for TFNBS/NBS pipelines.

    - `group1`/`group2` are arrays of shape (n_samples, N, N) with zero diagonal.
    - `net_labels` can be passed to cNBS/NI/FBC methods.
    - `effect_mask` is the signed/weighted topology mask (not multiplied by effect_size).
    """

    group1: ArrayF
    group2: ArrayF
    net_labels: np.ndarray
    effect_mask: ArrayF
    effect_size: float
    scenario: TopologyScenario
    meta: Dict[str, object] = field(default_factory=dict)

    def fisher_z(self) -> Tuple[ArrayF, ArrayF]:
        return fisher_r_to_z(self.group1), fisher_r_to_z(self.group2)


class TopologyDatasetGenerator:
    """
    Reusable dataset generator for topology scenarios.

    Example
    -------
    >>> gen = TopologyDatasetGenerator(n_nodes=60, n_modules=4, seed=1)
    >>> ds = gen.generate("chain", effect_size=0.3, n_samples=20, time_points=30)
    >>> ds.group1.shape
    (20, 60, 60)
    """

    def __init__(
        self,
        n_nodes: int = 60,
        n_modules: int = 4,
        intra_corr: float = 0.3,
        inter_corr: float = 0.05,
        uniform_corr: float = 0.15,
        noise_level: float = 0.05,
        seed: int = 42,
    ) -> None:
        self.n_nodes = int(n_nodes)
        self.n_modules = int(n_modules)
        self.intra_corr = float(intra_corr)
        self.inter_corr = float(inter_corr)
        self.uniform_corr = float(uniform_corr)
        self.noise_level = float(noise_level)
        self.seed = int(seed)

    def generate(
        self,
        scenario: Union[str, TopologyScenario],
        effect_size: float,
        *,
        n_samples: int = 20,
        n_samples_g1: Optional[int] = None,
        n_samples_g2: Optional[int] = None,
        time_points: int = 30,
        scenario_params: Optional[Dict[str, object]] = None,
        zero_diagonal: bool = True,
    ) -> TopologyDataset:
        scenario_obj = get_scenario(scenario) if isinstance(scenario, str) else scenario
        if scenario_params:
            scenario_obj = scenario_obj.with_mask_params(**scenario_params)

        n_samples_g1 = int(n_samples if n_samples_g1 is None else n_samples_g1)
        n_samples_g2 = int(n_samples_g1 if n_samples_g2 is None else n_samples_g2)
        time_points = int(time_points)

        param_token = _params_token(scenario_obj.mask_params)
        mask_seed = _stable_seed(self.seed, scenario_obj.name, "mask", param_token, self.n_nodes, self.n_modules)
        data_seed = _stable_seed(self.seed, scenario_obj.name, "data", param_token, self.n_nodes, self.n_modules, time_points)

        labels = scenario_obj.labels_fn(self.n_nodes, self.n_modules)
        rng_mask = np.random.default_rng(mask_seed)
        effect_mask = scenario_obj.mask_fn(labels, rng=rng_mask, **scenario_obj.mask_params)
        effect_mask = _ensure_symmetric_zero_diag(effect_mask)

        if scenario_obj.base_kind == "modular":
            intra_corr = self.intra_corr
            inter_corr = self.inter_corr
        elif scenario_obj.base_kind == "uniform":
            intra_corr = self.uniform_corr
            inter_corr = self.uniform_corr
        else:
            raise ValueError(f"Unknown base_kind: {scenario_obj.base_kind!r}")

        gen = ModularDatasetGenerator(
            N=self.n_nodes,
            n_modules=self.n_modules,
            intra_corr=intra_corr,
            inter_corr=inter_corr,
            noise_level=self.noise_level,
            seed=data_seed,
        )
        if not np.array_equal(gen.labels, labels):
            gen.labels = labels
            gen.rng = np.random.default_rng(data_seed)
            gen.base_cov = gen._create_base_covariance(intra_corr, inter_corr, self.noise_level)

        g1, g2, net_labels = gen.generate_data(
            effect_mask=effect_mask,
            effect_size=float(effect_size),
            n_samples_g1=n_samples_g1,
            n_samples_g2=n_samples_g2,
            time_points=time_points,
        )

        if zero_diagonal:
            diag = np.arange(self.n_nodes)
            g1[:, diag, diag] = 0.0
            g2[:, diag, diag] = 0.0

        meta = {
            "mask_seed": int(mask_seed),
            "data_seed": int(data_seed),
            "n_samples_g1": int(n_samples_g1),
            "n_samples_g2": int(n_samples_g2),
            "time_points": int(time_points),
            "intra_corr": float(intra_corr),
            "inter_corr": float(inter_corr),
            "noise_level": float(self.noise_level),
        }
        return TopologyDataset(
            group1=g1,
            group2=g2,
            net_labels=net_labels,
            effect_mask=effect_mask,
            effect_size=float(effect_size),
            scenario=scenario_obj,
            meta=meta,
        )


def _ensure_symmetric_zero_diag(matrix: ArrayF) -> ArrayF:
    matrix = np.asarray(matrix, dtype=np.float64)
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 0.0)
    return matrix


def _sorted_module_labels(n_nodes: int, n_modules: int) -> np.ndarray:
    """Balanced module labels, sorted so blocks appear as contiguous squares in plots."""
    return np.sort(np.arange(n_nodes) % n_modules)


def _imbalanced_module_labels(n_nodes: int, n_modules: int) -> np.ndarray:
    """
    Create intentionally imbalanced module sizes.

    Example (n_modules=4): ~50% nodes in module 0, remainder split across others.
    """
    if n_modules < 2:
        raise ValueError("n_modules must be >= 2 for imbalanced labels.")

    size0 = int(round(0.5 * n_nodes))
    remainder = n_nodes - size0
    base = remainder // (n_modules - 1)
    sizes = [size0] + [base] * (n_modules - 1)
    for i in range(remainder - base * (n_modules - 1)):
        sizes[1 + i] += 1

    labels: List[int] = []
    for idx, size in enumerate(sizes):
        labels.extend([idx] * size)
    return np.asarray(labels, dtype=int)


def _module_nodes(labels: np.ndarray, module_idx: int) -> np.ndarray:
    return np.where(labels == module_idx)[0]


def _mask_within_module(labels: np.ndarray, module_idx: int) -> ArrayF:
    n_nodes = labels.shape[0]
    mask = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    nodes = _module_nodes(labels, module_idx)
    if nodes.size < 2:
        return mask
    ii, jj = np.triu_indices(nodes.size, k=1)
    mask[nodes[ii], nodes[jj]] = 1.0
    return _ensure_symmetric_zero_diag(mask)


def _mask_between_modules(labels: np.ndarray, module_a: int, module_b: int) -> ArrayF:
    n_nodes = labels.shape[0]
    mask = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    nodes_a = _module_nodes(labels, module_a)
    nodes_b = _module_nodes(labels, module_b)
    if nodes_a.size == 0 or nodes_b.size == 0:
        return mask
    mask[np.ix_(nodes_a, nodes_b)] = 1.0
    return _ensure_symmetric_zero_diag(mask)


def _mask_hub(n_nodes: int, hub_node: int, n_spokes: int, rng: np.random.Generator) -> ArrayF:
    mask = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    candidates = [i for i in range(n_nodes) if i != hub_node]
    n_spokes = min(n_spokes, len(candidates))
    targets = rng.choice(candidates, size=n_spokes, replace=False)
    mask[hub_node, targets] = 1.0
    return _ensure_symmetric_zero_diag(mask)


def _mask_chain(n_nodes: int, length: int, rng: np.random.Generator) -> ArrayF:
    """Random simple path (edges share nodes; long thin component)."""
    mask = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    if n_nodes < 2:
        return mask

    current = int(rng.integers(0, n_nodes))
    visited = {current}
    for _ in range(length):
        candidates = list(set(range(n_nodes)) - visited)
        if not candidates:
            break
        nxt = int(rng.choice(candidates))
        mask[current, nxt] = 1.0
        visited.add(nxt)
        current = nxt
    return _ensure_symmetric_zero_diag(mask)


def _mask_fragmented_within_module(
    labels: np.ndarray,
    module_idx: int,
    sparsity: float,
    rng: np.random.Generator,
) -> ArrayF:
    """Random subset of within-module edges (often not one connected component)."""
    full = _mask_within_module(labels, module_idx)
    rows, cols = np.triu_indices_from(full, k=1)
    edge_idx = np.where(full[rows, cols] > 0)[0]
    if edge_idx.size == 0:
        return full * 0.0

    n_select = int(round(edge_idx.size * sparsity))
    n_select = max(1, min(n_select, edge_idx.size))
    chosen = rng.choice(edge_idx, size=n_select, replace=False)

    mask = np.zeros_like(full)
    mask[rows[chosen], cols[chosen]] = 1.0
    return _ensure_symmetric_zero_diag(mask)


def _mask_multi_clique_within_module(
    labels: np.ndarray,
    module_idx: int,
    n_clusters: int,
    nodes_per_cluster: int,
    rng: np.random.Generator,
) -> ArrayF:
    """Several disconnected cliques inside one module."""
    n_nodes = labels.shape[0]
    mask = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    nodes = _module_nodes(labels, module_idx).copy()
    rng.shuffle(nodes)

    for c in range(n_clusters):
        start = c * nodes_per_cluster
        end = start + nodes_per_cluster
        if end > nodes.size:
            break
        cluster = nodes[start:end]
        ii, jj = np.triu_indices(cluster.size, k=1)
        mask[cluster[ii], cluster[jj]] = 1.0

    return _ensure_symmetric_zero_diag(mask)


def _mask_scattered_cross_block(
    labels: np.ndarray,
    n_edges_per_block: int,
    rng: np.random.Generator,
) -> ArrayF:
    """Scattered edges across many blocks (noise-like stress pattern)."""
    n_nodes = labels.shape[0]
    n_modules = int(np.max(labels) + 1)
    mask = np.zeros((n_nodes, n_nodes), dtype=np.float64)

    for mod_i in range(n_modules):
        for mod_j in range(mod_i, n_modules):
            nodes_i = _module_nodes(labels, mod_i)
            nodes_j = _module_nodes(labels, mod_j)
            if nodes_i.size == 0 or nodes_j.size == 0:
                continue

            if mod_i == mod_j:
                rr, cc = np.triu_indices(nodes_i.size, k=1)
                pairs = np.stack([nodes_i[rr], nodes_i[cc]], axis=1)
            else:
                grid_i, grid_j = np.meshgrid(nodes_i, nodes_j, indexing="ij")
                pairs = np.stack([grid_i.ravel(), grid_j.ravel()], axis=1)

            if pairs.shape[0] == 0:
                continue

            n_pick = min(n_edges_per_block, pairs.shape[0])
            pick = rng.choice(pairs.shape[0], size=n_pick, replace=False)
            chosen = pairs[pick]
            mask[chosen[:, 0], chosen[:, 1]] = 1.0

    return _ensure_symmetric_zero_diag(mask)


def _mask_perfect_matching_within_module(labels: np.ndarray, module_idx: int, rng: np.random.Generator) -> ArrayF:
    """
    Many edges in one block but no shared nodes (topologically disconnected).

    Builds a single random perfect matching (or near-perfect if odd node count).
    """
    n_nodes = labels.shape[0]
    mask = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    nodes = _module_nodes(labels, module_idx).copy()
    rng.shuffle(nodes)
    n_pairs = nodes.size // 2
    for k in range(n_pairs):
        i = int(nodes[2 * k])
        j = int(nodes[2 * k + 1])
        mask[i, j] = 1.0
    return _ensure_symmetric_zero_diag(mask)


def _mask_cross_block_connected_chain(
    labels: np.ndarray,
    length: int,
    rng: np.random.Generator,
) -> ArrayF:
    """
    A single long topological component, but edges spread across many blocks.

    Construct a node path that alternates modules as much as possible.
    """
    n_nodes = labels.shape[0]
    n_modules = int(np.max(labels) + 1)
    nodes_by_module = {m: _module_nodes(labels, m).copy() for m in range(n_modules)}
    for m in nodes_by_module:
        rng.shuffle(nodes_by_module[m])

    # Round-robin node list across modules.
    rr_nodes: List[int] = []
    ptr = {m: 0 for m in range(n_modules)}
    while len(rr_nodes) < n_nodes:
        progressed = False
        for m in range(n_modules):
            if ptr[m] < nodes_by_module[m].size:
                rr_nodes.append(int(nodes_by_module[m][ptr[m]]))
                ptr[m] += 1
                progressed = True
        if not progressed:
            break

    if len(rr_nodes) < 2:
        return np.zeros((n_nodes, n_nodes), dtype=np.float64)

    length = min(length, len(rr_nodes) - 1)
    start = int(rng.integers(0, len(rr_nodes) - length))
    path = rr_nodes[start : start + length + 1]

    mask = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for a, b in zip(path[:-1], path[1:]):
        mask[a, b] = 1.0
    return _ensure_symmetric_zero_diag(mask)


def _mask_rich_club(
    n_nodes: int,
    n_hubs: int,
    rng: np.random.Generator,
    n_spokes_per_hub: int = 0,
) -> ArrayF:
    """
    Dense clique among hubs (rich-club). Optionally add hub->spoke edges.
    """
    n_hubs = max(2, min(n_hubs, n_nodes))
    hubs = rng.choice(n_nodes, size=n_hubs, replace=False)
    mask = np.zeros((n_nodes, n_nodes), dtype=np.float64)

    # Clique on hubs.
    for i in range(n_hubs):
        for j in range(i + 1, n_hubs):
            mask[int(hubs[i]), int(hubs[j])] = 1.0

    # Optional spokes.
    if n_spokes_per_hub > 0:
        all_nodes = np.arange(n_nodes)
        for hub in hubs:
            candidates = all_nodes[all_nodes != hub]
            spokes = rng.choice(candidates, size=min(n_spokes_per_hub, candidates.size), replace=False)
            mask[int(hub), spokes] = 1.0

    return _ensure_symmetric_zero_diag(mask)


def _mask_two_disconnected_cliques(
    labels: np.ndarray,
    module_idx: int,
    clique_size: int,
    rng: np.random.Generator,
) -> ArrayF:
    """Two equal-size disconnected cliques within one module."""
    n_nodes = labels.shape[0]
    nodes = _module_nodes(labels, module_idx).copy()
    rng.shuffle(nodes)
    if nodes.size < 2 * clique_size:
        clique_size = max(2, nodes.size // 2)
    if clique_size < 2:
        return np.zeros((n_nodes, n_nodes), dtype=np.float64)

    c1 = nodes[:clique_size]
    c2 = nodes[clique_size : 2 * clique_size]

    mask = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for cluster in (c1, c2):
        ii, jj = np.triu_indices(cluster.size, k=1)
        mask[cluster[ii], cluster[jj]] = 1.0
    return _ensure_symmetric_zero_diag(mask)


def _mask_partial_bipartite_between_modules(
    labels: np.ndarray,
    module_a: int,
    module_b: int,
    n_a: int,
    n_b: int,
    rng: np.random.Generator,
) -> ArrayF:
    """Subset A×B between-module complete bipartite block."""
    n_nodes = labels.shape[0]
    nodes_a = _module_nodes(labels, module_a).copy()
    nodes_b = _module_nodes(labels, module_b).copy()
    rng.shuffle(nodes_a)
    rng.shuffle(nodes_b)
    nodes_a = nodes_a[: min(n_a, nodes_a.size)]
    nodes_b = nodes_b[: min(n_b, nodes_b.size)]

    mask = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    if nodes_a.size == 0 or nodes_b.size == 0:
        return mask
    mask[np.ix_(nodes_a, nodes_b)] = 1.0
    return _ensure_symmetric_zero_diag(mask)


def _weighted_mask_gradient_chain(
    n_nodes: int,
    length: int,
    rng: np.random.Generator,
    min_weight: float = 0.2,
) -> ArrayF:
    """Connected component with a weight gradient (strong core -> weak tail)."""
    length = max(2, length)
    mask_binary = _mask_chain(n_nodes, length=length, rng=rng)
    rows, cols = np.where(np.triu(mask_binary, k=1) > 0)
    if rows.size == 0:
        return mask_binary

    # Assign a reproducible order of edges and a decreasing weight schedule.
    order = np.lexsort((cols, rows))
    rows = rows[order]
    cols = cols[order]

    weights = np.linspace(1.0, float(min_weight), num=rows.size, dtype=np.float64)
    weighted = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    weighted[rows, cols] = weights
    return _ensure_symmetric_zero_diag(weighted)


def _mask_checkerboard_within_module(labels: np.ndarray, module_idx: int) -> ArrayF:
    """
    Structured fragmented pattern inside a module (checkerboard-like).

    This creates many affected edges within the block, but with a regular pattern
    rather than a dense clique.
    """
    n_nodes = labels.shape[0]
    mask = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    nodes = _module_nodes(labels, module_idx)
    if nodes.size < 2:
        return mask

    # Use within-block coordinates (0..k-1) to define a simple checkerboard rule.
    k = nodes.size
    ii, jj = np.triu_indices(k, k=1)
    keep = ((ii + jj) % 2) == 0
    mask[nodes[ii[keep]], nodes[jj[keep]]] = 1.0
    return _ensure_symmetric_zero_diag(mask)


def _weighted_mask_core_periphery_within_module(
    labels: np.ndarray,
    module_idx: int,
    n_core: int,
    core_weight: float = 1.0,
    core_to_periphery_weight: float = 0.4,
    periphery_weight: float = 0.1,
) -> ArrayF:
    """
    Gradient inside one module: strong dense core + weaker periphery effects.

    This is useful for illustrating why threshold-free methods can outperform
    fixed-threshold NBS on connected but heterogeneous clusters.
    """
    n_nodes = labels.shape[0]
    nodes = _module_nodes(labels, module_idx)
    if nodes.size < 2:
        return np.zeros((n_nodes, n_nodes), dtype=np.float64)

    n_core = max(2, min(int(n_core), nodes.size))
    core = nodes[:n_core]
    periphery = nodes[n_core:]

    mask = np.zeros((n_nodes, n_nodes), dtype=np.float64)

    # Core clique.
    ii, jj = np.triu_indices(core.size, k=1)
    mask[core[ii], core[jj]] = core_weight

    # Core-to-periphery.
    if periphery.size > 0:
        mask[np.ix_(core, periphery)] = core_to_periphery_weight

    # Periphery-to-periphery.
    if periphery.size > 1:
        ii, jj = np.triu_indices(periphery.size, k=1)
        mask[periphery[ii], periphery[jj]] = periphery_weight

    return _ensure_symmetric_zero_diag(mask)


def _summarize_masked_effect(
    group1_r: ArrayF,
    group2_r: ArrayF,
    group1_z: ArrayF,
    group2_z: ArrayF,
    effect_gt_r: ArrayF,
    t_signed: ArrayF,
) -> Dict[str, float]:
    """
    Summarize realized effect magnitude on masked edges.

    Returns scalars computed on the upper triangle only (k=1).
    """
    n_nodes = effect_gt_r.shape[0]
    tri = np.triu_indices(n_nodes, k=1)

    gt_ut = effect_gt_r[tri]
    mask_ut = gt_ut != 0
    if not np.any(mask_ut):
        return {
            "n_edges": 0.0,
            "gt_r_mean": 0.0,
            "obs_r_mean": 0.0,
            "obs_z_mean": 0.0,
            "pooled_std_z_mean": 0.0,
            "cohen_d_z_mean": 0.0,
            "t_abs_median": 0.0,
            "t_abs_max": 0.0,
        }

    mean_g1_r = np.mean(group1_r, axis=0)
    mean_g2_r = np.mean(group2_r, axis=0)
    obs_diff_r = (mean_g2_r - mean_g1_r)[tri][mask_ut]

    mean_g1_z = np.mean(group1_z, axis=0)
    mean_g2_z = np.mean(group2_z, axis=0)
    obs_diff_z = (mean_g2_z - mean_g1_z)[tri][mask_ut]

    # Pooled std across subjects per edge (classical pooled variance), in Fisher-z domain.
    n1 = group1_z.shape[0]
    n2 = group2_z.shape[0]
    var1 = np.var(group1_z, axis=0, ddof=1)
    var2 = np.var(group2_z, axis=0, ddof=1)
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / max(1, (n1 + n2 - 2))
    pooled_std = np.sqrt(pooled_var)[tri][mask_ut]

    gt_vals = gt_ut[mask_ut]
    t_vals = t_signed[tri][mask_ut]

    pooled_std_z_mean = float(np.mean(pooled_std))
    obs_r_mean = float(np.mean(obs_diff_r))
    obs_z_mean = float(np.mean(obs_diff_z))
    cohen_d_z_mean = float(obs_z_mean / pooled_std_z_mean) if pooled_std_z_mean > 0 else 0.0

    return {
        "n_edges": float(np.sum(mask_ut)),
        "gt_r_mean": float(np.mean(gt_vals)),
        "obs_r_mean": obs_r_mean,
        "obs_z_mean": obs_z_mean,
        "pooled_std_z_mean": pooled_std_z_mean,
        "cohen_d_z_mean": cohen_d_z_mean,
        "t_abs_median": float(np.median(np.abs(t_vals))),
        "t_abs_max": float(np.max(np.abs(t_vals))),
    }


def _scenario_mask_within_module(
    labels: np.ndarray, *, rng: np.random.Generator, module_idx: int = 0
) -> ArrayF:
    return _mask_within_module(labels, module_idx=module_idx)


def _scenario_mask_between_modules(
    labels: np.ndarray, *, rng: np.random.Generator, module_a: int = 1, module_b: int = 2
) -> ArrayF:
    return _mask_between_modules(labels, module_a=module_a, module_b=module_b)


def _scenario_mask_hub(
    labels: np.ndarray, *, rng: np.random.Generator, hub_node: Optional[int] = None, n_spokes: int = 40
) -> ArrayF:
    n_nodes = labels.shape[0]
    if hub_node is None:
        hub_node = n_nodes // 2
    return _mask_hub(n_nodes, hub_node=int(hub_node), n_spokes=int(n_spokes), rng=rng)


def _scenario_mask_chain(
    labels: np.ndarray, *, rng: np.random.Generator, length: int = 30
) -> ArrayF:
    return _mask_chain(labels.shape[0], length=int(length), rng=rng)


def _scenario_mask_fragmented_within_module(
    labels: np.ndarray, *, rng: np.random.Generator, module_idx: int = 0, sparsity: float = 0.3
) -> ArrayF:
    return _mask_fragmented_within_module(labels, module_idx=int(module_idx), sparsity=float(sparsity), rng=rng)


def _scenario_mask_multi_clique_within_module(
    labels: np.ndarray,
    *,
    rng: np.random.Generator,
    module_idx: int = 0,
    n_clusters: int = 3,
    nodes_per_cluster: int = 4,
) -> ArrayF:
    return _mask_multi_clique_within_module(
        labels,
        module_idx=int(module_idx),
        n_clusters=int(n_clusters),
        nodes_per_cluster=int(nodes_per_cluster),
        rng=rng,
    )


def _scenario_mask_checkerboard_within_module(
    labels: np.ndarray, *, rng: np.random.Generator, module_idx: int = 0
) -> ArrayF:
    _ = rng
    return _mask_checkerboard_within_module(labels, module_idx=int(module_idx))


def _scenario_mask_scattered_cross_block(
    labels: np.ndarray, *, rng: np.random.Generator, n_edges_per_block: int = 6
) -> ArrayF:
    return _mask_scattered_cross_block(labels, n_edges_per_block=int(n_edges_per_block), rng=rng)


def _scenario_mask_within_plus_between(
    labels: np.ndarray,
    *,
    rng: np.random.Generator,
    within_module_idx: int = 0,
    between_module_a: int = 1,
    between_module_b: int = 2,
) -> ArrayF:
    _ = rng
    return _ensure_symmetric_zero_diag(
        _mask_within_module(labels, module_idx=int(within_module_idx))
        + _mask_between_modules(labels, module_a=int(between_module_a), module_b=int(between_module_b))
    )


def _scenario_mask_perfect_matching_within_module(
    labels: np.ndarray, *, rng: np.random.Generator, module_idx: int = 0
) -> ArrayF:
    return _mask_perfect_matching_within_module(labels, module_idx=int(module_idx), rng=rng)


def _scenario_mask_cross_block_connected_chain(
    labels: np.ndarray, *, rng: np.random.Generator, length: int = 30
) -> ArrayF:
    return _mask_cross_block_connected_chain(labels, length=int(length), rng=rng)


def _scenario_mask_rich_club(
    labels: np.ndarray, *, rng: np.random.Generator, n_hubs: int = 10, n_spokes_per_hub: int = 0
) -> ArrayF:
    return _mask_rich_club(
        labels.shape[0], n_hubs=int(n_hubs), rng=rng, n_spokes_per_hub=int(n_spokes_per_hub)
    )


def _scenario_mask_two_disconnected_cliques(
    labels: np.ndarray, *, rng: np.random.Generator, module_idx: int = 0, clique_size: int = 6
) -> ArrayF:
    return _mask_two_disconnected_cliques(labels, module_idx=int(module_idx), clique_size=int(clique_size), rng=rng)


def _scenario_mask_focal_clique_inside_module(
    labels: np.ndarray, *, rng: np.random.Generator, module_idx: int = 0, clique_size: int = 5
) -> ArrayF:
    return _mask_multi_clique_within_module(
        labels,
        module_idx=int(module_idx),
        n_clusters=1,
        nodes_per_cluster=int(clique_size),
        rng=rng,
    )


def _scenario_mask_partial_bipartite_between_modules(
    labels: np.ndarray,
    *,
    rng: np.random.Generator,
    module_a: int = 1,
    module_b: int = 2,
    n_a: int = 6,
    n_b: int = 6,
) -> ArrayF:
    return _mask_partial_bipartite_between_modules(
        labels, module_a=int(module_a), module_b=int(module_b), n_a=int(n_a), n_b=int(n_b), rng=rng
    )


def _scenario_mask_gradient_effect_chain(
    labels: np.ndarray, *, rng: np.random.Generator, length: int = 30, min_weight: float = 0.2
) -> ArrayF:
    return _weighted_mask_gradient_chain(labels.shape[0], length=int(length), rng=rng, min_weight=float(min_weight))


def _scenario_mask_gradient_core_periphery_within_module(
    labels: np.ndarray,
    *,
    rng: np.random.Generator,
    module_idx: int = 0,
    n_core: int = 6,
    core_weight: float = 1.0,
    core_to_periphery_weight: float = 0.4,
    periphery_weight: float = 0.1,
) -> ArrayF:
    _ = rng
    return _weighted_mask_core_periphery_within_module(
        labels,
        module_idx=int(module_idx),
        n_core=int(n_core),
        core_weight=float(core_weight),
        core_to_periphery_weight=float(core_to_periphery_weight),
        periphery_weight=float(periphery_weight),
    )


_SCENARIOS: List[TopologyScenario] = [
    TopologyScenario(
        name="within_module_dense",
        base_kind="modular",
        mask_fn=_scenario_mask_within_module,
        mask_params={"module_idx": 0},
        labels_fn=_sorted_module_labels,
    ),
    TopologyScenario(
        name="between_modules_dense",
        base_kind="modular",
        mask_fn=_scenario_mask_between_modules,
        mask_params={"module_a": 1, "module_b": 2},
        labels_fn=_sorted_module_labels,
    ),
    TopologyScenario(
        name="hub",
        base_kind="modular",
        mask_fn=_scenario_mask_hub,
        mask_params={"hub_node": None, "n_spokes": 40},
        labels_fn=_sorted_module_labels,
    ),
    TopologyScenario(
        name="chain",
        base_kind="modular",
        mask_fn=_scenario_mask_chain,
        mask_params={"length": 30},
        labels_fn=_sorted_module_labels,
    ),
    TopologyScenario(
        name="fragmented_within_module",
        base_kind="modular",
        mask_fn=_scenario_mask_fragmented_within_module,
        mask_params={"module_idx": 0, "sparsity": 0.3},
        labels_fn=_sorted_module_labels,
    ),
    TopologyScenario(
        name="multi_cluster_within_module",
        base_kind="modular",
        mask_fn=_scenario_mask_multi_clique_within_module,
        mask_params={"module_idx": 0, "n_clusters": 3, "nodes_per_cluster": 4},
        labels_fn=_sorted_module_labels,
    ),
    TopologyScenario(
        name="checkerboard_within_module",
        base_kind="modular",
        mask_fn=_scenario_mask_checkerboard_within_module,
        mask_params={"module_idx": 0},
        labels_fn=_sorted_module_labels,
    ),
    TopologyScenario(
        name="scattered_cross_block",
        base_kind="modular",
        mask_fn=_scenario_mask_scattered_cross_block,
        mask_params={"n_edges_per_block": 6},
        labels_fn=_sorted_module_labels,
    ),
    TopologyScenario(
        name="uniform_base_modular_effect",
        base_kind="uniform",
        mask_fn=_scenario_mask_within_module,
        mask_params={"module_idx": 0},
        labels_fn=_sorted_module_labels,
    ),
    TopologyScenario(
        name="within_plus_between",
        base_kind="modular",
        mask_fn=_scenario_mask_within_plus_between,
        mask_params={"within_module_idx": 0, "between_module_a": 1, "between_module_b": 2},
        labels_fn=_sorted_module_labels,
    ),
    TopologyScenario(
        name="perfect_matching_within_module",
        base_kind="modular",
        mask_fn=_scenario_mask_perfect_matching_within_module,
        mask_params={"module_idx": 0},
        labels_fn=_sorted_module_labels,
    ),
    TopologyScenario(
        name="cross_block_connected_chain",
        base_kind="modular",
        mask_fn=_scenario_mask_cross_block_connected_chain,
        mask_params={"length": 30},
        labels_fn=_sorted_module_labels,
    ),
    TopologyScenario(
        name="rich_club",
        base_kind="modular",
        mask_fn=_scenario_mask_rich_club,
        mask_params={"n_hubs": 10, "n_spokes_per_hub": 0},
        labels_fn=_sorted_module_labels,
    ),
    TopologyScenario(
        name="two_equal_disconnected_cliques",
        base_kind="modular",
        mask_fn=_scenario_mask_two_disconnected_cliques,
        mask_params={"module_idx": 0, "clique_size": 6},
        labels_fn=_sorted_module_labels,
    ),
    TopologyScenario(
        name="focal_clique_inside_module",
        base_kind="modular",
        mask_fn=_scenario_mask_focal_clique_inside_module,
        mask_params={"module_idx": 0, "clique_size": 5},
        labels_fn=_sorted_module_labels,
    ),
    TopologyScenario(
        name="partial_bipartite_between_modules",
        base_kind="modular",
        mask_fn=_scenario_mask_partial_bipartite_between_modules,
        mask_params={"module_a": 1, "module_b": 2, "n_a": 6, "n_b": 6},
        labels_fn=_sorted_module_labels,
    ),
    TopologyScenario(
        name="gradient_effect_chain",
        base_kind="modular",
        mask_fn=_scenario_mask_gradient_effect_chain,
        mask_params={"length": 30, "min_weight": 0.2},
        labels_fn=_sorted_module_labels,
    ),
    TopologyScenario(
        name="gradient_core_periphery_within_module",
        base_kind="modular",
        mask_fn=_scenario_mask_gradient_core_periphery_within_module,
        mask_params={
            "module_idx": 0,
            "n_core": 6,
            "core_weight": 1.0,
            "core_to_periphery_weight": 0.4,
            "periphery_weight": 0.1,
        },
        labels_fn=_sorted_module_labels,
    ),
    TopologyScenario(
        name="imbalanced_modules_within_effect",
        base_kind="modular",
        mask_fn=_scenario_mask_within_module,
        mask_params={"module_idx": 0},
        labels_fn=_imbalanced_module_labels,
    ),
]

_SCENARIO_BY_NAME: Dict[str, TopologyScenario] = {s.name: s for s in _SCENARIOS}


def list_scenarios() -> List[str]:
    return [s.name for s in _SCENARIOS]


def get_scenarios() -> List[TopologyScenario]:
    return list(_SCENARIOS)


def get_scenario(scenario: Union[str, TopologyScenario]) -> TopologyScenario:
    if isinstance(scenario, TopologyScenario):
        return scenario
    if not isinstance(scenario, str):
        raise TypeError("scenario must be a scenario name (str) or TopologyScenario.")
    try:
        return _SCENARIO_BY_NAME[scenario]
    except KeyError as exc:
        raise ValueError(f"Unknown scenario: {scenario!r}. Available: {list_scenarios()}") from exc


def _build_scenarios() -> List[TopologyScenario]:
    """Backward-compatible alias (kept for older example scripts)."""
    return get_scenarios()
