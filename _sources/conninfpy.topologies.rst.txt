conninfpy.topologies
====================

Named topology scenarios for benchmarking enhancement methods on
synthetic connectivity. Promoted from ``examples/`` to the public API
on 2026-04-18.

Provides 19 named scenarios covering the topologies most likely to
arise in real connectivity data: within-module-dense,
between-modules-dense, hub, rich-club, scattered, chain,
gradient-core-periphery, partial-bipartite, fragmented-within-module,
cross-block-connected-chain, plus secondary variants.

Public surface:

- :class:`TopologyDatasetGenerator` — main generator class
- :class:`TopologyDataset` — output container (``.group1``, ``.group2``,
  ``.mask``, ``.net_labels``, ``.scenario``)
- :class:`TopologyScenario` — scenario specification
- :func:`list_scenarios` — return all scenario names
- :func:`get_scenarios` — return all scenario specs
- :func:`get_scenario` — look up a specific scenario by name

The MICCAI 2026 power benchmark
(``examples/miccai_paper_reproducing/``) sweeps all enhancement methods
across all 19 topologies; its findings are summarized in the
"no-method-dominates-across-topologies" empirical message of the
ConnInfPy paper.

.. automodule:: conninfpy.topologies
   :members:
   :undoc-members:
   :show-inheritance:
