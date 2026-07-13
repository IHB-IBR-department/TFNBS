conninfpy.defaults
==================

Single source of truth for default parameters used across the package.

TFCE exponents (E, H) follow Smith & Nichols (2009) values across both the
scoring and permutation paths. Standard validation settings use
``E=0.3, H=3.0`` for empirical FDR control on network data — the
package uses these as global defaults since v2.1. every validation
pipeline in ``examples/`` passes
``e=0.3, h=3.0, n=10`` explicitly. Vinokur et al. (2023) report 75-fold
edge-count variation across (E, H) within Baggio's recommended range —
sensitivity is real and should be reported.

The only path-specific default is ``n`` (threshold integration steps):
``n=100`` for direct scoring (high resolution, single-shot), ``n=10`` for
permutation (Hao 2024 reports n=10 is sufficient for FDR control).

.. automodule:: conninfpy.defaults
   :members:
   :undoc-members:
   :show-inheritance:
