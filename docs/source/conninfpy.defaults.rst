conninfpy.defaults
==================

Single source of truth for default parameters used across the package.

TFCE exponents (E, H) follow Smith & Nichols (2009) values across both the
scoring and permutation paths. Hao et al. (2024) recommend
``E=0.4, H ∈ [3.0, 7.0]`` for empirical FDR < 10% on network data — the
package does not impose this globally to preserve backward compatibility,
but every validation pipeline in ``examples/`` passes
``e=0.4, h=3.0, n=10`` explicitly. Vinokur et al. (2023) report 75-fold
edge-count variation across (E, H) within Baggio's recommended range —
sensitivity is real and should be reported.

The only path-specific default is ``n`` (threshold integration steps):
``n=100`` for direct scoring (high resolution, single-shot), ``n=10`` for
permutation (Hao 2024 reports n=10 is sufficient for FDR control).

.. automodule:: conninfpy.defaults
   :members:
   :undoc-members:
   :show-inheritance:
