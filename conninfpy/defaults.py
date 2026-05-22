"""
Unified default parameters for ConnInfPy.

Single source of truth for all default constants used across modules.

TFCE exponent defaults (E, H) use Smith & Nichols (2009) values
``E=0.5, H=2.0`` across both the scoring and permutation paths.

Hao et al. (2024) recommend ``E=0.4, H in [3.0, 7.0]`` for empirical
FDR < 10% on network data — Baggio (2018) defaults overshoot FDR > 50%
in their benchmarks; Vinokur et al. (2023) report 75-fold variation in
detected edge counts across (E,H) within Baggio's recommended range.
The package does not impose Hao's defaults globally to preserve
backward compatibility, but every validation pipeline in
``examples/`` (ABIDE, Open-Close, simulation) calls
``compute_p_val(...)`` / ``compute_p_val_glm(...)`` with explicit
``e=0.4, h=3.0, n=10`` — that is the regime our paper figures use.

Users tuning (E,H) for a specific dataset should treat the exponents as
a sensitivity knob: ablate within Hao's recommended box
``E in [0.3, 0.5]``, ``H in [3.0, 7.0]`` and report stability.

The only path-specific default is ``n`` (threshold integration steps):
``n=100`` for scoring (high resolution, single-shot), ``n=10`` for
permutation (lower ``n`` keeps the permutation loop affordable).

The rationale for ``n=10`` inside the permutation loop differs between
the two error-control regimes ConnInfPy supports, and the literature
strictly speaking only validates one of them at this resolution:

* **FDR control (``method='bh_fdr_perm'``).** Hao et al. (2024)
  empirically benchmark TFNOS at ``E=0.4, H=3.0, n=10`` on synthetic
  network data and report FDR < 10% — this is the regime their result
  underwrites. Using ``n=10`` for the permutation-calibrated BH path
  is directly supported.

* **FWER control via max-statistic permutation.** No specific
  literature endorses ``n=10`` for this path; the choice is a
  pragmatic compromise between integration resolution and permutation
  budget. ConnInfPy's simulation calibration grid
  (1 000 null repeats × 9 methods × 4 sample sizes, see
  ``examples/simulation_validation/``) confirms that the
  one-sided FWER ≤ α rule still holds in all 36 cells at ``n=10``,
  but this is an empirical observation rather than a theoretical
  guarantee — users with budget for ``n=30+`` permutation-time
  integration steps can pass ``n=30`` to ``compute_p_val(...)`` /
  ``compute_p_val_glm(...)`` without changing any other default.

Threshold tie-breaking convention
---------------------------------
ConnInfPy's TFCE integral uses ``edge_t >= threshold`` (inclusive lower
bound) at each integration step. Smith & Nichols (2009) define the
TFCE indicator as ``e(h) = 1 if t > h``, i.e. strict inequality. The
inclusive convention is mathematically equivalent under continuous
``t``-statistics and computationally simpler. We retain ``>=`` because
the production FPR calibration on 1{,}000 null repeats × 9 methods × 4
sample sizes (see ``examples/simulation_validation/``) confirms all
36 cells PASS the one-sided FWER ≤ α rule with this choice. A future
release will switch to ``>`` to match the literal Smith & Nichols
formula and rerun calibration as a regression check.

References
----------
- Smith & Nichols (2009). Threshold-free cluster enhancement.
  NeuroImage 44(1):83-98.
- Baggio et al. (2018). TFNBS: a threshold-free method for the analysis
  of brain networks. Hum Brain Mapp 39(6):2289-2302.
- Vinokur et al. (2023). Parameter sensitivity of NBS / TFNBS in
  connectivity inference.
- Hao et al. (2024). TFNOS: threshold-free network omnibus statistics
  (recommends ``E=0.4, H in [3.0, 7.0]`` for FDR < 10%).
"""

# =============================================================================
# TFCE parameters (Smith & Nichols, 2009)
# =============================================================================

DEFAULT_EXTENT_EXPONENT: float = 0.5
"""Default extent exponent E. See module docstring for Hao 2024 alternative."""

DEFAULT_HEIGHT_EXPONENT: float = 2.0
"""Default height exponent H. See module docstring for Hao 2024 alternative."""

DEFAULT_START_THRESHOLD: float = 1.65
"""Initial threshold for cluster formation (p < 0.05 one-tailed)."""

# =============================================================================
# Integration resolution (path-specific)
# =============================================================================

DEFAULT_N_THRESHOLDS_SCORING: int = 100
"""Threshold integration steps for direct TFCE scoring (high resolution)."""

DEFAULT_N_THRESHOLDS_PERMUTATION: int = 10
"""Threshold integration steps inside the permutation loop (speed tradeoff;
Hao 2024 reports n=10 is sufficient for FDR control on network data)."""

# =============================================================================
# Permutation testing
# =============================================================================

DEFAULT_N_PERMUTATIONS: int = 1000
"""Default number of permutations for null distribution."""

# =============================================================================
# NBS defaults
# =============================================================================

DEFAULT_NBS_THRESHOLD: float = 2.0
"""Default t-statistic threshold for classical NBS edge inclusion."""

DEFAULT_NBS_STAT: str = "extent"
"""Default cluster statistic type for NBS ('extent' or 'intensity')."""

# =============================================================================
# FBC-TFNBS
# =============================================================================

DEFAULT_MIN_CLUSTER_SIZE: int = 3
"""Default minimum edges in a functional block to form a cluster (for FBC-TFNBS)."""
