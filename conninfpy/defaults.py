"""
Unified default parameters for TFNBS package.

Single source of truth for all default constants used across modules.
Parameters follow Hao et al. (2024) for TFCE on network data.
"""

# =============================================================================
# TFCE parameters (Smith & Nichols, 2009)
# =============================================================================

DEFAULT_EXTENT_EXPONENT: float = 0.5
"""Default extent exponent (E) for TFCE. Controls how much cluster size matters."""

DEFAULT_HEIGHT_EXPONENT: float = 2.0
"""Default height exponent (H) for TFCE. Controls how much threshold height matters."""

DEFAULT_START_THRESHOLD: float = 1.65
"""Default initial threshold for cluster formation (corresponds to p < 0.05 one-tailed)."""

# =============================================================================
# Integration resolution
# =============================================================================

DEFAULT_N_THRESHOLDS_SCORING: int = 100
"""Default number of threshold integration steps for direct TFCE scoring."""

DEFAULT_N_THRESHOLDS_PERMUTATION: int = 10
"""Default number of threshold steps during permutation testing (speed tradeoff)."""

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
