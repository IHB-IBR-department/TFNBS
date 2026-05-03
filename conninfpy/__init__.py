"""ConnInfPy — Connectivity Inference in Python.

Permutation-based statistical inference on brain connectivity networks (fMRI,
EEG). One permutation engine shared across nine inference methods, edge-wise
GLM (Freedman--Lane), parametric empirical-Bayes ComBat, and tail-approximation
acceleration.

Public quick-start
------------------
>>> from conninfpy import compute_p_val, fisher_r_to_z
>>> z = fisher_r_to_z(corr_matrices)
>>> p = compute_p_val(g1_z, g2_z, test_type='two-sample',
...                   method='tfnbs', e=0.4, h=3.0, n=10)
>>> p['positive']  # significant edges where group2 > group1

Returns a :class:`~conninfpy._compat.TailResult` with canonical keys
``'positive'`` and ``'negative'``. The legacy keys ``'g2>g1'`` and
``'g1>g2'`` from v1.x remain readable but emit a :class:`DeprecationWarning`
and will be removed in v2.1.
"""
__version__ = "2.0.0"

from ._analyze import AnalyzeResult, analyze
from ._compat import TailResult
from ._result import InferenceResult
from .defaults import (
    DEFAULT_EXTENT_EXPONENT,
    DEFAULT_HEIGHT_EXPONENT,
    DEFAULT_START_THRESHOLD,
    DEFAULT_N_THRESHOLDS_SCORING,
    DEFAULT_N_THRESHOLDS_PERMUTATION,
    DEFAULT_N_PERMUTATIONS,
    DEFAULT_NBS_THRESHOLD,
    DEFAULT_NBS_STAT,
    DEFAULT_MIN_CLUSTER_SIZE,
)

from .synth_datasets import generate_fc_matrices, ModularDatasetGenerator
from .topologies import (
    TopologyDataset,
    TopologyDatasetGenerator,
    TopologyScenario,
    get_scenario,
    get_scenarios,
    list_scenarios,
)

from .eeg_utils import (
    read_from_eeg_dataframe,
    reshape_eeg_data,
    inverse_reshape_eeg_data,
    EEGData,
    Electrodes,
    Bands,
    PairsElectrodes1020,
)
from .nbs_score import nbs_bct
from .pairwise_stats import (
    compute_p_val,
    compute_null_dist,
    compute_t_stat,
    compute_t_stat_diff,
    StatMethod,
    TestType,
)

from ._enhancement import (
    apply_tfnbs,
    apply_nbs,
    apply_cnbs,
    apply_ni_tfnbs,
    apply_fbc_tfnbs,
)

from .tfnbs_score import (
    get_tfnbs_score,
    get_network_informed_tfnbs_score,
    get_fbc_tfnbs_score,
)
from .utils import (
    fisher_r_to_z,
    fisher_z_to_r,
    get_components,
    binarize,
    create_prior_weights,
)

from .glm_stats import (
    GLMStatType,
    compute_glm_stat,
    compute_p_val_glm,
    compute_p_val_glm_multi,
    compute_p_val_paired_glm,
    build_design_matrix,
)

from .harmonize import (
    ComBatModel,
    CombatResult,
    combat_harmonize,
    combat_fit,
    combat_apply,
    compute_vif,
    design_diagnostics,
    flatten_upper,
    unflatten_upper,
    block_mass,
)

from .acceleration import (
    fit_gpd_tail,
    fit_gamma_tail,
    compute_p_values_accelerated,
)


__all__ = [
    # version + result types
    "__version__",
    "TailResult",
    "InferenceResult",
    "AnalyzeResult",
    "analyze",
    # defaults
    "DEFAULT_EXTENT_EXPONENT",
    "DEFAULT_HEIGHT_EXPONENT",
    "DEFAULT_START_THRESHOLD",
    "DEFAULT_N_THRESHOLDS_SCORING",
    "DEFAULT_N_THRESHOLDS_PERMUTATION",
    "DEFAULT_N_PERMUTATIONS",
    "DEFAULT_NBS_THRESHOLD",
    "DEFAULT_NBS_STAT",
    "DEFAULT_MIN_CLUSTER_SIZE",
    # statistical pipelines
    "compute_p_val",
    "compute_null_dist",
    "compute_t_stat",
    "compute_t_stat_diff",
    "StatMethod",
    "TestType",
    "compute_p_val_glm",
    "compute_p_val_glm_multi",
    "compute_p_val_paired_glm",
    "compute_glm_stat",
    "build_design_matrix",
    "GLMStatType",
    # enhancement (apply_*)
    "apply_tfnbs",
    "apply_nbs",
    "apply_cnbs",
    "apply_ni_tfnbs",
    "apply_fbc_tfnbs",
    # raw scoring
    "get_tfnbs_score",
    "get_network_informed_tfnbs_score",
    "get_fbc_tfnbs_score",
    "nbs_bct",
    # acceleration
    "fit_gpd_tail",
    "fit_gamma_tail",
    "compute_p_values_accelerated",
    # harmonization + design diagnostics
    "ComBatModel",
    "CombatResult",
    "combat_harmonize",
    "combat_fit",
    "combat_apply",
    "compute_vif",
    "design_diagnostics",
    "flatten_upper",
    "unflatten_upper",
    "block_mass",
    # synthetic data + topologies
    "generate_fc_matrices",
    "ModularDatasetGenerator",
    "TopologyDataset",
    "TopologyDatasetGenerator",
    "TopologyScenario",
    "get_scenario",
    "get_scenarios",
    "list_scenarios",
    # utilities
    "fisher_r_to_z",
    "fisher_z_to_r",
    "get_components",
    "binarize",
    "create_prior_weights",
    # EEG
    "read_from_eeg_dataframe",
    "reshape_eeg_data",
    "inverse_reshape_eeg_data",
    "EEGData",
    "Electrodes",
    "Bands",
    "PairsElectrodes1020",
]
