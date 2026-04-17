__version__ = "2.0.0"

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

from .eeg_utils import (
    read_from_eeg_dataframe,
    reshape_eeg_data,
    inverse_reshape_eeg_data,
    EEGData,
    Electrodes,
    Bands,
    PairsElectrodes1020
)
from .nbs_score import nbs_bct
from .pairwise_stats import (
    compute_p_val,
    compute_null_dist,
    compute_t_stat,
    compute_diffs,
    compute_t_stat_diff,
    compute_t_stat_ind,
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
    get_tfnbs_score_networkx,
    get_tfnbs_score,
    get_network_informed_tfnbs_score,
    get_fbc_tfnbs_score,
)
from .utils import fisher_r_to_z, fisher_z_to_r, get_components, binarize

from .glm_stats import (
    GLMStatType,
    compute_glm_stat,
    compute_p_val_glm,
    build_design_matrix,
)
