__version__ = "1.0.1"

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
    compute_t_stat_tfnbs,
    compute_t_stat_tfnbs_diffs,
    compute_t_stat,
    compute_diffs,
    compute_t_stat_diff,
    compute_t_stat_ind)

from .tfnbs_score import (
    get_tfnbs_score_networkx,
    get_tfnbs_score,
    get_network_informed_tfnbs_score,
    get_fbc_tfnbs_score,
    DEFAULT_MIN_CLUSTER_SIZE
)
from .utils import fisher_r_to_z, fisher_z_to_r, get_components, binarize

#__all__ = ['nbs_bct', 'compute_p_val', 'compute_t_stat_diff']
