# Methods: Network-Based Statistics for Connectome Inference

This section summarizes the scoring methods implemented in `tfnbs/tfnbs_score.py` and `tfnbs/nbs_score.py`. The focus is on the neuroscientific rationale and the statistical definitions, not code.

## Data and Statistical Framework

We analyze undirected functional connectomes represented as symmetric connectivity matrices (N x N, zero diagonal). Group contrasts are tested edge-wise with standard t-statistics, and positive/negative effects are treated separately (two one-tailed maps). Statistical inference uses permutation testing (paired, one-sample, or two-sample designs), with family-wise error rate (FWER) controlled by max-statistic across the enhanced score map.

## Classical NBS (Network-Based Statistic)

**Neuroscience rationale:** Effects in brain networks are spatially structured; true effects tend to form connected patterns rather than isolated edges. NBS boosts sensitivity by aggregating evidence within connected edge components.

**Definition:**
- Threshold the t-statistic matrix at a fixed value tau.
- Form a graph of suprathreshold edges (edges share a node).
- For each connected component, compute a cluster statistic:
  - **Extent:** number of edges in the component.
  - **Intensity:** sum of t-values in the component.
- Assign each suprathreshold edge the component statistic; others remain zero.

This corresponds to the scoring used by `get_nbs_score`.

## TFNBS (Threshold-Free NBS)

**Neuroscience rationale:** Cluster-defining thresholds are arbitrary and can miss weak but spatially distributed effects. TFNBS integrates evidence across a range of thresholds to reduce threshold dependence.

**Definition:** For each edge e,

TFNBS(e) = integral over h of [S(h, e)]^E * h^H dh,

where h is the threshold level, S(h, e) is the size of the connected component containing edge e at threshold h, and E/H are extent and height exponents. This is the graph analogue of TFCE and captures both focal and extended effects.

This corresponds to `get_tfnbs_score` (and the baseline/networkx variants for validation).

## cNBS (Constrained NBS)

**Neuroscience rationale:** Many effects are organized by large-scale functional systems (e.g., DMN, visual, motor). cNBS shifts inference to these predefined subnetworks.

**Definition (scoring step):**
- Use a network parcellation to define blocks (within- and between-network edge sets).
- For each block, compute the mean of all edge-wise t-statistics in that block.
- Assign each edge the mean of its block (blockwise mean score).

This corresponds to `get_cnbs_score`. Inference is then performed by permutation testing with max-statistic correction across subnetworks.

## NI-TFNBS (Network-Informed TFNBS)

**Neuroscience rationale:** Standard TFNBS is topologically sensitive but anatomically blind. NI-TFNBS introduces a soft anatomical prior: edges within dense functional blocks should contribute more to evidence than edges in sparse blocks.

**Block weighting:** For each functional block at threshold h:
- k = number of suprathreshold edges in the block.
- M = total possible edges in the block.
- Weight W_block = k / sqrt(M).

**Score:** For each topological component, the support is the sum of W_block over its edges. This weighted support replaces the unweighted component size in the TFNBS integral.

This corresponds to `get_network_informed_tfnbs_score`.

## FBC-TFNBS (Functional Block Clustering TFNBS)

**Neuroscience rationale:** Some disorders manifest as diffuse, block-confined effects (e.g., within-network hypoconnectivity) that do not form strong topological components. FBC-TFNBS clusters edges by functional block membership rather than node adjacency, suppressing isolated edges.

**Definition:** At each threshold h:
- Group suprathreshold edges by functional block (network pair).
- If a block has k >= min_cluster_size edges, those edges support each other with support = k.
- Blocks with fewer edges are suppressed (support = 0).
- Integrate support over thresholds using the TFNBS formula.

This corresponds to `get_fbc_tfnbs_score` and implements the “functional block clustering” idea highlighted in the project analyses.

## Summary of Method Roles

- **NBS:** Detects connected topological effects; sensitive to clusters defined by node adjacency.
- **TFNBS:** Threshold-free extension of NBS; integrates effects across thresholds.
- **cNBS:** Block-level inference; summarizes effects within predefined networks.
- **NI-TFNBS:** Hybrid of topology and functional priors via block-density weighting.
- **FBC-TFNBS:** Block-defined clustering that promotes within-network coherence and suppresses isolated edges.

These methods are designed for fMRI/EEG connectomics, where effects are expected to align with large-scale network organization rather than isolated edges.
