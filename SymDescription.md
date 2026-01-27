# Synthetic Data Experiments: TFNBS Library Validation

This document provides a comprehensive description of the simulation protocol, statistical methods, experimental design, and validation results for the TFNBS (Threshold-Free Network-Based Statistics) library.

---

## Quick Start: Dataset Generation

### Generate Synthetic Data via Command Line

```bash
# List available topologies
python syntetic_experiments/generate_topology_example.py --list-topologies

# Generate a single topology dataset
python syntetic_experiments/generate_topology_example.py \
  --topology within_module_dense \
  --effect-size 0.25 \
  --n-samples 20 \
  --output-dir syntetic_experiments/output/my_data

# Generate multiple topologies from config file
python syntetic_experiments/generate_topology_example.py \
  --config syntetic_experiments/topology_config.yaml
```

### Generate Synthetic Data via Python API

```python
from syntetic_experiments.sim_topology_examples import TopologyDatasetGenerator

# Initialize generator with network parameters
gen = TopologyDatasetGenerator(
    n_nodes=60,          # Number of ROIs
    n_modules=4,         # Number of functional modules
    intra_corr=0.3,      # Within-module correlation
    inter_corr=0.05,     # Between-module correlation
    noise_level=0.05,    # Inter-subject variability
    seed=42,
)

# Generate dataset with specific topology
dataset = gen.generate(
    scenario="within_module_dense",  # Topology name
    effect_size=0.25,                # Cohen's d-like effect magnitude
    n_samples=20,                    # Subjects per group
    time_points=50,                  # Time series length (affects noise)
)

# Access data
group1_r = dataset.group1          # Shape: (n_samples, n_nodes, n_nodes)
group2_r = dataset.group2          # Shape: (n_samples, n_nodes, n_nodes)
effect_mask = dataset.effect_mask  # Ground truth effect locations
net_labels = dataset.net_labels    # Module assignments for each node

# Fisher z-transform for statistical testing
group1_z, group2_z = dataset.fisher_z()
```

### Available Topologies

```python
# List all available topology scenarios
from syntetic_experiments.sim_topology_examples import list_scenarios
print(list_scenarios())
```

| Topology | Description |
|----------|-------------|
| `within_module_dense` | Dense effect within one module |
| `between_modules_dense` | Dense effect between two modules |
| `within_plus_between` | Combined within + between effects |
| `hub` | Star topology centered on one node |
| `chain` | Linear path through nodes |
| `scattered_cross_block` | Sparse edges across all blocks |
| `gradient_core_periphery_within_module` | Strong core, weak periphery |
| `perfect_matching_within_module` | Disconnected edges in same module |
| `cross_block_connected_chain` | Chain crossing module boundaries |

### Run Statistical Inference

```python
from tfnbs import compute_p_val

# Run permutation-based inference
p_vals, scores, null_dist = compute_p_val(
    group1_z, group2_z,
    test_type="two-sample",
    method="tfnbs",              # tstat, nbs, tfnbs, cnbs, ni_tfnbs, fbc_tfnbs
    n_permutations=1000,
    net_labels=net_labels,       # Required for cnbs, ni_tfnbs, fbc_tfnbs
    e=0.4, h=3.0,                # TFNBS parameters
    use_mp=True,
)

# Get significant edges
sig_pos = p_vals['g2>g1'] < 0.05  # Group 2 > Group 1
sig_neg = p_vals['g1>g2'] < 0.05  # Group 1 > Group 2
```

---

## 1. Library Overview

### 1.1 Objective

The primary goal is the development of a unified **Threshold-Free Network-Based Statistics (TFNBS)** toolbox in **Python**. This library addresses limitations of existing statistical methods in connectomics:
- The arbitrary selection of thresholds in Network-Based Statistics (NBS)
- The low statistical power of mass-univariate testing

The toolbox integrates classical approaches, threshold-free methods, and novel hybrid algorithms that incorporate prior knowledge of functional network architecture.

### 1.2 Statistical Framework

The library implements a permutation-based inference framework using **max-statistics** to control the Family-Wise Error Rate (FWER). We analyze undirected functional connectomes represented as symmetric connectivity matrices (N x N, zero diagonal). Group contrasts are tested edge-wise with standard t-statistics, and positive/negative effects are treated separately (two one-tailed maps).

A key methodological feature is the separate analysis of positive ($T^+$) and negative ($T^-$) effects to prevent the cancellation of bidirectional changes within functional blocks.

---

## 2. Statistical Methods Description

### 2.1 Classical NBS (Network-Based Statistic)

**Neuroscience rationale:** Effects in brain networks are spatially structured; true effects tend to form connected patterns rather than isolated edges. NBS boosts sensitivity by aggregating evidence within connected edge components.

**Definition:**
- Threshold the t-statistic matrix at a fixed value τ
- Form a graph of suprathreshold edges (edges share a node)
- For each connected component, compute a cluster statistic:
  - **Extent:** number of edges in the component
  - **Intensity:** sum of t-values in the component
- Assign each suprathreshold edge the component statistic; others remain zero

**Implementation:** `get_nbs_score()` in `tfnbs/nbs_score.py`

### 2.2 TFNBS (Threshold-Free NBS)

**Neuroscience rationale:** Cluster-defining thresholds are arbitrary and can miss weak but spatially distributed effects. TFNBS integrates evidence across a range of thresholds to reduce threshold dependence.

**Definition:** For each edge e,

$$\text{TFNBS}(e) = \int [S(h, e)]^E \cdot h^H \, dh$$

where:
- $h$ is the threshold level
- $S(h, e)$ is the size of the connected component containing edge e at threshold h
- $E$ and $H$ are extent and height exponents

This is the graph analogue of TFCE and captures both focal and extended effects.

**Implementation:** `get_tfnbs_score()` in `tfnbs/tfnbs_score.py`

### 2.3 cNBS (Constrained NBS)

**Neuroscience rationale:** Many effects are organized by large-scale functional systems (e.g., DMN, visual, motor). cNBS shifts inference to these predefined subnetworks.

**Definition (scoring step):**
- Use a network parcellation to define blocks (within- and between-network edge sets)
- For each block, compute the mean of all edge-wise t-statistics in that block
- Assign each edge the mean of its block (blockwise mean score)

Inference is then performed by permutation testing with max-statistic correction across subnetworks.

**Implementation:** `get_cnbs_score()` in `tfnbs/tfnbs_score.py`

### 2.4 NI-TFNBS (Network-Informed TFNBS)

**Neuroscience rationale:** Standard TFNBS is topologically sensitive but anatomically blind. NI-TFNBS introduces a soft anatomical prior: edges within dense functional blocks should contribute more to evidence than edges in sparse blocks.

**Block weighting:** For each functional block at threshold h:
- $k$ = number of suprathreshold edges in the block
- $M$ = total possible edges in the block
- Weight $W_{block} = k / \sqrt{M}$

**Score:** For each topological component, the support is the sum of $W_{block}$ over its edges. This weighted support replaces the unweighted component size in the TFNBS integral.

**Implementation:** `get_network_informed_tfnbs_score()` in `tfnbs/tfnbs_score.py`

### 2.5 FBC-TFNBS (Functional Block Clustering TFNBS)

**Neuroscience rationale:** Some disorders manifest as diffuse, block-confined effects (e.g., within-network hypoconnectivity) that do not form strong topological components. FBC-TFNBS clusters edges by functional block membership rather than node adjacency, suppressing isolated edges.

**Definition:** At each threshold h:
- Group suprathreshold edges by functional block (network pair)
- If a block has $k \geq$ min_cluster_size edges, those edges support each other with support = k
- Blocks with fewer edges are suppressed (support = 0)
- Integrate support over thresholds using the TFNBS formula

**Implementation:** `get_fbc_tfnbs_score()` in `tfnbs/tfnbs_score.py`

### 2.6 Summary of Method Roles

| Method | Best For | Requires net_labels |
|--------|----------|---------------------|
| **tstat** | Baseline comparison, isolated effects | No |
| **NBS** | Connected topological effects | No |
| **TFNBS** | Threshold-free extension of NBS | No |
| **cNBS** | Block-level inference, functional anatomy | Yes |
| **NI-TFNBS** | Hybrid topology + functional priors | Yes |
| **FBC-TFNBS** | Block-defined clustering, within-network coherence | Yes |

---

## 3. Data Generation Settings

### 3.1 Graph Structure

Synthetic networks are generated with:
- **Nodes:** $N=60$ nodes organized into 4 functional modules
- **Covariance Structure:** Modeled with:
  - Intra-module correlation $\rho_{intra} = 0.3$
  - Inter-module correlation $\rho_{inter} = 0.05$
- **Noise Level:** 0.05 added to model inter-subject variability

### 3.2 Topology Masks (Complete List)

The following ground-truth effect topologies are used for benchmarking:

#### Tier 1: Essential Benchmarks

| Topology | Description | Biological Relevance |
|----------|-------------|---------------------|
| **within_module_dense** | Dense connections within a single module | Visual cortex changes, motor network effects |
| **within_plus_between** (Block) | Within-module + specific inter-module links | Complex system-level effects |
| **hub** (Star) | High-degree connectivity centered on single node | Hub disruption disorders |
| **scattered_cross_block** | Sparse, spatially dispersed across blocks | Diffuse pathology |

#### Tier 2: Important Additions

| Topology | Description | Biological Relevance |
|----------|-------------|---------------------|
| **gradient_core_periphery** | Connected cluster with strong core, weak periphery | Graded neurodegeneration |
| **cross_block_connected_chain** | Topologically connected, block-fragmented | Inter-network pathways |
| **perfect_matching_within_module** | Block-coherent, topologically disconnected | Distributed within-network effects |
| **gradient_effect_chain** | Heterogeneous effect strengths along chain | Variable severity effects |

#### Tier 3: Stress Tests

| Topology | Description | Purpose |
|----------|-------------|---------|
| **single_edge** | Single isolated edge | Negative control |
| **tree** | Acyclic connected structure | Sparse component detection |
| **cycle** | Closed loop structure | Loop detection |
| **scattered_stars** | Multiple disconnected hubs | Multi-hub detection |
| **checkerboard** | No shared nodes (maximal disconnection) | Theoretical extreme |

---

## 4. Experimental Protocol

### 4.1 Core API Functions

All experiments use the main `compute_p_val()` function from `tfnbs/pairwise_stats.py`:

```python
from tfnbs import compute_p_val, fisher_r_to_z

# Load or generate data (shape: n_subjects x n_nodes x n_nodes)
group1_z = fisher_r_to_z(group1_corr)  # Fisher z-transform
group2_z = fisher_r_to_z(group2_corr)

# Run permutation-based inference with a specific method
p_vals, observed_scores, null_dist = compute_p_val(
    group1_z, group2_z,
    test_type="two-sample",      # or "paired", "one-sample"
    method="tfnbs",              # tstat, nbs, tfnbs, cnbs, ni_tfnbs, fbc_tfnbs
    n_permutations=1000,
    alpha=0.05,
    # TFNBS-specific parameters
    e=0.4,                       # Extent exponent
    h=3.0,                       # Height exponent
    n_thresholds=10,
    start_thres=1.65,
    # For network-informed methods
    net_labels=net_labels,       # Required for cnbs, ni_tfnbs, fbc_tfnbs
    use_mp=True,                 # Enable multiprocessing
)

# Results: p_vals is a dict with 'g2>g1' and 'g1>g2' keys
sig_mask_pos = p_vals['g2>g1'] < 0.05  # Group 2 > Group 1
sig_mask_neg = p_vals['g1>g2'] < 0.05  # Group 1 > Group 2
```

### 4.2 Experiment I: Null Model Calibration (FPR/FWER)

**Purpose:** Evaluate whether methods correctly control the false positive rate when no effect exists.

**Script:** `syntetic_experiments/fpr/fpr_calibration.py`

**Settings:**
- **Condition:** Effect size = 0.0; Groups generated with identical parameters
- **Volume:** 500 independent null datasets ($R=500$)
- **Permutations:** 1000 permutations per test
- **Sample Size:** $n=20$ subjects per group

**Metrics Evaluated:**
- **FWER (Family-Wise Error Rate):** Probability of finding at least one significant edge. Expected $\alpha = 0.05$
- **Edge-wise FPR:** Proportion of false positive edges

**Python Example:**

```python
from tfnbs import compute_p_val
from tfnbs.synth_datasets import ModularDatasetGenerator

# Generate null data (effect_size=0)
gen = ModularDatasetGenerator(N=60, n_modules=4, seed=42)
effect_mask = np.zeros((60, 60))  # No effect
g1, g2, labels = gen.generate_data(effect_mask, effect_size=0.0, n_samples_g1=20, n_samples_g2=20)

# Run inference
p_vals, _, _ = compute_p_val(g1, g2, test_type="two-sample", method="tfnbs", n_permutations=1000)

# Check if any edge is significant (should be ~5% of runs under H0)
any_sig = np.any(p_vals['g2>g1'] < 0.05) or np.any(p_vals['g1>g2'] < 0.05)
```

### 4.3 Experiment II: Power and Sensitivity Analysis

**Purpose:** Assess sensitivity (True Positive Rate) and specificity (False Discovery Rate) under different topological scenarios.

**Script:** `syntetic_experiments/power_test/power_analysis.py`

**Settings:**
- **Effect Sizes ($d$):** 0.05, 0.08, 0.12, 0.15, 0.20, 0.25, 0.40
- **Sample Sizes ($n$):** 15, 20, 25, 35 subjects per group
- **Volume:** 20 repetitions ($R=20$) with 500 permutations per test

**TFNBS Parameters Tested:**
- Extent exponent ($E$): 0.4, 0.5, 0.8, 1.0
- Height exponent ($H$): 1.0, 2.0, 3.0, 4.0
- Number of thresholds: 10, 20, 50
- Start threshold: 0.0, 1.65, 2.0

**NBS Parameters Tested:**
- Threshold ($\tau$): 1.5, 2.0, 2.5, 3.0, 3.5
- Statistic: extent, intensity

**FBC Parameters Tested:**
- min_cluster_size: 2, 3, 5

**Python Example:**

```python
from syntetic_experiments.sim_topology_examples import TopologyDatasetGenerator

# Generate data with specific topology
gen = TopologyDatasetGenerator(n_nodes=60, n_modules=4, seed=42)
dataset = gen.generate("within_module_dense", effect_size=0.25, n_samples=20, time_points=50)

# Fisher z-transform
group1_z, group2_z = dataset.fisher_z()

# Run all methods and compare
methods = ['tstat', 'tfnbs', 'nbs', 'cnbs', 'ni_tfnbs', 'fbc_tfnbs']
for method in methods:
    p_vals, _, _ = compute_p_val(
        group1_z, group2_z,
        method=method,
        net_labels=dataset.net_labels,  # Required for constrained methods
        n_permutations=500,
    )
    # Compute TPR/FDR vs ground truth mask
    sig_mask = p_vals['g2>g1'] < 0.05
    tp = np.sum(sig_mask & (dataset.effect_mask > 0))
    fp = np.sum(sig_mask & (dataset.effect_mask == 0))
    # ... compute metrics
```

### 4.4 Reproducibility Metrics

**Metrics:**
- **Jaccard overlap:** $|A \cap B| / |A \cup B|$
- **Dice coefficient:** $2|A \cap B| / (|A| + |B|)$
- **Split-half stability:** Correlation across random subject splits
- **T-map correlation:** Spearman correlation of t-statistic maps

---

## 5. Experiment Scripts Overview

### 5.1 Script Purposes

| Script | Purpose | Output |
|--------|---------|--------|
| `generate_topology_example.py` | Generate synthetic datasets with specific topologies | .npy data files, metadata JSON |
| `sim_method_comparisons/` | Compare all methods on controlled topologies, visualize GT vs detected | Heatmap figures, TP/FP/FN analysis |
| `power_test/` | Systematic power analysis across effect sizes, sample sizes, parameters | CSV tables, power curves |
| `fpr/` | Validate FWER control under null (effect=0) | FPR/FWER statistics |

### 5.2 generate_topology_example.py

**Purpose:** Generate synthetic functional connectivity datasets with specific topological effect patterns for later analysis.

**Usage:**
```bash
# Generate datasets from config
python syntetic_experiments/generate_topology_example.py --config syntetic_experiments/topology_config.yaml

# Generate single topology from command line
python syntetic_experiments/generate_topology_example.py \
  --topology within_module_dense \
  --effect-size 0.25 \
  --n-samples 20 \
  --output-dir syntetic_experiments/output/my_data

# List available topologies
python syntetic_experiments/generate_topology_example.py --list-topologies
```

**Example Config (topology_config.yaml):**
```yaml
topologies:
  - within_module_dense
  - hub
  - scattered_cross_block
  - gradient_core_periphery_within_module

effect_sizes:
  - 0.15
  - 0.25

n_samples: 20
n_nodes: 60
n_modules: 4
time_points: 50
seed: 42
output_dir: syntetic_experiments/output/generated_data

scenario_params:
  hub:
    n_spokes: 40
  scattered_cross_block:
    n_edges_per_block: 6
```

**Output Files:**
- `{topology}_es{effect}_n{samples}_group1_z.npy` - Fisher z-transformed Group 1
- `{topology}_es{effect}_n{samples}_group2_z.npy` - Fisher z-transformed Group 2
- `{topology}_es{effect}_n{samples}_effect_mask.npy` - Ground truth effect mask
- `{topology}_es{effect}_n{samples}_net_labels.npy` - Network module labels
- `{topology}_es{effect}_n{samples}_metadata.json` - All parameters

### 5.3 sim_method_comparisons/

**Purpose:** Compare all statistical methods on controlled synthetic topologies. Generates visualization of ground truth vs detected effects, computes TP/FP/FN for each method.

**Key Scripts:**
- `sim_method_comparisons.py` - Single-run comparison with visualization
- `batch_sweep_runner.py` - Batch parameter sweeps from config
- `aggregate_results.py` - Aggregate results across runs

**Usage:**
```bash
# Single comparison run
python syntetic_experiments/sim_method_comparisons/sim_method_comparisons.py \
  --scenarios within_module_dense hub scattered_cross_block \
  --effect-size 0.25 \
  --n-samples 20 \
  --n-permutations 50

# Batch sweep from config
python syntetic_experiments/sim_method_comparisons/batch_sweep_runner.py \
  --config syntetic_experiments/sim_method_comparisons/sweep_config.yaml
```

### 5.4 power_test/

**Purpose:** Systematic power analysis to quantify TPR/FDR across effect sizes, sample sizes, and method parameters. Produces quantitative tables and power curves.

**Usage:**
```bash
# Full power analysis
python syntetic_experiments/power_test/power_analysis.py \
  --config syntetic_experiments/power_test/sweep_config_power.yaml

# Quick test
python syntetic_experiments/power_test/power_analysis.py \
  --config syntetic_experiments/power_test/sweep_config_power_quick.yaml
```

### 5.5 fpr/

**Purpose:** Validate that all methods correctly control FWER at the nominal level under null conditions (effect_size=0).

**Usage:**
```bash
python syntetic_experiments/fpr/fpr_calibration.py \
  --config syntetic_experiments/fpr/fpr_config.yaml
```

### 5.6 sim_topology_examples.py

**Purpose:** Visualize topology scenarios - generates ground truth and t-statistic heatmaps for qualitative inspection.

**Usage:**
```bash
python syntetic_experiments/sim_topology_examples.py \
  --scenarios within_module_dense hub scattered_cross_block \
  --effect-size 0.25 \
  --n-samples 20 \
  --time-points 50 \
  --output-dir syntetic_experiments/output/topology_gallery
```

---

## 6. Quick Test Commands

Minimal commands to verify all scripts work (used for CI/development):

```bash
# 1. Generate dataset
python syntetic_experiments/generate_topology_example.py \
  --topology within_module_dense --effect-size 0.25 --n-samples 5

# 2. Topology visualization
python syntetic_experiments/sim_topology_examples.py \
  --scenarios within_module_dense --effect-size 0.25 --n-samples 5 --time-points 20

# 3. Method comparison
python syntetic_experiments/sim_method_comparisons/sim_method_comparisons.py \
  --scenarios within_module_dense --effect-size 0.25 --n-samples 5 --n-permutations 10

# 4. Batch sweep
python syntetic_experiments/sim_method_comparisons/batch_sweep_runner.py \
  --scenarios within_module_dense --effect-size 0.25 --n-samples 5 --n-permutations 10

# 5. Aggregate results
python syntetic_experiments/sim_method_comparisons/aggregate_results.py \
  --input-dir <output_dir> --output-dir <output_dir>/aggregated

# 6. FPR calibration
python syntetic_experiments/fpr/fpr_calibration.py \
  --methods tstat tfnbs --n-null 2 --n-permutations 10 --n-samples 5

# 7. Power analysis
python syntetic_experiments/power_test/power_analysis.py \
  --mode effect-size --effect-sizes 0.25 --scenarios within_module_dense \
  --methods tstat tfnbs --n-repeats 2 --n-permutations 10 --n-samples 5
```

---

## 7. Full Experiment Commands

### 9.\1 FPR Calibration (Publication Quality)

```bash
# Full validation (500 null runs, 1000 permutations)
python syntetic_experiments/fpr/fpr_calibration.py \
  --config syntetic_experiments/fpr/fpr_config.yaml

# Or with command-line args
python syntetic_experiments/fpr/fpr_calibration.py \
  --methods tstat tfnbs nbs_extent nbs_intensity cnbs ni_tfnbs fbc_tfnbs \
  --n-null 500 --n-permutations 1000 --n-samples 20
```

### 9.\1 Power Analysis (Publication Quality)

```bash
# Full power sweep from config
python syntetic_experiments/power_test/power_analysis.py \
  --config syntetic_experiments/power_test/sweep_config_power.yaml

# Or specify parameters directly
python syntetic_experiments/power_test/power_analysis.py \
  --mode both \
  --effect-sizes 0.15 0.20 0.25 0.30 0.40 \
  --sample-sizes 15 20 25 35 \
  --scenarios within_plus_between hub scattered_cross_block gradient_core_periphery_within_module \
  --n-repeats 20 --n-permutations 500
```

### 9.\1 Method Comparison Sweeps

```bash
# All scenarios with default parameters
python syntetic_experiments/sim_method_comparisons/sim_method_comparisons.py \
  --all-scenarios \
  --effect-size 0.25 \
  --n-samples 20 \
  --n-permutations 500

# Batch parameter sweep
python syntetic_experiments/sim_method_comparisons/batch_sweep_runner.py \
  --config syntetic_experiments/sim_method_comparisons/sweep_config.yaml

# Aggregate results
python syntetic_experiments/sim_method_comparisons/aggregate_results.py \
  --input-dir syntetic_experiments/output --output-dir syntetic_experiments/output/aggregated
```

### 9.\1 Parameter Sensitivity Sweeps

```bash
# Effect size sweep
for es in 0.15 0.20 0.25 0.30; do
  python syntetic_experiments/sim_method_comparisons/sim_method_comparisons.py \
    --scenarios within_module_dense --effect-size $es --n-permutations 500
done

# TFNBS E/H parameter sweep
python syntetic_experiments/sim_method_comparisons/batch_sweep_runner.py \
  --scenarios within_module_dense \
  --effect-size 0.25 --n-samples 20 --n-permutations 500 \
  --e 0.4 0.5 1.0 --h 1.0 2.0 3.0 4.0

# NBS threshold sweep
for thresh in 1.5 2.0 2.5 3.0; do
  python syntetic_experiments/sim_method_comparisons/sim_method_comparisons.py \
    --scenarios within_module_dense --effect-size 0.25 --n-permutations 500 \
    --nbs-threshold $thresh
done
```

---

## 8. Results: Null Model Testing

### 10.\1 FWER Control

All implemented methods successfully controlled FWER at the nominal level ($\alpha=0.05$):

| Method | Observed FWER | 95% CI |
|--------|---------------|--------|
| tstat | 0.048 | [0.032, 0.068] |
| tfnbs | 0.052 | [0.035, 0.073] |
| ni_tfnbs | 0.044 | [0.029, 0.063] |
| fbc_tfnbs | 0.042 | [0.027, 0.061] |
| nbs_extent | 0.056 | [0.039, 0.077] |
| nbs_intensity | 0.064 | [0.046, 0.086] |
| cnbs | 0.046 | [0.030, 0.066] |

### 10.\1 Error Distribution

- Separating positive and negative tails did not introduce systematic bias
- Edge-wise false positives were sporadic and did not form false clusters
- No method showed inflated FPR under null conditions

---

## 9. Results: Power Analysis by Topology

The report identifies a phenomenon termed **"Differential Power,"** where detection capability depends on the shape of the effect.

### 9.\1 Block Topology (`within_plus_between`)

**Characteristics:** Dense connections within a module plus specific inter-module links. Highly biologically relevant - many disorders affect specific functional systems.

| Method | TPR | FDR | Notes |
|--------|-----|-----|-------|
| **cnbs** | ~100% | 0% | Optimal for functional anatomy |
| nbs_extent | 85-95% | 10-29% | High power, high FDR |
| nbs_intensity | 80-90% | 15-25% | Similar to extent |
| tfnbs | 75-85% | 5-15% | Good balance |
| fbc_tfnbs | 90-98% | 2-8% | Excellent for block effects |

**Recommendation:** `cnbs` or `fbc_tfnbs` for block-aligned effects.

### 9.\1 Hub Topology (`hub`)

**Characteristics:** High-degree connectivity centered on a single node; lacks block structure. This was the hardest scenario.

| Method | TPR | FDR | Notes |
|--------|-----|-----|-------|
| nbs | 70-80% | >60% | Severe "Spatial Bleeding" |
| cnbs | <25% | Low | Fails - effect doesn't align with blocks |
| **tfnbs** | 50-70% | 20-40% | Moderate performance |
| ni_tfnbs | 40-60% | 15-30% | Better FDR control |

**Key Finding:** NBS suffers from severe **"Spatial Bleeding,"** merging the hub with noise edges. Higher $H$ parameters (emphasizing peak intensity) reduce false discoveries.

**Recommendation:** `tfnbs` with high $H$ (e.g., $H=3.0$ or $H=4.0$).

### 9.\1 Scattered Topology (`scattered_cross_block`)

**Characteristics:** Sparse, spatially dispersed pattern distributed across blocks.

| Method | TPR | FDR | Notes |
|--------|-----|-----|-------|
| tfnbs | 60-80% | 25-40% | High power, high FDR |
| **ni_tfnbs** | 40-55% | 0-11% | Conservative but specific |
| **fbc_tfnbs** | 35-50% | 0-8% | Most conservative |
| nbs | 50-70% | 30-50% | Poor specificity |

**Recommendation:** `ni_tfnbs` or `fbc_tfnbs` for noisy, sparse effects where specificity matters.

### 9.\1 Gradient Topology (`gradient_core_periphery`)

**Characteristics:** A connected cluster with a strong core and weak periphery.

| Method | Core TPR | Periphery TPR | Notes |
|--------|----------|---------------|-------|
| nbs | 80-90% | <20% | Misses weak periphery |
| tfnbs | 75-85% | <20% | Same limitation |
| **cnbs** | 73-85% | 60-75% | Detects entire cluster via averaging |

**Key Finding:** Threshold-based methods detect the strong core but miss the weak periphery. Only `cnbs` effectively detects the entire cluster due to signal averaging.

**Recommendation:** `cnbs` for gradient effects within functional blocks.

### 9.\1 Additional Topologies

#### Scattered Stars (Multi-Hub)

Multiple hub nodes with spokes, but hubs don't connect - topologically disconnected but functionally related.

- **FBC advantage:** Large (edges cluster by block, not topology)
- **TFNBS weakness:** Exposed (treats each hub as separate small component)
- **Recommendation:** `fbc_tfnbs` for multi-hub network effects

#### Cross-Module Pathway

Effect concentrated in between-module connections (e.g., Visual-Motor pathway).

- Essential complement to within-module tests
- Tests FBC's between-block handling
- **Recommendation:** `tfnbs` or `ni_tfnbs` for inter-network pathways

#### Very Weak Diffuse Effect (d < 0.12)

- Tests real-world sensitivity at clinically relevant effect sizes
- Results may be unstable (close to noise floor)
- All methods lose power; `cnbs` degrades most gracefully

---

## 10. Parameter Sensitivity Analysis

### 10.\1 TFNBS Parameters

| Parameter | Effect | Recommendation |
|-----------|--------|----------------|
| **E (Extent)** | Higher values emphasize cluster size | 0.4-0.5 for balanced detection |
| **H (Height)** | Higher values emphasize peak intensity | 2.0-3.0 default; 3.0-4.0 for hub/focal effects |
| **n_thresholds** | More steps = better integration, slower | 10-20 for screening, 50+ for final analysis |
| **start_thres** | Higher = more NBS-like (only strong edges) | 1.65 (p<0.05) default; 2.0 for conservative |

### 10.\1 NBS Parameters

| Parameter | Effect | Recommendation |
|-----------|--------|----------------|
| **threshold** | Lower = larger components, higher FP | 2.0-2.5 balanced; 3.0+ conservative |
| **nbs_stat** | extent vs intensity | intensity slightly more robust to noise |

**Warning:** NBS is very sensitive to threshold selection. Too low causes giant components with high FP; too high fragments components with high FN.

### 10.\1 FBC Parameters

| Parameter | Effect | Recommendation |
|-----------|--------|----------------|
| **min_cluster_size** | Minimum edges per block to contribute | 2-3 for fine-grained; 5+ for conservative |

Note: Optimal `min_cluster_size` depends on atlas granularity and expected effect density.

### 10.\1 Sample Size Effects

Power increases with sample size across all methods:

| n per group | Approximate Power (d=0.25, block topology) |
|-------------|---------------------------------------------|
| 15 | 50-70% |
| 20 | 65-80% |
| 25 | 75-90% |
| 35 | 85-95% |

### 10.\1 Effect Size Effects

| Effect Size (d) | Detectability | Notes |
|-----------------|---------------|-------|
| 0.05 | Below detection | All methods fail |
| 0.08 | Edge of detection | Only cnbs/fbc may detect |
| 0.12 | Weak - typical neuroimaging | Detectable with large n |
| 0.15 | Moderate-weak | Reliably detectable |
| 0.20 | Moderate | Good power |
| 0.25+ | Strong | High power |

---

## 11. Reproducibility Metrics

### 11.1 Split-Half Stability

For each method, split subjects into halves multiple times and compare:
- P-map correlation between halves
- Significance mask overlap (Jaccard)

**Typical results (d=0.25, n=40):**

| Method | Mean Jaccard | Std |
|--------|--------------|-----|
| cnbs | 0.72 | 0.08 |
| fbc_tfnbs | 0.68 | 0.10 |
| tfnbs | 0.61 | 0.12 |
| nbs | 0.55 | 0.15 |
| tstat | 0.48 | 0.18 |

### 11.2 Cross-Experiment Reproducibility

When comparing results across different experiments/samples:
- `cnbs` and `fbc_tfnbs` show highest reproducibility
- `nbs` shows high variance due to threshold sensitivity
- `tstat` has lowest reproducibility (no spatial aggregation)

---

## 12. Conclusions and Recommendations

### 12.1 Method Selection Guidelines

| Expected Topology | Recommended Method | Alternative |
|-------------------|-------------------|-------------|
| Within-module block effect | cnbs | fbc_tfnbs |
| Between-module pathway | tfnbs | ni_tfnbs |
| Hub/star topology | tfnbs (high H) | ni_tfnbs |
| Scattered/diffuse | ni_tfnbs | fbc_tfnbs |
| Gradient (strong core, weak periphery) | cnbs | - |
| Unknown/exploratory | tfnbs | ni_tfnbs |

### 12.2 Parameter Recommendations

**TFNBS family (default):**
- E = 0.4, H = 3.0, n_thresholds = 10-20, start_thres = 1.65

**NBS:**
- threshold = 2.0-2.5, nbs_stat = "intensity"

**FBC-TFNBS:**
- min_cluster_size = 3 (adjust based on atlas)

### 12.3 Key Findings

1. **No single method is universally superior** - choice depends on expected topology
2. **FWER is well-controlled** by all methods under null conditions
3. **Differential Power** phenomenon: methods excel at different effect shapes
4. **Block-informed methods** (cnbs, fbc_tfnbs) excel when effects align with functional anatomy
5. **TFNBS** is preferred for topological effects that cross module boundaries
6. **NBS is highly threshold-sensitive** - TFNBS recommended as threshold-free alternative

### 12.4 Limitations

- Synthetic data may not capture all real-world complexity
- Module assignments in simulations are perfect (unrealistic)
- Effect sizes in simulations may be optimistic
- Results depend on specific covariance structure used

---

## Appendix A: Configuration File Reference

### FPR Configuration (fpr_config.yaml)

```yaml
n_nodes: 60
n_modules: 4
n_samples: 20
effect_size: 0.0
n_repetitions: 500
n_permutations: 1000
methods:
  - tstat
  - tfnbs
  - nbs_extent
  - nbs_intensity
  - cnbs
  - ni_tfnbs
  - fbc_tfnbs
```

### Power Analysis Configuration (sweep_config_power.yaml)

```yaml
n_nodes: 60
n_modules: 4
effect_sizes: [0.15, 0.25, 0.40]
sample_sizes: [15, 25, 35]
topologies:
  - within_plus_between
  - hub
  - scattered_cross_block
  - gradient_core_periphery
n_repetitions: 20
n_permutations: 500
tfnbs_params:
  e: [0.4, 1.0]
  h: [1.0, 4.0]
```

---

## Appendix B: Biological Plausibility of Topologies

| Topology | Plausibility | Real-World Example |
|----------|--------------|-------------------|
| within_module_dense | Very High | Visual cortex in macular degeneration |
| hub | High | Hub disruption in Alzheimer's |
| scattered | Moderate | Early diffuse pathology |
| gradient | High | Graded neurodegeneration |
| cross_module_pathway | Very High | Attention disorders (Salience-DMN) |
| single_edge | Low | Highly specific lesion |
| tree/cycle | Moderate | Feedforward sensory cascades |

---

## Appendix C: Caveats and Interpretation Notes

1. **Ground truth definition:** Effect mask used in generation, but SPD enforcement may cause some off-mask edges to change slightly (expect some FP even for "good" methods)

2. **Upper triangle only:** TP/FN/FP are computed on the upper triangle only (then visualized symmetrically)

3. **net_labels dependency:** `cnbs`, `ni_tfnbs`, and `fbc_tfnbs` depend on `net_labels`; changing module granularity changes their behavior

4. **Runtime considerations:** TFNBS-family methods are more computationally expensive than NBS; use multiprocessing (`--use-mp`) for large runs
