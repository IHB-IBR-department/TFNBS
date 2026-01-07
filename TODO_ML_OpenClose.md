# TODO: Open/Close (HCP) — Reproducibility + Significant-Edge Methods + ML Transfer (IHB → RMET)

This TODO is a **detailed, implementation-oriented plan** to evaluate how different edge-selection/statistical methods behave on the **real Open/Close resting-state** dataset in:

- `datasets/OpenClose/HCP/corr/open_ihb.npy`, `datasets/OpenClose/HCP/corr/close_ihb.npy` (paired, `n=84`)
- `datasets/OpenClose/HCP/corr/open_rmet.npy`, `datasets/OpenClose/HCP/corr/close_rmet.npy` (paired, `n=63`)
- Atlas metadata: `datasets/OpenClose/HCP/HCPex_atlas_description.xlsx` (373 ROIs)

Key idea: we can test both **paired** and **two-sample** inference on the same data, then evaluate:
1) **reproducibility** of the Open/Close difference pattern across subsets (`ihb` vs `rmet`, and resampling), and
2) **downstream usefulness** of selected edges for **cross-experiment ML transfer** (train on `ihb`, test on `rmet`).

This plan is written to align with the existing API:
- `tfnbs/pairwise_stats.py::compute_p_val`
- `tfnbs/tfnbs_score.py` and `tfnbs/nbs_score.py` (methods)

---

## 0) Scope, definitions, and conventions (do first)

### 0.1 Data conventions
- Matrices are already **Fisher z-transformed** (per `datasets/OpenClose/HCP/corr/corr_code.txt`).
- Matrices are symmetric with zero diagonal (quickly confirmed on a sample).
- Ordering is said to be sorted by `HCPex_ID` (left hemi first, then right).

### 0.2 Naming / direction conventions (avoid sign confusion)
When using `compute_p_val(group1, group2, test_type="paired")`:
- set `group1 = open`, `group2 = close` (same subject order)
- diffs are computed internally as `close - open`
- outputs:
  - `p_vals["g2>g1"]`: **close > open** effects (positive diffs)
  - `p_vals["g1>g2"]`: **open > close** effects (negative diffs, made positive)

For all reports, keep the two tails separate and optionally combine into a signed map:
- `signed_mask = +1*(p_close_gt_open < alpha) - 1*(p_open_gt_close < alpha)`

### 0.3 Methods to compare (names map to code)
Using `tfnbs/pairwise_stats.py::StatMethod`:
- `tstat`: t-statistic + **permutation max correction** (edgewise map compared to null max)
- `nbs`: classic NBS (extent / intensity) + max correction over cluster score
- `cnbs`: constrained NBS (block mean) + max correction over block score
- `tfnbs`: standard TFNBS (topological clustering via shared-node components)
- `fbc_tfnbs`: FBC-TFNBS (functional-block clustering, needs `net_labels`)
- `ni_tfnbs`: **NO topological clustering** (edge-level block-density weighting; needs `net_labels`)

In your wording, “no-tfnbs” most naturally corresponds to `ni_tfnbs` (no topological clustering). Keep `tstat` as a separate baseline.

---

## 1) Inputs, labels, and basic QC

### 1.1 Load data (single source of truth)
Implement a loader utility (script-level or a small module) that returns:
- `open_ihb, close_ihb`: `(84, 373, 373)`
- `open_rmet, close_rmet`: `(63, 373, 373)`
- `n_nodes = 373`, `tri = np.triu_indices(n_nodes, k=1)`

Also verify:
- symmetry tolerance (should be exact symmetric here)
- diagonal all zeros
- finite values only

### 1.2 Build `net_labels` from the Excel atlas description
From `datasets/OpenClose/HCP/HCPex_atlas_description.xlsx`:
- sort rows by `HCPex_ID` to match `.npy` order
- produce two candidate `net_labels` arrays (shape `(373,)`, `int`):
  1) `Cortical_Division_Number` (24 unique values)
  2) `ColeAnticevic_functional_network` (14 unique values)

Important implementation detail:
- Many constrained scorers internally remap labels to contiguous `0..K-1` (this repo does), but do it explicitly anyway for safety when saving/plotting:
  - `labels_unique, labels_inv = np.unique(net_labels_raw, return_inverse=True)`
  - use `labels_inv` as the `net_labels` passed into `compute_p_val`

Deliverables:
- a small “labels summary” table: counts per label; label name mapping from the Excel columns
- a plot that shows block boundaries if needed (optional)

### 1.3 Define “subsets” for reproducibility checks
We have two natural subsets:
- **IHB** (84) and **RMET** (63): different experiments, different sample sizes
Additionally, create within-experiment resamples:
- split-half (multiple random seeds)
- bootstrap-by-subject (paired resampling)

### 1.4 Precompute p-value maps via config runner
Use the config-driven runner to generate and save p-maps per method/experiment:
- Config: `examples/openclose_pvals_config.yaml` (paired design, methods, parameters)
- Runner: `examples/openclose_pvals_runner.py`
- Command: `python -m examples.openclose_pvals_runner --config examples/openclose_pvals_config.yaml`
- Outputs:
  - per-method files: `results/openclose_hcp/pvals/pvals_<dataset>_<experiment>_<method_key>.pkl`
  - combined file: `results/openclose_hcp/pvals/pvals_<dataset>_<experiment>.pkl`
  - logs: `results/openclose_hcp/pvals/logs/*.log` (elapsed time + job done)

Notes:
- TFNBS-family methods output 3D p-maps when `e`/`h` are lists.
- NBS threshold lists and FBC min-cluster lists are concatenated to 3D maps.

---

## 2) Core analysis A — reproducibility of Open/Close difference patterns

Goal: quantify how stable the “difference signal” is under:
- dataset subset (`ihb` vs `rmet`)
- sampling variability (split-half / bootstrap)
- test design choice (paired vs two-sample)
- statistical method choice

### 2.1 Baseline “effect maps” (no multiple-comparisons decisions)
Compute these maps for each subset (`ihb`, `rmet`) and each design:

**Paired design**
- `t_signed = mean(close-open) / (std(close-open)/sqrt(n))` (signed, not separated)
- also store the repo’s separated one-tail `t_dict` via `compute_t_stat(open, close, test_type="paired")`

**Two-sample design (independent)**
- treat `open` and `close` as independent groups:
  - `compute_t_stat(open, close, test_type="two-sample")`

Similarity metrics between maps (report separately for each tail and also signed):
- Spearman correlation of:
  - signed `t` maps (upper triangle)
  - `-log10(p)` maps (upper triangle), when available
- Cosine similarity of signed `t` vectors (optional)

### 2.2 Method outputs: p-value maps and significance masks
For each subset (`ihb`, `rmet`), each design (`paired`, `two-sample`), and each method:
- compute `p_vals = compute_p_val(...)`
- create significance masks at `alpha`:
  - `mask_pos = (p_vals["g2>g1"] < alpha)`  (close > open)
  - `mask_neg = (p_vals["g1>g2"] < alpha)`  (open > close)
  - enforce symmetry and zero diagonal for masks

Metrics per (method, tail):
- `n_sig_edges`: number of significant edges (upper triangle count)
- stability of `n_sig_edges` under resampling (std / IQR)

Between-subset reproducibility metrics (ihb vs rmet):
- Jaccard overlap of masks (per tail): `|A∩B| / |A∪B|`
- Dice overlap (per tail): `2|A∩B| / (|A|+|B|)`
- “signed overlap”: agreement on direction among union of discovered edges

### 2.3 Resampling stability (within ihb and within rmet)
For each experiment separately, run resampling loops:

**Split-half**
- split subjects into halves (keep pairs aligned)
- compute p-maps/masks in each half
- quantify overlap between halves
- repeat `R` times (e.g., `R=50`) with fixed random seeds

**Bootstrap-by-subject**
- sample subjects with replacement, keeping paired open+close together
- compute p-maps/masks
- aggregate:
  - “selection frequency” per edge
  - stability of top edges / top blocks

Deliverables:
- a table: mean±std Jaccard across resamples per method
- an “edge selection frequency” heatmap for 1–2 best methods (optional; expensive)

### 2.4 Network/block-level reproducibility (uses `net_labels`)
For interpretability and cross-method comparison, aggregate to blocks:
- define block id for an edge as `(min(label_i,label_j), max(label_i,label_j))`
- per block compute:
  - number of significant edges
  - mean `-log10(p)` among significant edges
  - total mass: `sum(-log10(p))`

Then compute between-subset similarity on **block summaries**:
- correlation between block “mass” vectors
- overlap of top-K blocks (K=10, 20)

Run block-level analyses for both label schemes:
- `Cortical_Division_Number` (24 blocks)
- `ColeAnticevic_functional_network` (14 blocks)

---

## 3) Core analysis B — compare edge-extraction methods directly

### 3.1 Standardize decision rules
Because methods produce different score structures, standardize post-processing:
- primary: `p < alpha` (FWER max-corrected p-values from `compute_p_val`)
- secondary (optional): BH-FDR on the upper triangle per tail, to compare “liberal discovery modes”

Define a fixed alpha grid for comparability:
- `alpha ∈ {0.1, 0.05, 0.01, 0.005, 0.001}`

### 3.2 Parameter sweeps (start small, then refine)
Because 373×373 with permutations is expensive, do two stages:

**Stage 1 (cheap screening; e.g. 200 permutations)**
- `tstat`: no params
- `nbs`: `threshold ∈ {2.0, 2.5, 3.0}` × `nbs_stat ∈ {extent, intensity}`
- `tfnbs`: `e ∈ {0.4, 0.5}`, `h ∈ {2.0, 3.0}`, `n ∈ {10, 20}`, `start_thres ∈ {1.65, 2.0}`
- `cnbs`: depends only on `net_labels`
- `ni_tfnbs`: same grid as `tfnbs`, requires `net_labels`
- `fbc_tfnbs`: same grid + `min_cluster_size ∈ {2, 3, 5}`, requires `net_labels`

**Stage 2 (confirmation; e.g. 1000–5000 permutations)**
- pick 2–4 best settings per method based on:
  - split-half stability
  - ihb↔rmet reproducibility
  - interpretability (block concentration)
  - runtime feasibility

Deliverables:
- runtime table per method/setting (wall-time and sec/permutation)
- “stability vs parameters” summary (small multiples or a table)

---

## 4) ML transfer experiment — validate edge-detection methods (train ihb, test rmet)

Goal: **validate approaches for Open/Close network-difference detection** by checking whether the edges they recover (via `compute_p_val` p-maps) provide measurable benefit to a simple ML classifier, and **at what alpha level**.

Important: ML here is **not** used to generate edge p-values. Edge subsets come from the **statistical p-value maps** produced by the methods in `tfnbs/pairwise_stats.py::compute_p_val`.

### 4.0 Step-by-step execution plan (use saved p-maps)
Inputs:
- p-maps folder: `results/openclose_hcp/pvals/` (per-method pkl files)
- train data: IHB (open/close)
- test data: RMET (open/close)

Steps:
1) Load IHB/RMET data and build feature matrix from upper triangle (`tri`).
2) Fit **baseline** on all edges:
   - train: IHB (grouped CV by subject, tune C)
   - test: RMET
   - metrics: accuracy, balanced accuracy, ROC-AUC
3) For each per-method p-map file:
   - read `p_vals` and parameter metadata (`param_meta`), including param order.
   - for each parameter index:
     - compute `p_union = min(p_pos, p_neg)` (close>open vs open>close).
     - build mask: `p_union < alpha` (for each alpha grid value).
4) Train ML using only selected edges:
   - use the same IHB CV protocol as baseline.
   - evaluate on RMET with the final selected mask and tuned C.
5) Save results to `ml_transfer_results.csv`:
   - columns: method, method_key, param_idx, param_json (or param_* cols), alpha,
     n_edges, ihb_cv_accuracy, ihb_cv_bal_acc, ihb_cv_roc_auc,
     rmet_accuracy, rmet_bal_acc, rmet_roc_auc, baseline_flag, pmap_file.

Notes:
- Parameter order is stored in each p-map file:
  - NBS: `param_meta["param_pairs"]` (threshold, nbs_stat)
  - TFNBS: `param_meta["param_pairs"]` (e, h)
  - FBC-TFNBS: `param_meta["param_triplets"]` (min_cluster_size, e, h)
- For 2D p-maps, treat as single param set (`param_idx=0`).

### 4.1 Define the ML problem
Binary classification:
- label `y=0` for `open`, `y=1` for `close`

Dataset construction (treat open/close as samples, but keep pairing for splits):
- For `ihb`: 84 subjects → 168 samples
- For `rmet`: 63 subjects → 126 samples

Leakage constraint:
- Any train/validation split inside `ihb` must be **grouped by subject** so `open` and `close` from the same subject never split across folds.
  - Use `GroupKFold` / `GroupShuffleSplit` with `groups=subject_id`.

### 4.2 Feature representation and baseline model
Vectorize each correlation matrix to the upper triangle:
- `tri = np.triu_indices(n_nodes, k=1)`
- `x = matrix[tri]`, `p = 373*372/2 = 69378` features

Standardization:
- fit scaler on **ihb train** only; apply to ihb val and rmet test

Baseline classifier:
- Logistic Regression (L2, strong regularization)
- tune `C ∈ {1e-4, 1e-3, 1e-2, 1e-1, 1.0}` using grouped CV on `ihb`

Metrics:
- ROC-AUC, balanced accuracy, accuracy
- report also #features used (full vs subset size)

### 4.3 Building edge subsets from `compute_p_val` p-maps (per method)
For each method `m` (and a chosen parameter setting):
1) Compute p-maps **only on the training data** (ihb or ihb-fold-train):
   - `p_vals = compute_p_val(open_train, close_train, test_type="paired", method=m, ...)`
2) Convert to an undirected “edge significance” map to select features:
   - two tails: `p_pos = p_vals["g2>g1"]` (close>open), `p_neg = p_vals["g1>g2"]` (open>close)
   - recommended selection score: `p_union = np.minimum(p_pos, p_neg)`
     - rationale: for classification we care that an edge differs, not which direction; direction stays in the feature value.
   - select edges by `p_union < alpha`

Alpha as hyperparameter (main grid):
- `alpha ∈ {0.1, 0.05, 0.01, 0.005, 0.001}`

Note: `compute_p_val` returns max-corrected p-values (FWER-style) by construction; interpret `alpha` accordingly.

Optional sensitivity variants:
- direction-specific features: build two masks, `p_pos<alpha` and `p_neg<alpha`, and compare union vs separate
- BH-FDR on the upper triangle of `p_union` (if you want a more “discovery-oriented” mask)

### 4.4 ML evaluation protocol: “full edges vs method-selected subsets”
For each method `m` and each `alpha`:
1) On `ihb` (grouped CV):
   - in each fold:
     - compute method p-maps on fold-train only
     - derive mask at the current `alpha`
     - train LR on fold-train using only selected edges
     - evaluate on fold-val
   - aggregate mean±std performance and mean #edges selected
2) Choose the best `alpha` per method (and best `C` if re-tuned) **using ihb CV only**
3) Final evaluation (single shot):
   - compute mask on full `ihb` train using the chosen method+params+alpha
   - train LR on `ihb` using that mask
   - test on `rmet`

Always include baselines:
- LR on **all edges** (same `C` tuning protocol)
- optionally LR on random edge subsets with matched subset size (to quantify “selection benefit beyond dimensionality reduction”)

Deliverables:
- table: method × alpha → (#edges, ihb-CV metrics, rmet-test metrics)
- plot: performance vs alpha for each method (and #edges vs alpha)

### 4.5 Connect ML results back to network labels
For each stats-derived mask used in ML (especially best-performing alpha per method):
- summarize selected edges per block using `net_labels`:
  - counts and “mass” (e.g., mean `-log10(p_union)` over selected edges)
- compare block profiles across methods (rank correlation / overlap of top blocks)

This answers: “which method selects edges that both generalize and concentrate in interpretable networks?”

---

## 5) Recommended file/script layout (when implementing)

Suggested new scripts (examples-level, no library changes at first):
- `examples/openclose_hcp_reproducibility.py`
  - loads ihb/rmet
  - runs compute_t_stat / compute_p_val across methods
  - produces reproducibility tables + a small set of plots
- `examples/openclose_hcp_ml_transfer.py`
  - builds LR baseline
  - implements within-subject permutation for coefficient p-values
  - evaluates ihb→rmet transfer under different edge masks

Reusable helpers (optional, if scripts get big):
- `tfnbs/openclose_hcp.py` (data+labels loader, vectorization helpers)
- `tfnbs/ml_permutation.py` (paired label-swap permutation for LR)

Outputs (avoid touching `examples/output/` unless explicitly desired):
- store to a new top-level folder: `results/openclose_hcp/`
  - `results/openclose_hcp/tables/*.csv`
  - `results/openclose_hcp/figures/*.png`
  - `results/openclose_hcp/configs/*.json`
  - `results/openclose_hcp/artifacts/*.npz` (p-maps, masks, coefficients)

---

## 6) “Done” criteria / checklist

### Reproducibility
- [ ] Signed t-map correlation: `ihb` vs `rmet` (paired and two-sample)
- [ ] Per-method `p-map` similarity: `ihb` vs `rmet`
- [ ] Per-method mask overlap (Jaccard/Dice): `ihb` vs `rmet`
- [ ] Split-half stability (mean±std) per method
- [ ] Block-level summaries and their stability

### ML transfer
- [ ] Baseline LR tuned on ihb, evaluated on rmet
- [ ] Performance vs `alpha` and #edges selected per stats method
- [ ] Comparison: full-edges baseline vs method-selected subsets
- [ ] Block-level interpretation of selected edges

---

## 7) Open decisions to confirm before implementation (quick answers help)
1) For constrained methods (`cnbs`, `ni_tfnbs`, `fbc_tfnbs`), which label scheme is primary?
   - `Cortical_Division_Number` (24) vs `ColeAnticevic_functional_network` (14)
2) Should ML edge selection use `p_union = min(p_pos, p_neg)` (recommended) or select only one direction (close>open / open>close)?
3) Compute budget: realistic `n_permutations` for `compute_p_val` per fold and for the final ihb→rmet evaluation?
4) Do you want to tune `alpha` per method via ihb CV, or fix a small set (e.g., 0.05/0.01) and compare directly?
