# TODO: Real-Data Method Comparison Plan (Open/Close + ABIDE)

## Goal
Compare `compute_p_val(..., method=...)` methods on real connectome datasets, quantify sensitivity to parameters (TFNBS/NBS), and benchmark runtime scaling for large atlases.

Primary outputs:
- P-value maps per method (and significance masks under multiple decision rules).
- Between-method agreement (overlap/correlation) and resampling stability (edges + blocks).
- Sensitivity maps: outputs vs parameters (TFNBS: `e,h,n,start_thres`; NBS: `threshold,nbs_stat`; FBC: `min_cluster_size`).
- Runtime + memory benchmarks of `compute_p_val` across methods and atlas sizes.
- Optional ML-based *external validation* (not feature selection): do significant edges align with predictive signal?

Datasets:
- **Open/Close**: state classification (often paired/repeated measures).
- **ABIDE**: autism vs control classification, with atlas-based **functional labels** (`net_labels`) available.

---

## Important note about leakage (scope clarification)
This plan is **not** about using significant edges as features inside an ML pipeline. The primary goal is **method-to-method comparison** on the same data, so any “leakage” is shared across methods.

If ML is used at all, it is only as an *external reference* for “does the method highlight edges that look predictive?”, and all methods must use identical splits.

---

## Step 0 — Define standard inputs
For each subject/session sample:
- `X`: connectivity matrix, shape `(N, N)`, symmetric.
- Optional: `subject_id`, `site_id` (ABIDE), demographics/motion confounds.
- `net_labels`: shape `(N,)`, atlas functional labels (required for `cnbs`, `ni_tfnbs`, `fbc_tfnbs`).

Preprocessing conventions:
- Set diagonal to `0`.
- Apply Fisher r→z (`tfnbs.utils.fisher_r_to_z`) before stats.
- Keep a consistent node ordering (critical for `net_labels`).

---

## Step 1 — Dataset-specific testing modes

### A) Open/Close (likely paired / repeated measures)
Run both (as requested), even if the data are truly paired, to see how sensitive each method is to test design:

1) **Paired test**:
   - Arrange matrices so `group1=open`, `group2=closed` with the same subject order.
   - Call `compute_p_val(..., test_type="paired")`.
   - Note: sign-flip permutations assume exchangeability of within-subject differences.

2) **Two-sample test**:
   - Treat open and closed samples as independent groups (optionally subsample one session per subject if needed).
   - Call `compute_p_val(..., test_type="two-sample")`.
   - Use identical `n_permutations` and random seeds for comparability.

### B) ABIDE (multi-site, between-subject)
Primary inference is **two-sample** (ASD vs HC):
- Call `compute_p_val(..., test_type="two-sample")`.

Optional “robustness” variants (recommended if feasible):
- Per-site inference (run separately within site if sample sizes allow).
- Site-regressed matrices (edge-wise residualization) then re-run inference.

---

## Step 2 — Methods to compare (all via `compute_p_val`)
For each dataset/testing mode above, run the same set of methods:

Unconstrained:
- `method="tstat"` (baseline max-stat correction)
- `method="tfnbs"`
- `method="nbs"` with `nbs_stat="extent"`
- `method="nbs"` with `nbs_stat="intensity"`

Constrained (requires `net_labels`):
- `method="cnbs"`
- `method="ni_tfnbs"`
- `method="fbc_tfnbs"`

Save:
- `p_vals["g2>g1"]` and `p_vals["g1>g2"]` per method.
- Summary tables at edge- and block-level (see Step 4).

---

## Step 3 — Parameter sensitivity (focus on TFNBS + NBS)
Use a two-stage sweep (coarse → refined) because permutations are expensive, especially for large atlases.

### A) TFNBS-family (`tfnbs`, `ni_tfnbs`, `fbc_tfnbs`)
Grid (coarse):
- `e`: `{0.4, 0.5, 0.8, 1.0}`
- `h`: `{1.0, 2.0, 3.0}`
- `n`: `{20, 50}`
- `start_thres`: `{0.0, 1.65, 2.0}`
- FBC only: `min_cluster_size`: `{2, 3, 5}` (note dependence on atlas granularity)

Refined (after coarse):
- Narrow to 2–4 promising settings per method (based on stability + interpretability + runtime).
- Rerun with higher `n_permutations` for cleaner p-maps.

### B) NBS (`method="nbs"`)
Grid:
- `threshold`: `{1.5, 2.0, 2.5, 3.0, 3.5}`
- `nbs_stat`: `{"extent", "intensity"}`

Record sensitivity outcomes:
- number of significant edges
- component size distribution (for NBS)
- block-level concentration (for constrained methods)

---

## Step 4 — How to compare methods (no “significant edges as ML features”)
Use multiple, complementary criteria.

### A) Edge-level summaries (per tail)
For each method and parameter setting:
- `n_sig_fwer`: number of edges with `p < alpha` (FWER-corrected by construction).
- `n_sig_fdr`: number of edges after BH-FDR on the upper triangle (`q=0.05`).
- `sig_overlap`: Jaccard overlap between methods’ significance masks.
- `map_similarity`: Spearman correlation between `1-p` maps (upper triangle).

### B) Block-level summaries (using `net_labels`)
Aggregate to functional blocks (unordered network pairs):
- `sig_edges_per_block`
- `mean(-log10 p)` per block
- “top blocks” ranking per method

This makes cross-atlas comparisons easier when N changes.

---

## Step 5 — Stability via resampling (recommended)
Because there is no ground truth on real data, treat *repeatability* as a key metric.

Resampling options (pick 1–2):
- Split-half: randomly split subjects into halves multiple times; compare p-maps.
- Bootstrap: resample subjects within each group/condition; compare significance masks.
- For paired open/close: bootstrap by subject_id (resample pairs).

Quantify:
- Jaccard overlap of significant edges.
- Correlation of `1-p` maps.
- Stability of top blocks.

---

## Step 6 — Optional ML-based external validation (without using sig edges as features)
Use ML only as a *reference signal* for “what is predictive”, then test if methods highlight those edges.

Procedure:
1) Train a simple linear classifier on the full edge vector (upper triangle Fisher-z):
   - Open/Close: paired classification depends on how samples are defined; run both “paired-style” and “two-sample-style” splits if needed.
   - ABIDE: stratified and/or site-aware splits.
2) Extract edge importance:
   - absolute coefficients averaged across folds (or permutation importance).
3) Compare each method’s p-map to importance:
   - correlation between `-log10(p)` and `|coef|`
   - enrichment: fraction of top-K important edges that are significant by the method (vary K)

This yields a method comparison that is ML-informed without feeding selected edges into the predictor.

---

## Step 7 — Benchmark runtime and scaling (must-do for big atlases)
Time `compute_p_val` systematically, because this is often the real bottleneck.

What to record per run:
- wall time (seconds) for `compute_p_val`
- time per permutation (`sec / n_permutations`)
- peak memory if feasible (rough estimate is fine)
- method + parameters + `use_mp` + `n_processes`

Benchmark axes:
- Atlas size N (real atlases or subsampled nodes): e.g. `{ 200,  400}`
- `n_permutations`: e.g. `{50, 100, 500}` (or as budget allows)
- Methods: all of them, but prioritize TFNBS-family vs NBS vs CNBS

Deliverable:
- “runtime vs N” plots for each method (log scale recommended).
- Table of recommended defaults for “small atlas” vs “large atlas”.

---

## Step 8 — Reporting and artifacts
Per dataset × test mode × method × parameter setting:
- Summary CSV/Parquet: edge counts, block summaries, stability metrics, runtime.
- Figures:
  - p-map panels for a small set of representative settings
  - parameter sensitivity heatmaps (TFNBS) and threshold curves (NBS)
  - method similarity matrix (overlap/correlation)
  - runtime scaling plots

Save to:
- `examples/output/ml_testing/` (figures)
- `examples/output/ml_testing/results.csv` (tabular summary)
- `examples/output/ml_testing/configs/` (JSON/YAML configs used)

---

## Step 9 — Sanity checks
- Label permutation (quick): shuffle labels and ensure methods don’t produce stable “significant structure”.
- Tail sanity: check both `g2>g1` and `g1>g2` maps to avoid label-order mistakes.
- ABIDE site confounds (quick): run stratified vs site-aware splits for the optional ML validation step.

---

## Decisions to confirm (before implementation)
1) Open/Close dataset structure: paired per subject, or independent samples?
2) ABIDE features: which atlas, which preprocessing pipeline (motion regression, global signal)?
3) Primary metric: ROC-AUC vs balanced accuracy?
4) Compute budget: maximum `n_permutations` feasible for your largest atlas?
