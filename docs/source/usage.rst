Usage
=====

ConnInfPy provides three top-level inferential pipelines, all sharing a
common contract: edge-wise statistics → topology-aware enhancement →
permutation null → FWER / FDR-corrected p-values, with positive and
negative effects always tested separately.

Quick reference:

============================================================  ============================================================
Pipeline                                                       Use case
============================================================  ============================================================
:func:`conninfpy.compute_p_val`                                Group / paired / one-sample t-test
:func:`conninfpy.compute_p_val_glm`                            Continuous predictors with confound regression (Freedman–Lane)
:func:`conninfpy.compute_p_val_glm_multi`                      Several contrasts under a shared nuisance model in **one** permutation pass
:func:`conninfpy.compute_p_val_paired_glm`                     Paired A vs B with Δ-level confounds
:func:`conninfpy.analyze`                                      One-shot Fisher-z → ComBat → GLM/t-test → :class:`InferenceResult`
============================================================  ============================================================

Two helper layers wrap them:

- :mod:`conninfpy.harmonize` — multi-site ComBat harmonization + design diagnostics
- :mod:`conninfpy.acceleration` — GPD/gamma tail approximation for permutation acceleration

Input conventions
-----------------

- Connectivity tensors: shape ``(n_subjects, N, N)``, symmetric, zero diagonal.
- Edge weights: Fisher-z transformed correlation coefficients. Use
  :func:`conninfpy.fisher_r_to_z` before any inference call.
- The package returns :class:`~conninfpy.InferenceResult` objects, which
  behave as ``{'positive', 'negative'}`` dicts (canonical keys since
  v2.0) and additionally carry attributes ``positive`` / ``negative`` /
  ``method`` / ``n_permutations`` / ``acceleration`` / ``wall_time_s``.
  The legacy v1.x keys ``'g2>g1'`` / ``'g1>g2'`` (t-test family) remain
  readable but emit a :class:`DeprecationWarning` and will be removed in
  v2.1. F-stat omnibus tests still return ``{'omnibus': arr}``.

t-test pipeline — :func:`compute_p_val`
---------------------------------------

The simplest entry point. Computes per-edge t-statistics, applies an
enhancement operator, builds a permutation null of per-tail max
statistics, and returns FWER- or FDR-corrected p-maps with the +1
Phipson–Smyth correction.

.. code-block:: python

   from conninfpy import compute_p_val, fisher_r_to_z

   group1_z = fisher_r_to_z(group1_corr)   # (n1, N, N)
   group2_z = fisher_r_to_z(group2_corr)

   p = compute_p_val(
       group1_z, group2_z,
       test_type='two-sample',             # or 'paired', 'one-sample'
       method='tfnbs',
       n_permutations=1000,
       e=0.4, h=3.0, n=10,                 # Hao 2024 FDR-calibrated regime
       use_mp=True, random_state=42,
   )
   # p['g2>g1'] : (N, N) p-map for group2 > group1
   # p['g1>g2'] : (N, N) p-map for group1 > group2

GLM pipeline — :func:`compute_p_val_glm`
----------------------------------------

Edge-wise GLM with Freedman–Lane permutation. Use this when you have a
continuous predictor and want to control for nuisance regressors
(motion, age, sex, site).

Convenience API (recommended):

.. code-block:: python

   import numpy as np
   from conninfpy import compute_p_val_glm, fisher_r_to_z

   Y = fisher_r_to_z(connectivity_matrices)              # (n, N, N)
   age = np.array([25, 30, 28])                          # (n,)
   confounds = np.column_stack([motion, sex])            # (n, p)

   p = compute_p_val_glm(
       Y, interest=age, confounds=confounds,
       stat_type='tstat',                                 # or 'beta', 'fstat'
       method='tfnbs',
       n_permutations=1000,
       e=0.4, h=3.0, n=10,
       use_mp=True, random_state=42,
   )
   # p['positive'] : edges where age ↑ → connectivity ↑
   # p['negative'] : edges where age ↑ → connectivity ↓

Advanced API (full design matrix + contrast):

.. code-block:: python

   from conninfpy import build_design_matrix, compute_p_val_glm

   X, contrast = build_design_matrix(interest=age, confounds=motion_sex)
   p = compute_p_val_glm(
       Y, design_matrix=X, contrast=contrast,
       stat_type='tstat', method='tfnbs', n_permutations=1000,
   )

F-contrast (omnibus) for joint multi-row tests — e.g. ≥3-condition
designs or testing several predictors jointly:

.. code-block:: python

   X = np.column_stack([np.ones(n), age, age**2, sex, motion])
   contrast = np.array([
       [0, 1, 0, 0, 0],       # beta_age
       [0, 0, 1, 0, 0],       # beta_age_squared
   ])
   p = compute_p_val_glm(
       Y, design_matrix=X, contrast=contrast,
       stat_type='fstat', method='tfnbs', n_permutations=1000,
   )
   # F-pipeline returns a single non-negative tail:
   # p['omnibus'] : (N, N) FWER-corrected p-map

Multi-contrast GLM in one pass — :func:`compute_p_val_glm_multi`
----------------------------------------------------------------

If you need to test several contrasts of interest under the same
nuisance model — e.g. ``age``, ``sex``, and ``mean_fd`` separately
while treating the others as nuisance — calling
:func:`compute_p_val_glm` once per contrast wastes work: the
reduced-model residual fit and the per-permutation
``X_pinv @ Y_perm`` matrix multiplication are identical. The
multi-contrast wrapper does it in one pass.

.. code-block:: python

   from conninfpy import compute_p_val_glm_multi

   X = np.column_stack([np.ones(n), age, sex, mean_fd])  # (n, 4)
   contrasts = {
       "age":     np.array([0.0, 1.0, 0.0, 0.0]),
       "sex":     np.array([0.0, 0.0, 1.0, 0.0]),
       "motion":  np.array([0.0, 0.0, 0.0, 1.0]),
   }

   results = compute_p_val_glm_multi(
       Y, design_matrix=X, contrasts=contrasts,
       method="tfnbs", n_permutations=5000,
       acceleration="gpd", rng=42,
   )
   # results = {'age': InferenceResult, 'sex': ..., 'motion': ...}
   results["age"].positive          # (N, N) FWER p-map
   results["age"].n_significant(0.05)

For ``K`` contrasts the wall-time is roughly that of a single
:func:`compute_p_val_glm` call rather than ``K`` calls — typically a
3× speedup for the canonical age + sex + motion design.

By default the reduced model excludes any column touched by *any*
contrast in the dictionary. Pass ``nuisance_contrast=`` to override
explicitly, e.g. when you want sex and motion treated as nuisance for
the age contrast even though you also test them separately. F-stat /
multi-row contrasts are unsupported in this wrapper — call
:func:`compute_p_val_glm` once per omnibus test.

Paired A vs B with Δ-level confounds — :func:`compute_p_val_paired_glm`
------------------------------------------------------------------------

When you have a within-subject contrast (task A vs B) and confounds
that differ between conditions (e.g. condition-level motion):

.. code-block:: python

   from conninfpy import compute_p_val_paired_glm

   p = compute_p_val_paired_glm(
       Y_task_A, Y_task_B,                            # both (n, N, N)
       confounds_A=np.column_stack([motion_A, drowsiness_A]),
       confounds_B=np.column_stack([motion_B, drowsiness_B]),
       method='tfnbs', n_permutations=1000,
       e=0.4, h=3.0, n=10,
   )

With no confounds, this delegates to ``compute_p_val(test_type='paired')``
(sign-flip permutation, the exact non-asymptotic null). With Δ-level
confounds it constructs Δ_Y = Y^A − Y^B, Δ_C = C^A − C^B and runs a
one-sample GLM on the differences.

Methods (enhancement operators)
-------------------------------

``compute_p_val(..., method=...)`` and ``compute_p_val_glm(..., method=...)``
both accept:

==================  ================================================================
``method``          Operator
==================  ================================================================
``'tstat'``         No enhancement; per-tail max-statistic FWER on raw t / β / F
``'tfnbs'``         Threshold-Free NBS (Baggio 2018)
``'nbs'``           Classical Network-Based Statistic (Zalesky 2010); supply
                    ``threshold=2.0`` and ``nbs_stat='extent'`` or ``'intensity'``
``'cnbs'``          Constrained NBS (Noble & Scheinost 2020); supply ``net_labels=``
``'ni_tfnbs'``      Network-Informed TFNBS (this work); soft block-density prior;
                    supply ``net_labels=``
``'fbc_tfnbs'``     Functional-Block-Clustering TFNBS (this work); hard block
                    prior; supply ``net_labels=``, ``min_cluster_size=3``
``'bh_fdr'``        Parametric Benjamini–Hochberg FDR (no permutation)
``'bh_fdr_perm'``   Empirical edge-wise BH-FDR via permutation null
==================  ================================================================

Multi-site harmonization — :mod:`conninfpy.harmonize`
-----------------------------------------------------

Native NumPy implementation of parametric empirical-Bayes ComBat.
No ``neuroHarmonize`` or ``neurocombat`` dependency.

End-to-end with confound-aware GLM:

.. code-block:: python

   import numpy as np
   from conninfpy import (
       fisher_r_to_z,
       combat_harmonize, design_diagnostics,
       compute_p_val_glm,
   )

   # Y_corr : (n, N, N) per-subject Pearson correlation matrices
   # sites  : (n,) acquisition-site labels
   # age, sex, mean_fd : per-subject phenotype
   Y = fisher_r_to_z(Y_corr)

   # 1. Harmonize site effects, preserving age + sex + motion as biology
   preserve = np.column_stack([age, sex, mean_fd])
   Y = combat_harmonize(Y, sites=sites, preserve=preserve).Y_adjusted

   # 2. Sanity-check the design before running anything expensive
   X = np.column_stack([np.ones_like(age), age, sex, mean_fd])
   diag = design_diagnostics(X, names=["intercept", "age", "sex", "mean_fd"])
   # reports condition number + per-column VIF + plain-English flags

   # 3. TFNBS on age, partialling out sex + motion (Freedman–Lane)
   p = compute_p_val_glm(
       Y, interest=age,
       confounds=np.column_stack([sex, mean_fd]),
       stat_type='tstat', method='tfnbs',
       e=0.4, h=3.0, n=10,
       n_permutations=200, acceleration='gpd',
       use_mp=True, random_state=42,
   )

For cross-site machine-learning transfer (fit ComBat on training
cohorts, freeze, apply to held-out sites at test time), use
:func:`combat_fit` / :func:`combat_apply` separately.

Permutation acceleration — :mod:`conninfpy.acceleration`
--------------------------------------------------------

Set ``acceleration='gpd'`` (or ``'gamma'``) on either pipeline to
replace the empirical permutation p-value formula with a fitted
parametric tail (Generalized Pareto Distribution, Winkler 2016b).
Reduces the perm budget from ~5000 to ~200 with ~25× wall-clock saving.
Reproduces the empirical FWER-corrected p-values to within
``|Δ(-log10 p)| ≤ 0.001`` on >99% of edges in the ConnInfPy ABIDE Age
validation. A goodness-of-fit guard (Anderson–Darling on tail
exceedances) falls back to empirical p-values when the GPD does not
fit cleanly.

Topology-aware default integration steps
----------------------------------------

The TFCE integral is evaluated at ``n`` thresholds. Defaults:

- ``n=100`` for direct scoring (single-shot, high resolution; via
  :func:`conninfpy.get_tfnbs_score`)
- ``n=10`` inside the permutation loop (Hao 2024 reports n=10 is
  sufficient for FDR control on network data)

The exponents ``(e, h)`` accept either scalars or equal-length lists.
Lists are zipped pairwise into a single 3-D batched call, so a 16×11
(E, H) grid runs 176 combos in one ``compute_p_val`` invocation without
multiplying the runtime.

See :mod:`conninfpy.defaults` for the full list of constants and
citations (Smith & Nichols 2009, Baggio 2018, Vinokur 2023, Hao 2024).
