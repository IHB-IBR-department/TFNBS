Recommended Workflows
=====================

This page describes the **recommended end-to-end recipes** for the most
common connectivity-inference designs. Where the :doc:`usage` page is a
function-by-function reference, this page is task-oriented: pick the row in
the decision table that matches your study, then copy the recipe.

.. tip::

   For the conceptual background to these recipes — the per-edge statistic,
   network enhancement, the permutation null, and family-wise error control —
   see :doc:`approach`. Each recipe below is a particular configuration of that
   common four-step pipeline.

The recommended entry point is :func:`conninfpy.analyze`. It wraps the
standard pipeline —

.. code-block:: text

   Fisher r→z  →  optional ComBat  →  optional site-stratified permutation
              →  network enhancement  →  FWER-corrected p-value maps

— in a single call, and dispatches to the appropriate lower-level pipeline
based on the arguments supplied. The lower-level entry points
:func:`conninfpy.compute_p_val`, :func:`conninfpy.compute_p_val_glm`, and
:func:`conninfpy.compute_p_val_paired_glm` are needed only when full control
over the design matrix or contrast is required.

Selecting a workflow
--------------------

.. list-table::
   :header-rows: 1
   :widths: 40 35 25

   * - Study design
     - ``analyze()`` arguments
     - Section
   * - Two independent groups (patients vs controls)
     - ``group1=``, ``group2=``
     - :ref:`wf-two-sample`
   * - Continuous predictor + nuisance covariates
     - ``Y=``, ``interest=``, ``confounds=``
     - :ref:`wf-continuous`
   * - Several predictors, tested separately, one pass
     - ``Y=``, ``interest={name: vec}``
     - :ref:`wf-multi-interest`
   * - Multi-site continuous / group predictor
     - ``... + sites=``, ``harmonize=``
     - :ref:`wf-glm-sites`
   * - Paired / repeated conditions (no condition confounds)
     - ``group1=``, ``group2=``, ``test_type='paired'``
     - :ref:`wf-paired`
   * - Paired / repeated conditions + condition-varying confound
     - ``... + confounds_group1=``, ``confounds_group2=``
     - :ref:`wf-repeated-glm`
   * - Custom contrast / interaction / omnibus F-test
     - drop to :func:`conninfpy.compute_p_val_glm`
     - :ref:`wf-custom`

All paths share the same output contract: a dict-like
:class:`~conninfpy.InferenceResult` with canonical keys ``'positive'`` and
``'negative'`` (or ``'omnibus'`` for F-tests). Positive and negative effects
are **always tested separately**. For directional designs the orientation is
fixed:

- two-sample / paired: ``positive`` = ``group2 > group1``;
- GLM: ``positive`` = predictor ↑ → connectivity ↑.

Input conventions
-----------------

- Connectivity tensors are shape ``(n_subjects, N, N)``, symmetric, zero
  diagonal.
- :func:`analyze` applies Fisher r→z by default (``fisher_z=True``), so pass
  **raw correlation matrices**. If your matrices are already on the z-scale,
  pass ``fisher_z=False``. Lower-level functions do not apply the transform
  automatically — call :func:`conninfpy.fisher_r_to_z` first.
- Predictors, confounds, and ``sites`` must be row-aligned to the subject
  axis of the connectivity tensor.


.. _wf-two-sample:

Two independent groups
----------------------

The simplest design — two groups of subjects, no covariates.

.. code-block:: python

   from conninfpy import analyze

   out = analyze(
       group1=controls_corr,        # (n1, N, N) raw correlations
       group2=patients_corr,        # (n2, N, N)
       test_type='two-sample',
       method='tfnbs',
       e=0.4, h=3.0, n=10,          # FDR-calibrated regime (Hao 2024)
       n_permutations=1000,
       acceleration=None,           # exact empirical reference
       rng=42,
   )

   p_patients_higher = out.inference['positive']   # patients > controls
   p_controls_higher = out.inference['negative']   # controls > patients
   print(out.inference.n_significant(alpha=0.05))

.. note::

   ``compute_p_val(test_type='two-sample')`` uses **Welch's** (unequal
   variance) t-statistic unconditionally. Under unequal variances *combined
   with* unbalanced group sizes the exchangeability assumption behind the
   permutation null weakens and Type-I error can inflate (Anderson &
   Robinson 2001). Treat such results with caution; for publication-grade
   multi-site work prefer the GLM recipe below with a binary ``interest``
   indicator.


.. _wf-continuous:

Continuous predictor with confounds
-----------------------------------

Question: *does connectivity vary with age after controlling for sex and
head motion?* This is the GLM path (Freedman–Lane permutation), triggered by
passing ``Y=`` and ``interest=``.

.. code-block:: python

   import numpy as np
   from conninfpy import analyze

   confounds = np.column_stack([sex, mean_fd])

   out = analyze(
       Y,                            # (n, N, N) raw correlations
       interest=age,                 # continuous predictor
       confounds=confounds,          # nuisance regressors
       method='tfnbs',
       e=0.4, h=3.0, n=10,
       rng=42,
   )

   p_age_pos = out.inference['positive']   # older → higher connectivity
   p_age_neg = out.inference['negative']   # older → lower connectivity

A binary 0/1 ``interest`` column turns this into a confound-adjusted group
comparison — the preferred form of the two-sample test when covariates or
multiple sites are involved.

A single array ``interest`` tests one effect and returns one result. To
test **several predictors at once**, pass a dict — see
:ref:`wf-multi-interest`.


.. _wf-multi-interest:

Several predictors in one pass
------------------------------

To test several predictors **separately** under a shared nuisance model — e.g.
``age``, ``sex``, and ``mean_fd`` each as an effect of interest while the
others are controlled — pass ``interest`` as a **dict** ``{name: vector}``.
``analyze()`` builds one design matrix, shares the Freedman–Lane reduced-model
fit across predictors (:func:`conninfpy.compute_p_val_glm_multi`), and returns
a **dict** mapping each name to its own :class:`~conninfpy.InferenceResult`, at
approximately the cost of a *single* inference call rather than one per
predictor.

.. code-block:: python

   from conninfpy import analyze

   out = analyze(
       Y,
       interest={'age': age, 'sex': sex, 'mean_fd': mean_fd},
       confounds=motion,          # extra nuisance, shared by all predictors
       sites=site,                # ComBat + site-stratified permutation
       harmonize='combat_site_dummies_glm',
       method='tfnbs', e=0.4, h=3.0, n=10,
       n_permutations=5000, acceleration='gpd', rng=42,
   )

   out['age'].inference['positive']        # edges where age ↑ → connectivity ↑
   out['sex'].inference.n_significant(0.05)
   out['mean_fd'].significant_edges(atlas)  # AnalyzeResult per predictor

Notes:

- Each predictor is tested adjusting for the intercept, every ``confounds``
  column, **and the other interest predictors** (they all sit in the shared
  design). The shared ComBat diagnostics and warning flags are attached to
  every entry of the returned dict.
- Each dict **value** is a single 1-D regressor of shape ``(n_subjects,)``;
  the **key** names that predictor's result. An empty dict, or a value that
  is not 1-D, raises.
- Under the ComBat arms (``combat_only`` and ``combat_site_dummies_glm``), ComBat
  preserves only ``confounds`` and excludes **all** tested predictors — the
  same label-leak avoidance as the single-predictor case.
- The ``(E, H)`` grid sweep (:ref:`wf-eh-stability`) composes: pass sequences
  for ``e`` / ``h`` and every predictor's result carries the parameter axis.
- This path is for **separate** per-predictor tests. For a **joint**
  (omnibus) test of several predictors, use a multi-row F-contrast instead
  (:ref:`wf-custom`).


.. _wf-glm-sites:

Multi-site GLM with ComBat harmonization
----------------------------------------

The recommended pattern for most real fMRI analyses: a scientific predictor,
subject-level confounds, and multi-site data. Adding ``sites=`` engages two
coupled mechanisms — ComBat batch harmonization, and site-stratified
permutation (PALM ``-eb`` semantics, auto-set via ``strata=sites``).

.. code-block:: python

   import numpy as np
   from conninfpy import analyze

   confounds = np.column_stack([age, sex, mean_fd])

   out_primary = analyze(
       Y,
       interest=diagnosis,           # 0 = control, 1 = patient
       confounds=confounds,
       sites=site,                   # per-subject scanner / site label
       harmonize='combat_site_dummies_glm',  # primary paper recipe
       method='tfnbs',
       e=0.4, h=3.0, n=10,
       n_permutations=200,
       acceleration='gpd',           # fast exploratory inference
       rng=42,
   )

   print(out_primary.inference)
   print(out_primary.flags)                # plain-English provenance warnings
   print(out_primary.combat_diagnostics)   # includes 'strategy': 'combat_site_dummies_glm'

**What the call does, step by step:**

- ComBat fits with ``preserve = confounds`` (age, sex, motion). The tested
  variable — diagnosis — is *deliberately excluded* so the harmonization fit
  does not incorporate the labels the permutation will reshuffle (Nygaard 2016
  label-leak avoidance).
- The downstream GLM tests diagnosis with ``age + sex + mean_fd +
  site_dummies`` as nuisance. Site dummies absorb any additive shifts ComBat
  did not fully remove.
- ``sites=site`` auto-sets ``strata=site``, so the permutation respects
  site exchangeability blocks.

Choosing a harmonization strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``analyze()`` ships the three multi-site strategies used in the paper:
ComBat-only, ComBat plus site dummies in the GLM, and site dummies in the GLM
without ComBat. For paper-grade work, report the primary
``combat_site_dummies_glm`` result with ``combat_only`` and/or ``site_dummies_glm`` as
sensitivity arms, depending on the question.

.. list-table::
   :header-rows: 1
   :widths: 18 14 24 22 22

   * - ``harmonize=``
     - Strategy
     - What ComBat does
     - GLM nuisance design
     - When to use
   * - ``'combat_only'`` / ``'combat'``
     - ComBat only
     - Fits with ``preserve = confounds``; tested variable excluded
     - ``confounds``
     - Isolates what the ComBat transform contributes, without residual
       site fixed effects in the inferential GLM.
   * - ``'combat_site_dummies_glm'`` / ``'nuisance_only'`` / ``'d'``
     - ComBat + site GLM
     - Fits with ``preserve = confounds``; tested variable excluded
     - ``confounds + site dummies``
     - Primary recipe. Removes the Nygaard 2016 label leak and keeps residual
       site control in the inferential GLM. Requires ``sites=`` and
       ``confounds=``; GLM mode only.
   * - ``'site_dummies_glm'`` / ``None`` / ``'e'``
     - Site GLM
     - Skipped
     - ``confounds + site dummies``
     - No-ComBat sensitivity arm; useful when the inference result is the only
       deliverable or ComBat assumptions are questionable.
   * - ``'auto'`` (default)
     - dispatcher
     - ``combat_site_dummies_glm`` if ``sites + confounds``; ``site_dummies_glm`` if only
       ``sites``; none otherwise
     - (whichever selected strategy uses)
     - When the call shape unambiguously implies the recipe. Prefer
       explicit strategy names in paper scripts.

.. code-block:: python

   # Primary + sensitivity arms.
   common = dict(Y=Y, interest=diagnosis, confounds=confounds, sites=site,
                 method='tfnbs', e=0.4, h=3.0, n=10, rng=42)
   out_combat = analyze(**common, harmonize='combat_only')
   out_primary = analyze(**common, harmonize='combat_site_dummies_glm')
   out_site = analyze(**common, harmonize='site_dummies_glm')

.. note::

   **Two-sample mode + ``sites=``** has no defensible ComBat recipe (there is
   no interest column to preserve). ``analyze()`` skips ComBat and emits a
   flag recommending promotion of the analysis to a GLM with a binary
   ``interest`` indicator, which is the appropriate form for any multi-site
   group comparison.


.. _wf-paired:

Paired / repeated conditions
----------------------------

Question: *does connectivity change between condition A and condition B in
the same subjects?* Pass both conditions as ``group1`` / ``group2`` (row
aligned: ``group1[s]`` and ``group2[s]`` are the same subject) with
``test_type='paired'``. With no condition-varying confounds this uses the
**sign-flip** permutation null — the exact non-asymptotic test.

.. code-block:: python

   from conninfpy import analyze

   out = analyze(
       group1=rest_corr,             # condition A, (n, N, N)
       group2=task_corr,             # condition B, same subjects
       test_type='paired',
       method='tfnbs',
       e=0.4, h=3.0, n=10,
       rng=42,
   )

   p_task_higher = out.inference['positive']   # task > rest
   p_rest_higher = out.inference['negative']   # rest > task


.. _wf-repeated-glm:

Repeated-measures GLM (condition-varying confounds)
---------------------------------------------------

When a confound *differs between the two conditions for the same subject*
(e.g. condition-level head motion, arousal, or reaction time), pass it
through ``confounds_group1`` / ``confounds_group2``. ``analyze()`` then
routes to the **paired-difference GLM**: it forms
:math:`\Delta_Y = \mathrm{group2} - \mathrm{group1}` and the per-subject
confound difference, and tests the difference intercept with Freedman–Lane
permutation (via :func:`conninfpy.compute_p_val_paired_glm`).

.. code-block:: python

   from conninfpy import analyze

   out = analyze(
       group1=rest_corr,             # condition A, (n, N, N)
       group2=task_corr,             # condition B, same subjects
       test_type='paired',
       confounds_group1=fd_rest,     # motion during condition A
       confounds_group2=fd_task,     # motion during condition B
       method='tfnbs',
       e=0.4, h=3.0, n=10,
       n_permutations=1000,
       acceleration=None,
       rng=42,
   )

   p_task_higher = out.inference['positive']   # task > rest, motion-adjusted
   p_rest_higher = out.inference['negative']

Notes:

- **Orientation is identical** to the no-confound paired path:
  ``positive = group2 > group1``. (Internally the conditions are passed
  swapped so the tested intercept of ``Δ = group2 − group1`` keeps that
  sign.)
- Pass **both** ``confounds_group1`` and ``confounds_group2`` or neither.
  They are only valid with ``test_type='paired'``. Use ``confounds=`` (no
  suffix) only for the between-subject GLM path; passing it alongside
  ``group1``/``group2`` raises.
- **Subject-constant nuisances cancel** in the within-subject difference and
  do not need to be supplied — including additive **site** effects. So
  ``sites=`` with a paired design skips ComBat (it is unnecessary) while
  still stratifying the permutation; ``analyze()`` notes this in
  ``out.flags``.
- **Power caveat.** The paired GLM tests the difference *intercept*. When a
  single edge carries a very strong effect it dominates the max-statistic
  FWER null, and the GLM path is then **less powerful** than the no-confound
  sign-flip path. This is inherent to the intercept permutation test, not to
  the ``analyze()`` implementation. When no condition-varying confounds are
  present, the plain paired path (:ref:`wf-paired`) is preferable.


.. _wf-custom:

Custom contrasts, interactions, and omnibus F-tests
---------------------------------------------------

For interactions, custom categorical coding, or joint (omnibus) tests, build
the design matrix and contrast explicitly and call
:func:`conninfpy.compute_p_val_glm` directly — see the
:doc:`usage` page for the advanced API, including ``stat_type='fstat'`` for
multi-row contrasts (≥3-condition designs or jointly testing several
predictors), which returns a single ``'omnibus'`` p-map.


.. _wf-eh-stability:

Inference stability: sweep ``(E, H)`` in one call
-------------------------------------------------

The TFNBS-family enhancement (``'tfnbs'``, ``'ni_tfnbs'``, ``'fbc_tfnbs'``)
depends on two exponents — the extent exponent ``E`` and the height exponent
``H``. Published defaults disagree (Hao 2024 ``E=0.4, H=3.0``; Smith–Nichols
2009 ``E=0.5, H=2.0``; Baggio 2018 ``E=0.75, H=3.0``), and Vinokur 2023
reports up to 75-fold variation in edge counts across the ``(E, H)`` plane.
A single ``(E, H)`` result is therefore of limited value on its own; a finding
should be shown to be **stable across the plausible parameter range**.

``analyze()`` evaluates the grid at negligible additional cost. Pass
**equal-length sequences** for ``e`` and ``h`` and the whole grid is computed
in **one permutation pass**: the threshold-integration loop runs once and the
per-cell exponentiation is broadcast at the end, so a K-cell grid costs
approximately the wall-clock of a single cell. Passing sequences thus converts
a point estimate into a stability assessment at little extra cost.

.. code-block:: python

   from conninfpy import analyze

   # Three published-default (E, H) cells, zipped pairwise (not a cross
   # product): (0.4, 3.0), (0.5, 2.0), (0.75, 3.0).
   e_grid = [0.4, 0.5, 0.75]      # Hao, Smith–Nichols, Baggio
   h_grid = [3.0, 2.0, 3.0]

   out = analyze(
       Y, interest=diagnosis, confounds=confounds, sites=site,
       harmonize='combat_site_dummies_glm',
       method='tfnbs', e=e_grid, h=h_grid, n=10,
       n_permutations=5000, acceleration=None, rng=42,
   )

   r = out.inference
   r.is_grid                 # True
   r.positive.shape          # (N, N, 3) — one p-map per (E, H) cell
   r.e_grid                  # array([0.4, 0.5, 0.75])
   r.h_grid                  # array([3.0, 2.0, 3.0])
   r.n_significant(0.05)     # per-cell counts:
                             # {'positive': [k0, k1, k2], 'negative': [...]}

When the returned object is a grid (``is_grid == True``), the ``(N, N)``
exporters require a cell to project to. Pass ``param_idx=`` (or call
``.select()`` first); omitting it on a grid raises an explicit error rather
than selecting a cell implicitly:

.. code-block:: python

   sub = r.select(0)                                  # fresh 2D result for cell 0
   df  = r.significant_edges(atlas, param_idx=2)      # export cell 2 directly
   r.to_csv('edges_hao.csv', atlas=atlas, param_idx=0)

Use it to confirm a result is stable across the published-default cells
before reporting, or to run a denser sensitivity grid (e.g. a 6 × 6 sweep) at
the cost of a single inference call. The same ``e`` / ``h`` sequence syntax
works on every TFNBS-family path — two-sample, GLM, and the repeated-measures
GLM alike.


Interpreting and exporting results
----------------------------------

Every result is dict-like and carries metadata and exporters:

.. code-block:: python

   r = out.inference
   r.method, r.n_permutations, r.acceleration, r.harmonized
   r.n_significant(alpha=0.05)            # {'positive': k, 'negative': k}
   r.stat_signed                          # signed t / β effect map

   # ROI-aware edge table (needs an atlas)
   from conninfpy import AtlasInfo
   atlas = AtlasInfo.schaefer_200_yeo7()
   df = out.significant_edges(atlas, sort='network_pair', top_k=50)
   out.to_csv('edges.csv', atlas=atlas)

   # One-call publication figure
   from conninfpy.plot import summary_figure
   fig = summary_figure(out.inference, atlas=atlas, alpha=0.05, top_k=10)
   fig.savefig('summary.pdf', bbox_inches='tight')

Speed vs. publication runs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``analyze()`` defaults to ``n_permutations=200`` with ``acceleration='gpd'``
for fast exploration. For a final result use a larger empirical reference:

.. code-block:: python

   # Exploration (≈25× faster, GPD tail approximation with empirical fallback)
   analyze(Y, interest=age, confounds=confounds, n_permutations=200,
           acceleration='gpd', rng=42)

   # Publication (exact finite-permutation reference)
   analyze(Y, interest=age, confounds=confounds, n_permutations=5000,
           acceleration=None, rng=42)

See :doc:`usage` for the full enhancement-method table and acceleration
internals. The ``(E, H)`` stability sweep (:ref:`wf-eh-stability`) and the
several-predictors workflow (:ref:`wf-multi-interest`, which wraps
:func:`conninfpy.compute_p_val_glm_multi`) are covered above.
