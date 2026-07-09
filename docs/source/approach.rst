How Inference Works
===================

This page explains the **approach** behind ConnInfPy — the reasoning that
connects "connectivity matrices and a hypothesis" to "these edges are
significant." It is conceptual rather than formal: the aim is to make clear
*why* each step is present. For the recipes (which function to call) see
:doc:`workflows`; for the equations and citations see :doc:`references`.

The package applies a single inferential scheme across designs:

.. code-block:: text

   a statistic at each edge  →  network enhancement  →
   a null distribution from relabelling  →  comparison of observed to null

The remainder of this page covers each of these four steps and the reasoning
behind it.


Rationale for permutation testing
---------------------------------

A connectome has many edges (:math:`N(N{-}1)/2`), and they are **not
independent**: neighbouring regions covary, hubs participate in many edges, and
the correlations are bounded and non-normal. A parametric per-edge p-value
assumes a known distribution and independent tests; neither holds here, and
across thousands of edges even a small per-edge error rate yields many false
positives.

Permutation testing avoids both assumptions. Rather than *assume* a null
distribution, the method **constructs** one: the data are relabelled so that
the effect is removed, the full statistic is recomputed, and the process is
repeated. The only requirement is **exchangeability** — under the null, the
relabelled data are as probable as the observed data. This is a substantially
weaker assumption than normality, and the resulting null automatically reflects
the empirical covariance among edges.


Step 1 — a statistic at each edge
---------------------------------

At every edge a single statistic quantifies the strength and reliability of the
effect. The choice of statistic follows the design:

- **Two groups** → a *t*-statistic of the difference in means (Welch's, which
  does not assume equal group variances).
- **Same subjects, two conditions** → a one-sample *t* of the within-subject
  differences.
- **A predictor with confounds** (e.g. age, controlling for motion and sex) →
  a linear model fit at each edge, reading off the *t* for the predictor of
  interest.

Direction is meaningful, so ConnInfPy always splits the statistic into two
**separate** tails — ``positive`` (e.g. group2 > group1, or predictor ↑ →
connectivity ↑) and ``negative`` — and propagates them independently. Testing a
single two-sided statistic would allow a positive and a negative block to
cancel.

.. code-block:: python

   from conninfpy import compute_t_stat, fisher_r_to_z
   t = compute_t_stat(fisher_r_to_z(g1), fisher_r_to_z(g2),
                      test_type='two-sample')
   # t['g2>g1'] is the per-edge statistic for "group2 > group1"

Step 1 thus reduces the data to one strength-map per direction.


Step 2 — network enhancement
----------------------------

A connectivity effect rarely appears as a single isolated edge; it typically
appears as a **connected set** of edges that covary. An isolated edge with a
large statistic may be noise, whereas a coherent set of moderate edges is more
convincing evidence. **Enhancement** rewrites each edge's score to reflect the
network structure it participates in; the operators differ in how they define
and summarise that structure.

A useful way to picture enhancement: take the strength-map from step 1 and
progressively raise a threshold (a "water level"). At a low threshold most
edges are supra-threshold and form large connected components; as the threshold
rises, only the strongest edges remain, in small components. The operators
differ in how they read these components.

No enhancement — ``tstat``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each edge is tested on its own, with no network aggregation. Useful as a
baseline and for strong focal effects, but insensitive to the distributed weak
effects typical of brain connectivity — which the operators below are designed
to recover.

Fixed threshold — ``nbs``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Classical Network-Based Statistics. Fix a single threshold, identify the
connected components above it, and score each edge by the size of its component
(extent) or by the summed statistic within it (intensity). It is most powerful
when the effect forms a large connected component above the chosen threshold.
The threshold is arbitrary, however, and results can vary substantially with it
(Vinokur et al. 2023 report up to 75-fold variation in detected edges across
reasonable thresholds); report results at several thresholds when using it.

Threshold-free — ``tfnbs`` (the default)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rather than fix a single threshold, TFNBS integrates the cluster evidence over
the full range of thresholds:

.. math::

   \text{score}(e) \;=\; \sum_{\text{thresholds } h}
       (\text{size of the component containing } e \text{ at } h)^{E}\;\cdot\; h^{H}

At low thresholds an edge accrues weight from belonging to a spatially extended
component; at high thresholds from its own magnitude. Summing across thresholds
rewards an edge that is extended *or* strong, removing the dependence on any
single cutoff. ``E`` controls the contribution of component size and ``H`` that
of statistic height; they act as sensitivity parameters rather than fixed
constants, and can be swept at negligible cost to report stability
(:ref:`wf-eh-stability`). This is the recommended default when there is no
strong prior about where the effect is located.

Using a known network partition — ``cnbs``, ``ni_tfnbs``, ``fbc_tfnbs``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Given an atlas partition (e.g. Yeo-7 networks, lobes, or modules), the operator
can be directed to expect effects that respect those blocks:

- ``cnbs`` (constrained NBS) — averages the statistic within each predefined
  block and performs inference at the block level (e.g. testing whether the
  within-DMN block differs as a unit). Most powerful for block-aligned effects
  such as within-network hypoconnectivity; it can miss effects that span
  multiple blocks.
- ``ni_tfnbs`` (network-informed, **soft** prior) — applies TFNBS but reweights
  an edge's component by the activation density of its own block, favouring
  clusters concentrated within a known network. It attains the lowest
  false-discovery rate in our benchmarks, at the cost of reduced power when the
  effect does not follow the partition; it degrades toward plain TFNBS rather
  than inverting when the partition is misspecified.
- ``fbc_tfnbs`` (functional-block-clustering, **hard** prior) — treats each
  block as a single atomic cluster and requires a minimum number of active
  edges per block. It targets diffuse, block-confined effects that are spread
  too thinly to form a strong connected component.

Choosing an operator
~~~~~~~~~~~~~~~~~~~~~~

No operator is universally optimal: each encodes an assumption about the
*spatial shape* of the effect, and is strongest when that assumption holds and
weakest otherwise.

.. list-table::
   :header-rows: 1
   :widths: 18 30 30 22

   * - Operator
     - Assumes the effect is…
     - Best for
     - Prior needed
   * - ``tstat``
     - focal / very strong
     - baseline reference
     - none
   * - ``nbs``
     - one large connected component
     - strong connected effects
     - a threshold
   * - ``tfnbs``
     - connected at *some* scale
     - mixed focal + extended (default)
     - none
   * - ``cnbs``
     - aligned with known blocks
     - within-network effects, high power
     - ``net_labels``
   * - ``ni_tfnbs``
     - block-concentrated (soft)
     - low false-discovery priority
     - ``net_labels``
   * - ``fbc_tfnbs``
     - diffuse within a block (hard)
     - thin, block-confined effects
     - ``net_labels``

Because the choice encodes an assumption, the recommended practice is to report
two operators: a topology-agnostic reference (``tfnbs``) and the network-aware
operator matching the prior. Agreement between them strengthens a finding;
disagreement is itself informative about the effect's spatial shape.


Step 3 — the null distribution from relabelling (permutation schemes)
---------------------------------------------------------------------

Given the enhanced score-map from the observed data, its significance is
assessed against the distribution expected under the null. That distribution is
constructed by relabelling the data so as to remove the effect while preserving
all other structure, rerunning steps 1–2, and repeating many times to obtain a
collection of null score-maps.

The principle underlying every scheme is **exchangeability**: under the null,
certain labels are interchangeable, and a valid permutation is any reshuffle
that respects that interchangeability. The correct scheme is the relabelling
that removes exactly the tested effect while leaving every nuisance structure
intact — too little randomisation leaves the effect in the null (low power);
too much scrambles structure that should be held fixed (mis-calibrated
false-positive rate). Each design has its own scheme.

Two groups — shuffle the group labels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Removes:* the association between subject and group label. *Preserves:* each
subject's connectivity matrix. Under the null the two groups are exchangeable,
so the observed assignment is one of many equally probable assignments; the
null is generated by drawing other assignments and recomputing.

Paired conditions — sign-flip each subject's difference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Removes:* the direction of each within-subject change. *Preserves:* the
pairing and each subject's magnitude. If conditions A and B are interchangeable,
the labelling of each subject's difference as "A−B" or "B−A" is equiprobable,
so each subject's difference matrix is randomly negated. Because every sign
pattern is equally likely under the null, this yields an exact, distribution-
free null.

Predictor with confounds — Freedman–Lane
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A natural but incorrect approach is to shuffle the predictor of interest
directly. When the predictor is correlated with the confounds (e.g. age with
motion), shuffling it also destroys that correlation, so the permuted data no
longer match the model under the null and the test is mis-calibrated.
Freedman–Lane permutes only the part of the data the confounds cannot account
for:

#. Fit the reduced model (confounds only). Decompose each edge's data into the
   fitted part (explained by the confounds) and the residual.
#. Permute the residuals, then add the fitted part back, unchanged.
#. Recompute the predictor's statistic on the reconstructed data.

*Removes:* any genuine predictor effect (the residuals are now in random
subject order). *Preserves:* the confound structure, held fixed. Only the
residual — the component of the data the predictor of interest could explain —
is permuted, so the resulting null isolates the tested effect while controlling
the confounds.

Repeated measures with condition-varying confounds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The within-subject difference is formed first. Any quantity constant per
subject — including additive **site** effects — cancels in the subtraction.
Freedman–Lane is then applied to the differences to partial out the component
of the confound that varies between conditions. The two adjustments are applied
in the order that makes each straightforward.

Several predictors at once
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The costly part of Freedman–Lane — the reduced-model fit and the permuted
residuals — is computed once and reused for every predictor under the shared
nuisance model, yielding a calibrated null per predictor in a single
permutation run (see :ref:`wf-multi-interest`).

Multiple sites — permute within site
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Subjects from different scanners are **not** freely exchangeable, since a site
can shift the connectivity baseline. Passing ``sites=`` confines every
permutation to occur within each site block, never across blocks. *Removes:*
the effect, within each site. *Preserves:* the between-site differences, which
therefore remain in the null rather than leaking into the effect and inflating
false positives. This is the permutation component of multi-site handling;
ComBat (below) is the complementary harmonization component.

.. code-block:: python

   from conninfpy import compute_p_val, compute_p_val_glm, fisher_r_to_z
   import numpy as np

   # two groups: group labels are permuted internally
   compute_p_val(g1, g2, test_type='two-sample', method='tfnbs', rng=42)

   # paired: each subject's difference is sign-flipped internally (exact null)
   compute_p_val(rest, task, test_type='paired', method='tfnbs', rng=42)

   # predictor + confounds: Freedman–Lane handles the relabelling
   compute_p_val_glm(fisher_r_to_z(Y), interest=age,
                     confounds=np.column_stack([sex, motion]),
                     method='tfnbs', rng=42)

   # multi-site: pass sites= and permutation stays within each site block
   compute_p_val_glm(fisher_r_to_z(Y), interest=age,
                     confounds=np.column_stack([sex, motion]),
                     sites=site, method='tfnbs', rng=42)


Step 4 — comparison of observed to null
---------------------------------------

For each relabelled null score-map, a single value is retained: the maximum
enhanced score over all edges. Across permutations these maxima form the null
distribution of the largest score attributable to noise alone.

An observed edge is significant if its score exceeds this null-maximum
distribution. Its p-value is the proportion of permutation maxima that match or
exceed it:

.. math::

   p(e) \;=\; \frac{1 + \#\{\text{permutations whose max} \ge \text{score}(e)\}}
                    {1 + (\text{number of permutations})}

Two design choices are embedded in this formula:

- **Why the maximum?** Comparing every edge to the distribution of the
  *largest* null score controls the probability of **any** false positive
  across the connectome — strong family-wise error control — without requiring
  knowledge of the inter-edge covariance, which is already reflected in the
  permutations.
- **Why the** ``+1``\ **?** The observed data is itself one valid relabelling
  and is included in the count (Phipson & Smyth 2010). This makes the p-value
  exact and strictly positive, so a finite permutation set cannot return
  :math:`p = 0`.

To control the *false-discovery rate* instead (greater sensitivity at the cost
of some false positives), ``method='bh_fdr_perm'`` replaces the max-statistic
step with a per-edge permutation p-value and a Benjamini–Hochberg procedure.


Auxiliary components: harmonization and acceleration
----------------------------------------------------

**ComBat (multi-site harmonization).** Different scanners introduce additive
and multiplicative offsets that can exceed the biological signal. ComBat
estimates and removes these per-site offsets *before* step 1, while preserving
the covariates of interest, and excludes the *tested* variable from its model
so that the harmonization never incorporates the labels the permutation will
later shuffle (which would otherwise inflate false positives). The site
strategies and their trade-offs are described in :ref:`wf-glm-sites`.

**Acceleration (GPD).** Resolving a small p-value normally requires thousands
of permutations. ``acceleration='gpd'`` fits a generalized Pareto distribution
to the extreme tail of the null-maximum distribution, so ~200 permutations
approximate ~5000 (a ~25× reduction in runtime), with a goodness-of-fit check
that falls back to the empirical count when the fit is poor. It is appropriate
for exploration; a large empirical run is recommended for the final result.


End-to-end example
------------------

A single ``analyze()`` call runs all four steps, with ComBat and within-site
permutation included:

.. code-block:: python

   from conninfpy import analyze
   import numpy as np

   out = analyze(
       Y,                                   # raw correlations (Fisher-z applied)
       interest=diagnosis,                  # step 1: a t at each edge
       confounds=np.column_stack([age, sex, mean_fd]),
       sites=site,                          # ComBat + within-site permutation
       harmonize='combat_site_dummies_glm',
       method='tfnbs',                      # step 2: threshold-free enhancement
       n_permutations=5000, acceleration=None,   # steps 3–4: empirical null
       rng=42,
   )
   out.inference['positive']                # FWER p-map, diagnosis ↑ → conn ↑

The call composes the four steps: a *t*-statistic at each edge (controlling for
age, sex, and motion) → TFNBS enhancement over connected components → a
within-site Freedman–Lane null on the harmonized data → a family-wise-corrected
p-value from the max-statistic distribution with the ``+1`` correction. See
:doc:`workflows` for the design variants and :doc:`references` for the
supporting literature.
