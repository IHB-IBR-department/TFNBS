Usage
=====

ConnInfPy provides a Python implementation of permutation-based statistical inference for brain connectivity
networks (fMRI, EEG). It supports group comparisons (t-test) and continuous predictors with confound regression
(GLM via Freedman-Lane), together with enhancement methods including classical NBS, TFNBS, cNBS, network-informed
TFNBS, and functional-block clustering TFNBS. Null distributions are built via permutation, optionally across a
range of thresholds (TFCE), and inference can be accelerated with GPD/gamma tail approximation. Implementations
are vectorised and support multiprocessing.


Permutation-Based Inference (t-test)
=====================================

The main API for permutation-based p-values is :func:`conninfpy.pairwise_stats.compute_p_val`.

Inputs
------

- ``group1`` / ``group2``: arrays of shape ``(n_subjects, N, N)``, symmetric, with zero diagonal.
- Recommended preprocessing: Fisher r-to-z transform (``conninfpy.utils.fisher_r_to_z``) before inference.

Methods
-------

``compute_p_val(..., method=...)`` supports:

- ``"tstat"``: raw t-statistics (max-stat correction)
- ``"tfnbs"``: threshold-free NBS / TFCE-style enhancement
- ``"nbs"``: classical NBS with fixed threshold (``nbs_stat="extent"`` or ``"intensity"``)
- ``"cnbs"``: constrained NBS (requires ``net_labels``)
- ``"ni_tfnbs"``: network-informed TFNBS (requires ``net_labels``)
- ``"fbc_tfnbs"``: functional-block clustering TFNBS (requires ``net_labels``)

Minimal example (two-sample)
----------------------------

.. code-block:: python

   from conninfpy.pairwise_stats import compute_p_val
   from conninfpy.utils import fisher_r_to_z

   # group1, group2: (n_subjects, N, N), symmetric, diagonal=0
   group1_z = fisher_r_to_z(group1)
   group2_z = fisher_r_to_z(group2)

   p_vals = compute_p_val(
       group1_z,
       group2_z,
       n_permutations=1000,
       test_type="two-sample",
       method="tfnbs",
       use_mp=True,
   )

Notes
-----

- Constrained methods (``cnbs``, ``ni_tfnbs``, ``fbc_tfnbs``) require ``net_labels: ndarray[int]`` of shape ``(N,)``.
- NBS uses ``threshold`` and ``nbs_stat``.
- TFNBS-family uses ``e``, ``h``, ``n`` and ``start_thres`` (plus ``min_cluster_size`` for ``fbc_tfnbs``).


GLM Inference (continuous predictors + confounds)
==================================================

For continuous predictors (age, clinical scales) with confound regression, use
:func:`conninfpy.glm_stats.compute_p_val_glm` with Freedman-Lane permutation.

.. code-block:: python

   from conninfpy import compute_p_val_glm, fisher_r_to_z

   Y = fisher_r_to_z(connectivity_matrices)   # (n_subjects, N, N)
   p_vals = compute_p_val_glm(
       Y, interest=age, confounds=motion,
       method="tfnbs", n_permutations=1000,
   )
   # p_vals['positive'], p_vals['negative']

With GPD acceleration (Winkler 2016, ~25x fewer permutations required):

.. code-block:: python

   p_vals = compute_p_val_glm(
       Y, interest=age, confounds=motion,
       method="tfnbs", n_permutations=200, acceleration="gpd",
   )


Synthetic Method-Comparison Example
===================================

The repository includes a runnable comparison script that uses ``compute_p_val`` for all methods and saves heatmaps
(ground truth, t-stat, and ``1-p`` maps) to ``examples/output/``:

.. code-block:: bash

   python examples/sim_method_comparisons/sim_method_comparisons.py --all-scenarios --effect-size 0.25 --time-points 50 --n-permutations 50
