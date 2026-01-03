Usage
=====

This toolbox presents a Python implementation to use the TFNBS for fMRI and EEG functional connectivity data with the 
flexibility to use structured data (i.e. block diagonal matrices) corresponding to frequency range and connectivity matrix. 
While other network-oriented toolkits are composed of higher computational complexity, TFNBS is based on the construction 
of a null distribution which is permuted across many thresholds to identify statistical significance therefore optimizing 
performance. Our implementation of TFNBS follows efficient algorithms and allows usage of parallel computing cores, therefore 
massively reducing required computing resources over conventional approaches.


Permutation-Based Inference
===========================

The main API for permutation-based p-values is :func:`tfnbs.pairwise_stats.compute_p_val`.

Inputs
------

- ``group1`` / ``group2``: arrays of shape ``(n_subjects, N, N)``, symmetric, with zero diagonal.
- Recommended preprocessing: Fisher r-to-z transform (``tfnbs.utils.fisher_r_to_z``) before inference.

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

   from tfnbs.pairwise_stats import compute_p_val
   from tfnbs.utils import fisher_r_to_z

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


Synthetic Method-Comparison Example
===================================

The repository includes a runnable comparison script that uses ``compute_p_val`` for all methods and saves heatmaps
(ground truth, t-stat, and ``1-p`` maps) to ``examples/output/``:

.. code-block:: bash

   python examples/sim_method_comparisons.py --all-scenarios --effect-size 0.25 --time-points 50 --n-permutations 50
