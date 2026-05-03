conninfpy.harmonize
===================

Multi-site harmonization for connectivity data, with design-matrix
diagnostics.

Implements parametric empirical-Bayes ComBat (Johnson, Li & Rabinovic
2007; Fortin et al. 2017/2018) directly in NumPy — no dependency on
``neuroHarmonize`` or ``neurocombat``.

ComBat model per feature :math:`j`:

.. math::

   Y_{s,i,j} = \alpha_j + X_{s,i}\,\beta_j + \gamma_{s,j}
              + \delta_{s,j}\,\varepsilon_{s,i,j},
   \qquad \varepsilon \sim \mathcal{N}(0, \sigma_j^2),

where :math:`s` indexes site, :math:`i` indexes subject within site, and
:math:`X_{s,i}` holds covariates to preserve (age, sex, diagnosis).
Estimation uses an empirical-Bayes shrinkage of per-(site, feature)
location :math:`\gamma` and scale :math:`\delta^2` toward site-level
priors, iterated to convergence.

Use this when your cohort pools subjects from multiple scanners /
sites and you want to remove site-aligned variance that is *orthogonal*
to the biological covariates you care about.

The :func:`combat_fit` / :func:`combat_apply` separation supports
cross-site machine-learning transfer (fit ComBat on training cohorts,
apply to held-out sites at test time).

The :func:`compute_vif` and :func:`design_diagnostics` helpers are
included here because confound regression (in :mod:`~conninfpy.glm_stats`)
and harmonization both lean on a well-conditioned design matrix.

References
----------
Johnson, W. E., Li, C., & Rabinovic, A. (2007). Adjusting batch effects
in microarray expression data using empirical Bayes methods.
*Biostatistics*, 8(1):118–127. doi:10.1093/biostatistics/kxj037.

Fortin, J.-P. et al. (2017). Harmonization of multi-site diffusion
tensor imaging data. *NeuroImage* 161:149–170.

Fortin, J.-P. et al. (2018). Harmonization of cortical thickness
measurements across scanners and sites. *NeuroImage* 167:104–120.

.. automodule:: conninfpy.harmonize
   :members:
   :undoc-members:
   :show-inheritance:
