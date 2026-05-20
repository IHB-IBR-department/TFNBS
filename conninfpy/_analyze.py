"""High-level `analyze()` convenience entry-point.

Wraps the most common neuroimaging recipe in one function call:

    Fisher r→z (optional) → ComBat (optional) → design diagnostics →
    GLM with Freedman-Lane permutation → :class:`InferenceResult`.

Use the lower-level :func:`~conninfpy.compute_p_val` /
:func:`~conninfpy.compute_p_val_glm` entry-points when you need
fine-grained control. Use :func:`analyze` when the standard recipe
fits and you want one line of code.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt

from ._result import InferenceResult, OmnibusInferenceResult
from ._rng import RngLike
from .glm_stats import compute_p_val_glm
from .harmonize import combat_harmonize
from .pairwise_stats import StatMethod, compute_p_val
from .utils import fisher_r_to_z


@dataclass
class AnalyzeResult:
    """Bundle of inference + harmonization diagnostics from :func:`analyze`."""

    inference: Union[InferenceResult, OmnibusInferenceResult]
    combat_diagnostics: Optional[Dict[str, Any]] = None
    design_diagnostics: Optional[Dict[str, Any]] = None
    flags: list = field(default_factory=list)

    def __getitem__(self, key: str) -> Any:
        # Forward dict-style access to the underlying result so
        # `analyze(...)['positive']` or ['omnibus'] works without users
        # unwrapping.
        return self.inference[key]

    @property
    def positive(self) -> npt.NDArray[np.float64]:
        if isinstance(self.inference, OmnibusInferenceResult):
            raise AttributeError(
                "AnalyzeResult.positive is undefined for the F-stat omnibus "
                "path; use AnalyzeResult.omnibus instead."
            )
        return self.inference.positive

    @property
    def negative(self) -> npt.NDArray[np.float64]:
        if isinstance(self.inference, OmnibusInferenceResult):
            raise AttributeError(
                "AnalyzeResult.negative is undefined for the F-stat omnibus "
                "path; use AnalyzeResult.omnibus instead."
            )
        return self.inference.negative

    @property
    def omnibus(self) -> npt.NDArray[np.float64]:
        if not isinstance(self.inference, OmnibusInferenceResult):
            raise AttributeError(
                "AnalyzeResult.omnibus is only defined for the F-stat path; "
                "use .positive / .negative for t / β contrasts."
            )
        return self.inference.omnibus

    def __repr__(self) -> str:  # pragma: no cover (cosmetic)
        parts = [
            f"AnalyzeResult(inference={self.inference!r})",
        ]
        if self.combat_diagnostics is not None:
            ratio = self.combat_diagnostics.get(
                "between_site_variance_ratio_after_over_before"
            )
            if ratio is not None:
                parts.append(f"  combat: between-site var ratio={ratio:.3f}")
        if self.flags:
            parts.append("  flags: " + "; ".join(self.flags))
        return "\n".join(parts)


def analyze(
    Y: npt.NDArray[np.float64],
    *,
    interest: Optional[npt.NDArray[np.float64]] = None,
    confounds: Optional[npt.NDArray[np.float64]] = None,
    group1: Optional[npt.NDArray[np.float64]] = None,
    group2: Optional[npt.NDArray[np.float64]] = None,
    test_type: str = "two-sample",
    sites: Optional[Sequence[Any]] = None,
    preserve: Optional[npt.NDArray[np.float64]] = None,
    fisher_z: bool = True,
    method: Union[str, StatMethod] = "tfnbs",
    acceleration: Optional[str] = "gpd",
    n_permutations: int = 200,
    e: float = 0.4,
    h: float = 3.0,
    n: int = 10,
    rng: RngLike = None,
    verbose: bool = False,
    use_mp: bool = True,
    **method_kwargs: Any,
) -> AnalyzeResult:
    """Run the standard prepare → harmonize → GLM/two-sample → infer recipe.

    Parameters
    ----------
    Y : ndarray of shape (n_subjects, N, N)
        Per-subject connectivity matrices. Pass either this (with
        ``interest``/``confounds`` for GLM) **or** ``group1`` + ``group2``
        for the two-sample t-test pipeline.
    interest : ndarray, optional
        GLM regressor of interest (e.g. ``age`` or a 0/1 group dummy).
        Triggers the GLM pipeline.
    confounds : ndarray, optional
        Nuisance regressors for the GLM pipeline.
    group1, group2 : ndarray, optional
        Two-sample inputs of shape ``(n_i, N, N)``. Triggers the t-test
        pipeline. Mutually exclusive with ``interest``.
    test_type : str, default ``'two-sample'``
        Test type for the t-test pipeline (also accepts ``'paired'`` and
        ``'one-sample'``).
    sites : sequence, optional
        Per-subject site labels. If provided, ComBat harmonisation runs
        before inference AND the permutation engine is auto-stratified on
        ``sites`` (within-block exchangeability — equivalent to PALM's
        ``-eb`` option). Either ``Y`` or both ``group1``/``group2`` must
        share the site vector indexing — for the two-sample pipeline,
        pass concatenated ``[group1, group2]`` as ``Y`` if you want
        ComBat first. The auto-stratification prevents the shadow-of-H₀
        leak that occurs when ComBat is fit on observed labels but
        downstream permutation reshuffles across sites freely.
    preserve : ndarray, optional
        Covariates whose effect should be preserved through ComBat.
    fisher_z : bool, default ``True``
        Apply Fisher r→z to ``Y`` (or ``group1``/``group2``) first.
    method : str, default ``'tfnbs'``
        Enhancement method.
    acceleration : str, default ``'gpd'``
        ``'gpd'`` / ``'gamma'`` / ``None``. Defaults to GPD because at
        ``n_permutations=200`` GPD reproduces empirical 5{,}000-perm
        FWER p-values to within ``|Δ(-log10 p)| ≤ 0.001`` on >99% of
        edges (Winkler 2016).
    n_permutations : int, default ``200``
        Permutations. Default tuned to GPD-accelerated FWER.
    e, h, n : float, float, int
        TFNBS exponents and threshold integration steps. Defaults follow
        Hao 2024 ``(0.4, 3.0, 10)``.
    rng : int, ``numpy.random.Generator``, or ``None``
        Reproducibility handle.
    verbose : bool, default ``False``
        Show a progress bar during permutation.
    use_mp : bool, default ``True``
        Use multiprocessing for the permutation loop.
    **method_kwargs
        Forwarded to the underlying pipeline (e.g. ``net_labels``,
        ``min_cluster_size``, ``threshold``).

    Returns
    -------
    AnalyzeResult
        Wrapper carrying ``.inference`` (the
        :class:`~conninfpy.InferenceResult`) plus optional ComBat /
        design diagnostics and any plain-English warning flags.
    """
    flags: list = []
    combat_diag: Optional[Dict[str, Any]] = None

    glm_mode = interest is not None
    ttest_mode = group1 is not None or group2 is not None
    if glm_mode and ttest_mode:
        raise ValueError(
            "analyze() takes either (interest, [confounds]) for the GLM path or "
            "(group1, group2) for the t-test path, not both."
        )
    if not glm_mode and not ttest_mode:
        raise ValueError(
            "analyze() requires either an `interest` regressor or `group1`/`group2`."
        )

    # ---- Fisher r→z ----
    if fisher_z:
        if glm_mode:
            Y = fisher_r_to_z(Y)
        else:
            group1 = fisher_r_to_z(group1)
            group2 = fisher_r_to_z(group2)

    # ---- ComBat ----
    if sites is not None:
        if glm_mode:
            assert Y is not None
            res = combat_harmonize(Y, sites=np.asarray(sites), preserve=preserve)
            Y = res.Y_adjusted
            combat_diag = res.diagnostics
        else:
            assert group1 is not None and group2 is not None
            n1 = group1.shape[0]
            stacked = np.concatenate([group1, group2], axis=0)
            res = combat_harmonize(
                stacked, sites=np.asarray(sites), preserve=preserve
            )
            stacked = res.Y_adjusted
            group1, group2 = stacked[:n1], stacked[n1:]
            combat_diag = res.diagnostics
        ratio = combat_diag.get("between_site_variance_ratio_after_over_before")
        if ratio is not None and ratio > 0.5:
            flags.append(
                f"ComBat removed less than half of between-site variance "
                f"(ratio after/before = {ratio:.3f})."
            )

    # ---- Inference ----
    # Auto-stratify the permutation engine on sites when present (PALM -eb
    # semantics). This is the second half of the harmonization-stratification
    # pair: ComBat fits site means then permutation respects site structure.
    # An explicit strata= passed via **method_kwargs wins (e.g. for
    # study-specific blocking that doesn't match the site label).
    if "strata" in method_kwargs:
        user_strata = method_kwargs.pop("strata")
        auto_strata = (
            np.asarray(user_strata) if user_strata is not None else None
        )
    else:
        auto_strata = np.asarray(sites) if sites is not None else None
        if auto_strata is not None:
            flags.append(
                "strata= auto-set to `sites`; pass strata= explicitly "
                "(or strata=None) to override."
            )

    if glm_mode:
        result = compute_p_val_glm(
            Y, interest=interest, confounds=confounds,
            method=method, n_permutations=n_permutations,
            acceleration=acceleration,
            e=e, h=h, n=n, rng=rng, verbose=verbose, use_mp=use_mp,
            strata=auto_strata,
            **method_kwargs,
        )
    else:
        result = compute_p_val(
            group1, group2, test_type=test_type,
            method=method, n_permutations=n_permutations,
            acceleration=acceleration,
            e=e, h=h, n=n, rng=rng, verbose=verbose, use_mp=use_mp,
            strata=auto_strata,
            **method_kwargs,
        )

    # ---- Provenance threading ----
    # The lower-level compute_p_val* entry-points don't know that
    # analyze() ran ComBat upstream; thread that information onto the
    # result so downstream consumers (significant_edges, summary_figure,
    # exporters) can report what preprocessing applied.
    result.harmonized = sites is not None
    result.preserve_provided = preserve is not None
    result.combat_diagnostics = combat_diag
    # strata_provided was set inside compute_p_val{,_glm} from the strata=
    # argument; preserved here for clarity.

    return AnalyzeResult(
        inference=result,
        combat_diagnostics=combat_diag,
        flags=flags,
    )


__all__ = ["analyze", "AnalyzeResult"]
