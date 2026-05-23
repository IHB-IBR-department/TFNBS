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

    def significant_edges(self, *args: Any, **kwargs: Any) -> "Any":
        """Delegate to the underlying inference result's
        :meth:`~conninfpy.InferenceResult.significant_edges`.
        """
        return self.inference.significant_edges(*args, **kwargs)

    def to_csv(self, path: "Any", **kwargs: Any) -> None:
        """Delegate to the underlying inference result's
        :meth:`~conninfpy.InferenceResult.to_csv`.
        """
        return self.inference.to_csv(path, **kwargs)

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


def _as_2d(a: Any) -> np.ndarray:
    """Promote a 1D regressor to a column matrix, leave 2D untouched."""
    arr = np.asarray(a)
    if arr.ndim == 1:
        return arr[:, np.newaxis]
    return arr


def _site_dummies(
    site_id: Any, drop_first: bool = True
) -> npt.NDArray[np.float64]:
    """One-hot encode site labels with optional drop-first reference coding.

    Returns ``(n, K - drop_first)`` float64 dummy matrix. The first
    unique site (sorted) is the reference category when
    ``drop_first=True`` (the standard one). Returns an ``(n, 0)`` array
    when there is only one site — the GLM design is unchanged.
    """
    sites = np.asarray(site_id)
    unique, inverse = np.unique(sites, return_inverse=True)
    n_sites = len(unique)
    if n_sites < 2:
        return np.zeros((len(sites), 0), dtype=np.float64)
    start = 1 if drop_first else 0
    out = np.zeros((len(sites), n_sites - start), dtype=np.float64)
    for j in range(start, n_sites):
        out[inverse == j, j - start] = 1.0
    return out


def _resolve_strategy(
    harmonize: Any,
    *,
    glm_mode: bool,
    ttest_mode: bool,
    sites_provided: bool,
    confounds_provided: bool,
) -> Optional[str]:
    """Resolve the user-facing ``harmonize=`` to a canonical strategy.

    Returns one of ``{'d', 'e', None}``:

    * ``'d'`` — preserve confounds only through ComBat, site in GLM.
      Primary recipe.
    * ``'e'`` — skip ComBat, site in GLM. Sensitivity arm.
    * ``None`` — no ComBat, no site adjustment (no ``sites`` passed,
      or two-sample call where ComBat has no defensible recipe).
    """
    if harmonize in ("nuisance_only", "d"):
        return "d"
    if harmonize in ("e", None):
        return "e" if (glm_mode and sites_provided) else None
    if harmonize == "auto":
        if not sites_provided:
            return None
        if glm_mode:
            return "d" if confounds_provided else "e"
        # two-sample + sites has no defensible ComBat recipe without
        # an interest column to preserve. Skip harmonization; the
        # caller is flagged in analyze() so the demotion isn't silent.
        return None
    valid = {"auto", "nuisance_only", "d", "e", None}
    raise ValueError(
        f"harmonize= must be one of {valid}, got {harmonize!r}."
    )


def analyze(
    Y: Optional[npt.NDArray[np.float64]] = None,
    *,
    interest: Optional[npt.NDArray[np.float64]] = None,
    confounds: Optional[npt.NDArray[np.float64]] = None,
    group1: Optional[npt.NDArray[np.float64]] = None,
    group2: Optional[npt.NDArray[np.float64]] = None,
    test_type: str = "two-sample",
    sites: Optional[Sequence[Any]] = None,
    preserve: Optional[npt.NDArray[np.float64]] = None,
    harmonize: Optional[str] = "auto",
    fisher_z: bool = True,
    method: Union[str, StatMethod] = "tfnbs",
    acceleration: Optional[str] = "gpd",
    n_permutations: int = 200,
    e: Union[float, Sequence[float]] = 0.4,
    h: Union[float, Sequence[float]] = 3.0,
    n: int = 10,
    rng: RngLike = None,
    verbose: bool = False,
    use_mp: bool = True,
    **method_kwargs: Any,
) -> AnalyzeResult:
    """Run the standard prepare → harmonize → GLM/two-sample → infer recipe.

    Parameters
    ----------
    Y : ndarray of shape (n_subjects, N, N), optional
        Per-subject connectivity matrices for the GLM path. Pass this
        with ``interest``/``confounds``. Leave as ``None`` (or simply
        omit) when using the two-sample path via ``group1``/``group2``.
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
        Under Strategy D this is set automatically to ``confounds``;
        passing ``preserve`` explicitly overrides it and emits a flag.
        Strategy E does not run ComBat, so ``preserve`` is ignored.
    harmonize : str or ``None``, default ``'auto'``
        Selects the ComBat / site-handling strategy. See
        [[paper_combat_resolution_strategies]] for derivations and
        the calibration evidence behind the choice of D and E. Two
        strategies are supported:

        * ``'nuisance_only'`` / ``'d'`` (Strategy D, primary
          recipe) — ComBat fits with ``preserve = confounds`` only;
          the tested variable is deliberately omitted from the
          preserve design. Site dummies are appended to the
          downstream GLM nuisance design. Removes the Nygaard 2016
          label leak; mild signal attenuation when site and
          interest are correlated. Requires ``sites=`` and
          ``confounds=``; GLM mode only.
        * ``None`` / ``'e'`` (Strategy E, sensitivity arm) — skip
          ComBat entirely. Site dummies are appended to the GLM
          nuisance design. Calibrated by construction; gives up
          ComBat's multiplicative site shrinkage. Pair with
          Strategy D and report both.

        ``'auto'`` (default) dispatches based on the call shape:

        - GLM + ``sites`` + ``confounds`` → ``'d'`` (primary).
        - GLM + ``sites``, no ``confounds`` → ``'e'``.
        - GLM + no ``sites`` → no harmonization.
        - Two-sample + ``sites`` → skip ComBat with a flag asking
          the caller to promote to GLM with binary ``interest`` for
          the Strategy D recipe.
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
    e, h : float or sequence of float, default ``0.4``, ``3.0``
        TFNBS exponents. Pass equal-length sequences to evaluate a
        whole ``(E, H)`` grid in one call — the threshold loop runs
        once and the per-cell scores are broadcast at the end, so
        a K-cell grid costs ~the same wall-clock as a single cell.
        When arrays are passed the returned :class:`InferenceResult`
        carries the parameter axis (``result.is_grid == True``,
        ``result.e_grid``, ``result.h_grid``); use
        ``result.select(param_idx)`` or pass ``param_idx=`` to
        ``significant_edges`` / ``to_csv`` to project to a single
        cell. Hao 2024 default ``(0.4, 3.0)``.
    n : int, default ``10``
        Threshold integration steps.
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
    if glm_mode and Y is None:
        raise ValueError(
            "analyze() requires Y when using the GLM path (interest=, "
            "[confounds=]). Pass Y as the first positional argument."
        )

    strategy = _resolve_strategy(
        harmonize,
        glm_mode=glm_mode,
        ttest_mode=ttest_mode,
        sites_provided=sites is not None,
        confounds_provided=confounds is not None,
    )

    # Explicit Strategy-D guards (only fire for explicit 'nuisance_only' / 'd';
    # 'auto' resolves to 'd' only when these guards are already satisfied).
    if harmonize in ("nuisance_only", "d"):
        if sites is None:
            raise ValueError(
                "Strategy D requires sites=... — otherwise there is no "
                "ComBat step to control."
            )
        if confounds is None:
            raise ValueError(
                "Strategy D requires confounds= — otherwise the ComBat "
                "preserve design would be empty."
            )
        if not glm_mode:
            raise ValueError(
                "Strategy D is GLM-only (interest=, confounds=). For "
                "two-sample / paired designs use harmonize='auto' or None."
            )
    # Two-sample + sites resolves to no-ComBat (no defensible preserve).
    # Flag so the demotion is visible and the caller can promote to GLM.
    if ttest_mode and sites is not None and harmonize in ("auto", None):
        flags.append(
            "two-sample + sites: ComBat skipped (no interest column to "
            "preserve). For Strategy D, promote to GLM with a binary "
            "interest indicator."
        )

    # ---- Fisher r→z ----
    if fisher_z:
        if glm_mode:
            Y = fisher_r_to_z(Y)
        else:
            if group1 is not None:
                group1 = fisher_r_to_z(group1)
            if group2 is not None:
                group2 = fisher_r_to_z(group2)

    # ---- Strategy D: ComBat preserve = confounds only ----
    # The tested variable is deliberately omitted from ComBat so the
    # harmonization fit never sees the labels that the permutation will
    # reshuffle (Nygaard 2016 label-leak avoidance). Mild attenuation
    # when corr(site, interest) > 0; that's the accepted trade-off.
    if strategy == "d":
        if preserve is not None:
            flags.append(
                "preserve= overridden by Strategy D: ComBat preserve "
                "is set to confounds only."
            )
        preserve = _as_2d(confounds)
        flags.append(
            "preserve excludes interest (Strategy D); "
            "site dummies appended to GLM nuisance."
        )

    # ---- ComBat (only runs for Strategy D) ----
    if strategy == "d" and sites is not None:
        assert Y is not None
        res = combat_harmonize(Y, sites=np.asarray(sites), preserve=preserve)
        Y = res.Y_adjusted
        combat_diag = dict(res.diagnostics)
        combat_diag["strategy"] = "D"
        combat_diag["preserve_columns"] = "confounds_only"
        ratio = combat_diag.get("between_site_variance_ratio_after_over_before")
        if ratio is not None and ratio > 0.5:
            flags.append(
                f"ComBat removed less than half of between-site variance "
                f"(ratio after/before = {ratio:.3f})."
            )

    # ---- Site dummies → GLM nuisance for Strategy D and Strategy E ----
    # Strategy D: ComBat harmonized the matrix but did not see interest;
    # site dummies in the GLM still help in case any site-correlated noise
    # survives. Strategy E: no ComBat ran at all; site dummies are the
    # only site adjustment.
    if (
        glm_mode
        and sites is not None
        and strategy in ("d", "e")
    ):
        site_dum = _site_dummies(np.asarray(sites))
        if site_dum.shape[1] > 0:
            if confounds is None:
                confounds = site_dum
            else:
                confounds = np.column_stack([_as_2d(confounds), site_dum])
        if strategy == "e":
            flags.append(
                "no ComBat; site dummies appended to GLM nuisance "
                "(Strategy E)."
            )
            combat_diag = {"strategy": "E", "preserve_columns": None}

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
    # exporters) can report what preprocessing applied. With harmonize=None
    # (Strategy E) ComBat is skipped even when sites= is passed; harmonized
    # reflects whether the matrix was actually adjusted.
    result.harmonized = strategy in ("b", "d") and sites is not None
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
