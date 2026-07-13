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
from .glm_stats import (
    compute_p_val_glm,
    compute_p_val_glm_multi,
    compute_p_val_paired_glm,
)
from .harmonize import combat_harmonize
from .pairwise_stats import StatMethod, compute_p_val
from .utils import fisher_r_to_z


COMBAT_ONLY = "combat_only"
COMBAT_SITE_DUMMIES_GLM = "combat_site_dummies_glm"
SITE_DUMMIES_GLM = "site_dummies_glm"


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


def _build_multi_design(
    interest: Dict[str, Any],
    confounds: Optional[npt.NDArray[np.float64]],
) -> "tuple[np.ndarray, Dict[str, np.ndarray]]":
    """Assemble ``X = [intercept, confounds, *interest_columns]`` and one
    unit contrast per named predictor for the multi-contrast GLM.

    The intercept and every confound column are nuisance; each interest
    column gets a ``[0, ..., 0, 1, 0, ..., 0]`` contrast targeting it. The
    returned ``contrasts`` dict is keyed by the same names as ``interest``,
    ready for :func:`~conninfpy.compute_p_val_glm_multi`.
    """
    names = list(interest.keys())
    n = len(np.asarray(interest[names[0]]))
    blocks = [np.ones((n, 1), dtype=np.float64)]
    if confounds is not None:
        blocks.append(_as_2d(confounds).astype(np.float64))
    n_nuisance_cols = sum(b.shape[1] for b in blocks)

    interest_cols = [np.asarray(interest[nm], dtype=np.float64)[:, None]
                     for nm in names]
    X = np.hstack(blocks + interest_cols)
    p = X.shape[1]

    contrasts: Dict[str, np.ndarray] = {}
    for j, nm in enumerate(names):
        c = np.zeros(p, dtype=np.float64)
        c[n_nuisance_cols + j] = 1.0
        contrasts[nm] = c
    return X, contrasts


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

    Returns one of:

    * ``'combat_only'`` -- fit ComBat, then test on the harmonized matrix
      without adding site dummies to the GLM.
    * ``'combat_site_dummies_glm'`` -- fit ComBat, then add site dummies to the
      inferential GLM.
    * ``'site_dummies_glm'`` -- skip ComBat and model site directly in the
      inferential GLM.
    * ``None`` -- no ComBat, no site adjustment (no ``sites`` passed,
      or two-sample call where ComBat has no defensible recipe).

    The old ``'d'`` and ``'e'`` values are accepted as compatibility aliases
    for the historical table labels used during early development.
    """
    if harmonize in (
        COMBAT_SITE_DUMMIES_GLM,
        "combat_site_glm",
        "combat_plus_site_dummies_glm",
        "combat_plus_site_glm",
        "interest_orthogonal_combat",
        "interest_orthogonal",
        "nuisance_only",
        "d",
    ):
        return COMBAT_SITE_DUMMIES_GLM
    if harmonize in (
        COMBAT_ONLY,
        "combat",
        "combat_no_site_glm",
        "combat_no_site_dummies_glm",
    ):
        return COMBAT_ONLY
    if harmonize in (
        SITE_DUMMIES_GLM,
        "site_glm",
        "single_stage_site_glm",
        "single_stage_site_dummies_glm",
        "no_combat",
        "e",
        None,
    ):
        return SITE_DUMMIES_GLM if (glm_mode and sites_provided) else None
    if harmonize == "auto":
        if not sites_provided:
            return None
        if glm_mode:
            if confounds_provided:
                return COMBAT_SITE_DUMMIES_GLM
            return SITE_DUMMIES_GLM
        # two-sample + sites has no defensible ComBat recipe without
        # an interest column to preserve. Skip harmonization; the
        # caller is flagged in analyze() so the demotion isn't silent.
        return None
    raise ValueError(
        "harmonize= must be one of 'auto', 'combat_only', "
        "'combat_site_dummies_glm', 'site_dummies_glm', or None "
        f"(legacy aliases are accepted); got {harmonize!r}."
    )


def analyze(
    Y: Optional[npt.NDArray[np.float64]] = None,
    *,
    interest: Optional[
        Union[npt.NDArray[np.float64], Dict[str, npt.NDArray[np.float64]]]
    ] = None,
    confounds: Optional[npt.NDArray[np.float64]] = None,
    group1: Optional[npt.NDArray[np.float64]] = None,
    group2: Optional[npt.NDArray[np.float64]] = None,
    confounds_group1: Optional[npt.NDArray[np.float64]] = None,
    confounds_group2: Optional[npt.NDArray[np.float64]] = None,
    test_type: str = "two-sample",
    sites: Optional[Sequence[Any]] = None,
    preserve: Optional[npt.NDArray[np.float64]] = None,
    harmonize: Optional[str] = "auto",
    fisher_z: bool = True,
    method: Union[str, StatMethod] = "tfnbs",
    acceleration: Optional[str] = "gpd",
    n_permutations: int = 200,
    e: Union[float, Sequence[float]] = 0.3,
    h: Union[float, Sequence[float]] = 3.0,
    n: int = 10,
    rng: RngLike = None,
    verbose: bool = False,
    use_mp: bool = True,
    **method_kwargs: Any,
) -> Union[AnalyzeResult, Dict[str, AnalyzeResult]]:
    """Run the standard prepare → harmonize → GLM/two-sample → infer recipe.

    Parameters
    ----------
    Y : ndarray of shape (n_subjects, N, N), optional
        Per-subject connectivity matrices for the GLM path. Pass this
        with ``interest``/``confounds``. Leave as ``None`` (or simply
        omit) when using the two-sample path via ``group1``/``group2``.
    interest : ndarray or dict of ndarray, optional
        GLM regressor(s) of interest. Triggers the GLM pipeline.

        * **Single predictor** — a 1D array of shape ``(n_subjects,)``
          (or ``(n_subjects, 1)``), e.g. ``age`` or a 0/1 group dummy.
          ``analyze()`` returns one :class:`AnalyzeResult`.
        * **Several predictors** — a dict ``{name: vector}`` (e.g.
          ``{'age': age, 'sex': sex}``). Each predictor is tested under a
          shared nuisance model (intercept + ``confounds`` + the other
          predictors) in one Freedman–Lane permutation pass, via
          :func:`~conninfpy.compute_p_val_glm_multi`. ``analyze()`` then
          returns a ``dict`` mapping each name to its own
          :class:`AnalyzeResult`.

        A bare multi-column array or a Python list of regressors is
        rejected (it cannot carry predictor names and would silently test
        only the last column) — pass a dict instead. For a *joint*
        omnibus test of several predictors use a multi-row F-contrast via
        :func:`~conninfpy.compute_p_val_glm` (``stat_type='fstat'``).
    confounds : ndarray, optional
        Nuisance regressors for the GLM pipeline.
    group1, group2 : ndarray, optional
        Two-sample inputs of shape ``(n_i, N, N)``. Triggers the t-test
        pipeline. Mutually exclusive with ``interest``. For
        ``test_type='paired'`` they are the two within-subject
        conditions and must be row-aligned (``group1[s]`` and
        ``group2[s]`` are the same subject).
    confounds_group1, confounds_group2 : ndarray, optional
        Condition-varying confounds for the **repeated-measures GLM**
        path, shape ``(n_subjects,)`` or ``(n_subjects, k)``, aligned to
        ``group1`` / ``group2`` respectively. Only valid with
        ``test_type='paired'``; pass both or neither. When given, the
        paired difference ``group1 - group2`` is tested while the
        per-subject confound difference is partialled out
        (Freedman–Lane on the differences, via
        :func:`~conninfpy.compute_p_val_paired_glm`). Use these for a
        confound whose value differs between conditions for the same
        subject (e.g. condition-level motion, arousal, reaction time).
        Subject-constant nuisances (and additive site effects) cancel in
        the difference and do not need to be supplied. With no
        condition-varying confounds the paired path stays on the
        sign-flip t-test (exact non-asymptotic null). Either way the
        result keeps the ``positive = group2 > group1`` orientation.
    test_type : str, default ``'two-sample'``
        Test type for the t-test pipeline (also accepts ``'paired'`` and
        ``'one-sample'``). ``'paired'`` combined with
        ``confounds_group1`` / ``confounds_group2`` selects the
        repeated-measures GLM.
    sites : sequence, optional
        Per-subject site labels. If provided, the permutation engine is
        auto-stratified on ``sites`` (within-block exchangeability —
        equivalent to PALM's ``-eb`` option). ComBat runs when
        ``harmonize`` resolves to ``'combat_only'`` or
        ``'combat_site_dummies_glm'``. In GLM mode with sites but no confounds,
        ``'auto'`` resolves to ``'site_dummies_glm'`` (site dummies in the GLM,
        no ComBat). In two-sample mode with sites, ComBat is skipped and
        a flag asks the caller to promote the analysis to GLM with a
        binary ``interest``.
    preserve : ndarray, optional
        Covariates whose effect should be preserved through ComBat.
        Under ``'combat_only'`` and ``'combat_site_dummies_glm'`` this is set
        automatically to ``confounds``; passing ``preserve`` explicitly is
        overridden and emits a flag because the tested variable must remain
        outside the ComBat design. ``'site_dummies_glm'`` does not run ComBat, so
        ``preserve`` is ignored.
    harmonize : str or ``None``, default ``'auto'``
        Selects the ComBat / site-handling strategy. Three named multi-site
        strategies are supported, matching the paper comparison:

        * ``'combat_only'`` (aliases ``'combat'`` and
          ``'combat_no_site_glm'``) — fit ComBat with
          ``preserve = confounds`` only, then test on the harmonized matrix
          without adding site dummies to the downstream GLM. Use this to
          isolate what the ComBat transform itself contributes.
        * ``'combat_site_dummies_glm'`` (aliases ``'combat_site_glm'``,
          ``'combat_plus_site_glm'``, ``'nuisance_only'``, and legacy alias
          ``'d'``) —
          primary site-aware recipe. Fit ComBat with ``preserve = confounds``
          only, deliberately excluding the tested variable, then append site
          dummies to the downstream GLM nuisance design. Use this when the
          analysis needs both a harmonized matrix and residual site control
          during inference.
        * ``'site_dummies_glm'`` (aliases ``'site_glm'``,
          ``'single_stage_site_glm'``, ``'no_combat'`` and ``None``; legacy
          alias ``'e'``) — skip
          ComBat entirely and model site as fixed-effect nuisance dummies in
          the same GLM that tests the variable of interest. Use this when the
          only deliverable is the inference result, when ComBat assumptions
          are questionable, or as the no-ComBat site-aware sensitivity arm.

        ``'auto'`` (default) dispatches based on the call shape:

        - GLM + ``sites`` + ``confounds`` → ``'combat_site_dummies_glm'``.
        - GLM + ``sites``, no ``confounds`` → ``'site_dummies_glm'``.
        - GLM + no ``sites`` → no harmonization.
        - Two-sample + ``sites`` → skip ComBat with a flag asking
          the caller to promote to GLM with binary ``interest`` for
          the ``'combat_site_dummies_glm'`` recipe.
    fisher_z : bool, default ``True``
        Apply Fisher r→z to ``Y`` (or ``group1``/``group2``) first.
    method : str, default ``'tfnbs'``
        Enhancement method.
    acceleration : str, default ``'gpd'``
        ``'gpd'`` / ``'gamma'`` / ``None``. GPD and gamma are
        tail-approximation accelerators with empirical fallback; use
        ``None`` and a larger permutation budget when an exact empirical
        finite-permutation reference is required.
    n_permutations : int, default ``200``
        Permutations. The default is tuned for exploratory
        GPD-accelerated inference; final empirical runs typically use a
        larger value with ``acceleration=None``.
    e, h : float or sequence of float, default ``0.3``, ``3.0``
        TFNBS exponents. Pass equal-length sequences to evaluate a
        whole ``(E, H)`` grid in one call — the threshold loop runs
        once and the per-cell scores are broadcast at the end, so
        a K-cell grid costs ~the same wall-clock as a single cell.
        When arrays are passed the returned :class:`InferenceResult`
        carries the parameter axis (``result.is_grid == True``,
        ``result.e_grid``, ``result.h_grid``); use
        ``result.select(param_idx)`` or pass ``param_idx=`` to
        ``significant_edges`` / ``to_csv`` to project to a single
        cell. Validation-paper default ``(0.3, 3.0)``.
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
    AnalyzeResult or dict of AnalyzeResult
        For a single predictor / two-sample / paired call, one
        :class:`AnalyzeResult` carrying ``.inference`` (the
        :class:`~conninfpy.InferenceResult`) plus optional ComBat /
        design diagnostics and any plain-English warning flags. For a
        dict ``interest`` (several predictors), a ``dict`` mapping each
        predictor name to its own :class:`AnalyzeResult`; the shared
        ComBat diagnostics and flags are attached to every entry.
    """
    flags: list = []
    combat_diag: Optional[Dict[str, Any]] = None

    glm_mode = interest is not None
    multi_mode = glm_mode and isinstance(interest, dict)
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
    if multi_mode:
        # Several predictors → compute_p_val_glm_multi. Each value must be a
        # single 1D regressor; the dict key names the per-predictor result.
        if len(interest) == 0:
            raise ValueError(
                "interest={} is empty. Pass a non-empty dict {name: vector}, "
                "or a single array for one predictor."
            )
        for nm, vec in interest.items():
            v = np.asarray(vec)
            if v.ndim != 1:
                raise ValueError(
                    f"interest[{nm!r}] must be a 1D predictor of shape "
                    f"(n_subjects,); got shape {v.shape}. Each dict entry is "
                    "one regressor of interest."
                )
    elif glm_mode:
        # `interest` is a SINGLE predictor. A (n, k>1) array or a Python list
        # of regressors would be tested by build_design_matrix's contrast on
        # the *last* column only (silently demoting the rest to nuisance), and
        # a list like [age, sex] is read as shape (k, n) and crashes
        # downstream. Reject both with a pointer to the dict (multi) API.
        _interest_arr = np.asarray(interest)
        if _interest_arr.ndim >= 2 and not (
            _interest_arr.ndim == 2 and _interest_arr.shape[1] == 1
        ):
            raise ValueError(
                "interest= must be a single predictor of shape (n_subjects,) "
                f"or (n_subjects, 1); got array of shape {_interest_arr.shape}. "
                "To test several predictors, pass a dict "
                "interest={'name': vector, ...} (analyze() returns one result "
                "per predictor via compute_p_val_glm_multi), or build a "
                "multi-row contrast with compute_p_val_glm(..., "
                "stat_type='fstat') for a joint omnibus F-test."
            )

    # Repeated-measures GLM: paired conditions + condition-varying confounds.
    has_cg1 = confounds_group1 is not None
    has_cg2 = confounds_group2 is not None
    paired_glm_mode = has_cg1 or has_cg2
    if paired_glm_mode:
        if has_cg1 != has_cg2:
            raise ValueError(
                "Pass both confounds_group1 and confounds_group2, or neither — "
                "the repeated-measures GLM needs the confound in both conditions "
                "to form the per-subject difference."
            )
        if not ttest_mode:
            raise ValueError(
                "confounds_group1/confounds_group2 require the paired t-test "
                "inputs (group1=, group2=). For a between-subject continuous "
                "predictor with confounds use Y=, interest=, confounds=."
            )
        if test_type != "paired":
            raise ValueError(
                "confounds_group1/confounds_group2 are only valid with "
                f"test_type='paired'; got test_type={test_type!r}. Condition-"
                "varying confounds are differenced within subject, which only "
                "makes sense for a repeated-measures design."
            )
    if ttest_mode and confounds is not None:
        raise ValueError(
            "confounds= is the GLM-path nuisance design (used with interest=). "
            "For a paired design with condition-varying confounds use "
            "confounds_group1=/confounds_group2=; for a two-sample contrast with "
            "nuisance covariates promote to GLM with a binary interest= column."
        )

    strategy = _resolve_strategy(
        harmonize,
        glm_mode=glm_mode,
        ttest_mode=ttest_mode,
        sites_provided=sites is not None,
        confounds_provided=confounds is not None,
    )

    # Explicit ComBat guards (only fire for explicit requests; 'auto'
    # resolves to combat_site_dummies_glm only when these guards are satisfied).
    if harmonize in (
        COMBAT_ONLY,
        "combat",
        "combat_no_site_glm",
        "combat_no_site_dummies_glm",
        COMBAT_SITE_DUMMIES_GLM,
        "combat_site_glm",
        "combat_plus_site_glm",
        "combat_plus_site_dummies_glm",
        "interest_orthogonal_combat",
        "interest_orthogonal",
        "nuisance_only",
        "d",
    ):
        if sites is None:
            raise ValueError(
                f"{strategy} requires sites=...; otherwise "
                "there is no ComBat step to control."
            )
        if confounds is None:
            raise ValueError(
                f"{strategy} requires confounds=; otherwise "
                "the ComBat preserve design would be empty."
            )
        if not glm_mode:
            raise ValueError(
                f"{strategy} is GLM-only (interest=, confounds=). For "
                "two-sample / paired designs use "
                "harmonize='auto' or None."
            )
    # Two-sample + sites resolves to no-ComBat (no defensible preserve).
    # Flag so the demotion is visible and the caller can promote to GLM.
    if (
        ttest_mode
        and test_type != "paired"
        and sites is not None
        and harmonize in ("auto", None)
    ):
        flags.append(
            "two-sample + sites: ComBat skipped (no interest column to "
            "preserve). For combat_site_dummies_glm, promote to GLM "
            "with a binary interest indicator."
        )
    # Paired + sites: additive site effects are subject-constant across the
    # two conditions and cancel in the within-subject difference, so ComBat
    # is unnecessary. sites= still auto-stratifies the permutation below.
    if (
        ttest_mode
        and test_type == "paired"
        and sites is not None
        and harmonize in ("auto", None)
    ):
        flags.append(
            "paired + sites: ComBat skipped — additive site effects cancel in "
            "the within-subject difference; sites= still stratifies the "
            "permutation."
        )
    if strategy is None and preserve is not None:
        flags.append(
            "preserve= ignored because no ComBat harmonization strategy is "
            "active."
        )
        preserve = None

    # ---- Fisher r→z ----
    if fisher_z:
        if glm_mode:
            Y = fisher_r_to_z(Y)
        else:
            if group1 is not None:
                group1 = fisher_r_to_z(group1)
            if group2 is not None:
                group2 = fisher_r_to_z(group2)

    # ---- ComBat strategies: preserve = confounds only ----
    # The tested variable is deliberately omitted from ComBat so the
    # harmonization fit never sees the labels that the permutation will
    # reshuffle (Nygaard 2016 label-leak avoidance). Mild attenuation
    # when corr(site, interest) > 0; that's the accepted trade-off.
    if strategy in (COMBAT_ONLY, COMBAT_SITE_DUMMIES_GLM):
        if preserve is not None:
            flags.append(
                f"preserve= overridden by {strategy}: "
                "ComBat preserve is set to confounds only."
            )
        preserve = _as_2d(confounds)
        flags.append(
            f"{strategy}: preserve excludes interest."
        )

    # ---- ComBat (runs for combat_only and combat_site_dummies_glm) ----
    if strategy in (COMBAT_ONLY, COMBAT_SITE_DUMMIES_GLM) and sites is not None:
        assert Y is not None
        res = combat_harmonize(Y, sites=np.asarray(sites), preserve=preserve)
        Y = res.Y_adjusted
        combat_diag = dict(res.diagnostics)
        combat_diag["strategy"] = strategy
        if strategy == COMBAT_SITE_DUMMIES_GLM:
            combat_diag["legacy_strategy"] = "D"
        combat_diag["preserve_columns"] = "confounds_only"
        ratio = combat_diag.get("between_site_variance_ratio_after_over_before")
        if ratio is not None and ratio > 0.5:
            flags.append(
                f"ComBat removed less than half of between-site variance "
                f"(ratio after/before = {ratio:.3f})."
            )

    # ---- Site dummies → GLM nuisance for the site-GLM strategies ----
    # combat_site_dummies_glm: ComBat harmonized the matrix but did not see interest;
    # site dummies in the GLM still help in case any site-correlated noise
    # survives. site_dummies_glm: no ComBat ran at all; site dummies are the only site
    # adjustment.
    if (
        glm_mode
        and sites is not None
        and strategy in (COMBAT_SITE_DUMMIES_GLM, SITE_DUMMIES_GLM)
    ):
        site_dum = _site_dummies(np.asarray(sites))
        if site_dum.shape[1] > 0:
            if confounds is None:
                confounds = site_dum
            else:
                confounds = np.column_stack([_as_2d(confounds), site_dum])
        if strategy == COMBAT_SITE_DUMMIES_GLM:
            flags.append(
                "combat_site_dummies_glm: site dummies appended to GLM nuisance."
            )
        if strategy == SITE_DUMMIES_GLM:
            if preserve is not None:
                flags.append(
                    "preserve= ignored by site_dummies_glm because "
                    "ComBat does not run."
                )
                preserve = None
            flags.append(
                "site_dummies_glm: no ComBat; site dummies appended to GLM nuisance."
            )
            combat_diag = {
                "strategy": SITE_DUMMIES_GLM,
                "legacy_strategy": "E",
                "preserve_columns": None,
            }

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
            if (
                ttest_mode
                and test_type == "paired"
                and group1 is not None
                and auto_strata.shape[0] != group1.shape[0]
            ):
                raise ValueError(
                    "For paired inference, sites/strata must provide one label per "
                    f"paired subject ({group1.shape[0]}), got {auto_strata.shape[0]}."
                )
            flags.append(
                "strata= auto-set to `sites`; pass strata= explicitly "
                "(or strata=None) to override."
            )

    if multi_mode:
        # Several predictors under a shared nuisance model in one pass.
        # Build X = [intercept, confounds(+site dummies), *interest cols]
        # and a unit contrast per predictor. ComBat strategies already ran
        # above with preserve=confounds, excluding *all* tested columns.
        assert isinstance(interest, dict)
        X, contrasts = _build_multi_design(interest, confounds)
        results = compute_p_val_glm_multi(
            Y, design_matrix=X, contrasts=contrasts,
            method=method, n_permutations=n_permutations,
            acceleration=acceleration,
            e=e, h=h, n=n, rng=rng, verbose=verbose, use_mp=use_mp,
            strata=auto_strata,
            **method_kwargs,
        )
        out: Dict[str, AnalyzeResult] = {}
        for nm, res in results.items():
            res.harmonized = (
                strategy in (COMBAT_ONLY, COMBAT_SITE_DUMMIES_GLM)
                and sites is not None
            )
            res.preserve_provided = preserve is not None
            res.combat_diagnostics = combat_diag
            out[nm] = AnalyzeResult(
                inference=res,
                combat_diagnostics=combat_diag,
                flags=list(flags),
            )
        return out

    if glm_mode:
        result = compute_p_val_glm(
            Y, interest=interest, confounds=confounds,
            method=method, n_permutations=n_permutations,
            acceleration=acceleration,
            e=e, h=h, n=n, rng=rng, verbose=verbose, use_mp=use_mp,
            strata=auto_strata,
            **method_kwargs,
        )
    elif paired_glm_mode:
        # Repeated-measures GLM. Pass the conditions swapped (A=group2,
        # B=group1) so the tested intercept of Δ = A − B = group2 − group1
        # keeps analyze()'s `positive = group2 > group1` orientation, matching
        # the no-confound paired path (compute_p_val(group1, group2, paired)).
        result = compute_p_val_paired_glm(
            group2, group1,
            confounds_A=confounds_group2, confounds_B=confounds_group1,
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
    # exporters) can report what preprocessing applied. With site_dummies_glm, ComBat
    # is skipped even when sites= is passed; harmonized reflects whether the
    # matrix was actually adjusted.
    result.harmonized = (
        strategy in (COMBAT_ONLY, COMBAT_SITE_DUMMIES_GLM) and sites is not None
    )
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
