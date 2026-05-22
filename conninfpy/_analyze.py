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


def _column_names(a: Any) -> Optional[list]:
    """Return column names for a labeled 2D container, else None.

    Recognized: pandas DataFrame, numpy structured array. Plain ndarrays
    (1D or 2D) return None — the coupling check needs labels to compare.
    """
    if hasattr(a, "columns"):  # pandas DataFrame-like
        return list(a.columns)
    arr = np.asarray(a)
    if arr.dtype.names is not None:
        return list(arr.dtype.names)
    return None


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


def _design_coupling_leak(
    preserve: Any,
    interest: Optional[Any],
    confounds: Optional[Any],
) -> Optional[list]:
    """Return names of preserve-columns not present in the GLM design.

    Returns ``None`` when we cannot compare (any input is unlabeled);
    returns an empty list when all preserve columns are accounted for;
    returns a list of leaking column names otherwise.
    """
    pres_names = _column_names(preserve)
    if pres_names is None:
        return None  # raw ndarray: skip silently
    design_names: list = []
    for source in (interest, confounds):
        if source is None:
            continue
        names = _column_names(source)
        if names is None:
            return None  # mixed labeled/unlabeled — can't compare
        design_names.extend(names)
    leak = [c for c in pres_names if c not in design_names]
    return leak or None


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
        Covariates whose effect should be preserved through ComBat. If
        ``sites`` is provided but ``preserve`` is left ``None``,
        ``analyze`` auto-builds ``preserve`` from the design:
        ``np.column_stack([interest, confounds])`` in GLM mode, or a
        0/1 group indicator ``[0…0, 1…1]`` in two-sample mode. An
        explicit ``preserve=`` always wins. The auto-built default
        prevents a common pitfall in which biological variance is
        partially absorbed into the site adjustments. When ``preserve``
        and the GLM design are passed as labeled DataFrames (or numpy
        structured arrays), ``analyze`` additionally checks for columns
        present in ``preserve`` but absent from ``(interest, confounds)``
        and flags them — the test for that variance survives
        harmonization but isn't partialled out at inference.
    harmonize : str or ``None``, default ``'auto'``
        Selects the ComBat / site-handling strategy. See
        [[paper_combat_resolution_strategies]] for the methodological
        background; the short summary is:

        * ``'auto'`` (default, Strategy B) — historical wiring: ComBat
          fits with ``preserve = interest + confounds`` (the
          tested variable is preserved) and site is *not* added to the
          downstream GLM. Convenient but anti-conservative under
          ABIDE-style imbalance; treat the resulting p-values as
          exploratory.
        * ``'nuisance_only'`` (Strategy D) — ComBat fits with
          ``preserve = confounds`` only (the tested variable is
          deliberately omitted), then the downstream GLM tests interest
          with site dummies appended to the nuisance design. Removes
          the two-step label leak; mild signal attenuation when site
          and interest are correlated. Requires ``sites=`` and
          ``confounds=``; GLM mode only.
        * ``None`` (Strategy E) — skip ComBat entirely. Site dummies
          are appended to the downstream GLM nuisance design. The
          conservative calibrated reference. GLM mode only when
          ``sites is not None``; in two-sample mode this just means
          "no harmonization."
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

    valid_harmonize = {"auto", "nuisance_only", None}
    if harmonize not in valid_harmonize:
        raise ValueError(
            f"harmonize= must be one of {valid_harmonize}, got {harmonize!r}."
        )

    glm_mode = interest is not None
    ttest_mode = group1 is not None or group2 is not None
    test_type_str = test_type.value if hasattr(test_type, "value") else str(test_type)
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

    # Strategy-D / Strategy-E guards
    if harmonize == "nuisance_only":
        if sites is None:
            raise ValueError(
                "harmonize='nuisance_only' (Strategy D) requires sites=... "
                "— otherwise there is no ComBat step to control."
            )
        if confounds is None:
            raise ValueError(
                "harmonize='nuisance_only' (Strategy D) requires confounds= "
                "— otherwise the ComBat preserve design would be empty."
            )
        if not glm_mode:
            raise ValueError(
                "harmonize='nuisance_only' (Strategy D) is only supported in "
                "the GLM path (interest=, confounds=). For two-sample / paired "
                "designs use harmonize='auto' or harmonize=None."
            )
    if harmonize is None and ttest_mode and sites is not None:
        # In two-sample mode harmonize=None just means "skip ComBat". Site
        # dummies cannot be appended to a t-test design, so emit a clarifying
        # flag rather than silently doing nothing.
        flags.append(
            "harmonize=None in two-sample mode: ComBat skipped; site dummies "
            "have no place in the t-test design and were not added."
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

    # ---- Auto-preserve (PR-4 of implementation_plan_2026-05-19) ----
    # When ComBat runs, any variance the user wants preserved through
    # harmonization MUST be passed via preserve=; otherwise its
    # variance gets partially absorbed into the site adjustments.
    #
    # The default `harmonize='auto'` (Strategy B) auto-builds preserve
    # from the full design — interest + confounds in GLM mode, or a
    # group indicator in t-test mode.
    #
    # `harmonize='nuisance_only'` (Strategy D) overrides preserve to
    # confounds only — the tested variable is *deliberately omitted*
    # from ComBat so the harmonization fit never sees the labels that
    # will be permuted. This avoids the Nygaard 2016 two-step inflation
    # at the cost of mild attenuation under corr(site, interest) > 0.
    if harmonize == "nuisance_only":
        if preserve is not None:
            flags.append(
                "preserve= overridden by harmonize='nuisance_only': "
                "ComBat preserve is set to confounds only (Strategy D)."
            )
        preserve = _as_2d(confounds)
        flags.append(
            "preserve excludes interest (Strategy D); "
            "site dummies appended to GLM nuisance."
        )
    elif harmonize == "auto" and sites is not None and preserve is None:
        if glm_mode:
            cols = [c for c in (interest, confounds) if c is not None]
            if cols:
                preserve = np.column_stack(
                    [_as_2d(c) for c in cols]
                )
                flags.append(
                    "preserve auto-built from (interest, confounds); "
                    "pass preserve= explicitly to override."
                )
        else:
            assert group1 is not None
            if test_type_str != "one-sample":
                assert group2 is not None
                n1 = group1.shape[0]
                n2 = group2.shape[0]
                preserve = np.concatenate(
                    [np.zeros(n1), np.ones(n2)]
                )[:, np.newaxis]
                flags.append(
                    "preserve auto-built from the (g1 vs g2) group indicator; "
                    "pass preserve= explicitly to override."
                )

    # ---- Design coupling check (PR-4 / Tier 1.3) ----
    # If preserve and the GLM design are passed as labeled DataFrames /
    # structured arrays, warn when a column in preserve doesn't have a
    # counterpart in the GLM design — its variance survived
    # harmonization unadjusted at inference, a silent confound leak.
    # For raw ndarrays we can't compare column identity; silent skip.
    if (
        harmonize == "auto"
        and sites is not None
        and glm_mode
        and preserve is not None
    ):
        leak = _design_coupling_leak(preserve, interest, confounds)
        if leak:
            flags.append(
                "Column(s) {!r} in preserve= are not represented in "
                "(interest, confounds); their variance survives "
                "harmonization but is unadjusted in the contrast."
                .format(leak)
            )

    # ---- ComBat (skipped entirely when harmonize=None / Strategy E) ----
    if harmonize is not None and sites is not None:
        if glm_mode:
            assert Y is not None
            res = combat_harmonize(Y, sites=np.asarray(sites), preserve=preserve)
            Y = res.Y_adjusted
            combat_diag = dict(res.diagnostics)  # copy so we can annotate
        else:
            assert group1 is not None
            if test_type_str == "one-sample":
                res = combat_harmonize(
                    group1, sites=np.asarray(sites), preserve=preserve
                )
                group1 = res.Y_adjusted
                combat_diag = dict(res.diagnostics)
            else:
                assert group2 is not None
                n1 = group1.shape[0]
                stacked = np.concatenate([group1, group2], axis=0)
                res = combat_harmonize(
                    stacked, sites=np.asarray(sites), preserve=preserve
                )
                stacked = res.Y_adjusted
                group1, group2 = stacked[:n1], stacked[n1:]
                combat_diag = dict(res.diagnostics)
        # Annotate which strategy this ComBat fit corresponds to.
        if harmonize == "nuisance_only":
            combat_diag["strategy"] = "D"
            combat_diag["preserve_columns"] = "confounds_only"
        else:
            combat_diag["strategy"] = "B"
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
        and harmonize in ("nuisance_only", None)
    ):
        site_dum = _site_dummies(np.asarray(sites))
        if site_dum.shape[1] > 0:
            if confounds is None:
                confounds = site_dum
            else:
                confounds = np.column_stack([_as_2d(confounds), site_dum])
        if harmonize is None:
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
    result.harmonized = harmonize is not None and sites is not None
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
