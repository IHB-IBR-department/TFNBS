import os
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from conninfpy.atlas import AtlasInfo
from conninfpy.loaders import (
    AbideSchaeferLoader,
    OpenCloseLoader,
    MultiSiteOpenCloseLoader,
    StressTimeseriesLoader,
    ZerssenNiftiLoader,
    NumpyLoader,
    CSVDirectoryLoader,
    NiftiDirectoryLoader,
    FmriprepDerivativesLoader,
    TimeseriesDirectoryLoader,
    ConnectivityMatrixLoader,
    ConditionTimeseriesArrayLoader,
    CustomPreparedZerssenLoader,
    ManifestLoader,
    ChinaCloseCloseLoader
)

APP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = APP_DIR.parent

LOADER_CLASSES = {
    "AbideSchaeferLoader": AbideSchaeferLoader,
    "OpenCloseLoader": OpenCloseLoader,
    "MultiSiteOpenCloseLoader": MultiSiteOpenCloseLoader,
    "StressTimeseriesLoader": StressTimeseriesLoader,
    "ZerssenNiftiLoader": ZerssenNiftiLoader,
    "NumpyLoader": NumpyLoader,
    "CSVDirectoryLoader": CSVDirectoryLoader,
    "NiftiDirectoryLoader": NiftiDirectoryLoader,
    "FmriprepDerivativesLoader": FmriprepDerivativesLoader,
    "TimeseriesDirectoryLoader": TimeseriesDirectoryLoader,
    "ConnectivityMatrixLoader": ConnectivityMatrixLoader,
    "ConditionTimeseriesArrayLoader": ConditionTimeseriesArrayLoader,
    "CustomPreparedZerssenLoader": CustomPreparedZerssenLoader,
    "ManifestLoader": ManifestLoader,
    "ChinaCloseCloseLoader": ChinaCloseCloseLoader
}

def local_dataset_templates_enabled():
    """Whether local-only custom/beta built-in templates should be exposed."""
    return os.getenv("CONNINFPY_ENABLE_LOCAL_TEMPLATES", "").lower() in {"1", "true", "yes"}


def load_custom_datasets():
    if not local_dataset_templates_enabled():
        return {}
    path = os.path.expanduser("~/.conninfpy/custom_datasets.json")
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def safe_filename_part(text):
    """Return a compact filesystem-safe slug for generated downloads."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")
    return slug or "connectivity_analysis"

def current_contrast_name():
    """Build a stable label for exports and evidence packets."""
    plan = st.session_state.get("run_plan") or {}
    question = plan.get("question_type") or "connectivity_analysis"
    method = plan.get("method")
    return f"{question}_{method}" if method else question

def clear_downstream_results():
    """Clear outputs and completed background tasks invalidated upstream."""
    for key in (
        "inference_result",
        "edges_df",
        "companion_inference_result",
        "companion_edges_df",
        "companion_method",
        "decoded_df",
        "decoding_summary",
        "decoding_score",
        "evidence_packet",
        "narrative_text",
        "narrative_results",
        "run_plan",
    ):
        st.session_state[key] = None

    # Task objects retain their result payload after completion. Leaving one in
    # session state lets Tab 2 republish an old result after a dataset reload.
    # A still-running thread cannot be safely killed, but removing its task
    # handle prevents its stale payload from being promoted into this session.
    for task_key in ("inference_task", "decoding_task"):
        task = st.session_state.get(task_key)
        if task is not None and getattr(task, "status", None) == "running":
            task.status = "invalidated"
        st.session_state.pop(task_key, None)


def active_analysis_atlas(base_atlas=None):
    """Prefer ROI/dataset-specific atlas metadata over sidebar metadata."""
    if st.session_state.get("sub_atlas") is not None:
        return st.session_state.sub_atlas
    if st.session_state.get("dataset_atlas") is not None:
        return st.session_state.dataset_atlas
    return base_atlas


def atlas_for_annotation(atlas, n_nodes, *, warn=True):
    """Return atlas only when it matches the result dimensionality."""
    if atlas is None or len(atlas) == n_nodes:
        return atlas
    if warn:
        st.warning(
            f"Atlas metadata has {len(atlas)} ROIs, but the inference result has {n_nodes} nodes. "
            "The significant-edge table will be shown without atlas annotations."
        )
    return None


def data_is_fisher_z():
    """Use explicit ingestion metadata instead of value-range heuristics."""
    return st.session_state.get("connectivity_data_kind") == "fisher_z"


def result_is_stale(run_plan=None):
    """True when current loaded data/sidebar atlas no longer match a result."""
    plan = run_plan if run_plan is not None else st.session_state.get("run_plan")
    if not plan:
        return False
    current_hash = st.session_state.get("current_settings_hash")
    if plan.get("loaded_settings_hash") != current_hash:
        return True
    current_atlas_sig = st.session_state.get("active_atlas_signature")
    if plan.get("active_atlas_signature") != current_atlas_sig:
        return True
    return False


def effect_direction_labels(run_plan=None):
    """Return reader-facing labels for positive/negative statistical tails."""
    plan = run_plan if run_plan is not None else (st.session_state.get("run_plan") or {})
    question = plan.get("question_type")

    if question == "Group Difference":
        target = plan.get("target_group")
        reference = plan.get("reference_group")
        if target is None or reference is None:
            return {
                "positive": "Positive group effect",
                "negative": "Negative group effect",
                "positive_title": "Positive group effect",
                "negative_title": "Negative group effect",
            }
        
        target_str = str(target)
        reference_str = str(reference)
        if target_str.isdigit() or (target_str.startswith("Group") and target_str[5:].strip().isdigit()):
            target_lbl = f"Group {target_str}" if target_str.isdigit() else target_str
        else:
            target_lbl = target_str
            
        if reference_str.isdigit() or (reference_str.startswith("Group") and reference_str[5:].strip().isdigit()):
            reference_lbl = f"Group {reference_str}" if reference_str.isdigit() else reference_str
        else:
            reference_lbl = reference_str
            
        return {
            "positive": f"{target_lbl} > {reference_lbl}",
            "negative": f"{reference_lbl} > {target_lbl}",
            "positive_title": f"Higher connectivity in {target_lbl}",
            "negative_title": f"Higher connectivity in {reference_lbl}",
        }

    if question == "Paired Condition":
        target = plan.get("target_condition")
        baseline = plan.get("baseline_condition")
        if target is None or baseline is None:
            return {
                "positive": "Positive paired effect",
                "negative": "Negative paired effect",
                "positive_title": "Positive paired effect",
                "negative_title": "Negative paired effect",
            }
        return {
            "positive": f"{target} > {baseline}",
            "negative": f"{baseline} > {target}",
            "positive_title": f"{target} > {baseline}",
            "negative_title": f"{baseline} > {target}",
        }

    predictor = plan.get("interest_var") or plan.get("predictor") or "predictor"
    if question == "Continuous Predictor":
        return {
            "positive": f"Positive association with {predictor}",
            "negative": f"Negative association with {predictor}",
            "positive_title": f"Positive association with {predictor}",
            "negative_title": f"Negative association with {predictor}",
        }

    return {
        "positive": "Positive effect",
        "negative": "Negative effect",
        "positive_title": "Positive effect",
        "negative_title": "Negative effect",
    }

def resolve_project_path(user_path):
    """Resolve a user-entered path while keeping writes inside the project."""
    candidate = Path(user_path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    resolved = candidate.resolve()
    root = PROJECT_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Save path must stay inside the project: {root}")
    return resolved

def atlas_has_networks(atlas):
    """True when atlas has usable, non-placeholder network labels."""
    if atlas is None or getattr(atlas, "networks", None) is None:
        return False
    labels = [str(x).strip() for x in atlas.networks if str(x).strip()]
    return bool(labels) and set(labels) != {"Unknown"}

def atlas_has_coords(atlas):
    """True when atlas has finite MNI-like coordinates for all ROIs."""
    if atlas is None or getattr(atlas, "coords", None) is None:
        return False
    coords = np.asarray(atlas.coords)
    return coords.ndim == 2 and coords.shape[1] == 3 and np.isfinite(coords).all()

def atlas_label(atlas, fallback="No atlas metadata"):
    if atlas is None:
        return fallback
    return getattr(atlas, "source", None) or f"Custom atlas ({len(atlas)} ROIs)"

def atlas_from_dataframe(df, *, source="Custom atlas"):
    """Build AtlasInfo from a flexible custom atlas CSV dataframe."""
    column_map = {str(c).strip().lower(): c for c in df.columns}
    label_col = next(
        (column_map[c] for c in ("name", "label", "roi", "roi_name", "region", "parcel") if c in column_map),
        None,
    )
    if label_col is None:
        raise ValueError("Custom atlas CSV must include a label column such as name, label, roi, or roi_name.")

    network_col = next(
        (column_map[c] for c in ("network", "system", "module", "community", "lobe") if c in column_map),
        None,
    )
    hemi_col = next(
        (column_map[c] for c in ("hemisphere", "hemi", "side") if c in column_map),
        None,
    )

    labels = df[label_col].astype(str).str.strip().tolist()
    if any(not label for label in labels):
        raise ValueError("Custom atlas labels must be non-empty.")

    if network_col is None:
        networks = ["Unknown"] * len(labels)
    else:
        networks = df[network_col].fillna("Unknown").astype(str).str.strip().replace("", "Unknown").tolist()

    coords = None
    coord_cols = []
    for names in (("x", "y", "z"), ("mni_x", "mni_y", "mni_z")):
        if all(name in column_map for name in names):
            coord_cols = [column_map[name] for name in names]
            break
    if coord_cols:
        coords_df = df[coord_cols].apply(pd.to_numeric, errors="coerce")
        if not coords_df.isna().all().all():
            if coords_df.isna().any().any():
                raise ValueError("Custom atlas coordinate columns must be complete numeric x/y/z values or omitted.")
            coords = coords_df.to_numpy(dtype=float)

    hemisphere = None
    if hemi_col is not None:
        hemisphere = df[hemi_col].fillna("").astype(str).str.strip().tolist()
        if not any(hemisphere):
            hemisphere = None

    return AtlasInfo(
        labels=labels,
        networks=networks,
        coords=coords,
        hemisphere=hemisphere,
        source=source,
    )

def load_custom_atlas_csv(path_or_buffer, *, source="Custom atlas"):
    df = pd.read_csv(path_or_buffer)
    if df.empty:
        raise ValueError("Custom atlas CSV has no rows.")
    return atlas_from_dataframe(df, source=source)

def make_sub_atlas(atlas, roi_idx):
    """Create an AtlasInfo restricted to selected ROI indices."""
    if atlas is None:
        return None
    coords = atlas.coords[roi_idx] if atlas.coords is not None else None
    hemisphere = [atlas.hemisphere[i] for i in roi_idx] if atlas.hemisphere is not None else None
    return AtlasInfo(
        labels=[atlas.labels[i] for i in roi_idx],
        networks=[atlas.networks[i] for i in roi_idx],
        coords=coords,
        hemisphere=hemisphere,
        source=f"{atlas.source} (Subnetwork of {len(roi_idx)} ROIs)",
    )


HELP_TEXT = {
    "data_source": """
**Data Ingestion Methods**
- **Generate Synthetic Dataset**: Simulates groups/conditions and connectivity changes for testing and debugging.
- **Built-in / Standard Dataset Structures**: Pre-configured pipelines for standard cohorts like `ABIDE-mini` and `OpenClose`.
- **Dataset Manifest (data.yaml)**: Imports custom, full, or private local datasets using portable YAML manifest configurations.
""",
    "built_in_demo": """
**Built-in Demo Datasets**
- **ABIDE-mini**: Curated 5-site subset of ABIDE (204 subjects) preserving complete phenotypic columns and group balances.
- **OpenClose**: Paired Eyes-Open vs. Eyes-Closed resting-state dataset using a 182-ROI unified Schaefer-200 mask.
*Note: Demo datasets are for trying out workflow features and diagnostic tests, not definitive scientific analysis. Full ABIDE, stress, and local/private datasets should be imported from a manifest.*
""",
    "openclose_multisite": """
**Open-Close Paired Demo**

- **Both sites (IHB + China)** loads the paired Eyes-Open and Eyes-Closed observations from both cohorts and adds a `site` column.
- Every subject has both conditions at the same site. Use **Paired Condition** inference with `subject_id` and `condition`.
- This is a useful demonstration that the within-subject Open-Close contrast is less sensitive to between-site offsets than a between-subject comparison. The site column remains available for design inspection and sensitivity checks.
""",
    "china_close_close": """
**China Close-Close Test-Retest Example**

This is a paired null/control comparison of two Eyes-Closed runs from the same China participants. It is useful for checking method specificity: a valid Open-Close result should not be reproduced strongly when comparing repeated closed-eyes runs. One participant with an incomplete second run is excluded automatically.
""",
    "manifest_import": """
**Manifest Ingestion**
- A `data.yaml` file declares `schema_version`, `name`, a supported `loader`, and its file `paths`. Add `params` for loader options, `preprocessing` for requested connectivity construction, and `checks` for expected dimensions.
- Relative paths are resolved from the manifest's own folder. Use `BUILTIN:<file>` for packaged atlas assets, for example `BUILTIN:bna246.csv`.
- For ready connectivity matrices, choose the matching loader and include atlas metadata whenever ROI labels, networks, coordinates, or brain plots are needed.
- **Prepared Zerssen example:** use `loader: CustomPreparedZerssenLoader`; under `paths`, provide `prepared_npz`, `set84_bnt_ids`, and `atlas_metadata: BUILTIN:bna246.csv`. Set `params.matrix_key` to select the analysis array. `connectivity_pearson_z`, `connectivity_tangent`, and `connectivity_tangent_hc_ref` use the full 246-ROI BNA atlas; `set84m_pearson_z` and `set84w_pearson_z` use the ordered 84-ROI BNA subset defined by BNT IDs. Set `data_kind: fisher_z` for the Pearson-z arrays and `data_kind: correlation` for tangent arrays. Update `checks.expected_rois` to match the selected 84- or 246-ROI matrix.
- The dataset-provided atlas takes precedence after loading. The sidebar atlas is then only a reference for datasets without their own metadata.
- See `datasets/zerssen_prepared.yaml` for a working local template; replace its machine-specific paths with your own.
""",
    "source_type": """
**Source File Types**
- **fMRIPrep derivatives / preprocessed NIfTIs**: Volumetric neuroimaging directories that require anatomical ROI atlas extraction.
- **Timeseries directories / arrays**: Denoised timeseries matrices requiring connectivity correlation construction.
- **Connectivity matrices**: Ready-to-inference Fisher-z or Pearson correlation matrices.
""",
    "dataset_preview": """
**Dataset Preview Metrics**
- **Subjects**: Number of unique participants.
- **Observations**: Total runs/sessions (observations may exceed subjects for paired/repeated designs).
- **ROIs**: Parcellation node count (must align with the reference atlas).
- **File size/count**: Helps verify if files are correct and identify unintended full-dataset imports.
""",
    "source_validation": """
**Source Validation Checks**
- **Errors**: Structural issues (non-square matrices, NaNs, missing folders, phenotype mismatch) that block loading.
- **Warnings**: Non-blocking alerts (e.g. low-coverage ROIs, minor group imbalances) indicating potential limitations in interpretation.
""",
    "preprocessing": """
**Ingestion Preprocessing**
- **Correlation Estimator**: Converts timeseries data to connectivity matrices using Pearson or Spearman correlation.
- **Fisher r-to-z**: Standard transformation recommended before running statistical inference.
- **Diagonal Zeroing**: Diagonal elements are set to zero before Fisher z-transform to prevent infinite values.
""",
    "inference_ready": """
**Inference-Ready Checks**
- Enforces that data is a 3D connectivity tensor of shape `(observations, ROIs, ROIs)`.
- Verified symmetric, finite, and aligned with phenotype rows before running statistics.
""",
    "analysis_setup": """
**Analysis Design Binding**
- **Group comparison**: Compares discrete groups (e.g., patient vs. control) using a group column.
- **Paired comparison**: Evaluates within-subject changes across paired conditions (e.g., open vs. close).
- **Continuous predictor**: Models correlation against continuous behavioral scales (e.g. ADOS) with optional confound regression.
- **Site harmonization**: Adjusts for batch effects in multi-site designs using ComBat or ComBat-Apply strategies.
""",
    "run_plan": """
**Run Plan Review**
- **Permutation Budget**: Selects the statistical precision. Larger budgets (e.g., 5000+ permutations) are required for paper-grade inference.
- **GLM / Contrast**: Confirms interest predictors, covariates, and the contrast matrix.
- **Reloading**: If settings were changed after importing, remember to reload the dataset to refresh stats matrices.
""",
    "inference_results": """
**Inference Results**
- Displays the statistical significance of network edges after family-wise error rate (FWER) correction.
- Interactive network graphs show the enhanced subnetworks.
- Node summary lists anatomical regions sorted by degree burden.
""",
    "meta_analytic_decoding": """
**Meta-Analytic Decoding (NiMARE)**
Connects the coordinates of your significant functional subnetwork to public cognitive databases to associate brain patterns with specific literature terms.

### 📚 Databases
*   **[NeuroQuery](https://neuroquery.org/) (Recommended Default):**
    *   Uses a predictive machine-learning model mapping text semantic concepts to spatial activity.
    *   Excellent for **network-level functional profiling** because it handles synonyms, smooths over sparse literature, and generates coherent concept maps.
*   **[Neurosynth](https://neurosynth.org/):**
    *   The classic approach using exact keyword-association meta-analysis.
    *   Excellent for **high-specificity reverse inference** (e.g., finding exact localized regions uniquely active for a single cognitive task), but can be noisy or fragmented for whole networks.

### 🧠 Strategies
*   **Combined Region Decoding:**
    Merges all coordinates of nodes in the significant network into a single spatial pattern. Highly recommended for interpreting global network functional profiles.
*   **Discrete ROI Overlap:**
    Decodes each individual ROI independently, allowing a region-by-region audit of the subnetwork nodes.

### 💡 Usage Advice
1. Use **NeuroQuery + Combined Region Decoding** as your starting point to get a smooth, concept-level description of what your network is doing.
2. Cross-reference with **Neurosynth** to test for exact, highly specific literature associations for your top high-burden nodes.
3. Be cautious: meta-analytic associations are correlation-based context, not causal explanations.
""",
    "narrative_report": """
**Narrative Report**
- Generates a clinical-grade narrative report constrained by the meta-analytic and statistical evidence.
- Highlights key network hubs and cognitive associations.
""",
    "workspace_documentation": """
**Workspace Documentation**
- Indexes and displays active research notes, design plans, and implementation logs from the wiki.
"""
}


def render_help(key: str):
    if key in HELP_TEXT:
        with st.popover("ℹ️ Help Info", use_container_width=False):
            st.markdown(HELP_TEXT[key])


def list_manifest_files() -> list[str]:
    """List all YAML manifest files directly under the datasets/ folder of the project root."""
    datasets_dir = PROJECT_ROOT / "datasets"
    if not datasets_dir.exists():
        return []
    yamls = list(datasets_dir.glob("*.yaml")) + list(datasets_dir.glob("*.yml"))
    rel_yamls = []
    for y in yamls:
        try:
            rel_path = str(y.relative_to(PROJECT_ROOT))
            rel_yamls.append(rel_path)
        except Exception:
            rel_yamls.append(str(y))
    return sorted(rel_yamls)


def align_atlas_coordinates(target_atlas, reference_atlas):
    """Align and copy coordinates from reference_atlas to target_atlas by matching ROI names."""
    if target_atlas is None or reference_atlas is None:
        return target_atlas
    if getattr(reference_atlas, "coords", None) is None:
        return target_atlas
        
    ref_coords_map = {}
    for label, coord in zip(reference_atlas.labels, reference_atlas.coords):
        ref_coords_map[label.strip().lower()] = coord
        
    new_coords = []
    has_any = False
    for label in target_atlas.labels:
        clean = label.strip().lower()
        if clean in ref_coords_map:
            new_coords.append(ref_coords_map[clean])
            has_any = True
        else:
            new_coords.append([np.nan, np.nan, np.nan])
            
    if has_any:
        target_atlas.coords = np.array(new_coords)
    return target_atlas
