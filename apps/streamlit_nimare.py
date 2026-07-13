import os
from importlib.util import find_spec

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from conninfpy.atlas import AtlasInfo
from conninfpy.interpret.llm_narrative import load_dotenv_manually

# Import modular view components
from apps.components.data_ingestion import render_data_ingestion_view
from apps.components.design_inference import render_design_inference_view
from apps.components.inference_results import render_inference_results_view
from apps.components.meta_decoding import render_meta_decoding_view
from apps.components.narrative_report import render_narrative_report_view
from apps.components.workspace_docs import render_workspace_docs_view
from apps.utils.helpers import atlas_has_coords, atlas_has_networks, load_custom_atlas_csv

# Load env variables on startup
load_dotenv_manually()

# Page setup
st.set_page_config(
    page_title="ConnInfPy End-to-End Analysis Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Keep layout adjustments theme-neutral so Streamlit can follow light or dark mode.
st.markdown("""
<style>
    .app-title {
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.1rem;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .app-subtitle {
        font-size: 0.9rem;
        opacity: 0.72;
        margin-bottom: 1.2rem;
    }
    .custom-card {
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 16px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        border-radius: 0px;
        font-size: 0.9rem;
        font-weight: 500;
        padding-left: 12px;
        padding-right: 12px;
    }
</style>
""", unsafe_allow_html=True)


def decoding_is_enabled() -> bool:
    """Return whether this deployment can expose the optional NiMARE workflow."""
    try:
        configured = st.secrets.get(
            "CONNINFPY_ENABLE_DECODING",
            os.getenv("CONNINFPY_ENABLE_DECODING", "auto"),
        )
    except FileNotFoundError:
        configured = os.getenv("CONNINFPY_ENABLE_DECODING", "auto")
    configured = str(configured).strip().lower()
    if configured in {"0", "false", "no", "off"}:
        return False
    return find_spec("nimare") is not None


DECODING_ENABLED = decoding_is_enabled()

# ----------------- SIDEBAR GLOBAL SETTINGS -----------------
st.sidebar.markdown("## Global Settings")

# Instantiate default atlas based on selection
@st.cache_resource
def get_atlas(name):
    if name == "Schaefer-100 Yeo-7":
        return AtlasInfo.schaefer_100_yeo7()
    elif name == "Schaefer-200 Yeo-7":
        return AtlasInfo.schaefer_200_yeo7()
    elif name == "Schaefer-400 Yeo-7":
        return AtlasInfo.schaefer_400_yeo7()
    else:
        return AtlasInfo.bna_246()

atlas_mode = st.sidebar.radio(
    "Atlas Metadata",
    ["No atlas metadata", "Bundled atlas", "Custom atlas CSV"],
    index=1,
)

base_atlas = None
atlas_choice = atlas_mode

if atlas_mode == "Bundled atlas":
    atlas_choice = st.sidebar.selectbox(
        "Reference Parcellation",
        ["Schaefer-100 Yeo-7", "Schaefer-200 Yeo-7", "Schaefer-400 Yeo-7", "BNA-246"]
    )
    base_atlas = get_atlas(atlas_choice)

    # Sidebar resting state network visualization palette
    if "Schaefer" in atlas_choice:
        schaefer_palette = {
            "Schaefer-100 Yeo-7": "apps/assets/yeo7_schaefer100_sidebar_palette.png",
            "Schaefer-200 Yeo-7": "apps/assets/yeo7_schaefer200_sidebar_palette.png",
            "Schaefer-400 Yeo-7": "apps/assets/yeo7_schaefer400_sidebar_palette.png",
        }[atlas_choice]
        st.sidebar.image(
            schaefer_palette,
            use_container_width=True
        )
        st.sidebar.caption("Yeo-7 network palette mapping")
    elif atlas_choice == "BNA-246":
        st.sidebar.image(
            "apps/assets/bna246_sidebar_palette.png",
            use_container_width=True
        )
        st.sidebar.caption("Brainnetome lobe palette mapping")
elif atlas_mode == "Custom atlas CSV":
    custom_atlas_source = st.sidebar.radio("Custom atlas source", ["Path", "Upload"], horizontal=True)
    try:
        if custom_atlas_source == "Path":
            custom_atlas_path = st.sidebar.text_input("Atlas CSV path", value="")
            if custom_atlas_path:
                base_atlas = load_custom_atlas_csv(custom_atlas_path, source=f"Custom atlas: {custom_atlas_path}")
                atlas_choice = f"Custom atlas ({len(base_atlas)} ROIs)"
        else:
            uploaded_atlas = st.sidebar.file_uploader("Upload atlas CSV", type=["csv"])
            if uploaded_atlas is not None:
                base_atlas = load_custom_atlas_csv(uploaded_atlas, source=f"Custom atlas: {uploaded_atlas.name}")
                atlas_choice = f"Custom atlas ({len(base_atlas)} ROIs)"
    except Exception as e:
        st.sidebar.error(f"Could not load custom atlas: {e}")
        base_atlas = None
        atlas_choice = "Custom atlas CSV (not loaded)"

if base_atlas is None:
    st.sidebar.caption("Statistics can run without atlas metadata. ROI labels, network-aware methods, and NiMARE decoding need metadata.")
else:
    caps = []
    caps.append(f"{len(base_atlas)} ROIs")
    caps.append("networks" if atlas_has_networks(base_atlas) else "no networks")
    caps.append("coordinates" if atlas_has_coords(base_atlas) else "no coordinates")
    st.sidebar.caption("Atlas loaded: " + ", ".join(caps))

dataset_atlas = st.session_state.get("dataset_atlas")
if dataset_atlas is not None:
    st.sidebar.caption(
        "Active dataset atlas: "
        f"{len(dataset_atlas)} ROIs from {getattr(dataset_atlas, 'source', 'dataset metadata')}."
    )

st.session_state["active_atlas_signature"] = "|".join([
    atlas_mode,
    atlas_choice,
    str(len(base_atlas)) if base_atlas is not None else "0",
    str(getattr(base_atlas, "source", "")) if base_atlas is not None else "",
])

# ----------------- SESSION STATE INIT -----------------
DATA_TAB = "1. Data Ingestion & Preprocessing"
DESIGN_TAB = "2. Design & Inference"
RESULTS_TAB = "3. Inference Results"
DECODING_TAB = "4. Meta-Analytic Decoding"
NARRATIVE_TAB = "5. Narrative Report"
DOCS_TAB = "6. Workspace Documentation"

tabs_list = [
    DATA_TAB,
    DESIGN_TAB,
    RESULTS_TAB,
    DECODING_TAB,
    NARRATIVE_TAB,
    DOCS_TAB,
]
if "next_tab" in st.session_state and st.session_state.next_tab is not None:
    st.session_state.active_tab = st.session_state.next_tab
    st.session_state.next_tab = None

if "active_tab" not in st.session_state or st.session_state.active_tab not in tabs_list:
    st.session_state.active_tab = tabs_list[0]

if 'connectivity_data' not in st.session_state:
    st.session_state.connectivity_data = None  # (n_subjects, N, N)
if 'connectivity_data_kind' not in st.session_state:
    st.session_state.connectivity_data_kind = None
if 'pheno_df' not in st.session_state:
    st.session_state.pheno_df = None
if 'inference_result' not in st.session_state:
    st.session_state.inference_result = None
if 'edges_df' not in st.session_state:
    st.session_state.edges_df = None
if 'companion_inference_result' not in st.session_state:
    st.session_state.companion_inference_result = None
if 'companion_edges_df' not in st.session_state:
    st.session_state.companion_edges_df = None
if 'companion_method' not in st.session_state:
    st.session_state.companion_method = None
if 'decoded_df' not in st.session_state:
    st.session_state.decoded_df = None
if 'decoding_summary' not in st.session_state:
    st.session_state.decoding_summary = None
if 'decoding_score' not in st.session_state:
    st.session_state.decoding_score = None
if 'evidence_packet' not in st.session_state:
    st.session_state.evidence_packet = None
if 'narrative_text' not in st.session_state:
    st.session_state.narrative_text = None
if 'narrative_results' not in st.session_state:
    st.session_state.narrative_results = None
if 'sub_atlas' not in st.session_state:
    st.session_state.sub_atlas = None
if 'dataset_atlas' not in st.session_state:
    st.session_state.dataset_atlas = None
if 'roi_indices' not in st.session_state:
    st.session_state.roi_indices = None
if 'llm_provider' not in st.session_state:
    st.session_state.llm_provider = "mock"
if 'llm_model' not in st.session_state:
    st.session_state.llm_model = ""
if 'llm_api_key' not in st.session_state:
    st.session_state.llm_api_key = ""

# Title
title_col1, title_col2 = st.columns([5, 1])
with title_col1:
    st.markdown('<div class="app-title">ConnInfPy</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">End-to-end connectivity inference workspace</div>', unsafe_allow_html=True)
with title_col2:
    st.image("apps/assets/brain_connectivity_dashboard_right_panel.png", width=120)

# Stepper task states
decoding_task = st.session_state.get("decoding_task")
decoding_is_running = decoding_task is not None and getattr(decoding_task, "status", "") == "running"

inference_task = st.session_state.get("inference_task")
inference_is_running = inference_task is not None and getattr(inference_task, "status", "") == "running"

# Dynamic Progress Stepper Bar (Clinical Neuro Lab style, no emojis)
status_cols = st.columns(5)

# Ingestion status
with status_cols[0]:
    is_active = st.session_state.active_tab == DATA_TAB
    if st.session_state.connectivity_data is not None:
        label = "1. Data Ingestion (Loaded)"
    else:
        label = "1. Data Ingestion (Empty)"
    if st.button(label, key="nav_step_1", type="primary" if is_active else "secondary", use_container_width=True):
        st.session_state.next_tab = DATA_TAB
        st.rerun()
        
# Design & Inference status
with status_cols[1]:
    is_active = st.session_state.active_tab == DESIGN_TAB
    if inference_is_running:
        label = "2. Inference (Running)"
    elif st.session_state.inference_result is not None:
        label = "2. Inference (Completed)"
    elif st.session_state.connectivity_data is not None:
        label = "2. Inference (Ready)"
    else:
        label = "2. Inference (Awaiting Data)"
    if st.button(label, key="nav_step_2", type="primary" if is_active else "secondary", use_container_width=True):
        st.session_state.next_tab = DESIGN_TAB
        st.rerun()
        
# Results status
with status_cols[2]:
    is_active = st.session_state.active_tab == RESULTS_TAB
    if st.session_state.edges_df is not None and not st.session_state.edges_df.empty:
        label = "3. Results (Significant Edges)"
    else:
        label = "3. Results (Awaiting Inference)"
    if st.button(label, key="nav_step_3", type="primary" if is_active else "secondary", use_container_width=True):
        st.session_state.next_tab = RESULTS_TAB
        st.rerun()
        
# Decoding status
with status_cols[3]:
    is_active = st.session_state.active_tab == DECODING_TAB
    if not DECODING_ENABLED:
        label = "4. Decoding (Offline Version)"
    elif decoding_is_running:
        label = "4. Decoding (Running)"
    elif st.session_state.decoded_df is not None:
        label = "4. Decoding (Completed)"
    elif st.session_state.edges_df is not None and not st.session_state.edges_df.empty:
        label = "4. Decoding (Ready)"
    else:
        label = "4. Decoding (Awaiting Results)"
    if st.button(label, key="nav_step_4", type="primary" if is_active else "secondary", use_container_width=True):
        st.session_state.next_tab = DECODING_TAB
        st.rerun()

# Narrative status
with status_cols[4]:
    is_active = st.session_state.active_tab == NARRATIVE_TAB
    if not DECODING_ENABLED:
        label = "5. Narrative (Offline Decoding)"
    elif decoding_is_running:
        label = "5. Narrative (Waiting for Decoding)"
    elif st.session_state.narrative_text is not None:
        label = "5. Narrative (Generated)"
    elif st.session_state.evidence_packet is not None:
        label = "5. Narrative (Ready)"
    else:
        label = "5. Narrative (Awaiting Decoding)"
    if st.button(label, key="nav_step_5", type="primary" if is_active else "secondary", use_container_width=True):
        st.session_state.next_tab = NARRATIVE_TAB
        st.rerun()

st.markdown("---")

st.sidebar.markdown("---")
st.sidebar.markdown("## Documentation")
doc_active = st.session_state.active_tab == DOCS_TAB
if st.sidebar.button(
    "📖 Workspace Documentation",
    key="nav_docs_btn",
    type="primary" if doc_active else "secondary",
    use_container_width=True
):
    st.session_state.active_tab = DOCS_TAB
    st.rerun()

active_tab = st.session_state.active_tab

# ----------------- ROUTING TO modularized tabs -----------------
if active_tab == DATA_TAB:
    render_data_ingestion_view(base_atlas, atlas_choice, tabs_list)
elif active_tab == DESIGN_TAB:
    render_design_inference_view(base_atlas, tabs_list)
elif active_tab == RESULTS_TAB:
    render_inference_results_view(base_atlas)
elif active_tab == DECODING_TAB:
    render_meta_decoding_view(base_atlas, decoding_enabled=DECODING_ENABLED)
elif active_tab == NARRATIVE_TAB:
    if DECODING_ENABLED:
        render_narrative_report_view()
    else:
        st.markdown("### Narrative Report")
        st.info("Narrative reporting is available after meta-analytic decoding in the offline ConnInfPy version.")
elif active_tab == DOCS_TAB:
    render_workspace_docs_view()

# Trigger reload: 2026-07-09 17:03
