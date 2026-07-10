import os
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

# Styling Injection
st.markdown("""
<style>
    /* App background */
    .stApp {
        background-color: #F7F8FA;
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E5E7EB;
    }
    /* Typography */
    .app-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #202633;
        margin-bottom: 0.1rem;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .app-subtitle {
        font-size: 0.9rem;
        color: #6B7280;
        margin-bottom: 1.2rem;
    }
    /* Light functional panels */
    .custom-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        box-shadow: none;
    }
    /* Bordered containers background fill */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #F0F7FF;
        border: 1px solid #D6E4FF !important;
        border-radius: 8px !important;
        padding: 1.2rem !important;
    }
    /* Tab bar override */
    .stTabs [data-baseweb="tab-list"] {
        gap: 16px;
        border-bottom: 1px solid #E5E7EB;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        background-color: transparent;
        border-radius: 0px;
        color: #6B7280;
        font-size: 0.9rem;
        font-weight: 500;
        padding-left: 12px;
        padding-right: 12px;
        border-bottom: 2px solid transparent;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #202633;
    }
    .stTabs [aria-selected="true"] {
        color: #2563EB !important;
        border-bottom: 2px solid #2563EB !important;
        background-color: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

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
        st.sidebar.image(
            "apps/assets/yeo7_schaefer100_sidebar_palette.png",
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

st.session_state["active_atlas_signature"] = "|".join([
    atlas_mode,
    atlas_choice,
    str(len(base_atlas)) if base_atlas is not None else "0",
    str(getattr(base_atlas, "source", "")) if base_atlas is not None else "",
])

# ----------------- SESSION STATE INIT -----------------
tabs_list = [
    "1. Data Ingestion & Preprocessing",
    "2. Design & Inference",
    "3. Inference Results",
    "4. Meta-Analytic Decoding",
    "5. Narrative Report",
    "6. Workspace Documentation"
]
if "next_tab" in st.session_state and st.session_state.next_tab is not None:
    st.session_state.active_tab = st.session_state.next_tab
    st.session_state.next_tab = None

if "active_tab" not in st.session_state:
    st.session_state.active_tab = tabs_list[0]

if 'connectivity_data' not in st.session_state:
    st.session_state.connectivity_data = None  # (n_subjects, N, N)
if 'pheno_df' not in st.session_state:
    st.session_state.pheno_df = None
if 'inference_result' not in st.session_state:
    st.session_state.inference_result = None
if 'edges_df' not in st.session_state:
    st.session_state.edges_df = None
if 'decoded_df' not in st.session_state:
    st.session_state.decoded_df = None
if 'evidence_packet' not in st.session_state:
    st.session_state.evidence_packet = None
if 'narrative_text' not in st.session_state:
    st.session_state.narrative_text = None
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

# Dynamic Progress Stepper Bar (Clinical Neuro Lab style, no emojis)
status_cols = st.columns(5)

# Ingestion status
with status_cols[0]:
    is_active = st.session_state.active_tab == tabs_list[0]
    if st.session_state.connectivity_data is not None:
        label = "1. Data Ingestion (Loaded)"
    else:
        label = "1. Data Ingestion (Empty)"
    if st.button(label, key="nav_step_1", type="primary" if is_active else "secondary", use_container_width=True):
        st.session_state.next_tab = tabs_list[0]
        st.rerun()
        
# Design & Inference status
with status_cols[1]:
    is_active = st.session_state.active_tab == tabs_list[1]
    if st.session_state.inference_result is not None:
        label = "2. Inference (Completed)"
    elif st.session_state.connectivity_data is not None:
        label = "2. Inference (Ready)"
    else:
        label = "2. Inference (Awaiting Data)"
    if st.button(label, key="nav_step_2", type="primary" if is_active else "secondary", use_container_width=True):
        st.session_state.next_tab = tabs_list[1]
        st.rerun()
        
# Results status
with status_cols[2]:
    is_active = st.session_state.active_tab == tabs_list[2]
    if st.session_state.edges_df is not None and not st.session_state.edges_df.empty:
        label = "3. Results (Significant Edges)"
    else:
        label = "3. Results (Awaiting Inference)"
    if st.button(label, key="nav_step_3", type="primary" if is_active else "secondary", use_container_width=True):
        st.session_state.next_tab = tabs_list[2]
        st.rerun()
        
# Decoding status
with status_cols[3]:
    is_active = st.session_state.active_tab == tabs_list[3]
    if st.session_state.decoded_df is not None:
        label = "4. Decoding (Completed)"
    elif st.session_state.edges_df is not None and not st.session_state.edges_df.empty:
        label = "4. Decoding (Ready)"
    else:
        label = "4. Decoding (Awaiting Results)"
    if st.button(label, key="nav_step_4", type="primary" if is_active else "secondary", use_container_width=True):
        st.session_state.next_tab = tabs_list[3]
        st.rerun()
        
# Narrative status
with status_cols[4]:
    is_active = st.session_state.active_tab == tabs_list[4]
    if st.session_state.narrative_text is not None:
        label = "5. Narrative (Generated)"
    elif st.session_state.evidence_packet is not None:
        label = "5. Narrative (Ready)"
    else:
        label = "5. Narrative (Awaiting Decoding)"
    if st.button(label, key="nav_step_5", type="primary" if is_active else "secondary", use_container_width=True):
        st.session_state.next_tab = tabs_list[4]
        st.rerun()

st.markdown("---")

st.sidebar.markdown("---")
st.sidebar.markdown("## Navigation")

active_tab = st.sidebar.radio("Go to:", tabs_list, key="active_tab")

# ----------------- ROUTING TO modularized tabs -----------------
if active_tab == tabs_list[0]:
    render_data_ingestion_view(base_atlas, atlas_choice, tabs_list)
elif active_tab == tabs_list[1]:
    render_design_inference_view(base_atlas, tabs_list)
elif active_tab == tabs_list[2]:
    render_inference_results_view(base_atlas)
elif active_tab == tabs_list[3]:
    render_meta_decoding_view(base_atlas)
elif active_tab == tabs_list[4]:
    render_narrative_report_view()
elif active_tab == tabs_list[5]:
    render_workspace_docs_view()

# Trigger reload: 2026-07-09 17:03
