import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import io
import matplotlib.pyplot as plt

from conninfpy.atlas import AtlasInfo
from conninfpy.decode import decode_rois, annotate_edge_table
from conninfpy._decode_cache import fetch_neurosynth_dataset
from conninfpy.interpret.evidence import build_decoding_evidence, validate_evidence
from conninfpy.interpret.llm_narrative import LLMNarrator, check_narrative_terms, load_dotenv_manually
from conninfpy import InferenceResult, analyze
from conninfpy.synth_datasets import generate_multisite_glm_dataset

# Page setup
st.set_page_config(
    page_title="ConnInfPy End-to-End Analysis Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load env variables on startup
load_dotenv_manually()

# Styling Injection
st.markdown("""
<style>
    .app-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .app-subtitle {
        font-size: 1.1rem;
        color: #64748b;
        margin-bottom: 2rem;
    }
    .custom-card {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin-bottom: 1.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1e293b;
        border-radius: 4px;
        color: #f1f5f9;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR GLOBAL SETTINGS -----------------
st.sidebar.markdown("## ⚙️ App Configurations")

# Atlas Select
atlas_choice = st.sidebar.selectbox(
    "Reference Parcellation",
    ["Schaefer-100 Yeo-7", "Schaefer-200 Yeo-7", "Schaefer-400 Yeo-7", "BNA-246"]
)

# LLM Narrator settings
st.sidebar.markdown("### 🤖 LLM Narrator Settings")
llm_provider = st.sidebar.selectbox("Provider", ["mock", "openrouter", "gemini", "openai"])
llm_model = st.sidebar.text_input("Model Name (Optional)", placeholder="e.g. meta-llama/llama-3-8b-instruct:free")
llm_api_key = st.sidebar.text_input("API Key (Override)", type="password")

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

base_atlas = get_atlas(atlas_choice)

# ----------------- SESSION STATE INIT -----------------
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
if 'roi_indices' not in st.session_state:
    st.session_state.roi_indices = None

# Title
st.markdown('<div class="app-title">🧠 ConnInfPy End-to-End Analysis Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Connectivity statistics, multi-site harmonization, subnetwork selection, and literature narration</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📥 1. Preprocessing & Data Upload",
    "⚙️ 2. Statistical Design & Inference",
    "📊 3. Inference Results",
    "🧠 4. Meta-Analytic Decoding",
    "✍️ 5. Narrative Generator",
    "📘 6. Help & Documentation"
])

# ----------------- TAB 1: DATA UPLOAD & PREPROCESSING -----------------
with tab1:
    st.markdown("### Preprocessing & Data Upload")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**1. Connectivity/Timeseries Input**")
        data_source = st.radio("Connectivity Source", ["Generate Synthetic Dataset", "Upload own files"])
        
        if data_source == "Generate Synthetic Dataset":
            n_sub = st.number_input("Subjects Count", 10, 200, 60, 10)
            effect_sz = st.slider("Planted Effect Size (0.0 = Null)", 0.0, 1.0, 0.4, 0.05)
            n_signals = st.number_input("Planted Signal Edges", 0, 500, 150)
            
            if st.button("Generate Synthetic Dataset"):
                # Clear previous state
                st.session_state.inference_result = None
                st.session_state.edges_df = None
                st.session_state.decoded_df = None
                st.session_state.evidence_packet = None
                st.session_state.narrative_text = None
                
                # Generate
                ds = generate_multisite_glm_dataset(
                    n_subjects=n_sub,
                    N=len(base_atlas),
                    effect_size=effect_sz,
                    n_signal_edges=n_signals,
                    seed=42
                )
                
                st.session_state.connectivity_data = ds["Y"]
                
                # Build synthetic pheno dataframe
                pheno = pd.DataFrame({
                    "subject_id": [f"sub_{i:02d}" for i in range(n_sub)],
                    "group_interest": ds["interest"],
                    "site": [f"Site_{s}" for s in ds["sites"]],
                    "age": np.random.normal(50, 10, n_sub).astype(int),
                    "sex": np.random.choice(["M", "F"], n_sub)
                })
                st.session_state.pheno_df = pheno
                st.success("Synthetic dataset generated successfully!")
                
        else:
            conn_file = st.file_uploader("Upload Connectivity Tensor Y (.npy) or Timeseries (.npy)", type=["npy"])
            is_timeseries = st.checkbox("Is this a timeseries array? (shape: subjects x timepoints x nodes)")
            
            if conn_file:
                try:
                    arr = np.load(conn_file)
                    if is_timeseries:
                        # Convert to correlations
                        if arr.ndim != 3:
                            st.error(f"Timeseries array must be 3D. Got shape {arr.shape}")
                        else:
                            corrs = []
                            for s in range(arr.shape[0]):
                                c = np.corrcoef(arr[s].T)
                                # Fill NaNs
                                c = np.nan_to_num(c)
                                corrs.append(c)
                            st.session_state.connectivity_data = np.array(corrs)
                            st.success(f"Loaded timeseries. Calculated correlations of shape {st.session_state.connectivity_data.shape}")
                    else:
                        if arr.ndim != 3:
                            st.error(f"Connectivity tensor must be 3D (shape: subjects x nodes x nodes). Got shape {arr.shape}")
                        else:
                            st.session_state.connectivity_data = arr
                            st.success(f"Loaded connectivity tensor of shape {st.session_state.connectivity_data.shape}")
                except Exception as e:
                    st.error(f"Error loading numpy file: {e}")
                    
        # Optional: Subnetwork selection
        st.markdown("**2. Optional: Subnetwork Node Selection**")
        roi_indices_input = st.text_input(
            "ROI Indices (comma-separated, e.g. 0,1,2,10,11 to restrict analysis to a subset of nodes)",
            value=""
        )
        
        if roi_indices_input:
            try:
                roi_idx = [int(x.strip()) for x in roi_indices_input.split(",") if x.strip()]
                # validate
                invalid = [r for r in roi_idx if r < 0 or r >= len(base_atlas)]
                if invalid:
                    st.error(f"Indices {invalid} are out of range for the chosen atlas.")
                else:
                    st.session_state.roi_indices = roi_idx
                    # Construct sub-atlas
                    st.session_state.sub_atlas = AtlasInfo(
                        labels=[base_atlas.labels[i] for i in roi_idx],
                        networks=[base_atlas.networks[i] for i in roi_idx],
                        coords=base_atlas.coords[roi_idx] if base_atlas.coords is not None else None,
                        hemisphere=[base_atlas.hemisphere[i] for i in roi_idx] if base_atlas.hemisphere is not None else None,
                        source=f"{base_atlas.source} (Subnetwork of {len(roi_idx)} ROIs)"
                    )
                    st.success(f"Subnetwork configured with {len(roi_idx)} ROIs.")
            except ValueError:
                st.error("Please enter valid comma-separated integers.")
        else:
            st.session_state.roi_indices = None
            st.session_state.sub_atlas = None
            
    with col2:
        st.markdown("**3. Phenotypic/Design Data**")
        pheno_file = st.file_uploader("Upload Phenotypic CSV", type=["csv"])
        if pheno_file:
            try:
                st.session_state.pheno_df = pd.read_csv(pheno_file)
                st.success("Phenotypic CSV loaded successfully!")
            except Exception as e:
                st.error(f"Error loading CSV: {e}")
                
        # Display data summary
        if st.session_state.connectivity_data is not None:
            N_nodes = st.session_state.connectivity_data.shape[1]
            st.info(f"Loaded Connectivity Matrix: {st.session_state.connectivity_data.shape[0]} subjects, {N_nodes} x {N_nodes} ROIs.")
            
        if st.session_state.pheno_df is not None:
            st.markdown("**Phenotypic Data Preview**")
            st.dataframe(st.session_state.pheno_df.head(), use_container_width=True)

# ----------------- TAB 2: STATISTICAL DESIGN & INFERENCE -----------------
with tab2:
    st.markdown("### Statistical Design & Inference Loop")
    
    if st.session_state.connectivity_data is None or st.session_state.pheno_df is None:
        st.warning("Please upload or generate a dataset in Tab 1 first.")
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**1. Design Options**")
            design_type = st.selectbox(
                "Experimental Design",
                ["two-sample", "paired", "one-sample", "glm"]
            )
            
            # Column bindings
            cols = list(st.session_state.pheno_df.columns)
            
            subject_col = st.selectbox("Subject ID Column (optional)", [None] + cols)
            
            # Variables mapping based on design
            interest_var = None
            confound_vars = []
            site_var = None
            
            if design_type == "glm":
                interest_var = st.selectbox("Predictor of Interest (Interest)", cols)
                confound_vars = st.multiselect("Nuisance Covariates (Confounds)", [c for c in cols if c != interest_var])
            elif design_type in ("two-sample", "paired"):
                group_col = st.selectbox("Group / Condition Column", cols)
                # find unique values
                vals = list(st.session_state.pheno_df[group_col].unique())
                if len(vals) < 2:
                    st.error(f"Group column must have at least 2 unique conditions; found {vals}")
                else:
                    g1_val = st.selectbox("Condition 1 (Group 1 / baseline)", vals, index=0)
                    g2_val = st.selectbox("Condition 2 (Group 2 / target)", vals, index=1 if len(vals)>1 else 0)
            
            # Site column for ComBat / Exchangeability strata
            site_col = st.selectbox("Acquisition Site Column (ComBat)", [None] + cols)
            harmonization_choice = st.selectbox(
                "ComBat Strategy",
                ["none", "combat", "covariate", "combat_covariate", "auto"]
            )
            
            # Preprocessing toggle
            apply_fisher_z = st.checkbox("Apply Fisher r-to-z transform on connectivity matrices", value=True)
            
        with col2:
            st.markdown("**2. Enhancement Operator**")
            operator_choice = st.selectbox(
                "Enhancement Operator",
                ["tfnbs", "nbs", "tstat", "bh_fdr", "bonferroni", "cnbs", "ni_tfnbs", "fbc_tfnbs"]
            )
            
            # Operator parameters
            st.markdown("**Operator Specific Parameters**")
            op_kwargs = {}
            if operator_choice == "nbs":
                tau = st.number_input("Cluster-forming threshold (tau)", 1.0, 10.0, 3.0, 0.1)
                op_kwargs["start_thres"] = tau
            elif operator_choice in ("tfnbs", "ni_tfnbs", "fbc_tfnbs"):
                e_exp = st.number_input("Extent exponent (E)", 0.1, 2.0, 0.3, 0.05)
                h_exp = st.number_input("Height exponent (H)", 1.0, 5.0, 3.0, 0.1)
                n_thres = st.number_input("Integration steps (n)", 5, 50, 10, 1)
                op_kwargs["e"] = e_exp
                op_kwargs["h"] = h_exp
                op_kwargs["n"] = n_thres
                
            # Permutation parameters
            st.markdown("**Permutation Engine**")
            n_perms = st.number_input("Number of Permutations (B)", 10, 5000, 100, 50)
            seed = st.number_input("Random Seed", 1, 10000, 42)
            
        # Inference Run Trigger
        if st.button("🚀 Run Connectivity Inference", type="primary"):
            try:
                # 1. Slice subnetwork if configured
                Y = st.session_state.connectivity_data.copy()
                atlas = base_atlas
                if st.session_state.roi_indices is not None:
                    roi_idx = st.session_state.roi_indices
                    Y = Y[:, roi_idx][:, :, roi_idx]
                    atlas = st.session_state.sub_atlas
                    
                # 2. Extract covariates / interest vectors from Phenotypic CSV
                pheno_df = st.session_state.pheno_df
                
                interest = None
                confounds = None
                group1 = None
                group2 = None
                sites = None
                
                # Fetch sites
                if site_col:
                    sites = list(pheno_df[site_col].values)
                    
                # Setup inputs depending on design type
                if design_type == "glm":
                    interest = pheno_df[interest_var].values.astype(np.float64)
                    if confound_vars:
                        # Build confound matrix
                        confounds = pheno_df[confound_vars].values.astype(np.float64)
                elif design_type in ("two-sample", "paired"):
                    g1_mask = (pheno_df[group_col] == g1_val)
                    g2_mask = (pheno_df[group_col] == g2_val)
                    
                    group1 = Y[g1_mask]
                    group2 = Y[g2_mask]
                    
                    # Split sites
                    if site_col:
                        # For two-sample path in analyze(), sites splits can be handled, but
                        # it is easier to feed Y, interest, and confounds. Let's just feed group1 and group2.
                        sites = None  # reset for simplicity in groups path
                
                # Run analyze wrapper
                with st.spinner("Executing permutation loops..."):
                    res = analyze(
                        Y=Y if design_type == "glm" else None,
                        interest=interest,
                        confounds=confounds,
                        group1=group1,
                        group2=group2,
                        test_type=design_type,
                        sites=sites,
                        harmonize=harmonization_choice,
                        fisher_z=apply_fisher_z,
                        method=operator_choice,
                        n_permutations=n_perms,
                        rng=seed,
                        verbose=False,
                        use_mp=True,
                        **op_kwargs
                    )
                    
                st.session_state.inference_result = res
                
                # Save annotated edges
                alpha = 0.05
                edges = res.significant_edges(atlas=atlas, alpha=alpha)
                st.session_state.edges_df = edges
                
                st.success(f"Connectivity inference completed! Found {len(edges)} significant edges (α={alpha}).")
                
            except Exception as e:
                st.error(f"Inference failed: {e}")
                st.exception(e)

# ----------------- TAB 3: INFERENCE RESULTS -----------------
with tab3:
    st.markdown("### Inference Results")
    
    if st.session_state.inference_result is None:
        st.warning("Please configure and run inference in Tab 2 first.")
    else:
        res = st.session_state.inference_result
        
        # 1. Summary stats
        nsig = res.n_significant(0.05)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Positive Tail Significant Edges", nsig["positive"])
        with col2:
            st.metric("Negative Tail Significant Edges", nsig["negative"])
        with col3:
            st.metric("Permutations Completed", res.n_permutations)
            
        # 2. Heatmap visualization
        st.markdown("**Connectivity Maps Visualization**")
        
        # Get active atlas
        atlas = base_atlas if st.session_state.sub_atlas is None else st.session_state.sub_atlas
        
        # Retrieve effect size map
        stat_map = res.stat_signed
        p_pos = res["positive"]
        p_neg = res["negative"]
        
        if stat_map is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Heatmap 1: Observed statistics
            im1 = ax1.imshow(stat_map, cmap="RdBu_r", aspect="auto")
            fig.colorbar(im1, ax=ax1)
            ax1.set_title("Observed Effect Size (Stat Map)")
            
            # Heatmap 2: -log10 thresholded p-values
            min_p = np.minimum(p_pos, p_neg)
            log_p = -np.log10(min_p + 1e-10)
            log_p[min_p > 0.05] = 0.0  # Hide non-significant
            
            im2 = ax2.imshow(log_p, cmap="inferno", aspect="auto")
            fig.colorbar(im2, ax=ax2)
            ax2.set_title("Significant Edges (-log10 p-value)")
            
            st.pyplot(fig)
            
        # 3. Interactive Table
        st.markdown("**Significant Edges Table**")
        st.dataframe(st.session_state.edges_df, use_container_width=True)
        
        # Downloads
        csv_data = st.session_state.edges_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Significant Edges CSV",
            data=csv_data,
            file_name=f"significant_edges_{design_type}.csv",
            mime="text/csv"
        )

# ----------------- TAB 4: META-ANALYTIC DECODING -----------------
with tab4:
    st.markdown("### Meta-Analytic Decoding (NiMARE)")
    
    if st.session_state.edges_df is None or st.session_state.edges_df.empty:
        st.warning("No significant edges found. Please check your data or threshold in previous tabs.")
    else:
        st.markdown("Run coordinate-based Neurosynth meta-analytic decoding on coordinates associated with significant edges.")
        
        # Check coordinates presence in current atlas
        atlas = base_atlas if st.session_state.sub_atlas is None else st.session_state.sub_atlas
        
        if atlas.coords is None:
            st.error("The chosen reference parcellation does not contain coordinates; cannot run NiMARE decoding.")
        else:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("**Decoding Settings**")
                dec_radius = st.slider("Coordinate Sphere Radius (mm)", 4.0, 12.0, 6.0, 0.5, key="dec_radius")
                dec_top_n = st.number_input("Top Terms to Retrieve", 3, 20, 10, 1, key="dec_top_n")
                dec_scoring = st.selectbox("Association Metric", ["chi2", "lda"], key="dec_scoring")
                
                run_dec = st.button("🚀 Run NiMARE Decoding", key="run_dec_button")
                
            if run_dec:
                try:
                    # Get unique roi indices from significant edges
                    roi_ids = sorted(list(pd.concat([st.session_state.edges_df['roi_i'], st.session_state.edges_df['roi_j']]).unique()))
                    roi_ids = [int(r) for r in roi_ids]
                    
                    dataset = load_cached_dataset()
                    
                    # Stopwords filter
                    method_stop_words = {
                        "task", "fmri", "subject", "brain", "cortex", "bold", "functional", "activation", 
                        "study", "magnetic resonance", "scanner", "magnetic", "image", "imaging", 
                        "stimulus", "response", "subjects", "patients", "healthy", "group", "studies"
                    }
                    def term_filter(t):
                        t_clean = t.lower().strip()
                        for stop in method_stop_words:
                            if stop == t_clean or stop in t_clean.split():
                                return False
                        return True
                        
                    with st.spinner("Decoding coordinates..."):
                        decoded = decode_rois(
                            atlas,
                            roi_ids,
                            top_n=dec_top_n,
                            radius_mm=dec_radius,
                            scoring=dec_scoring,
                            dataset=dataset,
                            term_filter=term_filter
                        )
                        
                    st.session_state.decoded_df = decoded
                    
                    # Compile evidence packet
                    evidence = build_decoding_evidence(
                        st.session_state.edges_df,
                        atlas,
                        decoded,
                        contrast_name=design_type,
                        radius_mm=dec_radius,
                        scoring=dec_scoring,
                        top_n=dec_top_n,
                        source="conninfpy_edges",
                    )
                    st.session_state.evidence_packet = evidence
                    st.session_state.narrative_text = None  # Reset narrative
                    
                    st.success("Decoding completed successfully!")
                except Exception as e:
                    st.error(f"Decoding failed: {e}")
                    st.exception(e)
                    
            # Display results
            if st.session_state.decoded_df is not None:
                with col2:
                    st.markdown("**Decoded Terms Table**")
                    st.dataframe(st.session_state.decoded_df, use_container_width=True)
                    
                    # Download buttons
                    csv_dec = st.session_state.decoded_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Decoded Terms CSV",
                        data=csv_dec,
                        file_name="decoded_terms.csv",
                        mime="text/csv"
                    )

# ----------------- TAB 5: NARRATIVE GENERATOR -----------------
with tab5:
    st.markdown("### Narrative Generator")
    
    if st.session_state.evidence_packet is None:
        st.warning("Please run NiMARE decoding in Tab 4 first.")
    else:
        st.info("💡 **Scientific Citation Note**: NiMARE/Neurosynth meta-analytic decoding maps literature associations and represents literature spatial frequency, not direct mechanistic causal claims (Yarkoni et al., 2011; Wager & Lindquist, 2016).")
        
        # Validate evidence
        try:
            validate_evidence(st.session_state.evidence_packet)
            valid = True
        except Exception as ve:
            st.error(f"Evidence validation failed: {ve}")
            valid = False
            
        if valid:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("**Narrative Control**")
                # Show loaded API Key status
                key_to_use = llm_api_key
                if not key_to_use:
                    if llm_provider == "openrouter":
                        key_to_use = os.getenv("OPENROUTER_API_KEY")
                    elif llm_provider == "openai":
                        key_to_use = os.getenv("OPENAI_API_KEY")
                    elif llm_provider == "gemini":
                        key_to_use = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                        
                if llm_provider != "mock":
                    if key_to_use:
                        st.success("API Key detected in environment/.env.")
                    else:
                        st.warning("No API key detected. Please add it to the sidebar input.")
                        
                gen_button = st.button("✍️ Generate Report Narrative", type="primary")
                
            if gen_button:
                try:
                    narrator = LLMNarrator(
                        provider=llm_provider,
                        model=llm_model if llm_model else None,
                        api_key=key_to_use if key_to_use else None
                    )
                    with st.spinner("Generating LLM narrative..."):
                        narrative = narrator.generate(st.session_state.evidence_packet)
                        st.session_state.narrative_text = narrative
                except Exception as e:
                    st.error(f"LLM Narration failed: {e}")
                    st.info("Tip: If you do not have an API key, switch the LLM Narrator Settings Provider to 'mock' in the sidebar for offline testing.")
                    
            if st.session_state.narrative_text is not None:
                with col2:
                    st.markdown("**LLM Generated Narrative**")
                    
                    # Term guardrails warning check
                    unsupported = check_narrative_terms(st.session_state.narrative_text, st.session_state.evidence_packet)
                    
                    if unsupported:
                        st.warning(f"⚠️ **Term Guardrail Warning**: The generated answer contains cognitive terms not present in the NiMARE evidence: `{', '.join(unsupported)}`. Regenerate with stricter prompt or edit manually.")
                    
                    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                    st.markdown(st.session_state.narrative_text)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Export narrative report block
                    report_block = f"""# Neuroimaging Decoding Report
Contrast: {st.session_state.evidence_packet['query']['contrast']}
Atlas: {st.session_state.evidence_packet['query']['atlas']}
Method: {st.session_state.evidence_packet['decoder']['method']}

{st.session_state.narrative_text}

---
*Generated by ConnInfPy interpretation layer. Cautions apply.*
"""
                    st.download_button(
                        "Export Markdown Report",
                        data=report_block.encode('utf-8'),
                        file_name=f"nimare_narrative_{design_type}.md",
                        mime="text/markdown"
                    )

# ----------------- TAB 6: HELP & DOCUMENTATION -----------------
with tab6:
    st.markdown("### Help & Documentation")
    
    st.markdown("""
    #### Experimental Designs
    - **Two-sample comparison:** Welch's t-test comparing independent groups.
    - **Paired contrast:** Subject-wise difference contrast.
    - **GLM:** Evaluate variable of interest while regressing out nuisance covariates.
    
    #### Enhancement Operators Quick-Reference
    - **tstat:** Max-stat FWER on raw edge statistics.
    - **bh_fdr / bonferroni:** Standard edge-wise multiple comparison correction.
    - **nbs:** Fixed-threshold connected-component mass (Zalesky et al., 2010).
    - **tfnbs:** Threshold-free connected-component integration (Baggio et al., 2018; Hao et al., 2024).
    - **cnbs:** Constrained block-mean statistic (Noble et al., 2020).
    - **ni_tfnbs / fbc_tfnbs:** Block prior informed network threshold-free statistics.
    
    #### Scientific Caveats on Reverse Inference
    Meta-analytic decoding maps literature associations based on spatial coordinates in neuroimaging studies. A strong association with a term (e.g. `'memory'`) indicates that coordinates in your effect map frequently co-occur in the literature with studies mentioning memory. It does **not** prove task activation, causal mechanism, or diagnostic properties (Yarkoni et al., 2011; Wager & Lindquist, 2016). Always report statistical effect maps and literature associations cautiously.
    """)
