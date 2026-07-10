import streamlit as st
from apps.utils.helpers import render_help

def render_workspace_docs_view():
    col_t, col_h = st.columns([0.8, 0.2])
    with col_t:
        st.markdown("### Help & Documentation")
    with col_h:
        render_help("workspace_documentation")
    
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
