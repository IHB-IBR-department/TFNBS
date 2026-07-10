import pandas as pd
import streamlit as st
from conninfpy.decode import decode_rois
from conninfpy._decode_cache import fetch_neurosynth_dataset
from conninfpy.interpret.evidence import build_decoding_evidence
from apps.utils.helpers import active_analysis_atlas, atlas_has_coords, current_contrast_name, render_help, result_is_stale

def render_meta_decoding_view(base_atlas):
    col_t, col_h = st.columns([0.8, 0.2])
    with col_t:
        st.markdown("### Meta-Analytic Decoding (NiMARE)")
    with col_h:
        render_help("meta_analytic_decoding")
    
    is_stale = result_is_stale()

    if is_stale:
        st.error("❌ **Stale Results:** The dataset configuration has changed. You must re-run inference in Tab 2 before running decoding.")
    elif st.session_state.edges_df is None or st.session_state.edges_df.empty:
        st.warning("No significant edges found. Please check your data or threshold in previous tabs.")
    else:
        st.markdown("Run coordinate-based Neurosynth meta-analytic decoding on coordinates associated with significant edges.")
        
        # Check coordinates presence in current atlas
        atlas = active_analysis_atlas(base_atlas)
        
        if atlas is None:
            st.info("Atlas metadata is disabled. NiMARE decoding requires ROI coordinates from a bundled or custom atlas.")
        elif not atlas_has_coords(atlas):
            st.info("The active atlas does not contain complete x/y/z coordinates, so NiMARE decoding is unavailable.")
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
                    
                    with st.spinner("⏳ Loading/Downloading Neurosynth Dataset... (First-time run downloads ~150MB of database files; this can take 2-5 minutes depending on network speed. Please wait.)"):
                        dataset = fetch_neurosynth_dataset()
                    
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
                        
                    with st.spinner("🧠 Performing Neurosynth decoding analysis... (This matches spatial coordinate spheres and scores literature associations; this takes about 1-2 minutes.)"):
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
                    contrast_name = current_contrast_name()
                    evidence = build_decoding_evidence(
                        st.session_state.edges_df,
                        atlas,
                        decoded,
                        contrast_name=contrast_name,
                        radius_mm=dec_radius,
                        scoring=dec_scoring,
                        top_n=dec_top_n,
                        source="conninfpy_edges",
                    )
                    st.session_state.evidence_packet = evidence
                    st.session_state.narrative_text = None  # Reset narrative
                    
                    # Calculate summary and score
                    from conninfpy.interpret.evidence import summarize_decoded_terms, score_decoding_evidence
                    summary = summarize_decoded_terms(decoded, st.session_state.edges_df, atlas)
                    score = score_decoding_evidence(summary)
                    st.session_state.decoding_summary = summary
                    st.session_state.decoding_score = score
                    
                    st.success("Decoding completed successfully!")
                except Exception as e:
                    st.error(f"Decoding failed: {e}")
                    st.exception(e)
                    
            # Display results
            if st.session_state.decoded_df is not None:
                summary = st.session_state.get("decoding_summary")
                score = st.session_state.get("decoding_score")
                if summary is None or score is None:
                    from conninfpy.interpret.evidence import summarize_decoded_terms, score_decoding_evidence
                    summary = summarize_decoded_terms(st.session_state.decoded_df, st.session_state.edges_df, atlas)
                    score = score_decoding_evidence(summary)
                    st.session_state.decoding_summary = summary
                    st.session_state.decoding_score = score
                    
                quality = score["evidence_quality"]
                explanation = score["explanation"]
                report_sentence = score["report_sentence"]
                
                with col2:
                    st.markdown("#### 📝 Decoded Summary")
                    if quality == "informative":
                        st.success(f"**Evidence Quality:** `informative`\n\n{explanation}")
                    elif quality == "weak":
                        st.info(f"**Evidence Quality:** `weak`\n\n{explanation}")
                    elif quality == "generic":
                        st.warning(f"**Evidence Quality:** `generic`\n\n{explanation}")
                    else:
                        st.error(f"**Evidence Quality:** `inconclusive`\n\n{explanation}")
                        
                    st.markdown(f"**Suggested Interpretation:**\n> {report_sentence}")
                    
                    # Caveats
                    with st.expander("⚠️ Scientific Caveats on Reverse Inference", expanded=False):
                        for caveat in st.session_state.evidence_packet.get("caveats", []):
                            st.markdown(f"- {caveat}")
                            
                    # High-burden ROIs
                    st.markdown("#### 🎯 High-Burden Endpoint ROIs")
                    top_rois_df = pd.DataFrame(summary["top_endpoint_rois"])
                    st.dataframe(top_rois_df, use_container_width=True)
                    
                    # Filtered Term Summary
                    st.markdown("#### 🔍 Filtered Term Summary")
                    agg_terms = summary["aggregated_terms"]
                    if agg_terms:
                        table_data = []
                        for t in agg_terms:
                            table_data.append({
                                "Term": t["term"],
                                "Weighted Count": t["weighted_count"],
                                "ROI Count": t["roi_count"],
                                "Networks": ", ".join(t["networks"]),
                                "Best Rank": t["best_rank"],
                                "Max Score": t["max_score"]
                            })
                        agg_df = pd.DataFrame(table_data)
                        st.dataframe(agg_df, use_container_width=True)
                    else:
                        st.info("No terms remained after stop-word filtering.")
                        table_data = []
                        
                    # Raw expander
                    with st.expander("📋 Raw Decoded Terms (Audit)", expanded=False):
                        st.dataframe(st.session_state.decoded_df, use_container_width=True)
                        
                    # Downloads
                    st.markdown("#### 💾 Downloads")
                    col_d1, col_d2, col_d3 = st.columns(3)
                    
                    csv_dec = st.session_state.decoded_df.to_csv(index=False).encode('utf-8')
                    with col_d1:
                        st.download_button(
                            "Raw Terms CSV",
                            data=csv_dec,
                            file_name="decoded_terms_raw.csv",
                            mime="text/csv"
                        )
                        
                    if agg_terms:
                        csv_filt = agg_df.to_csv(index=False).encode('utf-8')
                        with col_d2:
                            st.download_button(
                                "Filtered Terms CSV",
                                data=csv_filt,
                                file_name="decoded_terms_filtered.csv",
                                mime="text/csv"
                            )
                            
                    # Evidence JSON
                    import json
                    evidence_json = json.dumps(st.session_state.evidence_packet, indent=2).encode('utf-8')
                    
                    # Markdown Report
                    report_md = f"""# NiMARE/Neurosynth Decoding Report

- **Contrast:** {st.session_state.evidence_packet['query']['contrast']}
- **Atlas:** {st.session_state.evidence_packet['query']['atlas']}
- **Evidence Quality:** {quality}
- **Interpretation:** {report_sentence}

## High-Burden ROIs
{pd.DataFrame(summary['top_endpoint_rois']).to_markdown(index=False) if summary['top_endpoint_rois'] else "None"}

## Top Filtered Terms
{pd.DataFrame(table_data).to_markdown(index=False) if agg_terms else "None"}
""".encode('utf-8')

                    with col_d3:
                        st.download_button(
                            "Report Markdown",
                            data=report_md,
                            file_name="decoding_report.md",
                            mime="text/markdown"
                        )
