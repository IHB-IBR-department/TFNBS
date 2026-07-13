import pandas as pd
import streamlit as st
import threading
import traceback
from conninfpy.decode import decode_rois
from conninfpy._decode_cache import fetch_neurosynth_dataset
from conninfpy.interpret.evidence import build_decoding_evidence
from apps.utils.helpers import active_analysis_atlas, atlas_has_coords, current_contrast_name, render_help, result_is_stale

class DecodingTask:
    def __init__(self):
        self.status = "idle"  # idle, running, success, failed
        self.progress_message = ""  # e.g., "Downloading database..."
        self.result = None
        self.evidence = None
        self.summary = None
        self.score = None
        self.error = None

def _run_decoding_background(
    task, atlas, roi_ids, dec_database, dec_strategy, dec_radius, dec_top_n, dec_scoring, edges_df, contrast_name
):
    try:
        # Check cache existence to set appropriate progress message
        import os
        from conninfpy._decode_cache import get_cache_dir
        c_dir = get_cache_dir()
        pkl_name = "neuroquery_dataset.pkl" if dec_database.lower() == "neuroquery" else "neurosynth_dataset.pkl"
        pkl_path = c_dir / pkl_name
        
        raw_dir = c_dir / ("raw_nq" if dec_database.lower() == "neuroquery" else "raw")
        raw_files_available = raw_dir.exists() and any(
            path.is_file() for path in raw_dir.rglob("*")
        )
        if pkl_path.exists():
            task.progress_message = f"📂 Loading cached {dec_database} database from disk..."
        elif raw_files_available:
            task.progress_message = (
                f"🧱 Building the compact {dec_database} cache from downloaded files "
                "(no new download required)..."
            )
        else:
            task.progress_message = (
                f"📥 Downloading and building the {dec_database} cache "
                "(first run can take several minutes)..."
            )

        # Fetch selected dataset
        if dec_database.lower() == "neuroquery":
            from conninfpy._decode_cache import fetch_neuroquery_dataset
            dataset = fetch_neuroquery_dataset()
        else:
            from conninfpy._decode_cache import fetch_neurosynth_dataset
            dataset = fetch_neurosynth_dataset()
            
        # Stopwords filter
        task.progress_message = "🧹 Preparing terms and method stopwords filters..."
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

        if dec_strategy == "Combined Region Decoding":
            task.progress_message = f"🧠 Running Combined Region Decoding on {dec_database}..."
            from conninfpy.decode import decode_combined_rois
            combined = decode_combined_rois(
                atlas,
                roi_ids,
                top_n=dec_top_n,
                radius_mm=dec_radius,
                dataset=dataset,
                dataset_name=dec_database.lower(),
                term_filter=term_filter
            )
            decoded = combined.assign(
                roi_id=-1,
                roi_name="Combined Pattern",
                network="All Networks"
            )
        else:
            task.progress_message = f"🧠 Running Discrete ROI Overlap Decoding on {dec_database}..."
            decoded = decode_rois(
                atlas,
                roi_ids,
                top_n=dec_top_n,
                radius_mm=dec_radius,
                scoring=dec_scoring,
                dataset=dataset,
                term_filter=term_filter
            )

        task.progress_message = "📊 Compiling statistical evidence packets..."
        from conninfpy.interpret.evidence import default_term_filter, build_decoding_evidence
        decoded_filtered = decoded[decoded["term"].apply(default_term_filter)]
        evidence = build_decoding_evidence(
            edges_df,
            atlas,
            decoded_filtered,
            contrast_name=contrast_name,
            radius_mm=dec_radius,
            scoring=dec_scoring,
            top_n=dec_top_n,
            source="conninfpy_edges",
            backend="NiMARE",
            dataset_name=dec_database,
            decoder_method="CombinedDecoder" if dec_strategy == "Combined Region Decoding" else "NeurosynthDecoder"
        )
        
        task.progress_message = "✍️ Scoring evidence and generating report..."
        from conninfpy.interpret.evidence import summarize_decoded_terms, score_decoding_evidence
        summary = summarize_decoded_terms(decoded, edges_df, atlas)
        score = score_decoding_evidence(summary)

        task.result = decoded
        task.evidence = evidence
        task.summary = summary
        task.score = score
        task.status = "success"
        task.progress_message = "✅ Completed!"
    except Exception as e:
        task.error = f"{e}\n{traceback.format_exc()}"
        task.status = "failed"
        task.progress_message = "❌ Failed"

def render_meta_decoding_view(base_atlas, *, decoding_enabled: bool = True):
    if "decoding_task" not in st.session_state:
        st.session_state.decoding_task = DecodingTask()
        
    col_t, col_h = st.columns([0.8, 0.2])
    with col_t:
        st.markdown("### Meta-Analytic Decoding (NiMARE)")
    with col_h:
        render_help("meta_analytic_decoding")

    if not decoding_enabled:
        st.info(
            "Meta-analytic decoding is available in the offline ConnInfPy version, "
            "which includes the optional NiMARE dependency and local reference datasets."
        )
        return
    
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
                dec_database = st.selectbox("Database / Source", ["NeuroQuery", "Neurosynth"], key="dec_database")
                dec_strategy = st.selectbox("Strategy / Unit of Decoding", ["Combined Region Decoding", "Discrete ROI Overlap"], key="dec_strategy")
                dec_radius = st.slider("Coordinate Sphere Radius (mm)", 4.0, 12.0, 6.0, 0.5, key="dec_radius")
                dec_top_n = st.number_input("Top Terms to Retrieve", 3, 20, 10, 1, key="dec_top_n")
                
                dec_scoring = "chi2"
                if dec_strategy == "Discrete ROI Overlap" and dec_database == "Neurosynth":
                    dec_scoring = st.selectbox("Association Metric", ["chi2", "lda"], key="dec_scoring")
                
                # Get unique roi indices from significant edges
                roi_ids = sorted(list(pd.concat([st.session_state.edges_df['roi_i'], st.session_state.edges_df['roi_j']]).unique()))
                roi_ids = [int(r) for r in roi_ids]
                
                task = st.session_state.decoding_task
                if task.status == "idle":
                    run_dec = st.button("🚀 Run NiMARE Decoding", key="run_dec_button")
                    if run_dec:
                        task.status = "running"
                        task.result = None
                        task.evidence = None
                        task.summary = None
                        task.score = None
                        task.error = None
                        
                        st.session_state.decoded_df = None
                        st.session_state.evidence_packet = None
                        st.session_state.decoding_summary = None
                        st.session_state.decoding_score = None
                        st.session_state.narrative_text = None
                        
                        # Start background thread
                        t = threading.Thread(
                            target=_run_decoding_background,
                            args=(
                                task,
                                atlas,
                                roi_ids,
                                dec_database,
                                dec_strategy,
                                dec_radius,
                                dec_top_n,
                                dec_scoring,
                                st.session_state.edges_df,
                                current_contrast_name()
                            )
                        )
                        t.daemon = True
                        t.start()
                        st.rerun()
                elif task.status == "running":
                    def render_running_status():
                        """Poll the decoder thread without restarting its work."""
                        current_task = st.session_state.decoding_task
                        if current_task.status != "running":
                            if hasattr(st, "fragment"):
                                st.rerun(scope="app")
                            st.rerun()

                        st.info(f"⏳ **Status:** {current_task.progress_message}")
                        st.caption("First-time cache builds can take several minutes. Cached runs are much faster.")
                        st.caption("This status checks automatically every 2 seconds; decoding continues in the background.")

                    if hasattr(st, "fragment"):
                        st.fragment(run_every=2.0)(render_running_status)()
                    else:  # pragma: no cover - compatibility with Streamlit < 1.37
                        render_running_status()
                        if st.button("Refresh Status", key="refresh_dec_status"):
                            st.rerun()
                else:
                    if st.button("🚀 Run New Decoding", key="run_dec_new_button"):
                        task.status = "idle"
                        st.rerun()
                        
            # Handle finished background tasks
            task = st.session_state.decoding_task
            if task.status == "success":
                if st.session_state.decoded_df is None:
                    st.session_state.decoded_df = task.result
                    st.session_state.evidence_packet = task.evidence
                    st.session_state.decoding_summary = task.summary
                    st.session_state.decoding_score = task.score
                    st.success("🎉 Decoding completed successfully!")
            elif task.status == "failed":
                st.error("❌ Decoding failed!")
                st.text_area("Error Traceback", value=task.error or "Unknown error", height=150)
                if st.button("Reset & Try Again", key="reset_failed_task"):
                    task.status = "idle"
                    st.rerun()
                    
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
                    
                    def _to_markdown_simple(df: pd.DataFrame) -> str:
                        if df.empty:
                            return ""
                        headers = [str(col) for col in df.columns]
                        lines = []
                        lines.append("| " + " | ".join(headers) + " |")
                        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                        for _, row in df.iterrows():
                            row_str = "| " + " | ".join(str(val) for val in row) + " |"
                            lines.append(row_str)
                        return "\n".join(lines)
                    
                    # Markdown Report
                    report_md = f"""# NiMARE/Neurosynth Decoding Report

- **Contrast:** {st.session_state.evidence_packet['query']['contrast']}
- **Atlas:** {st.session_state.evidence_packet['query']['atlas']}
- **Evidence Quality:** {quality}
- **Interpretation:** {report_sentence}

## High-Burden ROIs
{_to_markdown_simple(pd.DataFrame(summary['top_endpoint_rois'])) if summary['top_endpoint_rois'] else "None"}

## Top Filtered Terms
{_to_markdown_simple(pd.DataFrame(table_data)) if agg_terms else "None"}
""".encode('utf-8')

                    with col_d3:
                        st.download_button(
                            "Report Markdown",
                            data=report_md,
                            file_name="decoding_report.md",
                            mime="text/markdown"
                        )
            elif task.status == "running":
                with col2:
                    st.markdown("#### 📝 Decoded Summary")
                    st.info("⏳ **NiMARE Decoding is running in the background...**")
                    st.markdown(f"""
                    **Current Step:** {task.progress_message}
                    
                    *   You can switch to other tabs or browse around the workspace.
                    *   When you want to check if the results are ready, click the **Refresh Status** button in the left column.
                    """)
