import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from io import BytesIO

from conninfpy.plot import plot_connectome_graph
from apps.utils.helpers import (
    active_analysis_atlas,
    atlas_has_coords,
    atlas_has_networks,
    current_contrast_name,
    effect_direction_labels,
    result_is_stale,
    safe_filename_part,
    render_help,
)

def _render_summary_metrics(res, edges_df, method_name, direction_labels):
    st.markdown(f"##### 📊 {method_name} Summary")
    nsig = res.n_significant(0.05)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(direction_labels["positive"], nsig["positive"], help="Edges significant in the positive statistical tail.")
    with col2:
        st.metric(direction_labels["negative"], nsig["negative"], help="Edges significant in the negative statistical tail.")
    with col3:
        st.metric("Permutations Completed", res.n_permutations)

def _render_ground_truth_metrics(edges_df, effect_mask, method_name):
    if effect_mask is not None:
        st.markdown(f"##### 🎯 Ground Truth Comparison ({method_name})")
        true_mask = (effect_mask > 0)
        true_mask_triu = np.triu(true_mask, 1)
        
        n_nodes = true_mask.shape[0]
        possible_edges = int(n_nodes * (n_nodes - 1) / 2)
        
        n_true_pos_edges = int(np.sum(true_mask_triu))
        n_true_neg_edges = possible_edges - n_true_pos_edges
        
        detected_edges_set = set()
        if edges_df is not None and not edges_df.empty:
            for _, row in edges_df.iterrows():
                i = int(row['roi_i'])
                j = int(row['roi_j'])
                u, v = min(i, j), max(i, j)
                detected_edges_set.add((u, v))
        
        tp_count = 0
        fp_count = 0
        for u, v in detected_edges_set:
            if true_mask_triu[u, v]:
                tp_count += 1
            else:
                fp_count += 1
        
        fn_count = n_true_pos_edges - tp_count
        
        tp_rate = (tp_count / n_true_pos_edges) if n_true_pos_edges > 0 else 0.0
        fp_rate = (fp_count / n_true_neg_edges) if n_true_neg_edges > 0 else 0.0
        fn_rate = (fn_count / n_true_pos_edges) if n_true_pos_edges > 0 else 0.0
        
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric("Sensitivity / Power (TPR)", f"{tp_rate*100:.1f}%", help=f"True Positives: {tp_count} detected out of {n_true_pos_edges} active ground-truth edges")
        with m_col2:
            st.metric("Type I Error Rate (FPR)", f"{fp_rate*100:.1f}%", help=f"False Positives: {fp_count} detected out of {n_true_neg_edges} null ground-truth edges")
        with m_col3:
            st.metric("Type II Error Rate (FNR)", f"{fn_rate*100:.1f}%", help=f"False Negatives: {fn_count} missed out of {n_true_pos_edges} active ground-truth edges")
        
        if len(detected_edges_set) > 0:
            fdr = fp_count / len(detected_edges_set)
            st.markdown(f"**False Discovery Rate (FDR):** `{fdr*100:.1f}%` (proportion of false positives among all detected edges)")

def _render_heatmaps(res, base_atlas, method_name):
    st.markdown(f"##### 🗺️ Connectivity Maps ({method_name})")
    
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
        plt.close(fig)

def _active_atlas(base_atlas):
    return active_analysis_atlas(base_atlas)



def _filter_edges_for_tail(edges_df, atlas, tail, alpha, top_n, rank_by, selected_networks, network_filter_mode):
    if edges_df is None or edges_df.empty:
        return edges_df.copy() if edges_df is not None else pd.DataFrame()

    p_col = "p_positive" if tail == "positive" else "p_negative"
    df = edges_df.copy()
    df = df[df[p_col] <= alpha]

    if selected_networks and atlas_has_networks(atlas):
        networks = np.asarray(atlas.networks, dtype=object)
        roi_i_net = df["roi_i"].astype(int).map(lambda idx: networks[idx])
        roi_j_net = df["roi_j"].astype(int).map(lambda idx: networks[idx])
        selected = set(selected_networks)
        if network_filter_mode == "Both endpoints":
            keep = roi_i_net.isin(selected) & roi_j_net.isin(selected)
        else:
            keep = roi_i_net.isin(selected) | roi_j_net.isin(selected)
        df = df[keep]

    if df.empty:
        return df

    if rank_by == "|t|":
        df = df.assign(_rank_value=df["t_signed"].abs())
        df = df.sort_values("_rank_value", ascending=False)
    else:
        df = df.sort_values(p_col, ascending=True)

    return df.head(int(top_n)).drop(columns=[c for c in ["_rank_value"] if c in df.columns])

def _download_figure_button(fig, label, filename, key):
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=220, bbox_inches="tight", facecolor="white")
    st.download_button(
        label,
        data=buffer.getvalue(),
        file_name=filename,
        mime="image/png",
        key=key,
    )

def _render_brain_connectome_graphs(edges_df, base_atlas, method_name, direction_labels):
    atlas = _active_atlas(base_atlas)
    st.markdown(f"##### 🧠 Brain-Space Effect Graphs ({method_name})")

    if atlas is None:
        st.info("Brain graph plotting requires atlas metadata with ROI coordinates.")
        return
    if not atlas_has_coords(atlas):
        st.info("Brain graph plotting requires complete atlas x/y/z coordinates.")
        return
    if edges_df is None or edges_df.empty:
        st.info("No significant edges available to plot.")
        return

    network_options = list(dict.fromkeys(str(n) for n in atlas.networks)) if atlas_has_networks(atlas) else []
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1, 1, 1, 1])
    with ctrl1:
        alpha = st.slider("Graph p-value cutoff", 0.001, 0.100, 0.050, 0.001, key=f"{method_name}_graph_alpha")
        top_n = st.slider("Top edges per direction", 10, 500, 100, 10, key=f"{method_name}_graph_top_n")
    with ctrl2:
        display_mode = st.selectbox("Brain view", ["z", "ortho", "x", "y"], index=0, key=f"{method_name}_graph_display")
        weight_by = st.selectbox("Edge weight", ["-log10(p)", "|t|"], index=0, key=f"{method_name}_graph_weight")
    with ctrl3:
        color_options = ["Network", "Degree"]
        if getattr(atlas, "hemisphere", None) is not None:
            color_options.insert(1, "Hemisphere")
        color_by = st.selectbox("Node color", color_options, index=0, key=f"{method_name}_graph_color")
        rank_by = st.selectbox("Top edge ranking", ["p-value", "|t|"], index=0, key=f"{method_name}_graph_rank")
    with ctrl4:
        selected_networks = []
        network_filter_mode = "Any endpoint"
        if network_options:
            selected_networks = st.multiselect("Networks", network_options, default=[], key=f"{method_name}_graph_networks")
            network_filter_mode = st.radio("Network filter", ["Any endpoint", "Both endpoints"], horizontal=True, key=f"{method_name}_graph_network_mode")
        else:
            st.caption("No network labels available for filtering.")

    pos_edges = _filter_edges_for_tail(edges_df, atlas, "positive", alpha, top_n, rank_by, selected_networks, network_filter_mode)
    neg_edges = _filter_edges_for_tail(edges_df, atlas, "negative", alpha, top_n, rank_by, selected_networks, network_filter_mode)

    col_pos, col_neg = st.columns(2)
    with col_pos:
        st.markdown(f"**{direction_labels['positive']}: {len(pos_edges)} edges**")
        if pos_edges.empty:
            st.info(f"No edges for {direction_labels['positive']} pass the current graph filters.")
        else:
            fig_pos = plot_connectome_graph(
                edges_df=edges_df,
                atlas=atlas,
                tail="positive",
                alpha=alpha,
                top_n=top_n,
                display_mode=display_mode,
                color_by=color_by,
                weight_by=weight_by,
                rank_by=rank_by,
                selected_networks=selected_networks,
                network_filter_mode=network_filter_mode,
                title=direction_labels["positive_title"]
            )
            st.pyplot(fig_pos, use_container_width=True)
            _download_figure_button(fig_pos, f"Download {direction_labels['positive']} graph PNG", f"{safe_filename_part(direction_labels['positive'])}_connectome_{safe_filename_part(method_name)}.png", f"{method_name}_pos_graph_download")
            plt.close(fig_pos)

    with col_neg:
        st.markdown(f"**{direction_labels['negative']}: {len(neg_edges)} edges**")
        if neg_edges.empty:
            st.info(f"No edges for {direction_labels['negative']} pass the current graph filters.")
        else:
            fig_neg = plot_connectome_graph(
                edges_df=edges_df,
                atlas=atlas,
                tail="negative",
                alpha=alpha,
                top_n=top_n,
                display_mode=display_mode,
                color_by=color_by,
                weight_by=weight_by,
                rank_by=rank_by,
                selected_networks=selected_networks,
                network_filter_mode=network_filter_mode,
                title=direction_labels["negative_title"]
            )
            st.pyplot(fig_neg, use_container_width=True)
            _download_figure_button(fig_neg, f"Download {direction_labels['negative']} graph PNG", f"{safe_filename_part(direction_labels['negative'])}_connectome_{safe_filename_part(method_name)}.png", f"{method_name}_neg_graph_download")
            plt.close(fig_neg)

def _render_table_and_download(edges_df, contrast_name, file_suffix):
    st.markdown("##### 📋 Significant Edges Table")
    st.dataframe(edges_df, use_container_width=True)
    
    csv_data = edges_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        f"Download {file_suffix} Significant Edges CSV",
        data=csv_data,
        file_name=f"significant_edges_{safe_filename_part(contrast_name)}_{file_suffix}.csv",
        mime="text/csv"
    )

def _render_comparative_view(primary_res, primary_edges, comp_res, comp_edges, primary_name, comp_name, base_atlas, effect_mask, direction_labels):
    st.markdown("#### 🔄 Side-by-Side Comparison")
    
    # 1. Compare summary metrics in two columns
    col_p, col_c = st.columns(2)
    with col_p:
        st.markdown(f"**Primary Analysis ({primary_name})**")
        nsig_p = primary_res.n_significant(0.05)
        st.write(f"- **{direction_labels['positive']}:** `{nsig_p['positive']}`")
        st.write(f"- **{direction_labels['negative']}:** `{nsig_p['negative']}`")
        st.write(f"- **Permutations:** `{primary_res.n_permutations}`")
    with col_c:
        st.markdown(f"**Sensitivity Companion ({comp_name})**")
        nsig_c = comp_res.n_significant(0.05)
        st.write(f"- **{direction_labels['positive']}:** `{nsig_c['positive']}`")
        st.write(f"- **{direction_labels['negative']}:** `{nsig_c['negative']}`")
        st.write(f"- **Permutations:** `{comp_res.n_permutations}`")
        
    # 2. Compare ground truth side-by-side if synthetic
    if effect_mask is not None:
        st.divider()
        st.markdown("##### 🎯 Ground Truth Comparison")
        
        true_mask = (effect_mask > 0)
        true_mask_triu = np.triu(true_mask, 1)
        n_nodes = true_mask.shape[0]
        possible_edges = int(n_nodes * (n_nodes - 1) / 2)
        n_true_pos_edges = int(np.sum(true_mask_triu))
        n_true_neg_edges = possible_edges - n_true_pos_edges
        
        # Helper to compute TPR, FPR, FNR, FDR
        def _get_metrics(edges_df):
            detected = set()
            if edges_df is not None and not edges_df.empty:
                for _, row in edges_df.iterrows():
                    u, v = min(int(row['roi_i']), int(row['roi_j'])), max(int(row['roi_i']), int(row['roi_j']))
                    detected.add((u, v))
            tp = sum(1 for u, v in detected if true_mask_triu[u, v])
            fp = len(detected) - tp
            fn = n_true_pos_edges - tp
            tpr = tp / n_true_pos_edges if n_true_pos_edges > 0 else 0.0
            fpr = fp / n_true_neg_edges if n_true_neg_edges > 0 else 0.0
            fnr = fn / n_true_pos_edges if n_true_pos_edges > 0 else 0.0
            fdr = fp / len(detected) if len(detected) > 0 else 0.0
            return tp, fp, fn, tpr, fpr, fnr, fdr
            
        tp_p, fp_p, fn_p, tpr_p, fpr_p, fnr_p, fdr_p = _get_metrics(primary_edges)
        tp_c, fp_c, fn_c, tpr_c, fpr_c, fnr_c, fdr_c = _get_metrics(comp_edges)
        
        col_m_p, col_m_c = st.columns(2)
        with col_m_p:
            st.markdown(f"**{primary_name} Performance**")
            st.metric("Sensitivity / Power (TPR)", f"{tpr_p*100:.1f}%", help=f"{tp_p} / {n_true_pos_edges}")
            st.metric("Type I Error Rate (FPR)", f"{fpr_p*100:.1f}%", help=f"{fp_p} / {n_true_neg_edges}")
            st.metric("False Discovery Rate (FDR)", f"{fdr_p*100:.1f}%", help=f"{fp_p} / {len(primary_edges) if primary_edges is not None else 0}")
        with col_m_c:
            st.markdown(f"**{comp_name} Performance**")
            st.metric("Sensitivity / Power (TPR)", f"{tpr_c*100:.1f}%", help=f"{tp_c} / {n_true_pos_edges}")
            st.metric("Type I Error Rate (FPR)", f"{fpr_c*100:.1f}%", help=f"{fp_c} / {n_true_neg_edges}")
            st.metric("False Discovery Rate (FDR)", f"{fdr_c*100:.1f}%", help=f"{fp_c} / {len(comp_edges) if comp_edges is not None else 0}")

    # 3. Compare heatmaps side-by-side
    st.divider()
    st.markdown("##### 🗺️ Edge Significance Maps Comparison (-log10 p-value)")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Primary Map
    p_pos_p = primary_res["positive"]
    p_neg_p = primary_res["negative"]
    min_p_p = np.minimum(p_pos_p, p_neg_p)
    log_p_p = -np.log10(min_p_p + 1e-10)
    log_p_p[min_p_p > 0.05] = 0.0
    im1 = ax1.imshow(log_p_p, cmap="inferno", aspect="auto")
    fig.colorbar(im1, ax=ax1)
    ax1.set_title(f"{primary_name} Significant Edges")
    
    # Companion Map
    p_pos_c = comp_res["positive"]
    p_neg_c = comp_res["negative"]
    min_p_c = np.minimum(p_pos_c, p_neg_c)
    log_p_c = -np.log10(min_p_c + 1e-10)
    log_p_c[min_p_c > 0.05] = 0.0
    im2 = ax2.imshow(log_p_c, cmap="inferno", aspect="auto")
    fig.colorbar(im2, ax=ax2)
    ax2.set_title(f"{comp_name} Significant Edges")
    
    st.pyplot(fig)
    plt.close(fig)

def render_inference_results_view(base_atlas):
    col_t, col_h = st.columns([0.8, 0.2])
    with col_t:
        st.markdown("### Inference Results")
    with col_h:
        render_help("inference_results")
    
    is_stale = result_is_stale()

    if is_stale:
        st.warning("⚠️ **Stale Results Detected:** The dataset configuration, parcellation, or preprocessing settings have changed in Tab 1 since these inference results were generated. Please re-run the inference in Tab 2 to obtain up-to-date results.")
        
    if st.session_state.inference_result is None:
        st.warning("Please configure and run inference in Tab 2 first.")
    else:
        # Determine if companion result is available
        companion_res_wrapper = st.session_state.get("companion_inference_result")
        
        primary_method = st.session_state.run_plan.get("method", "Primary").upper()
        effect_mask = st.session_state.get("_synthetic_effect_mask")
        contrast_name = current_contrast_name()
        direction_labels = effect_direction_labels(st.session_state.get("run_plan"))
        
        if companion_res_wrapper is not None:
            companion_method = st.session_state.get("companion_method", "Companion").upper()
            
            view_mode = st.radio(
                "Select Result View Mode:",
                [f"Primary Run ({primary_method})", f"Sensitivity Companion ({companion_method})", "Side-by-Side Comparison"],
                horizontal=True
            )
            st.write("")
            
            if view_mode == f"Primary Run ({primary_method})":
                res_wrapper = st.session_state.inference_result
                res = getattr(res_wrapper, "inference", res_wrapper)
                edges_df = st.session_state.edges_df
                
                _render_summary_metrics(res, edges_df, primary_method, direction_labels)
                _render_ground_truth_metrics(edges_df, effect_mask, primary_method)
                _render_heatmaps(res, base_atlas, primary_method)
                _render_brain_connectome_graphs(edges_df, base_atlas, primary_method, direction_labels)
                _render_table_and_download(edges_df, contrast_name, primary_method)
                
            elif view_mode == f"Sensitivity Companion ({companion_method})":
                res = getattr(companion_res_wrapper, "inference", companion_res_wrapper)
                edges_df = st.session_state.companion_edges_df
                
                _render_summary_metrics(res, edges_df, companion_method, direction_labels)
                _render_ground_truth_metrics(edges_df, effect_mask, companion_method)
                _render_heatmaps(res, base_atlas, companion_method)
                _render_brain_connectome_graphs(edges_df, base_atlas, companion_method, direction_labels)
                _render_table_and_download(edges_df, contrast_name, companion_method)
                
            else: # Side-by-Side
                primary_res = getattr(st.session_state.inference_result, "inference", st.session_state.inference_result)
                primary_edges = st.session_state.edges_df
                comp_res = getattr(companion_res_wrapper, "inference", companion_res_wrapper)
                comp_edges = st.session_state.companion_edges_df
                
                _render_comparative_view(primary_res, primary_edges, comp_res, comp_edges, primary_method, companion_method, base_atlas, effect_mask, direction_labels)
                
                # Show tables in tabs
                tab_p, tab_c = st.tabs([f"Primary Edges ({primary_method})", f"Companion Edges ({companion_method})"])
                with tab_p:
                    _render_table_and_download(primary_edges, contrast_name, primary_method)
                with tab_c:
                    _render_table_and_download(comp_edges, contrast_name, companion_method)
        else:
            # Traditional single run view
            res_wrapper = st.session_state.inference_result
            res = getattr(res_wrapper, "inference", res_wrapper)
            edges_df = st.session_state.edges_df
            
            _render_summary_metrics(res, edges_df, primary_method, direction_labels)
            _render_ground_truth_metrics(edges_df, effect_mask, primary_method)
            _render_heatmaps(res, base_atlas, primary_method)
            _render_brain_connectome_graphs(edges_df, base_atlas, primary_method, direction_labels)
            _render_table_and_download(edges_df, contrast_name, primary_method)
