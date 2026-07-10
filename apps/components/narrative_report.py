import os
import streamlit as st
from conninfpy.interpret.evidence import validate_evidence
from conninfpy.interpret.llm_narrative import LLMNarrator, check_narrative_terms
from apps.utils.helpers import current_contrast_name, safe_filename_part, render_help, result_is_stale

def render_narrative_report_view():
    col_t, col_h = st.columns([0.8, 0.2])
    with col_t:
        st.markdown("### Narrative Generator")
    with col_h:
        render_help("narrative_report")
    
    # Collapsible LLM Config
    with st.expander("🤖 LLM Settings & API Keys", expanded=False):
        provider = st.selectbox(
            "Provider", 
            ["mock", "openrouter", "gemini", "openai"], 
            index=["mock", "openrouter", "gemini", "openai"].index(st.session_state.llm_provider)
        )
        
        selected_models = []
        if provider == "openrouter":
            available_models = [
                "deepseek/deepseek-v4-pro",
                "google/gemini-3.5-flash",
                "qwen/qwen3.7-max",
                "z-ai/glm-5.2",
                "minimax/minimax-m3",
                "moonshotai/kimi-k2.6"
            ]
            default_selection = [st.session_state.llm_model] if st.session_state.llm_model in available_models else ["deepseek/deepseek-v4-pro"]
            selected_models = st.multiselect(
                "OpenRouter Model(s)",
                available_models,
                default=default_selection,
                help="Select one or more models to query sequentially and compare outputs."
            )
            model_val = selected_models[0] if selected_models else "deepseek/deepseek-v4-pro"
        else:
            model_val = st.text_input(
                "Model Name (Optional)", 
                value=st.session_state.llm_model, 
                placeholder="e.g. gpt-4o-mini"
            )
            selected_models = [model_val] if model_val else []
            
        api_key = st.text_input(
            "API Key (Override)", 
            value=st.session_state.llm_api_key, 
            type="password"
        )
        
        # Save back to session state
        st.session_state.llm_provider = provider
        st.session_state.llm_model = model_val
        st.session_state.llm_api_key = api_key

    is_stale = result_is_stale()

    if is_stale:
        st.error("❌ **Stale Results:** The dataset configuration has changed. You must re-run inference in Tab 2 before generating the narrative report.")
    elif st.session_state.evidence_packet is None:
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
                key_to_use = st.session_state.llm_api_key
                if not key_to_use:
                    if st.session_state.llm_provider == "openrouter":
                        key_to_use = os.getenv("OPENROUTER_API_KEY")
                    elif st.session_state.llm_provider == "openai":
                        key_to_use = os.getenv("OPENAI_API_KEY")
                    elif st.session_state.llm_provider == "gemini":
                        key_to_use = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                        
                if st.session_state.llm_provider != "mock":
                    if key_to_use:
                        st.success("API Key detected in environment/.env.")
                    else:
                        st.warning("No API key detected. Please add it to the settings expander above.")
                        
                gen_button = st.button("✍️ Generate Report Narrative", type="primary")
                
            if gen_button:
                if not selected_models:
                    st.error("Please select or specify at least one model.")
                else:
                    st.session_state.narrative_results = {}
                    for model in selected_models:
                        try:
                            narrator = LLMNarrator(
                                provider=st.session_state.llm_provider,
                                model=model if model else None,
                                api_key=key_to_use if key_to_use else None
                            )
                            with st.spinner(f"Generating narrative using {model}..."):
                                narrative = narrator.generate(st.session_state.evidence_packet)
                                st.session_state.narrative_results[model] = {
                                    "text": narrative,
                                    "usage": narrator.last_usage
                                }
                        except Exception as e:
                            st.error(f"LLM Narration failed for {model}: {e}")
                    # Clear single-text cache to avoid mixed state
                    st.session_state.narrative_text = "multi_model"
                    
            results = st.session_state.get("narrative_results")
            if results:
                with col2:
                    st.markdown("**LLM Generated Narrative Comparison**")
                    
                    # Create tabs for each model
                    model_tab_names = [m.split("/")[-1] for m in results.keys()]
                    model_tabs = st.tabs(model_tab_names)
                    
                    for idx, (m_name, m_data) in enumerate(results.items()):
                        with model_tabs[idx]:
                            st.markdown(f"##### 🤖 Model: `{m_name}`")
                            
                            # Term guardrails warning check
                            unsupported = check_narrative_terms(m_data["text"], st.session_state.evidence_packet)
                            if unsupported:
                                st.warning(f"⚠️ **Term Guardrail Warning**: The generated answer contains cognitive terms not present in the NiMARE evidence: `{', '.join(unsupported)}`.")
                            
                            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                            st.markdown(m_data["text"])
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Usage
                            usage = m_data.get("usage")
                            if usage is not None:
                                st.caption(
                                    f"📊 **LLM Resource Consumption:** Input: `{usage['prompt_tokens']}` tokens | "
                                    f"Output: `{usage['completion_tokens']}` tokens | "
                                    f"Total: `{usage['total_tokens']}` tokens | "
                                    f"Estimated Cost: **${usage['cost_usd']:.5f}**"
                                )
                            
                            # Export narrative report block
                            report_block = f"""# Neuroimaging Decoding Report
Contrast: {st.session_state.evidence_packet['query']['contrast']}
Atlas: {st.session_state.evidence_packet['query']['atlas']}
Method: {st.session_state.evidence_packet['decoder']['method']}
Model: {m_name}

{m_data["text"]}

---
*Generated by ConnInfPy interpretation layer. Cautions apply.*
"""
                            st.download_button(
                                f"Export Markdown Report ({m_name.split('/')[-1]})",
                                data=report_block.encode('utf-8'),
                                file_name=f"nimare_narrative_{m_name.split('/')[-1]}_{safe_filename_part(current_contrast_name())}.md",
                                mime="text/markdown",
                                key=f"download_{m_name}"
                            )
