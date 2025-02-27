import gradio as gr
from rag_core.retrieval import search_faiss
from rag_core.llm import query_llm
import ollama

AVAILABLE_APIS = ["gpt-4o", "deepseek"]


def list_ollama_models():
    """List available Ollama models."""
    try:
        models = ollama.list()
        return [model['model'] for model in models['models']]
    except Exception as e:
        return [f"Error fetching Ollama models: {e}"]


def format_search_results(search_results):
    """Format search results into HTML."""
    html_content = "<div style='max-height:600px; overflow:auto;'>"

    for item in search_results:
        video_url = item.get("video_link", "")
        video_name = item.get("video_name", "")
        similarity = item.get("similarity", 0)
        entire = item.get("entire", False)

        youtube_id = None
        if "youtube.com/watch?v=" in video_url:
            youtube_id = video_url.split("watch?v=")[-1].split("&")[0]

        html_content += "<div style='border:1px solid #ccc; margin:10px; padding:10px;'>"
        html_content += f"<h4>Video Name: {video_name}</h4>"
        html_content += f"<p><strong>Similarity:</strong> {similarity:.3f} | <strong>Entire Video Retrieved:</strong> {entire}</p>"

        if youtube_id:
            embed_url = f"https://www.youtube.com/embed/{youtube_id}"
            html_content += f"""
                <div style='margin-bottom:10px;'>
                  <iframe width="400" height="225" 
                          src="{embed_url}" 
                          frameborder="0" 
                          allowfullscreen>
                  </iframe>
                </div>
            """
        elif video_url:
            html_content += f"<p><a href='{video_url}' target='_blank'>Open Video Link</a></p>"

        html_content += "</div>"

    html_content += "</div>"
    return html_content


def rag_search(query, prompt_mod, top_k, threshold, model_choice, api_choice, ollama_choice, api_url, api_token):
    """Perform FAISS search and LLM query."""
    search_results,_,_ = search_faiss(query, top_k=top_k, min_threshold=threshold)
    full_prompt = f"{prompt_mod}\n\nQuery:\n{query}\n\nSearch Results:\n{search_results}"
    llm_response = query_llm(full_prompt, model_choice, api_choice, ollama_choice, api_url, api_token)
    search_results_html = format_search_results(search_results)

    return search_results, llm_response, search_results_html


def toggle_model_options(choice):
    """Toggle visibility between Local Ollama and API models."""
    if choice == "Local Ollama":
        return (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))
    else:
        return (gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True))


def set_api_template(selected_api):
    """Auto-populate API URL based on selected API."""
    if selected_api == "gpt-4o":
        return gr.update(value="https://api.openai.com/v1/chat/completions")
    elif selected_api == "deepseek":
        return gr.update(value="https://api.deepseek.org/v2/chat/completions")
    return gr.update(value="")


def launch_ui():
    """Launch the Gradio UI."""
    with gr.Blocks() as demo:
        gr.Markdown("## 🔍 老高宇宙 RAG Search with LLM")
        gr.Markdown("Enter a query, modify the prompt if needed, choose a model, and see the results.")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🔽 Input")
                query_input = gr.Textbox(label="Enter your query")
                prompt_mod_input = gr.Textbox(
                    label="Modify prompt (optional)",
                    value="请基于召回的内容整理答案，并附上视频出处"
                )

                top_k_slider = gr.Slider(minimum=5, maximum=30, step=1, value=5, label="Top-K Segments to Recall")
                threshold_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.85,
                                             label="Similarity Threshold")

                model_choice = gr.Radio(["Local Ollama", "API"], label="Choose Model", value="Local Ollama")

                ollama_model_choice = gr.Dropdown(
                    list_ollama_models(), label="Select Local Model", visible=True
                )

                api_choice = gr.Dropdown(
                    AVAILABLE_APIS, label="Select API", visible=False
                )

                api_url = gr.Textbox(label="API URL", visible=False)
                api_token = gr.Textbox(label="API Key", type="password", visible=False)

                model_choice.change(
                    fn=toggle_model_options,
                    inputs=model_choice,
                    outputs=[ollama_model_choice, api_choice, api_url, api_token]
                )

                api_choice.change(
                    fn=set_api_template,
                    inputs=api_choice,
                    outputs=api_url
                )

                search_button = gr.Button("🔎 Search")

            with gr.Column(scale=1):
                gr.Markdown("### 📜 Output")
                search_results_output = gr.JSON(label="Search Results", height=300)
                llm_response_output = gr.Markdown(label="LLM Response")
                search_results_html = gr.HTML(label="Video & Clips", elem_id="video_clips_display")

        search_button.click(
            fn=rag_search,
            inputs=[
                query_input, prompt_mod_input, top_k_slider, threshold_slider,
                model_choice, api_choice, ollama_model_choice, api_url, api_token
            ],
            outputs=[search_results_output, llm_response_output, search_results_html]
        )

    demo.launch()
