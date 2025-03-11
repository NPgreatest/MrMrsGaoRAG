import gradio as gr
import asyncio
from api_providers.silicon_flow.api import SiliconFlowLLMAPI
from rag_core.retrieval import search_faiss


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

async def rag_search(query, prompt_mod, top_k, threshold):
    """Perform FAISS search and query LLM."""
    try:
        results, video_links, str_for_llm = await search_faiss(query, top_k, threshold)
        search_results_html = format_search_results(results)
        full_prompt = f"{prompt_mod}\n\n用户的问题:\n{query}\n\n召回的内容:\n{str_for_llm}"
        api_provider = SiliconFlowLLMAPI()
        llm_task = asyncio.create_task(api_provider.query_llm(full_prompt))
        llm_response = await llm_task
        return results, llm_response, search_results_html
    except Exception as e:
        return [], f"Error during search: {e}", ""


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
                    value="请基于老高视频中的内容片段，以老高第一人称的视角，并且运用片段中的口语化语言风格解答问题，并且引用片段时要提到在哪期视频中提到的（只用说视频的标题简略版本，比如\"就像在《【震撼】她通過一個獨特的方法看到了真正的天堂》这期视频里提到的\" 变为 \"我在天堂那期影片里提到过\"，不要说完整标题）"
                )
                top_k_slider = gr.Slider(minimum=5, maximum=30, step=1, value=5, label="Top-K Segments to Recall")
                threshold_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.85, label="Similarity Threshold")
                search_button = gr.Button("🔎 Search")
            with gr.Column(scale=1):
                gr.Markdown("### 📜 Output")
                search_results_output = gr.JSON(label="Search Results", height=300)
                llm_response_output = gr.Markdown(label="LLM Response")
                search_results_html = gr.HTML(label="Video & Clips", elem_id="video_clips_display")
        search_button.click(
            fn=rag_search,
            inputs=[query_input, prompt_mod_input, top_k_slider, threshold_slider],
            outputs=[search_results_output, llm_response_output, search_results_html]
        )
    demo.launch()
