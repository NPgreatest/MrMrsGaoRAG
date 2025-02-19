import json
import os
import faiss
import gradio as gr
from sentence_transformers import SentenceTransformer
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import ollama

# 使用本地 M3E embedding 模型
embedding_model = SentenceTransformer("moka-ai/m3e-base")

# 设定数据目录
TRANSCRIBE_DIR = "./transcribe/"
FAISS_INDEX_PATH = "./faiss/faiss_index"

# 仅保留两个可用 API
AVAILABLE_APIS = [
    "gpt-4o",
    "deepseek"
]


def list_ollama_models():
    """列出本地 Ollama 可用的模型"""
    try:
        models = ollama.list()
        print(models)
        return [model['model'] for model in models['models']]
    except Exception as e:
        return [f"Error fetching Ollama models {e}"]


def search_faiss(query, top_k=5, min_threshold=0.85):
    """Search FAISS and retrieve matching text and metadata.

    - If the similarity is above `min_threshold`, retrieve the entire video's transcript.
    - Skip subsequent segments from the same video if the full video has been retrieved.

    Args:
        query (str): The input search query.
        top_k (int): Number of top similar results to retrieve.
        min_threshold (float): The similarity threshold to retrieve the entire video.

    Returns:
        list: A list of matching results.
    """
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(f"{FAISS_INDEX_PATH}_metadata.json"):
        return [{"error": "Faiss index or metadata file not found."}]

    # Load FAISS index
    index = faiss.read_index(FAISS_INDEX_PATH)

    # Compute query embedding
    query_embedding = embedding_model.encode([query], convert_to_tensor=True).cpu().numpy()
    distances, indices = index.search(query_embedding, top_k)

    # Load metadata
    with open(f"{FAISS_INDEX_PATH}_metadata.json", "r", encoding="utf-8") as f:
        metadata_list = {entry["id"]: entry for entry in json.load(f)}

    results = []
    retrieved_videos = set()  # Keep track of retrieved full videos

    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue

        metadata = metadata_list.get(int(idx), {})
        video_id = metadata.get("video_id", None)
        video_name = metadata.get("video_name", "Unknown")
        video_link = metadata.get("video_link", "")
        text = metadata.get("text", "Unknown")

        # Compute similarity (convert FAISS L2 distance to similarity)
        similarity = 1 / (1 + distances[0][i])

        print(f'current sim {similarity}')

        if similarity >= min_threshold and video_id not in retrieved_videos:
            # Retrieve the entire video's transcript
            full_video_text = " ".join(
                meta["text"] for meta in metadata_list.values() if meta["video_id"] == video_id
            )

            results.append({
                "video_id": video_id,
                "video_name": video_name,
                "text": full_video_text,
                "video_link": video_link,
                "similarity": similarity,
                "entire": True  # Mark that the full video was retrieved
            })

            retrieved_videos.add(video_id)  # Mark this video as fully retrieved
        elif video_id not in retrieved_videos:  # Skip segments if the entire video was already retrieved
            results.append({
                "video_id": video_id,
                "video_name": video_name,
                "text": text,
                "video_link": video_link,
                "similarity": similarity,
                "entire": False  # This is only a single segment
            })

    return results



def format_search_results(search_results):
    """
    Takes the list of results from search_faiss and returns an HTML string
    containing embedded YouTube videos (if possible) and text snippets.
    """
    html_content = "<div style='max-height:600px; overflow:auto;'>"

    for item in search_results:
        video_url = item.get("video_link", "")
        video_name = item.get("video_name", "")
        text = item.get("text", "")
        similarity = item.get("similarity", 0)
        entire = item.get("entire", False)

        # Attempt to parse out the YouTube video ID if it's a standard link
        youtube_id = None
        if "youtube.com/watch?v=" in video_url:
            # Naive parsing – might need more robust logic if there are extra URL params
            youtube_id = video_url.split("watch?v=")[-1].split("&")[0]

        # Build a block of HTML per search result
        html_content += "<div style='border:1px solid #ccc; margin:10px; padding:10px;'>"
        html_content += f"<h4>Video Name: {video_name}</h4>"
        html_content += f"<p><strong>Similarity:</strong> {similarity:.3f} | <strong>Entire Video Retrieved:</strong> {entire}</p>"

        # If we found a YouTube ID, embed it:
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
        else:
            # Otherwise, just link it if it's not empty
            if video_url:
                html_content += f"<p><a href='{video_url}' target='_blank'>Open Video Link</a></p>"

        # # Show the text snippet or the entire text
        # html_content += f"<p>{text}</p>"
        html_content += "</div>"  # End of item block

    html_content += "</div>"
    return html_content


def query_llm(prompt, model_choice, api_choice=None, ollama_model_choice=None, api_url=None, api_token=None):
    """Query LLM using LangChain and return the response text."""
    print(f"Selected model choice: {model_choice}")

    # If user chooses Local Ollama
    if model_choice == "Local Ollama":
        try:
            response = ollama.chat(
                model=ollama_model_choice,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.get('message', {}).get('content', "No response from Ollama.")
        except Exception as e:
            return f"Error querying Local Ollama: {e}"

    # If user chooses an API-based model
    elif model_choice == "API":
        if not api_url or not api_token:
            return "Error: API URL and API token are required for API-based models."

        try:
            llm = ChatOpenAI(
                model_name=api_choice,
                openai_api_key=api_token,
            )
            response = llm([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            return f"Error querying API model: {e}"

    else:
        return "Invalid model choice. Please select 'Local Ollama' or 'API'."


def rag_search(query, prompt_modification,top_k,threshold_slider, model_choice, api_choice, ollama_model_choice, api_url, api_token):
    """整合 Faiss 搜索和 LLM 查询"""
    # 首先使用 Faiss 搜索
    search_results = search_faiss(query,top_k = top_k, min_threshold=threshold_slider)

    # 将搜索结果添加到提示中
    full_prompt = f"{prompt_modification}\n\nQuery:\n{query}\n\nSearch Results:\n{search_results}"

    # 查询 LLM
    llm_response = query_llm(full_prompt, model_choice, api_choice, ollama_model_choice, api_url, api_token)

    search_results_html = format_search_results(search_results)

    return search_results, llm_response, search_results_html




# ----- Gradio UI -----
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

            top_k_slider = gr.Slider(
                minimum=5,
                maximum=30,
                step=1,
                value=5,
                label="Select Top-K Segments to Recall"
            )

            threshold_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=0.85,
                label="Similarity Threshold for Retrieving Full Video"
            )

            # Choose between Local Ollama or API
            model_choice = gr.Radio(["Local Ollama", "API"], label="Choose Model", value="Local Ollama")

            # Local model dropdown
            ollama_model_choice = gr.Dropdown(
                list_ollama_models(),
                label="Select Local Model",
                visible=True
            )

            # API choice dropdown
            api_choice = gr.Dropdown(
                AVAILABLE_APIS,
                label="Select API",
                visible=False
            )

            # API URL text field
            api_url = gr.Textbox(
                label="API URL",
                visible=False
            )

            # API Token (password input)
            api_token = gr.Textbox(
                label="API Key",
                type="password",
                visible=False
            )


            def toggle_model_options(choice):
                """
                Toggle visibility of Local Model vs. API fields.
                """
                if choice == "Local Ollama":
                    # Show Local model, hide API selection
                    return (
                        gr.update(visible=True),  # ollama_model_choice
                        gr.update(visible=False),  # api_choice
                        gr.update(visible=False),  # api_url
                        gr.update(visible=False)  # api_token
                    )
                else:
                    # Hide Local model, show API selection
                    return (
                        gr.update(visible=False),
                        gr.update(visible=True),
                        gr.update(visible=True),
                        gr.update(visible=True)
                    )


            def set_api_template(selected_api):
                """
                Auto-populate the API URL template based on the API selected.
                """
                if selected_api == "gpt-4o":
                    return gr.update(value="https://api.openai.com/v1/chat/completions")
                elif selected_api == "deepseek":
                    return gr.update(value="https://api.deepseek.org/v2/chat/completions")
                else:
                    return gr.update(value="")


            # When model changes between Local Ollama and API
            model_choice.change(
                fn=toggle_model_options,
                inputs=model_choice,
                outputs=[ollama_model_choice, api_choice, api_url, api_token]
            )

            # When user picks one of the two APIs
            api_choice.change(
                fn=set_api_template,
                inputs=api_choice,
                outputs=api_url
            )

            search_button = gr.Button("🔎 Search")

        with gr.Column(scale=1):
            gr.Markdown("### 📜 Output")
            search_results_output = gr.JSON(label="Search Results",height=300)
            llm_response_output = gr.Markdown(label="LLM Response")

            search_results_html = gr.HTML(label="Video & Clips", elem_id="video_clips_display")



    # Connect button click to rag_search
    search_button.click(
        fn=rag_search,
        inputs=[
            query_input,
            prompt_mod_input,
            top_k_slider,
            threshold_slider,
            model_choice,
            api_choice,
            ollama_model_choice,
            api_url,
            api_token
        ],
        outputs=[
            search_results_output,
            llm_response_output,
            search_results_html
        ]
    )

if __name__ == "__main__":
    demo.launch()
