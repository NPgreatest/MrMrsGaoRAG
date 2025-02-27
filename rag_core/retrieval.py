import json
import os

import faiss

from rag_core.embedding import encode_text

FAISS_INDEX_PATH = "./faiss/faiss_hnsw_index"


def process_results(results):
    """Process the search results to extract unique video links and concatenate text content."""

    video_links = list(set(result["video_link"] for result in results if result["video_link"]))

    str_for_llm = "\n\n".join(
        f"【{result['video_name']}】\n{result['text']}"
        for result in results
    )

    return video_links, str_for_llm


def search_faiss(query, top_k=5, min_threshold=0.85, context_window=2):
    """Search FAISS and retrieve matching results with context handling."""
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(f"{FAISS_INDEX_PATH}_metadata.json"):
        return [{"error": "Faiss index or metadata file not found."}]

    index = faiss.read_index(FAISS_INDEX_PATH)
    query_embedding = encode_text([query])
    distances, indices = index.search(query_embedding, top_k)

    with open(f"{FAISS_INDEX_PATH}_metadata.json", "r", encoding="utf-8") as f:
        metadata_list = {entry["id"]: entry for entry in json.load(f)}

    results = []
    processed_segs = set()

    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue

        metadata = metadata_list.get(int(idx), {})
        id = metadata.get("id", None)
        video_id = metadata.get("video_id", None)
        video_name = metadata.get("video_name", "Unknown")
        video_link = metadata.get("video_link", "")
        text = metadata.get("text", "Unknown")
        similarity = 1 / (1 + distances[0][i])

        if id in processed_segs:
            continue

        if similarity >= min_threshold:
            neighbors = [metadata_list.get(int(idx + offset), {}) for offset in range(-context_window, context_window + 1) if (idx + offset) in metadata_list]
            context_texts = [neighbor.get("text", "") for neighbor in neighbors if neighbor]
            overlapped_text = " ... ".join(context_texts)
        else:
            overlapped_text = text

        results.append({
            "video_id": video_id, "video_name": video_name,
            "text": overlapped_text, "video_link": video_link,
            "similarity": similarity
        })

        processed_segs.add(id)

    video_links, str_for_llm =process_results(results)

    return results, video_links, str_for_llm
