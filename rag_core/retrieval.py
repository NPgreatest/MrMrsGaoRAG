import os
import json
import faiss
import numpy as np
from rag_core.embedding import encode_text

FAISS_INDEX_PATH = "./faiss/faiss_index"


def search_faiss(query, top_k=5, min_threshold=0.85):
    """Search FAISS and retrieve matching results."""
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(f"{FAISS_INDEX_PATH}_metadata.json"):
        return [{"error": "Faiss index or metadata file not found."}]

    index = faiss.read_index(FAISS_INDEX_PATH)
    query_embedding = encode_text([query])
    distances, indices = index.search(query_embedding, top_k)

    with open(f"{FAISS_INDEX_PATH}_metadata.json", "r", encoding="utf-8") as f:
        metadata_list = {entry["id"]: entry for entry in json.load(f)}

    results = []
    retrieved_videos = set()

    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue

        metadata = metadata_list.get(int(idx), {})
        video_id = metadata.get("video_id", None)
        video_name = metadata.get("video_name", "Unknown")
        video_link = metadata.get("video_link", "")
        text = metadata.get("text", "Unknown")
        similarity = 1 / (1 + distances[0][i])

        if similarity >= min_threshold and video_id not in retrieved_videos:
            full_video_text = " ".join(
                meta["text"] for meta in metadata_list.values() if meta["video_id"] == video_id
            )
            results.append({
                "video_id": video_id, "video_name": video_name,
                "text": full_video_text, "video_link": video_link,
                "similarity": similarity, "entire": True
            })
            retrieved_videos.add(video_id)
        elif video_id not in retrieved_videos:
            results.append({
                "video_id": video_id, "video_name": video_name,
                "text": text, "video_link": video_link,
                "similarity": similarity, "entire": False
            })

    return results
