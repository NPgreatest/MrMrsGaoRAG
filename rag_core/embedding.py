import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

from api_providers.silicon_flow.api import SiliconFlowLLMAPI

# Set embedding model
embedding_model = SentenceTransformer("moka-ai/m3e-base")

# Paths
TRANSCRIBE_DIR = "./transcribe/"
FAISS_INDEX_PATH = "./faiss/faiss_hnsw_index"


def encode_text(text_list):
    # api_provider = SiliconFlowLLMAPI()
    # return api_provider.encode_text(text_list,show_progress=True).cpu().numpy()
    return embedding_model.encode(text_list,show_progress=True)


def split_text(text, chunk_size=300, overlap=20):
    """Splits text into chunks using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap, separators=["。", "！", "？", "\n", " "]
    )
    return splitter.split_text(text)


def extract_video_links():
    # Read JSON data from video_url.json
    config_file = "./configs/video_url.json"
    with open(config_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Create a dictionary for quick lookup
    video_url_map = {item["title"]: item["url"] for item in data}

    return video_url_map


def process_json_files():
    """Reads JSON transcription files and extracts metadata."""
    all_texts, metadata_list = [], []
    video_id_map = {}
    video_url_map = extract_video_links()

    for idx, filename in enumerate(os.listdir(TRANSCRIBE_DIR)):
        if not filename.endswith(".json"):
            continue

        file_path = os.path.join(TRANSCRIBE_DIR, filename)
        video_name = os.path.splitext(filename)[0]

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "segments" not in data:
                print(f"Warning: {file_path} missing 'segments' field")
                continue

            video_id = video_id_map.setdefault(video_name, idx)
            video_link = video_url_map.get(video_name, "")

            full_text = " ".join(seg["text"] for seg in data["segments"])
            text_chunks = split_text(full_text)

            all_texts.extend(text_chunks)
            metadata_list.extend([
                {"video_id": video_id, "video_name": video_name, "text": chunk, "video_link": video_link}
                for chunk in text_chunks
            ])

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    return all_texts, metadata_list


def build_faiss_index():
    """Builds an HNSW-based FAISS index from transcript data."""
    all_texts, metadata_list = process_json_files()

    if not all_texts:
        print("No text data found.")
        return
    print("try to embedding the texts")
    embeddings = encode_text(all_texts)
    print("done embedding the texts")
    dimension = embeddings.shape[1]

    # Create an HNSW index for fast searching
    index = faiss.IndexHNSWFlat(dimension, 32)  # 32 is the number of neighbors in the HNSW graph
    index = faiss.IndexIDMap2(index)

    ids = np.arange(len(all_texts))
    index.add_with_ids(embeddings, ids)

    faiss.write_index(index, FAISS_INDEX_PATH)

    metadata_json = [
        {"id": int(id_), "video_id": meta["video_id"], "video_name": meta["video_name"],
         "text": meta["text"], "video_link": meta["video_link"]}
        for id_, meta in zip(ids, metadata_list)
    ]

    with open(f"{FAISS_INDEX_PATH}_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_json, f, ensure_ascii=False, indent=4)

    print(f"FAISS HNSW index built with {len(all_texts)} text entries.")


if __name__ == "__main__":
    build_faiss_index()
