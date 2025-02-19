import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Set embedding model
embedding_model = SentenceTransformer("moka-ai/m3e-base")

# Paths
TRANSCRIBE_DIR = "./transcribe/"
FAISS_INDEX_PATH = "./faiss/faiss_index"


def encode_text(text_list):
    """Generate embeddings for a list of texts."""
    return embedding_model.encode(text_list, convert_to_tensor=True).cpu().numpy()


def split_text(text, chunk_size=300, max_chunk_size=500):
    """Splits text into chunks while ensuring natural breaking points."""
    chunks, current_chunk = [], []
    current_length = 0

    for char in text:
        current_chunk.append(char)
        current_length += 1
        if current_length >= chunk_size and (char in "。！？ \n" or current_length >= max_chunk_size):
            chunks.append("".join(current_chunk))
            current_chunk, current_length = [], 0

    if current_chunk:
        chunks.append("".join(current_chunk))

    return chunks


def process_json_files():
    """Reads JSON transcription files and extracts metadata."""
    all_texts, metadata_list = [], []
    video_id_map = {}

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
            video_link = data.get("video_link", "")

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
    """Builds a FAISS index from transcript data."""
    all_texts, metadata_list = process_json_files()

    if not all_texts:
        print("No text data found.")
        return

    embeddings = encode_text(all_texts)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
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

    print(f"FAISS index built with {len(all_texts)} text entries.")


if __name__ == "__main__":
    build_faiss_index()
