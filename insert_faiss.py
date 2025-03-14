import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 使用本地 M3E embedding 模型
embedding_model = SentenceTransformer("moka-ai/m3e-base")

# 设定数据目录
TRANSCRIBE_DIR = "./transcribe/"
FAISS_INDEX_PATH = "./faiss/faiss_index"


def split_text(text, chunk_size=300, max_chunk_size=500):
    """将文本按 chunk_size 进行分割，并避免过长的 chunk"""
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
    """读取 JSON 并提取文本和元数据"""
    all_texts, metadata_list = [], []
    video_id_map = {}  # 用于存储视频 ID，确保唯一

    for idx, filename in enumerate(os.listdir(TRANSCRIBE_DIR)):
        if not filename.endswith(".json"):
            continue

        file_path = os.path.join(TRANSCRIBE_DIR, filename)
        video_name = os.path.splitext(filename)[0]

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "segments" not in data:
                print(f"警告：{file_path} 缺少 'segments' 字段")
                continue

            # 生成 video_id
            video_id = video_id_map.setdefault(video_name, idx)

            # 解析视频链接（如果存在）
            video_link = data.get("video_link", "")

            # 提取并分割文本
            full_text = " ".join(seg["text"] for seg in data["segments"])
            text_chunks = split_text(full_text)

            # 记录所有文本及对应的 metadata
            all_texts.extend(text_chunks)
            metadata_list.extend([
                {"video_id": video_id, "video_name": video_name, "text": chunk, "video_link": video_link}
                for chunk in text_chunks
            ])

        except Exception as e:
            print(f"处理 {filename} 时出错: {e}")

    return all_texts, metadata_list


def build_faiss_index():
    """构建 FAISS 索引并存储"""
    all_texts, metadata_list = process_json_files()

    if not all_texts:
        print("未找到任何文本数据")
        return

    # 生成嵌入向量
    embeddings = embedding_model.encode(all_texts, convert_to_tensor=True).cpu().numpy()

    # FAISS 索引
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIDMap2(index)  # 允许存储 ID 以匹配 metadata

    # 为每个文本分配一个唯一的 ID
    ids = np.arange(len(all_texts))
    index.add_with_ids(embeddings, ids)

    # 存储索引
    faiss.write_index(index, FAISS_INDEX_PATH)

    # **存储 metadata 为 JSON**
    metadata_json = [
        {"id": int(id_), "video_id": meta["video_id"], "video_name": meta["video_name"],
         "text": meta["text"], "video_link": meta["video_link"]}
        for id_, meta in zip(ids, metadata_list)
    ]

    with open(f"{FAISS_INDEX_PATH}_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_json, f, ensure_ascii=False, indent=4)

    print(f"已创建 {len(all_texts)} 条文本的 FAISS 向量索引，并存储了 metadata")


if __name__ == "__main__":
    build_faiss_index()
