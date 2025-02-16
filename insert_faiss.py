import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 使用本地 M3E embedding 模型
embedding_model = SentenceTransformer("moka-ai/m3e-base")

# 设定数据目录
TRANSCRIBE_DIR = "./transcribe/"
FAISS_INDEX_PATH = "faiss_index"


def split_text(text, chunk_size=300, max_chunk_size=500):
    """将文本按 chunk_size 进行分割，并避免过长的 chunk"""
    chunks, current_chunk = [], []
    current_length = 0

    for char in text:
        current_chunk.append(char)
        current_length += 1
        if current_length >= chunk_size and (char == ' ' or current_length >= max_chunk_size):
            chunks.append("".join(current_chunk))
            current_chunk, current_length = [], 0

    if current_chunk:
        chunks.append("".join(current_chunk))

    return chunks


def process_json_files():
    """读取 JSON 并提取文本和元数据"""
    all_texts, metadata_list = [], []

    for filename in os.listdir(TRANSCRIBE_DIR):
        if not filename.endswith(".json"):
            continue

        file_path = os.path.join(TRANSCRIBE_DIR, filename)
        title = os.path.splitext(filename)[0]

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "segments" not in data:
                print(f"警告：{file_path} 缺少 'segments' 字段")
                continue

            full_text = " ".join(seg["text"] for seg in data["segments"])
            text_chunks = split_text(full_text)

            all_texts.extend(text_chunks)
            metadata_list.extend([{"video_name": title, "text": chunk} for chunk in text_chunks])

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
    metadata_json = [{"id": int(id_), "video_name": meta["video_name"], "text": meta["text"]}
                     for id_, meta in zip(ids, metadata_list)]

    with open(f"{FAISS_INDEX_PATH}_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_json, f, ensure_ascii=False, indent=4)

    print(f"已创建 {len(all_texts)} 条文本的 FAISS 向量索引，并存储了 metadata")


def search_faiss(query, top_k=5):
    """搜索 Faiss 并返回匹配的文本和 metadata"""
    # 加载 Faiss 索引
    index = faiss.read_index(FAISS_INDEX_PATH)

    # 计算查询向量
    query_embedding = embedding_model.encode([query], convert_to_tensor=True).cpu().numpy()

    # 搜索最近的向量
    distances, indices = index.search(query_embedding, top_k)

    # 读取 metadata
    with open(f"{FAISS_INDEX_PATH}_metadata.json", "r", encoding="utf-8") as f:
        metadata_list = {entry["id"]: entry for entry in json.load(f)}

    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        metadata = metadata_list.get(int(idx), {})
        results.append({
            "video_name": metadata.get("video_name", "Unknown"),
            "text": metadata.get("text", "Unknown"),
            "distance": distances[0][i]
        })

    return results


if __name__ == "__main__":
    build_faiss_index()

    # 测试查询
    query_text = "外星人遗迹"
    results = search_faiss(query_text, top_k=3)

    for result in results:
        print(f"匹配到的文件: {result['video_name']}, 片段内容: {result['text']}, 相似度: {result['distance']:.4f}")
