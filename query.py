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
    # 测试查询
    query_text = "古文明中，石头雕刻不可思议的地方"
    results = search_faiss(query_text, top_k=10)

    for result in results:
        print(f"匹配到的文件: {result['video_name']}, 片段内容: {result['text']}, 相似度: {result['distance']:.4f}")
