import numpy as np
import requests
import yaml


CONFIG_PATH = "./config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

API_URL = "https://api.siliconflow.cn/v1"
API_MODEL = config["API_MODEL"]
EMBEDDING_MODEL = config["EMBEDDING_MODEL"]
API_TOKEN = config["API_TOKEN"]
HEADERS = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

# Paths
TRANSCRIBE_DIR = "./transcribe/"
FAISS_INDEX_PATH = "./faiss/faiss_hnsw_index"


class SiliconFlowLLMAPI:
    def query_llm(self,prompt):
        """Queries DeepSeek AI using the Silicon Flow API."""
        if not API_TOKEN:
            return "Error: DeepSeek API token required."

        payload = {
            "model": API_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "max_tokens": 512,
            "stop": ["null"],
            "temperature": 0.7,
            "top_p": 0.7,
            "top_k": 50,
            "frequency_penalty": 0.5,
            "n": 1,
            "response_format": {"type": "text"},
        }

        try:
            response = requests.post(f"{API_URL}/chat/completions", json=payload, headers=HEADERS)
            response.raise_for_status()
            response_data = response.json()
            choices = response_data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", f"No response from Silicon Flow {API_MODEL}.")
            return "No response from DeepSeek AI."
        except Exception as e:
            return f"No response from Silicon Flow {API_MODEL}, error = {e}."



    def encode_text(self,text_list):
        """Generate embeddings using the Silicon Flow API."""
        embeddings = []
        for text in text_list:
            payload = {
                "model": EMBEDDING_MODEL,
                "input": text,
                "encoding_format": "float"
            }
            try:
                response = requests.post(f"{API_URL}/embeddings", json=payload, headers=HEADERS)
                response.raise_for_status()
                data = response.json()
                embeddings.append(data["data"][0]["embedding"])
            except Exception as e:
                print(f"Error generating embedding: {text[:30]}... {e}")
                embeddings.append([0] * 1024)  # Default zero vector
        return np.array(embeddings, dtype=np.float32)




if __name__ == "__main__":
    """Test the SiliconFlowLLMAPI functions."""
    api = SiliconFlowLLMAPI()
    test_prompt = "What is the meaning of life?"
    llm_response = api.query_llm(test_prompt)
    print(f"LLM Response:\n{llm_response}\n")
    # Test embedding generation
    test_texts = ["Silicon Flow API is powerful.", "DeepSeek AI provides great embeddings."]
    embeddings = api.encode_text(test_texts)
    print(f"Embeddings Shape: {embeddings.shape}")
    print(f"Embeddings result: {embeddings.data.__str__()}")