import numpy as np
import requests
import yaml
from tqdm import tqdm


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


import httpx


import httpx
import traceback

class SiliconFlowLLMAPI:
    async def query_llm(self, prompt):
        """Queries DeepSeek AI using the Silicon Flow API asynchronously."""
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
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(f"{API_URL}/chat/completions", json=payload, headers=HEADERS)

                # Debugging: Print raw response details
                print(f"Response Status: {response.status_code}")
                print(f"Response Headers: {response.headers}")
                print(f"Response Content: {response.text}")

                response.raise_for_status()  # Raise an error if not 200

                response_data = response.json()
                choices = response_data.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", f"No response from Silicon Flow {API_MODEL}.")
                return "No response from DeepSeek AI."

        except httpx.HTTPStatusError as e:
            print(f"HTTP Error: {e.response.status_code}")
            print(f"Response Content: {e.response.text}")
            return f"HTTP Error: {e.response.status_code}, Response: {e.response.text}"

        except httpx.RequestError as e:
            print(f"Request Error: {str(e)}")
            traceback.print_exc()  # Print full traceback for debugging
            return f"Request Error: {str(e)}"

        except Exception as e:
            print(f"Unexpected Error: {str(e)}")
            traceback.print_exc()
            return f"Unexpected Error: {str(e)}"


    def encode_text(self, text_list, show_progress=False):
        """Generate embeddings using the Silicon Flow API."""
        embeddings = []
        iterator = tqdm(text_list, desc="Encoding Text", disable=not show_progress)

        for text in iterator:
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


import asyncio
import numpy as np

if __name__ == "__main__":
    """Test the SiliconFlowLLMAPI functions."""
    api = SiliconFlowLLMAPI()


    # Test LLM Query (Async)
    async def test_llm():
        test_prompt = "What LLM are you?"
        llm_response = await api.query_llm(test_prompt)  # Use await inside an async function
        print(f"LLM Response:\n{llm_response}\n")


    asyncio.run(test_llm())  # Run the async function in a blocking manner

    # Test Embedding Generation (Sync)
    test_texts = ["Silicon Flow API is powerful.", "DeepSeek AI provides great embeddings."]
    embeddings = api.encode_text(test_texts, show_progress=True)

    print(f"Embeddings Shape: {embeddings.shape}")
    print(f"Embeddings result: {embeddings}")
