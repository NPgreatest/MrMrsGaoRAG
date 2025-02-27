import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

deepseek_token = os.getenv("DEEPSEEK_API_KEY")
model = os.getenv("DEEPSEEK_MODEL")

def query_llm(prompt):
    """Queries DeepSeek AI using the Silicon Flow API."""
    if not deepseek_token:
        return "Error: DeepSeek API token required."

    url = "https://api.siliconflow.cn/v1/chat/completions"
    payload = {
        "model": model,
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
    headers = {
        "Authorization": f"Bearer {deepseek_token}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response_data = response.json()
        choices = response_data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", f"No response from Silicon flow {model}.")
        return "No response from DeepSeek AI."
    except Exception as e:
        return f"No response from Silicon flow {model}."





# Example usage
if __name__ == "__main__":
    prompt = "Hello, how does this work?"
    response = query_llm(prompt)
    print(response)
