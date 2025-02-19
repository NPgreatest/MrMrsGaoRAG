import ollama
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

def query_llm(prompt, model_choice, api_choice=None, ollama_model_choice=None, api_url=None, api_token=None):
    """Queries either Local Ollama or API-based LLM."""
    if model_choice == "Local Ollama":
        try:
            response = ollama.chat(model=ollama_model_choice, messages=[{"role": "user", "content": prompt}])
            return response.get('message', {}).get('content', "No response from Ollama.")
        except Exception as e:
            return f"Error querying Local Ollama: {e}"

    elif model_choice == "API":
        if not api_url or not api_token:
            return "Error: API URL and API token required."

        try:
            llm = ChatOpenAI(model_name=api_choice, openai_api_key=api_token)
            response = llm([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            return f"Error querying API model: {e}"

    return "Invalid model choice."
