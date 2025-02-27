from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_core.retrieval import search_faiss
from rag_core.llm import query_llm
from typing import Optional
import logging
import datetime
import ollama

app = FastAPI()

# Configure logging (append-only mode)
log_filename = "./log/query_log.txt"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    encoding="utf-8",
    filemode="a",  # "a" ensures the log file is append-only
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RequestBody(BaseModel):
    query: str
    prompt_mod: Optional[str] = "请基于召回的内容整理答案，并附上视频出处"
    top_k: Optional[int] = 5
    threshold: Optional[float] = 0.85
    api_choice: Optional[str] = None
    api_url: Optional[str] = None
    api_token: Optional[str] = None

def log_request(request: Request, request_body: RequestBody):
    """Logs incoming request details to a file in append mode."""
    client_ip = request.client.host
    log_message = (
        f"IP: {client_ip} | Query: {request_body.query} | "
    )
    logging.info(log_message)

@app.post("/api/search")
async def api_search(request: Request, request_body: RequestBody):
    try:
        # Log request details
        log_request(request, request_body)

        # Perform FAISS search
        results, video_links, str_for_llm = search_faiss(
            request_body.query, top_k=request_body.top_k, min_threshold=request_body.threshold
        )

        full_prompt = f"{request_body.prompt_mod}\n\n用户的问题:\n{request_body.query}\n\n召回的内容:\n{str_for_llm}"

        # Query LLM
        llm_response = query_llm(full_prompt)

        return {
            "search_results": results,
            "llm_response": llm_response,
            "video_links": video_links
        }

    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=443,  # Use port 443 for HTTPS
        ssl_keyfile="./ssl/key.pem",
        ssl_certfile="./ssl/cert.pem"
    )
