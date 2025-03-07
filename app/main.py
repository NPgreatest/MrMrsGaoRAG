import logging
import asyncio
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from api_providers.silicon_flow.api import SiliconFlowLLMAPI
from rag_core.retrieval import search_faiss

app = FastAPI()

# Configure logging (append-only mode)
log_filename = "./log/query_log.txt"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    encoding="utf-8",
    filemode="a",
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
    prompt_mod: Optional[str] = "请基于召回的内容整理答案，以老高的风格简短概括，在300字以内。"
    top_k: Optional[int] = 5
    threshold: Optional[float] = 0.85
    api_choice: Optional[str] = None
    api_url: Optional[str] = None
    api_token: Optional[str] = None

def log_request(request: Request, request_body: RequestBody):
    """Logs incoming request details to a file in append mode."""
    client_ip = request.client.host
    log_message = f"IP: {client_ip} | Query: {request_body.query}"
    logging.info(log_message)

@app.post("/api/search")
async def api_search(request: Request, request_body: RequestBody):
    try:
        log_request(request, request_body)

        # Perform FAISS search with timeout
        faiss_task = asyncio.create_task(
            search_faiss(
                request_body.query,
                top_k=request_body.top_k,
                min_threshold=request_body.threshold
            )
        )

        # Wait for FAISS results
        results, video_links, str_for_llm = await asyncio.wait_for(faiss_task, timeout=10.0)

        # Prepare prompt for LLM
        full_prompt = f"{request_body.prompt_mod}\n\n用户的问题:\n{request_body.query}\n\n召回的内容:\n{str_for_llm}"
        print("begin query llm")
        # Query LLM with timeout
        api_provider = SiliconFlowLLMAPI()
        llm_task = asyncio.create_task(api_provider.query_llm(full_prompt))
        llm_response = await asyncio.wait_for(llm_task, timeout=50.0)

        return {
            "search_results": results,
            "llm_response": llm_response,
            "video_links": video_links
        }

    except asyncio.TimeoutError:
        logging.error("Timeout occurred while processing request")
        raise HTTPException(status_code=504, detail="Request timeout, please try again.")
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, workers=4, timeout_keep_alive=60.0)



# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(
#         app,
#         host="0.0.0.0",
#         port=443,  # Use port 443 for HTTPS
#         ssl_keyfile="./ssl/key.pem",
#         ssl_certfile="./ssl/cert.pem"
#     )


