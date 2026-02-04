from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
import uvicorn
import os

from rlm.rlm import RLM, RLMConfig

app = FastAPI(title="RLM API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    context: Union[str, List[str]]
    config: Optional[Dict[str, Any]] = None

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/query")
async def rlm_query(request: QueryRequest):
    try:
        # Initialize config with provided overrides
        config_data = request.config or {}
        config = RLMConfig(**config_data)
        
        rlm = RLM(config=config)
        result = rlm.query(query=request.query, context=request.context, verbose=True)
        
        return {
            "answer": result.answer,
            "success": result.success,
            "iterations": result.iterations,
            "total_cost": result.total_cost,
            "usage_summary": result.usage_summary,
            "trajectory": result.trajectory,
            "error": result.error
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text = content.decode("utf-8")
        return {"filename": file.filename, "content": text, "length": len(text)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read file: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
