from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import pandas as pd
import uvicorn
import os
import logging
import time

# Configure FastAPI-specific logging
fastapi_logger = logging.getLogger("fastapi")
fastapi_logger.setLevel(logging.INFO)
fastapi_handler = logging.FileHandler('fastapi_rag.log')
fastapi_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
fastapi_logger.addHandler(fastapi_handler)

# Initialize your existing RAG system
# from advanced_rag import RAGSystem, RAGConfig, preprocess_sales_data  # Import your actual module
from ai_based_rag import RAGSystem, RAGConfig, preprocess_sales_data
app = FastAPI(
    title="Sales Data RAG API",
    description="API for querying structured sales data using RAG pipeline",
    version="1.0.0",
    openapi_url="/openapi.json"
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG system on startup
rag_system = None

class QueryRequest(BaseModel):
    question: str = Field(..., description="The question to ask about sales data")
    collection_name: str = Field("sales_data_v2", description="Qdrant collection name")
    search_limit: int = Field(10, description="Number of context chunks to retrieve")
    model: str = Field("llama3-70b-8192", description="LLM model to use for generation")
    temperature: float = Field(0.1, description="LLM temperature parameter")
    max_tokens: int = Field(1024, description="Max tokens for LLM response")

class QueryResponse(BaseModel):
    success: bool
    answer: Optional[str] = None
    context: Optional[List[str]] = None
    search_results: Optional[List[dict]] = None
    execution_time: float
    error: Optional[str] = None

class IngestRequest(BaseModel):
    file_path: str = Field(..., description="Path to CSV file to ingest")
    collection_name: str = Field("sales_data_v2", description="Collection to ingest into")

class IngestResponse(BaseModel):
    success: bool
    message: str
    document_count: int = 0
    execution_time: float

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    global rag_system
    try:
        config = RAGConfig(
            qdrant_url=os.getenv('QDRANT_URL'),
            qdrant_api_key=os.getenv('QDRANT_API_KEY'),
            groq_api_key=os.getenv('GROQ_API_KEY')
        )
        rag_system = RAGSystem(config)
        fastapi_logger.info("RAG system initialized successfully")
    except Exception as e:
        fastapi_logger.critical(f"Failed to initialize RAG system: {str(e)}")
        raise RuntimeError("System initialization failed") from e

@app.post("/query", response_model=QueryResponse, summary="Query the RAG system")
async def query_rag_system(request: QueryRequest = Body(...)):
    """Query the RAG system with a natural language question"""
    start_time = time.time()
    response = {
        "success": False,
        "answer": None,
        "context": None,
        "search_results": None,
        "execution_time": 0,
        "error": None
    }
    
    try:
        if not rag_system:
            response["error"] = "RAG system not initialized"
            return response
        
        result = rag_system.query(
            collection_name=request.collection_name,
            question=request.question,
            search_limit=request.search_limit,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        response.update(result)
        fastapi_logger.info(f"Query successful: '{request.question}'")
        return response
    except Exception as e:
        response["error"] = f"Query processing failed: {str(e)}"
        response["execution_time"] = time.time() - start_time
        fastapi_logger.error(f"Query failed: {request.question} - {str(e)}")
        return response

@app.post("/ingest", response_model=IngestResponse, summary="Ingest new sales data")
async def ingest_data(request: IngestRequest = Body(...)):
    """Ingest new sales data from a CSV file"""
    start_time = time.time()
    response = {
        "success": False,
        "message": "",
        "document_count": 0,
        "execution_time": 0
    }
    
    try:
        if not rag_system:
            response["message"] = "RAG system not initialized"
            return response
        
        # Load and preprocess data
        try:
            sales_df = pd.read_csv(request.file_path)
            documents, metadata = preprocess_sales_data(sales_df)
            response["document_count"] = len(documents)
        except Exception as e:
            response["message"] = f"Data loading failed: {str(e)}"
            return response
        
        # Add documents to vector store
        success = rag_system.add_documents(
            collection_name=request.collection_name,
            documents=documents,
            metadata=metadata
        )
        
        if success:
            response["success"] = True
            response["message"] = f"Ingested {len(documents)} documents into '{request.collection_name}'"
            fastapi_logger.info(response["message"])
        else:
            response["message"] = "Document ingestion failed"
            fastapi_logger.error(response["message"])
        
        return response
    except Exception as e:
        response["message"] = f"Ingestion failed: {str(e)}"
        fastapi_logger.error(response["message"])
        return response
    finally:
        response["execution_time"] = time.time() - start_time

@app.get("/collections", summary="List available collections")
async def list_collections():
    """Get list of available collections in Qdrant"""
    try:
        if not rag_system:
            return {"error": "RAG system not initialized"}
        collections = rag_system.list_collections()
        return {"collections": collections}
    except Exception as e:
        fastapi_logger.error(f"Failed to list collections: {str(e)}")
        return {"error": str(e)}

@app.get("/collection/{collection_name}", summary="Get collection info")
async def get_collection_info(collection_name: str):
    """Get information about a specific collection"""
    try:
        if not rag_system:
            return {"error": "RAG system not initialized"}
        info = rag_system.get_collection_info(collection_name)
        if not info:
            return JSONResponse(
                status_code=404,
                content={"error": f"Collection '{collection_name}' not found"}
            )
        return info
    except Exception as e:
        fastapi_logger.error(f"Failed to get collection info: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_config=None,
        access_log=False
    )