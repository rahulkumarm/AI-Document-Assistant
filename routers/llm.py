from fastapi import APIRouter, HTTPException, Query
from llm.llama_runner import get_llama_runner
from vector_store.chroma import get_chroma_store
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/llm", tags=["LLM"])

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repeat_penalty: Optional[float] = None
    stop: Optional[List[str]] = None

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None

class RAGRequest(BaseModel):
    query: str
    k: Optional[int] = 5
    filename_filter: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    system_message: Optional[str] = None

class GenerateResponse(BaseModel):
    response: str
    generation_time: Optional[float] = None
    prompt_length: int
    response_length: int

class RAGResponse(BaseModel):
    query: str
    response: str
    context_chunks: List[Dict[str, Any]]
    generation_time: Optional[float] = None

@router.get("/info")
async def get_llm_info():
    """
    Get information about the loaded LLM model.
    """
    try:
        llm_runner = get_llama_runner()
        return llm_runner.get_model_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting LLM info: {str(e)}")

@router.post("/test")
async def test_llm():
    """
    Test the LLM with a simple prompt.
    """
    try:
        llm_runner = get_llama_runner()
        return llm_runner.test_generation()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error testing LLM: {str(e)}")

@router.post("/generate", response_model=GenerateResponse)
async def generate_response(request: GenerateRequest):
    """
    Generate a response from the LLM given a prompt.
    """
    try:
        llm_runner = get_llama_runner()
        
        import time
        start_time = time.time()
        
        response = llm_runner.generate_response(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repeat_penalty=request.repeat_penalty,
            stop=request.stop
        )
        
        generation_time = time.time() - start_time
        
        return GenerateResponse(
            response=response,
            generation_time=generation_time,
            prompt_length=len(request.prompt),
            response_length=len(response)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@router.post("/chat", response_model=GenerateResponse)
async def chat_response(request: ChatRequest):
    """
    Generate a response using chat format.
    """
    try:
        llm_runner = get_llama_runner()
        
        # Convert Pydantic models to dict
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        import time
        start_time = time.time()
        
        response = llm_runner.generate_chat_response(
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        generation_time = time.time() - start_time
        
        # Calculate total prompt length
        prompt_length = sum(len(msg.content) for msg in request.messages)
        
        return GenerateResponse(
            response=response,
            generation_time=generation_time,
            prompt_length=prompt_length,
            response_length=len(response)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating chat response: {str(e)}")

@router.post("/rag", response_model=RAGResponse)
async def rag_query(request: RAGRequest):
    """
    Perform RAG (Retrieval-Augmented Generation) query.
    Retrieves relevant context from vector store and generates response.
    """
    try:
        # Get services
        llm_runner = get_llama_runner()
        chroma_store = get_chroma_store()
        
        # Retrieve relevant context
        where_filter = None
        if request.filename_filter:
            where_filter = {"filename": request.filename_filter}
        
        context_results = chroma_store.query_similar_chunks(
            query=request.query,
            k=request.k,
            where=where_filter
        )
        
        if not context_results:
            raise HTTPException(status_code=404, detail="No relevant context found in vector store")
        
        # Extract text chunks for context
        context_chunks = [result["document"] for result in context_results]
        
        # Create RAG prompt
        rag_prompt = llm_runner.create_rag_prompt(
            query=request.query,
            context_chunks=context_chunks,
            system_message=request.system_message
        )
        
        # Generate response
        import time
        start_time = time.time()
        
        response = llm_runner.generate_response(
            prompt=rag_prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        generation_time = time.time() - start_time
        
        return RAGResponse(
            query=request.query,
            response=response,
            context_chunks=context_results,
            generation_time=generation_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing RAG query: {str(e)}")

@router.post("/rag/simple")
async def simple_rag_query(
    query: str = Query(..., description="Question to ask"),
    k: int = Query(5, description="Number of context chunks to retrieve"),
    filename_filter: str = Query(None, description="Filter by filename"),
    max_tokens: int = Query(300, description="Maximum tokens to generate")
):
    """
    Simple RAG query endpoint with query parameters.
    """
    try:
        request = RAGRequest(
            query=query,
            k=k,
            filename_filter=filename_filter,
            max_tokens=max_tokens
        )
        return await rag_query(request)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in simple RAG query: {str(e)}")

@router.get("/models/available")
async def list_available_models():
    """
    List available GGUF models in the models directory.
    """
    try:
        import os
        models_dir = os.getenv("MODELS_DIR", "./models")
        
        if not os.path.exists(models_dir):
            return {"models": [], "models_dir": models_dir, "exists": False}
        
        # Find GGUF files
        gguf_files = []
        for file in os.listdir(models_dir):
            if file.endswith(('.gguf', '.bin')):
                file_path = os.path.join(models_dir, file)
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                gguf_files.append({
                    "filename": file,
                    "path": file_path,
                    "size_mb": round(file_size, 2)
                })
        
        return {
            "models": gguf_files,
            "models_dir": models_dir,
            "exists": True,
            "total_models": len(gguf_files)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}") 