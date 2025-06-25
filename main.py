from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from routers import pdf, llm, rag
from services.rag_pipeline import get_rag_pipeline
from pydantic import BaseModel
import os

# Load environment variables
load_dotenv()

app = FastAPI(
    title="AI Project Backend",
    description="FastAPI backend for AI project with PDF parsing, embedding, vector store, LLM, and RAG capabilities",
    version="1.0.0"
)

# Include routers
app.include_router(pdf.router)
app.include_router(llm.router)
app.include_router(rag.router)

class AskRequest(BaseModel):
    file_id: str
    question: str

class AskResponse(BaseModel):
    answer: str

@app.get("/")
async def hello_world():
    return {"message": "Hello World from AI Project Backend!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API is running"}

@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Ask a question about a specific document.
    This endpoint is used by the frontend ChatBox component.
    """
    try:
        rag_pipeline = get_rag_pipeline()
        
        answer = rag_pipeline.answer_question(
            user_query=request.question,
            file_id=request.file_id,
            k=3,
            max_tokens=500,
            temperature=0.7
        )
        
        return AskResponse(answer=answer)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "True").lower() == "true"
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug
    ) 