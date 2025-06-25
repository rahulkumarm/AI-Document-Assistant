from fastapi import APIRouter, HTTPException, Query
from services.rag_pipeline import get_rag_pipeline
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["RAG Pipeline"])

class QuestionRequest(BaseModel):
    user_query: str
    file_id: str
    k: Optional[int] = 3
    max_tokens: Optional[int] = 500
    temperature: Optional[float] = 0.7
    include_context_info: Optional[bool] = False

class AdvancedQuestionRequest(BaseModel):
    user_query: str
    file_id: Optional[str] = None
    k: Optional[int] = 5
    max_tokens: Optional[int] = 500
    temperature: Optional[float] = 0.7
    system_message: Optional[str] = None

class QuestionResponse(BaseModel):
    answer: str
    file_id: str
    metadata: Optional[Dict[str, Any]] = None

class AdvancedQuestionResponse(BaseModel):
    answer: str
    context_chunks: list
    metadata: Dict[str, Any]

@router.post("/question", response_model=QuestionResponse)
async def answer_question(request: QuestionRequest):
    """
    Answer a question based on a specific document using RAG pipeline.
    Uses the research paper prompt format.
    """
    try:
        rag_pipeline = get_rag_pipeline()
        
        answer = rag_pipeline.answer_question(
            user_query=request.user_query,
            file_id=request.file_id,
            k=request.k,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            include_context_info=request.include_context_info
        )
        
        return QuestionResponse(
            answer=answer,
            file_id=request.file_id,
            metadata={
                "k": request.k,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature
            }
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")

@router.post("/question/simple")
async def simple_question(
    user_query: str = Query(..., description="The question to ask"),
    file_id: str = Query(..., description="The document filename to search in"),
    k: int = Query(3, description="Number of context chunks to retrieve"),
    max_tokens: int = Query(500, description="Maximum tokens for response"),
    temperature: float = Query(0.7, description="Temperature for generation")
):
    """
    Simple question endpoint using query parameters.
    """
    try:
        request = QuestionRequest(
            user_query=user_query,
            file_id=file_id,
            k=k,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return await answer_question(request)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in simple question: {str(e)}")

@router.post("/question/advanced", response_model=AdvancedQuestionResponse)
async def answer_question_advanced(request: AdvancedQuestionRequest):
    """
    Advanced question answering with detailed response information.
    Can search across all documents or filter by specific file.
    """
    try:
        rag_pipeline = get_rag_pipeline()
        
        result = rag_pipeline.answer_question_advanced(
            user_query=request.user_query,
            file_id=request.file_id,
            k=request.k,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            system_message=request.system_message
        )
        
        return AdvancedQuestionResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in advanced question: {str(e)}")

@router.get("/info")
async def get_rag_info():
    """
    Get information about the RAG pipeline configuration and status.
    """
    try:
        rag_pipeline = get_rag_pipeline()
        return rag_pipeline.get_pipeline_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting RAG info: {str(e)}")

@router.post("/batch/questions")
async def batch_questions(
    questions: list,
    file_id: str = Query(..., description="Document to search in"),
    k: int = Query(3, description="Context chunks per question"),
    max_tokens: int = Query(300, description="Max tokens per answer")
):
    """
    Answer multiple questions for the same document in batch.
    """
    try:
        if not questions:
            raise HTTPException(status_code=400, detail="No questions provided")
        
        if len(questions) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 questions per batch")
        
        rag_pipeline = get_rag_pipeline()
        results = []
        
        for i, question in enumerate(questions):
            try:
                answer = rag_pipeline.answer_question(
                    user_query=question,
                    file_id=file_id,
                    k=k,
                    max_tokens=max_tokens
                )
                results.append({
                    "question_index": i,
                    "question": question,
                    "answer": answer,
                    "status": "success"
                })
            except Exception as e:
                results.append({
                    "question_index": i,
                    "question": question,
                    "answer": f"Error: {str(e)}",
                    "status": "error"
                })
        
        return {
            "file_id": file_id,
            "total_questions": len(questions),
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in batch questions: {str(e)}")

@router.post("/compare/documents")
async def compare_documents_question(
    user_query: str = Query(..., description="Question to ask across documents"),
    file_ids: list = Query(..., description="List of document filenames to compare"),
    k: int = Query(3, description="Context chunks per document")
):
    """
    Ask the same question across multiple documents and compare answers.
    """
    try:
        if not file_ids:
            raise HTTPException(status_code=400, detail="No file IDs provided")
        
        if len(file_ids) > 5:
            raise HTTPException(status_code=400, detail="Maximum 5 documents for comparison")
        
        rag_pipeline = get_rag_pipeline()
        results = []
        
        for file_id in file_ids:
            try:
                answer = rag_pipeline.answer_question(
                    user_query=user_query,
                    file_id=file_id,
                    k=k,
                    max_tokens=400
                )
                results.append({
                    "file_id": file_id,
                    "answer": answer,
                    "status": "success"
                })
            except Exception as e:
                results.append({
                    "file_id": file_id,
                    "answer": f"Error: {str(e)}",
                    "status": "error"
                })
        
        return {
            "question": user_query,
            "documents_compared": len(file_ids),
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comparing documents: {str(e)}") 