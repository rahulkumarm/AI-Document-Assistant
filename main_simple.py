from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import uuid

app = FastAPI(
    title="AI Project Backend (Simplified)",
    description="Simplified FastAPI backend for testing",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:5175", "http://localhost:5176"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    file_id: str
    question: str

class AskResponse(BaseModel):
    answer: str

class UploadResponse(BaseModel):
    document_id: str
    filename: str
    total_pages: int
    total_chunks: int
    embeddings_generated: bool
    message: str

@app.get("/")
async def hello_world():
    return {"message": "Hello World from AI Project Backend!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API is running"}

@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    save_to_vector_store: bool = Form(True)
):
    """
    Simplified upload endpoint for testing.
    Returns a mock response that matches the expected format.
    """
    try:
        print(f"Upload request received - filename: {file.filename}, content_type: {file.content_type}")
        
        # Validate file type
        if not file.filename or not file.filename.lower().endswith('.pdf'):
            print(f"Invalid file type: {file.filename}")
            raise HTTPException(status_code=422, detail="Only PDF files are allowed")
        
        # Read file content to ensure it's valid
        content = await file.read()
        print(f"File content length: {len(content)} bytes")
        
        # Reset file pointer for potential future use
        await file.seek(0)
        
        # Generate a mock document ID
        document_id = str(uuid.uuid4())
        print(f"Generated document ID: {document_id}")
        
        # Mock response for testing
        mock_response = UploadResponse(
            document_id=document_id,
            filename=file.filename,
            total_pages=5,  # Mock page count
            total_chunks=20,  # Mock chunk count
            embeddings_generated=save_to_vector_store,
            message=f"Successfully processed {file.filename}. Mock response - actual PDF processing will be connected once dependencies are resolved."
        )
        
        print(f"Returning successful response for {file.filename}")
        return mock_response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Simplified ask endpoint for testing.
    Returns a mock response for now.
    """
    try:
        # Mock response for testing
        mock_answer = f"This is a mock response to your question: '{request.question}' about document '{request.file_id}'. The actual RAG pipeline will be connected once the dependency issues are resolved."
        
        return AskResponse(answer=mock_answer)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_simple:app", host="0.0.0.0", port=8000, reload=True) 