from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from services.pdf_parser import PDFParser
from services.embedder import get_embedder
from vector_store.chroma import get_chroma_store
from models.pdf_models import PDFUploadResponse, DocumentListResponse, TextChunk
import os

router = APIRouter(prefix="/pdf", tags=["PDF"])

# Initialize services
pdf_parser = PDFParser()

@router.post("/upload", response_model=PDFUploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    generate_embeddings: bool = Query(False, description="Whether to generate embeddings for the text chunks"),
    save_to_vector_store: bool = Query(False, description="Whether to save embeddings to ChromaDB vector store")
):
    """
    Upload a PDF file and extract text content by page.
    Optionally generate embeddings and save to vector store.
    """
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    if file.content_type != 'application/pdf':
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF file")
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Parse PDF and extract text
        text_chunks = pdf_parser.extract_text_from_pdf(file_content, file.filename)
        
        # Generate embeddings if requested
        embeddings = None
        if generate_embeddings and text_chunks:
            embedder = get_embedder()
            text_only = [chunk["text"] for chunk in text_chunks]
            embeddings = embedder.generate_embeddings(text_only)
            
            # Add embeddings to text chunks
            for i, chunk in enumerate(text_chunks):
                chunk["embedding"] = embeddings[i]
        
        # Save to vector store if requested and embeddings were generated
        saved_ids = None
        if save_to_vector_store and embeddings:
            chroma_store = get_chroma_store()
            text_only = [chunk["text"] for chunk in text_chunks]
            
            # Prepare metadata
            metadata = {
                "filename": file.filename,
                "upload_timestamp": text_chunks[0].get("upload_timestamp") if text_chunks else None,
                "source": "pdf_upload",
                "total_pages": len(text_chunks)
            }
            
            saved_ids = chroma_store.save_embeddings(text_only, embeddings, metadata)
        
        # Convert to Pydantic models
        text_chunk_models = [TextChunk(**chunk) for chunk in text_chunks]
        
        response_data = {
            "message": "PDF uploaded and processed successfully",
            "filename": file.filename,
            "total_pages": len(text_chunks),
            "text_chunks": text_chunk_models
        }
        
        # Add vector store info if saved
        if saved_ids:
            response_data["message"] += f" and saved to vector store"
            response_data["vector_store_ids"] = saved_ids
        
        return PDFUploadResponse(**response_data)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@router.post("/generate-embeddings/{filename}")
async def generate_embeddings_for_document(
    filename: str,
    save_to_vector_store: bool = Query(False, description="Whether to save embeddings to ChromaDB vector store")
):
    """
    Generate embeddings for a previously uploaded document.
    """
    try:
        # Get document text chunks
        text_chunks = pdf_parser.get_document_text_chunks(filename)
        
        if not text_chunks:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Extract text from chunks
        text_only = [chunk["text"] for chunk in text_chunks]
        
        # Generate embeddings
        embedder = get_embedder()
        embeddings = embedder.generate_embeddings(text_only)
        
        # Save to vector store if requested
        saved_ids = None
        if save_to_vector_store:
            chroma_store = get_chroma_store()
            
            # Prepare metadata
            metadata = {
                "filename": filename,
                "source": "generate_embeddings_endpoint",
                "total_chunks": len(text_chunks)
            }
            
            saved_ids = chroma_store.save_embeddings(text_only, embeddings, metadata)
        
        response_data = {
            "message": f"Generated embeddings for {filename}",
            "filename": filename,
            "total_chunks": len(embeddings),
            "embedding_dimension": len(embeddings[0]) if embeddings else 0,
            "embeddings": embeddings
        }
        
        if saved_ids:
            response_data["message"] += " and saved to vector store"
            response_data["vector_store_ids"] = saved_ids
        
        return response_data
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")

@router.post("/search")
async def search_similar_chunks(
    query: str = Query(..., description="Search query text"),
    k: int = Query(5, description="Number of similar chunks to return"),
    filename_filter: str = Query(None, description="Filter results by filename")
):
    """
    Search for similar text chunks in the vector store.
    """
    try:
        chroma_store = get_chroma_store()
        
        # Prepare metadata filter if filename is specified
        where_filter = None
        if filename_filter:
            where_filter = {"filename": filename_filter}
        
        # Query similar chunks
        results = chroma_store.query_similar_chunks(
            query=query, 
            k=k, 
            where=where_filter
        )
        
        return {
            "query": query,
            "total_results": len(results),
            "results": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching chunks: {str(e)}")

@router.get("/vector-store/stats")
async def get_vector_store_stats():
    """
    Get statistics about the vector store.
    """
    try:
        chroma_store = get_chroma_store()
        stats = chroma_store.get_collection_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting vector store stats: {str(e)}")

@router.delete("/vector-store/clear")
async def clear_vector_store():
    """
    Clear all data from the vector store.
    """
    try:
        chroma_store = get_chroma_store()
        chroma_store.clear_collection()
        return {"message": "Vector store cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing vector store: {str(e)}")

@router.delete("/vector-store/document/{filename}")
async def delete_document_from_vector_store(filename: str):
    """
    Delete all chunks for a specific document from the vector store.
    """
    try:
        chroma_store = get_chroma_store()
        deleted_count = chroma_store.delete_chunks_by_metadata({"filename": filename})
        return {
            "message": f"Deleted {deleted_count} chunks for document '{filename}'",
            "deleted_chunks": deleted_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document from vector store: {str(e)}")

@router.get("/embedder/info")
async def get_embedder_info():
    """
    Get information about the embedder model.
    """
    try:
        embedder = get_embedder()
        return embedder.get_model_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting embedder info: {str(e)}")

@router.get("/documents", response_model=DocumentListResponse)
async def get_stored_documents():
    """
    Get list of stored documents and their metadata.
    """
    documents = pdf_parser.get_stored_documents()
    return DocumentListResponse(
        total_documents=len(documents),
        documents=documents
    ) 