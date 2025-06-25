import fitz  # PyMuPDF
from typing import List, Dict, Any
from datetime import datetime
import io

class PDFParser:
    def __init__(self):
        # In-memory storage for parsed documents
        self.stored_documents: Dict[str, Dict[str, Any]] = {}
    
    def extract_text_from_pdf(self, file_content: bytes, filename: str) -> List[str]:
        """
        Extract text from PDF file content by page.
        
        Args:
            file_content (bytes): PDF file content as bytes
            filename (str): Name of the PDF file
        
        Returns:
            List[str]: List of text chunks, one per page
        """
        try:
            # Open PDF from bytes
            pdf_document = fitz.open(stream=file_content, filetype="pdf")
            
            text_chunks = []
            page_count = len(pdf_document)
            
            # Extract text from each page
            for page_num in range(page_count):
                page = pdf_document.load_page(page_num)
                text = page.get_text()
                
                # Clean up the text (remove extra whitespace)
                cleaned_text = " ".join(text.split())
                
                if cleaned_text.strip():  # Only add non-empty pages
                    text_chunks.append({
                        "page_number": page_num + 1,
                        "text": cleaned_text
                    })
            
            # Close the document
            pdf_document.close()
            
            # Store the document in memory
            self._store_document(filename, text_chunks, page_count)
            
            return text_chunks
            
        except Exception as e:
            raise Exception(f"Failed to extract text from PDF: {str(e)}")
    
    def _store_document(self, filename: str, text_chunks: List[Dict[str, Any]], page_count: int):
        """
        Store document information in memory.
        
        Args:
            filename (str): Name of the PDF file
            text_chunks (List[Dict]): Extracted text chunks with page numbers
            page_count (int): Total number of pages in the document
        """
        self.stored_documents[filename] = {
            "filename": filename,
            "upload_timestamp": datetime.now().isoformat(),
            "total_pages": page_count,
            "total_chunks": len(text_chunks),
            "text_chunks": text_chunks
        }
    
    def get_stored_documents(self) -> List[Dict[str, Any]]:
        """
        Get list of all stored documents with metadata.
        
        Returns:
            List[Dict]: List of document metadata (without full text content)
        """
        documents = []
        for filename, doc_data in self.stored_documents.items():
            documents.append({
                "filename": doc_data["filename"],
                "upload_timestamp": doc_data["upload_timestamp"],
                "total_pages": doc_data["total_pages"],
                "total_chunks": doc_data["total_chunks"]
            })
        return documents
    
    def get_document_by_filename(self, filename: str) -> Dict[str, Any]:
        """
        Get a specific document by filename.
        
        Args:
            filename (str): Name of the PDF file
        
        Returns:
            Dict: Document data including text chunks
        """
        return self.stored_documents.get(filename)
    
    def get_document_text_chunks(self, filename: str) -> List[Dict[str, Any]]:
        """
        Get text chunks for a specific document.
        
        Args:
            filename (str): Name of the PDF file
        
        Returns:
            List[Dict]: Text chunks with page numbers
        """
        doc_data = self.stored_documents.get(filename)
        if doc_data:
            return doc_data["text_chunks"]
        return []
    
    def clear_all_documents(self):
        """
        Clear all stored documents from memory.
        """
        self.stored_documents.clear() 