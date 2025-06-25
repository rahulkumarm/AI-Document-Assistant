from vector_store.chroma import get_chroma_store
from llm.llama_runner import get_llama_runner
from typing import Optional, Dict, Any, List
import logging
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self):
        """
        Initialize the RAG pipeline with vector store and LLM components.
        """
        self.chroma_store = get_chroma_store()
        self.llm_runner = get_llama_runner()
        
        # Default configuration from environment
        self.default_k = int(os.getenv("RAG_DEFAULT_K", "3"))
        self.default_max_tokens = int(os.getenv("RAG_MAX_TOKENS", "500"))
        self.default_temperature = float(os.getenv("RAG_TEMPERATURE", "0.7"))
        
        logger.info("RAG Pipeline initialized successfully")
    
    def answer_question(
        self, 
        user_query: str, 
        file_id: str,
        k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        include_context_info: bool = False
    ) -> str:
        """
        Answer a question based on document context using RAG pipeline.
        
        Args:
            user_query (str): The user's question
            file_id (str): The filename/document ID to search within
            k (int, optional): Number of context chunks to retrieve
            max_tokens (int, optional): Maximum tokens for LLM response
            temperature (float, optional): Temperature for LLM generation
            include_context_info (bool): Whether to include context metadata in response
        
        Returns:
            str: Generated answer based on document context
        """
        if not user_query.strip():
            raise ValueError("User query cannot be empty")
        
        if not file_id.strip():
            raise ValueError("File ID cannot be empty")
        
        # Use defaults if not provided
        k = k or self.default_k
        max_tokens = max_tokens or self.default_max_tokens
        temperature = temperature or self.default_temperature
        
        logger.info(f"Processing RAG query: '{user_query[:100]}...' for file: {file_id}")
        logger.info(f"Parameters: k={k}, max_tokens={max_tokens}, temperature={temperature}")
        
        try:
            # Step 1: Retrieve relevant context chunks
            context_results = self.chroma_store.query_similar_chunks(
                query=user_query,
                k=k,
                where={"filename": file_id}
            )
            
            if not context_results:
                logger.warning(f"No context found for file: {file_id}")
                return f"I couldn't find any relevant information in the document '{file_id}' to answer your question. Please make sure the document has been uploaded and processed."
            
            # Step 2: Extract text chunks for context
            context_chunks = [result["document"] for result in context_results]
            
            logger.info(f"Retrieved {len(context_chunks)} context chunks")
            
            # Step 3: Format the research paper prompt
            formatted_prompt = self._format_research_paper_prompt(user_query, context_chunks)
            
            # Step 4: Generate response using LLM
            response = self.llm_runner.generate_response(
                prompt=formatted_prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            logger.info(f"Generated response: {len(response)} characters")
            
            # Optionally include context information
            if include_context_info:
                context_info = self._format_context_info(context_results)
                response += f"\n\n---\nContext sources: {context_info}"
            
            return response
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {str(e)}")
            raise Exception(f"Failed to answer question: {str(e)}")
    
    def _format_research_paper_prompt(self, user_query: str, context_chunks: List[str]) -> str:
        """
        Format the prompt specifically for research paper context.
        
        Args:
            user_query (str): The user's question
            context_chunks (List[str]): List of relevant text chunks
        
        Returns:
            str: Formatted prompt for the LLM
        """
        # Format context chunks
        formatted_context = ""
        for i, chunk in enumerate(context_chunks, 1):
            formatted_context += f"[chunk {i}]\n{chunk.strip()}\n\n"
        
        # Create the research paper prompt
        prompt = f"""You are reading a research paper. Based on the following context, answer the question clearly.

Context:
{formatted_context.strip()}

Question: {user_query}"""
        
        return prompt
    
    def _format_context_info(self, context_results: List[Dict[str, Any]]) -> str:
        """
        Format context information for inclusion in response.
        
        Args:
            context_results (List[Dict]): Context results from vector search
        
        Returns:
            str: Formatted context information
        """
        context_info = []
        for i, result in enumerate(context_results, 1):
            metadata = result.get("metadata", {})
            page_num = metadata.get("chunk_index", "unknown")
            similarity = result.get("similarity", 0)
            context_info.append(f"Chunk {i} (Page {page_num}, Similarity: {similarity:.3f})")
        
        return ", ".join(context_info)
    
    def answer_question_advanced(
        self,
        user_query: str,
        file_id: Optional[str] = None,
        k: int = 5,
        max_tokens: int = 500,
        temperature: float = 0.7,
        system_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Advanced RAG pipeline with detailed response information.
        
        Args:
            user_query (str): The user's question
            file_id (str, optional): Filter by specific file
            k (int): Number of context chunks to retrieve
            max_tokens (int): Maximum tokens for response
            temperature (float): Temperature for generation
            system_message (str, optional): Custom system message
        
        Returns:
            Dict: Detailed response with context and metadata
        """
        try:
            # Prepare filter
            where_filter = {"filename": file_id} if file_id else None
            
            # Retrieve context
            context_results = self.chroma_store.query_similar_chunks(
                query=user_query,
                k=k,
                where=where_filter
            )
            
            if not context_results:
                return {
                    "answer": "No relevant context found to answer the question.",
                    "context_chunks": [],
                    "metadata": {"status": "no_context", "file_id": file_id}
                }
            
            # Extract chunks
            context_chunks = [result["document"] for result in context_results]
            
            # Use custom system message or default
            if system_message:
                prompt = self.llm_runner.create_rag_prompt(
                    query=user_query,
                    context_chunks=context_chunks,
                    system_message=system_message
                )
            else:
                prompt = self._format_research_paper_prompt(user_query, context_chunks)
            
            # Generate response
            import time
            start_time = time.time()
            
            response = self.llm_runner.generate_response(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            generation_time = time.time() - start_time
            
            return {
                "answer": response,
                "context_chunks": context_results,
                "metadata": {
                    "status": "success",
                    "file_id": file_id,
                    "chunks_used": len(context_chunks),
                    "generation_time": generation_time,
                    "parameters": {
                        "k": k,
                        "max_tokens": max_tokens,
                        "temperature": temperature
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error in advanced RAG pipeline: {str(e)}")
            return {
                "answer": f"Error processing question: {str(e)}",
                "context_chunks": [],
                "metadata": {"status": "error", "error": str(e)}
            }
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about the RAG pipeline configuration.
        
        Returns:
            Dict: Pipeline configuration and status
        """
        try:
            # Get vector store stats
            vector_stats = self.chroma_store.get_collection_stats()
            
            # Get LLM info
            llm_info = self.llm_runner.get_model_info()
            
            return {
                "pipeline_status": "ready",
                "default_config": {
                    "k": self.default_k,
                    "max_tokens": self.default_max_tokens,
                    "temperature": self.default_temperature
                },
                "vector_store": vector_stats,
                "llm_model": {
                    "model_path": llm_info.get("model_path"),
                    "model_exists": llm_info.get("model_exists"),
                    "context_window": llm_info.get("n_ctx")
                }
            }
            
        except Exception as e:
            return {
                "pipeline_status": "error",
                "error": str(e)
            }

# Global RAG pipeline instance (singleton pattern)
_rag_pipeline: Optional[RAGPipeline] = None

def get_rag_pipeline() -> RAGPipeline:
    """
    Get or create a global RAG pipeline instance (singleton pattern).
    
    Returns:
        RAGPipeline: Global RAG pipeline instance
    """
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline 