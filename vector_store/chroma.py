import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import os
import uuid
from datetime import datetime
from dotenv import load_dotenv
import logging
from services.embedder import get_embedder

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChromaVectorStore:
    def __init__(self, persist_directory: Optional[str] = None, collection_name: Optional[str] = None):
        """
        Initialize ChromaDB vector store.
        
        Args:
            persist_directory (str, optional): Directory to persist ChromaDB data
            collection_name (str, optional): Name of the collection to use
        """
        # Set default values from environment or use defaults
        self.persist_directory = persist_directory or os.getenv("CHROMA_DB_PATH", "./chroma_db")
        self.collection_name = collection_name or os.getenv("COLLECTION_NAME", "documents")
        
        logger.info(f"Initializing ChromaDB at: {self.persist_directory}")
        logger.info(f"Collection name: {self.collection_name}")
        
        try:
            # Create ChromaDB client with persistence
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Document chunks with embeddings"}
            )
            
            logger.info(f"ChromaDB initialized successfully")
            logger.info(f"Collection '{self.collection_name}' ready")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise Exception(f"Failed to initialize ChromaDB: {str(e)}")
    
    def save_embeddings(
        self, 
        chunks: List[str], 
        embeddings: List[List[float]], 
        metadata: Dict[str, Any]
    ) -> List[str]:
        """
        Save text chunks with their embeddings to ChromaDB.
        
        Args:
            chunks (List[str]): List of text chunks
            embeddings (List[List[float]]): List of embedding vectors
            metadata (Dict[str, Any]): Metadata to associate with the chunks
        
        Returns:
            List[str]: List of generated IDs for the saved chunks
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        if not chunks:
            return []
        
        logger.info(f"Saving {len(chunks)} chunks to ChromaDB")
        
        try:
            # Generate unique IDs for each chunk
            ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
            
            # Prepare metadata for each chunk
            chunk_metadata = []
            for i, chunk in enumerate(chunks):
                chunk_meta = {
                    **metadata,  # Include common metadata
                    "chunk_index": i,
                    "chunk_length": len(chunk),
                    "timestamp": datetime.now().isoformat(),
                    "chunk_id": ids[i]
                }
                chunk_metadata.append(chunk_meta)
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=chunks,
                metadatas=chunk_metadata,
                ids=ids
            )
            
            logger.info(f"Successfully saved {len(chunks)} chunks to ChromaDB")
            return ids
            
        except Exception as e:
            logger.error(f"Failed to save embeddings: {str(e)}")
            raise Exception(f"Failed to save embeddings: {str(e)}")
    
    def query_similar_chunks(
        self, 
        query: str, 
        k: int = 5, 
        where: Optional[Dict[str, Any]] = None,
        include_distances: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Query for similar chunks using text query.
        
        Args:
            query (str): Query text to search for similar chunks
            k (int): Number of similar chunks to return. Defaults to 5
            where (Dict, optional): Metadata filter conditions
            include_distances (bool): Whether to include similarity distances
        
        Returns:
            List[Dict]: List of similar chunks with metadata and distances
        """
        if not query.strip():
            return []
        
        logger.info(f"Querying for {k} similar chunks")
        logger.info(f"Query: {query[:100]}...")
        
        try:
            # Generate embedding for the query
            embedder = get_embedder()
            query_embedding = embedder.generate_single_embedding(query)
            
            # Prepare query parameters
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": k
            }
            
            # Add metadata filter if provided
            if where:
                query_params["where"] = where
            
            # Include what we want in results
            include_list = ["documents", "metadatas", "ids"]
            if include_distances:
                include_list.append("distances")
            
            query_params["include"] = include_list
            
            # Query the collection
            results = self.collection.query(**query_params)
            
            # Format results
            formatted_results = []
            for i in range(len(results["ids"][0])):
                result = {
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i]
                }
                
                if include_distances and "distances" in results:
                    result["distance"] = results["distances"][0][i]
                    result["similarity"] = 1 - results["distances"][0][i]  # Convert distance to similarity
                
                formatted_results.append(result)
            
            logger.info(f"Found {len(formatted_results)} similar chunks")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to query similar chunks: {str(e)}")
            raise Exception(f"Failed to query similar chunks: {str(e)}")
    
    def query_by_embedding(
        self, 
        query_embedding: List[float], 
        k: int = 5, 
        where: Optional[Dict[str, Any]] = None,
        include_distances: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Query for similar chunks using embedding vector.
        
        Args:
            query_embedding (List[float]): Query embedding vector
            k (int): Number of similar chunks to return
            where (Dict, optional): Metadata filter conditions
            include_distances (bool): Whether to include similarity distances
        
        Returns:
            List[Dict]: List of similar chunks with metadata and distances
        """
        logger.info(f"Querying by embedding vector for {k} similar chunks")
        
        try:
            # Prepare query parameters
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": k
            }
            
            if where:
                query_params["where"] = where
            
            include_list = ["documents", "metadatas", "ids"]
            if include_distances:
                include_list.append("distances")
            
            query_params["include"] = include_list
            
            # Query the collection
            results = self.collection.query(**query_params)
            
            # Format results
            formatted_results = []
            for i in range(len(results["ids"][0])):
                result = {
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i]
                }
                
                if include_distances and "distances" in results:
                    result["distance"] = results["distances"][0][i]
                    result["similarity"] = 1 - results["distances"][0][i]
                
                formatted_results.append(result)
            
            logger.info(f"Found {len(formatted_results)} similar chunks")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to query by embedding: {str(e)}")
            raise Exception(f"Failed to query by embedding: {str(e)}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dict: Collection statistics
        """
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "total_chunks": count,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            return {"error": str(e)}
    
    def delete_chunks_by_metadata(self, where: Dict[str, Any]) -> int:
        """
        Delete chunks based on metadata filter.
        
        Args:
            where (Dict): Metadata filter conditions
        
        Returns:
            int: Number of chunks deleted
        """
        try:
            # Get chunks matching the filter first
            results = self.collection.get(where=where, include=["ids"])
            chunk_ids = results["ids"]
            
            if chunk_ids:
                # Delete the chunks
                self.collection.delete(ids=chunk_ids)
                logger.info(f"Deleted {len(chunk_ids)} chunks")
                return len(chunk_ids)
            else:
                logger.info("No chunks found matching the filter")
                return 0
                
        except Exception as e:
            logger.error(f"Failed to delete chunks: {str(e)}")
            raise Exception(f"Failed to delete chunks: {str(e)}")
    
    def clear_collection(self):
        """
        Clear all data from the collection.
        """
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Document chunks with embeddings"}
            )
            logger.info(f"Collection '{self.collection_name}' cleared")
            
        except Exception as e:
            logger.error(f"Failed to clear collection: {str(e)}")
            raise Exception(f"Failed to clear collection: {str(e)}")

# Global ChromaDB instance (singleton pattern)
_chroma_instance: Optional[ChromaVectorStore] = None

def get_chroma_store() -> ChromaVectorStore:
    """
    Get or create a global ChromaDB instance (singleton pattern).
    
    Returns:
        ChromaVectorStore: Global ChromaDB instance
    """
    global _chroma_instance
    if _chroma_instance is None:
        _chroma_instance = ChromaVectorStore()
    return _chroma_instance 