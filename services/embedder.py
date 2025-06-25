from sentence_transformers import SentenceTransformer
from typing import List, Optional
import torch
import numpy as np
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Embedder:
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the embedder with a sentence-transformer model.
        
        Args:
            model_name (str, optional): HuggingFace model name. Defaults to BAAI/bge-base-en
            device (str, optional): Device to use ('cuda', 'cpu', 'mps'). Auto-detected if None
        """
        # Set default model from environment or use BAAI/bge-base-en
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en")
        
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():  # Apple Silicon
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        logger.info(f"Initializing embedder with model: {self.model_name}")
        logger.info(f"Using device: {self.device}")
        
        # Load the model
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("Embedder model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise Exception(f"Failed to initialize embedder: {str(e)}")
    
    def generate_embeddings(
        self, 
        text_chunks: List[str], 
        batch_size: int = 32,
        show_progress: bool = True,
        normalize_embeddings: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of text chunks with batching for efficiency.
        
        Args:
            text_chunks (List[str]): List of text strings to embed
            batch_size (int): Batch size for processing. Defaults to 32
            show_progress (bool): Whether to show progress bar. Defaults to True
            normalize_embeddings (bool): Whether to normalize embeddings. Defaults to True
        
        Returns:
            List[List[float]]: List of embedding vectors
        """
        if not text_chunks:
            return []
        
        logger.info(f"Generating embeddings for {len(text_chunks)} text chunks")
        logger.info(f"Batch size: {batch_size}")
        
        try:
            # Generate embeddings with batching
            embeddings = self.model.encode(
                text_chunks,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=normalize_embeddings,
                convert_to_numpy=True
            )
            
            # Convert numpy arrays to lists for JSON serialization
            embeddings_list = embeddings.tolist()
            
            logger.info(f"Successfully generated {len(embeddings_list)} embeddings")
            logger.info(f"Embedding dimension: {len(embeddings_list[0]) if embeddings_list else 0}")
            
            return embeddings_list
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise Exception(f"Failed to generate embeddings: {str(e)}")
    
    def generate_single_embedding(self, text: str, normalize: bool = True) -> List[float]:
        """
        Generate embedding for a single text string.
        
        Args:
            text (str): Text string to embed
            normalize (bool): Whether to normalize the embedding. Defaults to True
        
        Returns:
            List[float]: Embedding vector
        """
        try:
            embedding = self.model.encode(
                [text],
                normalize_embeddings=normalize,
                convert_to_numpy=True
            )
            return embedding[0].tolist()
            
        except Exception as e:
            logger.error(f"Failed to generate single embedding: {str(e)}")
            raise Exception(f"Failed to generate single embedding: {str(e)}")
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by the model.
        
        Returns:
            int: Embedding dimension
        """
        return self.model.get_sentence_embedding_dimension()
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            dict: Model information including name, device, and embedding dimension
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "embedding_dimension": self.get_embedding_dimension(),
            "max_seq_length": getattr(self.model, 'max_seq_length', 'Unknown')
        }
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1 (List[float]): First embedding vector
            embedding2 (List[float]): Second embedding vector
        
        Returns:
            float: Cosine similarity score (-1 to 1)
        """
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Compute cosine similarity
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Failed to compute similarity: {str(e)}")
            raise Exception(f"Failed to compute similarity: {str(e)}")

# Global embedder instance (singleton pattern)
_embedder_instance: Optional[Embedder] = None

def get_embedder() -> Embedder:
    """
    Get or create a global embedder instance (singleton pattern).
    
    Returns:
        Embedder: Global embedder instance
    """
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = Embedder()
    return _embedder_instance 