from llama_cpp import Llama
from typing import Optional, Dict, Any
import os
from dotenv import load_dotenv
import logging
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LlamaRunner:
    def __init__(self, model_path: Optional[str] = None, **kwargs):
        """
        Initialize the Llama model runner.
        
        Args:
            model_path (str, optional): Path to the GGUF model file
            **kwargs: Additional parameters for model initialization
        """
        # Get model configuration from environment variables
        self.model_path = model_path or os.getenv("MODEL_PATH", "./models/llama-2-7b-chat.q4_0.gguf")
        
        # Model parameters from environment with defaults
        self.n_ctx = int(os.getenv("MODEL_N_CTX", "2048"))  # Context window size
        self.n_threads = int(os.getenv("MODEL_N_THREADS", "-1"))  # CPU threads (-1 = auto)
        self.n_gpu_layers = int(os.getenv("MODEL_N_GPU_LAYERS", "0"))  # GPU layers (0 = CPU only)
        self.temperature = float(os.getenv("MODEL_TEMPERATURE", "0.7"))
        self.top_p = float(os.getenv("MODEL_TOP_P", "0.9"))
        self.top_k = int(os.getenv("MODEL_TOP_K", "40"))
        self.max_tokens = int(os.getenv("MODEL_MAX_TOKENS", "512"))
        self.repeat_penalty = float(os.getenv("MODEL_REPEAT_PENALTY", "1.1"))
        
        # Override with any provided kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        logger.info(f"Initializing Llama model from: {self.model_path}")
        logger.info(f"Context window: {self.n_ctx}, GPU layers: {self.n_gpu_layers}")
        
        # Check if model file exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            # Initialize the Llama model
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False  # Set to True for debugging
            )
            
            logger.info("Llama model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Llama model: {str(e)}")
            raise Exception(f"Failed to initialize Llama model: {str(e)}")
    
    def generate_response(
        self, 
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        stop: Optional[list] = None
    ) -> str:
        """
        Generate a response from the model given a prompt.
        
        Args:
            prompt (str): Input prompt for the model
            max_tokens (int, optional): Maximum tokens to generate
            temperature (float, optional): Sampling temperature
            top_p (float, optional): Top-p sampling parameter
            top_k (int, optional): Top-k sampling parameter
            repeat_penalty (float, optional): Repetition penalty
            stop (list, optional): Stop sequences
        
        Returns:
            str: Generated response text
        """
        if not prompt.strip():
            return ""
        
        # Use instance defaults if parameters not provided
        generation_params = {
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature or self.temperature,
            "top_p": top_p or self.top_p,
            "top_k": top_k or self.top_k,
            "repeat_penalty": repeat_penalty or self.repeat_penalty,
            "stop": stop or ["</s>", "<|im_end|>", "\n\n"]  # Common stop sequences
        }
        
        logger.info(f"Generating response for prompt: {prompt[:100]}...")
        logger.info(f"Generation params: {generation_params}")
        
        try:
            start_time = time.time()
            
            # Generate response
            response = self.llm(
                prompt,
                **generation_params,
                echo=False  # Don't include the prompt in the output
            )
            
            generation_time = time.time() - start_time
            
            # Extract the generated text
            generated_text = response["choices"][0]["text"].strip()
            
            logger.info(f"Response generated in {generation_time:.2f}s")
            logger.info(f"Generated {len(generated_text)} characters")
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
            raise Exception(f"Failed to generate response: {str(e)}")
    
    def generate_chat_response(
        self,
        messages: list,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate a response using chat format.
        
        Args:
            messages (list): List of message dictionaries with 'role' and 'content'
            max_tokens (int, optional): Maximum tokens to generate
            temperature (float, optional): Sampling temperature
            **kwargs: Additional generation parameters
        
        Returns:
            str: Generated response text
        """
        try:
            # Format messages into a single prompt
            prompt = self._format_chat_prompt(messages)
            
            return self.generate_response(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
        except Exception as e:
            logger.error(f"Failed to generate chat response: {str(e)}")
            raise Exception(f"Failed to generate chat response: {str(e)}")
    
    def _format_chat_prompt(self, messages: list) -> str:
        """
        Format chat messages into a prompt string.
        
        Args:
            messages (list): List of message dictionaries
            
        Returns:
            str: Formatted prompt string
        """
        # Simple chat format - can be customized based on model requirements
        formatted_prompt = ""
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                formatted_prompt += f"System: {content}\n\n"
            elif role == "user":
                formatted_prompt += f"Human: {content}\n\n"
            elif role == "assistant":
                formatted_prompt += f"Assistant: {content}\n\n"
        
        # Add assistant prompt for response
        formatted_prompt += "Assistant:"
        
        return formatted_prompt
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dict: Model information and parameters
        """
        return {
            "model_path": self.model_path,
            "n_ctx": self.n_ctx,
            "n_threads": self.n_threads,
            "n_gpu_layers": self.n_gpu_layers,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "repeat_penalty": self.repeat_penalty,
            "model_exists": os.path.exists(self.model_path),
            "model_size_mb": os.path.getsize(self.model_path) / (1024 * 1024) if os.path.exists(self.model_path) else 0
        }
    
    def test_generation(self) -> Dict[str, Any]:
        """
        Test the model with a simple prompt.
        
        Returns:
            Dict: Test results including prompt, response, and timing
        """
        test_prompt = "Hello! How are you today?"
        
        try:
            start_time = time.time()
            response = self.generate_response(test_prompt, max_tokens=50)
            generation_time = time.time() - start_time
            
            return {
                "success": True,
                "prompt": test_prompt,
                "response": response,
                "generation_time": generation_time,
                "response_length": len(response)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "prompt": test_prompt
            }
    
    def create_rag_prompt(self, query: str, context_chunks: list, system_message: Optional[str] = None) -> str:
        """
        Create a RAG (Retrieval-Augmented Generation) prompt.
        
        Args:
            query (str): User's question
            context_chunks (list): List of relevant text chunks
            system_message (str, optional): System message for the model
        
        Returns:
            str: Formatted RAG prompt
        """
        # Default system message for RAG
        default_system = """You are a helpful assistant that answers questions based on the provided context. 
Use only the information from the context to answer the question. If the answer cannot be found in the context, say so."""
        
        system_msg = system_message or default_system
        
        # Format context
        context_text = "\n\n".join([f"Context {i+1}: {chunk}" for i, chunk in enumerate(context_chunks)])
        
        # Create RAG prompt
        rag_prompt = f"""System: {system_msg}

Context:
{context_text}

Question: {query}

Answer:"""
        
        return rag_prompt

# Global LlamaRunner instance (singleton pattern)
_llama_instance: Optional[LlamaRunner] = None

def get_llama_runner() -> LlamaRunner:
    """
    Get or create a global LlamaRunner instance (singleton pattern).
    
    Returns:
        LlamaRunner: Global LlamaRunner instance
    """
    global _llama_instance
    if _llama_instance is None:
        _llama_instance = LlamaRunner()
    return _llama_instance 