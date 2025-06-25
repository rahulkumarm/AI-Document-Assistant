#!/usr/bin/env python3
"""
Example script demonstrating the Embedder service usage.
This script can be run independently to test the embedder functionality.
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.embedder import Embedder

def main():
    print("🚀 Embedder Service Example")
    print("=" * 50)
    
    # Sample text chunks
    text_chunks = [
        "FastAPI is a modern, fast web framework for building APIs with Python.",
        "Machine learning models can process and understand natural language text.",
        "Vector embeddings represent text as numerical vectors in high-dimensional space.",
        "Sentence transformers are neural networks trained to create meaningful embeddings.",
        "ChromaDB is a vector database designed for storing and querying embeddings."
    ]
    
    try:
        # Initialize embedder
        print("📦 Initializing embedder...")
        embedder = Embedder()
        
        # Get model info
        print("\n📋 Model Information:")
        model_info = embedder.get_model_info()
        for key, value in model_info.items():
            print(f"  {key}: {value}")
        
        # Generate embeddings
        print(f"\n🔄 Generating embeddings for {len(text_chunks)} text chunks...")
        embeddings = embedder.generate_embeddings(text_chunks, batch_size=2)
        
        print(f"\n✅ Successfully generated {len(embeddings)} embeddings")
        print(f"📏 Embedding dimension: {len(embeddings[0])}")
        
        # Show first few values of first embedding
        print(f"\n🔍 First embedding (first 10 values): {embeddings[0][:10]}")
        
        # Test single embedding
        print("\n🔄 Testing single embedding generation...")
        single_text = "This is a test sentence for single embedding."
        single_embedding = embedder.generate_single_embedding(single_text)
        print(f"📏 Single embedding dimension: {len(single_embedding)}")
        
        # Test similarity computation
        print("\n🔄 Testing similarity computation...")
        similarity = embedder.compute_similarity(embeddings[0], embeddings[1])
        print(f"🎯 Similarity between first two chunks: {similarity:.4f}")
        
        # Test similarity between similar texts
        similar_text1 = "Machine learning is a subset of artificial intelligence."
        similar_text2 = "AI and machine learning are closely related fields."
        
        emb1 = embedder.generate_single_embedding(similar_text1)
        emb2 = embedder.generate_single_embedding(similar_text2)
        similarity_high = embedder.compute_similarity(emb1, emb2)
        
        print(f"🎯 Similarity between similar texts: {similarity_high:.4f}")
        
        print("\n🎉 Embedder example completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 