#!/usr/bin/env python3
"""
Example script demonstrating the ChromaDB vector store usage.
This script can be run independently to test the vector store functionality.
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vector_store.chroma import ChromaVectorStore
from services.embedder import Embedder

def main():
    print("🚀 ChromaDB Vector Store Example")
    print("=" * 50)
    
    # Sample documents
    documents = [
        "FastAPI is a modern web framework for building APIs with Python based on standard Python type hints.",
        "Machine learning enables computers to learn and make decisions from data without being explicitly programmed.",
        "Vector databases store and query high-dimensional vectors efficiently for similarity search applications.",
        "Natural language processing helps computers understand, interpret and generate human language.",
        "ChromaDB is an open-source embedding database that makes it easy to build LLM applications."
    ]
    
    try:
        # Initialize services
        print("📦 Initializing embedder and vector store...")
        embedder = Embedder()
        vector_store = ChromaVectorStore()
        
        # Clear existing data for clean example
        print("🧹 Clearing existing data...")
        vector_store.clear_collection()
        
        # Generate embeddings
        print(f"\n🔄 Generating embeddings for {len(documents)} documents...")
        embeddings = embedder.generate_embeddings(documents, batch_size=2)
        
        # Prepare metadata
        metadata = {
            "source": "example_script",
            "domain": "ai_ml_tech",
            "language": "english"
        }
        
        # Save to vector store
        print("💾 Saving embeddings to ChromaDB...")
        saved_ids = vector_store.save_embeddings(documents, embeddings, metadata)
        print(f"✅ Saved {len(saved_ids)} documents with IDs: {saved_ids[:2]}...")
        
        # Get collection stats
        print("\n📊 Collection Statistics:")
        stats = vector_store.get_collection_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Test similarity search
        print("\n🔍 Testing Similarity Search:")
        test_queries = [
            "What is machine learning?",
            "How do vector databases work?",
            "Tell me about web frameworks"
        ]
        
        for query in test_queries:
            print(f"\n🔎 Query: '{query}'")
            results = vector_store.query_similar_chunks(query, k=3)
            
            for i, result in enumerate(results, 1):
                print(f"  {i}. Similarity: {result.get('similarity', 0):.3f}")
                print(f"     Text: {result['document'][:80]}...")
                print(f"     Metadata: {result['metadata']['source']}")
        
        # Test filtering by metadata
        print(f"\n🎯 Testing Metadata Filtering:")
        filtered_results = vector_store.query_similar_chunks(
            "artificial intelligence", 
            k=2, 
            where={"domain": "ai_ml_tech"}
        )
        
        print(f"Found {len(filtered_results)} results with domain filter:")
        for result in filtered_results:
            print(f"  - {result['document'][:60]}...")
        
        # Test query by embedding
        print(f"\n🧮 Testing Query by Embedding Vector:")
        query_embedding = embedder.generate_single_embedding("database technology")
        embedding_results = vector_store.query_by_embedding(query_embedding, k=2)
        
        print(f"Found {len(embedding_results)} results using embedding query:")
        for result in embedding_results:
            print(f"  - Similarity: {result.get('similarity', 0):.3f}")
            print(f"    Text: {result['document'][:60]}...")
        
        # Test deletion
        print(f"\n🗑️ Testing Document Deletion:")
        deleted_count = vector_store.delete_chunks_by_metadata({"source": "example_script"})
        print(f"Deleted {deleted_count} documents")
        
        # Final stats
        final_stats = vector_store.get_collection_stats()
        print(f"Final collection size: {final_stats['total_chunks']} documents")
        
        print("\n🎉 ChromaDB example completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 