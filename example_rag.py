#!/usr/bin/env python3
"""
Example script demonstrating the RAG Pipeline usage.
This script shows how to use the complete RAG pipeline to answer questions
based on document context.
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.rag_pipeline import RAGPipeline
from vector_store.chroma import ChromaVectorStore
from services.embedder import Embedder

def main():
    print("🚀 RAG Pipeline Example")
    print("=" * 50)
    
    # Sample document content (simulating a research paper)
    sample_documents = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can analyze data, identify patterns, and make decisions.",
        "Deep learning is a specialized subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data. It has revolutionized fields like computer vision, natural language processing, and speech recognition.",
        "Natural language processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret, and manipulate human language. NLP combines computational linguistics with statistical and machine learning models.",
        "Computer vision is a field of artificial intelligence that enables computers to interpret and understand visual information from the world. It involves developing algorithms to process, analyze, and understand digital images and videos.",
        "Reinforcement learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize cumulative reward. It's inspired by behavioral psychology and has applications in robotics and game playing."
    ]
    
    sample_filename = "ai_research_paper.pdf"
    
    try:
        # Initialize services
        print("📦 Initializing RAG pipeline components...")
        embedder = Embedder()
        vector_store = ChromaVectorStore()
        rag_pipeline = RAGPipeline()
        
        # Clear existing data for clean example
        print("🧹 Clearing existing data...")
        vector_store.clear_collection()
        
        # Generate embeddings and store in vector database
        print(f"\n💾 Processing sample document: {sample_filename}")
        embeddings = embedder.generate_embeddings(sample_documents)
        
        # Prepare metadata
        metadata = {
            "filename": sample_filename,
            "source": "example_script",
            "document_type": "research_paper",
            "total_chunks": len(sample_documents)
        }
        
        # Save to vector store
        saved_ids = vector_store.save_embeddings(sample_documents, embeddings, metadata)
        print(f"✅ Stored {len(saved_ids)} chunks in vector database")
        
        # Get pipeline info
        print("\n📋 RAG Pipeline Information:")
        pipeline_info = rag_pipeline.get_pipeline_info()
        print(f"  Status: {pipeline_info['pipeline_status']}")
        print(f"  Vector Store: {pipeline_info['vector_store']['total_chunks']} chunks")
        print(f"  LLM Model: {pipeline_info['llm_model']['model_exists']}")
        
        # Test question answering
        print("\n🔍 Testing Question Answering:")
        test_questions = [
            "What is machine learning?",
            "How does deep learning differ from traditional machine learning?",
            "What are the applications of computer vision?",
            "Explain reinforcement learning and its applications.",
            "What is the relationship between NLP and artificial intelligence?"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n🔎 Question {i}: {question}")
            
            try:
                # Use the main answer_question method
                answer = rag_pipeline.answer_question(
                    user_query=question,
                    file_id=sample_filename,
                    k=3,
                    max_tokens=200,
                    temperature=0.7,
                    include_context_info=True
                )
                
                print(f"🤖 Answer: {answer}")
                
            except Exception as e:
                print(f"❌ Error answering question: {str(e)}")
        
        # Test advanced question answering
        print(f"\n🔬 Testing Advanced Question Answering:")
        advanced_question = "Compare machine learning and deep learning approaches"
        
        try:
            advanced_result = rag_pipeline.answer_question_advanced(
                user_query=advanced_question,
                file_id=sample_filename,
                k=4,
                max_tokens=300,
                temperature=0.6
            )
            
            print(f"🔎 Question: {advanced_question}")
            print(f"🤖 Answer: {advanced_result['answer']}")
            print(f"📊 Metadata:")
            print(f"  - Status: {advanced_result['metadata']['status']}")
            print(f"  - Chunks used: {advanced_result['metadata']['chunks_used']}")
            print(f"  - Generation time: {advanced_result['metadata']['generation_time']:.2f}s")
            
        except Exception as e:
            print(f"❌ Error in advanced question: {str(e)}")
        
        # Test with no context scenario
        print(f"\n🚫 Testing Question with No Relevant Context:")
        irrelevant_question = "What is the recipe for chocolate cake?"
        
        try:
            no_context_answer = rag_pipeline.answer_question(
                user_query=irrelevant_question,
                file_id=sample_filename,
                k=2
            )
            print(f"🔎 Question: {irrelevant_question}")
            print(f"🤖 Answer: {no_context_answer}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        # Test with custom system message
        print(f"\n🎯 Testing Custom System Message:")
        custom_question = "Summarize the key concepts in this document"
        custom_system = "You are a technical summarizer. Provide a concise bullet-point summary of the key concepts mentioned in the context."
        
        try:
            custom_result = rag_pipeline.answer_question_advanced(
                user_query=custom_question,
                file_id=sample_filename,
                k=5,
                max_tokens=250,
                system_message=custom_system
            )
            
            print(f"🔎 Question: {custom_question}")
            print(f"🤖 Summary: {custom_result['answer']}")
            
        except Exception as e:
            print(f"❌ Error with custom system message: {str(e)}")
        
        # Clean up
        print(f"\n🧹 Cleaning up example data...")
        vector_store.delete_chunks_by_metadata({"filename": sample_filename})
        
        print("\n🎉 RAG Pipeline example completed successfully!")
        print("\n💡 Next steps:")
        print("  1. Upload real PDF documents using the /pdf/upload endpoint")
        print("  2. Use the /rag/question endpoint to ask questions about your documents")
        print("  3. Try the advanced features like batch questions and document comparison")
        
    except FileNotFoundError as e:
        print(f"❌ Model file not found: {str(e)}")
        print("💡 Please download a GGUF model and update the MODEL_PATH in your .env file")
        print("   The RAG pipeline requires both embeddings and LLM models to work")
        return 1
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 