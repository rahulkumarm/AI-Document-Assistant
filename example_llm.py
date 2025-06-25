#!/usr/bin/env python3
"""
Example script demonstrating the LLM Runner usage.
This script can be run independently to test the LLM functionality.
Note: Requires a GGUF model file to be available at the specified path.
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm.llama_runner import LlamaRunner

def main():
    print("🚀 LLM Runner Example")
    print("=" * 50)
    
    try:
        # Initialize LLM runner
        print("📦 Initializing LLM runner...")
        llm_runner = LlamaRunner()
        
        # Get model info
        print("\n📋 Model Information:")
        model_info = llm_runner.get_model_info()
        for key, value in model_info.items():
            print(f"  {key}: {value}")
        
        # Test basic generation
        print("\n🔄 Testing Basic Generation:")
        test_result = llm_runner.test_generation()
        
        if test_result["success"]:
            print(f"✅ Test successful!")
            print(f"  Prompt: {test_result['prompt']}")
            print(f"  Response: {test_result['response']}")
            print(f"  Generation time: {test_result['generation_time']:.2f}s")
        else:
            print(f"❌ Test failed: {test_result['error']}")
            return 1
        
        # Test different prompts
        print("\n🔄 Testing Different Prompts:")
        test_prompts = [
            "What is artificial intelligence?",
            "Explain machine learning in simple terms.",
            "Write a short poem about technology."
        ]
        
        for prompt in test_prompts:
            print(f"\n🔎 Prompt: '{prompt}'")
            try:
                response = llm_runner.generate_response(
                    prompt=prompt,
                    max_tokens=100,
                    temperature=0.7
                )
                print(f"🤖 Response: {response}")
            except Exception as e:
                print(f"❌ Error: {str(e)}")
        
        # Test chat format
        print("\n💬 Testing Chat Format:")
        messages = [
            {"role": "system", "content": "You are a helpful assistant that gives concise answers."},
            {"role": "user", "content": "What are the benefits of using vector databases?"}
        ]
        
        try:
            chat_response = llm_runner.generate_chat_response(
                messages=messages,
                max_tokens=150,
                temperature=0.6
            )
            print(f"🤖 Chat Response: {chat_response}")
        except Exception as e:
            print(f"❌ Chat Error: {str(e)}")
        
        # Test RAG prompt creation
        print("\n📚 Testing RAG Prompt Creation:")
        query = "What is machine learning?"
        context_chunks = [
            "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data.",
            "Machine learning algorithms can identify patterns in data and make predictions without being explicitly programmed.",
            "Common types of machine learning include supervised learning, unsupervised learning, and reinforcement learning."
        ]
        
        rag_prompt = llm_runner.create_rag_prompt(query, context_chunks)
        print(f"📝 RAG Prompt Created:")
        print(f"{rag_prompt[:200]}...")
        
        # Generate response using RAG prompt
        try:
            rag_response = llm_runner.generate_response(
                prompt=rag_prompt,
                max_tokens=200,
                temperature=0.5
            )
            print(f"\n🤖 RAG Response: {rag_response}")
        except Exception as e:
            print(f"❌ RAG Error: {str(e)}")
        
        print("\n🎉 LLM example completed successfully!")
        
    except FileNotFoundError as e:
        print(f"❌ Model file not found: {str(e)}")
        print("💡 Please download a GGUF model and update the MODEL_PATH in your .env file")
        print("   Example: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF")
        return 1
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 