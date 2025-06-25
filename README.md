# AI Project Backend

A FastAPI backend for AI project with PDF parsing, embedding, and RAG capabilities.

## Project Structure

```
.
├── main.py              # FastAPI app entry point
├── routers/             # API endpoints
├── services/            # Business logic (PDF parsing, embedding, LLM, RAG)
├── models/              # Pydantic request/response schemas
├── vector_store/        # ChromaDB interaction logic
├── llm/                 # llama-cpp-python integration
├── requirements.txt     # Python dependencies
├── example_embedder.py  # Embedder service example
├── example_chroma.py    # ChromaDB vector store example
├── example_llm.py       # LLM runner example
├── example_rag.py       # RAG pipeline example
└── README.md           # This file
```

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with the following variables:
   ```
   # FastAPI Configuration
   DEBUG=True
   HOST=0.0.0.0
   PORT=8000
   
   # LLM Configuration
   MODEL_PATH=./models/llama-2-7b-chat.q4_0.gguf
   MODEL_N_CTX=2048
   MODEL_N_THREADS=-1
   MODEL_N_GPU_LAYERS=0
   MODEL_TEMPERATURE=0.7
   MODEL_TOP_P=0.9
   MODEL_TOP_K=40
   MODEL_MAX_TOKENS=512
   MODEL_REPEAT_PENALTY=1.1
   
   # RAG Pipeline Configuration
   RAG_DEFAULT_K=3
   RAG_MAX_TOKENS=500
   RAG_TEMPERATURE=0.7
   
   # ChromaDB Configuration
   CHROMA_DB_PATH=./chroma_db
   COLLECTION_NAME=documents
   
   # Embedding Configuration
   EMBEDDING_MODEL=BAAI/bge-base-en
   ```

4. Download a GGUF model (optional - for LLM functionality):
   ```bash
   mkdir -p models
   # Download your preferred GGUF model to the models directory
   # Example: wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.q4_0.gguf -O models/llama-2-7b-chat.q4_0.gguf
   ```

5. Run the application:
   ```bash
   python main.py
   ```
   
   Or using uvicorn directly:
   ```bash
   uvicorn main:app --reload
   ```

## API Endpoints

### Core Endpoints
- `GET /` - Hello World endpoint
- `GET /health` - Health check endpoint

### PDF Processing
- `POST /pdf/upload` - Upload and process PDF files
  - Query params: `generate_embeddings`, `save_to_vector_store`
- `GET /pdf/documents` - List processed documents
- `POST /pdf/generate-embeddings/{filename}` - Generate embeddings for a document

### Vector Store & Search
- `POST /pdf/search` - Search similar text chunks
- `GET /pdf/vector-store/stats` - Get vector store statistics
- `DELETE /pdf/vector-store/clear` - Clear vector store
- `DELETE /pdf/vector-store/document/{filename}` - Delete document from vector store

### LLM Endpoints
- `GET /llm/info` - Get LLM model information
- `POST /llm/test` - Test LLM with a simple prompt
- `POST /llm/generate` - Generate response from prompt
- `POST /llm/chat` - Generate response using chat format
- `POST /llm/rag` - Perform RAG query (retrieval + generation)
- `POST /llm/rag/simple` - Simple RAG query with URL parameters
- `GET /llm/models/available` - List available GGUF models

### RAG Pipeline Endpoints
- `GET /rag/info` - Get RAG pipeline information and status
- `POST /rag/question` - Answer question using research paper format
- `POST /rag/question/simple` - Simple question with URL parameters
- `POST /rag/question/advanced` - Advanced question with detailed response
- `POST /rag/batch/questions` - Answer multiple questions for same document
- `POST /rag/compare/documents` - Compare answers across multiple documents

### Embedder & LLM Info
- `GET /pdf/embedder/info` - Get embedder model information

## Features

### 📄 PDF Processing
- Extract text from PDF files page by page
- Clean and structure text content
- Store documents in memory with metadata

### 🔢 Embedding Generation
- Uses BAAI/bge-base-en model for high-quality embeddings
- Efficient batching for multiple texts
- Device auto-detection (CUDA/MPS/CPU)
- Cosine similarity calculations

### 💾 Vector Storage
- ChromaDB for persistent vector storage
- Metadata filtering and search
- Efficient similarity search
- Document management (add/delete/clear)

### 🤖 LLM Integration
- GGUF model support via llama-cpp-python
- Configurable generation parameters
- Chat format support
- RAG prompt templates
- CPU and GPU acceleration support

### 🔗 RAG Pipeline
- Complete retrieval-augmented generation workflow
- Research paper-specific prompt formatting
- Configurable context retrieval (k chunks)
- Batch question processing
- Document comparison capabilities
- Advanced response with metadata

## Usage Examples

### Upload PDF with Full Processing
```bash
curl -X POST "http://localhost:8000/pdf/upload?generate_embeddings=true&save_to_vector_store=true" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"
```

### Ask Question About Document (RAG)
```bash
curl -X POST "http://localhost:8000/rag/question/simple?user_query=What is machine learning?&file_id=document.pdf&k=3"
```

### Advanced RAG Question
```bash
curl -X POST "http://localhost:8000/rag/question/advanced" \
  -H "Content-Type: application/json" \
  -d '{
    "user_query": "Compare the different approaches mentioned in the paper",
    "file_id": "research_paper.pdf",
    "k": 5,
    "max_tokens": 400,
    "temperature": 0.6,
    "system_message": "You are a research assistant. Provide a detailed comparison."
  }'
```

### Batch Questions
```bash
curl -X POST "http://localhost:8000/rag/batch/questions?file_id=document.pdf" \
  -H "Content-Type: application/json" \
  -d '["What is the main topic?", "What are the key findings?", "What are the limitations?"]'
```

### Search Similar Content
```bash
curl -X POST "http://localhost:8000/pdf/search?query=machine learning&k=5" \
  -H "Content-Type: application/json"
```

### Generate LLM Response
```bash
curl -X POST "http://localhost:8000/llm/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is artificial intelligence?", "max_tokens": 100}'
```

### RAG Query (Simple)
```bash
curl -X POST "http://localhost:8000/llm/rag/simple?query=What is machine learning?&k=3&max_tokens=200"
```

### Chat with LLM
```bash
curl -X POST "http://localhost:8000/llm/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain quantum computing"}
    ],
    "max_tokens": 150
  }'
```

### Get Vector Store Stats
```bash
curl -X GET "http://localhost:8000/pdf/vector-store/stats"
```

## Example Scripts

Run the example scripts to test individual components:

```bash
# Test embedder functionality
python example_embedder.py

# Test ChromaDB vector store
python example_chroma.py

# Test LLM functionality (requires GGUF model)
python example_llm.py

# Test complete RAG pipeline (requires GGUF model)
python example_rag.py
```

## Complete RAG Workflow

1. **Upload Document**: Upload PDF with embeddings and vector store enabled
   ```bash
   curl -X POST "http://localhost:8000/pdf/upload?generate_embeddings=true&save_to_vector_store=true" \
     -F "file=@research_paper.pdf"
   ```

2. **Ask Questions**: Use the RAG pipeline to ask questions about the document
   ```bash
   curl -X POST "http://localhost:8000/rag/question/simple?user_query=What are the main findings?&file_id=research_paper.pdf"
   ```

3. **Advanced Analysis**: Use batch processing for comprehensive analysis
   ```bash
   curl -X POST "http://localhost:8000/rag/batch/questions?file_id=research_paper.pdf" \
     -d '["What is the methodology?", "What are the results?", "What are the conclusions?"]'
   ```

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `./models/llama-2-7b-chat.q4_0.gguf` | Path to GGUF model file |
| `MODEL_N_CTX` | `2048` | Context window size |
| `MODEL_N_THREADS` | `-1` | CPU threads (-1 = auto) |
| `MODEL_N_GPU_LAYERS` | `0` | GPU layers (0 = CPU only) |
| `MODEL_TEMPERATURE` | `0.7` | Sampling temperature |
| `MODEL_TOP_P` | `0.9` | Top-p sampling |
| `MODEL_TOP_K` | `40` | Top-k sampling |
| `MODEL_MAX_TOKENS` | `512` | Maximum tokens to generate |
| `MODEL_REPEAT_PENALTY` | `1.1` | Repetition penalty |
| `RAG_DEFAULT_K` | `3` | Default context chunks for RAG |
| `RAG_MAX_TOKENS` | `500` | Default max tokens for RAG |
| `RAG_TEMPERATURE` | `0.7` | Default temperature for RAG |
| `CHROMA_DB_PATH` | `./chroma_db` | ChromaDB storage path |
| `COLLECTION_NAME` | `documents` | ChromaDB collection name |
| `EMBEDDING_MODEL` | `BAAI/bge-base-en` | HuggingFace embedding model |

The API documentation is available at `http://localhost:8000/docs` when running. 