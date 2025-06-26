from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import uuid
import chromadb
from sentence_transformers import SentenceTransformer
import PyPDF2
import io
from typing import List, Dict, Any
import logging
import re

# Optional Ollama import (not available on Railway)
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Ollama not available - LLM features will be disabled")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Project Backend (Enhanced)",
    description="Enhanced FastAPI backend with real RAG functionality using ChromaDB and embeddings",
    version="1.0.0"
)

# Custom CORS origin checker for Vercel deployments
def check_cors_origin(origin: str) -> bool:
    """Check if origin is allowed for CORS"""
    allowed_patterns = [
        r"^http://localhost:\d+$",  # Local development
        r"^https://.*\.vercel\.app$",  # Any Vercel deployment
        r"^https://.*\.railway\.app$",  # Any Railway deployment
        r"^https://ai-document-assistant.*\.vercel\.app$",  # Specific project deployments
    ]
    
    for pattern in allowed_patterns:
        if re.match(pattern, origin):
            return True
    return False

# Add CORS middleware with explicit origins (better Railway compatibility)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", 
        "http://localhost:5174", 
        "http://localhost:5175", 
        "http://localhost:5176",
        "https://ai-document-assistant-blmet8y2n-rahul-kumars-projects-bbeb7f6d.vercel.app",
        "https://ai-document-assistant-git-main-rahul-kumars-projects-bbeb7f6d.vercel.app",
        "https://ai-document-assistant.vercel.app",
        "https://ai-document-assistant-production.up.railway.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="documents")

# Initialize embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Llama model via Ollama (if available)
llm_available = False
OLLAMA_MODEL = None

if OLLAMA_AVAILABLE:
    logger.info("Initializing Llama 3.2 via Ollama...")
    try:
        # Test connection to Ollama
        models = ollama.list()
        available_models = [model.model for model in models.models]
        logger.info(f"Available Ollama models: {available_models}")
        
        if 'llama3.2:3b' in available_models:
            OLLAMA_MODEL = 'llama3.2:3b'
            logger.info("Llama 3.2 3B model ready!")
            llm_available = True
            
            # Test the model with a simple query
            test_response = ollama.generate(
                model=OLLAMA_MODEL,
                prompt="Test: What is 2+2?",
                options={'num_predict': 10}
            )
            logger.info(f"Model test successful: {test_response['response'][:50]}...")
            
        else:
            logger.warning("Llama 3.2 not found. Please run: ollama pull llama3.2:3b")
            
    except Exception as e:
        logger.error(f"Error connecting to Ollama: {str(e)}")
        logger.info("Make sure Ollama is running: brew services start ollama")
else:
    logger.info("Ollama not available - running in embedding/search-only mode")

class AskRequest(BaseModel):
    file_id: str
    question: str

class AskResponse(BaseModel):
    answer: str

class UploadResponse(BaseModel):
    document_id: str
    filename: str
    total_pages: int
    total_chunks: int
    embeddings_generated: bool
    message: str

def extract_text_from_pdf(pdf_content: bytes) -> List[str]:
    """
    Advanced PDF text extraction with intelligent parsing and context-aware chunking.
    Designed to match ChatGPT-level document understanding.
    """
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        text_chunks = []
        full_document_text = ""
        page_texts = []
        
        # First pass: Extract all text and build full document context
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text.strip():
                # Advanced text cleaning
                text = text.replace('\n', ' ').replace('\r', ' ')
                text = ' '.join(text.split())  # Remove extra whitespace
                page_texts.append((page_num + 1, text))
                full_document_text += f" [PAGE {page_num + 1}] " + text
        
        logger.info(f"Extracted text from {len(page_texts)} pages, total length: {len(full_document_text)}")
        
        # Advanced document structure analysis
        import re
        
        # Extract document metadata (title, authors, abstract, keywords)
        first_page_text = page_texts[0][1] if page_texts else ""
        logger.info(f"First page text length: {len(first_page_text)}")
        logger.info(f"First 500 characters: {first_page_text[:500]}")
        
        # Create comprehensive metadata chunk
        if first_page_text:
            # Try to identify title (usually first substantial text)
            title_patterns = [
                r'^([A-Z][^.!?]*(?:[.!?]|$))',  # First capitalized sentence
                r'^(.*?)(?:\n|Abstract|ABSTRACT|\d+\.|\w+@)',  # Text before abstract or sections
            ]
            
            potential_titles = []
            for pattern in title_patterns:
                matches = re.findall(pattern, first_page_text, re.MULTILINE | re.DOTALL)
                if matches:
                    title = matches[0].strip()
                    if 5 < len(title) < 200 and not title.lower().startswith(('the', 'a ', 'an ')):
                        potential_titles.append(title)
            
            # Extract authors (names typically follow title)
            author_patterns = [
                r'([A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # Full names
                r'([A-Z]\. [A-Z][a-z]+)',  # Initials + last name
            ]
            
            potential_authors = []
            for pattern in author_patterns:
                matches = re.findall(pattern, first_page_text)
                potential_authors.extend(matches[:10])  # Limit to reasonable number
            
            # Create rich metadata chunk
            metadata_chunk = {
                'text': first_page_text,
                'page': 1,
                'chunk_index': len(text_chunks),
                'chunk_type': 'metadata',
                'extracted_title': potential_titles[0] if potential_titles else None,
                'extracted_authors': potential_authors[:5] if potential_authors else None,
                'content_type': 'document_metadata'
            }
            text_chunks.append(metadata_chunk)
            logger.info(f"Created metadata chunk - Title: {potential_titles[0] if potential_titles else 'None'}")
            logger.info(f"Authors found: {potential_authors[:3] if potential_authors else 'None'}")
        
        # Advanced figure and table extraction
        def extract_figures_and_tables(text, page_num):
            figures_tables = []
            
            # Enhanced figure patterns
            figure_patterns = [
                r'(Figure\s+\d+[:\.].*?)(?=Figure\s+\d+|Table\s+\d+|\n\n|\.|$)',
                r'(Fig\.\s+\d+[:\.].*?)(?=Fig\.\s+\d+|Table\s+\d+|\n\n|\.|$)',
                r'(figure\s+\d+[:\.].*?)(?=figure\s+\d+|table\s+\d+|\n\n|\.|$)',
                # Capture figure references and surrounding context
                r'([^.!?]*?(?:Figure|Fig\.|figure)\s+\d+[^.!?]*[.!?])',
                # Caption-like patterns
                r'(\([A-Z]\)[^.!?]*(?:Figure|Fig\.|figure)\s+\d+[^.!?]*[.!?])',
            ]
            
            for pattern in figure_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    clean_match = ' '.join(match.split())
                    if len(clean_match) > 15:  # Meaningful content
                        # Extract figure number
                        fig_num_match = re.search(r'(?:Figure|Fig\.|figure)\s+(\d+)', clean_match, re.IGNORECASE)
                        fig_number = fig_num_match.group(1) if fig_num_match else 'unknown'
                        
                        figures_tables.append({
                            'text': clean_match,
                            'page': page_num,
                            'chunk_index': len(text_chunks) + len(figures_tables),
                            'chunk_type': 'figure',
                            'figure_number': fig_number,
                            'content_type': f'figure_{fig_number}'
                        })
            
            # Enhanced table patterns
            table_patterns = [
                r'(Table\s+\d+[:\.].*?)(?=Table\s+\d+|Figure\s+\d+|\n\n|\.|$)',
                r'(table\s+\d+[:\.].*?)(?=table\s+\d+|figure\s+\d+|\n\n|\.|$)',
                r'([^.!?]*?Table\s+\d+[^.!?]*[.!?])',
            ]
            
            for pattern in table_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    clean_match = ' '.join(match.split())
                    if len(clean_match) > 15:
                        table_num_match = re.search(r'Table\s+(\d+)', clean_match, re.IGNORECASE)
                        table_number = table_num_match.group(1) if table_num_match else 'unknown'
                        
                        figures_tables.append({
                            'text': clean_match,
                            'page': page_num,
                            'chunk_index': len(text_chunks) + len(figures_tables),
                            'chunk_type': 'table',
                            'table_number': table_number,
                            'content_type': f'table_{table_number}'
                        })
            
            return figures_tables
        
        # Extract section headers and organize content
        def extract_sections(text, page_num):
            sections = []
            
            section_patterns = [
                r'(\d+\.\s+[A-Z][^.]*?)(?=\d+\.\s+[A-Z]|$)',  # 1. Introduction
                r'(\d+\.\d+\s+[A-Z][^.]*?)(?=\d+\.\d+\s+[A-Z]|$)',  # 1.1 Background
                r'((?:Abstract|Introduction|Background|Related Work|Methodology|Methods|Experiments|Results|Discussion|Conclusion|References|Acknowledgments)\b.*?)(?=(?:Abstract|Introduction|Background|Related Work|Methodology|Methods|Experiments|Results|Discussion|Conclusion|References|Acknowledgments)\b|$)',
            ]
            
            for pattern in section_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    clean_match = ' '.join(match.split())
                    if len(clean_match) > 20:  # Substantial content
                        # Extract section name
                        section_name = re.match(r'^[^.]*', clean_match).group(0) if re.match(r'^[^.]*', clean_match) else 'unknown'
                        
                        sections.append({
                            'text': clean_match[:1000],  # Limit section chunk size
                            'page': page_num,
                            'chunk_index': len(text_chunks) + len(sections),
                            'chunk_type': 'section',
                            'section_name': section_name.strip(),
                            'content_type': f'section_{section_name.lower().replace(" ", "_")}'
                        })
            
            return sections
        
        # Process each page for figures, tables, and sections
        all_figures_tables = []
        all_sections = []
        
        for page_num, text in page_texts:
            figures_tables = extract_figures_and_tables(text, page_num)
            sections = extract_sections(text, page_num)
            
            all_figures_tables.extend(figures_tables)
            all_sections.extend(sections)
        
        # Add extracted figures and tables
        text_chunks.extend(all_figures_tables)
        text_chunks.extend(all_sections)
        
        logger.info(f"Extracted {len(all_figures_tables)} figure/table chunks")
        logger.info(f"Extracted {len(all_sections)} section chunks")
        
        # Intelligent content chunking with context preservation
        def create_intelligent_chunks(text, page_num, chunk_size=600):
            """Create overlapping chunks that preserve context and meaning."""
            chunks = []
            
            # Split into sentences first
            sentence_endings = r'[.!?]'
            sentences = re.split(f'({sentence_endings})', text)
            
            # Reconstruct sentences properly
            reconstructed_sentences = []
            for i in range(0, len(sentences), 2):
                if i + 1 < len(sentences):
                    sentence = sentences[i] + sentences[i + 1]
                else:
                    sentence = sentences[i]
                
                sentence = sentence.strip()
                if len(sentence) > 10:  # Filter out very short fragments
                    reconstructed_sentences.append(sentence)
            
            # Create overlapping chunks
            current_chunk = ""
            overlap_buffer = ""
            
            for i, sentence in enumerate(reconstructed_sentences):
                # Check if adding this sentence would exceed chunk size
                if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                    # Create chunk with overlap from previous chunk
                    final_chunk = overlap_buffer + current_chunk if overlap_buffer else current_chunk
                    
                    chunks.append({
                        'text': final_chunk.strip(),
                        'page': page_num,
                        'chunk_index': len(text_chunks) + len(chunks),
                        'chunk_type': 'content',
                        'content_type': 'contextual_content',
                        'sentence_start': max(0, i - 3),  # Track position for context
                        'sentence_end': i
                    })
                    
                    # Prepare overlap for next chunk (last 2 sentences)
                    overlap_sentences = reconstructed_sentences[max(0, i-2):i]
                    overlap_buffer = ' '.join(overlap_sentences) + ' ' if overlap_sentences else ''
                    current_chunk = sentence + ' '
                else:
                    current_chunk += sentence + ' '
            
            # Add final chunk
            if current_chunk.strip():
                final_chunk = overlap_buffer + current_chunk if overlap_buffer else current_chunk
                chunks.append({
                    'text': final_chunk.strip(),
                    'page': page_num,
                    'chunk_index': len(text_chunks) + len(chunks),
                    'chunk_type': 'content',
                    'content_type': 'contextual_content'
                })
            
            return chunks
        
        # Create intelligent content chunks for each page
        for page_num, text in page_texts:
            content_chunks = create_intelligent_chunks(text, page_num)
            text_chunks.extend(content_chunks)
        
        # Log final statistics
        chunk_types = {}
        for chunk in text_chunks:
            chunk_type = chunk.get('chunk_type', 'unknown')
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        
        logger.info(f"Final extraction complete: {len(text_chunks)} total chunks")
        logger.info(f"Chunk distribution: {chunk_types}")
        
        return text_chunks
        
    except Exception as e:
        logger.error(f"Error in advanced text extraction: {str(e)}")
        raise

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for text chunks."""
    try:
        embeddings = embedder.encode(texts)
        return embeddings.tolist()
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise

def store_document_chunks(document_id: str, filename: str, chunks: List[Dict[str, Any]]):
    """Store document chunks in ChromaDB with enhanced metadata preservation."""
    try:
        texts = [chunk['text'] for chunk in chunks]
        embeddings = generate_embeddings(texts)
        
        ids = [f"{document_id}_{i}" for i in range(len(chunks))]
        metadatas = []
        
        for chunk in chunks:
            # Create comprehensive metadata preserving all extracted information
            metadata = {
                'filename': filename,
                'document_id': document_id,
                'page': chunk['page'],
                'chunk_index': chunk['chunk_index'],
                'chunk_type': chunk.get('chunk_type', 'content'),
                'content_type': chunk.get('content_type', 'unknown')
            }
            
            # Add type-specific metadata
            if chunk.get('chunk_type') == 'metadata':
                if chunk.get('extracted_title'):
                    metadata['extracted_title'] = chunk['extracted_title']
                if chunk.get('extracted_authors'):
                    metadata['extracted_authors'] = ','.join(chunk['extracted_authors'])
            
            elif chunk.get('chunk_type') == 'figure':
                if chunk.get('figure_number'):
                    metadata['figure_number'] = chunk['figure_number']
            
            elif chunk.get('chunk_type') == 'table':
                if chunk.get('table_number'):
                    metadata['table_number'] = chunk['table_number']
            
            elif chunk.get('chunk_type') == 'section':
                if chunk.get('section_name'):
                    metadata['section_name'] = chunk['section_name']
            
            elif chunk.get('chunk_type') == 'content':
                if chunk.get('sentence_start') is not None:
                    metadata['sentence_start'] = chunk['sentence_start']
                if chunk.get('sentence_end') is not None:
                    metadata['sentence_end'] = chunk['sentence_end']
            
            metadatas.append(metadata)
        
        collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Stored {len(chunks)} enhanced chunks for document {document_id}")
        
        # Log storage statistics
        chunk_stats = {}
        for chunk in chunks:
            chunk_type = chunk.get('chunk_type', 'unknown')
            chunk_stats[chunk_type] = chunk_stats.get(chunk_type, 0) + 1
        logger.info(f"Storage stats: {chunk_stats}")
        
    except Exception as e:
        logger.error(f"Error storing enhanced document chunks: {str(e)}")
        raise

def query_relevant_chunks(query: str, document_id: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Advanced multi-stage retrieval system with intelligent context understanding.
    Designed to match ChatGPT-level document understanding and retrieval.
    """
    try:
        query_embedding = embedder.encode([query])[0].tolist()
        query_lower = query.lower()
        
        # Advanced query analysis
        import re
        
        # Detect query type with more sophisticated patterns
        metadata_keywords = ['title', 'author', 'abstract', 'summary', 'paper', 'document', 'article', 'study', 'research', 'who wrote', 'what is the name']
        figure_keywords = ['figure', 'fig', 'chart', 'graph', 'plot', 'diagram', 'illustration', 'image']
        table_keywords = ['table', 'data', 'results', 'comparison', 'statistics', 'numbers']
        section_keywords = ['section', 'introduction', 'background', 'methods', 'methodology', 'results', 'discussion', 'conclusion', 'abstract']
        equation_keywords = ['equation', 'formula', 'mathematical', 'calculation', 'math']
        
        # Extract specific references
        figure_refs = re.findall(r'figure\s*(\d+)', query_lower)
        table_refs = re.findall(r'table\s*(\d+)', query_lower)
        section_refs = re.findall(r'section\s*(\d+)', query_lower)
        
        # Determine query intent with confidence scoring
        intent_scores = {
            'metadata': sum(1 for kw in metadata_keywords if kw in query_lower),
            'figure': sum(1 for kw in figure_keywords if kw in query_lower) + len(figure_refs) * 2,
            'table': sum(1 for kw in table_keywords if kw in query_lower) + len(table_refs) * 2,
            'section': sum(1 for kw in section_keywords if kw in query_lower) + len(section_refs) * 2,
            'equation': sum(1 for kw in equation_keywords if kw in query_lower),
            'general': 0
        }
        
        # Determine primary intent
        primary_intent = max(intent_scores.items(), key=lambda x: x[1])[0]
        intent_confidence = intent_scores[primary_intent]
        
        logger.info(f"Query analysis - Intent: {primary_intent} (confidence: {intent_confidence})")
        logger.info(f"Specific references - Figures: {figure_refs}, Tables: {table_refs}, Sections: {section_refs}")
        
        # Multi-stage retrieval strategy
        relevant_chunks = []
        
        # Stage 1: Intent-specific targeted search
        if primary_intent == 'metadata' and intent_confidence > 0:
            # Metadata search with high precision
            metadata_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where={"$and": [{"document_id": document_id}, {"chunk_type": "metadata"}]}
            )
            
            if metadata_results['documents'] and metadata_results['documents'][0]:
                for i, doc in enumerate(metadata_results['documents'][0]):
                    relevant_chunks.append({
                        'text': doc,
                        'metadata': metadata_results['metadatas'][0][i],
                        'distance': metadata_results['distances'][0][i] if metadata_results['distances'] else 0,
                        'relevance_score': 1.0 - (metadata_results['distances'][0][i] if metadata_results['distances'] else 0),
                        'source': 'metadata_targeted'
                    })
        
        elif primary_intent == 'figure' and intent_confidence > 0:
            # Multi-step figure search
            
            # Step 1: Direct figure chunk search
            figure_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=k * 2,
                where={"$and": [{"document_id": document_id}, {"chunk_type": "figure"}]}
            )
            
            # Step 2: Prioritize specific figure numbers
            if figure_results['documents'] and figure_results['documents'][0]:
                for i, doc in enumerate(figure_results['documents'][0]):
                    boost_score = 0
                    # Boost score for exact figure number matches
                    if figure_refs:
                        for fig_num in figure_refs:
                            if any(pattern in doc.lower() for pattern in [f'figure {fig_num}', f'fig. {fig_num}', f'fig {fig_num}']):
                                boost_score += 2.0
                    
                    base_score = 1.0 - (figure_results['distances'][0][i] if figure_results['distances'] else 0)
                    final_score = base_score + boost_score
                    
                    relevant_chunks.append({
                        'text': doc,
                        'metadata': figure_results['metadatas'][0][i],
                        'distance': figure_results['distances'][0][i] if figure_results['distances'] else 0,
                        'relevance_score': final_score,
                        'source': 'figure_targeted'
                    })
            
            # Step 3: Search content for figure references and descriptions
            content_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=k * 3,
                where={"$and": [{"document_id": document_id}, {"chunk_type": "content"}]}
            )
            
            if content_results['documents'] and content_results['documents'][0]:
                for i, doc in enumerate(content_results['documents'][0]):
                    # Check for figure mentions and descriptions
                    figure_mention_score = 0
                    doc_lower = doc.lower()
                    
                    # Score based on figure-related content
                    if any(word in doc_lower for word in ['figure', 'fig.', 'shown in', 'illustrated', 'depicts']):
                        figure_mention_score += 0.5
                    
                    # Boost for specific figure numbers
                    if figure_refs:
                        for fig_num in figure_refs:
                            if any(pattern in doc_lower for pattern in [f'figure {fig_num}', f'fig. {fig_num}', f'fig {fig_num}']):
                                figure_mention_score += 1.5
                    
                    if figure_mention_score > 0:
                        base_score = 1.0 - (content_results['distances'][0][i] if content_results['distances'] else 0)
                        final_score = base_score + figure_mention_score
                        
                        relevant_chunks.append({
                            'text': doc,
                            'metadata': content_results['metadatas'][0][i],
                            'distance': content_results['distances'][0][i] if content_results['distances'] else 0,
                            'relevance_score': final_score,
                            'source': 'content_figure_mention'
                        })
        
        elif primary_intent == 'table' and intent_confidence > 0:
            # Similar multi-step approach for tables
            table_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=k * 2,
                where={"$and": [{"document_id": document_id}, {"chunk_type": "table"}]}
            )
            
            if table_results['documents'] and table_results['documents'][0]:
                for i, doc in enumerate(table_results['documents'][0]):
                    boost_score = 0
                    if table_refs:
                        for table_num in table_refs:
                            if f'table {table_num}' in doc.lower():
                                boost_score += 2.0
                    
                    base_score = 1.0 - (table_results['distances'][0][i] if table_results['distances'] else 0)
                    final_score = base_score + boost_score
                    
                    relevant_chunks.append({
                        'text': doc,
                        'metadata': table_results['metadatas'][0][i],
                        'distance': table_results['distances'][0][i] if table_results['distances'] else 0,
                        'relevance_score': final_score,
                        'source': 'table_targeted'
                    })
        
        elif primary_intent == 'section' and intent_confidence > 0:
            # Section-specific search
            section_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=k * 2,
                where={"$and": [{"document_id": document_id}, {"chunk_type": "section"}]}
            )
            
            if section_results['documents'] and section_results['documents'][0]:
                for i, doc in enumerate(section_results['documents'][0]):
                    base_score = 1.0 - (section_results['distances'][0][i] if section_results['distances'] else 0)
                    
                    relevant_chunks.append({
                        'text': doc,
                        'metadata': section_results['metadatas'][0][i],
                        'distance': section_results['distances'][0][i] if section_results['distances'] else 0,
                        'relevance_score': base_score,
                        'source': 'section_targeted'
                    })
        
        # Stage 2: General semantic search to fill remaining slots
        remaining_slots = k - len(relevant_chunks)
        if remaining_slots > 0:
            general_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=remaining_slots * 2,  # Get extra to filter out duplicates
                where={"document_id": document_id}
            )
            
            # Track already included chunks to avoid duplicates
            existing_texts = {chunk['text'] for chunk in relevant_chunks}
            
            if general_results['documents'] and general_results['documents'][0]:
                for i, doc in enumerate(general_results['documents'][0]):
                    if doc not in existing_texts and len(relevant_chunks) < k:
                        base_score = 1.0 - (general_results['distances'][0][i] if general_results['distances'] else 0)
                        
                        relevant_chunks.append({
                            'text': doc,
                            'metadata': general_results['metadatas'][0][i],
                            'distance': general_results['distances'][0][i] if general_results['distances'] else 0,
                            'relevance_score': base_score,
                            'source': 'general_semantic'
                        })
        
        # Stage 3: Intelligent re-ranking based on query context
        def calculate_contextual_relevance(chunk, query_lower):
            """Calculate additional relevance based on query context."""
            text_lower = chunk['text'].lower()
            context_score = 0
            
            # Keyword matching with weights
            query_words = [word for word in query_lower.split() if len(word) > 2]
            word_matches = sum(1 for word in query_words if word in text_lower)
            context_score += (word_matches / len(query_words)) * 0.5 if query_words else 0
            
            # Question-specific scoring
            if any(q_word in query_lower for q_word in ['what', 'how', 'why', 'when', 'where', 'who']):
                # Boost chunks that might contain answers
                if any(ans_word in text_lower for ans_word in ['because', 'therefore', 'due to', 'result', 'shows', 'demonstrates']):
                    context_score += 0.3
            
            # Temporal/causal language
            if any(temp_word in query_lower for temp_word in ['process', 'method', 'approach', 'technique']):
                if any(method_word in text_lower for method_word in ['first', 'then', 'next', 'finally', 'step', 'procedure']):
                    context_score += 0.3
            
            return context_score
        
        # Apply contextual re-ranking
        for chunk in relevant_chunks:
            contextual_boost = calculate_contextual_relevance(chunk, query_lower)
            chunk['relevance_score'] += contextual_boost
            chunk['contextual_boost'] = contextual_boost
        
        # Sort by final relevance score
        relevant_chunks.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Stage 4: Ensure we have the best k chunks
        final_chunks = relevant_chunks[:k]
        
        # Log retrieval performance
        if final_chunks:
            avg_score = sum(chunk['relevance_score'] for chunk in final_chunks) / len(final_chunks)
            sources = [chunk['source'] for chunk in final_chunks]
            logger.info(f"Retrieved {len(final_chunks)} chunks with avg relevance: {avg_score:.3f}")
            logger.info(f"Sources: {sources}")
        
        return final_chunks
        
    except Exception as e:
        logger.error(f"Error in advanced chunk retrieval: {str(e)}")
        raise

def generate_answer_simple(question: str, context_chunks: List[Dict[str, Any]]) -> str:
    """Generate a simple answer based on context without LLM."""
    if not context_chunks:
        return "No relevant information found in the document."
    
    # Check if this is a document metadata question
    metadata_keywords = ['title', 'author', 'abstract', 'summary', 'paper', 'document', 'article', 'study', 'research']
    is_metadata_query = any(keyword in question.lower() for keyword in metadata_keywords)
    
    if is_metadata_query:
        # For metadata questions, prioritize metadata chunks and look for specific patterns
        metadata_chunks = [chunk for chunk in context_chunks if chunk['metadata'].get('chunk_type') == 'metadata']
        
        if metadata_chunks:
            # Use the metadata chunk (full first page)
            best_chunk = metadata_chunks[0]
            text = best_chunk['text']
            
            # Try to extract specific information based on question type
            question_lower = question.lower()
            
            if 'title' in question_lower:
                # Look for title patterns - usually at the beginning, often in caps or bold formatting
                # Split into sentences and look for the first substantial sentence
                sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]
                if sentences:
                    # The title is often the first substantial text
                    potential_title = sentences[0].strip()
                    # Clean up common artifacts
                    potential_title = potential_title.replace('arXiv:', '').strip()
                    if len(potential_title) > 5 and len(potential_title) < 200:
                        return f"The title appears to be: \"{potential_title}\" (from page {best_chunk['metadata']['page']})"
            
            elif 'author' in question_lower:
                # Look for author patterns - names often appear after title
                # Common patterns: "FirstName LastName", "F. LastName", email addresses
                import re
                
                # Look for email patterns first (authors often have emails)
                email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                emails = re.findall(email_pattern, text)
                
                # Look for name patterns (capitalized words that could be names)
                name_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
                potential_names = re.findall(name_pattern, text)
                
                authors = []
                if potential_names:
                    # Filter out common non-name words
                    common_words = ['The', 'This', 'That', 'With', 'From', 'And', 'For', 'In', 'On', 'At', 'By']
                    filtered_names = [name for name in potential_names if not any(word in name for word in common_words)]
                    authors.extend(filtered_names[:5])  # Limit to first 5 potential names
                
                if authors:
                    author_list = ', '.join(authors)
                    return f"The authors appear to be: {author_list} (from page {best_chunk['metadata']['page']})"
                elif emails:
                    return f"Found author email addresses: {', '.join(emails[:3])} (from page {best_chunk['metadata']['page']})"
            
            # Fallback: return relevant portion of metadata chunk
            return f"Based on the document metadata: {text[:500]}{'...' if len(text) > 500 else ''} (from page {best_chunk['metadata']['page']})"
    
    # Regular processing for non-metadata questions
    best_chunk = context_chunks[0]  # Most relevant chunk
    
    # Try to find sentences that might contain the answer
    sentences = best_chunk['text'].split('.')
    relevant_sentences = []
    
    question_words = question.lower().split()
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 20:  # Ignore very short sentences
            # Check if sentence contains question-related words
            sentence_lower = sentence.lower()
            if any(word in sentence_lower for word in question_words if len(word) > 3):
                relevant_sentences.append(sentence)
    
    if relevant_sentences:
        answer = ". ".join(relevant_sentences[:2])  # Take top 2 relevant sentences
    else:
        answer = best_chunk['text'][:400]  # Fallback to beginning of chunk
    
    # Add page reference
    page_refs = list(set([chunk['metadata']['page'] for chunk in context_chunks[:2]]))
    page_info = f" (from page{'s' if len(page_refs) > 1 else ''} {', '.join(map(str, page_refs))})"
    
    return f"{answer.strip()}{page_info}"

def generate_answer(question: str, context_chunks: List[Dict[str, Any]]) -> str:
    """
    Advanced answer generation with multi-step reasoning and context synthesis.
    Designed to match ChatGPT-level understanding and response quality.
    """
    try:
        if not llm_available:
            return generate_answer_simple(question, context_chunks)
        
        # Advanced query analysis for response strategy
        question_lower = question.lower()
        
        # Analyze question type for tailored response approach
        question_types = {
            'factual': any(word in question_lower for word in ['what is', 'who is', 'when', 'where', 'which']),
            'explanatory': any(word in question_lower for word in ['why', 'how', 'explain', 'describe']),
            'comparative': any(word in question_lower for word in ['compare', 'difference', 'versus', 'vs', 'better']),
            'analytical': any(word in question_lower for word in ['analyze', 'evaluate', 'assess', 'significance']),
            'procedural': any(word in question_lower for word in ['process', 'method', 'approach', 'steps', 'procedure']),
            'quantitative': any(word in question_lower for word in ['how much', 'how many', 'percentage', 'rate', 'number']),
            'summary': any(word in question_lower for word in ['summarize', 'overview', 'main points', 'bullet points'])
        }
        
        primary_type = max(question_types.items(), key=lambda x: x[1])[0] if any(question_types.values()) else 'general'
        
        # Detect specific elements being asked about
        specific_elements = {
            'title': any(word in question_lower for word in ['title', 'name of paper', 'called']),
            'authors': any(word in question_lower for word in ['author', 'wrote', 'written by']),
            'abstract': any(word in question_lower for word in ['abstract', 'summary']),
            'figures': any(word in question_lower for word in ['figure', 'fig', 'chart', 'graph']),
            'tables': any(word in question_lower for word in ['table', 'data']),
            'methods': any(word in question_lower for word in ['method', 'approach', 'technique']),
            'results': any(word in question_lower for word in ['result', 'finding', 'outcome']),
            'conclusion': any(word in question_lower for word in ['conclusion', 'conclude'])
        }
        
        active_elements = [elem for elem, active in specific_elements.items() if active]
        
        logger.info(f"Question analysis - Type: {primary_type}, Elements: {active_elements}")
        
        # Organize context by relevance and type
        context_by_type = {
            'metadata': [],
            'figure': [],
            'table': [],
            'section': [],
            'content': []
        }
        
        for chunk in context_chunks:
            chunk_type = chunk['metadata'].get('chunk_type', 'content')
            context_by_type[chunk_type].append(chunk)
        
        # Extract specific numbers or references mentioned in question
        import re
        mentioned_figures = re.findall(r'figure\s*(\d+)', question_lower)
        mentioned_tables = re.findall(r'table\s*(\d+)', question_lower)
        
        # Build comprehensive context with smart prioritization
        context_sections = []
        
        # Prioritize based on question intent
        if 'title' in active_elements or 'authors' in active_elements:
            # Metadata questions get priority
            for chunk in context_by_type['metadata']:
                context_sections.append(f"[METADATA - Page {chunk['metadata']['page']}]:\n{chunk['text']}")
        
        if 'figures' in active_elements or mentioned_figures:
            # Figure-specific context
            relevant_figures = []
            if mentioned_figures:
                # Prioritize specific figures
                for chunk in context_by_type['figure']:
                    chunk_text = chunk['text'].lower()
                    for fig_num in mentioned_figures:
                        if any(pattern in chunk_text for pattern in [f'figure {fig_num}', f'fig. {fig_num}', f'fig {fig_num}']):
                            relevant_figures.insert(0, chunk)  # Prioritize
                            break
                    else:
                        relevant_figures.append(chunk)
            else:
                relevant_figures = context_by_type['figure']
            
            for chunk in relevant_figures[:2]:  # Limit to avoid overwhelming
                context_sections.append(f"[FIGURE - Page {chunk['metadata']['page']}]:\n{chunk['text']}")
        
        if 'tables' in active_elements or mentioned_tables:
            # Table-specific context
            relevant_tables = context_by_type['table']
            if mentioned_tables:
                # Prioritize specific tables
                relevant_tables = []
                for chunk in context_by_type['table']:
                    chunk_text = chunk['text'].lower()
                    for table_num in mentioned_tables:
                        if f'table {table_num}' in chunk_text:
                            relevant_tables.insert(0, chunk)
                            break
                    else:
                        relevant_tables.append(chunk)
            
            for chunk in relevant_tables[:2]:
                context_sections.append(f"[TABLE - Page {chunk['metadata']['page']}]:\n{chunk['text']}")
        
        # Add section context if relevant
        if context_by_type['section'] and any(elem in active_elements for elem in ['methods', 'results', 'conclusion']):
            for chunk in context_by_type['section'][:2]:
                context_sections.append(f"[SECTION - Page {chunk['metadata']['page']}]:\n{chunk['text']}")
        
        # Add high-quality content chunks
        content_chunks_sorted = sorted(context_by_type['content'], 
                                     key=lambda x: x.get('relevance_score', 0), 
                                     reverse=True)
        
        for chunk in content_chunks_sorted[:3]:  # Top 3 content chunks
            context_sections.append(f"[CONTENT - Page {chunk['metadata']['page']}]:\n{chunk['text']}")
        
        combined_context = "\n\n".join(context_sections)
        
        # Create sophisticated prompt based on question type
        if primary_type == 'factual' and 'title' in active_elements:
            prompt = f"""You are an expert document analyst. Based on the provided context, identify and extract the exact title of the document.

Context from the document:
{combined_context}

Question: {question}

ANALYSIS INSTRUCTIONS:
1. Look for the document title, which is typically:
   - At the very beginning of the document
   - The first substantial piece of text
   - Usually formatted prominently
   - Often followed by author names or institutional affiliations

2. The title should be:
   - Complete and accurate
   - Free from formatting artifacts
   - The main title, not a section header

3. If you find multiple title-like elements, choose the most prominent one that represents the main document title.

Provide a direct, accurate answer stating the title clearly."""

        elif primary_type == 'factual' and 'authors' in active_elements:
            prompt = f"""You are an expert document analyst. Based on the provided context, identify and list the authors of the document.

Context from the document:
{combined_context}

Question: {question}

ANALYSIS INSTRUCTIONS:
1. Look for author names, which typically appear:
   - After the title
   - Before institutional affiliations
   - In a specific author section
   - Sometimes with superscript numbers linking to affiliations

2. Author names usually follow patterns like:
   - "FirstName LastName"
   - "F. LastName" 
   - "FirstName MiddleInitial LastName"

3. Institutional affiliations, email addresses, and department names are NOT author names.

4. Present the authors in the order they appear in the document.

Provide a clear list of the document authors."""

        elif 'figures' in active_elements or mentioned_figures:
            specific_refs = f" specifically Figure {', '.join(mentioned_figures)}" if mentioned_figures else ""
            prompt = f"""You are an expert document analyst. Based on the provided context, provide detailed information about the figure(s){specific_refs} mentioned in the question.

Context from the document:
{combined_context}

Question: {question}

ANALYSIS INSTRUCTIONS:
1. Look for figure descriptions, captions, and references in the context.

2. For each relevant figure, provide:
   - What the figure shows or illustrates
   - The purpose or significance of the figure
   - Key findings or insights presented
   - Any methodological details if relevant

3. If the figure is referenced but not fully described, explain what you can determine from the available context.

4. Be specific about which figure you're discussing (Figure 1, Figure 2, etc.).

5. If multiple figures are relevant, address each one separately.

Provide a comprehensive explanation of the figure(s) based on the available information."""

        elif 'tables' in active_elements or mentioned_tables:
            specific_refs = f" specifically Table {', '.join(mentioned_tables)}" if mentioned_tables else ""
            prompt = f"""You are an expert document analyst. Based on the provided context, provide detailed information about the table(s){specific_refs} mentioned in the question.

Context from the document:
{combined_context}

Question: {question}

ANALYSIS INSTRUCTIONS:
1. Look for table descriptions, captions, data presentations, and references in the context.

2. For each relevant table, provide:
   - What data the table presents
   - The structure and organization of the data
   - Key findings or patterns in the data
   - Statistical significance if mentioned
   - Comparisons being made

3. If specific data points are mentioned, include them in your response.

4. Be specific about which table you're discussing (Table 1, Table 2, etc.).

Provide a comprehensive explanation of the table(s) and their significance."""

        elif primary_type == 'summary':
            prompt = f"""You are an expert document analyst. Based on the provided context, create a comprehensive and well-structured summary as requested.

Context from the document:
{combined_context}

Question: {question}

ANALYSIS INSTRUCTIONS:
1. Create a structured response that addresses the specific request (bullet points, overview, etc.).

2. Organize information logically and coherently.

3. Include the most important and relevant information from the context.

4. If bullet points are requested, create clear, concise points.

5. Ensure each point or section adds value and insight.

6. Reference specific pages when providing information.

Provide a well-organized and informative summary based on the available context."""

        elif primary_type == 'explanatory':
            prompt = f"""You are an expert document analyst. Based on the provided context, provide a detailed explanation that addresses the "how" or "why" aspects of the question.

Context from the document:
{combined_context}

Question: {question}

ANALYSIS INSTRUCTIONS:
1. Provide a thorough explanation that addresses the underlying mechanisms, reasons, or processes.

2. Use specific evidence from the context to support your explanation.

3. If the question asks "how," focus on processes, methods, and mechanisms.

4. If the question asks "why," focus on reasons, causes, and motivations.

5. Connect different pieces of information to create a coherent explanation.

6. Include relevant details that enhance understanding.

Provide a comprehensive explanation based on the available evidence."""

        else:
            # General sophisticated prompt
            prompt = f"""You are an expert document analyst with deep comprehension capabilities. Based on the provided context, answer the question with thorough analysis and insight.

Context from the document:
{combined_context}

Question: {question}

ANALYSIS INSTRUCTIONS:
1. Carefully analyze all provided context to understand the full scope of relevant information.

2. Synthesize information from multiple sources within the context to provide a comprehensive answer.

3. Be specific and accurate, using only information supported by the context.

4. If the context contains partial information, clearly indicate what you can determine and what limitations exist.

5. Provide specific page references for key information.

6. Structure your response clearly and logically.

7. If the question cannot be fully answered from the context, explain what information is available and what is missing.

Provide a thorough, well-reasoned response based on careful analysis of the context."""

        try:
            # Generate response using Ollama with optimized parameters
            logger.info(f"Generating advanced answer for {primary_type} question about {active_elements}")
            
            response = ollama.generate(
                model=OLLAMA_MODEL,
                prompt=prompt,
                options={
                    'temperature': 0.1,  # Low temperature for accuracy
                    'top_p': 0.9,
                    'top_k': 40,
                    'num_predict': 400,  # Allow for detailed responses
                    'repeat_penalty': 1.1,
                }
            )
            
            answer = response['response'].strip()
            
            # Advanced post-processing
            if answer:
                # Clean up any leaked instruction text
                cleanup_patterns = [
                    r'ANALYSIS INSTRUCTIONS:.*$',
                    r'Context from the document:.*?Question:.*?(?=ANALYSIS)',
                    r'Based on the provided context[,:]?\s*',
                    r'The context (?:shows|indicates|suggests|provides).*?,?\s*'
                ]
                
                for pattern in cleanup_patterns:
                    answer = re.sub(pattern, '', answer, flags=re.IGNORECASE | re.DOTALL)
                
                # Improve formatting
                answer = answer.strip()
                
                # Add confidence and source indicators
                source_pages = list(set([chunk['metadata']['page'] for chunk in context_chunks[:3]]))
                confidence_indicators = [
                    chunk.get('relevance_score', 0) for chunk in context_chunks[:3]
                ]
                avg_confidence = sum(confidence_indicators) / len(confidence_indicators) if confidence_indicators else 0
                
                # Add contextual information if not already present
                if not any(f"page {page}" in answer.lower() for page in source_pages):
                    if len(source_pages) == 1:
                        answer += f"\n\n(Based on analysis of page {source_pages[0]})"
                    else:
                        answer += f"\n\n(Based on analysis of pages {', '.join(map(str, source_pages))})"
                
                # Add confidence note for low-confidence answers
                if avg_confidence < 0.6:
                    answer += f"\n\n*Note: This answer is based on limited context from the document. Additional information may be available in other sections.*"
                
                logger.info(f"Generated advanced answer with confidence: {avg_confidence:.3f}")
                return answer
            
            # Fallback if processing failed
            logger.warning("Advanced answer processing failed, using enhanced fallback")
            return generate_answer_simple(question, context_chunks)
            
        except Exception as e:
            logger.error(f"Error with advanced Llama generation: {str(e)}")
            return generate_answer_simple(question, context_chunks)
        
    except Exception as e:
        logger.error(f"Error in advanced answer generation: {str(e)}")
        return f"Error generating answer: {str(e)}"

@app.get("/")
async def hello_world():
    return {"message": "Hello World from AI Project Backend (Enhanced with Llama 3.2)!"}

@app.get("/health")
async def health_check():
    llm_status = "loaded" if llm_available else "fallback mode"
    return {
        "status": "healthy", 
        "message": f"Enhanced API is running with real RAG functionality (LLM: {llm_status})"
    }

@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    save_to_vector_store: bool = Form(True)
):
    """
    Enhanced upload endpoint with real PDF processing and vector storage.
    """
    try:
        print(f"Upload request received - filename: {file.filename}, content_type: {file.content_type}")
        
        # Validate file type
        if not file.filename or not file.filename.lower().endswith('.pdf'):
            print(f"Invalid file type: {file.filename}")
            raise HTTPException(status_code=422, detail="Only PDF files are allowed")
        
        # Read file content
        content = await file.read()
        print(f"File content length: {len(content)} bytes")
        
        # Generate document ID
        document_id = str(uuid.uuid4())
        print(f"Generated document ID: {document_id}")
        
        # Extract text from PDF
        text_chunks = extract_text_from_pdf(content)
        print(f"Extracted {len(text_chunks)} text chunks")
        
        # Store in vector database if requested
        embeddings_generated = False
        if save_to_vector_store and text_chunks:
            store_document_chunks(document_id, file.filename, text_chunks)
            embeddings_generated = True
        
        response = UploadResponse(
            document_id=document_id,
            filename=file.filename,
            total_pages=max([chunk['page'] for chunk in text_chunks]) if text_chunks else 0,
            total_chunks=len(text_chunks),
            embeddings_generated=embeddings_generated,
            message=f"Successfully processed {file.filename} with real RAG functionality (Llama 3.2)."
        )
        
        print(f"Returning successful response for {file.filename}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Enhanced ask endpoint with real RAG functionality using Llama 3.2.
    """
    try:
        print(f"Question received: '{request.question}' for document: {request.file_id}")
        
        # Query relevant chunks
        relevant_chunks = query_relevant_chunks(request.question, request.file_id, k=5)
        
        if not relevant_chunks:
            return AskResponse(
                answer=f"I couldn't find any relevant information in the document to answer your question. Please make sure the document has been uploaded and processed."
            )
        
        print(f"Found {len(relevant_chunks)} relevant chunks")
        
        # Generate answer using Llama 3.2
        answer = generate_answer(request.question, relevant_chunks)
        
        return AskResponse(answer=answer)
        
    except Exception as e:
        print(f"Error answering question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_enhanced:app", host="0.0.0.0", port=8000, reload=True) 