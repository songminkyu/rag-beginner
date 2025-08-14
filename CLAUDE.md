# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a comprehensive RAG (Retrieval-Augmented Generation) learning repository focused on 2025 LLM RAG technology, with special optimization for Korean language processing using EXAONE 4.0 models. The repository provides systematic tutorials, real-world project examples, and supports multiple LLM providers.

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# Install dependencies
pip install -e .
# Or with uv (if available):
uv pip install -e .
```

### Running Examples and Tutorials
```bash
# Basic RAG example
python src/tutorials/01_getting_started/hello_rag.py

# API comparison
python src/tutorials/01_getting_started/api_comparison.py

# Run specific examples
python src/examples/model_example/gpt_oss.py
python src/examples/multimodal_rag_example/multimodal_rag.py
```

### EXAONE Model Setup
```bash
# Setup Ollama and EXAONE models (Linux/Mac)
chmod +x scripts/setup_ollama.sh
./scripts/setup_ollama.sh

# Setup EXAONE Python environment
python scripts/setup_exaone.py
```

### Testing
```bash
# Run basic tests (if pytest is available)
pytest src/tests/ -v

# Test individual components
python src/main.py  # Main entry point test
python src/utils/jax_flax_test.py  # JAX/Flax compatibility test
```

### Linting and Formatting
```bash
# Format code with black
black src/

# Lint with flake8
flake8 src/

# Type checking with mypy
mypy src/
```

## Repository Architecture

### Core System Design

The RAG system follows a modular architecture with clear separation of concerns:

**Core Components (`src/core/`)**:
- **LLM Providers** (`llm_providers/`): Abstracted LLM interfaces supporting OpenAI, Claude, and flexible local models
  - `BaseLLMProvider`: Standard interface for all LLM providers
  - `OpenAIProvider`, `ClaudeProvider`, `FlexibleLocalProvider`: Specific implementations
  - `FlexibleLocalProvider`: Supports multiple local model types (Ollama + HuggingFace)
  - Unified response format through `LLMResponse` and `ChatMessage` classes
- **Data Processing** (`data_processing/`): Document processing and embedding generation
  - `VectorStore`: Multiple vector database backends (ChromaDB, FAISS, In-Memory)
  - `EmbeddingGenerator`: Text-to-vector conversion with Korean optimization
  - `DocumentLoader`, `TextSplitter`: Document ingestion and chunking
- **Retrieval System** (`retrieval/`): Advanced search and retrieval mechanisms
  - `BaseRetriever`: Standard retrieval interface
  - `HybridRetriever`: Combines semantic + keyword search
  - Support for query expansion, reranking, and filtering
- **Evaluation** (`evaluation/`): RAG system assessment tools
  - RAGAS, BLEU, BERTScore metrics integration
  - Automated benchmarking and performance analysis

### Framework Integration (`src/frameworks/`)

**LangChain Examples** (`langchain_examples/`):
- Basic RAG: Simple Q&A, document chat, retrieval chains
- Advanced RAG: Conversational RAG with memory, multi-query RAG, hierarchical retrieval
- Agentic RAG: Agent-based RAG systems

### Configuration Management (`src/config/`)

Centralized configuration system supporting:
- API keys and model configurations
- Vector store settings (ChromaDB, FAISS, Pinecone)
- Provider-specific optimizations for Korean vs English processing

### Flexible Local Model Integration

The system provides flexible support for multiple local model types:
- **Strategy Pattern**: Automatic selection between Ollama and HuggingFace backends
- **Multi-Model Support**: Llama, Mistral, EXAONE, CodeLlama, Phi, Qwen, and more
- **Korean Optimization**: Special handling for Korean models (EXAONE, KoGPT, etc.)
- **Auto-Detection**: Intelligent model type detection and parameter optimization
- **Fallback System**: Automatic fallback between different model backends

### Vector Store Strategy

Multi-backend vector storage with intelligent selection:
- **InMemory**: Development and prototyping
- **ChromaDB**: Medium-scale applications with persistence
- **FAISS**: High-performance production deployments
- **Pinecone**: Cloud-scale vector search

### Advanced Retrieval Patterns

The retrieval system supports sophisticated search strategies:
- **Hybrid Search**: Semantic + lexical retrieval with RRF (Reciprocal Rank Fusion)
- **Query Enhancement**: Automatic query expansion and reformulation
- **Multi-stage Reranking**: Cross-encoder and LLM-based reranking
- **Metadata Filtering**: Advanced filtering with range queries and operators

## Key Patterns

### Provider Initialization Pattern
All LLM providers follow a consistent initialization pattern:
```python
provider = create_provider(provider_type, config)
response = provider.generate(prompt, system_prompt=system_prompt)
```

### RAG Pipeline Pattern
Standard RAG execution follows this pattern:
```python
# 1. Document processing
documents = loader.load_documents(source)
chunks = splitter.split_documents(documents)

# 2. Vector storage
vector_store.add_documents(chunks, embedding_generator)

# 3. Retrieval
results = retriever.retrieve(query, k=5)

# 4. Generation
context = format_context(results)
answer = llm_provider.generate(query, context)
```

### Korean Optimization Pattern
Korean text processing requires specific considerations:
- Use Korean-optimized embedding models (multilingual variants)
- EXAONE models for Korean-specific prompting
- Proper tokenization handling for Hangul text

### Error Handling Pattern
The system uses defensive error handling:
- Graceful fallbacks between provider types (local → OpenAI → Claude)
- Comprehensive logging with structured error information
- Provider availability checking before initialization

## Environment Variables

Required environment variables for full functionality:
```bash
# OpenAI (optional)
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Anthropic Claude (optional)
ANTHROPIC_API_KEY=your_anthropic_key
CLAUDE_MODEL=claude-3-5-sonnet-20241022

# Local Models (flexible)
LOCAL_MODEL_NAME=llama3.1:8b
LOCAL_MODEL_TYPE=auto
HF_MODEL_NAME=microsoft/DialoGPT-medium
OLLAMA_BASE_URL=http://localhost:11434
TORCH_DTYPE=float16
KOREAN_OPTIMIZED=true

# Vector Store Configuration
CHROMA_PERSIST_DIR=./chroma_db
FAISS_INDEX_DIR=./faiss_indexes
```

## Important Implementation Details

### Memory Management
- Vector stores implement lazy loading and caching strategies
- EXAONE models support dynamic batch sizing based on available GPU memory
- Document chunking optimized for Korean sentence boundaries

### Korean Language Processing
- **Multi-Model Korean Support**: EXAONE, KoGPT, KcELECTRA, and multilingual models
- **Automatic Korean Optimization**: Korean-specific prompt templates and parameter tuning
- **Korean Embeddings**: Support for ko-sroberta-multitask, KoSimCSE, and multilingual models
- **Intelligent Model Selection**: Automatic selection of Korean-optimized models when available

### Model Compatibility
- **Flexible Local Models**: Supports Ollama (llama3.1, mistral, exaone4, etc.) and HuggingFace models
- **Automatic Strategy Selection**: Intelligent choice between Ollama and HuggingFace backends
- **Fallback Chain**: Local (flexible) → OpenAI → Claude with graceful degradation
- **Provider Capability Checking**: Embedding support, streaming, model availability validation
- **Unified Response Format**: Consistent interface across all providers and strategies

### Performance Considerations
- **Vector Store Scaling**: FAISS for large-scale (>10K docs), ChromaDB for development
- **Local Model Optimization**: Dynamic memory management, torch dtype selection, caching
- **Strategy-Based Performance**: Ollama for served models, HuggingFace for fine-tuned models
- **Batch Processing**: Optimized embedding generation and inference batching
- **GPU Acceleration**: Automatic device detection and memory optimization