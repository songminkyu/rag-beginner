"""
Core module for RAG system.

This module provides the fundamental components for building RAG systems:
- LLM providers for different AI models
- Data processing utilities
- Retrieval systems
- Evaluation metrics
"""

__version__ = "0.1.0"

from .llm_providers import (
    BaseLLMProvider,
    OpenAIProvider,
    ClaudeProvider,
    LocalLLMProvider,
)

from .data_processing import (
    DocumentLoader,
    TextSplitter,
    EmbeddingGenerator,
    VectorStore,
)

from .retrieval import (
    BaseRetriever,
    SemanticRetriever,
    KeywordRetriever,
    HybridRetriever,
)

from .evaluation import (
    RAGEvaluator,
    Metrics,
    Benchmark,
)

__all__ = [
    # LLM Providers
    "BaseLLMProvider",
    "OpenAIProvider", 
    "ClaudeProvider",
    "LocalLLMProvider",
    
    # Data Processing
    "DocumentLoader",
    "TextSplitter",
    "EmbeddingGenerator",
    "VectorStore",
    
    # Retrieval
    "BaseRetriever",
    "SemanticRetriever", 
    "KeywordRetriever",
    "HybridRetriever",
    
    # Evaluation
    "RAGEvaluator",
    "Metrics",
    "Benchmark",
]