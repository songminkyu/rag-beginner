"""
API Configuration for LLM Providers
2025년 최신 API 설정 관리
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class OpenAIConfig:
    """OpenAI API 설정"""
    api_key: str
    organization_id: Optional[str] = None
    model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-large"
    max_tokens: int = 4096
    temperature: float = 0.1
    
    @classmethod
    def from_env(cls) -> "OpenAIConfig":
        return cls(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            organization_id=os.getenv("OPENAI_ORG_ID"),
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "4096")),
            temperature=float(os.getenv("TEMPERATURE", "0.1")),
        )


@dataclass
class ClaudeConfig:
    """Claude (Anthropic) API 설정"""
    api_key: str
    model: str = "claude-3-5-sonnet-20241022"
    max_tokens: int = 8192
    temperature: float = 0.1
    
    @classmethod
    def from_env(cls) -> "ClaudeConfig":
        return cls(
            api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            model=os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022"),
            max_tokens=int(os.getenv("CLAUDE_MAX_TOKENS", "8192")),
            temperature=float(os.getenv("TEMPERATURE", "0.1")),
        )


@dataclass
class LocalModelConfig:
    """로컬 모델 (EXAONE via Hugging Face Transformers) 설정"""
    model: str = "LGAI-EXAONE/EXAONE-4.0-1.2B"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    device: str = "auto"
    torch_dtype: str = "bfloat16"
    max_tokens: int = 4096
    temperature: float = 0.1
    korean_optimized: bool = True
    low_cpu_mem_usage: bool = True
    use_cache: bool = True
    
    @classmethod
    def from_env(cls) -> "LocalModelConfig":
        return cls(
            model=os.getenv("EXAONE_MODEL", "LGAI-EXAONE/EXAONE-4.0-1.2B"),
            embedding_model=os.getenv("LOCAL_EMBEDDING_MODEL", 
                                   "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
            device=os.getenv("DEVICE", "auto"),
            torch_dtype=os.getenv("TORCH_DTYPE", "bfloat16"),
            max_tokens=int(os.getenv("MAX_TOKENS", "4096")),
            temperature=float(os.getenv("TEMPERATURE", "0.1")),
            korean_optimized=os.getenv("KOREAN_OPTIMIZED", "true").lower() == "true",
            low_cpu_mem_usage=os.getenv("LOW_CPU_MEM_USAGE", "true").lower() == "true",
            use_cache=os.getenv("USE_CACHE", "true").lower() == "true",
        )


@dataclass
class VectorStoreConfig:
    """벡터 스토어 설정"""
    store_type: str = "chroma"
    path: str = "./data/vector_stores/chromadb"
    dimension: int = 1536
    collection_name: str = "rag_documents"
    
    # ChromaDB specific
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    
    # Pinecone specific
    pinecone_api_key: Optional[str] = None
    pinecone_environment: Optional[str] = None
    pinecone_index_name: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> "VectorStoreConfig":
        return cls(
            store_type=os.getenv("VECTOR_STORE_TYPE", "chroma"),
            path=os.getenv("VECTOR_STORE_PATH", "./data/vector_stores/chromadb"),
            dimension=int(os.getenv("VECTOR_DIMENSION", "1536")),
            collection_name=os.getenv("CHROMA_COLLECTION_NAME", "rag_documents"),
            chroma_host=os.getenv("CHROMA_HOST", "localhost"),
            chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
            pinecone_api_key=os.getenv("PINECONE_API_KEY"),
            pinecone_environment=os.getenv("PINECONE_ENVIRONMENT"),
            pinecone_index_name=os.getenv("PINECONE_INDEX_NAME"),
        )


@dataclass
class ProcessingConfig:
    """데이터 처리 설정"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_documents: int = 1000
    supported_extensions: list = None
    
    def __post_init__(self):
        if self.supported_extensions is None:
            self.supported_extensions = [".pdf", ".docx", ".txt", ".md", ".html"]
    
    @classmethod
    def from_env(cls) -> "ProcessingConfig":
        extensions_str = os.getenv("SUPPORTED_EXTENSIONS", ".pdf,.docx,.txt,.md,.html")
        extensions = [ext.strip() for ext in extensions_str.split(",")]
        
        return cls(
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            max_documents=int(os.getenv("MAX_DOCUMENTS", "1000")),
            supported_extensions=extensions,
        )


@dataclass
class RetrievalConfig:
    """검색 설정"""
    top_k: int = 5
    similarity_threshold: float = 0.7
    rerank_top_k: int = 3
    search_type: str = "similarity"  # similarity, mmr, similarity_score_threshold
    
    @classmethod
    def from_env(cls) -> "RetrievalConfig":
        return cls(
            top_k=int(os.getenv("RETRIEVAL_TOP_K", "5")),
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.7")),
            rerank_top_k=int(os.getenv("RERANK_TOP_K", "3")),
            search_type=os.getenv("SEARCH_TYPE", "similarity"),
        )


class APIConfigManager:
    """API 설정 관리자"""
    
    def __init__(self):
        self.openai = OpenAIConfig.from_env()
        self.claude = ClaudeConfig.from_env()
        self.local = LocalModelConfig.from_env()
        self.vector_store = VectorStoreConfig.from_env()
        self.processing = ProcessingConfig.from_env()
        self.retrieval = RetrievalConfig.from_env()
    
    def get_provider_config(self, provider: str) -> Dict[str, Any]:
        """특정 제공자의 설정 반환"""
        configs = {
            "openai": self.openai,
            "claude": self.claude,
            "local": self.local,
        }
        
        config = configs.get(provider.lower())
        if not config:
            raise ValueError(f"Unsupported provider: {provider}")
        
        return config.__dict__
    
    def validate_config(self, provider: str) -> bool:
        """설정 유효성 검사"""
        if provider.lower() == "openai":
            return bool(self.openai.api_key)
        elif provider.lower() == "claude":
            return bool(self.claude.api_key)
        elif provider.lower() == "local":
            # 로컬 모델은 GPU/CPU 사용 가능 여부 확인
            try:
                import torch
                return torch.cuda.is_available() or True  # CPU라도 사용 가능
            except ImportError:
                return False
        
        return False
    
    def get_available_providers(self) -> list:
        """사용 가능한 제공자 목록 반환"""
        providers = []
        
        if self.validate_config("openai"):
            providers.append("openai")
        if self.validate_config("claude"):
            providers.append("claude")
        if self.validate_config("local"):
            providers.append("local")
        
        return providers


# 전역 설정 인스턴스
config_manager = APIConfigManager()


def get_config_summary() -> Dict[str, Any]:
    """설정 요약 정보 반환"""
    return {
        "available_providers": config_manager.get_available_providers(),
        "vector_store_type": config_manager.vector_store.store_type,
        "chunk_size": config_manager.processing.chunk_size,
        "retrieval_top_k": config_manager.retrieval.top_k,
        "supported_extensions": config_manager.processing.supported_extensions,
    }


if __name__ == "__main__":
    # 설정 테스트
    print("=== LLM RAG Learning Repository Configuration ===")
    print(f"Available providers: {config_manager.get_available_providers()}")
    print(f"Vector store: {config_manager.vector_store.store_type}")
    print(f"Configuration summary: {get_config_summary()}")