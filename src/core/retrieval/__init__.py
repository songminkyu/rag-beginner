"""
Retrieval Package
다양한 검색 방식을 지원하는 검색기 패키지
"""

from .base_retriever import (
    BaseRetriever,
    RetrievalResult,
    RetrievedDocument
)

from .semantic_retriever import (
    SemanticRetriever,
    MultiQuerySemanticRetriever
)

from .keyword_retriever import (
    KeywordRetriever,
    BM25Retriever,
    TFIDFRetriever
)

from .hybrid_retriever import (
    HybridRetriever,
    RRFHybridRetriever,
    WeightedHybridRetriever
)

__all__ = [
    # Base classes
    "BaseRetriever",
    "RetrievalResult", 
    "RetrievedDocument",
    
    # Semantic retrieval
    "SemanticRetriever",
    "MultiQuerySemanticRetriever",
    
    # Keyword retrieval
    "KeywordRetriever",
    "BM25Retriever",
    "TFIDFRetriever",
    
    # Hybrid retrieval
    "HybridRetriever",
    "RRFHybridRetriever",
    "WeightedHybridRetriever",
]


def create_retriever(retriever_type: str, **kwargs):
    """검색기 타입에 따른 검색기 생성"""
    
    retriever_type = retriever_type.lower()
    
    if retriever_type == "semantic":
        return SemanticRetriever(**kwargs)
    elif retriever_type == "keyword":
        return KeywordRetriever(**kwargs)
    elif retriever_type == "bm25":
        return BM25Retriever(**kwargs)
    elif retriever_type == "tfidf":
        return TFIDFRetriever(**kwargs)
    elif retriever_type == "hybrid":
        return HybridRetriever(**kwargs)
    elif retriever_type == "rrf_hybrid":
        return RRFHybridRetriever(**kwargs)
    elif retriever_type == "weighted_hybrid":
        return WeightedHybridRetriever(**kwargs)
    else:
        raise ValueError(f"지원하지 않는 검색기 타입: {retriever_type}")


def get_retriever_info():
    """검색기별 정보 반환"""
    return {
        "semantic": {
            "name": "Semantic Retriever",
            "description": "임베딩 기반 의미적 검색",
            "pros": ["의미 이해", "다국어 지원", "문맥 고려"],
            "cons": ["계산 비용", "임베딩 모델 의존성"],
            "use_case": "의미적 유사성이 중요한 경우"
        },
        "keyword": {
            "name": "Keyword Retriever", 
            "description": "키워드 기반 검색",
            "pros": ["빠른 속도", "정확한 키워드 매칭", "해석 가능"],
            "cons": ["동의어 처리 어려움", "문맥 이해 부족"],
            "use_case": "정확한 키워드 매칭이 중요한 경우"
        },
        "hybrid": {
            "name": "Hybrid Retriever",
            "description": "의미적 + 키워드 검색 결합",
            "pros": ["두 방식의 장점", "더 나은 검색 성능", "균형적 접근"],
            "cons": ["복잡성 증가", "파라미터 튜닝 필요"],
            "use_case": "최고의 검색 성능이 필요한 경우"
        }
    }