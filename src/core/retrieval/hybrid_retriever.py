"""
Hybrid Retriever Module
의미적 검색과 키워드 검색을 결합하는 하이브리드 검색기
"""

import time
import logging
from typing import List, Dict, Any, Optional
from .base_retriever import BaseRetriever, RetrievalResult, RetrievedDocument, MultiRetriever

logger = logging.getLogger(__name__)


class HybridRetriever(MultiRetriever):
    """하이브리드 검색기 (의미적 + 키워드 검색 결합)"""
    
    def __init__(
        self,
        semantic_retriever: BaseRetriever,
        keyword_retriever: BaseRetriever,
        config: Dict[str, Any]
    ):
        # MultiRetriever 초기화
        retrievers = [semantic_retriever, keyword_retriever]
        super().__init__(retrievers, config)
        
        self.semantic_retriever = semantic_retriever
        self.keyword_retriever = keyword_retriever
        
        # 하이브리드 파라미터
        self.semantic_weight = config.get("semantic_weight", 0.7)
        self.keyword_weight = config.get("keyword_weight", 0.3)
        self.normalization_method = config.get("normalization_method", "min_max")  # min_max, z_score
        self.diversity_penalty = config.get("diversity_penalty", 0.1)
        
        # 가중치 정규화
        total_weight = self.semantic_weight + self.keyword_weight
        if total_weight > 0:
            self.semantic_weight /= total_weight
            self.keyword_weight /= total_weight
        
        logger.info(f"하이브리드 검색기 초기화: semantic={self.semantic_weight:.2f}, keyword={self.keyword_weight:.2f}")
    
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> RetrievalResult:
        """하이브리드 검색 수행"""
        
        start_time = time.time()
        query = self.preprocess_query(query)
        k = k or self.top_k
        
        try:
            # 개별 검색기로 검색 (더 많은 결과 가져오기)
            search_k = min(k * 3, 100)  # 결합을 위해 더 많은 후보 가져오기
            
            semantic_result = self.semantic_retriever.retrieve(query, search_k, filters, **kwargs)
            keyword_result = self.keyword_retriever.retrieve(query, search_k, filters, **kwargs)
            
            # 점수 정규화
            semantic_docs_normalized = self._normalize_scores(
                semantic_result.documents, 
                "semantic"
            )
            keyword_docs_normalized = self._normalize_scores(
                keyword_result.documents, 
                "keyword"
            )
            
            # 결과 결합
            combined_docs = self._combine_hybrid_results(
                semantic_docs_normalized,
                keyword_docs_normalized,
                k
            )
            
            # 후처리
            final_docs = self.postprocess_results(combined_docs, k)
            
            retrieval_time = time.time() - start_time
            
            return RetrievalResult(
                query=query,
                documents=final_docs,
                total_results=len(final_docs),
                retrieval_time=retrieval_time,
                retrieval_method="hybrid",
                metadata={
                    "semantic_weight": self.semantic_weight,
                    "keyword_weight": self.keyword_weight,
                    "normalization_method": self.normalization_method,
                    "semantic_results": len(semantic_result.documents),
                    "keyword_results": len(keyword_result.documents),
                    "semantic_time": semantic_result.retrieval_time,
                    "keyword_time": keyword_result.retrieval_time
                }
            )
            
        except Exception as e:
            logger.error(f"하이브리드 검색 오류: {e}")
            return RetrievalResult(
                query=query,
                documents=[],
                total_results=0,
                retrieval_time=time.time() - start_time,
                retrieval_method="hybrid",
                metadata={"error": str(e)}
            )
    
    def _normalize_scores(
        self,
        documents: List[RetrievedDocument],
        method_name: str
    ) -> List[RetrievedDocument]:
        """검색 결과 점수 정규화"""
        
        if not documents:
            return documents
        
        scores = [doc.score for doc in documents]
        
        if self.normalization_method == "min_max":
            # Min-Max 정규화
            min_score = min(scores)
            max_score = max(scores)
            
            if max_score == min_score:
                normalized_scores = [1.0] * len(scores)
            else:
                normalized_scores = [(score - min_score) / (max_score - min_score) for score in scores]
        
        elif self.normalization_method == "z_score":
            # Z-Score 정규화
            import statistics
            
            if len(scores) < 2:
                normalized_scores = [1.0] * len(scores)
            else:
                mean_score = statistics.mean(scores)
                std_score = statistics.stdev(scores)
                
                if std_score == 0:
                    normalized_scores = [1.0] * len(scores)
                else:
                    z_scores = [(score - mean_score) / std_score for score in scores]
                    # Z-score를 [0, 1] 범위로 변환 (sigmoid 함수 사용)
                    import math
                    normalized_scores = [1 / (1 + math.exp(-z)) for z in z_scores]
        
        else:
            # 정규화 없음
            normalized_scores = scores
        
        # 정규화된 점수로 문서 업데이트
        normalized_docs = []
        for doc, norm_score in zip(documents, normalized_scores):
            new_doc = RetrievedDocument(
                id=doc.id,
                content=doc.content,
                metadata=doc.metadata,
                score=norm_score,
                retrieval_method=f"{method_name}_normalized"
            )
            normalized_docs.append(new_doc)
        
        return normalized_docs
    
    def _combine_hybrid_results(
        self,
        semantic_docs: List[RetrievedDocument],
        keyword_docs: List[RetrievedDocument],
        k: int
    ) -> List[RetrievedDocument]:
        """의미적 검색과 키워드 검색 결과를 결합"""
        
        # 문서별 점수 결합
        doc_scores = {}
        
        # 의미적 검색 결과 추가
        for doc in semantic_docs:
            doc_scores[doc.id] = {
                "document": doc,
                "semantic_score": doc.score,
                "keyword_score": 0.0,
                "final_score": 0.0
            }
        
        # 키워드 검색 결과 추가
        for doc in keyword_docs:
            if doc.id in doc_scores:
                doc_scores[doc.id]["keyword_score"] = doc.score
            else:
                doc_scores[doc.id] = {
                    "document": doc,
                    "semantic_score": 0.0,
                    "keyword_score": doc.score,
                    "final_score": 0.0
                }
        
        # 최종 점수 계산 (가중 평균)
        for doc_id, scores in doc_scores.items():
            final_score = (
                scores["semantic_score"] * self.semantic_weight +
                scores["keyword_score"] * self.keyword_weight
            )
            
            # 다양성 보너스 (두 방법 모두에서 점수를 받은 경우)
            if scores["semantic_score"] > 0 and scores["keyword_score"] > 0:
                diversity_bonus = self.diversity_penalty * min(scores["semantic_score"], scores["keyword_score"])
                final_score += diversity_bonus
            
            scores["final_score"] = final_score
        
        # 점수순 정렬
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x["final_score"], reverse=True)
        
        # RetrievedDocument 객체로 변환
        combined_docs = []
        for item in sorted_docs[:k]:
            doc = item["document"]
            doc.score = item["final_score"]
            doc.retrieval_method = "hybrid"
            
            # 메타데이터에 세부 점수 추가
            doc.metadata["hybrid_scores"] = {
                "semantic": item["semantic_score"],
                "keyword": item["keyword_score"],
                "final": item["final_score"]
            }
            
            combined_docs.append(doc)
        
        return combined_docs


class RRFHybridRetriever(MultiRetriever):
    """Reciprocal Rank Fusion을 사용하는 하이브리드 검색기"""
    
    def __init__(
        self,
        semantic_retriever: BaseRetriever,
        keyword_retriever: BaseRetriever,
        config: Dict[str, Any]
    ):
        # MultiRetriever 초기화 (RRF 사용)
        rrf_config = {**config, "combination_method": "rrf"}
        retrievers = [semantic_retriever, keyword_retriever]
        super().__init__(retrievers, rrf_config)
        
        self.semantic_retriever = semantic_retriever
        self.keyword_retriever = keyword_retriever
        self.rrf_k = config.get("rrf_k", 60)
        
        logger.info(f"RRF 하이브리드 검색기 초기화: rrf_k={self.rrf_k}")
    
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> RetrievalResult:
        """RRF 하이브리드 검색 수행"""
        
        start_time = time.time()
        query = self.preprocess_query(query)
        k = k or self.top_k
        
        try:
            # 개별 검색기로 검색
            search_k = min(k * 2, 100)
            
            semantic_result = self.semantic_retriever.retrieve(query, search_k, filters, **kwargs)
            keyword_result = self.keyword_retriever.retrieve(query, search_k, filters, **kwargs)
            
            # RRF 결합
            combined_docs = self._rrf_combine_results(
                [semantic_result, keyword_result],
                k
            )
            
            # 후처리
            final_docs = self.postprocess_results(combined_docs, k)
            
            retrieval_time = time.time() - start_time
            
            return RetrievalResult(
                query=query,
                documents=final_docs,
                total_results=len(final_docs),
                retrieval_time=retrieval_time,
                retrieval_method="rrf_hybrid",
                metadata={
                    "rrf_k": self.rrf_k,
                    "semantic_results": len(semantic_result.documents),
                    "keyword_results": len(keyword_result.documents),
                    "semantic_time": semantic_result.retrieval_time,
                    "keyword_time": keyword_result.retrieval_time
                }
            )
            
        except Exception as e:
            logger.error(f"RRF 하이브리드 검색 오류: {e}")
            return RetrievalResult(
                query=query,
                documents=[],
                total_results=0,
                retrieval_time=time.time() - start_time,
                retrieval_method="rrf_hybrid",
                metadata={"error": str(e)}
            )
    
    def _rrf_combine_results(
        self,
        results: List[RetrievalResult],
        k: int
    ) -> List[RetrievedDocument]:
        """RRF로 결과 결합"""
        
        doc_scores = {}
        
        for result in results:
            for rank, doc in enumerate(result.documents):
                if doc.id not in doc_scores:
                    doc_scores[doc.id] = {
                        "document": doc,
                        "rrf_score": 0.0
                    }
                
                # RRF 점수 계산: 1 / (k + rank)
                rrf_score = 1.0 / (self.rrf_k + rank + 1)
                doc_scores[doc.id]["rrf_score"] += rrf_score
        
        # 점수순 정렬
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x["rrf_score"], reverse=True)
        
        # RetrievedDocument 객체로 변환
        combined_docs = []
        for item in sorted_docs[:k]:
            doc = item["document"]
            doc.score = item["rrf_score"]
            doc.retrieval_method = "rrf_hybrid"
            combined_docs.append(doc)
        
        return combined_docs


class WeightedHybridRetriever(MultiRetriever):
    """가중치 기반 하이브리드 검색기"""
    
    def __init__(
        self,
        retrievers: List[BaseRetriever],
        weights: List[float],
        config: Dict[str, Any]
    ):
        """
        Args:
            retrievers: 검색기 리스트
            weights: 각 검색기에 대한 가중치
            config: 설정
        """
        if len(retrievers) != len(weights):
            raise ValueError("검색기 개수와 가중치 개수가 일치하지 않습니다.")
        
        # 가중치 정규화
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        
        # MultiRetriever 초기화 (weighted 사용)
        weighted_config = {**config, "combination_method": "weighted", "weights": weights}
        super().__init__(retrievers, weighted_config)
        
        self.weights = weights
        self.normalization_method = config.get("normalization_method", "min_max")
        
        logger.info(f"가중치 하이브리드 검색기 초기화: weights={self.weights}")
    
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> RetrievalResult:
        """가중치 하이브리드 검색 수행"""
        
        start_time = time.time()
        query = self.preprocess_query(query)
        k = k or self.top_k
        
        try:
            # 개별 검색기로 검색
            search_k = min(k * 2, 100)
            
            retrieval_results = []
            for retriever in self.retrievers:
                result = retriever.retrieve(query, search_k, filters, **kwargs)
                retrieval_results.append(result)
            
            # 점수 정규화
            normalized_results = []
            for result in retrieval_results:
                normalized_docs = self._normalize_scores(result.documents)
                normalized_result = RetrievalResult(
                    query=result.query,
                    documents=normalized_docs,
                    total_results=result.total_results,
                    retrieval_time=result.retrieval_time,
                    retrieval_method=result.retrieval_method + "_normalized"
                )
                normalized_results.append(normalized_result)
            
            # 가중치 결합
            combined_docs = self._weighted_combine_results(
                normalized_results,
                k
            )
            
            # 후처리
            final_docs = self.postprocess_results(combined_docs, k)
            
            retrieval_time = time.time() - start_time
            
            return RetrievalResult(
                query=query,
                documents=final_docs,
                total_results=len(final_docs),
                retrieval_time=retrieval_time,
                retrieval_method="weighted_hybrid",
                metadata={
                    "weights": self.weights,
                    "normalization_method": self.normalization_method,
                    "component_results": [len(r.documents) for r in retrieval_results],
                    "component_times": [r.retrieval_time for r in retrieval_results]
                }
            )
            
        except Exception as e:
            logger.error(f"가중치 하이브리드 검색 오류: {e}")
            return RetrievalResult(
                query=query,
                documents=[],
                total_results=0,
                retrieval_time=time.time() - start_time,
                retrieval_method="weighted_hybrid",
                metadata={"error": str(e)}
            )
    
    def _normalize_scores(self, documents: List[RetrievedDocument]) -> List[RetrievedDocument]:
        """점수 정규화"""
        
        if not documents:
            return documents
        
        scores = [doc.score for doc in documents]
        
        if self.normalization_method == "min_max":
            min_score = min(scores)
            max_score = max(scores)
            
            if max_score == min_score:
                normalized_scores = [1.0] * len(scores)
            else:
                normalized_scores = [(score - min_score) / (max_score - min_score) for score in scores]
        
        elif self.normalization_method == "z_score":
            import statistics
            
            if len(scores) < 2:
                normalized_scores = [1.0] * len(scores)
            else:
                mean_score = statistics.mean(scores)
                std_score = statistics.stdev(scores)
                
                if std_score == 0:
                    normalized_scores = [1.0] * len(scores)
                else:
                    z_scores = [(score - mean_score) / std_score for score in scores]
                    import math
                    normalized_scores = [1 / (1 + math.exp(-z)) for z in z_scores]
        
        else:
            normalized_scores = scores
        
        # 정규화된 문서 생성
        normalized_docs = []
        for doc, norm_score in zip(documents, normalized_scores):
            new_doc = RetrievedDocument(
                id=doc.id,
                content=doc.content,
                metadata=doc.metadata,
                score=norm_score,
                retrieval_method=doc.retrieval_method + "_normalized"
            )
            normalized_docs.append(new_doc)
        
        return normalized_docs
    
    def _weighted_combine_results(
        self,
        results: List[RetrievalResult],
        k: int
    ) -> List[RetrievedDocument]:
        """가중치 기반 결과 결합"""
        
        doc_scores = {}
        
        for result, weight in zip(results, self.weights):
            for doc in result.documents:
                if doc.id not in doc_scores:
                    doc_scores[doc.id] = {
                        "document": doc,
                        "weighted_score": 0.0,
                        "component_scores": {}
                    }
                
                # 가중 점수 추가
                doc_scores[doc.id]["weighted_score"] += doc.score * weight
                doc_scores[doc.id]["component_scores"][result.retrieval_method] = doc.score
        
        # 점수순 정렬
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x["weighted_score"], reverse=True)
        
        # RetrievedDocument 객체로 변환
        combined_docs = []
        for item in sorted_docs[:k]:
            doc = item["document"]
            doc.score = item["weighted_score"]
            doc.retrieval_method = "weighted_hybrid"
            
            # 메타데이터에 세부 점수 추가
            doc.metadata["component_scores"] = item["component_scores"]
            doc.metadata["weighted_score"] = item["weighted_score"]
            
            combined_docs.append(doc)
        
        return combined_docs


def create_hybrid_retriever(
    retriever_type: str,
    semantic_retriever: BaseRetriever,
    keyword_retriever: BaseRetriever,
    config: Dict[str, Any]
) -> BaseRetriever:
    """하이브리드 검색기 생성 헬퍼 함수"""
    
    retriever_type = retriever_type.lower()
    
    if retriever_type == "hybrid" or retriever_type == "weighted_hybrid":
        return HybridRetriever(semantic_retriever, keyword_retriever, config)
    
    elif retriever_type == "rrf" or retriever_type == "rrf_hybrid":
        return RRFHybridRetriever(semantic_retriever, keyword_retriever, config)
    
    else:
        raise ValueError(f"지원하지 않는 하이브리드 검색기 타입: {retriever_type}")


def create_multi_retriever(
    retrievers: List[BaseRetriever],
    combination_method: str = "rrf",
    weights: Optional[List[float]] = None,
    config: Optional[Dict[str, Any]] = None
) -> BaseRetriever:
    """다중 검색기 생성 헬퍼 함수"""
    
    config = config or {}
    config["combination_method"] = combination_method
    
    if combination_method == "weighted" and weights:
        return WeightedHybridRetriever(retrievers, weights, config)
    else:
        return MultiRetriever(retrievers, config)