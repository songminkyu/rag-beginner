"""
Base Retriever Module
모든 검색기가 구현해야 하는 기본 인터페이스
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDocument:
    """검색된 문서 표준화 클래스"""
    id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    retrieval_method: str
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "score": self.score,
            "retrieval_method": self.retrieval_method
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RetrievedDocument':
        """딕셔너리에서 생성"""
        return cls(**data)


@dataclass
class RetrievalResult:
    """검색 결과 표준화 클래스"""
    query: str
    documents: List[RetrievedDocument]
    total_results: int
    retrieval_time: float
    retrieval_method: str
    metadata: Optional[Dict[str, Any]] = None
    
    def get_top_k(self, k: int) -> List[RetrievedDocument]:
        """상위 k개 문서 반환"""
        return self.documents[:k]
    
    def filter_by_score(self, min_score: float) -> List[RetrievedDocument]:
        """최소 점수 이상인 문서들 반환"""
        return [doc for doc in self.documents if doc.score >= min_score]
    
    def filter_by_metadata(self, filters: Dict[str, Any]) -> List[RetrievedDocument]:
        """메타데이터 조건에 맞는 문서들 반환"""
        filtered_docs = []
        
        for doc in self.documents:
            match = True
            for key, value in filters.items():
                if key not in doc.metadata or doc.metadata[key] != value:
                    match = False
                    break
            
            if match:
                filtered_docs.append(doc)
        
        return filtered_docs
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "query": self.query,
            "documents": [doc.to_dict() for doc in self.documents],
            "total_results": self.total_results,
            "retrieval_time": self.retrieval_time,
            "retrieval_method": self.retrieval_method,
            "metadata": self.metadata
        }


class BaseRetriever(ABC):
    """검색기 기본 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get("name", self.__class__.__name__)
        self.top_k = config.get("top_k", 10)
        self.min_score = config.get("min_score", 0.0)
        self.enable_reranking = config.get("enable_reranking", False)
        self.max_query_length = config.get("max_query_length", 1000)
    
    @abstractmethod
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> RetrievalResult:
        """쿼리에 대한 관련 문서 검색"""
        pass
    
    @abstractmethod
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        **kwargs
    ) -> bool:
        """문서들을 검색 인덱스에 추가"""
        pass
    
    @abstractmethod
    def update_document(
        self,
        doc_id: str,
        document: Dict[str, Any],
        **kwargs
    ) -> bool:
        """특정 문서 업데이트"""
        pass
    
    @abstractmethod
    def delete_documents(
        self,
        doc_ids: List[str],
        **kwargs
    ) -> bool:
        """문서들을 검색 인덱스에서 삭제"""
        pass
    
    @abstractmethod
    def get_index_info(self) -> Dict[str, Any]:
        """인덱스 정보 반환"""
        pass
    
    def preprocess_query(self, query: str) -> str:
        """쿼리 전처리"""
        
        # 길이 제한
        if len(query) > self.max_query_length:
            query = query[:self.max_query_length]
            logger.warning(f"쿼리가 최대 길이를 초과하여 잘림: {self.max_query_length}")
        
        # 기본 정리
        query = query.strip()
        
        # 빈 쿼리 확인
        if not query:
            raise ValueError("빈 쿼리는 처리할 수 없습니다.")
        
        return query
    
    def postprocess_results(
        self,
        documents: List[RetrievedDocument],
        k: Optional[int] = None,
        min_score: Optional[float] = None
    ) -> List[RetrievedDocument]:
        """검색 결과 후처리"""
        
        # 최소 점수 필터링
        if min_score is not None:
            documents = [doc for doc in documents if doc.score >= min_score]
        elif self.min_score > 0:
            documents = [doc for doc in documents if doc.score >= self.min_score]
        
        # 상위 k개 선택
        if k is not None:
            documents = documents[:k]
        elif self.top_k:
            documents = documents[:self.top_k]
        
        # 리랭킹 (하위 클래스에서 구현)
        if self.enable_reranking:
            documents = self.rerank_documents(documents)
        
        return documents
    
    def rerank_documents(
        self,
        documents: List[RetrievedDocument]
    ) -> List[RetrievedDocument]:
        """문서 리랭킹 (기본 구현은 그대로 반환)"""
        return documents
    
    def batch_retrieve(
        self,
        queries: List[str],
        k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[RetrievalResult]:
        """여러 쿼리에 대한 배치 검색"""
        
        results = []
        
        for query in queries:
            try:
                result = self.retrieve(query, k, filters, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"배치 검색 중 오류 발생 (쿼리: {query[:50]}...): {e}")
                # 빈 결과로 대체
                empty_result = RetrievalResult(
                    query=query,
                    documents=[],
                    total_results=0,
                    retrieval_time=0.0,
                    retrieval_method=self.name,
                    metadata={"error": str(e)}
                )
                results.append(empty_result)
        
        return results
    
    def search_with_expansion(
        self,
        query: str,
        expansion_terms: List[str],
        k: Optional[int] = None,
        expansion_weight: float = 0.3,
        **kwargs
    ) -> RetrievalResult:
        """쿼리 확장을 통한 검색"""
        
        # 확장된 쿼리 생성
        expanded_query = query + " " + " ".join(expansion_terms)
        
        # 원본 쿼리와 확장된 쿼리로 각각 검색
        original_result = self.retrieve(query, k, **kwargs)
        expanded_result = self.retrieve(expanded_query, k, **kwargs)
        
        # 결과 결합 (간단한 점수 가중 평균)
        combined_docs = {}
        
        # 원본 결과 추가
        for doc in original_result.documents:
            combined_docs[doc.id] = doc
        
        # 확장 결과 추가 (가중치 적용)
        for doc in expanded_result.documents:
            if doc.id in combined_docs:
                # 기존 문서의 점수와 가중 평균
                original_score = combined_docs[doc.id].score
                new_score = (original_score * (1 - expansion_weight) + 
                           doc.score * expansion_weight)
                combined_docs[doc.id].score = new_score
            else:
                # 새 문서 추가 (가중치 적용)
                doc.score *= expansion_weight
                combined_docs[doc.id] = doc
        
        # 점수순 정렬
        final_docs = sorted(combined_docs.values(), key=lambda x: x.score, reverse=True)
        
        # 결과 반환
        return RetrievalResult(
            query=query,
            documents=final_docs[:k] if k else final_docs,
            total_results=len(final_docs),
            retrieval_time=original_result.retrieval_time + expanded_result.retrieval_time,
            retrieval_method=f"{self.name}_expanded",
            metadata={
                "expansion_terms": expansion_terms,
                "expansion_weight": expansion_weight
            }
        )
    
    def get_similar_documents(
        self,
        doc_id: str,
        k: int = 5,
        **kwargs
    ) -> RetrievalResult:
        """특정 문서와 유사한 문서들 검색"""
        
        # 기본 구현: 문서 내용을 쿼리로 사용
        # 하위 클래스에서 더 효율적인 방법으로 오버라이드 가능
        
        try:
            # 문서 정보 가져오기 (하위 클래스에서 구현해야 함)
            doc_info = self.get_document_by_id(doc_id)
            if not doc_info:
                raise ValueError(f"문서를 찾을 수 없습니다: {doc_id}")
            
            # 문서 내용으로 검색
            result = self.retrieve(doc_info["content"], k + 1, **kwargs)  # +1 for self
            
            # 자기 자신 제외
            filtered_docs = [doc for doc in result.documents if doc.id != doc_id]
            
            return RetrievalResult(
                query=f"similar_to:{doc_id}",
                documents=filtered_docs[:k],
                total_results=len(filtered_docs),
                retrieval_time=result.retrieval_time,
                retrieval_method=f"{self.name}_similar",
                metadata={"source_doc_id": doc_id}
            )
            
        except Exception as e:
            logger.error(f"유사 문서 검색 오류: {e}")
            return RetrievalResult(
                query=f"similar_to:{doc_id}",
                documents=[],
                total_results=0,
                retrieval_time=0.0,
                retrieval_method=f"{self.name}_similar",
                metadata={"error": str(e)}
            )
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """문서 ID로 문서 정보 조회 (하위 클래스에서 구현)"""
        raise NotImplementedError("하위 클래스에서 구현해야 합니다.")
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """검색 통계 정보 반환"""
        
        return {
            "retriever_name": self.name,
            "retriever_type": self.__class__.__name__,
            "config": self.config,
            "index_info": self.get_index_info()
        }
    
    def validate_query(self, query: str) -> bool:
        """쿼리 유효성 검사"""
        
        if not query or not query.strip():
            return False
        
        if len(query) > self.max_query_length:
            logger.warning(f"쿼리 길이가 제한을 초과합니다: {len(query)} > {self.max_query_length}")
        
        return True
    
    def validate_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """문서 리스트 유효성 검사"""
        
        if not documents:
            return False
        
        required_fields = ["id", "content"]
        
        for doc in documents:
            for field in required_fields:
                if field not in doc:
                    logger.error(f"문서에 필수 필드가 없습니다: {field}")
                    return False
        
        return True
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, top_k={self.top_k})"


class MultiRetriever(BaseRetriever):
    """여러 검색기를 결합하는 기본 클래스"""
    
    def __init__(self, retrievers: List[BaseRetriever], config: Dict[str, Any]):
        super().__init__(config)
        self.retrievers = retrievers
        self.combination_method = config.get("combination_method", "rrf")  # rrf, weighted, max
        self.weights = config.get("weights", None)
        
        if self.weights and len(self.weights) != len(self.retrievers):
            raise ValueError("가중치 개수가 검색기 개수와 일치하지 않습니다.")
    
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> RetrievalResult:
        """여러 검색기 결과를 결합하여 검색"""
        
        start_time = time.time()
        query = self.preprocess_query(query)
        k = k or self.top_k
        
        # 각 검색기로 검색
        retrieval_results = []
        total_time = 0.0
        
        for retriever in self.retrievers:
            try:
                result = retriever.retrieve(query, k, filters, **kwargs)
                retrieval_results.append(result)
                total_time += result.retrieval_time
                
            except Exception as e:
                logger.error(f"검색기 {retriever.name} 오류: {e}")
                # 빈 결과로 대체
                empty_result = RetrievalResult(
                    query=query,
                    documents=[],
                    total_results=0,
                    retrieval_time=0.0,
                    retrieval_method=retriever.name
                )
                retrieval_results.append(empty_result)
        
        # 결과 결합
        combined_docs = self.combine_results(retrieval_results, k)
        
        # 후처리
        final_docs = self.postprocess_results(combined_docs, k)
        
        retrieval_time = time.time() - start_time
        
        return RetrievalResult(
            query=query,
            documents=final_docs,
            total_results=len(final_docs),
            retrieval_time=retrieval_time,
            retrieval_method=f"Multi({self.combination_method})",
            metadata={
                "component_retrievers": [r.name for r in self.retrievers],
                "combination_method": self.combination_method,
                "component_times": [r.retrieval_time for r in retrieval_results]
            }
        )
    
    def combine_results(
        self,
        results: List[RetrievalResult],
        k: int
    ) -> List[RetrievedDocument]:
        """결과 결합 로직"""
        
        if self.combination_method == "rrf":
            return self._rrf_combine(results, k)
        elif self.combination_method == "weighted":
            return self._weighted_combine(results, k)
        elif self.combination_method == "max":
            return self._max_combine(results, k)
        else:
            raise ValueError(f"지원하지 않는 결합 방법: {self.combination_method}")
    
    def _rrf_combine(self, results: List[RetrievalResult], k: int) -> List[RetrievedDocument]:
        """Reciprocal Rank Fusion으로 결과 결합"""
        
        rrf_k = 60  # RRF 파라미터
        doc_scores = {}
        
        for result in results:
            for rank, doc in enumerate(result.documents):
                if doc.id not in doc_scores:
                    doc_scores[doc.id] = {
                        "document": doc,
                        "score": 0.0
                    }
                
                # RRF 점수 계산
                rrf_score = 1.0 / (rrf_k + rank + 1)
                doc_scores[doc.id]["score"] += rrf_score
        
        # 점수순 정렬
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
        
        # RetrievedDocument 객체로 변환
        combined_docs = []
        for item in sorted_docs[:k]:
            doc = item["document"]
            doc.score = item["score"]
            doc.retrieval_method = f"RRF({doc.retrieval_method})"
            combined_docs.append(doc)
        
        return combined_docs
    
    def _weighted_combine(self, results: List[RetrievalResult], k: int) -> List[RetrievedDocument]:
        """가중치 기반 결과 결합"""
        
        if not self.weights:
            # 균등 가중치
            weights = [1.0 / len(results)] * len(results)
        else:
            weights = self.weights
        
        doc_scores = {}
        
        for result, weight in zip(results, weights):
            for doc in result.documents:
                if doc.id not in doc_scores:
                    doc_scores[doc.id] = {
                        "document": doc,
                        "score": 0.0
                    }
                
                # 가중 점수 추가
                doc_scores[doc.id]["score"] += doc.score * weight
        
        # 점수순 정렬
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
        
        # RetrievedDocument 객체로 변환
        combined_docs = []
        for item in sorted_docs[:k]:
            doc = item["document"]
            doc.score = item["score"]
            doc.retrieval_method = f"Weighted({doc.retrieval_method})"
            combined_docs.append(doc)
        
        return combined_docs
    
    def _max_combine(self, results: List[RetrievalResult], k: int) -> List[RetrievedDocument]:
        """최대 점수 기반 결과 결합"""
        
        doc_scores = {}
        
        for result in results:
            for doc in result.documents:
                if doc.id not in doc_scores or doc.score > doc_scores[doc.id]["score"]:
                    doc_scores[doc.id] = {
                        "document": doc,
                        "score": doc.score
                    }
        
        # 점수순 정렬
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
        
        # RetrievedDocument 객체로 변환
        combined_docs = []
        for item in sorted_docs[:k]:
            doc = item["document"]
            doc.retrieval_method = f"Max({doc.retrieval_method})"
            combined_docs.append(doc)
        
        return combined_docs
    
    def add_documents(self, documents: List[Dict[str, Any]], **kwargs) -> bool:
        """모든 검색기에 문서 추가"""
        
        success = True
        for retriever in self.retrievers:
            try:
                result = retriever.add_documents(documents, **kwargs)
                if not result:
                    success = False
                    logger.error(f"검색기 {retriever.name}에 문서 추가 실패")
            except Exception as e:
                success = False
                logger.error(f"검색기 {retriever.name}에 문서 추가 오류: {e}")
        
        return success
    
    def update_document(self, doc_id: str, document: Dict[str, Any], **kwargs) -> bool:
        """모든 검색기에서 문서 업데이트"""
        
        success = True
        for retriever in self.retrievers:
            try:
                result = retriever.update_document(doc_id, document, **kwargs)
                if not result:
                    success = False
                    logger.error(f"검색기 {retriever.name}에서 문서 업데이트 실패")
            except Exception as e:
                success = False
                logger.error(f"검색기 {retriever.name}에서 문서 업데이트 오류: {e}")
        
        return success
    
    def delete_documents(self, doc_ids: List[str], **kwargs) -> bool:
        """모든 검색기에서 문서 삭제"""
        
        success = True
        for retriever in self.retrievers:
            try:
                result = retriever.delete_documents(doc_ids, **kwargs)
                if not result:
                    success = False
                    logger.error(f"검색기 {retriever.name}에서 문서 삭제 실패")
            except Exception as e:
                success = False
                logger.error(f"검색기 {retriever.name}에서 문서 삭제 오류: {e}")
        
        return success
    
    def get_index_info(self) -> Dict[str, Any]:
        """모든 검색기의 인덱스 정보"""
        
        info = {
            "multi_retriever": True,
            "combination_method": self.combination_method,
            "component_retrievers": []
        }
        
        for retriever in self.retrievers:
            try:
                retriever_info = retriever.get_index_info()
                info["component_retrievers"].append({
                    "name": retriever.name,
                    "info": retriever_info
                })
            except Exception as e:
                info["component_retrievers"].append({
                    "name": retriever.name,
                    "error": str(e)
                })
        
        return info