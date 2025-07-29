"""
Semantic Retriever Module
임베딩 기반 의미적 검색을 수행하는 모듈
"""

import time
import logging
from typing import List, Dict, Any, Optional
from .base_retriever import BaseRetriever, RetrievalResult, RetrievedDocument

logger = logging.getLogger(__name__)


class SemanticRetriever(BaseRetriever):
    """의미적 검색기 (임베딩 기반)"""
    
    def __init__(
        self,
        vector_store,
        embedding_generator,
        config: Dict[str, Any]
    ):
        super().__init__(config)
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.similarity_threshold = config.get("similarity_threshold", 0.7)
        self.use_mmr = config.get("use_mmr", False)  # Maximal Marginal Relevance
        self.mmr_lambda = config.get("mmr_lambda", 0.5)
        
        logger.info(f"의미적 검색기 초기화: {self.name}")
    
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> RetrievalResult:
        """의미적 유사성을 기반으로 문서 검색"""
        
        start_time = time.time()
        query = self.preprocess_query(query)
        k = k or self.top_k
        
        try:
            # 쿼리 임베딩 생성
            query_embedding = self.embedding_generator.embed_text(query)
            
            # 벡터 스토어에서 검색
            search_result = self.vector_store.search(
                query_embedding=query_embedding,
                k=k * 2 if self.use_mmr else k,  # MMR 사용시 더 많은 결과 가져오기
                filters=filters,
                **kwargs
            )
            
            # VectorDocument를 RetrievedDocument로 변환
            retrieved_docs = []
            for vector_doc in search_result.documents:
                retrieved_doc = RetrievedDocument(
                    id=vector_doc.id,
                    content=vector_doc.content,
                    metadata=vector_doc.metadata,
                    score=vector_doc.score or 0.0,
                    retrieval_method="semantic"
                )
                retrieved_docs.append(retrieved_doc)
            
            # MMR 적용 (다양성 증진)
            if self.use_mmr:
                retrieved_docs = self.apply_mmr(
                    query_embedding, 
                    retrieved_docs, 
                    k
                )
            
            # 후처리
            final_docs = self.postprocess_results(retrieved_docs, k)
            
            retrieval_time = time.time() - start_time
            
            return RetrievalResult(
                query=query,
                documents=final_docs,
                total_results=len(final_docs),
                retrieval_time=retrieval_time,
                retrieval_method="semantic",
                metadata={
                    "similarity_threshold": self.similarity_threshold,
                    "use_mmr": self.use_mmr,
                    "embedding_model": self.embedding_generator.model_name,
                    "vector_store_type": self.vector_store.__class__.__name__
                }
            )
            
        except Exception as e:
            logger.error(f"의미적 검색 오류: {e}")
            return RetrievalResult(
                query=query,
                documents=[],
                total_results=0,
                retrieval_time=time.time() - start_time,
                retrieval_method="semantic",
                metadata={"error": str(e)}
            )
    
    def apply_mmr(
        self,
        query_embedding: List[float],
        documents: List[RetrievedDocument],
        k: int
    ) -> List[RetrievedDocument]:
        """Maximal Marginal Relevance를 적용하여 다양성 증진"""
        
        if len(documents) <= k:
            return documents
        
        import numpy as np
        
        # 문서 임베딩 생성 (필요한 경우)
        doc_embeddings = []
        for doc in documents:
            if hasattr(doc, 'embedding') and doc.embedding:
                doc_embeddings.append(doc.embedding)
            else:
                # 임베딩이 없으면 새로 생성
                embedding = self.embedding_generator.embed_text(doc.content[:500])  # 처음 500자
                doc_embeddings.append(embedding)
        
        query_emb = np.array(query_embedding)
        doc_embs = np.array(doc_embeddings)
        
        # MMR 알고리즘
        selected_indices = []
        remaining_indices = list(range(len(documents)))
        
        # 첫 번째 문서는 가장 유사한 것 선택
        similarities = np.dot(doc_embs, query_emb) / (
            np.linalg.norm(doc_embs, axis=1) * np.linalg.norm(query_emb)
        )
        first_idx = np.argmax(similarities)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # 나머지 문서들 선택
        while len(selected_indices) < k and remaining_indices:
            mmr_scores = []
            
            for idx in remaining_indices:
                # 쿼리와의 유사도
                query_sim = similarities[idx]
                
                # 이미 선택된 문서들과의 최대 유사도
                max_selected_sim = 0
                for selected_idx in selected_indices:
                    selected_sim = np.dot(doc_embs[idx], doc_embs[selected_idx]) / (
                        np.linalg.norm(doc_embs[idx]) * np.linalg.norm(doc_embs[selected_idx])
                    )
                    max_selected_sim = max(max_selected_sim, selected_sim)
                
                # MMR 점수 계산
                mmr_score = (self.mmr_lambda * query_sim - 
                           (1 - self.mmr_lambda) * max_selected_sim)
                mmr_scores.append((idx, mmr_score))
            
            # 가장 높은 MMR 점수 선택
            best_idx, _ = max(mmr_scores, key=lambda x: x[1])
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        # 선택된 문서들 반환
        selected_docs = [documents[idx] for idx in selected_indices]
        return selected_docs
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        **kwargs
    ) -> bool:
        """문서들을 벡터 스토어에 추가"""
        
        if not self.validate_documents(documents):
            return False
        
        try:
            # VectorDocument 형식으로 변환
            from ..data_processing.vector_store import VectorDocument
            
            vector_docs = []
            texts = [doc["content"] for doc in documents]
            
            # 배치로 임베딩 생성
            embeddings = self.embedding_generator.embed_texts(texts)
            
            for doc, embedding in zip(documents, embeddings):
                vector_doc = VectorDocument(
                    id=doc["id"],
                    content=doc["content"],
                    embedding=embedding,
                    metadata=doc.get("metadata", {})
                )
                vector_docs.append(vector_doc)
            
            # 벡터 스토어에 추가
            result = self.vector_store.add_documents(vector_docs, **kwargs)
            
            logger.info(f"의미적 검색기에 {len(documents)}개 문서 추가")
            return len(result) > 0
            
        except Exception as e:
            logger.error(f"문서 추가 오류: {e}")
            return False
    
    def update_document(
        self,
        doc_id: str,
        document: Dict[str, Any],
        **kwargs
    ) -> bool:
        """문서 업데이트"""
        
        try:
            # 새 임베딩 생성
            embedding = self.embedding_generator.embed_text(document["content"])
            
            from ..data_processing.vector_store import VectorDocument
            vector_doc = VectorDocument(
                id=doc_id,
                content=document["content"],
                embedding=embedding,
                metadata=document.get("metadata", {})
            )
            
            result = self.vector_store.update_document(vector_doc)
            logger.info(f"문서 업데이트: {doc_id}")
            return result
            
        except Exception as e:
            logger.error(f"문서 업데이트 오류: {e}")
            return False
    
    def delete_documents(
        self,
        doc_ids: List[str],
        **kwargs
    ) -> bool:
        """문서들을 벡터 스토어에서 삭제"""
        
        try:
            result = self.vector_store.delete_documents(doc_ids)
            logger.info(f"{len(doc_ids)}개 문서 삭제")
            return result
            
        except Exception as e:
            logger.error(f"문서 삭제 오류: {e}")
            return False
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """문서 ID로 문서 정보 조회"""
        
        try:
            vector_doc = self.vector_store.get_document(doc_id)
            if vector_doc:
                return {
                    "id": vector_doc.id,
                    "content": vector_doc.content,
                    "metadata": vector_doc.metadata
                }
            return None
            
        except Exception as e:
            logger.error(f"문서 조회 오류: {e}")
            return None
    
    def get_index_info(self) -> Dict[str, Any]:
        """인덱스 정보 반환"""
        
        try:
            vector_store_info = self.vector_store.get_collection_info()
            embedding_info = self.embedding_generator.get_model_info()
            
            return {
                "retriever_type": "semantic",
                "vector_store": vector_store_info,
                "embedding_generator": embedding_info,
                "similarity_threshold": self.similarity_threshold,
                "use_mmr": self.use_mmr,
                "mmr_lambda": self.mmr_lambda
            }
            
        except Exception as e:
            logger.error(f"인덱스 정보 조회 오류: {e}")
            return {"error": str(e)}
    
    def compute_document_similarity(
        self,
        doc1_id: str,
        doc2_id: str
    ) -> Optional[float]:
        """두 문서 간의 의미적 유사도 계산"""
        
        try:
            doc1 = self.get_document_by_id(doc1_id)
            doc2 = self.get_document_by_id(doc2_id)
            
            if not doc1 or not doc2:
                return None
            
            # 임베딩 생성
            emb1 = self.embedding_generator.embed_text(doc1["content"])
            emb2 = self.embedding_generator.embed_text(doc2["content"])
            
            # 코사인 유사도 계산
            import numpy as np
            
            emb1_arr = np.array(emb1)
            emb2_arr = np.array(emb2)
            
            similarity = np.dot(emb1_arr, emb2_arr) / (
                np.linalg.norm(emb1_arr) * np.linalg.norm(emb2_arr)
            )
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"문서 유사도 계산 오류: {e}")
            return None


class MultiQuerySemanticRetriever(SemanticRetriever):
    """다중 쿼리 의미적 검색기"""
    
    def __init__(
        self,
        vector_store,
        embedding_generator,
        query_generator,  # LLM 제공자
        config: Dict[str, Any]
    ):
        super().__init__(vector_store, embedding_generator, config)
        self.query_generator = query_generator
        self.num_queries = config.get("num_queries", 3)
        self.query_combination_method = config.get("query_combination_method", "rrf")
        
        logger.info(f"다중 쿼리 의미적 검색기 초기화: {self.name}")
    
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> RetrievalResult:
        """다중 쿼리를 생성하여 의미적 검색 수행"""
        
        start_time = time.time()
        query = self.preprocess_query(query)
        k = k or self.top_k
        
        try:
            # 다중 쿼리 생성
            generated_queries = self.generate_multiple_queries(query)
            all_queries = [query] + generated_queries
            
            # 각 쿼리로 검색
            all_results = []
            for q in all_queries:
                result = super().retrieve(q, k, filters, **kwargs)
                all_results.append(result)
            
            # 결과 결합
            combined_docs = self.combine_query_results(all_results, k)
            
            # 후처리
            final_docs = self.postprocess_results(combined_docs, k)
            
            retrieval_time = time.time() - start_time
            
            return RetrievalResult(
                query=query,
                documents=final_docs,
                total_results=len(final_docs),
                retrieval_time=retrieval_time,
                retrieval_method="multi_query_semantic",
                metadata={
                    "original_query": query,
                    "generated_queries": generated_queries,
                    "num_queries": len(all_queries),
                    "combination_method": self.query_combination_method
                }
            )
            
        except Exception as e:
            logger.error(f"다중 쿼리 의미적 검색 오류: {e}")
            return RetrievalResult(
                query=query,
                documents=[],
                total_results=0,
                retrieval_time=time.time() - start_time,
                retrieval_method="multi_query_semantic",
                metadata={"error": str(e)}
            )
    
    def generate_multiple_queries(self, original_query: str) -> List[str]:
        """원본 쿼리를 기반으로 다양한 쿼리 생성"""
        
        try:
            prompt = f"""주어진 질문을 다른 관점에서 표현한 {self.num_queries}개의 질문을 생성해주세요.
원본 질문과 같은 의미이지만 다른 표현을 사용해주세요.

원본 질문: {original_query}

생성할 질문들:
1."""
            
            response = self.query_generator.generate(prompt)
            
            # 응답에서 질문들 추출
            lines = response.content.split('\n')
            queries = []
            
            for line in lines:
                line = line.strip()
                # 번호 패턴 제거 (1., 2., -, * 등)
                import re
                cleaned = re.sub(r'^[\d\-\*\•]\.\s*', '', line)
                cleaned = re.sub(r'^[\-\*\•]\s*', '', cleaned)
                
                if cleaned and cleaned != original_query:
                    queries.append(cleaned)
            
            # 목표 개수만큼 반환
            return queries[:self.num_queries]
            
        except Exception as e:
            logger.error(f"다중 쿼리 생성 오류: {e}")
            return []
    
    def combine_query_results(
        self,
        results: List[RetrievalResult],
        k: int
    ) -> List[RetrievedDocument]:
        """여러 쿼리 결과를 결합"""
        
        if self.query_combination_method == "rrf":
            return self._rrf_combine_queries(results, k)
        elif self.query_combination_method == "weighted":
            return self._weighted_combine_queries(results, k)
        elif self.query_combination_method == "union":
            return self._union_combine_queries(results, k)
        else:
            # 기본값: 첫 번째 결과만 사용
            return results[0].documents[:k] if results else []
    
    def _rrf_combine_queries(
        self,
        results: List[RetrievalResult],
        k: int
    ) -> List[RetrievedDocument]:
        """RRF로 쿼리 결과 결합"""
        
        rrf_k = 60
        doc_scores = {}
        
        for result in results:
            for rank, doc in enumerate(result.documents):
                if doc.id not in doc_scores:
                    doc_scores[doc.id] = {
                        "document": doc,
                        "score": 0.0
                    }
                
                # RRF 점수 추가
                rrf_score = 1.0 / (rrf_k + rank + 1)
                doc_scores[doc.id]["score"] += rrf_score
        
        # 점수순 정렬
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
        
        combined_docs = []
        for item in sorted_docs[:k]:
            doc = item["document"]
            doc.score = item["score"]
            doc.retrieval_method = "multi_query_rrf"
            combined_docs.append(doc)
        
        return combined_docs
    
    def _weighted_combine_queries(
        self,
        results: List[RetrievalResult],
        k: int
    ) -> List[RetrievedDocument]:
        """가중치 기반 쿼리 결과 결합 (원본 쿼리에 더 높은 가중치)"""
        
        # 첫 번째는 원본 쿼리이므로 높은 가중치
        weights = [0.5] + [0.5 / (len(results) - 1)] * (len(results) - 1) if len(results) > 1 else [1.0]
        
        doc_scores = {}
        
        for result, weight in zip(results, weights):
            for doc in result.documents:
                if doc.id not in doc_scores:
                    doc_scores[doc.id] = {
                        "document": doc,
                        "score": 0.0
                    }
                
                doc_scores[doc.id]["score"] += doc.score * weight
        
        # 점수순 정렬
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
        
        combined_docs = []
        for item in sorted_docs[:k]:
            doc = item["document"]
            doc.score = item["score"]
            doc.retrieval_method = "multi_query_weighted"
            combined_docs.append(doc)
        
        return combined_docs
    
    def _union_combine_queries(
        self,
        results: List[RetrievalResult],
        k: int
    ) -> List[RetrievedDocument]:
        """합집합 방식으로 쿼리 결과 결합 (중복 제거)"""
        
        seen_docs = set()
        combined_docs = []
        
        # 첫 번째 결과부터 순서대로 추가
        for result in results:
            for doc in result.documents:
                if doc.id not in seen_docs:
                    seen_docs.add(doc.id)
                    doc.retrieval_method = "multi_query_union"
                    combined_docs.append(doc)
                    
                    if len(combined_docs) >= k:
                        return combined_docs
        
        return combined_docs