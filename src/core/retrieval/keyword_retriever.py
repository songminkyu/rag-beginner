"""
Keyword Retriever Module
키워드 기반 검색을 수행하는 모듈 (BM25, TF-IDF 등)
"""

import re
import math
import time
import logging
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Set
from .base_retriever import BaseRetriever, RetrievalResult, RetrievedDocument

# 외부 라이브러리 (optional imports)
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import konlpy
    from konlpy.tag import Okt, Mecab
    HAS_KONLPY = True
except ImportError:
    HAS_KONLPY = False

logger = logging.getLogger(__name__)


class KeywordRetriever(BaseRetriever):
    """기본 키워드 검색기 (단순 텍스트 매칭)"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.case_sensitive = config.get("case_sensitive", False)
        self.use_stemming = config.get("use_stemming", False)
        self.stop_words = set(config.get("stop_words", []))
        self.language = config.get("language", "korean")
        
        # 한국어 처리를 위한 토크나이저
        self.tokenizer = None
        if self.language == "korean" and HAS_KONLPY:
            try:
                self.tokenizer = Okt()
                logger.info("한국어 토크나이저 (Okt) 초기화")
            except Exception as e:
                logger.warning(f"한국어 토크나이저 초기화 실패: {e}")
        
        # 기본 한국어 불용어
        if self.language == "korean" and not self.stop_words:
            self.stop_words = {
                "은", "는", "이", "가", "을", "를", "에", "에서", "로", "으로",
                "와", "과", "의", "도", "만", "에게", "한테", "부터", "까지",
                "이다", "있다", "없다", "되다", "하다", "그", "저", "이", "그것",
                "저것", "여기", "거기", "저기", "때문에", "그리고", "또한",
                "하지만", "그러나", "그래서", "따라서"
            }
        
        logger.info(f"키워드 검색기 초기화: {self.name}")
    
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> RetrievalResult:
        """키워드 매칭으로 문서 검색"""
        
        start_time = time.time()
        query = self.preprocess_query(query)
        k = k or self.top_k
        
        try:
            # 쿼리 토큰화
            query_tokens = self.tokenize_text(query)
            
            # 문서별 점수 계산
            doc_scores = []
            
            for doc_id, doc_data in self.documents.items():
                # 필터 적용
                if filters and not self._match_filters(doc_data.get("metadata", {}), filters):
                    continue
                
                # 키워드 매칭 점수 계산
                score = self.calculate_keyword_score(query_tokens, doc_data["content"], doc_id)
                
                if score > 0:
                    retrieved_doc = RetrievedDocument(
                        id=doc_id,
                        content=doc_data["content"],
                        metadata=doc_data.get("metadata", {}),
                        score=score,
                        retrieval_method="keyword"
                    )
                    doc_scores.append(retrieved_doc)
            
            # 점수순 정렬
            doc_scores.sort(key=lambda x: x.score, reverse=True)
            
            # 후처리
            final_docs = self.postprocess_results(doc_scores, k)
            
            retrieval_time = time.time() - start_time
            
            return RetrievalResult(
                query=query,
                documents=final_docs,
                total_results=len(final_docs),
                retrieval_time=retrieval_time,
                retrieval_method="keyword",
                metadata={
                    "query_tokens": query_tokens,
                    "case_sensitive": self.case_sensitive,
                    "language": self.language,
                    "total_candidate_docs": len([d for d in self.documents.values() 
                                               if not filters or self._match_filters(d.get("metadata", {}), filters)])
                }
            )
            
        except Exception as e:
            logger.error(f"키워드 검색 오류: {e}")
            return RetrievalResult(
                query=query,
                documents=[],
                total_results=0,
                retrieval_time=time.time() - start_time,
                retrieval_method="keyword",
                metadata={"error": str(e)}
            )
    
    def tokenize_text(self, text: str) -> List[str]:
        """텍스트 토큰화"""
        
        if not self.case_sensitive:
            text = text.lower()
        
        if self.tokenizer and self.language == "korean":
            try:
                # 한국어 형태소 분석
                tokens = self.tokenizer.morphs(text)
            except Exception as e:
                logger.warning(f"한국어 토큰화 실패, 기본 방식 사용: {e}")
                tokens = self._basic_tokenize(text)
        else:
            tokens = self._basic_tokenize(text)
        
        # 불용어 제거
        tokens = [token for token in tokens if token not in self.stop_words]
        
        return tokens
    
    def _basic_tokenize(self, text: str) -> List[str]:
        """기본 토큰화 (정규식 기반)"""
        
        # 한글, 영문, 숫자만 추출
        if self.language == "korean":
            pattern = r'[가-힣a-zA-Z0-9]+'
        else:
            pattern = r'[a-zA-Z0-9]+'
        
        tokens = re.findall(pattern, text)
        return tokens
    
    def calculate_keyword_score(
        self,
        query_tokens: List[str],
        doc_content: str,
        doc_id: str
    ) -> float:
        """키워드 매칭 점수 계산"""
        
        doc_tokens = self.tokenize_text(doc_content)
        doc_token_count = Counter(doc_tokens)
        
        score = 0.0
        matched_tokens = 0
        
        for token in query_tokens:
            if token in doc_token_count:
                # TF (Term Frequency) 계산
                tf = doc_token_count[token] / len(doc_tokens) if doc_tokens else 0
                
                # 간단한 IDF 근사 (전체 문서 대비)
                docs_with_token = sum(1 for doc_data in self.documents.values() 
                                    if token in self.tokenize_text(doc_data["content"]))
                idf = math.log(len(self.documents) / (docs_with_token + 1)) + 1
                
                # TF-IDF 점수 추가
                score += tf * idf
                matched_tokens += 1
        
        # 매칭된 토큰 비율로 정규화
        if query_tokens:
            score *= (matched_tokens / len(query_tokens))
        
        return score
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        **kwargs
    ) -> bool:
        """문서들을 키워드 인덱스에 추가"""
        
        if not self.validate_documents(documents):
            return False
        
        try:
            for doc in documents:
                self.documents[doc["id"]] = {
                    "content": doc["content"],
                    "metadata": doc.get("metadata", {})
                }
            
            logger.info(f"키워드 검색기에 {len(documents)}개 문서 추가")
            return True
            
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
            self.documents[doc_id] = {
                "content": document["content"],
                "metadata": document.get("metadata", {})
            }
            
            logger.info(f"문서 업데이트: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"문서 업데이트 오류: {e}")
            return False
    
    def delete_documents(
        self,
        doc_ids: List[str],
        **kwargs
    ) -> bool:
        """문서들을 키워드 인덱스에서 삭제"""
        
        try:
            for doc_id in doc_ids:
                if doc_id in self.documents:
                    del self.documents[doc_id]
            
            logger.info(f"{len(doc_ids)}개 문서 삭제")
            return True
            
        except Exception as e:
            logger.error(f"문서 삭제 오류: {e}")
            return False
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """문서 ID로 문서 정보 조회"""
        
        if doc_id in self.documents:
            doc_data = self.documents[doc_id]
            return {
                "id": doc_id,
                "content": doc_data["content"],
                "metadata": doc_data["metadata"]
            }
        return None
    
    def get_index_info(self) -> Dict[str, Any]:
        """인덱스 정보 반환"""
        
        # 전체 토큰 통계
        total_tokens = 0
        unique_tokens = set()
        
        for doc_data in self.documents.values():
            tokens = self.tokenize_text(doc_data["content"])
            total_tokens += len(tokens)
            unique_tokens.update(tokens)
        
        return {
            "retriever_type": "keyword",
            "total_documents": len(self.documents),
            "total_tokens": total_tokens,
            "unique_tokens": len(unique_tokens),
            "language": self.language,
            "case_sensitive": self.case_sensitive,
            "use_stemming": self.use_stemming,
            "stop_words_count": len(self.stop_words),
            "has_korean_tokenizer": self.tokenizer is not None
        }
    
    def _match_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """필터 조건 확인"""
        
        for key, value in filters.items():
            if key not in metadata:
                return False
            
            if isinstance(value, dict):
                # 범위 쿼리 지원
                if "$gt" in value and metadata[key] <= value["$gt"]:
                    return False
                if "$lt" in value and metadata[key] >= value["$lt"]:
                    return False  
                if "$gte" in value and metadata[key] < value["$gte"]:
                    return False
                if "$lte" in value and metadata[key] > value["$lte"]:
                    return False
                if "$in" in value and metadata[key] not in value["$in"]:
                    return False
                if "$nin" in value and metadata[key] in value["$nin"]:
                    return False
            else:
                # 정확한 매치
                if metadata[key] != value:
                    return False
        
        return True


class BM25Retriever(BaseRetriever):
    """BM25 알고리즘 기반 검색기"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        if not HAS_BM25:
            raise ImportError("BM25 검색을 위해 rank-bm25를 설치해주세요: pip install rank-bm25")
        
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.bm25 = None
        self.tokenized_docs = []
        self.doc_ids = []
        
        # BM25 파라미터
        self.k1 = config.get("k1", 1.2)
        self.b = config.get("b", 0.75)
        self.epsilon = config.get("epsilon", 0.25)
        
        # 토큰화 설정
        self.language = config.get("language", "korean")
        self.stop_words = set(config.get("stop_words", []))
        
        # 한국어 토크나이저
        self.tokenizer = None
        if self.language == "korean" and HAS_KONLPY:
            try:
                self.tokenizer = Okt()
                logger.info("BM25 한국어 토크나이저 (Okt) 초기화")
            except Exception:
                pass
        
        # 기본 한국어 불용어
        if self.language == "korean" and not self.stop_words:
            self.stop_words = {
                "은", "는", "이", "가", "을", "를", "에", "에서", "로", "으로",
                "와", "과", "의", "도", "만", "에게", "한테", "부터", "까지",
                "이다", "있다", "없다", "되다", "하다", "그", "저", "이", "그것",
                "저것", "여기", "거기", "저기", "때문에", "그리고", "또한",
                "하지만", "그러나", "그래서", "따라서"
            }
        
        logger.info(f"BM25 검색기 초기화: k1={self.k1}, b={self.b}")
    
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> RetrievalResult:
        """BM25로 문서 검색"""
        
        start_time = time.time()
        query = self.preprocess_query(query)
        k = k or self.top_k
        
        try:
            if not self.bm25:
                logger.warning("BM25 인덱스가 없습니다. 빈 결과 반환.")
                return RetrievalResult(
                    query=query,
                    documents=[],
                    total_results=0,
                    retrieval_time=time.time() - start_time,
                    retrieval_method="bm25",
                    metadata={"error": "인덱스 없음"}
                )
            
            # 쿼리 토큰화
            query_tokens = self.tokenize_text(query)
            
            # BM25 점수 계산
            scores = self.bm25.get_scores(query_tokens)
            
            # 문서-점수 쌍 생성
            doc_scores = []
            for i, score in enumerate(scores):
                if score > 0:
                    doc_id = self.doc_ids[i]
                    doc_data = self.documents[doc_id]
                    
                    # 필터 적용
                    if filters and not self._match_filters(doc_data.get("metadata", {}), filters):
                        continue
                    
                    retrieved_doc = RetrievedDocument(
                        id=doc_id,
                        content=doc_data["content"],
                        metadata=doc_data.get("metadata", {}),
                        score=float(score),
                        retrieval_method="bm25"
                    )
                    doc_scores.append(retrieved_doc)
            
            # 점수순 정렬
            doc_scores.sort(key=lambda x: x.score, reverse=True)
            
            # 후처리
            final_docs = self.postprocess_results(doc_scores, k)
            
            retrieval_time = time.time() - start_time
            
            return RetrievalResult(
                query=query,
                documents=final_docs,
                total_results=len(final_docs),
                retrieval_time=retrieval_time,
                retrieval_method="bm25",
                metadata={
                    "query_tokens": query_tokens,
                    "bm25_params": {"k1": self.k1, "b": self.b, "epsilon": self.epsilon},
                    "total_candidate_docs": len([d for d in self.documents.values() 
                                               if not filters or self._match_filters(d.get("metadata", {}), filters)])
                }
            )
            
        except Exception as e:
            logger.error(f"BM25 검색 오류: {e}")
            return RetrievalResult(
                query=query,
                documents=[],
                total_results=0,
                retrieval_time=time.time() - start_time,
                retrieval_method="bm25",
                metadata={"error": str(e)}
            )
    
    def tokenize_text(self, text: str) -> List[str]:
        """텍스트 토큰화 (KeywordRetriever와 동일)"""
        
        text = text.lower()
        
        if self.tokenizer and self.language == "korean":
            try:
                tokens = self.tokenizer.morphs(text)
            except Exception:
                tokens = self._basic_tokenize(text)
        else:
            tokens = self._basic_tokenize(text)
        
        # 불용어 제거
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 1]
        
        return tokens
    
    def _basic_tokenize(self, text: str) -> List[str]:
        """기본 토큰화"""
        
        if self.language == "korean":
            pattern = r'[가-힣a-zA-Z0-9]+'
        else:
            pattern = r'[a-zA-Z0-9]+'
        
        tokens = re.findall(pattern, text)
        return tokens
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        **kwargs
    ) -> bool:
        """문서들을 BM25 인덱스에 추가"""
        
        if not self.validate_documents(documents):
            return False
        
        try:
            # 기존 문서에 추가
            for doc in documents:
                self.documents[doc["id"]] = {
                    "content": doc["content"],
                    "metadata": doc.get("metadata", {})
                }
            
            # BM25 인덱스 재구성
            self._rebuild_bm25_index()
            
            logger.info(f"BM25 검색기에 {len(documents)}개 문서 추가")
            return True
            
        except Exception as e:
            logger.error(f"문서 추가 오류: {e}")
            return False
    
    def _rebuild_bm25_index(self):
        """BM25 인덱스 재구성"""
        
        self.tokenized_docs = []
        self.doc_ids = []
        
        for doc_id, doc_data in self.documents.items():
            tokens = self.tokenize_text(doc_data["content"])
            self.tokenized_docs.append(tokens)
            self.doc_ids.append(doc_id)
        
        if self.tokenized_docs:
            self.bm25 = BM25Okapi(
                self.tokenized_docs,
                k1=self.k1,
                b=self.b,
                epsilon=self.epsilon
            )
            logger.info(f"BM25 인덱스 재구성 완료: {len(self.tokenized_docs)}개 문서")
        else:
            self.bm25 = None
    
    def update_document(
        self,
        doc_id: str,
        document: Dict[str, Any],
        **kwargs
    ) -> bool:
        """문서 업데이트"""
        
        try:
            self.documents[doc_id] = {
                "content": document["content"],
                "metadata": document.get("metadata", {})
            }
            
            # BM25 인덱스 재구성
            self._rebuild_bm25_index()
            
            logger.info(f"문서 업데이트 및 BM25 인덱스 재구성: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"문서 업데이트 오류: {e}")
            return False
    
    def delete_documents(
        self,
        doc_ids: List[str],
        **kwargs
    ) -> bool:
        """문서들을 BM25 인덱스에서 삭제"""
        
        try:
            for doc_id in doc_ids:
                if doc_id in self.documents:
                    del self.documents[doc_id]
            
            # BM25 인덱스 재구성
            self._rebuild_bm25_index()
            
            logger.info(f"{len(doc_ids)}개 문서 삭제 및 BM25 인덱스 재구성")
            return True
            
        except Exception as e:
            logger.error(f"문서 삭제 오류: {e}")
            return False
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """문서 ID로 문서 정보 조회"""
        
        if doc_id in self.documents:
            doc_data = self.documents[doc_id]
            return {
                "id": doc_id,
                "content": doc_data["content"],
                "metadata": doc_data["metadata"]
            }
        return None
    
    def get_index_info(self) -> Dict[str, Any]:
        """인덱스 정보 반환"""
        
        vocab_size = len(self.bm25.idf) if self.bm25 else 0
        avg_doc_len = sum(len(doc) for doc in self.tokenized_docs) / len(self.tokenized_docs) if self.tokenized_docs else 0
        
        return {
            "retriever_type": "bm25",
            "total_documents": len(self.documents),
            "vocabulary_size": vocab_size,
            "average_document_length": avg_doc_len,
            "bm25_params": {
                "k1": self.k1,
                "b": self.b,
                "epsilon": self.epsilon
            },
            "language": self.language,
            "stop_words_count": len(self.stop_words),
            "has_korean_tokenizer": self.tokenizer is not None
        }
    
    def _match_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """필터 조건 확인 (KeywordRetriever와 동일)"""
        
        for key, value in filters.items():
            if key not in metadata:
                return False
            
            if isinstance(value, dict):
                if "$gt" in value and metadata[key] <= value["$gt"]:
                    return False
                if "$lt" in value and metadata[key] >= value["$lt"]:
                    return False  
                if "$gte" in value and metadata[key] < value["$gte"]:
                    return False
                if "$lte" in value and metadata[key] > value["$lte"]:
                    return False
                if "$in" in value and metadata[key] not in value["$in"]:
                    return False
                if "$nin" in value and metadata[key] in value["$nin"]:
                    return False
            else:
                if metadata[key] != value:
                    return False
        
        return True


class TFIDFRetriever(BaseRetriever):
    """TF-IDF 기반 검색기"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        if not HAS_SKLEARN:
            raise ImportError("TF-IDF 검색을 위해 scikit-learn을 설치해주세요: pip install scikit-learn")
        
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.vectorizer = None
        self.tfidf_matrix = None
        self.doc_ids = []
        
        # TF-IDF 파라미터
        self.max_features = config.get("max_features", 10000)
        self.min_df = config.get("min_df", 1)
        self.max_df = config.get("max_df", 0.95)
        self.ngram_range = tuple(config.get("ngram_range", [1, 2]))
        
        # 토큰화 설정
        self.language = config.get("language", "korean")
        self.stop_words = set(config.get("stop_words", []))
        
        # 한국어 토크나이저
        self.tokenizer = None
        if self.language == "korean" and HAS_KONLPY:
            try:
                self.tokenizer = Okt()
                logger.info("TF-IDF 한국어 토크나이저 (Okt) 초기화")
            except Exception:
                pass
        
        logger.info(f"TF-IDF 검색기 초기화: max_features={self.max_features}")
    
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> RetrievalResult:
        """TF-IDF로 문서 검색"""
        
        start_time = time.time()
        query = self.preprocess_query(query)
        k = k or self.top_k
        
        try:
            if self.vectorizer is None or self.tfidf_matrix is None:
                logger.warning("TF-IDF 인덱스가 없습니다. 빈 결과 반환.")
                return RetrievalResult(
                    query=query,
                    documents=[],
                    total_results=0,
                    retrieval_time=time.time() - start_time,
                    retrieval_method="tfidf",
                    metadata={"error": "인덱스 없음"}
                )
            
            # 쿼리 벡터화
            query_vector = self.vectorizer.transform([self._preprocess_text_for_tfidf(query)])
            
            # 코사인 유사도 계산
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # 문서-점수 쌍 생성
            doc_scores = []
            for i, score in enumerate(similarities):
                if score > 0:
                    doc_id = self.doc_ids[i]
                    doc_data = self.documents[doc_id]
                    
                    # 필터 적용
                    if filters and not self._match_filters(doc_data.get("metadata", {}), filters):
                        continue
                    
                    retrieved_doc = RetrievedDocument(
                        id=doc_id,
                        content=doc_data["content"],
                        metadata=doc_data.get("metadata", {}),
                        score=float(score),
                        retrieval_method="tfidf"
                    )
                    doc_scores.append(retrieved_doc)
            
            # 점수순 정렬
            doc_scores.sort(key=lambda x: x.score, reverse=True)
            
            # 후처리
            final_docs = self.postprocess_results(doc_scores, k)
            
            retrieval_time = time.time() - start_time
            
            return RetrievalResult(
                query=query,
                documents=final_docs,
                total_results=len(final_docs),
                retrieval_time=retrieval_time,
                retrieval_method="tfidf",
                metadata={
                    "tfidf_params": {
                        "max_features": self.max_features,
                        "min_df": self.min_df,
                        "max_df": self.max_df,
                        "ngram_range": self.ngram_range
                    },
                    "vocabulary_size": len(self.vectorizer.vocabulary_) if self.vectorizer else 0,
                    "total_candidate_docs": len([d for d in self.documents.values() 
                                               if not filters or self._match_filters(d.get("metadata", {}), filters)])
                }
            )
            
        except Exception as e:
            logger.error(f"TF-IDF 검색 오류: {e}")
            return RetrievalResult(
                query=query,
                documents=[],
                total_results=0,
                retrieval_time=time.time() - start_time,
                retrieval_method="tfidf",
                metadata={"error": str(e)}
            )
    
    def _preprocess_text_for_tfidf(self, text: str) -> str:
        """TF-IDF를 위한 텍스트 전처리"""
        
        if self.tokenizer and self.language == "korean":
            try:
                tokens = self.tokenizer.morphs(text.lower())
                tokens = [token for token in tokens if token not in self.stop_words and len(token) > 1]
                return " ".join(tokens)
            except Exception:
                pass
        
        # 기본 전처리
        text = text.lower()
        if self.language == "korean":
            pattern = r'[가-힣a-zA-Z0-9]+'
        else:
            pattern = r'[a-zA-Z0-9]+'
        
        tokens = re.findall(pattern, text)
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 1]
        
        return " ".join(tokens)
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        **kwargs
    ) -> bool:
        """문서들을 TF-IDF 인덱스에 추가"""
        
        if not self.validate_documents(documents):
            return False
        
        try:
            # 기존 문서에 추가
            for doc in documents:
                self.documents[doc["id"]] = {
                    "content": doc["content"],
                    "metadata": doc.get("metadata", {})
                }
            
            # TF-IDF 인덱스 재구성
            self._rebuild_tfidf_index()
            
            logger.info(f"TF-IDF 검색기에 {len(documents)}개 문서 추가")
            return True
            
        except Exception as e:
            logger.error(f"문서 추가 오류: {e}")
            return False
    
    def _rebuild_tfidf_index(self):
        """TF-IDF 인덱스 재구성"""
        
        if not self.documents:
            self.vectorizer = None
            self.tfidf_matrix = None
            self.doc_ids = []
            return
        
        # 문서 전처리
        processed_docs = []
        self.doc_ids = []
        
        for doc_id, doc_data in self.documents.items():
            processed_text = self._preprocess_text_for_tfidf(doc_data["content"])
            processed_docs.append(processed_text)
            self.doc_ids.append(doc_id)
        
        # TF-IDF 벡터라이저 생성
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range,
            stop_words=None  # 이미 전처리에서 제거됨
        )
        
        # TF-IDF 매트릭스 생성
        self.tfidf_matrix = self.vectorizer.fit_transform(processed_docs)
        
        logger.info(f"TF-IDF 인덱스 재구성 완료: {len(processed_docs)}개 문서, "
                   f"어휘 크기: {len(self.vectorizer.vocabulary_)}")
    
    def update_document(
        self,
        doc_id: str,
        document: Dict[str, Any],
        **kwargs
    ) -> bool:
        """문서 업데이트"""
        
        try:
            self.documents[doc_id] = {
                "content": document["content"],
                "metadata": document.get("metadata", {})
            }
            
            # TF-IDF 인덱스 재구성
            self._rebuild_tfidf_index()
            
            logger.info(f"문서 업데이트 및 TF-IDF 인덱스 재구성: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"문서 업데이트 오류: {e}")
            return False
    
    def delete_documents(
        self,
        doc_ids: List[str],
        **kwargs
    ) -> bool:
        """문서들을 TF-IDF 인덱스에서 삭제"""
        
        try:
            for doc_id in doc_ids:
                if doc_id in self.documents:
                    del self.documents[doc_id]
            
            # TF-IDF 인덱스 재구성
            self._rebuild_tfidf_index()
            
            logger.info(f"{len(doc_ids)}개 문서 삭제 및 TF-IDF 인덱스 재구성")
            return True
            
        except Exception as e:
            logger.error(f"문서 삭제 오류: {e}")
            return False
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """문서 ID로 문서 정보 조회"""
        
        if doc_id in self.documents:
            doc_data = self.documents[doc_id]
            return {
                "id": doc_id,
                "content": doc_data["content"],
                "metadata": doc_data["metadata"]
            }
        return None
    
    def get_index_info(self) -> Dict[str, Any]:
        """인덱스 정보 반환"""
        
        return {
            "retriever_type": "tfidf",
            "total_documents": len(self.documents),
            "vocabulary_size": len(self.vectorizer.vocabulary_) if self.vectorizer else 0,
            "tfidf_params": {
                "max_features": self.max_features,
                "min_df": self.min_df,
                "max_df": self.max_df,
                "ngram_range": self.ngram_range
            },
            "language": self.language,
            "stop_words_count": len(self.stop_words),
            "has_korean_tokenizer": self.tokenizer is not None
        }
    
    def _match_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """필터 조건 확인"""
        
        for key, value in filters.items():
            if key not in metadata:
                return False
            
            if isinstance(value, dict):
                if "$gt" in value and metadata[key] <= value["$gt"]:
                    return False
                if "$lt" in value and metadata[key] >= value["$lt"]:
                    return False  
                if "$gte" in value and metadata[key] < value["$gte"]:
                    return False
                if "$lte" in value and metadata[key] > value["$lte"]:
                    return False
                if "$in" in value and metadata[key] not in value["$in"]:
                    return False
                if "$nin" in value and metadata[key] in value["$nin"]:
                    return False
            else:
                if metadata[key] != value:
                    return False
        
        return True