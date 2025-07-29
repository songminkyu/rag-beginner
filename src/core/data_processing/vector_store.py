"""
Vector Store Module
임베딩을 저장하고 검색하는 벡터 스토어 모듈
"""

import os
import json
import pickle
import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path

# 외부 라이브러리 (optional imports)
try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    import pinecone
    HAS_PINECONE = True
except ImportError:
    HAS_PINECONE = False

logger = logging.getLogger(__name__)


@dataclass
class VectorDocument:
    """벡터 문서 표준화 클래스"""
    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]
    score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VectorDocument':
        """딕셔너리에서 생성"""
        return cls(**data)


@dataclass
class SearchResult:
    """검색 결과 표준화 클래스"""
    documents: List[VectorDocument]
    query: str
    total_results: int
    search_time: float
    metadata: Optional[Dict[str, Any]] = None


class VectorStore(ABC):
    """벡터 스토어 기본 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.collection_name = config.get("collection_name", "default")
        self.dimension = config.get("dimension", 768)
        self.metric = config.get("metric", "cosine")  # cosine, euclidean, inner_product
    
    @abstractmethod
    def add_documents(
        self,
        documents: List[VectorDocument],
        **kwargs
    ) -> List[str]:
        """문서들을 벡터 스토어에 추가"""
        pass
    
    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> SearchResult:
        """임베딩으로 유사한 문서 검색"""
        pass
    
    @abstractmethod
    def delete_documents(self, ids: List[str]) -> bool:
        """문서들을 벡터 스토어에서 삭제"""
        pass
    
    @abstractmethod
    def get_document(self, doc_id: str) -> Optional[VectorDocument]:
        """특정 ID의 문서 조회"""
        pass
    
    @abstractmethod
    def update_document(self, document: VectorDocument) -> bool:
        """문서 업데이트"""
        pass
    
    @abstractmethod
    def get_collection_info(self) -> Dict[str, Any]:
        """컬렉션 정보 조회"""
        pass
    
    def search_by_text(
        self,
        query_text: str,
        embedding_generator,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> SearchResult:
        """텍스트로 검색 (임베딩 생성 후 검색)"""
        
        query_embedding = embedding_generator.embed_text(query_text)
        return self.search(query_embedding, k, filters, **kwargs)
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        embedding_generator = None,
        **kwargs
    ) -> List[str]:
        """텍스트 리스트를 임베딩으로 변환하여 추가"""
        
        if embedding_generator is None:
            raise ValueError("임베딩 생성기가 필요합니다.")
        
        # ID 생성
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]
        
        # 메타데이터 생성
        if metadatas is None:
            metadatas = [{}] * len(texts)
        
        # 임베딩 생성
        embeddings = embedding_generator.embed_texts(texts)
        
        # VectorDocument 객체 생성
        documents = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            doc = VectorDocument(
                id=ids[i],
                content=text,
                embedding=embedding,
                metadata=metadatas[i]
            )
            documents.append(doc)
        
        return self.add_documents(documents, **kwargs)


class InMemoryVectorStore(VectorStore):
    """메모리 기반 벡터 스토어 (개발/테스트용)"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.documents: Dict[str, VectorDocument] = {}
        self.embeddings = []
        self.doc_ids = []
        
        logger.info("InMemory 벡터 스토어 초기화")
    
    def add_documents(
        self,
        documents: List[VectorDocument],
        **kwargs
    ) -> List[str]:
        """문서들을 메모리에 추가"""
        
        added_ids = []
        
        for doc in documents:
            self.documents[doc.id] = doc
            self.embeddings.append(doc.embedding)
            self.doc_ids.append(doc.id)
            added_ids.append(doc.id)
        
        logger.info(f"메모리 벡터 스토어에 {len(documents)}개 문서 추가")
        return added_ids
    
    def search(
        self,
        query_embedding: List[float],
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> SearchResult:
        """코사인 유사도로 검색"""
        
        import time
        start_time = time.time()
        
        if not self.embeddings:
            return SearchResult(
                documents=[],
                query="",
                total_results=0,
                search_time=time.time() - start_time
            )
        
        # 코사인 유사도 계산
        query_arr = np.array(query_embedding)
        embeddings_arr = np.array(self.embeddings)
        
        # 정규화
        query_norm = query_arr / np.linalg.norm(query_arr)
        embeddings_norm = embeddings_arr / np.linalg.norm(embeddings_arr, axis=1, keepdims=True)
        
        # 유사도 계산
        similarities = np.dot(embeddings_norm, query_norm)
        
        # 필터링 적용
        valid_indices = list(range(len(similarities)))
        if filters:
            valid_indices = []
            for i, doc_id in enumerate(self.doc_ids):
                doc = self.documents[doc_id]
                if self._match_filters(doc.metadata, filters):
                    valid_indices.append(i)
        
        # 유효한 결과만 추출
        valid_similarities = [(similarities[i], i) for i in valid_indices]
        
        # 상위 k개 선택
        top_results = sorted(valid_similarities, key=lambda x: x[0], reverse=True)[:k]
        
        # 결과 문서 생성
        result_documents = []
        for similarity, idx in top_results:
            doc_id = self.doc_ids[idx]
            doc = self.documents[doc_id]
            
            # 점수 추가
            result_doc = VectorDocument(
                id=doc.id,
                content=doc.content,
                embedding=doc.embedding,
                metadata=doc.metadata,
                score=float(similarity)
            )
            result_documents.append(result_doc)
        
        search_time = time.time() - start_time
        
        return SearchResult(
            documents=result_documents,
            query="",
            total_results=len(valid_indices),
            search_time=search_time
        )
    
    def delete_documents(self, ids: List[str]) -> bool:
        """문서들을 메모리에서 삭제"""
        
        try:
            for doc_id in ids:
                if doc_id in self.documents:
                    # 문서 삭제
                    del self.documents[doc_id]
                    
                    # 임베딩과 ID 리스트에서도 삭제
                    if doc_id in self.doc_ids:
                        idx = self.doc_ids.index(doc_id)
                        del self.embeddings[idx]
                        del self.doc_ids[idx]
            
            logger.info(f"메모리 벡터 스토어에서 {len(ids)}개 문서 삭제")
            return True
            
        except Exception as e:
            logger.error(f"문서 삭제 오류: {e}")
            return False
    
    def get_document(self, doc_id: str) -> Optional[VectorDocument]:
        """특정 ID의 문서 조회"""
        return self.documents.get(doc_id)
    
    def update_document(self, document: VectorDocument) -> bool:
        """문서 업데이트"""
        
        try:
            if document.id in self.documents:
                # 기존 문서 위치 찾기
                idx = self.doc_ids.index(document.id)
                
                # 업데이트
                self.documents[document.id] = document
                self.embeddings[idx] = document.embedding
                
                logger.info(f"문서 업데이트: {document.id}")
                return True
            else:
                # 새 문서로 추가
                return len(self.add_documents([document])) > 0
                
        except Exception as e:
            logger.error(f"문서 업데이트 오류: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """컬렉션 정보 조회"""
        
        return {
            "collection_name": self.collection_name,
            "total_documents": len(self.documents),
            "dimension": self.dimension,
            "metric": self.metric,
            "store_type": "InMemory"
        }
    
    def save_to_file(self, filepath: str):
        """메모리 내용을 파일로 저장"""
        
        data = {
            "config": self.config,
            "documents": {doc_id: doc.to_dict() for doc_id, doc in self.documents.items()},
            "doc_ids": self.doc_ids,
            "embeddings": self.embeddings
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"벡터 스토어를 파일에 저장: {filepath}")
    
    def load_from_file(self, filepath: str):
        """파일에서 메모리로 로드"""
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.config = data["config"]
        self.documents = {doc_id: VectorDocument.from_dict(doc_data) 
                         for doc_id, doc_data in data["documents"].items()}
        self.doc_ids = data["doc_ids"]
        self.embeddings = data["embeddings"]
        
        logger.info(f"파일에서 벡터 스토어 로드: {filepath}")
    
    def _match_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """필터 조건 확인"""
        
        for key, value in filters.items():
            if key not in metadata:
                return False
            
            if isinstance(value, dict):
                # 범위 쿼리 지원 ($gt, $lt, $gte, $lte, $in, $nin)
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


class ChromaVectorStore(VectorStore):
    """ChromaDB 벡터 스토어"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        if not HAS_CHROMADB:
            raise ImportError("ChromaDB를 사용하려면 chromadb를 설치해주세요: pip install chromadb")
        
        # ChromaDB 설정
        self.persist_directory = config.get("persist_directory", "./chroma_db")
        self.host = config.get("host", None)
        self.port = config.get("port", None)
        
        # 클라이언트 초기화
        if self.host and self.port:
            # 서버 모드
            self.client = chromadb.HttpClient(host=self.host, port=self.port)
        else:
            # 로컬 모드
            self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # 컬렉션 생성 또는 가져오기
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"dimension": self.dimension, "metric": self.metric}
        )
        
        logger.info(f"ChromaDB 벡터 스토어 초기화: {self.collection_name}")
    
    def add_documents(
        self,
        documents: List[VectorDocument],
        **kwargs
    ) -> List[str]:
        """문서들을 ChromaDB에 추가"""
        
        if not documents:
            return []
        
        ids = [doc.id for doc in documents]
        embeddings = [doc.embedding for doc in documents]
        documents_content = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents_content,
                metadatas=metadatas
            )
            
            logger.info(f"ChromaDB에 {len(documents)}개 문서 추가")
            return ids
            
        except Exception as e:
            logger.error(f"ChromaDB 문서 추가 오류: {e}")
            raise e
    
    def search(
        self,
        query_embedding: List[float],
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> SearchResult:
        """ChromaDB에서 검색"""
        
        import time
        start_time = time.time()
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filters,
                include=["documents", "metadatas", "distances"]
            )
            
            # 결과 변환
            result_documents = []
            if results["ids"]:
                for i, doc_id in enumerate(results["ids"][0]):
                    doc = VectorDocument(
                        id=doc_id,
                        content=results["documents"][0][i],
                        embedding=[],  # ChromaDB에서는 임베딩을 반환하지 않음
                        metadata=results["metadatas"][0][i] or {},
                        score=1.0 - results["distances"][0][i]  # 거리를 유사도로 변환
                    )
                    result_documents.append(doc)
            
            search_time = time.time() - start_time
            
            return SearchResult(
                documents=result_documents,
                query="",
                total_results=len(result_documents),
                search_time=search_time
            )
            
        except Exception as e:
            logger.error(f"ChromaDB 검색 오류: {e}")
            raise e
    
    def delete_documents(self, ids: List[str]) -> bool:
        """ChromaDB에서 문서 삭제"""
        
        try:
            self.collection.delete(ids=ids)
            logger.info(f"ChromaDB에서 {len(ids)}개 문서 삭제")
            return True
            
        except Exception as e:
            logger.error(f"ChromaDB 문서 삭제 오류: {e}")
            return False
    
    def get_document(self, doc_id: str) -> Optional[VectorDocument]:
        """ChromaDB에서 특정 문서 조회"""
        
        try:
            results = self.collection.get(
                ids=[doc_id],
                include=["documents", "metadatas"]
            )
            
            if results["ids"]:
                return VectorDocument(
                    id=results["ids"][0],
                    content=results["documents"][0],
                    embedding=[],
                    metadata=results["metadatas"][0] or {}
                )
            
            return None
            
        except Exception as e:
            logger.error(f"ChromaDB 문서 조회 오류: {e}")
            return None
    
    def update_document(self, document: VectorDocument) -> bool:
        """ChromaDB에서 문서 업데이트"""
        
        try:
            self.collection.update(
                ids=[document.id],
                embeddings=[document.embedding],
                documents=[document.content],
                metadatas=[document.metadata]
            )
            
            logger.info(f"ChromaDB에서 문서 업데이트: {document.id}")
            return True
            
        except Exception as e:
            logger.error(f"ChromaDB 문서 업데이트 오류: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """ChromaDB 컬렉션 정보 조회"""
        
        try:
            count = self.collection.count()
            
            return {
                "collection_name": self.collection_name,
                "total_documents": count,
                "dimension": self.dimension,
                "metric": self.metric,
                "store_type": "ChromaDB",
                "persist_directory": self.persist_directory
            }
            
        except Exception as e:
            logger.error(f"ChromaDB 컬렉션 정보 조회 오류: {e}")
            return {"error": str(e)}


class FAISSVectorStore(VectorStore):
    """FAISS 벡터 스토어"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        if not HAS_FAISS:
            raise ImportError("FAISS를 사용하려면 faiss를 설치해주세요: pip install faiss-cpu 또는 faiss-gpu")
        
        self.index_file = config.get("index_file", f"faiss_{self.collection_name}.index")
        self.metadata_file = config.get("metadata_file", f"faiss_{self.collection_name}_metadata.json")
        
        # FAISS 인덱스 생성
        self.index_type = config.get("index_type", "IndexFlatIP")  # IndexFlatIP, IndexFlatL2, IndexIVFFlat
        
        if self.index_type == "IndexFlatIP":
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "IndexFlatL2":
            self.index = faiss.IndexFlatL2(self.dimension)
        else:
            # 기본값
            self.index = faiss.IndexFlatIP(self.dimension)
        
        # 메타데이터 저장
        self.documents_metadata: Dict[int, Dict[str, Any]] = {}
        self.id_to_index: Dict[str, int] = {}
        self.index_to_id: Dict[int, str] = {}
        self.next_index = 0
        
        # 기존 인덱스 로드
        self._load_index()
        
        logger.info(f"FAISS 벡터 스토어 초기화: {self.collection_name}")
    
    def add_documents(
        self,
        documents: List[VectorDocument],
        **kwargs
    ) -> List[str]:
        """문서들을 FAISS에 추가"""
        
        if not documents:
            return []
        
        embeddings = np.array([doc.embedding for doc in documents]).astype('float32')
        
        # FAISS에 추가
        start_idx = self.next_index
        self.index.add(embeddings)
        
        added_ids = []
        for i, doc in enumerate(documents):
            idx = start_idx + i
            self.id_to_index[doc.id] = idx
            self.index_to_id[idx] = doc.id
            self.documents_metadata[idx] = {
                "id": doc.id,
                "content": doc.content,
                "metadata": doc.metadata
            }
            added_ids.append(doc.id)
        
        self.next_index += len(documents)
        
        # 인덱스 저장
        self._save_index()
        
        logger.info(f"FAISS에 {len(documents)}개 문서 추가")
        return added_ids
    
    def search(
        self,
        query_embedding: List[float],
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> SearchResult:
        """FAISS에서 검색"""
        
        import time
        start_time = time.time()
        
        query_vector = np.array([query_embedding]).astype('float32')
        
        # FAISS 검색 (더 많은 결과를 가져와서 필터링)
        search_k = k * 3 if filters else k
        scores, indices = self.index.search(query_vector, min(search_k, self.index.ntotal))
        
        # 결과 변환 및 필터링
        result_documents = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # 유효하지 않은 인덱스
                continue
            
            if idx not in self.documents_metadata:
                continue
            
            doc_data = self.documents_metadata[idx]
            
            # 필터 적용
            if filters and not self._match_filters(doc_data["metadata"], filters):
                continue
            
            doc = VectorDocument(
                id=doc_data["id"],
                content=doc_data["content"],
                embedding=[],  # 메모리 절약을 위해 빈 리스트
                metadata=doc_data["metadata"],
                score=float(score)
            )
            result_documents.append(doc)
            
            if len(result_documents) >= k:
                break
        
        search_time = time.time() - start_time
        
        return SearchResult(
            documents=result_documents,
            query="",
            total_results=len(result_documents),
            search_time=search_time
        )
    
    def delete_documents(self, ids: List[str]) -> bool:
        """FAISS에서 문서 삭제 (실제로는 메타데이터만 삭제)"""
        
        try:
            for doc_id in ids:
                if doc_id in self.id_to_index:
                    idx = self.id_to_index[doc_id]
                    
                    # 메타데이터에서 삭제
                    if idx in self.documents_metadata:
                        del self.documents_metadata[idx]
                    
                    # 매핑에서 삭제
                    del self.id_to_index[doc_id]
                    del self.index_to_id[idx]
            
            # 인덱스 저장
            self._save_index()
            
            logger.info(f"FAISS에서 {len(ids)}개 문서 삭제 (메타데이터)")
            return True
            
        except Exception as e:
            logger.error(f"FAISS 문서 삭제 오류: {e}")
            return False
    
    def get_document(self, doc_id: str) -> Optional[VectorDocument]:
        """FAISS에서 특정 문서 조회"""
        
        if doc_id not in self.id_to_index:
            return None
        
        idx = self.id_to_index[doc_id]
        if idx not in self.documents_metadata:
            return None
        
        doc_data = self.documents_metadata[idx]
        
        return VectorDocument(
            id=doc_data["id"],
            content=doc_data["content"],
            embedding=[],
            metadata=doc_data["metadata"]
        )
    
    def update_document(self, document: VectorDocument) -> bool:
        """FAISS에서 문서 업데이트 (새로 추가)"""
        
        # FAISS는 업데이트를 직접 지원하지 않으므로 삭제 후 추가
        if document.id in self.id_to_index:
            self.delete_documents([document.id])
        
        return len(self.add_documents([document])) > 0
    
    def get_collection_info(self) -> Dict[str, Any]:
        """FAISS 컬렉션 정보 조회"""
        
        return {
            "collection_name": self.collection_name,
            "total_documents": len(self.documents_metadata),
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "metric": self.metric,
            "store_type": "FAISS",
            "index_type": self.index_type,
            "index_file": self.index_file
        }
    
    def _save_index(self):
        """FAISS 인덱스와 메타데이터 저장"""
        
        # FAISS 인덱스 저장
        faiss.write_index(self.index, self.index_file)
        
        # 메타데이터 저장
        metadata = {
            "documents_metadata": self.documents_metadata,
            "id_to_index": self.id_to_index,
            "index_to_id": self.index_to_id,
            "next_index": self.next_index,
            "config": self.config
        }
        
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    def _load_index(self):
        """FAISS 인덱스와 메타데이터 로드"""
        
        if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
            try:
                # FAISS 인덱스 로드
                self.index = faiss.read_index(self.index_file)
                
                # 메타데이터 로드
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # 인덱스는 문자열로 저장되므로 정수로 변환
                self.documents_metadata = {int(k): v for k, v in metadata["documents_metadata"].items()}
                self.id_to_index = metadata["id_to_index"]
                self.index_to_id = {int(k): v for k, v in metadata["index_to_id"].items()}
                self.next_index = metadata["next_index"]
                
                logger.info(f"FAISS 인덱스 로드: {len(self.documents_metadata)}개 문서")
                
            except Exception as e:
                logger.error(f"FAISS 인덱스 로드 오류: {e}")
    
    def _match_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """필터 조건 확인 (InMemoryVectorStore와 동일)"""
        
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


def create_vector_store(
    store_type: str,
    config: Dict[str, Any]
) -> VectorStore:
    """스토어 타입에 따른 벡터 스토어 생성"""
    
    store_type = store_type.lower()
    
    if store_type == "inmemory" or store_type == "memory":
        return InMemoryVectorStore(config)
    
    elif store_type == "chroma" or store_type == "chromadb":
        return ChromaVectorStore(config)
    
    elif store_type == "faiss":
        return FAISSVectorStore(config)
    
    else:
        raise ValueError(f"지원하지 않는 벡터 스토어 타입: {store_type}")


def get_recommended_vector_stores() -> Dict[str, Dict[str, Any]]:
    """추천 벡터 스토어 정보"""
    
    return {
        "inmemory": {
            "name": "In-Memory Vector Store",
            "description": "메모리 기반, 개발/테스트용",
            "pros": ["빠른 속도", "간단한 설정", "의존성 없음"],
            "cons": ["메모리 제한", "영구 저장 필요시 별도 저장"],
            "use_case": "프로토타입, 소규모 데이터"
        },
        "chroma": {
            "name": "ChromaDB",
            "description": "경량 벡터 데이터베이스",
            "pros": ["간단한 설정", "영구 저장", "필터링 지원"],
            "cons": ["중간 규모 데이터에 적합"],
            "use_case": "중소규모 애플리케이션"
        },
        "faiss": {
            "name": "FAISS",
            "description": "Meta의 고성능 벡터 검색",
            "pros": ["매우 빠른 검색", "대용량 데이터", "다양한 인덱스"],
            "cons": ["복잡한 설정", "메타데이터 관리 필요"],
            "use_case": "대규모 프로덕션 환경"
        }
    }