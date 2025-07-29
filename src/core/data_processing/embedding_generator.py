"""
Embedding Generator Module
다양한 임베딩 모델을 사용하여 텍스트 임베딩을 생성하는 모듈
"""

import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import time
from dataclasses import dataclass

# 외부 라이브러리 (optional imports)
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """임베딩 결과 표준화 클래스"""
    embeddings: List[List[float]]
    model: str
    dimension: int
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        
        # 차원 정보 검증
        if self.embeddings and len(self.embeddings) > 0:
            actual_dim = len(self.embeddings[0])
            if self.dimension != actual_dim:
                logger.warning(f"차원 불일치: 설정값 {self.dimension}, 실제값 {actual_dim}")
                self.dimension = actual_dim


class EmbeddingGenerator(ABC):
    """임베딩 생성기 기본 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get("model", "unknown")
        self.dimension = config.get("dimension", 768)
        self.batch_size = config.get("batch_size", 32)
        self.normalize = config.get("normalize", True)
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """단일 텍스트 임베딩 생성"""
        pass
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """여러 텍스트 임베딩 생성"""
        pass
    
    def embed_documents(
        self,
        texts: List[str],
        show_progress: bool = False
    ) -> EmbeddingResult:
        """문서 리스트에 대한 임베딩 생성 (배치 처리)"""
        
        if not texts:
            return EmbeddingResult(
                embeddings=[],
                model=self.model_name,
                dimension=self.dimension,
                metadata={"total_texts": 0}
            )
        
        all_embeddings = []
        
        # 배치 단위로 처리
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            if show_progress:
                logger.info(f"임베딩 진행률: {i+len(batch)}/{len(texts)}")
            
            try:
                batch_embeddings = self.embed_texts(batch)
                all_embeddings.extend(batch_embeddings)
                
                # API 속도 제한 대응
                if hasattr(self, '_should_delay') and self._should_delay():
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"배치 {i//self.batch_size + 1} 임베딩 실패: {e}")
                # 실패한 배치는 0으로 채운 임베딩으로 대체
                zero_embeddings = [[0.0] * self.dimension] * len(batch)
                all_embeddings.extend(zero_embeddings)
        
        return EmbeddingResult(
            embeddings=all_embeddings,
            model=self.model_name,
            dimension=self.dimension,
            metadata={
                "total_texts": len(texts),
                "batch_size": self.batch_size,
                "generator": self.__class__.__name__
            }
        )
    
    def get_dimension(self) -> int:
        """임베딩 차원 반환"""
        return self.dimension
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            "generator": self.__class__.__name__,
            "model": self.model_name,
            "dimension": self.dimension,
            "batch_size": self.batch_size,
            "normalize": self.normalize,
        }
    
    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """임베딩 정규화"""
        if not self.normalize:
            return embedding
        
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        
        return (np.array(embedding) / norm).tolist()
    
    def _normalize_embeddings(self, embeddings: List[List[float]]) -> List[List[float]]:
        """여러 임베딩 정규화"""
        if not self.normalize:
            return embeddings
        
        normalized = []
        for embedding in embeddings:
            normalized.append(self._normalize_embedding(embedding))
        
        return normalized


class OpenAIEmbeddingGenerator(EmbeddingGenerator):
    """OpenAI 임베딩 생성기"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        if not HAS_OPENAI:
            raise ImportError("OpenAI 임베딩을 사용하려면 openai를 설치해주세요: pip install openai")
        
        self.api_key = config.get("api_key", "")
        self.organization = config.get("organization", None)
        self.base_url = config.get("base_url", None)
        
        # OpenAI 클라이언트 초기화
        self.client = openai.OpenAI(
            api_key=self.api_key,
            organization=self.organization,
            base_url=self.base_url
        )
        
        # 모델별 차원 정보
        self.model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        
        # 차원 설정 업데이트
        if self.model_name in self.model_dimensions:
            self.dimension = self.model_dimensions[self.model_name]
        
        logger.info(f"OpenAI 임베딩 생성기 초기화: {self.model_name}")
    
    def embed_text(self, text: str) -> List[float]:
        """단일 텍스트 임베딩 생성"""
        
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=text,
                encoding_format="float"
            )
            
            embedding = response.data[0].embedding
            return self._normalize_embedding(embedding)
            
        except Exception as e:
            logger.error(f"OpenAI 임베딩 생성 오류: {e}")
            raise e
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """여러 텍스트 임베딩 생성"""
        
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=texts,
                encoding_format="float"
            )
            
            embeddings = [data.embedding for data in response.data]
            return self._normalize_embeddings(embeddings)
            
        except Exception as e:
            logger.error(f"OpenAI 배치 임베딩 생성 오류: {e}")
            raise e
    
    def _should_delay(self) -> bool:
        """API 속도 제한을 위한 지연 필요 여부"""
        return True  # OpenAI API 속도 제한 대응


class HuggingFaceEmbeddingGenerator(EmbeddingGenerator):
    """HuggingFace SentenceTransformers 임베딩 생성기"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError(
                "HuggingFace 임베딩을 사용하려면 sentence-transformers를 설치해주세요: "
                "pip install sentence-transformers"
            )
        
        self.device = config.get("device", "cpu")
        self.trust_remote_code = config.get("trust_remote_code", False)
        
        # GPU 사용 가능 여부 확인
        if HAS_TORCH and self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 모델 로드
        logger.info(f"HuggingFace 모델 로딩 중: {self.model_name}")
        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
            trust_remote_code=self.trust_remote_code
        )
        
        # 실제 차원 업데이트
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"HuggingFace 임베딩 생성기 초기화: {self.model_name} (dim: {self.dimension})")
    
    def embed_text(self, text: str) -> List[float]:
        """단일 텍스트 임베딩 생성"""
        
        try:
            embedding = self.model.encode([text], normalize_embeddings=self.normalize)[0]
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"HuggingFace 임베딩 생성 오류: {e}")
            raise e
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """여러 텍스트 임베딩 생성"""
        
        try:
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=self.normalize,
                batch_size=self.batch_size,
                show_progress_bar=False
            )
            
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"HuggingFace 배치 임베딩 생성 오류: {e}")
            raise e
    
    def _should_delay(self) -> bool:
        """로컬 모델이므로 지연 불필요"""
        return False


class OllamaEmbeddingGenerator(EmbeddingGenerator):
    """Ollama 임베딩 생성기"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        try:
            import ollama
            self.client = ollama.Client(
                host=config.get("host", "http://localhost:11434")
            )
        except ImportError:
            raise ImportError("Ollama 임베딩을 사용하려면 ollama를 설치해주세요: pip install ollama")
        
        self.host = config.get("host", "http://localhost:11434")
        
        logger.info(f"Ollama 임베딩 생성기 초기화: {self.model_name}")
    
    def embed_text(self, text: str) -> List[float]:
        """단일 텍스트 임베딩 생성"""
        
        try:
            response = self.client.embeddings(
                model=self.model_name,
                prompt=text
            )
            
            embedding = response['embedding']
            return self._normalize_embedding(embedding)
            
        except Exception as e:
            logger.error(f"Ollama 임베딩 생성 오류: {e}")
            raise e
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """여러 텍스트 임베딩 생성 (순차 처리)"""
        
        embeddings = []
        
        for text in texts:
            try:
                embedding = self.embed_text(text)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Ollama 텍스트 임베딩 실패: {e}")
                embeddings.append([0.0] * self.dimension)
        
        return embeddings
    
    def _should_delay(self) -> bool:
        """로컬 모델이므로 지연 불필요"""
        return False


def create_embedding_generator(
    generator_type: str,
    config: Dict[str, Any]
) -> EmbeddingGenerator:
    """생성기 타입에 따른 임베딩 생성기 생성"""
    
    generator_type = generator_type.lower()
    
    if generator_type == "openai":
        return OpenAIEmbeddingGenerator(config)
    
    elif generator_type == "huggingface" or generator_type == "sentence-transformers":
        return HuggingFaceEmbeddingGenerator(config)
    
    elif generator_type == "ollama":
        return OllamaEmbeddingGenerator(config)
    
    else:
        raise ValueError(f"지원하지 않는 임베딩 생성기 타입: {generator_type}")


def get_recommended_models() -> Dict[str, List[str]]:
    """추천 임베딩 모델 목록"""
    
    return {
        "openai": [
            "text-embedding-3-small",  # 성능 vs 비용 균형
            "text-embedding-3-large",  # 최고 성능
            "text-embedding-ada-002",  # 기본 모델
        ],
        "huggingface": [
            "sentence-transformers/all-MiniLM-L6-v2",  # 빠르고 가벼움
            "sentence-transformers/all-mpnet-base-v2",  # 고품질
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # 다국어
            "intfloat/multilingual-e5-large",  # 한국어 지원
            "jhgan/ko-sroberta-multitask",  # 한국어 특화
        ],
        "ollama": [
            "nomic-embed-text",  # 일반용
            "all-minilm",  # 가벼운 모델
            "mxbai-embed-large",  # 고성능
        ]
    }


def compare_embeddings(
    text: str,
    generators: List[EmbeddingGenerator]
) -> Dict[str, Dict[str, Any]]:
    """여러 임베딩 생성기 성능 비교"""
    
    results = {}
    
    for generator in generators:
        start_time = time.time()
        
        try:
            embedding = generator.embed_text(text)
            end_time = time.time()
            
            results[generator.model_name] = {
                "success": True,
                "dimension": len(embedding),
                "generation_time": end_time - start_time,
                "generator_type": generator.__class__.__name__,
                "first_values": embedding[:5],  # 처음 5개 값만 표시
            }
            
        except Exception as e:
            end_time = time.time()
            
            results[generator.model_name] = {
                "success": False,
                "error": str(e),
                "generation_time": end_time - start_time,
                "generator_type": generator.__class__.__name__,
            }
    
    return results


def calculate_embedding_similarity(
    embedding1: List[float],
    embedding2: List[float],
    method: str = "cosine"
) -> float:
    """두 임베딩 간의 유사도 계산"""
    
    emb1 = np.array(embedding1)
    emb2 = np.array(embedding2)
    
    if method == "cosine":
        # 코사인 유사도
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    elif method == "euclidean":
        # 유클리드 거리 (유사도로 변환)
        distance = np.linalg.norm(emb1 - emb2)
        return 1.0 / (1.0 + distance)
    
    elif method == "manhattan":
        # 맨하탄 거리 (유사도로 변환)
        distance = np.sum(np.abs(emb1 - emb2))
        return 1.0 / (1.0 + distance)
    
    else:
        raise ValueError(f"지원하지 않는 유사도 계산 방식: {method}")


def create_default_embedding_generator() -> EmbeddingGenerator:
    """기본 임베딩 생성기 생성 (HuggingFace 사용)"""
    
    config = {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384,
        "device": "auto",
        "normalize": True,
        "batch_size": 32,
    }
    
    return create_embedding_generator("huggingface", config)