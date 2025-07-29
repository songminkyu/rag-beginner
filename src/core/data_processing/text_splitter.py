"""
Text Splitter Module
텍스트를 효과적인 청크로 분할하는 모듈
"""

import re
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass

# 외부 라이브러리 (optional imports)
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """텍스트 청크 표준화 클래스"""
    content: str
    metadata: Dict[str, Any]
    start_index: int = 0
    end_index: int = 0
    chunk_id: Optional[str] = None
    overlap_with_previous: bool = False
    overlap_with_next: bool = False
    
    def __post_init__(self):
        if self.chunk_id is None:
            # 시작 인덱스를 기반으로 ID 생성
            self.chunk_id = f"chunk_{self.start_index}_{self.end_index}"
        
        if self.end_index == 0:
            self.end_index = self.start_index + len(self.content)


class TextSplitter(ABC):
    """텍스트 분할기 기본 클래스"""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        length_function: Optional[Callable[[str], int]] = None,
        keep_separator: bool = True,
        add_start_index: bool = False,
    ):
        """
        Args:
            chunk_size: 청크의 최대 크기
            chunk_overlap: 청크 간 겹치는 부분의 크기
            length_function: 텍스트 길이를 계산하는 함수
            keep_separator: 구분자를 결과에 포함할지 여부
            add_start_index: 시작 인덱스를 메타데이터에 추가할지 여부
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function or len
        self.keep_separator = keep_separator
        self.add_start_index = add_start_index
    
    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """텍스트를 청크로 분할"""
        pass
    
    def create_chunks(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[TextChunk]:
        """텍스트를 청크 객체로 분할"""
        
        splits = self.split_text(text)
        chunks = []
        
        base_metadata = metadata or {}
        current_index = 0
        
        for i, split in enumerate(splits):
            # 청크별 메타데이터 생성
            chunk_metadata = {
                **base_metadata,
                "chunk_index": i,
                "total_chunks": len(splits),
                "chunk_size": len(split),
                "splitter": self.__class__.__name__,
            }
            
            if self.add_start_index:
                start_index = text.find(split, current_index)
                if start_index == -1:
                    start_index = current_index
                else:
                    current_index = start_index + len(split)
            else:
                start_index = 0
            
            chunk = TextChunk(
                content=split,
                metadata=chunk_metadata,
                start_index=start_index,
                end_index=start_index + len(split),
                chunk_id=f"chunk_{i}_{start_index}",
                overlap_with_previous=i > 0 and self.chunk_overlap > 0,
                overlap_with_next=i < len(splits) - 1 and self.chunk_overlap > 0,
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """분할된 텍스트를 적절한 크기로 병합"""
        
        docs = []
        current_doc = []
        total = 0
        
        for split in splits:
            _len = self.length_function(split)
            
            if total + _len + (len(current_doc) * len(separator)) > self.chunk_size:
                if current_doc:
                    doc = separator.join(current_doc).strip()
                    if doc:
                        docs.append(doc)
                    
                    # 오버랩 처리
                    while (
                        total > self.chunk_overlap
                        and current_doc
                        and total + _len + (len(current_doc) * len(separator)) > self.chunk_size
                    ):
                        total -= self.length_function(current_doc[0]) + len(separator)
                        current_doc = current_doc[1:]
            
            current_doc.append(split)
            total += _len + len(separator)
        
        # 마지막 청크 추가
        if current_doc:
            doc = separator.join(current_doc).strip()
            if doc:
                docs.append(doc)
        
        return docs


class RecursiveTextSplitter(TextSplitter):
    """재귀적 텍스트 분할기"""
    
    def __init__(
        self,
        separators: Optional[List[str]] = None,
        is_separator_regex: bool = False,
        **kwargs
    ):
        """
        Args:
            separators: 사용할 구분자 목록 (우선순위 순)
            is_separator_regex: 구분자가 정규식인지 여부
        """
        super().__init__(**kwargs)
        
        # 기본 구분자 (한국어와 영어 모두 고려)
        self._separators = separators or [
            "\n\n",  # 문단 구분
            "\n",    # 줄 구분
            ". ",    # 영어 문장 구분
            "。",     # 일본어/중국어 문장 구분
            "? ",    # 영어 의문문
            "？",     # 중국어/일본어 의문문
            "! ",    # 영어 감탄문
            "！",     # 중국어/일본어 감탄문
            ";",     # 세미콜론
            ",",     # 쉼표
            " ",     # 공백
            "",      # 마지막 fallback
        ]
        
        self.is_separator_regex = is_separator_regex
    
    def split_text(self, text: str) -> List[str]:
        """재귀적으로 텍스트 분할"""
        
        return self._split_text(text, self._separators)
    
    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """주어진 구분자로 텍스트 분할"""
        
        final_chunks = []
        separator = separators[-1]
        new_separators = []
        
        for i, _s in enumerate(separators):
            _separator = _s if self.is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1:]
                break
        
        _separator = separator if self.is_separator_regex else re.escape(separator)
        splits = self._split_text_with_regex(text, _separator, self.keep_separator)
        
        # 좋은 분할이 된 경우 merge 처리
        _good_splits = []
        for s in splits:
            if self.length_function(s) < self.chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, separator)
            final_chunks.extend(merged_text)
        
        return final_chunks
    
    def _split_text_with_regex(
        self,
        text: str,
        separator: str,
        keep_separator: bool
    ) -> List[str]:
        """정규식을 사용하여 텍스트 분할"""
        
        if separator:
            if keep_separator:
                # 구분자를 유지하면서 분할
                _splits = re.split(f"({separator})", text)
                splits = [_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)]
                if len(_splits) % 2 == 0:
                    splits += _splits[-1:]
                splits = [_splits[0]] + splits
            else:
                splits = re.split(separator, text)
        else:
            splits = list(text)
        
        return [s for s in splits if s != ""]


class SemanticTextSplitter(TextSplitter):
    """의미적 텍스트 분할기 (문장 임베딩 기반)"""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.8,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError(
                "SemanticTextSplitter를 사용하려면 sentence-transformers를 설치해주세요: "
                "pip install sentence-transformers"
            )
        
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.model = SentenceTransformer(model_name)
        
        logger.info(f"의미적 텍스트 분할기 초기화: {model_name}")
    
    def split_text(self, text: str) -> List[str]:
        """의미적 유사성을 기반으로 텍스트 분할"""
        
        # 먼저 문장 단위로 분할
        sentences = self._split_into_sentences(text)
        
        if len(sentences) <= 1:
            return [text]
        
        # 문장들의 임베딩 계산
        embeddings = self.model.encode(sentences)
        
        # 유사성 기반으로 청크 생성
        chunks = []
        current_chunk = [sentences[0]]
        current_embedding = embeddings[0:1]
        
        for i in range(1, len(sentences)):
            # 현재 청크와 다음 문장의 유사성 계산
            next_embedding = embeddings[i:i+1]
            
            # 현재 청크의 평균 임베딩과 비교
            avg_embedding = current_embedding.mean(axis=0, keepdims=True)
            similarity = self._cosine_similarity(avg_embedding, next_embedding)[0][0]
            
            # 유사성이 임계값보다 높고 청크 크기가 제한을 넘지 않으면 추가
            if (similarity >= self.similarity_threshold and 
                self.length_function(' '.join(current_chunk + [sentences[i]])) <= self.chunk_size):
                
                current_chunk.append(sentences[i])
                current_embedding = embeddings[:i+1] if len(current_embedding) == 1 else embeddings[len(chunks):i+1]
                
            else:
                # 현재 청크 완료
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentences[i]]
                current_embedding = embeddings[i:i+1]
        
        # 마지막 청크 추가
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """텍스트를 문장 단위로 분할"""
        
        # 한국어와 영어 문장 끝 패턴
        sentence_endings = r'[.!?。！？]\s+'
        sentences = re.split(sentence_endings, text)
        
        # 빈 문장 제거 및 정리
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _cosine_similarity(self, a, b):
        """코사인 유사도 계산"""
        import numpy as np
        return np.dot(a, b.T) / (np.linalg.norm(a, axis=1, keepdims=True) * np.linalg.norm(b, axis=1))


class TokenBasedTextSplitter(TextSplitter):
    """토큰 기반 텍스트 분할기"""
    
    def __init__(
        self,
        model_name: str = "gpt-4",
        encoding_name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        if not HAS_TIKTOKEN:
            raise ImportError(
                "TokenBasedTextSplitter를 사용하려면 tiktoken을 설치해주세요: "
                "pip install tiktoken"
            )
        
        self.model_name = model_name
        self.encoding_name = encoding_name
        
        # 인코딩 초기화
        if encoding_name:
            self.encoding = tiktoken.get_encoding(encoding_name)
        else:
            try:
                self.encoding = tiktoken.encoding_for_model(model_name)
            except KeyError:
                logger.warning(f"모델 {model_name}의 인코딩을 찾을 수 없습니다. cl100k_base를 사용합니다.")
                self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # 토큰 길이 함수로 오버라이드
        self.length_function = self._token_length
        
        logger.info(f"토큰 기반 텍스트 분할기 초기화: {model_name}")
    
    def _token_length(self, text: str) -> int:
        """텍스트의 토큰 수 계산"""
        return len(self.encoding.encode(text))
    
    def split_text(self, text: str) -> List[str]:
        """토큰 기반으로 텍스트 분할"""
        
        # 토큰으로 인코딩
        tokens = self.encoding.encode(text)
        
        if len(tokens) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(tokens):
            # 청크 크기만큼 토큰 추출
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            
            # 토큰을 텍스트로 디코딩
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            # 오버랩을 고려하여 다음 시작점 설정
            start = end - self.chunk_overlap
            
            if start >= end:
                break
        
        return chunks


class CharacterTextSplitter(TextSplitter):
    """문자 기반 단순 텍스트 분할기"""
    
    def __init__(self, separator: str = "\n\n", **kwargs):
        super().__init__(**kwargs)
        self.separator = separator
    
    def split_text(self, text: str) -> List[str]:
        """지정된 구분자로 텍스트 분할"""
        
        if self.separator:
            splits = text.split(self.separator)
        else:
            splits = list(text)
        
        return self._merge_splits(splits, self.separator)


class KoreanTextSplitter(RecursiveTextSplitter):
    """한국어 특화 텍스트 분할기"""
    
    def __init__(self, **kwargs):
        # 한국어 특화 구분자
        korean_separators = [
            "\n\n",   # 문단 구분
            "\n",     # 줄 구분
            ". ",     # 영어 문장
            "。",     # 한자 문장
            "? ",     # 영어 의문문
            "？",     # 중국어/일본어 의문문  
            "! ",     # 영어 감탄문
            "！",     # 중국어/일본어 감탄문
            "다. ",   # 한국어 서술형 종결어미
            "요. ",   # 한국어 존댓말 종결어미
            "까? ",   # 한국어 의문형 종결어미
            "네. ",   # 한국어 대답
            "니다. ", # 한국어 존댓말 서술형
            "습니다. ", # 한국어 존댓말 서술형
            "; ",     # 세미콜론
            ", ",     # 쉼표
            " ",      # 공백
            "",       # fallback
        ]
        
        super().__init__(separators=korean_separators, **kwargs)


def create_text_splitter(
    splitter_type: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    **kwargs
) -> TextSplitter:
    """분할기 타입에 따른 텍스트 분할기 생성"""
    
    splitter_type = splitter_type.lower()
    
    base_config = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        **kwargs
    }
    
    if splitter_type == "recursive":
        return RecursiveTextSplitter(**base_config)
    
    elif splitter_type == "semantic":
        return SemanticTextSplitter(**base_config)
    
    elif splitter_type == "token":
        return TokenBasedTextSplitter(**base_config)
    
    elif splitter_type == "character":
        return CharacterTextSplitter(**base_config)
    
    elif splitter_type == "korean":
        return KoreanTextSplitter(**base_config)
    
    else:
        raise ValueError(f"지원하지 않는 분할기 타입: {splitter_type}")


def get_optimal_chunk_size(
    text: str,
    target_chunks: int = 10,
    min_chunk_size: int = 200,
    max_chunk_size: int = 2000
) -> int:
    """텍스트에 대한 최적 청크 크기 계산"""
    
    text_length = len(text)
    optimal_size = text_length // target_chunks
    
    # 최소/최대 크기 범위 내로 조정
    optimal_size = max(min_chunk_size, min(optimal_size, max_chunk_size))
    
    return optimal_size