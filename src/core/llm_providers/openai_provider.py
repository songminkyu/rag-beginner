"""
OpenAI API Provider
OpenAI GPT 모델들을 위한 제공자 구현
"""

import openai
from typing import Dict, Any, List, Optional, Iterator
import time
import logging
from .base_provider import BaseLLMProvider, EmbeddingProvider, LLMResponse, ChatMessage, parse_llm_response

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API 제공자"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key", "")
        self.organization = config.get("organization", None)
        self.base_url = config.get("base_url", None)
        
        # OpenAI 클라이언트 초기화
        self.client = openai.OpenAI(
            api_key=self.api_key,
            organization=self.organization,
            base_url=self.base_url
        )
        
        # 모델별 기본 설정
        self.model_configs = {
            "gpt-4": {"max_tokens": 8192, "context_window": 8192},
            "gpt-4-turbo": {"max_tokens": 4096, "context_window": 128000},
            "gpt-4o": {"max_tokens": 4096, "context_window": 128000},
            "gpt-3.5-turbo": {"max_tokens": 4096, "context_window": 16385},
        }
        
        # 현재 모델 설정 적용
        if self.model_name in self.model_configs:
            model_config = self.model_configs[self.model_name]
            self.max_tokens = min(self.max_tokens, model_config["max_tokens"])
            self.context_window = model_config["context_window"]
        else:
            self.context_window = 4096
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """단일 프롬프트로 텍스트 생성"""
        
        messages = []
        
        # 시스템 프롬프트 추가
        system_content = self.prepare_system_prompt(system_prompt)
        if system_content:
            messages.append({"role": "system", "content": system_content})
        
        # 사용자 프롬프트 추가
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                top_p=kwargs.get("top_p", 1.0),
                frequency_penalty=kwargs.get("frequency_penalty", 0.0),
                presence_penalty=kwargs.get("presence_penalty", 0.0),
            )
            
            return parse_llm_response(response, "openai")
            
        except Exception as e:
            logger.error(f"OpenAI API 호출 오류: {e}")
            raise e
    
    def chat(
        self,
        messages: List[ChatMessage],
        **kwargs
    ) -> LLMResponse:
        """채팅 형태로 텍스트 생성"""
        
        # ChatMessage를 OpenAI 형식으로 변환
        openai_messages = []
        for msg in messages:
            openai_messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=openai_messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                top_p=kwargs.get("top_p", 1.0),
                frequency_penalty=kwargs.get("frequency_penalty", 0.0),
                presence_penalty=kwargs.get("presence_penalty", 0.0),
            )
            
            return parse_llm_response(response, "openai")
            
        except Exception as e:
            logger.error(f"OpenAI Chat API 호출 오류: {e}")
            raise e
    
    def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Iterator[str]:
        """스트리밍으로 텍스트 생성"""
        
        messages = []
        
        # 시스템 프롬프트 추가
        system_content = self.prepare_system_prompt(system_prompt)
        if system_content:
            messages.append({"role": "system", "content": system_content})
        
        # 사용자 프롬프트 추가
        messages.append({"role": "user", "content": prompt})
        
        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                top_p=kwargs.get("top_p", 1.0),
                frequency_penalty=kwargs.get("frequency_penalty", 0.0),
                presence_penalty=kwargs.get("presence_penalty", 0.0),
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"OpenAI Streaming API 호출 오류: {e}")
            raise e
    
    def stream_chat(
        self,
        messages: List[ChatMessage],
        **kwargs
    ) -> Iterator[str]:
        """스트리밍으로 채팅"""
        
        # ChatMessage를 OpenAI 형식으로 변환
        openai_messages = []
        for msg in messages:
            openai_messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=openai_messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                top_p=kwargs.get("top_p", 1.0),
                frequency_penalty=kwargs.get("frequency_penalty", 0.0),
                presence_penalty=kwargs.get("presence_penalty", 0.0),
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"OpenAI Chat Streaming API 호출 오류: {e}")
            raise e
    
    def get_embedding(self, text: str) -> List[float]:
        """텍스트 임베딩 생성"""
        embedding_model = self.config.get("embedding_model", "text-embedding-3-small")
        
        try:
            response = self.client.embeddings.create(
                model=embedding_model,
                input=text,
                encoding_format="float"
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"OpenAI Embedding API 호출 오류: {e}")
            raise e
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """여러 텍스트의 임베딩 생성"""
        embedding_model = self.config.get("embedding_model", "text-embedding-3-small")
        
        try:
            response = self.client.embeddings.create(
                model=embedding_model,
                input=texts,
                encoding_format="float"
            )
            
            return [data.embedding for data in response.data]
            
        except Exception as e:
            logger.error(f"OpenAI Embeddings API 호출 오류: {e}")
            raise e
    
    def estimate_tokens(self, text: str) -> int:
        """토큰 수 추정 (OpenAI 방식)"""
        # 간단한 추정: 영어는 4자당 1토큰, 한국어는 2자당 1토큰
        korean_chars = sum(1 for char in text if '\uac00' <= char <= '\ud7af')
        other_chars = len(text) - korean_chars
        
        estimated_tokens = (korean_chars // 2) + (other_chars // 4)
        return max(1, estimated_tokens)
    
    def validate_config(self) -> bool:
        """설정 유효성 검사"""
        required_fields = ["api_key", "model"]
        return all(field in self.config and self.config[field] for field in required_fields)
    
    def get_available_models(self) -> List[str]:
        """사용 가능한 모델 목록 반환"""
        try:
            models = self.client.models.list()
            return [model.id for model in models.data if "gpt" in model.id]
        except Exception as e:
            logger.error(f"모델 목록 조회 오류: {e}")
            return list(self.model_configs.keys())


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI 임베딩 전용 제공자"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key", "")
        self.organization = config.get("organization", None)
        self.base_url = config.get("base_url", None)
        
        # OpenAI 클라이언트 초기화
        self.client = openai.OpenAI(
            api_key=self.api_key,
            organization=self.organization,
            base_url=self.base_url
        )
        
        # 임베딩 모델별 차원 정보
        self.embedding_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        
        # 차원 설정 업데이트
        if self.model_name in self.embedding_dimensions:
            self.dimension = self.embedding_dimensions[self.model_name]
    
    def embed_text(self, text: str) -> List[float]:
        """단일 텍스트 임베딩"""
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=text,
                encoding_format="float"
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"OpenAI Embedding API 호출 오류: {e}")
            raise e
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """여러 텍스트 임베딩"""
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=texts,
                encoding_format="float"
            )
            
            return [data.embedding for data in response.data]
            
        except Exception as e:
            logger.error(f"OpenAI Embeddings API 호출 오류: {e}")
            raise e
    
    def batch_embed_texts(
        self, 
        texts: List[str], 
        batch_size: int = 100,
        delay: float = 0.1
    ) -> List[List[float]]:
        """대량 텍스트 배치 임베딩"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                embeddings = self.embed_texts(batch)
                all_embeddings.extend(embeddings)
                
                # API 속도 제한 대응
                if delay > 0:
                    time.sleep(delay)
                    
            except Exception as e:
                logger.error(f"배치 {i//batch_size + 1} 임베딩 오류: {e}")
                # 실패한 배치는 빈 임베딩으로 채움
                all_embeddings.extend([[0.0] * self.dimension] * len(batch))
        
        return all_embeddings


def create_openai_provider(config: Dict[str, Any]) -> OpenAIProvider:
    """OpenAI 제공자 생성 헬퍼 함수"""
    
    # 기본 설정
    default_config = {
        "model": "gpt-4o-mini",
        "temperature": 0.1,
        "max_tokens": 4096,
        "embedding_model": "text-embedding-3-small"
    }
    
    # 사용자 설정과 병합
    final_config = {**default_config, **config}
    
    return OpenAIProvider(final_config)


def create_openai_embedding_provider(config: Dict[str, Any]) -> OpenAIEmbeddingProvider:
    """OpenAI 임베딩 제공자 생성 헬퍼 함수"""
    
    # 기본 설정
    default_config = {
        "embedding_model": "text-embedding-3-small",
        "dimension": 1536
    }
    
    # 사용자 설정과 병합
    final_config = {**default_config, **config}
    final_config["model"] = final_config["embedding_model"]
    
    return OpenAIEmbeddingProvider(final_config)