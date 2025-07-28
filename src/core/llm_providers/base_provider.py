"""
Base LLM Provider Interface
모든 LLM 제공자가 구현해야 하는 기본 인터페이스
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Iterator
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """LLM 응답 표준화 클래스"""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ChatMessage:
    """채팅 메시지 표준화 클래스"""
    role: str  # system, user, assistant
    content: str
    metadata: Optional[Dict[str, Any]] = None


class BaseLLMProvider(ABC):
    """LLM 제공자 기본 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get("model", "unknown")
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 4096)
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """단일 프롬프트로 텍스트 생성"""
        pass
    
    @abstractmethod
    def chat(
        self,
        messages: List[ChatMessage],
        **kwargs
    ) -> LLMResponse:
        """채팅 형태로 텍스트 생성"""
        pass
    
    @abstractmethod
    def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Iterator[str]:
        """스트리밍으로 텍스트 생성"""
        pass
    
    @abstractmethod
    def stream_chat(
        self,
        messages: List[ChatMessage],
        **kwargs
    ) -> Iterator[str]:
        """스트리밍으로 채팅"""
        pass
    
    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """텍스트 임베딩 생성"""
        pass
    
    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """여러 텍스트의 임베딩 생성"""
        pass
    
    def estimate_tokens(self, text: str) -> int:
        """토큰 수 추정 (기본 구현)"""
        # 간단한 추정: 4자 = 1토큰 (영어 기준)
        # 한국어는 더 복잡하지만 기본 구현으로 사용
        return len(text) // 4
    
    def validate_config(self) -> bool:
        """설정 유효성 검사"""
        required_fields = ["model"]
        return all(field in self.config for field in required_fields)
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            "provider": self.__class__.__name__,
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "config_valid": self.validate_config(),
        }
    
    def prepare_system_prompt(self, system_prompt: Optional[str] = None) -> Optional[str]:
        """시스템 프롬프트 준비"""
        default_system = "You are a helpful assistant."
        
        if system_prompt:
            return system_prompt
        
        # 한국어 모델인 경우 한국어 시스템 프롬프트 사용
        if "exaone" in self.model_name.lower() or "korean" in self.model_name.lower():
            return "당신은 도움이 되는 AI 어시스턴트입니다. 한국어로 정확하고 유용한 답변을 제공해주세요."
        
        return default_system
    
    def format_rag_prompt(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """RAG용 프롬프트 포맷팅"""
        
        # 한국어 모델용 프롬프트
        if "exaone" in self.model_name.lower() or "korean" in self.model_name.lower():
            rag_template = """다음 컨텍스트 정보를 바탕으로 질문에 답변해주세요.

컨텍스트:
{context}

질문: {query}

답변:"""
        else:
            # 영어 모델용 프롬프트
            rag_template = """Please answer the question based on the following context information.

Context:
{context}

Question: {query}

Answer:"""
        
        return rag_template.format(context=context, query=query)
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name})"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name}, temperature={self.temperature})"


class EmbeddingProvider(ABC):
    """임베딩 제공자 기본 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get("embedding_model", "unknown")
        self.dimension = config.get("dimension", 1536)
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """단일 텍스트 임베딩"""
        pass
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """여러 텍스트 임베딩"""
        pass
    
    def get_dimension(self) -> int:
        """임베딩 차원 반환"""
        return self.dimension
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            "provider": self.__class__.__name__,
            "model": self.model_name,
            "dimension": self.dimension,
        }


def create_chat_messages(
    query: str,
    context: Optional[str] = None,
    system_prompt: Optional[str] = None,
    conversation_history: Optional[List[ChatMessage]] = None
) -> List[ChatMessage]:
    """채팅 메시지 생성 헬퍼 함수"""
    
    messages = []
    
    # 시스템 메시지 추가
    if system_prompt:
        messages.append(ChatMessage(role="system", content=system_prompt))
    
    # 대화 히스토리 추가
    if conversation_history:
        messages.extend(conversation_history)
    
    # 현재 쿼리 추가 (컨텍스트가 있으면 포함)
    if context:
        user_content = f"Context: {context}\n\nQuestion: {query}"
    else:
        user_content = query
    
    messages.append(ChatMessage(role="user", content=user_content))
    
    return messages


def parse_llm_response(response: Any, provider_type: str) -> LLMResponse:
    """제공자별 응답을 표준 형태로 변환"""
    
    if provider_type == "openai":
        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            finish_reason=response.choices[0].finish_reason,
        )
    
    elif provider_type == "claude":
        return LLMResponse(
            content=response.content[0].text,
            model=response.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            finish_reason=response.stop_reason,
        )
    
    elif provider_type == "local":
        # Ollama 등 로컬 모델의 응답 형태
        if hasattr(response, 'response'):
            content = response.response
        elif isinstance(response, dict):
            content = response.get('response', str(response))
        else:
            content = str(response)
        
        return LLMResponse(
            content=content,
            model="local",
            metadata={"provider": "local"}
        )
    
    else:
        # 기본 처리
        return LLMResponse(
            content=str(response),
            model="unknown",
        )