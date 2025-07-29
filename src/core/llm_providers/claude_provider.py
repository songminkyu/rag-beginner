"""
Claude API Provider
Anthropic Claude 모델들을 위한 제공자 구현
"""

import anthropic
from typing import Dict, Any, List, Optional, Iterator
import time
import logging
from .base_provider import BaseLLMProvider, LLMResponse, ChatMessage, parse_llm_response

logger = logging.getLogger(__name__)


class ClaudeProvider(BaseLLMProvider):
    """Claude API 제공자"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key", "")
        self.base_url = config.get("base_url", None)
        
        # Claude 클라이언트 초기화
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
            
        self.client = anthropic.Anthropic(**client_kwargs)
        
        # 모델별 기본 설정
        self.model_configs = {
            "claude-3-5-sonnet-20241022": {"max_tokens": 8192, "context_window": 200000},
            "claude-3-5-haiku-20241022": {"max_tokens": 8192, "context_window": 200000},
            "claude-3-opus-20240229": {"max_tokens": 4096, "context_window": 200000},
            "claude-3-sonnet-20240229": {"max_tokens": 4096, "context_window": 200000},
            "claude-3-haiku-20240307": {"max_tokens": 4096, "context_window": 200000},
        }
        
        # 현재 모델 설정 적용
        if self.model_name in self.model_configs:
            model_config = self.model_configs[self.model_name]
            self.max_tokens = min(self.max_tokens, model_config["max_tokens"])
            self.context_window = model_config["context_window"]
        else:
            self.context_window = 200000
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """단일 프롬프트로 텍스트 생성"""
        
        messages = [{"role": "user", "content": prompt}]
        
        # 시스템 프롬프트 준비
        system_content = self.prepare_system_prompt(system_prompt)
        
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                system=system_content if system_content else "",
                messages=messages,
                top_p=kwargs.get("top_p", 1.0),
                top_k=kwargs.get("top_k", 0),
            )
            
            return parse_llm_response(response, "claude")
            
        except Exception as e:
            logger.error(f"Claude API 호출 오류: {e}")
            raise e
    
    def chat(
        self,
        messages: List[ChatMessage],
        **kwargs
    ) -> LLMResponse:
        """채팅 형태로 텍스트 생성"""
        
        # ChatMessage를 Claude 형식으로 변환
        claude_messages = []
        system_message = None
        
        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                claude_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                system=system_message if system_message else "",
                messages=claude_messages,
                top_p=kwargs.get("top_p", 1.0),
                top_k=kwargs.get("top_k", 0),
            )
            
            return parse_llm_response(response, "claude")
            
        except Exception as e:
            logger.error(f"Claude Chat API 호출 오류: {e}")
            raise e
    
    def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Iterator[str]:
        """스트리밍으로 텍스트 생성"""
        
        messages = [{"role": "user", "content": prompt}]
        
        # 시스템 프롬프트 준비
        system_content = self.prepare_system_prompt(system_prompt)
        
        try:
            with self.client.messages.stream(
                model=self.model_name,
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                system=system_content if system_content else "",
                messages=messages,
                top_p=kwargs.get("top_p", 1.0),
                top_k=kwargs.get("top_k", 0),
            ) as stream:
                for text in stream.text_stream:
                    yield text
                    
        except Exception as e:
            logger.error(f"Claude Streaming API 호출 오류: {e}")
            raise e
    
    def stream_chat(
        self,
        messages: List[ChatMessage],
        **kwargs
    ) -> Iterator[str]:
        """스트리밍으로 채팅"""
        
        # ChatMessage를 Claude 형식으로 변환
        claude_messages = []
        system_message = None
        
        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                claude_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        try:
            with self.client.messages.stream(
                model=self.model_name,
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                system=system_message if system_message else "",
                messages=claude_messages,
                top_p=kwargs.get("top_p", 1.0),
                top_k=kwargs.get("top_k", 0),
            ) as stream:
                for text in stream.text_stream:
                    yield text
                    
        except Exception as e:
            logger.error(f"Claude Chat Streaming API 호출 오류: {e}")
            raise e
    
    def get_embedding(self, text: str) -> List[float]:
        """텍스트 임베딩 생성 (Claude는 자체 임베딩 미지원)"""
        logger.warning("Claude는 자체 임베딩을 지원하지 않습니다. OpenAI 임베딩을 사용하세요.")
        raise NotImplementedError("Claude는 임베딩 API를 제공하지 않습니다.")
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """여러 텍스트의 임베딩 생성 (Claude는 자체 임베딩 미지원)"""
        logger.warning("Claude는 자체 임베딩을 지원하지 않습니다. OpenAI 임베딩을 사용하세요.")
        raise NotImplementedError("Claude는 임베딩 API를 제공하지 않습니다.")
    
    def estimate_tokens(self, text: str) -> int:
        """토큰 수 추정 (Claude 방식)"""
        # Claude의 토큰 계산은 복잡하지만 간단히 추정
        # 영어: 4자당 1토큰, 한국어: 3자당 1토큰
        korean_chars = sum(1 for char in text if '\uac00' <= char <= '\ud7af')
        other_chars = len(text) - korean_chars
        
        estimated_tokens = (korean_chars // 3) + (other_chars // 4)
        return max(1, estimated_tokens)
    
    def validate_config(self) -> bool:
        """설정 유효성 검사"""
        required_fields = ["api_key", "model"]
        return all(field in self.config and self.config[field] for field in required_fields)
    
    def get_available_models(self) -> List[str]:
        """사용 가능한 모델 목록 반환"""
        return list(self.model_configs.keys())
    
    def count_tokens(self, text: str) -> int:
        """실제 토큰 수 계산 (Claude API 사용)"""
        try:
            response = self.client.messages.count_tokens(
                model=self.model_name,
                messages=[{"role": "user", "content": text}]
            )
            return response.input_tokens
        except Exception as e:
            logger.warning(f"토큰 카운팅 오류, 추정값 사용: {e}")
            return self.estimate_tokens(text)
    
    def get_context_window_size(self) -> int:
        """컨텍스트 윈도우 크기 반환"""
        return self.context_window
    
    def prepare_conversation_messages(
        self,
        conversation_history: List[ChatMessage],
        current_query: str,
        system_prompt: Optional[str] = None,
        max_history_tokens: int = 100000
    ) -> List[ChatMessage]:
        """대화 히스토리를 포함한 메시지 준비"""
        
        messages = []
        
        # 시스템 메시지 추가
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
        
        # 토큰 제한을 고려한 히스토리 추가
        history_tokens = 0
        recent_history = []
        
        # 최신 대화부터 역순으로 추가
        for msg in reversed(conversation_history):
            msg_tokens = self.estimate_tokens(msg.content)
            if history_tokens + msg_tokens > max_history_tokens:
                break
            
            recent_history.insert(0, msg)
            history_tokens += msg_tokens
        
        messages.extend(recent_history)
        
        # 현재 쿼리 추가
        messages.append(ChatMessage(role="user", content=current_query))
        
        return messages


def create_claude_provider(config: Dict[str, Any]) -> ClaudeProvider:
    """Claude 제공자 생성 헬퍼 함수"""
    
    # 기본 설정
    default_config = {
        "model": "claude-3-5-sonnet-20241022",
        "temperature": 0.1,
        "max_tokens": 4096,
    }
    
    # 사용자 설정과 병합
    final_config = {**default_config, **config}
    
    return ClaudeProvider(final_config)


class ClaudeRAGHelper:
    """Claude RAG 전용 헬퍼 클래스"""
    
    def __init__(self, provider: ClaudeProvider):
        self.provider = provider
    
    def generate_search_queries(
        self,
        user_query: str,
        num_queries: int = 3
    ) -> List[str]:
        """검색용 다중 쿼리 생성"""
        
        prompt = f"""주어진 질문에 대해 효과적인 검색을 위한 {num_queries}개의 다양한 검색 쿼리를 생성해주세요.

원본 질문: {user_query}

검색 쿼리들은 다음 조건을 만족해야 합니다:
1. 서로 다른 관점에서 접근
2. 핵심 키워드 포함
3. 구체적이고 명확한 표현

검색 쿼리만 번호 없이 한 줄씩 출력해주세요."""

        try:
            response = self.provider.generate(prompt)
            queries = [q.strip() for q in response.content.split('\n') if q.strip()]
            return queries[:num_queries] if len(queries) >= num_queries else [user_query]
        except Exception as e:
            logger.error(f"쿼리 생성 오류: {e}")
            return [user_query]
    
    def compress_context(
        self,
        context: str,
        query: str,
        max_tokens: int = 4000
    ) -> str:
        """컨텍스트 압축"""
        
        if self.provider.estimate_tokens(context) <= max_tokens:
            return context
        
        prompt = f"""다음 컨텍스트에서 주어진 질문과 관련된 핵심 정보만 추출하여 요약해주세요.

질문: {query}

컨텍스트:
{context}

요약된 핵심 정보:"""

        try:
            response = self.provider.generate(
                prompt,
                max_tokens=max_tokens
            )
            return response.content
        except Exception as e:
            logger.error(f"컨텍스트 압축 오류: {e}")
            return context[:max_tokens*4]  # 간단한 자르기 fallback
    
    def evaluate_relevance(
        self,
        query: str,
        context: str,
        threshold: float = 0.7
    ) -> float:
        """컨텍스트 관련성 평가"""
        
        prompt = f"""다음 컨텍스트가 주어진 질문에 얼마나 관련이 있는지 0.0에서 1.0 사이의 점수로 평가해주세요.

질문: {query}

컨텍스트: {context}

점수만 숫자로만 답해주세요 (예: 0.8):"""

        try:
            response = self.provider.generate(prompt, max_tokens=10)
            score = float(response.content.strip())
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"관련성 평가 오류: {e}")
            return 0.5  # 기본값