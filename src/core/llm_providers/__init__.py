"""
LLM Providers Package
다양한 LLM 제공자들을 위한 통합 패키지
"""

from .base_provider import (
    BaseLLMProvider,
    EmbeddingProvider,
    LLMResponse,
    ChatMessage,
    create_chat_messages,
    parse_llm_response
)

from .openai_provider import (
    OpenAIProvider,
    OpenAIEmbeddingProvider,
    create_openai_provider,
    create_openai_embedding_provider
)

from .claude_provider import (
    ClaudeProvider,
    ClaudeRAGHelper,
    create_claude_provider
)

from .local_provider import (
    LocalProvider,
    # 다른 로컬 제공자들이 있다면 여기에 추가
)

__all__ = [
    # Base classes
    "BaseLLMProvider",
    "EmbeddingProvider",
    "LLMResponse",
    "ChatMessage",
    "create_chat_messages",
    "parse_llm_response",
    
    # OpenAI
    "OpenAIProvider",
    "OpenAIEmbeddingProvider",
    "create_openai_provider",
    "create_openai_embedding_provider",
    
    # Claude
    "ClaudeProvider",
    "ClaudeRAGHelper",
    "create_claude_provider",
    
    # Local
    "LocalProvider",
]


def create_provider(provider_type: str, config: dict):
    """제공자 타입에 따른 LLM 제공자 생성"""
    
    provider_type = provider_type.lower()
    
    if provider_type == "openai":
        return create_openai_provider(config)
    elif provider_type == "claude":
        return create_claude_provider(config)
    elif provider_type == "local":
        return LocalProvider(config)
    else:
        raise ValueError(f"지원하지 않는 제공자 타입: {provider_type}")


def get_available_providers():
    """사용 가능한 제공자 목록 반환"""
    return ["openai", "claude", "local"]


def get_provider_info():
    """제공자별 정보 반환"""
    return {
        "openai": {
            "name": "OpenAI",
            "models": ["gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-3.5-turbo"],
            "supports_embedding": True,
            "supports_streaming": True,
        },
        "claude": {
            "name": "Anthropic Claude",
            "models": [
                "claude-3-5-sonnet-20241022",
                "claude-3-5-haiku-20241022",
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307"
            ],
            "supports_embedding": False,
            "supports_streaming": True,
        },
        "local": {
            "name": "Local Models (Ollama)",
            "models": ["exaone-4.0", "llama3", "mistral"],
            "supports_embedding": True,
            "supports_streaming": True,
        }
    }