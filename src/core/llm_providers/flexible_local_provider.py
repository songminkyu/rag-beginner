"""
Flexible Local Model Provider
다양한 로컬 모델을 지원하는 유연한 제공자 (Ollama + HuggingFace)
"""

import os
import logging
import torch
from typing import Dict, Any, List, Optional, Iterator, Union
from abc import ABC, abstractmethod

from .base_provider import BaseLLMProvider, EmbeddingProvider, LLMResponse, ChatMessage

logger = logging.getLogger(__name__)


class LocalModelStrategy(ABC):
    """로컬 모델 전략 인터페이스"""
    
    @abstractmethod
    def load_model(self, config: Dict[str, Any]) -> bool:
        """모델 로드"""
        pass
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """텍스트 생성"""
        pass
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """채팅 형태 생성"""
        pass
    
    @abstractmethod
    def cleanup(self):
        """리소스 정리"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        pass


class OllamaStrategy(LocalModelStrategy):
    """Ollama 기반 모델 전략"""
    
    def __init__(self):
        self.client = None
        self.model_name = None
        self.base_url = None
    
    def load_model(self, config: Dict[str, Any]) -> bool:
        """Ollama 모델 로드"""
        try:
            import ollama
            
            self.base_url = config.get("ollama_base_url", "http://localhost:11434")
            self.model_name = config.get("model", "llama3.1:8b")
            
            self.client = ollama.Client(host=self.base_url)
            
            # 서버 연결 확인
            self.client.list()
            
            # 모델 존재 확인
            models = self.client.list()
            model_names = [model['name'] for model in models['models']]
            
            if self.model_name not in model_names:
                logger.info(f"모델 다운로드 중: {self.model_name}")
                self.client.pull(self.model_name)
            
            logger.info(f"Ollama 모델 로드 완료: {self.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Ollama 모델 로드 실패: {e}")
            return False
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Ollama 텍스트 생성"""
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': kwargs.get('temperature', 0.1),
                    'num_predict': kwargs.get('max_new_tokens', 1024),
                    'top_p': kwargs.get('top_p', 0.9),
                    'repeat_penalty': kwargs.get('repeat_penalty', 1.0),
                }
            )
            return response['response']
            
        except Exception as e:
            logger.error(f"Ollama 생성 오류: {e}")
            raise
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Ollama 채팅"""
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                options={
                    'temperature': kwargs.get('temperature', 0.1),
                    'num_predict': kwargs.get('max_new_tokens', 1024),
                    'top_p': kwargs.get('top_p', 0.9),
                    'repeat_penalty': kwargs.get('repeat_penalty', 1.0),
                }
            )
            return response['message']['content']
            
        except Exception as e:
            logger.error(f"Ollama 채팅 오류: {e}")
            raise
    
    def cleanup(self):
        """Ollama 리소스 정리"""
        self.client = None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Ollama 모델 정보"""
        return {
            "strategy": "ollama",
            "model": self.model_name,
            "base_url": self.base_url,
            "available": self.client is not None
        }


class HuggingFaceStrategy(LocalModelStrategy):
    """HuggingFace Transformers 기반 모델 전략"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.device = None
    
    def load_model(self, config: Dict[str, Any]) -> bool:
        """HuggingFace 모델 로드"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            self.model_name = config.get("hf_model", "microsoft/DialoGPT-medium")
            self.device = config.get("device", "auto")
            torch_dtype = config.get("torch_dtype", "float16")
            
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # 패딩 토큰 설정
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 모델 로드
            torch_dtype_obj = getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype_obj,
                device_map=self.device,
                low_cpu_mem_usage=config.get("low_cpu_mem_usage", True),
                trust_remote_code=True
            )
            
            logger.info(f"HuggingFace 모델 로드 완료: {self.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"HuggingFace 모델 로드 실패: {e}")
            return False
    
    def generate(self, prompt: str, **kwargs) -> str:
        """HuggingFace 텍스트 생성"""
        try:
            # 토크나이징
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            
            if hasattr(self.model, 'device'):
                inputs = inputs.to(self.model.device)
            
            # 생성 파라미터
            generation_kwargs = {
                "max_new_tokens": kwargs.get("max_new_tokens", 1024),
                "temperature": kwargs.get("temperature", 0.1),
                "do_sample": kwargs.get("temperature", 0.1) > 0,
                "top_p": kwargs.get("top_p", 0.9),
                "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                "use_cache": kwargs.get("use_cache", True),
            }
            
            # 생성
            with torch.no_grad():
                outputs = self.model.generate(inputs, **generation_kwargs)
            
            # 디코딩 (입력 제거)
            generated_tokens = outputs[0][len(inputs[0]):]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"HuggingFace 생성 오류: {e}")
            raise
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """HuggingFace 채팅 (간단한 구현)"""
        try:
            # 메시지를 단순한 프롬프트로 변환
            prompt_parts = []
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                if role == 'system':
                    prompt_parts.append(f"System: {content}")
                elif role == 'user':
                    prompt_parts.append(f"Human: {content}")
                elif role == 'assistant':
                    prompt_parts.append(f"Assistant: {content}")
            
            prompt_parts.append("Assistant:")
            prompt = "\\n".join(prompt_parts)
            
            return self.generate(prompt, **kwargs)
            
        except Exception as e:
            logger.error(f"HuggingFace 채팅 오류: {e}")
            raise
    
    def cleanup(self):
        """HuggingFace 리소스 정리"""
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer  
            self.tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_model_info(self) -> Dict[str, Any]:
        """HuggingFace 모델 정보"""
        return {
            "strategy": "huggingface",
            "model": self.model_name,
            "device": str(self.model.device) if self.model else "unknown",
            "available": self.model is not None
        }


class FlexibleLocalProvider(BaseLLMProvider):
    """유연한 로컬 모델 제공자"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.config = config
        self.model_name = config.get("model", "microsoft/DialoGPT-medium")
        self.model_type = config.get("model_type", "auto")
        self.korean_optimized = config.get("korean_optimized", True)
        
        self.strategy: Optional[LocalModelStrategy] = None
        self._load_strategy()
    
    def _load_strategy(self):
        """적절한 모델 전략 로드"""
        
        # 자동 감지
        if self.model_type == "auto":
            self.model_type = self._detect_model_type()
        
        try:
            if self._should_use_ollama():
                logger.info("Ollama 전략 사용")
                self.strategy = OllamaStrategy()
            else:
                logger.info("HuggingFace 전략 사용")
                self.strategy = HuggingFaceStrategy()
            
            # 모델 로드 시도
            success = self.strategy.load_model(self.config)
            if not success:
                logger.warning("첫 번째 전략 실패, 대체 전략 시도")
                self._try_fallback_strategy()
                
        except Exception as e:
            logger.error(f"모델 전략 로드 실패: {e}")
            self._try_fallback_strategy()
    
    def _detect_model_type(self) -> str:
        """모델 타입 자동 감지"""
        model_name = self.model_name.lower()
        
        if "exaone" in model_name:
            return "exaone"
        elif "llama" in model_name:
            return "llama"
        elif "mistral" in model_name:
            return "mistral"
        elif "codellama" in model_name:
            return "codellama"
        elif "phi" in model_name:
            return "phi"
        elif "qwen" in model_name:
            return "qwen"
        elif "gpt" in model_name:
            return "gpt"
        else:
            return "generic"
    
    def _should_use_ollama(self) -> bool:
        """Ollama 사용 여부 결정"""
        
        # 모델 이름이 Ollama 형식인지 확인
        if ":" in self.model_name and "/" not in self.model_name:
            return True
        
        # Ollama 서버 연결 가능한지 확인
        try:
            import requests
            base_url = self.config.get("ollama_base_url", "http://localhost:11434")
            response = requests.get(f"{base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def _try_fallback_strategy(self):
        """대체 전략 시도"""
        try:
            if isinstance(self.strategy, OllamaStrategy):
                logger.info("HuggingFace 대체 전략 시도")
                self.strategy = HuggingFaceStrategy()
            else:
                logger.info("Ollama 대체 전략 시도")  
                self.strategy = OllamaStrategy()
            
            success = self.strategy.load_model(self.config)
            if not success:
                raise Exception("모든 전략 실패")
                
        except Exception as e:
            logger.error(f"대체 전략도 실패: {e}")
            raise Exception("사용 가능한 로컬 모델이 없습니다")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> LLMResponse:
        """텍스트 생성"""
        if not self.strategy:
            raise RuntimeError("모델 전략이 로드되지 않았습니다")
        
        try:
            # 시스템 프롬프트 포함
            full_prompt = self._prepare_prompt(prompt, system_prompt)
            
            # 모델별 파라미터 조정
            adjusted_kwargs = self._adjust_parameters(kwargs)
            
            # 생성
            content = self.strategy.generate(full_prompt, **adjusted_kwargs)
            
            return LLMResponse(
                content=content,
                model=self.model_name,
                metadata={
                    "provider": "flexible_local",
                    "strategy": self.strategy.__class__.__name__,
                    "model_type": self.model_type,
                }
            )
            
        except Exception as e:
            logger.error(f"생성 오류: {e}")
            raise
    
    def chat(self, messages: List[ChatMessage], **kwargs) -> LLMResponse:
        """채팅 형태 생성"""
        if not self.strategy:
            raise RuntimeError("모델 전략이 로드되지 않았습니다")
        
        try:
            # ChatMessage를 dict로 변환
            message_dicts = []
            for msg in messages:
                message_dicts.append({
                    "role": msg.role,
                    "content": msg.content
                })
            
            # 모델별 파라미터 조정
            adjusted_kwargs = self._adjust_parameters(kwargs)
            
            # 생성
            content = self.strategy.chat(message_dicts, **adjusted_kwargs)
            
            return LLMResponse(
                content=content,
                model=self.model_name,
                metadata={
                    "provider": "flexible_local",
                    "strategy": self.strategy.__class__.__name__,
                    "model_type": self.model_type,
                }
            )
            
        except Exception as e:
            logger.error(f"채팅 오류: {e}")
            raise
    
    def stream_generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Iterator[str]:
        """스트리밍 생성 (시뮬레이션)"""
        response = self.generate(prompt, system_prompt, **kwargs)
        
        # 단어별로 스트리밍 시뮬레이션
        words = response.content.split()
        for i, word in enumerate(words):
            if i == 0:
                yield word
            else:
                yield " " + word
    
    def stream_chat(self, messages: List[ChatMessage], **kwargs) -> Iterator[str]:
        """스트리밍 채팅 (시뮬레이션)"""
        response = self.chat(messages, **kwargs)
        
        # 단어별로 스트리밍 시뮬레이션
        words = response.content.split()
        for i, word in enumerate(words):
            if i == 0:
                yield word
            else:
                yield " " + word
    
    def get_embedding(self, text: str) -> List[float]:
        """임베딩은 별도 제공자 사용"""
        raise NotImplementedError("임베딩은 LocalEmbeddingProvider를 사용하세요")
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """임베딩은 별도 제공자 사용"""
        raise NotImplementedError("임베딩은 LocalEmbeddingProvider를 사용하세요")
    
    def _prepare_prompt(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """프롬프트 준비"""
        if not system_prompt:
            system_prompt = self.prepare_system_prompt()
        
        if system_prompt:
            return f"{system_prompt}\\n\\nUser: {prompt}\\nAssistant:"
        else:
            return prompt
    
    def _adjust_parameters(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """모델 타입별 파라미터 조정"""
        adjusted = kwargs.copy()
        
        # 모델별 최적화
        if self.model_type == "exaone":
            adjusted.setdefault("repeat_penalty", 1.0)
            adjusted.setdefault("temperature", 0.1)
        elif self.model_type in ["llama", "mistral"]:
            adjusted.setdefault("temperature", 0.1)
            adjusted.setdefault("top_p", 0.9)
        
        # 한국어 최적화
        if self.korean_optimized:
            adjusted.setdefault("temperature", min(adjusted.get("temperature", 0.1), 0.6))
        
        return adjusted
    
    def validate_config(self) -> bool:
        """설정 검증"""
        return self.strategy is not None
    
    def cleanup(self):
        """리소스 정리"""
        if self.strategy:
            self.strategy.cleanup()
            self.strategy = None
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보"""
        base_info = {
            "provider": "flexible_local",
            "model": self.model_name,
            "model_type": self.model_type,
            "korean_optimized": self.korean_optimized,
        }
        
        if self.strategy:
            base_info.update(self.strategy.get_model_info())
        
        return base_info


class LocalEmbeddingProvider(EmbeddingProvider):
    """로컬 임베딩 제공자 (기존과 동일)"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.model_name = config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        self.korean_optimized = config.get("korean_optimized", True)
        
        # 한국어 특화 모델 선택
        if self.korean_optimized and "korean" not in self.model_name.lower():
            self.model_name = config.get(
                "embedding_model",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
        
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"임베딩 모델 로드 완료: {self.model_name} (차원: {self.dimension})")
        except Exception as e:
            logger.error(f"임베딩 모델 로드 실패: {e}")
            raise
    
    def embed_text(self, text: str) -> List[float]:
        """단일 텍스트 임베딩"""
        try:
            embedding = self.model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"텍스트 임베딩 오류: {e}")
            raise
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """여러 텍스트 임베딩"""
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            logger.error(f"텍스트들 임베딩 오류: {e}")
            raise


def create_flexible_local_provider(config: Dict[str, Any]) -> FlexibleLocalProvider:
    """유연한 로컬 제공자 생성"""
    return FlexibleLocalProvider(config)


def create_local_embedding_provider(config: Dict[str, Any]) -> LocalEmbeddingProvider:
    """로컬 임베딩 제공자 생성"""
    return LocalEmbeddingProvider(config)


if __name__ == "__main__":
    # 테스트 코드
    from src.config.api_config import config_manager
    
    config = config_manager.get_provider_config("local")
    
    # 유연한 로컬 제공자 테스트
    provider = FlexibleLocalProvider(config)
    
    # 간단한 테스트
    try:
        response = provider.generate("안녕하세요! 간단한 인사를 해주세요.")
        print(f"Response: {response.content}")
        print(f"Model info: {provider.get_model_info()}")
    except Exception as e:
        print(f"테스트 실패: {e}")
    finally:
        provider.cleanup()