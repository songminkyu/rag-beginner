"""
Local Model Provider (EXAONE via Ollama)
EXAONE 4.0 모델을 Ollama를 통해 사용하는 제공자
"""

import ollama
import requests
from typing import Dict, Any, List, Optional, Iterator
from sentence_transformers import SentenceTransformer
import logging

from .base_provider import BaseLLMProvider, EmbeddingProvider, LLMResponse, ChatMessage, parse_llm_response

logger = logging.getLogger(__name__)


class LocalLLMProvider(BaseLLMProvider):
    """로컬 LLM 제공자 (EXAONE via Ollama)"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.client = ollama.Client(host=self.base_url)
        self.korean_optimized = config.get("korean_optimized", True)
        
        # 모델 존재 확인 및 다운로드
        self._ensure_model_available()
    
    def _ensure_model_available(self):
        """모델이 사용 가능한지 확인하고, 없으면 다운로드"""
        try:
            # 사용 가능한 모델 목록 확인
            models = self.client.list()
            model_names = [model['name'] for model in models['models']]
            
            if self.model_name not in model_names:
                logger.info(f"Downloading model: {self.model_name}")
                self.client.pull(self.model_name)
                logger.info(f"Model {self.model_name} downloaded successfully")
            else:
                logger.info(f"Model {self.model_name} is already available")
                
        except Exception as e:
            logger.warning(f"Could not verify model availability: {e}")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """단일 프롬프트로 텍스트 생성"""
        
        try:
            # EXAONE 모델의 특별한 포맷 적용
            if "exaone" in self.model_name.lower():
                prompt = self._format_exaone_prompt(prompt, system_prompt)
            
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': kwargs.get('temperature', self.temperature),
                    'num_predict': kwargs.get('max_tokens', self.max_tokens),
                    'top_p': kwargs.get('top_p', 0.95),
                    'repeat_penalty': kwargs.get('repeat_penalty', 1.0),  # EXAONE에서 중요
                }
            )
            
            return LLMResponse(
                content=response['response'],
                model=self.model_name,
                metadata={
                    'provider': 'local',
                    'done': response.get('done', True),
                    'total_duration': response.get('total_duration'),
                    'load_duration': response.get('load_duration'),
                    'prompt_eval_count': response.get('prompt_eval_count'),
                    'eval_count': response.get('eval_count'),
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise
    
    def chat(
        self,
        messages: List[ChatMessage],
        **kwargs
    ) -> LLMResponse:
        """채팅 형태로 텍스트 생성"""
        
        try:
            # ChatMessage를 Ollama 형태로 변환
            ollama_messages = []
            for msg in messages:
                ollama_messages.append({
                    'role': msg.role,
                    'content': msg.content
                })
            
            response = self.client.chat(
                model=self.model_name,
                messages=ollama_messages,
                options={
                    'temperature': kwargs.get('temperature', self.temperature),
                    'num_predict': kwargs.get('max_tokens', self.max_tokens),
                    'top_p': kwargs.get('top_p', 0.95),
                    'repeat_penalty': kwargs.get('repeat_penalty', 1.0),
                }
            )
            
            return LLMResponse(
                content=response['message']['content'],
                model=self.model_name,
                metadata={
                    'provider': 'local',
                    'done': response.get('done', True),
                    'total_duration': response.get('total_duration'),
                    'load_duration': response.get('load_duration'),
                    'prompt_eval_count': response.get('prompt_eval_count'),
                    'eval_count': response.get('eval_count'),
                }
            )
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            raise
    
    def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Iterator[str]:
        """스트리밍으로 텍스트 생성"""
        
        try:
            if "exaone" in self.model_name.lower():
                prompt = self._format_exaone_prompt(prompt, system_prompt)
            
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                stream=True,
                options={
                    'temperature': kwargs.get('temperature', self.temperature),
                    'num_predict': kwargs.get('max_tokens', self.max_tokens),
                    'top_p': kwargs.get('top_p', 0.95),
                    'repeat_penalty': kwargs.get('repeat_penalty', 1.0),
                }
            )
            
            for chunk in response:
                if 'response' in chunk:
                    yield chunk['response']
                    
        except Exception as e:
            logger.error(f"Error in streaming generation: {e}")
            raise
    
    def stream_chat(
        self,
        messages: List[ChatMessage],
        **kwargs
    ) -> Iterator[str]:
        """스트리밍으로 채팅"""
        
        try:
            ollama_messages = []
            for msg in messages:
                ollama_messages.append({
                    'role': msg.role,
                    'content': msg.content
                })
            
            response = self.client.chat(
                model=self.model_name,
                messages=ollama_messages,
                stream=True,
                options={
                    'temperature': kwargs.get('temperature', self.temperature),
                    'num_predict': kwargs.get('max_tokens', self.max_tokens),
                    'top_p': kwargs.get('top_p', 0.95),
                    'repeat_penalty': kwargs.get('repeat_penalty', 1.0),
                }
            )
            
            for chunk in response:
                if 'message' in chunk and 'content' in chunk['message']:
                    yield chunk['message']['content']
                    
        except Exception as e:
            logger.error(f"Error in streaming chat: {e}")
            raise
    
    def get_embedding(self, text: str) -> List[float]:
        """텍스트 임베딩 생성 (외부 모델 사용)"""
        # Ollama는 현재 임베딩을 직접 지원하지 않으므로 
        # SentenceTransformer 등을 사용
        raise NotImplementedError("Use LocalEmbeddingProvider for embeddings")
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """여러 텍스트의 임베딩 생성"""
        raise NotImplementedError("Use LocalEmbeddingProvider for embeddings")
    
    def _format_exaone_prompt(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """EXAONE 모델용 프롬프트 포맷팅"""
        
        # EXAONE Deep 모델은 <thought> 태그를 사용하여 추론 과정을 표시
        if "deep" in self.model_name.lower():
            if system_prompt:
                formatted = f"{system_prompt}\n\n<thought>\n이 질문에 대해 단계별로 생각해보겠습니다.\n</thought>\n\n{prompt}"
            else:
                formatted = f"<thought>\n{prompt}에 대해 단계별로 생각해보겠습니다.\n</thought>\n\n{prompt}"
        else:
            # 일반 EXAONE 모델
            if system_prompt:
                formatted = f"{system_prompt}\n\n{prompt}"
            else:
                formatted = prompt
        
        return formatted
    
    def validate_config(self) -> bool:
        """설정 유효성 검사"""
        try:
            # Ollama 서버 연결 확인
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_available_models(self) -> List[str]:
        """사용 가능한 모델 목록 반환"""
        try:
            models = self.client.list()
            return [model['name'] for model in models['models']]
        except:
            return []


class LocalEmbeddingProvider(EmbeddingProvider):
    """로컬 임베딩 제공자 (SentenceTransformer 기반)"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 한국어 특화 모델 선택
        if config.get("korean_optimized", True):
            # 한국어-영어 다국어 모델
            self.model_name = config.get(
                "embedding_model",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
        else:
            self.model_name = config.get(
                "embedding_model",
                "sentence-transformers/all-MiniLM-L6-v2"
            )
        
        try:
            self.model = SentenceTransformer(self.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Loaded embedding model: {self.model_name} (dim: {self.dimension})")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def embed_text(self, text: str) -> List[float]:
        """단일 텍스트 임베딩"""
        try:
            embedding = self.model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            raise
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """여러 텍스트 임베딩"""
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            logger.error(f"Error embedding texts: {e}")
            raise
    
    def get_dimension(self) -> int:
        """임베딩 차원 반환"""
        return self.dimension


def create_local_provider(config: Dict[str, Any]) -> LocalLLMProvider:
    """로컬 LLM 제공자 생성 헬퍼 함수"""
    return LocalLLMProvider(config)


def create_local_embedding_provider(config: Dict[str, Any]) -> LocalEmbeddingProvider:
    """로컬 임베딩 제공자 생성 헬퍼 함수"""
    return LocalEmbeddingProvider(config)


def setup_exaone_model(model_name: str = "exaone-deep:32b", base_url: str = "http://localhost:11434"):
    """EXAONE 모델 설정 및 다운로드"""
    
    try:
        client = ollama.Client(host=base_url)
        
        # 서버 연결 확인
        try:
            client.list()
            logger.info("Ollama server is running")
        except:
            logger.error("Ollama server is not running. Please start Ollama first.")
            return False
        
        # 모델 존재 확인
        models = client.list()
        model_names = [model['name'] for model in models['models']]
        
        if model_name not in model_names:
            logger.info(f"Downloading EXAONE model: {model_name}")
            logger.info("This may take a while for the first time...")
            
            # 모델 다운로드
            client.pull(model_name)
            logger.info(f"Successfully downloaded {model_name}")
        else:
            logger.info(f"EXAONE model {model_name} is already available")
        
        # 테스트 생성
        logger.info("Testing model generation...")
        response = client.generate(
            model=model_name,
            prompt="안녕하세요! 간단한 인사말을 해주세요.",
            options={'num_predict': 50}
        )
        
        logger.info(f"Test response: {response['response'][:100]}...")
        logger.info("EXAONE model setup completed successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"Error setting up EXAONE model: {e}")
        return False


def get_exaone_usage_guide() -> str:
    """EXAONE 모델 사용 가이드"""
    
    guide = """
    🚀 EXAONE 4.0 사용 가이드
    
    1. 모델 종류:
       - exaone-deep:2.4b  (작은 모델, 빠른 추론)
       - exaone-deep:7.8b  (중간 모델, 균형잡힌 성능)
       - exaone-deep:32b   (큰 모델, 최고 성능)
    
    2. 한국어 최적화:
       - 한국어 질문에 한국어로 답변
       - 한국 문화와 맥락을 이해
       - 한국어 문서 처리에 최적화
    
    3. 추론 모드:
       - EXAONE Deep 모델은 <thought> 태그를 사용
       - 단계별 추론 과정을 보여줌
       - 수학 문제나 복잡한 질문에 유용
    
    4. 성능 팁:
       - repeat_penalty를 1.0으로 설정 권장
       - temperature는 0.1-0.6 사이 권장
       - 한국어 질문 시 더 나은 성능
    
    5. 메모리 요구사항:
       - 2.4B: 4GB RAM
       - 7.8B: 8GB RAM  
       - 32B: 32GB RAM
    """
    
    return guide


if __name__ == "__main__":
    # 테스트 코드
    from config.api_config import LocalModelConfig
    
    config = LocalModelConfig.from_env()
    
    # EXAONE 모델 설정
    setup_success = setup_exaone_model(config.model, config.base_url)
    
    if setup_success:
        # 로컬 제공자 테스트
        provider = LocalLLMProvider(config.__dict__)
        
        # 간단한 테스트
        response = provider.generate("안녕하세요! 자기소개를 해주세요.")
        print(f"Response: {response.content}")
        
        # 임베딩 제공자 테스트  
        embedding_provider = LocalEmbeddingProvider(config.__dict__)
        
        # 임베딩 테스트
        embedding = embedding_provider.embed_text("안녕하세요")
        print(f"Embedding dimension: {len(embedding)}")
        
        print(get_exaone_usage_guide())
    else:
        print("Failed to setup EXAONE model. Please check Ollama installation.")