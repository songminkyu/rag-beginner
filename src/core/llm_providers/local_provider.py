"""
Local Model Provider (EXAONE via Hugging Face Transformers)
EXAONE 4.0 모델을 Hugging Face Transformers를 통해 사용하는 제공자
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, List, Optional, Iterator
from sentence_transformers import SentenceTransformer
import logging
import gc
import ollama
from .base_provider import BaseLLMProvider, EmbeddingProvider, LLMResponse, ChatMessage, parse_llm_response

logger = logging.getLogger(__name__)


class LocalProvider(BaseLLMProvider):
    """로컬 LLM 제공자 (EXAONE via Hugging Face Transformers)"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get("model", "LGAI-EXAONE/EXAONE-4.0-1.2B")
        self.device = config.get("device", "auto")
        self.torch_dtype = config.get("torch_dtype", "bfloat16")
        self.korean_optimized = config.get("korean_optimized", True)
        
        # GPU 메모리 최적화 설정
        self.low_cpu_mem_usage = config.get("low_cpu_mem_usage", True)
        self.use_cache = config.get("use_cache", True)
        
        self.model = None
        self.tokenizer = None
        
        # 모델 로드
        self._load_model()
    
    def _load_model(self):
        """EXAONE 모델 로드"""
        try:
            logger.info(f"Loading EXAONE model: {self.model_name}")
            
            # 특별한 transformers 버전 필요 (EXAONE 4.0 지원)
            try:
                # EXAONE 4.0 전용 transformers 설치 안내
                import transformers
                logger.info(f"Transformers version: {transformers.__version__}")
            except ImportError:
                logger.error("Please install transformers with EXAONE support:")
                logger.error("pip install git+https://github.com/lgai-exaone/transformers@add-exaone4")
                raise
            
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logger.info("Tokenizer loaded successfully")
            
            # 모델 로드 (메모리 최적화)
            torch_dtype = getattr(torch, self.torch_dtype) if isinstance(self.torch_dtype, str) else self.torch_dtype
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map=self.device,
                low_cpu_mem_usage=self.low_cpu_mem_usage,
                trust_remote_code=True
            )
            
            logger.info(f"Model loaded successfully on device: {self.model.device}")
            logger.info(f"Model dtype: {self.model.dtype}")
            
            # 메모리 사용량 출력
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3
                logger.info(f"GPU memory used: {memory_used:.2f} GB")
                
        except Exception as e:
            logger.error(f"Failed to load EXAONE model: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """단일 프롬프트로 텍스트 생성"""
        
        try:
            # 채팅 메시지 형태로 변환
            messages = self._prepare_messages(prompt, system_prompt)
            
            # 토크나이징
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )
            
            # 생성 파라미터 설정
            generation_kwargs = {
                "max_new_tokens": kwargs.get("max_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
                "do_sample": kwargs.get("temperature", self.temperature) > 0,
                "top_p": kwargs.get("top_p", 0.95),
                "pad_token_id": self.tokenizer.eos_token_id,
                "use_cache": self.use_cache,
            }
            
            # EXAONE 모델 특화 설정
            if "exaone" in self.model_name.lower():
                generation_kwargs.update({
                    "repetition_penalty": kwargs.get("repetition_penalty", 1.0),  # EXAONE에서 중요
                    "length_penalty": kwargs.get("length_penalty", 1.0),
                })
            
            # 텍스트 생성
            with torch.no_grad():
                output = self.model.generate(
                    input_ids.to(self.model.device),
                    **generation_kwargs
                )
            
            # 입력 토큰 제거하고 출력만 디코딩
            generated_tokens = output[0][len(input_ids[0]):]
            response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return LLMResponse(
                content=response_text.strip(),
                model=self.model_name,
                metadata={
                    "provider": "local_transformers",
                    "input_length": len(input_ids[0]),
                    "output_length": len(generated_tokens),
                    "total_tokens": len(output[0]),
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
            # ChatMessage를 Hugging Face 형태로 변환
            hf_messages = []
            for msg in messages:
                hf_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
            
            # 토크나이징
            input_ids = self.tokenizer.apply_chat_template(
                hf_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )
            
            # 생성 파라미터 설정
            generation_kwargs = {
                "max_new_tokens": kwargs.get("max_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
                "do_sample": kwargs.get("temperature", self.temperature) > 0,
                "top_p": kwargs.get("top_p", 0.95),
                "pad_token_id": self.tokenizer.eos_token_id,
                "use_cache": self.use_cache,
            }
            
            # 텍스트 생성
            with torch.no_grad():
                output = self.model.generate(
                    input_ids.to(self.model.device),
                    **generation_kwargs
                )
            
            # 응답 디코딩
            generated_tokens = output[0][len(input_ids[0]):]
            response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return LLMResponse(
                content=response_text.strip(),
                model=self.model_name,
                metadata={
                    "provider": "local_transformers",
                    "input_length": len(input_ids[0]),
                    "output_length": len(generated_tokens),
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
        
        # Hugging Face Transformers의 스트리밍은 복잡하므로 기본 생성 후 청크로 반환
        try:
            response = self.generate(prompt, system_prompt, **kwargs)
            
            # 단어별로 스트리밍 시뮬레이션
            words = response.content.split()
            for i, word in enumerate(words):
                if i == 0:
                    yield word
                else:
                    yield " " + word
                    
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
            response = self.chat(messages, **kwargs)
            
            # 단어별로 스트리밍 시뮬레이션
            words = response.content.split()
            for i, word in enumerate(words):
                if i == 0:
                    yield word
                else:
                    yield " " + word
                    
        except Exception as e:
            logger.error(f"Error in streaming chat: {e}")
            raise
    
    def get_embedding(self, text: str) -> List[float]:
        """텍스트 임베딩 생성 (외부 모델 사용)"""
        raise NotImplementedError("Use LocalEmbeddingProvider for embeddings")
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """여러 텍스트의 임베딩 생성"""
        raise NotImplementedError("Use LocalEmbeddingProvider for embeddings")
    
    def _prepare_messages(self, prompt: str, system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        """프롬프트를 메시지 형태로 변환"""
        
        messages = []
        
        # 시스템 프롬프트 추가
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        elif self.korean_optimized:
            # 한국어 기본 시스템 프롬프트
            messages.append({
                "role": "system", 
                "content": "당신은 도움이 되는 AI 어시스턴트입니다. 한국어로 정확하고 유용한 답변을 제공해주세요."
            })
        
        # 사용자 프롬프트 추가
        messages.append({"role": "user", "content": prompt})
        
        return messages
    
    def validate_config(self) -> bool:
        """설정 유효성 검사"""
        try:
            return self.model is not None and self.tokenizer is not None
        except:
            return False
    
    def cleanup(self):
        """메모리 정리"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        logger.info("Model cleanup completed")


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


def create_local_provider(config: Dict[str, Any]) -> LocalProvider:
    """로컬 LLM 제공자 생성 헬퍼 함수"""
    return LocalProvider(config)


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
    from src.config.api_config import LocalModelConfig
    
    config = LocalModelConfig.from_env()
    
    # EXAONE 모델 설정
    setup_success = setup_exaone_model(config.model, config.base_url)
    
    if setup_success:
        # 로컬 제공자 테스트
        provider = LocalProvider(config.__dict__)
        
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