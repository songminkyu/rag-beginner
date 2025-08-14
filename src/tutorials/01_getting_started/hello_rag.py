"""
Hello RAG - 첫 번째 RAG 시스템 구현
간단한 RAG 시스템을 구현하여 기본 개념을 이해합니다.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import math
from src.config.api_config import config_manager
# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 환경 변수 로드
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("python-dotenv를 설치해주세요: pip install python-dotenv")

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleRAG:
    """간단한 RAG 시스템 구현"""
    
    def __init__(
        self,
        llm_provider_type: str = "openai",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """SimpleRAG 초기화"""
        
        self.llm_provider_type = llm_provider_type
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents = []  # 간단한 인메모리 저장
        
        # 컴포넌트 초기화
        self._initialize_components()
    
    def _initialize_components(self):
        """RAG 시스템 컴포넌트 초기화"""
        
        # LLM 제공자 초기화
        self.llm_provider = self._create_llm_provider()
        
        logger.info("RAG 시스템 컴포넌트 초기화 완료")
    
    def _create_llm_provider(self):
        """LLM 제공자 생성"""
        
        if self.llm_provider_type == "openai":
            try:
                from src.core.llm_providers.openai_provider import OpenAIProvider, OpenAIEmbeddingProvider
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
                
                config = {
                    "api_key": api_key,
                    "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    "temperature": 0.1,
                    "max_tokens": 1000,
                    "embedding_model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
                }
                
                # 임베딩 제공자도 생성
                self.embedding_provider = OpenAIEmbeddingProvider(config)
                
                return OpenAIProvider(config)
            except ImportError:
                logger.error("OpenAI 제공자를 로드할 수 없습니다.")
                raise
        
        elif self.llm_provider_type == "claude":
            try:
                from src.core.llm_providers.claude_provider import ClaudeProvider
                
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    raise ValueError("ANTHROPIC_API_KEY 환경변수가 설정되지 않았습니다.")
                
                config = {
                    "api_key": api_key,
                    "model": os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022"),
                    "temperature": 0.1,
                    "max_tokens": 1000
                }
                
                # Claude는 임베딩을 지원하지 않으므로 fallback 임베딩 사용
                try:
                    from src.core.llm_providers.flexible_local_provider import LocalEmbeddingProvider
                    embedding_config = {
                        "embedding_model": os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
                        "korean_optimized": os.getenv("KOREAN_OPTIMIZED", "true").lower() == "true"
                    }
                    self.embedding_provider = LocalEmbeddingProvider(embedding_config)
                except Exception as emb_error:
                    logger.warning(f"로컬 임베딩 제공자 로드 실패: {emb_error}")
                    logger.info("기본 임베딩 제공자를 사용합니다")
                    
                    # 간단한 임베딩 제공자 fallback
                    class SimpleEmbeddingProvider:
                        def embed_text(self, text):
                            return [0.0] * 384  # 기본 차원
                        
                        def embed_texts(self, texts):
                            return [[0.0] * 384 for _ in texts]
                    
                    self.embedding_provider = SimpleEmbeddingProvider()
                
                return ClaudeProvider(config)
            except ImportError:
                logger.error("Claude 제공자를 로드할 수 없습니다.")
                raise
        
        elif self.llm_provider_type == "local":
            try:
                # Try to import local providers with dependency check
                import torch
                logger.info("PyTorch found, attempting to load local providers...")
                import pyarrow
                
                # Check for sentence-transformers compatibility
                try:
                    from src.core.llm_providers.flexible_local_provider import FlexibleLocalProvider, LocalEmbeddingProvider
                    
                    config = {
                        "model": os.getenv("LOCAL_MODEL_NAME", "LGAI-EXAONE/EXAONE-4.0-1.2B"),
                        "model_type": os.getenv("LOCAL_MODEL_TYPE", "auto"),
                        "model_size": os.getenv("LOCAL_MODEL_SIZE", "medium"),
                        "hf_model": os.getenv("HF_MODEL_NAME", "microsoft/DialoGPT-medium"),
                        "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                        "device": os.getenv("DEVICE", "auto"),
                        "torch_dtype": os.getenv("TORCH_DTYPE", "float16"),
                        "max_tokens": int(os.getenv("MAX_TOKENS", "4096")),
                        "max_new_tokens": int(os.getenv("MAX_NEW_TOKENS", "1024")),
                        "korean_optimized": os.getenv("KOREAN_OPTIMIZED", "true").lower() == "true",
                        "low_cpu_mem_usage": os.getenv("LOW_CPU_MEM_USAGE", "true").lower() == "true",
                        "use_cache": os.getenv("USE_CACHE", "true").lower() == "true"
                    }
                    
                    # 로컬 임베딩 제공자 생성
                    embedding_config = {
                        "embedding_model": os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
                        "korean_optimized": os.getenv("KOREAN_OPTIMIZED", "true").lower() == "true"
                    }
                    self.embedding_provider = LocalEmbeddingProvider(embedding_config)
                    
                    return FlexibleLocalProvider(config)
                    
                except Exception as st_error:
                    logger.warning(f"SentenceTransformers 호환성 문제: {st_error}")
                    logger.info("기본 로컬 제공자를 사용합니다 (임베딩 없음)")
                    
                    # 완전히 독립적인 fallback 클래스들 - 임포트 없이 구현
                    from dataclasses import dataclass
                    from typing import Dict, Any, Optional
                    
                    @dataclass
                    class LLMResponse:
                        content: str
                        model: str
                        metadata: Dict[str, Any]
                    
                    class SimpleLocalProvider:
                        def __init__(self, config):
                            self.config = config
                            self.temperature = config.get("temperature", 0.1)
                            self.max_tokens = config.get("max_tokens", 1000)
                            
                        def generate(self, prompt, system_prompt=None, **kwargs):
                            return LLMResponse(
                                content="로컬 모델이 사용할 수 없습니다. OpenAI 또는 Claude API를 사용해주세요.",
                                model="local_fallback",
                                metadata={"provider": "fallback"}
                            )
                    
                    # 간단한 임베딩 제공자 fallback
                    class SimpleEmbeddingProvider:
                        def embed_text(self, text):
                            return [0.0] * 384  # 기본 차원
                        
                        def embed_texts(self, texts):
                            return [[0.0] * 384 for _ in texts]
                    
                    config = {
                        "korean_optimized": os.getenv("KOREAN_OPTIMIZED", "true").lower() == "true",
                        "max_tokens": int(os.getenv("MAX_TOKENS", "4096")),
                        "max_new_tokens": int(os.getenv("MAX_NEW_TOKENS", "1024"))
                    }
                    self.embedding_provider = SimpleEmbeddingProvider()
                    
                    return SimpleLocalProvider(config)
                    
            except ImportError as torch_error:
                logger.error(f"PyTorch를 찾을 수 없습니다: {torch_error}")
                logger.error("로컬 제공자를 사용하려면 PyTorch를 설치해주세요: pip install torch")
                raise ValueError("로컬 제공자 사용을 위해 PyTorch가 필요합니다.")
            except Exception as e:
                logger.error(f"로컬 제공자 초기화 실패: {e}")
                raise ValueError(f"로컬 제공자를 로드할 수 없습니다: {e}")
        
        else:
            raise ValueError(f"지원하지 않는 LLM 제공자: {self.llm_provider_type}")
    
    def add_documents(self, documents: List[str], metadata_list: Optional[List[Dict]] = None):
        """문서들을 시스템에 추가"""
        
        logger.info(f"{len(documents)}개 문서를 처리 중...")
        
        for i, doc in enumerate(documents):
            # 문서를 청크로 분할
            chunks = self._simple_text_splitter(doc)
            
            # 메타데이터 준비
            doc_metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else {}
            
            for j, chunk in enumerate(chunks):
                chunk_data = {
                    "content": chunk,
                    "metadata": {
                        "document_id": i,
                        "chunk_id": j,
                        "source": doc_metadata.get("source", f"document_{i}"),
                        **doc_metadata
                    }
                }
                self.documents.append(chunk_data)
        
        logger.info(f"총 {len(self.documents)}개 청크가 추가됨")
    
    def _simple_text_splitter(self, text: str) -> List[str]:
        """간단한 텍스트 분할기"""
        
        # 간단한 분할: 텍스트가 청크 크기보다 작으면 그대로 반환
        if len(text) <= self.chunk_size:
            return [text]
        
        # 텍스트를 청크로 분할
        chunks = []
        start = 0
        
        while start < len(text):
            # 청크 끝 위치 계산
            end = start + self.chunk_size
            
            # 마지막 청크가 아니라면 적절한 분할점 찾기
            if end < len(text):
                # 공백이나 문장 부호에서 분할하려고 시도
                for i in range(end, start + self.chunk_size - self.chunk_overlap, -1):
                    if text[i] in ' \n\t.!?':
                        end = i + 1
                        break
            
            # 청크 추출
            chunk = text[start:end].strip()
            if chunk:  # 빈 청크는 추가하지 않음
                chunks.append(chunk)
            
            # 다음 시작점 계산 (오버랩 적용)
            start = end - self.chunk_overlap
            
            # 무한 루프 방지
            if start >= len(text):
                break
        
        return chunks
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """쿼리와 관련된 문서들을 검색"""
        
        if not self.documents:
            return []
        
        # 유사도 계산 (간단한 키워드 기반 또는 임베딩 기반)
        scored_docs = []
        for doc in self.documents:
            score = self._simple_similarity(query, doc["content"])
            scored_docs.append({
                "content": doc["content"],
                "metadata": doc["metadata"],
                "similarity_score": score
            })
        
        # 유사도 순으로 정렬
        scored_docs.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        # 상위 k개 반환
        top_docs = scored_docs[:k]
        
        logger.info(f"검색 완료: {len(top_docs)}개 문서 검색됨")
        return top_docs
    
    def _simple_similarity(self, query: str, text: str) -> float:
        """간단한 유사도 계산 (키워드 기반)"""
        
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        
        if not query_words or not text_words:
            return 0.0
        
        intersection = query_words.intersection(text_words)
        union = query_words.union(text_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """코사인 유사도 계산"""
        import math
        
        dot_product = sum(x * y for x, y in zip(a, b))
        magnitude_a = math.sqrt(sum(x * x for x in a))
        magnitude_b = math.sqrt(sum(x * x for x in b))
        
        if magnitude_a == 0 or magnitude_b == 0:
            return 0
        
        return dot_product / (magnitude_a * magnitude_b)
    
    def generate_answer(self, query: str, context: str) -> str:
        """컨텍스트를 바탕으로 답변 생성"""
        
        # RAG 프롬프트 생성
        if hasattr(self.llm_provider, 'format_rag_prompt'):
            prompt = self.llm_provider.format_rag_prompt(query=query, context=context)
        else:
            # 기본 프롬프트 템플릿
            prompt = f"""다음 컨텍스트 정보를 바탕으로 질문에 답변해주세요.
                    컨텍스트:
                    {context}
                    
                    질문: {query}
                    
                    답변:"""
        
        # 답변 생성
        response = self.llm_provider.generate(prompt)
        return response.content
    
    def query(self, question: str, k: int = 5) -> Dict[str, Any]:
        """RAG 질의응답 실행"""
        
        logger.info(f"질문: {question}")
        
        # 1. 관련 문서 검색
        retrieved_docs = self.retrieve(question, k=k)
        
        if not retrieved_docs:
            return {
                "question": question,
                "answer": "관련된 정보를 찾을 수 없습니다.",
                "sources": [],
                "context": ""
            }
        
        # 2. 컨텍스트 구성
        context_parts = []
        sources = []
        
        for doc in retrieved_docs:
            if doc["similarity_score"] > 0:  # 유사도가 0보다 큰 것만
                context_parts.append(doc["content"])
                
                # 소스 정보 추가
                source_info = {
                    "content": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"],
                    "metadata": doc["metadata"],
                    "similarity_score": doc["similarity_score"]
                }
                sources.append(source_info)
        
        if not context_parts:
            return {
                "question": question,
                "answer": "관련된 정보를 찾을 수 없습니다.",
                "sources": [],
                "context": ""
            }
        
        context = "\n\n".join(context_parts)
        
        # 3. 답변 생성
        try:
            answer = self.generate_answer(question, context)
        except Exception as e:
            logger.error(f"답변 생성 오류: {e}")
            answer = f"답변 생성 중 오류가 발생했습니다: {e}"
        
        result = {
            "question": question,
            "answer": answer,
            "sources": sources,
            "context": context,
            "retrieved_count": len(sources)
        }
        
        logger.info(f"답변 생성 완료")
        return result


def demo_korean_rag():
    """한국어 RAG 데모"""
    print("🚀 한국어 RAG 시스템 데모")
    print("=" * 50)
    
    # 한국어 문서 샘플
    korean_documents = [
        "인공지능(AI)은 인간의 학습능력, 추론능력, 지각능력, 자연언어의 이해능력 등을 컴퓨터 프로그램으로 실현한 기술이다.",
        "머신러닝은 인공지능의 한 분야로, 컴퓨터가 학습할 수 있도록 하는 알고리즘과 기법을 개발하는 분야이다.",
        "딥러닝은 머신러닝의 한 분야로, 인공신경망을 기반으로 하여 컴퓨터가 사람처럼 학습할 수 있도록 하는 기술이다.",
        "자연어처리(NLP)는 인간의 언어를 컴퓨터가 이해하고 처리할 수 있도록 하는 인공지능 기술이다.",
        "컴퓨터 비전은 컴퓨터가 디지털 이미지나 비디오를 이해하고 분석할 수 있도록 하는 인공지능 분야이다.",
        "로봇공학은 로봇의 설계, 제작, 작동에 관한 기술을 다루는 공학 분야이다.",
        "빅데이터는 기존 데이터베이스 관리도구로 데이터를 수집, 저장, 관리, 분석할 수 있는 역량을 넘어서는 대량의 정형 또는 비정형 데이터를 말한다."
    ]
    
    # SimpleRAG 초기화 (유연한 로컬 모델 사용)
    try:
        rag = SimpleRAG("local")
        model_info = config_manager.get_local_model_info()
        print(f"✅ 로컬 모델을 사용합니다: {model_info.get('model', 'unknown')}")
    except Exception as e:
        print(f"로컬 모델 초기화 실패: {e}")
        print("OpenAI API를 사용합니다...")
        rag = SimpleRAG("openai")
    
    # 문서 추가
    rag.add_documents(korean_documents)
    
    # 질문들
    questions = [
        "인공지능이 무엇인가요?",
        "머신러닝과 딥러닝의 차이점은?",
        "자연어처리 기술에 대해 설명해주세요",
        "빅데이터란 무엇인가요?"
    ]
    
    # 각 질문에 대해 RAG 실행
    for question in questions:
        print(f"\n❓ 질문: {question}")
        print("-" * 30)
        
        result = rag.query(question)
        
        print(f"💡 답변: {result['answer']}")
        similarities = [source['similarity_score'] for source in result['sources']]
        print(f"📊 유사도: {[f'{sim:.3f}' for sim in similarities]}")
        print(f"📚 참고 문서: {len(result['sources'])}개")


def demo_english_rag():
    """영어 RAG 데모"""
    print("\n\n🚀 English RAG System Demo")
    print("=" * 50)
    
    # 영어 문서 샘플
    english_documents = [
        "Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.",
        "Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data.",
        "Natural Language Processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret and manipulate human language.",
        "Computer vision is a field of artificial intelligence that trains computers to interpret and understand the visual world.",
        "Reinforcement learning is an area of machine learning where an agent learns to behave in an environment by performing actions and receiving rewards or penalties.",
        "Big data refers to extremely large datasets that may be analyzed computationally to reveal patterns, trends, and associations."
    ]
    
    # SimpleRAG 초기화
    try:
        rag = SimpleRAG("local")
        model_info = config_manager.get_local_model_info()
        print(f"✅ 로컬 모델을 사용합니다: {model_info.get('model', 'unknown')}")
    except Exception as e:
        print(f"로컬 모델 실패: {e}")
        rag = SimpleRAG("openai")
        print("✅ OpenAI GPT 모델을 사용합니다.")
    
    # 문서 추가
    rag.add_documents(english_documents)
    
    # 영어 질문들
    questions = [
        "What is machine learning?",
        "How does deep learning differ from traditional machine learning?",
        "What is natural language processing used for?",
        "Explain computer vision"
    ]
    
    # 각 질문에 대해 RAG 실행
    for question in questions:
        print(f"\n❓ Question: {question}")
        print("-" * 30)
        
        result = rag.query(question)
        
        print(f"💡 Answer: {result['answer']}")
        similarities = [source['similarity_score'] for source in result['sources']]
        print(f"📊 Similarities: {[f'{sim:.3f}' for sim in similarities]}")
        print(f"📚 Sources: {len(result['sources'])} documents")


def interactive_rag():
    """대화형 RAG 시스템"""
    print("\n\n💬 대화형 RAG 시스템")
    print("=" * 50)
    print("종료하려면 'quit' 또는 'exit'를 입력하세요")
    
    # 기본 문서 로드
    documents = [
        "Python은 1991년 귀도 반 로섬이 개발한 고급 프로그래밍 언어입니다.",
        "Django는 Python으로 작성된 웹 프레임워크로, 빠른 개발과 깔끔한 디자인을 지원합니다.",
        "FastAPI는 현대적이고 빠른 웹 API를 구축하기 위한 Python 프레임워크입니다.",
        "NumPy는 Python에서 과학 계산을 위한 기본 패키지입니다.",
        "Pandas는 데이터 조작 및 분석을 위한 Python 라이브러리입니다.",
        "TensorFlow는 Google에서 개발한 오픈소스 머신러닝 프레임워크입니다.",
        "PyTorch는 Facebook에서 개발한 딥러닝 프레임워크입니다."
    ]
    
    # RAG 시스템 초기화
    available_providers = config_manager.get_available_providers()
    print(f"사용 가능한 제공자: {available_providers}")
    
    rag = None
    for provider in ["local", "openai", "claude"]:
        if provider in available_providers:
            try:
                rag = SimpleRAG(provider)
                if provider == "local":
                    model_info = config_manager.get_local_model_info()
                    print(f"✅ 로컬 모델을 사용합니다: {model_info.get('model', 'unknown')}")
                elif provider == "openai":
                    print("✅ OpenAI GPT 모델을 사용합니다.")
                else:
                    print("✅ Claude 모델을 사용합니다.")
                break
            except Exception as e:
                print(f"{provider} 제공자 실패: {e}")
                continue
    
    if not rag:
        print("❌ 사용 가능한 제공자가 없습니다.")
        return
    
    rag.add_documents(documents)
    print(f"📚 {len(documents)}개의 문서가 로드되었습니다.")
    
    while True:
        try:
            question = input("\n🤔 질문을 입력하세요: ").strip()
            
            if question.lower() in ['quit', 'exit', '종료', '끝']:
                print("👋 RAG 시스템을 종료합니다.")
                break
            
            if not question:
                continue
            
            print("🔍 검색 중...")
            result = rag.query(question)
            
            print(f"\n💡 답변: {result['answer']}")
            print(f"📊 검색된 문서 개수: {len(result['sources'])}")
            
            # 상세 정보 표시 여부 묻기
            show_details = input("\n상세 정보를 보시겠습니까? (y/n): ").lower()
            if show_details in ['y', 'yes', 'ㅇ']:
                print("\n📚 참고 문서:")
                for i, source in enumerate(result['sources'], 1):
                    print(f"  {i}. {source}")
                similarities = [source['similarity_score'] for source in result['sources']]
                print(f"\n📊 유사도 점수: {[f'{sim:.3f}' for sim in similarities]}")
        
        except KeyboardInterrupt:
            print("\n👋 RAG 시스템을 종료합니다.")
            break
        except Exception as e:
            print(f"❌ 오류가 발생했습니다: {e}")


def main():
    """메인 함수"""
    print("🎯 Hello RAG - 첫 번째 RAG 예제")
    print("=" * 50)
    
    # 사용 가능한 제공자 확인
    available_providers = config_manager.get_available_providers()
    print(f"✅ 사용 가능한 LLM 제공자: {available_providers}")
    
    if not available_providers:
        print("❌ 사용 가능한 LLM 제공자가 없습니다.")
        print("환경변수를 설정하거나 Ollama를 설치해주세요.")
        return
    
    try:
        # 데모 실행
        demo_korean_rag()
        # demo_english_rag()
        
        # 대화형 RAG
        # interactive_rag()
        
    except Exception as e:
        logger.error(f"Demo 실행 중 오류: {e}")
        print("\n❌ 데모 실행에 실패했습니다.")
        print("설정을 확인하고 다시 시도해주세요.")


if __name__ == "__main__":
    main()