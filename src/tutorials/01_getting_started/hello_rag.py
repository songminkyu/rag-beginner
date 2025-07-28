"""
Hello RAG - 첫 번째 RAG 예제
5분만에 RAG 시스템 구축하기
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from typing import List, Dict, Any
import logging
from config.api_config import config_manager
from core.llm_providers.local_provider import LocalLLMProvider, LocalEmbeddingProvider
from core.llm_providers.base_provider import ChatMessage

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleRAG:
    """간단한 RAG 시스템"""
    
    def __init__(self, provider_type: str = "local"):
        self.provider_type = provider_type
        self.documents = []
        self.embeddings = []
        
        # LLM 제공자 초기화
        if provider_type == "local":
            config = config_manager.local.__dict__
            self.llm = LocalLLMProvider(config)
            self.embedding_provider = LocalEmbeddingProvider(config)
        elif provider_type == "openai":
            from core.llm_providers.openai_provider import OpenAIProvider
            config = config_manager.openai.__dict__
            self.llm = OpenAIProvider(config)
            self.embedding_provider = OpenAIProvider(config)
        elif provider_type == "claude":
            from core.llm_providers.claude_provider import ClaudeProvider
            config = config_manager.claude.__dict__
            self.llm = ClaudeProvider(config)
            # Claude는 임베딩을 제공하지 않으므로 로컬 임베딩 사용
            self.embedding_provider = LocalEmbeddingProvider(config_manager.local.__dict__)
        else:
            raise ValueError(f"Unsupported provider: {provider_type}")
        
        logger.info(f"Initialized SimpleRAG with {provider_type} provider")
    
    def add_documents(self, documents: List[str]):
        """문서들을 추가하고 임베딩 생성"""
        logger.info(f"Adding {len(documents)} documents...")
        
        self.documents.extend(documents)
        
        # 임베딩 생성
        new_embeddings = self.embedding_provider.embed_texts(documents)
        self.embeddings.extend(new_embeddings)
        
        logger.info(f"Total documents: {len(self.documents)}")
    
    def similarity_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """유사도 검색으로 관련 문서 찾기"""
        if not self.documents:
            return []
        
        # 쿼리 임베딩 생성
        query_embedding = self.embedding_provider.embed_text(query)
        
        # 코사인 유사도 계산
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            similarities.append({
                'index': i,
                'document': self.documents[i],
                'similarity': similarity
            })
        
        # 유사도 기준으로 정렬
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similarities[:top_k]
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """코사인 유사도 계산"""
        import math
        
        dot_product = sum(x * y for x, y in zip(a, b))
        magnitude_a = math.sqrt(sum(x * x for x in a))
        magnitude_b = math.sqrt(sum(x * x for x in b))
        
        if magnitude_a == 0 or magnitude_b == 0:
            return 0
        
        return dot_product / (magnitude_a * magnitude_b)
    
    def query(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """RAG 쿼리 실행"""
        logger.info(f"Processing query: {question}")
        
        # 관련 문서 검색
        relevant_docs = self.similarity_search(question, top_k)
        
        if not relevant_docs:
            return {
                'question': question,
                'answer': '관련 문서를 찾을 수 없습니다.',
                'sources': [],
                'context': ''
            }
        
        # 컨텍스트 구성
        context = "\n\n".join([doc['document'] for doc in relevant_docs])
        
        # RAG 프롬프트 생성
        rag_prompt = self.llm.format_rag_prompt(question, context)
        
        # LLM으로 답변 생성
        response = self.llm.generate(rag_prompt)
        
        return {
            'question': question,
            'answer': response.content,
            'sources': [doc['document'][:100] + '...' for doc in relevant_docs],
            'context': context,
            'similarities': [doc['similarity'] for doc in relevant_docs]
        }


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
    
    # SimpleRAG 초기화 (로컬 EXAONE 모델 사용)
    try:
        rag = SimpleRAG("local")
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
        print(f"📊 유사도: {[f'{sim:.3f}' for sim in result['similarities']]}")
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
    except:
        rag = SimpleRAG("openai")
    
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
        print(f"📊 Similarities: {[f'{sim:.3f}' for sim in result['similarities']]}")
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
    try:
        rag = SimpleRAG("local")
        print("✅ 로컬 EXAONE 모델을 사용합니다.")
    except:
        try:
            rag = SimpleRAG("openai")
            print("✅ OpenAI GPT 모델을 사용합니다.")
        except:
            rag = SimpleRAG("claude")
            print("✅ Claude 모델을 사용합니다.")
    
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
                print(f"\n📊 유사도 점수: {[f'{sim:.3f}' for sim in result['similarities']]}")
        
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
        demo_english_rag()
        
        # 대화형 RAG
        interactive_rag()
        
    except Exception as e:
        logger.error(f"Demo 실행 중 오류: {e}")
        print("\n❌ 데모 실행에 실패했습니다.")
        print("설정을 확인하고 다시 시도해주세요.")


if __name__ == "__main__":
    main()