# 🚀 LLM RAG Learning Repository

**2025년 최신 LLM RAG 기술 학습을 위한 종합 저장소**

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-orange.svg)](https://langchain.com)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.12%2B-purple.svg)](https://llamaindex.ai)
[![EXAONE](https://img.shields.io/badge/EXAONE-4.0-red.svg)](https://huggingface.co/LGAI-EXAONE)

## 📖 개요

이 저장소는 **Retrieval-Augmented Generation (RAG)** 기술을 체계적으로 학습할 수 있도록 설계된 종합 학습 플랫폼입니다. 2025년 최신 기술 스택을 기반으로 하며, 특히 **한국어 처리에 최적화된 EXAONE 4.0 모델**을 지원합니다.

### ✨ 주요 특징

- 🔥 **2025년 최신 기술**: LangChain 0.2+, LlamaIndex 0.12+, Gradio 4.0+ 
- 🇰🇷 **한국어 특화**: EXAONE 4.0 모델과 한국어 문서 처리 최적화
- 🛠️ **다양한 LLM 지원**: OpenAI GPT-4, Claude-3.5, 로컬 EXAONE 모델
- 📚 **체계적 학습**: 기초부터 고급까지 6단계 튜토리얼
- 🎯 **실무 중심**: 4가지 실제 프로젝트 예제
- 📊 **포괄적 평가**: RAGAS, BLEU, BERTScore 등 다양한 메트릭

## 🚀 빠른 시작

### 1. 저장소 클론 및 설치

```bash
# 저장소 클론
git clone https://github.com/your-username/llm-rag-learning.git
cd llm-rag-learning

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
pip install -e .
```

### 2. 환경 설정

```bash
# 환경변수 파일 생성
cp .env.example .env

# .env 파일에서 API 키 설정
# OPENAI_API_KEY=your-openai-key
# ANTHROPIC_API_KEY=your-claude-key
```

### 3. EXAONE 로컬 모델 설정 (선택사항)

```bash
# Ollama 및 EXAONE 모델 설정
chmod +x scripts/setup_ollama.sh
./scripts/setup_ollama.sh
```

### 4. 첫 번째 RAG 실행

```bash
# Hello RAG 예제 실행
python tutorials/01_getting_started/hello_rag.py
```

## 📁 저장소 구조

```
llm-rag-learning/
├── 🔧 core/                    # 핵심 기능 모듈
│   ├── llm_providers/          # LLM 제공자 (OpenAI, Claude, EXAONE)
│   ├── data_processing/        # 데이터 처리 및 임베딩
│   ├── retrieval/              # 검색 엔진
│   └── evaluation/             # 평가 시스템
├── 🎓 tutorials/               # 단계별 학습 튜토리얼
│   ├── 01_getting_started/     # 시작하기
│   ├── 02_data_preparation/    # 데이터 준비
│   ├── 03_basic_rag/          # 기본 RAG
│   ├── 04_advanced_rag/       # 고급 RAG
│   ├── 05_local_models/       # 로컬 모델
│   └── 06_production_ready/   # 프로덕션 준비
├── 🏗️ frameworks/             # 프레임워크별 구현
│   ├── langchain_examples/     # LangChain 예제
│   └── llamaindex_examples/    # LlamaIndex 예제
├── 🚀 projects/               # 실습 프로젝트
│   ├── chatbot/               # 챗봇 프로젝트
│   ├── document_qa/           # 문서 Q&A
│   ├── knowledge_base/        # 지식베이스
│   └── multi_agent_rag/       # 멀티 에이전트 RAG
└── 📊 data/                   # 샘플 데이터 및 벡터 스토어
```

## 🎓 학습 로드맵

### 📚 단계별 튜토리얼

| 단계 | 내용 | 예상 시간 | 난이도 |
|------|------|----------|--------|
| **1단계** | [시작하기](tutorials/01_getting_started/) | 30분 | ⭐ |
| **2단계** | [데이터 준비](tutorials/02_data_preparation/) | 1시간 | ⭐⭐ |
| **3단계** | [기본 RAG](tutorials/03_basic_rag/) | 2시간 | ⭐⭐ |
| **4단계** | [고급 RAG](tutorials/04_advanced_rag/) | 3시간 | ⭐⭐⭐ |
| **5단계** | [로컬 모델](tutorials/05_local_models/) | 2시간 | ⭐⭐⭐ |
| **6단계** | [프로덕션](tutorials/06_production_ready/) | 4시간 | ⭐⭐⭐⭐ |

### 🏗️ 실습 프로젝트

1. **💬 RAG 챗봇**: Gradio 기반 대화형 인터페이스
2. **📄 문서 Q&A**: PDF/DOCX 문서 기반 질의응답 시스템
3. **🧠 지식베이스**: 기업 내부 지식 관리 시스템
4. **🤖 멀티 에이전트**: 협업하는 AI 에이전트 시스템

## 🛠️ 지원 기술

### 🤖 LLM 제공자

| 제공자 | 모델 | 특징 |
|--------|------|------|
| **OpenAI** | GPT-4o, GPT-4 Turbo | 높은 성능, 다양한 기능 |
| **Anthropic** | Claude-3.5 Sonnet | 안전성, 긴 컨텍스트 |
| **LG AI** | EXAONE 4.0 (32B/7.8B/2.4B) | 한국어 특화, 로컬 실행 |

### 📊 벡터 스토어

- **ChromaDB**: 로컬 개발용
- **FAISS**: 고성능 검색
- **Pinecone**: 클라우드 벡터 DB

### 🔧 프레임워크

- **LangChain**: 복잡한 워크플로우와 에이전트
- **LlamaIndex**: 효율적인 데이터 인덱싱
- **Gradio**: 현대적인 웹 인터페이스

## 💡 사용 예제

### 기본 RAG 시스템

```python
from core.llm_providers.local_provider import LocalLLMProvider
from tutorials.hello_rag import SimpleRAG

# EXAONE 모델로 RAG 시스템 초기화
rag = SimpleRAG("local")

# 문서 추가
documents = [
    "인공지능은 인간의 지능을 모방하는 기술입니다.",
    "머신러닝은 AI의 한 분야로 데이터로부터 학습합니다."
]
rag.add_documents(documents)

# 질문하기
result = rag.query("인공지능이 무엇인가요?")
print(f"답변: {result['answer']}")
```

### LangChain 기반 고급 RAG

```python
from langchain.chains import RetrievalQA
from frameworks.langchain_examples.advanced_rag import ConversationalRAG

# 대화형 RAG 시스템
conv_rag = ConversationalRAG(
    llm_provider="local",
    vector_store="chroma"
)

# 연속 대화
response1 = conv_rag.chat("RAG가 무엇인가요?")
response2 = conv_rag.chat("그럼 어떤 장점이 있나요?")  # 이전 대화 기억
```

## 🔧 고급 기능

### 🎯 하이브리드 검색
- 의미적 검색 + 키워드 검색
- 재순위 매기기 (Reranking)
- 쿼리 확장 및 개선

### 🧠 메모리 관리
- 대화 히스토리 유지
- 컨텍스트 압축
- 동적 메모리 할당

### 📊 성능 최적화
- 배치 처리
- 캐싱 시스템
- GPU 가속

### 🔍 평가 시스템
```python
from core.evaluation import RAGEvaluator

evaluator = RAGEvaluator()
results = evaluator.evaluate(
    questions=test_questions,
    answers=generated_answers,
    ground_truth=expected_answers,
    metrics=["ragas", "bleu", "bertscore"]
)
```

## 🌟 EXAONE 4.0 특별 기능

### 🇰🇷 한국어 최적화
- 한국어 문서 처리에 특화
- 한국 문화 컨텍스트 이해
- 한글 토크나이저 지원

### 🧠 추론 모드
```python
# EXAONE Deep 모델의 단계별 추론
prompt = "<thought>\n수학 문제를 단계별로 풀어보겠습니다.\n</thought>\n\n2x + 5 = 15를 풀어주세요."
response = exaone_provider.generate(prompt)
```

### ⚡ 성능 최적화
- Hybrid Attention 구조
- QK-Reorder-Norm 기법
- 효율적인 메모리 사용

## 📊 성능 벤치마크

### 모델별 성능 비교
| 모델 | 한국어 성능 | 영어 성능 | 추론 속도 | 메모리 사용 |
|------|-------------|-----------|-----------|-------------|
| EXAONE-32B | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | 32GB |
| EXAONE-7.8B | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | 8GB |
| GPT-4 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | API |
| Claude-3.5 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | API |

## 🤝 기여하기

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📚 참고 자료

### 공식 문서
- [LangChain Documentation](https://python.langchain.com/)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [EXAONE Model Card](https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-32B)
- [Ollama Documentation](https://ollama.ai/)

### 유용한 링크
- [RAG 논문 모음](docs/references.md)
- [한국어 NLP 리소스](docs/korean_nlp.md)
- [성능 튜닝 가이드](docs/performance_tuning.md)
- [배포 가이드](docs/deployment.md)

## ❓ FAQ

<details>
<summary><b>Q: EXAONE 모델을 사용하려면 GPU가 필요한가요?</b></summary>

A: 필수는 아니지만 권장됩니다. CPU에서도 실행 가능하지만, GPU를 사용하면 훨씬 빠른 추론이 가능합니다.
- EXAONE-2.4B: CPU로도 충분
- EXAONE-7.8B: GPU 권장
- EXAONE-32B: GPU 필수
</details>

<details>
<summary><b>Q: 어떤 모델부터 시작하는 것이 좋나요?</b></summary>

A: 학습 목적이라면 다음 순서를 권장합니다:
1. OpenAI GPT-4 (API가 있다면)
2. EXAONE-7.8B (로컬 환경)
3. EXAONE-32B (고성능 필요시)
</details>

<details>
<summary><b>Q: 상업적 용도로 사용 가능한가요?</b></summary>

A: MIT 라이센스로 상업적 사용이 가능합니다. 단, 각 LLM 모델의 라이센스를 별도로 확인해주세요.
</details>

## 📄 라이센스

이 프로젝트는 [MIT License](LICENSE) 하에 배포됩니다.

## 🙏 감사의 말

- [LangChain](https://github.com/langchain-ai/langchain) 팀
- [LlamaIndex](https://github.com/run-llama/llama_index) 팀  
- [LG AI Research](https://www.lgresearch.ai/) EXAONE 팀
- [Ollama](https://github.com/ollama/ollama) 팀

---

<div align="center">

**🌟 Star this repository if you find it helpful! 🌟**

Made with ❤️ for the Korean AI Community

</div>