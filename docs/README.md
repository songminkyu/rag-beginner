# LLM RAG 학습 저장소 구조

```

rag-beginner/
│
├── README.md                           # 저장소 소개 및 전체적인 학습 가이드
├── requirements.txt                    # 필요한 패키지 리스트
├── setup.py                           # 프로젝트 설정
├── .env.example                       # 환경변수 예시 파일
├── .gitignore                         # Git 제외 파일 설정
├── docs/                              # 문서화
│   ├── README.md                      # 프로젝트 경로 가이드
│   ├── getting_started.md             # 시작 가이드
│   ├── api_reference.md               # API 레퍼런스
│   ├── best_practices.md              # 모범 사례
│   ├── troubleshooting.md             # 문제 해결
│   ├── performance_tuning.md          # 성능 튜닝
│   └── contributing.md                # 기여 가이드		     # 문서화              
│
├── docker/                            # Docker 설정
│   ├── Dockerfile                     # 메인 Docker 파일
│   ├── docker-compose.yml             # Docker Compose 설정
│   ├── requirements.docker.txt        # Docker용 requirements
│   └── entrypoint.sh                  # Docker 진입점
│
└──src/
    ├── config/                            # 설정 파일들
    │   ├── __init__.py
    │   ├── api_config.py                  # API 설정 (OpenAI, Claude, 로컬)
    │   ├── model_config.py                # 모델별 설정
    │   └── vector_store_config.py         # 벡터 스토어 설정
    │
    ├── core/                              # 핵심 기능 모듈
    │   ├── __init__.py
    │   ├── llm_providers/                 # LLM 제공자별 구현
    │   │   ├── __init__.py
    │   │   ├── openai_provider.py         # OpenAI API 연동
    │   │   ├── claude_provider.py         # Claude API 연동
    │   │   ├── local_provider.py          # 로컬 모델 (EXAONE) 연동
    │   │   └── base_provider.py           # 공통 인터페이스
    │   │
    │   ├── data_processing/               # 데이터 처리
    │   │   ├── __init__.py
    │   │   ├── document_loader.py         # 문서 로딩 (PDF, DOCX, TXT)
    │   │   ├── text_splitter.py           # 텍스트 청킹
    │   │   ├── embedding_generator.py     # 임베딩 생성
    │   │   └── vector_store.py            # 벡터 저장소 관리
    │   │
    │   ├── retrieval/                     # 검색 및 검색기
    │   │   ├── __init__.py
    │   │   ├── base_retriever.py          # 기본 검색기
    │   │   ├── hybrid_retriever.py        # 하이브리드 검색
    │   │   ├── semantic_retriever.py      # 의미적 검색
    │   │   └── keyword_retriever.py       # 키워드 검색
    │   │
    │   └── evaluation/                    # 평가 및 메트릭
    │       ├── __init__.py
    │       ├── rag_evaluator.py           # RAG 성능 평가
    │       ├── metrics.py                 # 평가 지표
    │       └── benchmark.py               # 벤치마킹 도구
    │
    ├── frameworks/                        # 프레임워크별 구현
    │   ├── __init__.py
    │   │
    │   ├── langchain_examples/            # LangChain 기반 예제
    │   │   ├── __init__.py
    │   │   ├── basic_rag/                 # 기본 RAG 구현
    │   │   │   ├── simple_qa.py           # 간단한 Q&A 시스템
    │   │   │   ├── document_chat.py       # 문서 기반 채팅
    │   │   │   └── retrieval_chain.py     # 검색 체인
    │   │   │
    │   │   ├── advanced_rag/              # 고급 RAG 패턴
    │   │   │   ├── conversational_rag.py  # 대화형 RAG
    │   │   │   ├── multi_query_rag.py     # 다중 쿼리 RAG
    │   │   │   ├── hierarchical_rag.py    # 계층적 RAG
    │   │   │   └── agentic_rag.py         # 에이전트 기반 RAG
    │   │   │
    │   │   └── integration/               # 통합 예제
    │   │       ├── langchain_llamaindex.py # LangChain + LlamaIndex
    │   │       ├── memory_management.py   # 메모리 관리
    │   │       └── tool_integration.py    # 툴 연동
    │   │
    │   └── llamaindex_examples/           # LlamaIndex 기반 예제
    │       ├── __init__.py
    │       ├── basic_indexing/            # 기본 인덱싱
    │       │   ├── document_index.py      # 문서 인덱스
    │       │   ├── vector_index.py        # 벡터 인덱스
    │       │   └── graph_index.py         # 그래프 인덱스
    │       │
    │       ├── query_engines/             # 쿼리 엔진
    │       │   ├── basic_query.py         # 기본 쿼리
    │       │   ├── sub_question.py        # 서브 질문 쿼리
    │       │   ├── tree_summarize.py      # 트리 요약 쿼리
    │       │   └── router_query.py        # 라우터 쿼리
    │       │
    │       └── advanced_features/         # 고급 기능
    │           ├── auto_merging.py        # 자동 병합
    │           ├── query_rewriting.py     # 쿼리 재작성
    │           └── hybrid_retrieval.py    # 하이브리드 검색
    │
    ├── tutorials/                         # 단계별 튜토리얼
    │   ├── 01_getting_started/            # 시작하기
    │   │   ├── README.md                  # 튜토리얼 소개
    │   │   ├── setup_environment.py       # 환경 설정
    │   │   ├── hello_rag.py               # 첫 번째 RAG 예제
    │   │   └── api_comparison.py          # API 비교 예제
    │   │
    │   ├── 02_data_preparation/           # 데이터 준비
    │   │   ├── README.md
    │   │   ├── load_documents.py          # 문서 로딩
    │   │   ├── text_chunking.py           # 텍스트 청킹 전략
    │   │   ├── embedding_strategies.py    # 임베딩 전략
    │   │   └── vector_store_setup.py      # 벡터 스토어 설정
    │   │
    │   ├── 03_basic_rag/                  # 기본 RAG
    │   │   ├── README.md
    │   │   ├── simple_retrieval.py        # 간단한 검색
    │   │   ├── basic_generation.py        # 기본 생성
    │   │   ├── end_to_end_rag.py         # 종단간 RAG
    │   │   └── evaluation_basics.py       # 기본 평가
    │   │
    │   ├── 04_advanced_rag/               # 고급 RAG
    │   │   ├── README.md
    │   │   ├── multi_document_rag.py      # 다중 문서 RAG
    │   │   ├── conversational_memory.py   # 대화 메모리
    │   │   ├── query_enhancement.py       # 쿼리 개선
    │   │   └── context_compression.py     # 컨텍스트 압축
    │   │
    │   ├── 05_local_models/               # 로컬 모델
    │   │   ├── README.md
    │   │   ├── ollama_setup.py            # Ollama 설정
    │   │   ├── exaone_integration.py      # EXAONE 모델 연동
    │   │   ├── korean_rag.py              # 한국어 RAG
    │   │   └── performance_optimization.py # 성능 최적화
    │   │
    │   └── 06_production_ready/           # 프로덕션 준비
    │       ├── README.md
    │       ├── scalability.py             # 확장성
    │       ├── monitoring.py              # 모니터링
    │       ├── deployment.py              # 배포
    │       └── api_server.py              # API 서버
    │
    ├── projects/                          # 실습 프로젝트
    │   ├── chatbot/                       # 챗봇 프로젝트
    │   │   ├── README.md
    │   │   ├── app.py                     # Gradio 기반 웹앱
    │   │   ├── backend.py                 # 백엔드 로직
    │   │   └── requirements.txt           # 프로젝트별 요구사항
    │   │
    │   ├── document_qa/                   # 문서 Q&A 시스템
    │   │   ├── README.md
    │   │   ├── streamlit_app.py           # Streamlit 앱
    │   │   ├── document_processor.py      # 문서 처리기
    │   │   └── qa_engine.py               # Q&A 엔진
    │   │
    │   ├── knowledge_base/                # 지식베이스 구축
    │   │   ├── README.md
    │   │   ├── kb_builder.py              # 지식베이스 구축기
    │   │   ├── search_interface.py        # 검색 인터페이스
    │   │   └── update_pipeline.py         # 업데이트 파이프라인
    │   │
    │   └── multi_agent_rag/               # 멀티 에이전트 RAG
    │       ├── README.md
    │       ├── agent_coordinator.py       # 에이전트 코디네이터
    │       ├── specialized_agents.py      # 전문 에이전트들
    │       └── collaborative_rag.py       # 협업 RAG
    │
    ├── notebooks/                         # Jupyter 노트북
    │   ├── exploratory/                   # 탐색적 분석
    │   │   ├── framework_comparison.ipynb # 프레임워크 비교
    │   │   ├── embedding_analysis.ipynb   # 임베딩 분석
    │   │   └── retrieval_strategies.ipynb # 검색 전략 분석
    │   │
    │   ├── experiments/                   # 실험 노트북
    │   │   ├── parameter_tuning.ipynb     # 파라미터 튜닝
    │   │   ├── model_comparison.ipynb     # 모델 비교
    │   │   └── performance_analysis.ipynb # 성능 분석
    │   │
    │   └── visualization/                 # 시각화
    │       ├── embedding_viz.ipynb        # 임베딩 시각화
    │       ├── retrieval_viz.ipynb        # 검색 결과 시각화
    │       └── metrics_dashboard.ipynb    # 메트릭 대시보드
    │
    ├── data/                              # 샘플 데이터
    │   ├── documents/                     # 샘플 문서들
    │   │   ├── korean/                    # 한국어 문서
    │   │   │   ├── news_articles/         # 뉴스 기사
    │   │   │   ├── academic_papers/       # 학술 논문
    │   │   │   └── business_docs/         # 비즈니스 문서
    │   │   │
    │   │   └── english/                   # 영어 문서
    │   │       ├── technical_docs/        # 기술 문서
    │   │       ├── research_papers/       # 연구 논문
    │   │       └── general_knowledge/     # 일반 지식
    │   │
    │   ├── datasets/                      # 평가용 데이터셋
    │   │   ├── qa_pairs.json             # Q&A 쌍
    │   │   ├── benchmark_data.json        # 벤치마크 데이터
    │   │   └── evaluation_sets/           # 평가 세트
    │   │
    │   └── vector_stores/                 # 벡터 스토어 데이터
    │       ├── chromadb/                  # ChromaDB 데이터
    │       ├── faiss/                     # FAISS 인덱스
    │       └── pinecone/                  # Pinecone 데이터
    │
    ├── tests/                             # 테스트 코드
    │   ├── __init__.py
    │   ├── unit/                          # 단위 테스트
    │   │   ├── test_llm_providers.py      # LLM 제공자 테스트
    │   │   ├── test_data_processing.py    # 데이터 처리 테스트
    │   │   ├── test_retrieval.py          # 검색 테스트
    │   │   └── test_evaluation.py         # 평가 테스트
    │   │
    │   ├── integration/                   # 통합 테스트
    │   │   ├── test_langchain_flow.py     # LangChain 플로우 테스트
    │   │   ├── test_llamaindex_flow.py    # LlamaIndex 플로우 테스트
    │   │   └── test_end_to_end.py         # 종단간 테스트
    │   │
    │   └── performance/                   # 성능 테스트
    │       ├── benchmark_retrieval.py     # 검색 성능 벤치마크
    │       ├── benchmark_generation.py    # 생성 성능 벤치마크
    │       └── load_testing.py            # 부하 테스트
    │
    ├── scripts/                           # 유틸리티 스크립트
    │   ├── setup_ollama.sh                # Ollama 설정 스크립트
    │   ├── download_models.py             # 모델 다운로드
    │   ├── prepare_data.py                # 데이터 준비
    │   ├── run_evaluation.py              # 평가 실행
    │   └── deploy.py                      # 배포 스크립트
    │
    └── examples/                          # 간단한 예제들
        ├── quick_start/                   # 빠른 시작 예제
        │   ├── 5min_rag.py                # 5분만에 RAG 구현
        │   ├── api_comparison.py          # API 비교
        │   └── local_vs_cloud.py          # 로컬 vs 클라우드
        │
        ├── use_cases/                     # 사용 사례별 예제
        │   ├── customer_support.py        # 고객 지원
        │   ├── research_assistant.py      # 연구 보조
        │   ├── code_documentation.py      # 코드 문서화
        │   └── legal_document_search.py   # 법률 문서 검색
        │
        └── integrations/                  # 통합 예제
            ├── gradio_interface.py        # Gradio 인터페이스
            ├── streamlit_dashboard.py     # Streamlit 대시보드
            ├── fastapi_server.py          # FastAPI 서버
            └── websocket_chat.py          # WebSocket 채팅
```

## 주요 특징

### 📚 학습 친화적 구조
- **점진적 학습**: 기초부터 고급까지 단계별 구성
- **실무 중심**: 실제 프로젝트에 적용 가능한 예제
- **한국어 특화**: EXAONE 모델과 한국어 데이터 지원

### 🔧 2025년 최신 기술 스택
- **LangChain**: 복잡한 워크플로우와 에이전트 구현
- **LlamaIndex**: 효율적인 데이터 인덱싱과 검색
- **Gradio**: 현대적인 웹 인터페이스
- **EXAONE 4.0**: LG AI 연구소의 한국어 특화 모델

### 🚀 다양한 LLM 지원
- **OpenAI API**: GPT-4, GPT-4 Turbo 등
- **Claude API**: Claude-3.5 Sonnet, Claude-3 Opus 등
- **로컬 모델**: EXAONE 4.0 (Ollama 통해)

### 🎯 실습 중심 학습
- **튜토리얼**: 6단계 체계적 학습 과정
- **프로젝트**: 4가지 실무 프로젝트
- **노트북**: 탐색적 분석과 실험
- **예제**: 빠른 이해를 위한 간단한 예제

### 🔍 포괄적 평가 시스템
- **성능 메트릭**: RAGAS, BLEU, BERTScore 등
- **벤치마킹**: 모델별, 전략별 성능 비교
- **시각화**: 결과 분석을 위한 대시보드