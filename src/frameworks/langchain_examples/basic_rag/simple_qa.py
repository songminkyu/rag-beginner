"""
Simple QA with LangChain
LangChain을 사용한 간단한 Q&A 시스템 구현
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

# LangChain imports (optional)
try:
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    from langchain.schema import Document
    from langchain.vectorstores.base import VectorStore as LangChainVectorStore
    from langchain.embeddings.base import Embeddings
    from langchain.llms.base import LLM
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

logger = logging.getLogger(__name__)


class SimpleQA:
    """LangChain을 사용한 간단한 Q&A 시스템"""
    
    def __init__(
        self,
        llm,  # LangChain LLM 또는 우리의 provider
        embeddings,  # LangChain Embeddings 또는 우리의 embedding generator
        vector_store,  # LangChain VectorStore 또는 우리의 vector store
        config: Optional[Dict[str, Any]] = None
    ):
        self.config = config or {}
        self.language = self.config.get("language", "korean")
        
        # LangChain 사용 여부 확인
        self.use_langchain = HAS_LANGCHAIN and self.config.get("use_langchain", True)
        
        if self.use_langchain and HAS_LANGCHAIN:
            self._setup_langchain_qa(llm, embeddings, vector_store)
        else:
            self._setup_custom_qa(llm, embeddings, vector_store)
        
        logger.info(f"SimpleQA 초기화 완료: LangChain={self.use_langchain}")
    
    def _setup_langchain_qa(self, llm, embeddings, vector_store):
        """LangChain을 사용한 Q&A 설정"""
        
        self.llm = llm
        self.embeddings = embeddings
        self.vector_store = vector_store
        
        # 프롬프트 템플릿 설정
        if self.language == "korean":
            template = """다음 컨텍스트를 사용하여 질문에 답변해주세요. 컨텍스트에서 답을 찾을 수 없다면 "주어진 정보로는 답변할 수 없습니다"라고 말해주세요.

컨텍스트: {context}

질문: {question}
답변:"""
        else:
            template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}
Answer:"""
        
        self.prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # RetrievalQA 체인 생성
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(),
            chain_type_kwargs={"prompt": self.prompt},
            return_source_documents=True
        )
        
        logger.info("LangChain RetrievalQA 체인 설정 완료")
    
    def _setup_custom_qa(self, llm, embeddings, vector_store):
        """커스텀 Q&A 설정 (LangChain 없이)"""
        
        self.llm = llm
        self.embeddings = embeddings
        self.vector_store = vector_store
        
        logger.info("커스텀 Q&A 설정 완료")
    
    def ask(self, question: str, k: int = 5) -> Dict[str, Any]:
        """질문에 대한 답변 생성"""
        
        if self.use_langchain and HAS_LANGCHAIN:
            return self._ask_langchain(question, k)
        else:
            return self._ask_custom(question, k)
    
    def _ask_langchain(self, question: str, k: int) -> Dict[str, Any]:
        """LangChain을 사용한 질문 답변"""
        
        try:
            # 검색 개수 설정
            self.qa_chain.retriever.search_kwargs = {"k": k}
            
            # 질문 실행
            result = self.qa_chain({"query": question})
            
            return {
                "question": question,
                "answer": result["result"],
                "source_documents": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in result["source_documents"]
                ],
                "method": "langchain"
            }
            
        except Exception as e:
            logger.error(f"LangChain Q&A 오류: {e}")
            return {
                "question": question,
                "answer": "답변 생성 중 오류가 발생했습니다.",
                "source_documents": [],
                "error": str(e),
                "method": "langchain"
            }
    
    def _ask_custom(self, question: str, k: int) -> Dict[str, Any]:
        """커스텀 Q&A (LangChain 없이)"""
        
        try:
            # 1. 검색 수행
            if hasattr(self.vector_store, 'search'):
                # 우리의 vector store 사용
                query_embedding = self.embeddings.embed_text(question)
                search_result = self.vector_store.search(query_embedding, k=k)
                retrieved_docs = search_result.documents
            else:
                # 다른 벡터 스토어 인터페이스
                retrieved_docs = []
            
            # 2. 컨텍스트 준비
            contexts = []
            source_documents = []
            
            for doc in retrieved_docs:
                contexts.append(doc.content)
                source_documents.append({
                    "content": doc.content,
                    "metadata": doc.metadata
                })
            
            combined_context = "\n\n".join(contexts)
            
            # 3. 프롬프트 생성
            if self.language == "korean":
                prompt = f"""다음 컨텍스트를 사용하여 질문에 답변해주세요.

컨텍스트:
{combined_context}

질문: {question}
답변:"""
            else:
                prompt = f"""Use the following context to answer the question.

Context:
{combined_context}

Question: {question}
Answer:"""
            
            # 4. 답변 생성
            if hasattr(self.llm, 'generate'):
                # 우리의 LLM provider 사용
                response = self.llm.generate(prompt)
                answer = response.content
            else:
                # 다른 LLM 인터페이스
                answer = str(self.llm(prompt))
            
            return {
                "question": question,
                "answer": answer,
                "source_documents": source_documents,
                "method": "custom"
            }
            
        except Exception as e:
            logger.error(f"커스텀 Q&A 오류: {e}")
            return {
                "question": question,
                "answer": "답변 생성 중 오류가 발생했습니다.",
                "source_documents": [],
                "error": str(e),
                "method": "custom"
            }
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """문서 추가"""
        
        try:
            if self.use_langchain and HAS_LANGCHAIN:
                return self._add_documents_langchain(documents)
            else:
                return self._add_documents_custom(documents)
                
        except Exception as e:
            logger.error(f"문서 추가 오류: {e}")
            return False
    
    def _add_documents_langchain(self, documents: List[Dict[str, Any]]) -> bool:
        """LangChain을 사용한 문서 추가"""
        
        try:
            # Document 객체로 변환
            langchain_docs = []
            for doc in documents:
                langchain_doc = Document(
                    page_content=doc["content"],
                    metadata=doc.get("metadata", {})
                )
                langchain_docs.append(langchain_doc)
            
            # 벡터 스토어에 추가
            self.vector_store.add_documents(langchain_docs)
            
            logger.info(f"LangChain 벡터 스토어에 {len(documents)}개 문서 추가")
            return True
            
        except Exception as e:
            logger.error(f"LangChain 문서 추가 오류: {e}")
            return False
    
    def _add_documents_custom(self, documents: List[Dict[str, Any]]) -> bool:
        """커스텀 문서 추가"""
        
        try:
            # 우리의 vector store 사용
            if hasattr(self.vector_store, 'add_documents'):
                return self.vector_store.add_documents(documents)
            else:
                logger.warning("벡터 스토어가 문서 추가를 지원하지 않습니다.")
                return False
                
        except Exception as e:
            logger.error(f"커스텀 문서 추가 오류: {e}")
            return False
    
    def batch_ask(self, questions: List[str], k: int = 5) -> List[Dict[str, Any]]:
        """여러 질문에 대한 배치 처리"""
        
        results = []
        
        for i, question in enumerate(questions):
            logger.info(f"질문 처리 중 ({i+1}/{len(questions)}): {question[:50]}...")
            
            try:
                result = self.ask(question, k)
                results.append(result)
                
            except Exception as e:
                logger.error(f"질문 {i} 처리 오류: {e}")
                results.append({
                    "question": question,
                    "answer": "답변 생성 중 오류가 발생했습니다.",
                    "source_documents": [],
                    "error": str(e)
                })
        
        return results
    
    def get_config(self) -> Dict[str, Any]:
        """현재 설정 반환"""
        
        return {
            "language": self.language,
            "use_langchain": self.use_langchain,
            "has_langchain": HAS_LANGCHAIN,
            "llm_type": type(self.llm).__name__,
            "embeddings_type": type(self.embeddings).__name__,
            "vector_store_type": type(self.vector_store).__name__
        }


def create_simple_qa_from_documents(
    documents: List[Dict[str, Any]],
    llm_provider,
    embedding_generator,
    vector_store_config: Optional[Dict[str, Any]] = None,
    qa_config: Optional[Dict[str, Any]] = None
) -> SimpleQA:
    """문서로부터 SimpleQA 시스템 생성"""
    
    from ....core.data_processing.vector_store import create_vector_store
    from ....core.data_processing.vector_store import VectorDocument
    
    # 벡터 스토어 설정
    vector_store_config = vector_store_config or {
        "collection_name": "simple_qa",
        "dimension": embedding_generator.get_dimension()
    }
    
    # 벡터 스토어 생성
    vector_store = create_vector_store("inmemory", vector_store_config)
    
    # 문서를 벡터 문서로 변환
    vector_documents = []
    for i, doc in enumerate(documents):
        # 임베딩 생성
        embedding = embedding_generator.embed_text(doc["content"])
        
        vector_doc = VectorDocument(
            id=doc.get("id", f"doc_{i}"),
            content=doc["content"],
            embedding=embedding,
            metadata=doc.get("metadata", {})
        )
        vector_documents.append(vector_doc)
    
    # 벡터 스토어에 문서 추가
    vector_store.add_documents(vector_documents)
    
    # SimpleQA 생성
    qa_config = qa_config or {}
    qa_system = SimpleQA(llm_provider, embedding_generator, vector_store, qa_config)
    
    logger.info(f"문서 기반 SimpleQA 생성 완료: {len(documents)}개 문서")
    return qa_system


def create_simple_qa_from_files(
    file_paths: List[Path],
    llm_provider,
    embedding_generator,
    document_loader_config: Optional[Dict[str, Any]] = None,
    vector_store_config: Optional[Dict[str, Any]] = None,
    qa_config: Optional[Dict[str, Any]] = None
) -> SimpleQA:
    """파일들로부터 SimpleQA 시스템 생성"""
    
    from ....core.data_processing.document_loader import auto_detect_loader
    
    # 문서 로딩
    documents = []
    for file_path in file_paths:
        try:
            loader = auto_detect_loader(file_path)
            loaded_docs = loader.load(file_path)
            
            for doc in loaded_docs:
                documents.append({
                    "id": doc.doc_id,
                    "content": doc.content,
                    "metadata": doc.metadata
                })
                
        except Exception as e:
            logger.error(f"파일 로딩 오류 ({file_path}): {e}")
    
    # SimpleQA 생성
    return create_simple_qa_from_documents(
        documents, llm_provider, embedding_generator,
        vector_store_config, qa_config
    )