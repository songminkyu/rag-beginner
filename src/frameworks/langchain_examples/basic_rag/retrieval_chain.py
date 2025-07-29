"""
Retrieval Chain with LangChain
LangChain을 사용한 검색 체인 구현
"""

import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

# LangChain imports (optional)
try:
    from langchain.chains import LLMChain
    from langchain.chains.base import Chain
    from langchain.prompts import PromptTemplate
    from langchain.schema import Document
    from langchain.callbacks.manager import CallbackManagerForChainRun
    from langchain.vectorstores.base import VectorStore as LangChainVectorStore
    from langchain.embeddings.base import Embeddings
    from langchain.llms.base import LLM
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

logger = logging.getLogger(__name__)


class RetrievalChain:
    """LangChain을 사용한 검색 체인 시스템"""
    
    def __init__(
        self,
        llm,
        embeddings,
        vector_store,
        config: Optional[Dict[str, Any]] = None
    ):
        self.config = config or {}
        self.language = self.config.get("language", "korean")
        self.chain_type = self.config.get("chain_type", "map_reduce")  # stuff, map_reduce, refine
        self.return_intermediate_steps = self.config.get("return_intermediate_steps", False)
        self.return_source_documents = self.config.get("return_source_documents", True)
        
        # LangChain 사용 여부
        self.use_langchain = HAS_LANGCHAIN and self.config.get("use_langchain", True)
        
        if self.use_langchain and HAS_LANGCHAIN:
            self._setup_langchain_chain(llm, embeddings, vector_store)
        else:
            self._setup_custom_chain(llm, embeddings, vector_store)
        
        logger.info(f"RetrievalChain 초기화: LangChain={self.use_langchain}, Type={self.chain_type}")
    
    def _setup_langchain_chain(self, llm, embeddings, vector_store):
        """LangChain을 사용한 체인 설정"""
        
        self.llm = llm
        self.embeddings = embeddings
        self.vector_store = vector_store
        
        # 프롬프트 템플릿 설정
        self._setup_prompts()
        
        # 체인 타입별 설정
        if self.chain_type == "stuff":
            self._setup_stuff_chain()
        elif self.chain_type == "map_reduce":
            self._setup_map_reduce_chain()
        elif self.chain_type == "refine":
            self._setup_refine_chain()
        else:
            raise ValueError(f"지원하지 않는 체인 타입: {self.chain_type}")
        
        logger.info(f"LangChain {self.chain_type} 체인 설정 완료")
    
    def _setup_prompts(self):
        """프롬프트 템플릿 설정"""
        
        if self.language == "korean":
            # Stuff 체인용 프롬프트
            self.stuff_prompt = PromptTemplate(
                template="""다음 문서들을 사용하여 질문에 답변해주세요.

문서들:
{context}

질문: {question}

답변시 고려사항:
1. 문서의 정보를 정확히 활용하세요
2. 추론이 필요한 경우 근거를 명시하세요
3. 확실하지 않은 정보는 추측하지 마세요

답변:""",
                input_variables=["context", "question"]
            )
            
            # Map 단계 프롬프트
            self.map_prompt = PromptTemplate(
                template="""다음 문서를 사용하여 질문에 대한 부분적 답변을 제공해주세요.

문서:
{context}

질문: {question}

부분 답변:""",
                input_variables=["context", "question"]
            )
            
            # Reduce 단계 프롬프트
            self.reduce_prompt = PromptTemplate(
                template="""다음 부분 답변들을 종합하여 최종 답변을 작성해주세요.

부분 답변들:
{summaries}

질문: {question}

최종 답변:""",
                input_variables=["summaries", "question"]
            )
            
            # Refine 프롬프트
            self.refine_prompt = PromptTemplate(
                template="""기존 답변을 새로운 문서의 정보로 개선해주세요.

기존 답변:
{existing_answer}

새로운 문서:
{context}

질문: {question}

개선된 답변:""",
                input_variables=["existing_answer", "context", "question"]
            )
            
        else:
            # English prompts
            self.stuff_prompt = PromptTemplate(
                template="""Use the following documents to answer the question.

Documents:
{context}

Question: {question}

Guidelines:
1. Use information from the documents accurately
2. Provide reasoning when making inferences
3. Don't guess if you're not certain

Answer:""",
                input_variables=["context", "question"]
            )
            
            self.map_prompt = PromptTemplate(
                template="""Use the following document to provide a partial answer to the question.

Document:
{context}

Question: {question}

Partial Answer:""",
                input_variables=["context", "question"]
            )
            
            self.reduce_prompt = PromptTemplate(
                template="""Combine the following partial answers into a final answer.

Partial Answers:
{summaries}

Question: {question}

Final Answer:""",
                input_variables=["summaries", "question"]
            )
            
            self.refine_prompt = PromptTemplate(
                template="""Improve the existing answer using the new document.

Existing Answer:
{existing_answer}

New Document:
{context}

Question: {question}

Improved Answer:""",
                input_variables=["existing_answer", "context", "question"]
            )
    
    def _setup_stuff_chain(self):
        """Stuff 체인 설정 - 모든 문서를 하나의 프롬프트로"""
        
        self.llm_chain = LLMChain(
            llm=self.llm,
            prompt=self.stuff_prompt,
            verbose=False
        )
    
    def _setup_map_reduce_chain(self):
        """Map-Reduce 체인 설정 - 문서별 처리 후 결합"""
        
        # Map 체인
        self.map_chain = LLMChain(
            llm=self.llm,
            prompt=self.map_prompt,
            verbose=False
        )
        
        # Reduce 체인
        self.reduce_chain = LLMChain(
            llm=self.llm,
            prompt=self.reduce_prompt,
            verbose=False
        )
    
    def _setup_refine_chain(self):
        """Refine 체인 설정 - 순차적 개선"""
        
        # 초기 답변 체인
        initial_prompt = PromptTemplate(
            template=self.stuff_prompt.template,
            input_variables=["context", "question"]
        )
        
        self.initial_chain = LLMChain(
            llm=self.llm,
            prompt=initial_prompt,
            verbose=False
        )
        
        # 개선 체인
        self.refine_chain = LLMChain(
            llm=self.llm,
            prompt=self.refine_prompt,
            verbose=False
        )
    
    def _setup_custom_chain(self, llm, embeddings, vector_store):
        """커스텀 체인 설정 (LangChain 없이)"""
        
        self.llm = llm
        self.embeddings = embeddings
        self.vector_store = vector_store
        
        logger.info("커스텀 검색 체인 설정 완료")
    
    def run(
        self,
        question: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """검색 체인 실행"""
        
        if self.use_langchain and HAS_LANGCHAIN:
            return self._run_langchain(question, k, filters)
        else:
            return self._run_custom(question, k, filters)
    
    def _run_langchain(
        self,
        question: str,
        k: int,
        filters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """LangChain을 사용한 체인 실행"""
        
        try:
            # 1. 문서 검색
            retriever = self.vector_store.as_retriever()
            retriever.search_kwargs = {"k": k}
            if filters:
                retriever.search_kwargs["filters"] = filters
            
            documents = retriever.get_relevant_documents(question)
            
            # 2. 체인 타입별 처리
            if self.chain_type == "stuff":
                result = self._run_stuff_chain(question, documents)
            elif self.chain_type == "map_reduce":
                result = self._run_map_reduce_chain(question, documents)
            elif self.chain_type == "refine":
                result = self._run_refine_chain(question, documents)
            
            # 3. 응답 구성
            response = {
                "question": question,
                "answer": result["answer"],
                "chain_type": self.chain_type,
                "method": "langchain",
                "timestamp": datetime.now().isoformat()
            }
            
            if self.return_source_documents:
                response["source_documents"] = [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in documents
                ]
            
            if self.return_intermediate_steps and "intermediate_steps" in result:
                response["intermediate_steps"] = result["intermediate_steps"]
            
            return response
            
        except Exception as e:
            logger.error(f"LangChain 체인 실행 오류: {e}")
            return {
                "question": question,
                "answer": "체인 실행 중 오류가 발생했습니다.",
                "chain_type": self.chain_type,
                "method": "langchain",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _run_stuff_chain(self, question: str, documents: List[Document]) -> Dict[str, Any]:
        """Stuff 체인 실행"""
        
        # 모든 문서를 하나의 컨텍스트로 결합
        context = "\n\n".join([doc.page_content for doc in documents])
        
        # LLM 실행
        result = self.llm_chain.run(context=context, question=question)
        
        return {"answer": result}
    
    def _run_map_reduce_chain(self, question: str, documents: List[Document]) -> Dict[str, Any]:
        """Map-Reduce 체인 실행"""
        
        intermediate_steps = []
        
        # Map 단계: 각 문서별로 부분 답변 생성
        partial_answers = []
        for i, doc in enumerate(documents):
            partial_result = self.map_chain.run(
                context=doc.page_content,
                question=question
            )
            partial_answers.append(partial_result)
            
            if self.return_intermediate_steps:
                intermediate_steps.append({
                    "step": f"map_{i}",
                    "document": doc.page_content[:200] + "...",
                    "partial_answer": partial_result
                })
        
        # Reduce 단계: 부분 답변들을 종합
        summaries = "\n\n".join([
            f"답변 {i+1}: {answer}"
            for i, answer in enumerate(partial_answers)
        ])
        
        final_answer = self.reduce_chain.run(
            summaries=summaries,
            question=question
        )
        
        result = {"answer": final_answer}
        if self.return_intermediate_steps:
            intermediate_steps.append({
                "step": "reduce",
                "summaries": summaries,
                "final_answer": final_answer
            })
            result["intermediate_steps"] = intermediate_steps
        
        return result
    
    def _run_refine_chain(self, question: str, documents: List[Document]) -> Dict[str, Any]:
        """Refine 체인 실행"""
        
        intermediate_steps = []
        
        if not documents:
            return {"answer": "검색된 문서가 없습니다."}
        
        # 첫 번째 문서로 초기 답변 생성
        current_answer = self.initial_chain.run(
            context=documents[0].page_content,
            question=question
        )
        
        if self.return_intermediate_steps:
            intermediate_steps.append({
                "step": "initial",
                "document": documents[0].page_content[:200] + "...",
                "answer": current_answer
            })
        
        # 나머지 문서들로 순차적 개선
        for i, doc in enumerate(documents[1:], 1):
            improved_answer = self.refine_chain.run(
                existing_answer=current_answer,
                context=doc.page_content,
                question=question
            )
            
            if self.return_intermediate_steps:
                intermediate_steps.append({
                    "step": f"refine_{i}",
                    "document": doc.page_content[:200] + "...",
                    "previous_answer": current_answer,
                    "improved_answer": improved_answer
                })
            
            current_answer = improved_answer
        
        result = {"answer": current_answer}
        if self.return_intermediate_steps:
            result["intermediate_steps"] = intermediate_steps
        
        return result
    
    def _run_custom(
        self,
        question: str,
        k: int,
        filters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """커스텀 체인 실행 (LangChain 없이)"""
        
        try:
            # 1. 문서 검색
            if hasattr(self.vector_store, 'search'):
                query_embedding = self.embeddings.embed_text(question)
                search_result = self.vector_store.search(
                    query_embedding, k=k, filters=filters
                )
                retrieved_docs = search_result.documents
            else:
                retrieved_docs = []
            
            # 2. 체인 타입별 처리
            if self.chain_type == "stuff":
                answer = self._custom_stuff_chain(question, retrieved_docs)
            elif self.chain_type == "map_reduce":
                answer = self._custom_map_reduce_chain(question, retrieved_docs)
            elif self.chain_type == "refine":
                answer = self._custom_refine_chain(question, retrieved_docs)
            else:
                answer = self._custom_stuff_chain(question, retrieved_docs)
            
            # 3. 응답 구성
            response = {
                "question": question,
                "answer": answer,
                "chain_type": self.chain_type,
                "method": "custom",
                "timestamp": datetime.now().isoformat()
            }
            
            if self.return_source_documents:
                response["source_documents"] = [
                    {
                        "content": doc.content,
                        "metadata": doc.metadata
                    }
                    for doc in retrieved_docs
                ]
            
            return response
            
        except Exception as e:
            logger.error(f"커스텀 체인 실행 오류: {e}")
            return {
                "question": question,
                "answer": "체인 실행 중 오류가 발생했습니다.",
                "chain_type": self.chain_type,
                "method": "custom",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _custom_stuff_chain(self, question: str, documents: List) -> str:
        """커스텀 Stuff 체인"""
        
        if not documents:
            return "검색된 문서가 없습니다."
        
        # 모든 문서 결합
        context = "\n\n".join([doc.content for doc in documents])
        
        # 프롬프트 생성
        if self.language == "korean":
            prompt = f"""다음 문서들을 사용하여 질문에 답변해주세요.

문서들:
{context}

질문: {question}

답변:"""
        else:
            prompt = f"""Use the following documents to answer the question.

Documents:
{context}

Question: {question}

Answer:"""
        
        # 답변 생성
        if hasattr(self.llm, 'generate'):
            response = self.llm.generate(prompt)
            return response.content
        else:
            return str(self.llm(prompt))
    
    def _custom_map_reduce_chain(self, question: str, documents: List) -> str:
        """커스텀 Map-Reduce 체인"""
        
        if not documents:
            return "검색된 문서가 없습니다."
        
        # Map 단계: 각 문서별 부분 답변
        partial_answers = []
        for doc in documents:
            if self.language == "korean":
                prompt = f"""다음 문서를 사용하여 질문에 대한 부분적 답변을 제공해주세요.

문서:
{doc.content}

질문: {question}

부분 답변:"""
            else:
                prompt = f"""Use the following document to provide a partial answer.

Document:
{doc.content}

Question: {question}

Partial Answer:"""
            
            if hasattr(self.llm, 'generate'):
                response = self.llm.generate(prompt)
                partial_answers.append(response.content)
            else:
                partial_answers.append(str(self.llm(prompt)))
        
        # Reduce 단계: 부분 답변 결합
        summaries = "\n\n".join([
            f"답변 {i+1}: {answer}"
            for i, answer in enumerate(partial_answers)
        ])
        
        if self.language == "korean":
            final_prompt = f"""다음 부분 답변들을 종합하여 최종 답변을 작성해주세요.

부분 답변들:
{summaries}

질문: {question}

최종 답변:"""
        else:
            final_prompt = f"""Combine the following partial answers into a final answer.

Partial Answers:
{summaries}

Question: {question}

Final Answer:"""
        
        if hasattr(self.llm, 'generate'):
            response = self.llm.generate(final_prompt)
            return response.content
        else:
            return str(self.llm(final_prompt))
    
    def _custom_refine_chain(self, question: str, documents: List) -> str:
        """커스텀 Refine 체인"""
        
        if not documents:
            return "검색된 문서가 없습니다."
        
        # 첫 번째 문서로 초기 답변
        first_doc = documents[0]
        if self.language == "korean":
            initial_prompt = f"""다음 문서를 사용하여 질문에 답변해주세요.

문서:
{first_doc.content}

질문: {question}

답변:"""
        else:
            initial_prompt = f"""Use the following document to answer the question.

Document:
{first_doc.content}

Question: {question}

Answer:"""
        
        if hasattr(self.llm, 'generate'):
            response = self.llm.generate(initial_prompt)
            current_answer = response.content
        else:
            current_answer = str(self.llm(initial_prompt))
        
        # 나머지 문서들로 순차적 개선
        for doc in documents[1:]:
            if self.language == "korean":
                refine_prompt = f"""기존 답변을 새로운 문서의 정보로 개선해주세요.

기존 답변:
{current_answer}

새로운 문서:
{doc.content}

질문: {question}

개선된 답변:"""
            else:
                refine_prompt = f"""Improve the existing answer using the new document.

Existing Answer:
{current_answer}

New Document:
{doc.content}

Question: {question}

Improved Answer:"""
            
            if hasattr(self.llm, 'generate'):
                response = self.llm.generate(refine_prompt)
                current_answer = response.content
            else:
                current_answer = str(self.llm(refine_prompt))
        
        return current_answer
    
    def batch_run(
        self,
        questions: List[str],
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """여러 질문에 대한 배치 처리"""
        
        results = []
        
        for i, question in enumerate(questions):
            logger.info(f"체인 실행 중 ({i+1}/{len(questions)}): {question[:50]}...")
            
            try:
                result = self.run(question, k, filters)
                results.append(result)
                
            except Exception as e:
                logger.error(f"질문 {i} 처리 오류: {e}")
                results.append({
                    "question": question,
                    "answer": "체인 실행 중 오류가 발생했습니다.",
                    "chain_type": self.chain_type,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        return results
    
    def get_config(self) -> Dict[str, Any]:
        """현재 설정 반환"""
        
        return {
            "language": self.language,
            "chain_type": self.chain_type,
            "use_langchain": self.use_langchain,
            "has_langchain": HAS_LANGCHAIN,
            "return_intermediate_steps": self.return_intermediate_steps,
            "return_source_documents": self.return_source_documents,
            "llm_type": type(self.llm).__name__,
            "embeddings_type": type(self.embeddings).__name__,
            "vector_store_type": type(self.vector_store).__name__
        }


def create_retrieval_chain_from_documents(
    documents: List[Dict[str, Any]],
    llm_provider,
    embedding_generator,
    chain_type: str = "stuff",
    vector_store_config: Optional[Dict[str, Any]] = None,
    chain_config: Optional[Dict[str, Any]] = None
) -> RetrievalChain:
    """문서로부터 RetrievalChain 시스템 생성"""
    
    from ....core.data_processing.vector_store import create_vector_store
    from ....core.data_processing.vector_store import VectorDocument
    
    # 벡터 스토어 설정
    vector_store_config = vector_store_config or {
        "collection_name": "retrieval_chain",
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
    
    # RetrievalChain 생성
    chain_config = chain_config or {}
    chain_config["chain_type"] = chain_type
    
    retrieval_chain = RetrievalChain(
        llm_provider, embedding_generator, vector_store, chain_config
    )
    
    logger.info(f"문서 기반 RetrievalChain 생성 완료: {len(documents)}개 문서, Type={chain_type}")
    return retrieval_chain