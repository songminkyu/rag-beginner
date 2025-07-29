"""
Multi-Query RAG with LangChain
LangChain을 사용한 다중 쿼리 RAG 시스템
"""

import logging
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# LangChain imports (optional)
try:
    from langchain.retrievers.multi_query import MultiQueryRetriever
    from langchain.prompts import PromptTemplate
    from langchain.schema import Document
    from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
    from langchain.chains import LLMChain
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

logger = logging.getLogger(__name__)


class MultiQueryRAG:
    """LangChain을 사용한 다중 쿼리 RAG 시스템"""
    
    def __init__(
        self,
        llm,
        embeddings,
        vector_store,
        config: Optional[Dict[str, Any]] = None
    ):
        self.config = config or {}
        self.language = self.config.get("language", "korean")
        self.num_queries = self.config.get("num_queries", 3)
        self.enable_query_fusion = self.config.get("enable_query_fusion", True)
        self.enable_parallel_search = self.config.get("enable_parallel_search", True)
        self.diversity_threshold = self.config.get("diversity_threshold", 0.7)
        self.max_documents_per_query = self.config.get("max_documents_per_query", 5)
        self.return_source_documents = self.config.get("return_source_documents", True)
        self.return_query_analysis = self.config.get("return_query_analysis", False)
        
        # LangChain 사용 여부
        self.use_langchain = HAS_LANGCHAIN and self.config.get("use_langchain", True)
        
        if self.use_langchain and HAS_LANGCHAIN:
            self._setup_langchain_multi_query(llm, embeddings, vector_store)
        else:
            self._setup_custom_multi_query(llm, embeddings, vector_store)
        
        logger.info(f"MultiQueryRAG 초기화: LangChain={self.use_langchain}, Queries={self.num_queries}")
    
    def _setup_langchain_multi_query(self, llm, embeddings, vector_store):
        """LangChain을 사용한 다중 쿼리 RAG 설정"""
        
        self.llm = llm
        self.embeddings = embeddings
        self.vector_store = vector_store
        
        # 쿼리 생성 프롬프트 설정
        self._setup_query_generation_prompts()
        
        # MultiQueryRetriever 설정
        self.multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=self.vector_store.as_retriever(),
            llm=self.llm,
            prompt=self.query_generation_prompt,
            parser_key="queries"
        )
        
        # 답변 생성 프롬프트 설정
        self._setup_answer_prompts()
        
        logger.info("LangChain MultiQueryRetriever 설정 완료")
    
    def _setup_query_generation_prompts(self):
        """쿼리 생성 프롬프트 설정"""
        
        if self.language == "korean":
            query_template = """당신은 AI 언어 모델 도우미입니다. 사용자의 질문에 대해 벡터 데이터베이스에서 관련 문서를 검색하는 것이 목표입니다.
사용자의 질문에 대해 다양한 관점에서 {num_queries}개의 서로 다른 질문을 생성해주세요.

원본 질문: {question}

각 질문은 다음 조건을 만족해야 합니다:
1. 원본 질문과 동일한 정보를 찾을 수 있어야 함
2. 서로 다른 키워드와 표현을 사용해야 함
3. 다양한 관점과 접근 방식을 반영해야 함
4. 명확하고 구체적이어야 함

{num_queries}개의 질문을 한 줄씩 작성해주세요:"""
        else:
            query_template = """You are an AI language model assistant. Your task is to generate {num_queries} different versions of the given question to retrieve relevant documents from a vector database.

Original question: {question}

Each question should:
1. Seek the same information as the original question
2. Use different keywords and expressions
3. Reflect various perspectives and approaches
4. Be clear and specific

Provide {num_queries} questions, one per line:"""
        
        self.query_generation_prompt = PromptTemplate(
            template=query_template,
            input_variables=["question", "num_queries"]
        )
    
    def _setup_answer_prompts(self):
        """답변 생성 프롬프트 설정"""
        
        if self.language == "korean":
            answer_template = """다음 문서들을 사용하여 질문에 답변해주세요. 여러 쿼리로 검색된 문서들이므로 중복될 수 있습니다.

검색된 문서들:
{context}

원본 질문: {question}
사용된 쿼리들: {queries}

답변 시 고려사항:
1. 모든 관련 문서의 정보를 종합적으로 활용하세요
2. 중복 정보는 한 번만 언급하세요
3. 다양한 관점의 정보가 있다면 모두 포함하세요
4. 확실하지 않은 정보는 추측하지 마세요

답변:"""
        else:
            answer_template = """Use the following documents to answer the question. These documents were retrieved using multiple queries, so there may be duplicates.

Retrieved Documents:
{context}

Original Question: {question}
Queries Used: {queries}

Guidelines:
1. Synthesize information from all relevant documents
2. Avoid repeating duplicate information
3. Include diverse perspectives if available
4. Don't guess information that's not provided

Answer:"""
        
        self.answer_prompt = PromptTemplate(
            template=answer_template,
            input_variables=["context", "question", "queries"]
        )
    
    def _setup_custom_multi_query(self, llm, embeddings, vector_store):
        """커스텀 다중 쿼리 RAG 설정 (LangChain 없이)"""
        
        self.llm = llm
        self.embeddings = embeddings
        self.vector_store = vector_store
        
        # 프롬프트 설정
        self._setup_query_generation_prompts()
        self._setup_answer_prompts()
        
        logger.info("커스텀 MultiQueryRAG 설정 완료")
    
    def search(
        self,
        question: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """다중 쿼리 검색 수행"""
        
        if self.use_langchain and HAS_LANGCHAIN:
            return self._search_langchain(question, k, filters)
        else:
            return self._search_custom(question, k, filters)
    
    def _search_langchain(
        self,
        question: str,
        k: int,
        filters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """LangChain을 사용한 다중 쿼리 검색"""
        
        try:
            # 검색 설정
            self.multi_query_retriever.retriever.search_kwargs = {"k": k}
            if filters:
                self.multi_query_retriever.retriever.search_kwargs["filters"] = filters
            
            # 다중 쿼리 검색 수행
            documents = self.multi_query_retriever.get_relevant_documents(question)
            
            # 쿼리 생성 기록 (LangChain 내부에서 생성됨)
            generated_queries = self._generate_queries_langchain(question)
            
            response = {
                "question": question,
                "generated_queries": generated_queries,
                "documents": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in documents
                ],
                "total_documents": len(documents),
                "method": "langchain",
                "timestamp": datetime.now().isoformat()
            }
            
            return response
            
        except Exception as e:
            logger.error(f"LangChain 다중 쿼리 검색 오류: {e}")
            return {
                "question": question,
                "error": str(e),
                "method": "langchain",
                "timestamp": datetime.now().isoformat()
            }
    
    def _generate_queries_langchain(self, question: str) -> List[str]:
        """LangChain을 사용한 쿼리 생성"""
        
        try:
            # 쿼리 생성 체인
            query_chain = LLMChain(
                llm=self.llm,
                prompt=self.query_generation_prompt
            )
            
            result = query_chain.run(
                question=question,
                num_queries=self.num_queries
            )
            
            # 결과 파싱
            queries = [q.strip() for q in result.split('\n') if q.strip()]
            
            # 원본 질문 포함
            if question not in queries:
                queries.insert(0, question)
            
            return queries[:self.num_queries + 1]
            
        except Exception as e:
            logger.warning(f"쿼리 생성 실패: {e}")
            return [question]
    
    def _search_custom(
        self,
        question: str,
        k: int,
        filters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """커스텀 다중 쿼리 검색 (LangChain 없이)"""
        
        try:
            # 1. 다중 쿼리 생성
            generated_queries = self._generate_queries_custom(question)
            
            # 2. 각 쿼리로 검색 수행
            all_documents = []
            query_results = {}
            
            if self.enable_parallel_search:
                # 병렬 검색
                all_documents, query_results = self._parallel_search(
                    generated_queries, k, filters
                )
            else:
                # 순차 검색
                all_documents, query_results = self._sequential_search(
                    generated_queries, k, filters
                )
            
            # 3. 문서 중복 제거 및 다양성 확보
            unique_documents = self._deduplicate_documents(all_documents)
            
            response = {
                "question": question,
                "generated_queries": generated_queries,
                "query_results": query_results if self.return_query_analysis else None,
                "documents": [
                    {
                        "content": doc.content,
                        "metadata": doc.metadata
                    }
                    for doc in unique_documents
                ],
                "total_documents": len(unique_documents),
                "method": "custom",
                "timestamp": datetime.now().isoformat()
            }
            
            return response
            
        except Exception as e:
            logger.error(f"커스텀 다중 쿼리 검색 오류: {e}")
            return {
                "question": question,
                "error": str(e),
                "method": "custom",
                "timestamp": datetime.now().isoformat()
            }
    
    def _generate_queries_custom(self, question: str) -> List[str]:
        """커스텀 쿼리 생성"""
        
        try:
            prompt = self.query_generation_prompt.format(
                question=question,
                num_queries=self.num_queries
            )
            
            if hasattr(self.llm, 'generate'):
                response = self.llm.generate(prompt)
                result = response.content
            else:
                result = str(self.llm(prompt))
            
            # 결과 파싱
            queries = []
            for line in result.split('\n'):
                line = line.strip()
                # 번호나 특수문자 제거
                cleaned = line.lstrip('123456789.- ').strip()
                if cleaned and len(cleaned) > 5:
                    queries.append(cleaned)
            
            # 원본 질문 포함
            if question not in queries:
                queries.insert(0, question)
            
            return queries[:self.num_queries + 1]
            
        except Exception as e:
            logger.warning(f"쿼리 생성 실패: {e}")
            return [question]
    
    def _parallel_search(
        self,
        queries: List[str],
        k: int,
        filters: Optional[Dict[str, Any]]
    ) -> tuple:
        """병렬 검색 수행"""
        
        all_documents = []
        query_results = {}
        
        with ThreadPoolExecutor(max_workers=min(len(queries), 5)) as executor:
            # 각 쿼리에 대한 검색 작업 제출
            future_to_query = {
                executor.submit(self._search_single_query, query, k, filters): query
                for query in queries
            }
            
            # 결과 수집
            for future in as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    documents = future.result()
                    all_documents.extend(documents)
                    query_results[query] = {
                        "documents_found": len(documents),
                        "status": "success"
                    }
                except Exception as e:
                    logger.warning(f"쿼리 '{query}' 검색 실패: {e}")
                    query_results[query] = {
                        "documents_found": 0,
                        "status": "failed",
                        "error": str(e)
                    }
        
        return all_documents, query_results
    
    def _sequential_search(
        self,
        queries: List[str],
        k: int,
        filters: Optional[Dict[str, Any]]
    ) -> tuple:
        """순차 검색 수행"""
        
        all_documents = []
        query_results = {}
        
        for query in queries:
            try:
                documents = self._search_single_query(query, k, filters)
                all_documents.extend(documents)
                query_results[query] = {
                    "documents_found": len(documents),
                    "status": "success"
                }
            except Exception as e:
                logger.warning(f"쿼리 '{query}' 검색 실패: {e}")
                query_results[query] = {
                    "documents_found": 0,
                    "status": "failed",
                    "error": str(e)
                }
        
        return all_documents, query_results
    
    def _search_single_query(
        self,
        query: str,
        k: int,
        filters: Optional[Dict[str, Any]]
    ) -> List:
        """단일 쿼리 검색"""
        
        if hasattr(self.vector_store, 'search'):
            query_embedding = self.embeddings.embed_text(query)
            search_result = self.vector_store.search(
                query_embedding, 
                k=self.max_documents_per_query, 
                filters=filters
            )
            return search_result.documents
        else:
            return []
    
    def _deduplicate_documents(self, documents: List) -> List:
        """문서 중복 제거 및 다양성 확보"""
        
        if not documents:
            return []
        
        # 내용 기반 중복 제거
        seen_content = set()
        unique_documents = []
        
        for doc in documents:
            # 내용의 해시값으로 중복 확인
            content_hash = hash(doc.content[:500])  # 첫 500자로 해시
            
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_documents.append(doc)
        
        # 다양성 임계값 적용 (선택적)
        if self.diversity_threshold < 1.0:
            unique_documents = self._apply_diversity_filter(unique_documents)
        
        return unique_documents
    
    def _apply_diversity_filter(self, documents: List) -> List:
        """다양성 필터 적용"""
        
        if len(documents) <= 1:
            return documents
        
        try:
            # 문서들의 임베딩 생성
            doc_embeddings = []
            for doc in documents:
                embedding = self.embeddings.embed_text(doc.content[:200])
                doc_embeddings.append(embedding)
            
            # 다양성 기반 선택 (간단한 휴리스틱)
            selected_documents = [documents[0]]  # 첫 번째 문서는 항상 포함
            selected_embeddings = [doc_embeddings[0]]
            
            for i, (doc, embedding) in enumerate(zip(documents[1:], doc_embeddings[1:]), 1):
                # 기존 선택된 문서들과의 유사도 확인
                max_similarity = 0
                for sel_embedding in selected_embeddings:
                    similarity = self._cosine_similarity(embedding, sel_embedding)
                    max_similarity = max(max_similarity, similarity)
                
                # 다양성 임계값 이하면 추가
                if max_similarity < self.diversity_threshold:
                    selected_documents.append(doc)
                    selected_embeddings.append(embedding)
            
            return selected_documents
            
        except Exception as e:
            logger.warning(f"다양성 필터 적용 실패: {e}")
            return documents
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """코사인 유사도 계산"""
        
        try:
            import numpy as np
            
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            
            dot_product = np.dot(vec1, vec2)
            norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)
            
            return dot_product / norms if norms != 0 else 0
            
        except ImportError:
            # NumPy가 없는 경우 간단한 구현
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5
            
            return dot_product / (norm1 * norm2) if norm1 * norm2 != 0 else 0
    
    def ask(
        self,
        question: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """질문에 대한 답변 생성 (검색 + 답변 생성)"""
        
        # 1. 다중 쿼리 검색
        search_result = self.search(question, k, filters)
        
        if "error" in search_result:
            return search_result
        
        try:
            # 2. 검색된 문서들로부터 답변 생성
            documents = search_result["documents"]
            generated_queries = search_result["generated_queries"]
            
            # 컨텍스트 준비
            contexts = [doc["content"] for doc in documents]
            combined_context = "\n\n".join(contexts)
            
            # 쿼리 목록 준비
            queries_text = "\n".join([f"- {q}" for q in generated_queries])
            
            # 프롬프트 생성
            prompt = self.answer_prompt.format(
                context=combined_context,
                question=question,
                queries=queries_text
            )
            
            # 답변 생성
            if hasattr(self.llm, 'generate'):
                response = self.llm.generate(prompt)
                answer = response.content
            else:
                answer = str(self.llm(prompt))
            
            # 응답 구성
            result = {
                "question": question,
                "answer": answer,
                "generated_queries": generated_queries,
                "search_method": search_result["method"],
                "timestamp": datetime.now().isoformat()
            }
            
            if self.return_source_documents:
                result["source_documents"] = documents
            
            if self.return_query_analysis and "query_results" in search_result:
                result["query_analysis"] = search_result["query_results"]
            
            return result
            
        except Exception as e:
            logger.error(f"답변 생성 오류: {e}")
            return {
                "question": question,
                "answer": "답변 생성 중 오류가 발생했습니다.",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def batch_ask(
        self,
        questions: List[str],
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """여러 질문에 대한 배치 처리"""
        
        results = []
        
        for i, question in enumerate(questions):
            logger.info(f"다중 쿼리 처리 중 ({i+1}/{len(questions)}): {question[:50]}...")
            
            try:
                result = self.ask(question, k, filters)
                results.append(result)
                
            except Exception as e:
                logger.error(f"질문 {i} 처리 오류: {e}")
                results.append({
                    "question": question,
                    "answer": "처리 중 오류가 발생했습니다.",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        return results
    
    def analyze_query_performance(self, question: str, k: int = 5) -> Dict[str, Any]:
        """쿼리 성능 분석"""
        
        try:
            # 다중 쿼리 생성
            generated_queries = self._generate_queries_custom(question)
            
            # 각 쿼리별 검색 성능 분석
            query_analysis = {}
            
            for query in generated_queries:
                start_time = datetime.now()
                
                try:
                    documents = self._search_single_query(query, k, None)
                    
                    duration = (datetime.now() - start_time).total_seconds()
                    
                    query_analysis[query] = {
                        "documents_found": len(documents),
                        "search_duration_seconds": duration,
                        "status": "success",
                        "unique_documents": len(set(doc.id for doc in documents if hasattr(doc, 'id')))
                    }
                    
                except Exception as e:
                    query_analysis[query] = {
                        "documents_found": 0,
                        "search_duration_seconds": 0,
                        "status": "failed",
                        "error": str(e)
                    }
            
            # 전체 성능 요약
            total_documents = sum(q.get("documents_found", 0) for q in query_analysis.values())
            avg_duration = sum(q.get("search_duration_seconds", 0) for q in query_analysis.values()) / len(query_analysis)
            success_rate = sum(1 for q in query_analysis.values() if q.get("status") == "success") / len(query_analysis)
            
            return {
                "original_question": question,
                "generated_queries": generated_queries,
                "query_analysis": query_analysis,
                "performance_summary": {
                    "total_queries": len(generated_queries),
                    "total_documents_found": total_documents,
                    "average_search_duration": avg_duration,
                    "success_rate": success_rate,
                    "queries_per_document": len(generated_queries) / max(total_documents, 1)
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"쿼리 성능 분석 오류: {e}")
            return {"error": str(e)}
    
    def get_config(self) -> Dict[str, Any]:
        """현재 설정 반환"""
        
        return {
            "language": self.language,
            "num_queries": self.num_queries,
            "enable_query_fusion": self.enable_query_fusion,
            "enable_parallel_search": self.enable_parallel_search,
            "diversity_threshold": self.diversity_threshold,
            "max_documents_per_query": self.max_documents_per_query,
            "use_langchain": self.use_langchain,
            "has_langchain": HAS_LANGCHAIN,
            "return_source_documents": self.return_source_documents,
            "return_query_analysis": self.return_query_analysis,
            "llm_type": type(self.llm).__name__,
            "embeddings_type": type(self.embeddings).__name__,
            "vector_store_type": type(self.vector_store).__name__
        }