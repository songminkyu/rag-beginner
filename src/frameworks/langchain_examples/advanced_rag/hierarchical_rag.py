"""
Hierarchical RAG with LangChain
LangChain을 사용한 계층형 RAG 시스템
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

# LangChain imports (optional)
try:
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    from langchain.schema import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

logger = logging.getLogger(__name__)


@dataclass
class HierarchicalDocument:
    """계층형 문서 구조"""
    id: str
    level: int  # 0: 원본, 1: 섹션, 2: 청크
    parent_id: Optional[str]
    content: str
    summary: Optional[str]
    metadata: Dict[str, Any]
    children_ids: List[str]


class HierarchicalRAG:
    """LangChain을 사용한 계층형 RAG 시스템"""
    
    def __init__(
        self,
        llm,
        embeddings,
        vector_store,
        config: Optional[Dict[str, Any]] = None
    ):
        self.config = config or {}
        self.language = self.config.get("language", "korean")
        self.max_levels = self.config.get("max_levels", 3)  # 0: 문서, 1: 섹션, 2: 청크
        self.chunk_sizes = self.config.get("chunk_sizes", [4000, 1000, 300])
        self.chunk_overlaps = self.config.get("chunk_overlaps", [200, 100, 50])
        self.enable_summarization = self.config.get("enable_summarization", True)
        self.summary_compression_ratio = self.config.get("summary_compression_ratio", 0.3)
        self.search_strategy = self.config.get("search_strategy", "hierarchical")  # hierarchical, flat, adaptive
        self.return_source_documents = self.config.get("return_source_documents", True)
        self.return_hierarchy_info = self.config.get("return_hierarchy_info", False)
        
        # LangChain 사용 여부
        self.use_langchain = HAS_LANGCHAIN and self.config.get("use_langchain", True)
        
        # 계층형 문서 저장소
        self.document_hierarchy = {}  # doc_id -> HierarchicalDocument
        self.level_indices = {i: [] for i in range(self.max_levels)}  # level -> [doc_ids]
        
        if self.use_langchain and HAS_LANGCHAIN:
            self._setup_langchain_hierarchical_rag(llm, embeddings, vector_store)
        else:
            self._setup_custom_hierarchical_rag(llm, embeddings, vector_store)
        
        logger.info(f"HierarchicalRAG 초기화: LangChain={self.use_langchain}, Levels={self.max_levels}")
    
    def _setup_langchain_hierarchical_rag(self, llm, embeddings, vector_store):
        """LangChain을 사용한 계층형 RAG 설정"""
        
        self.llm = llm
        self.embeddings = embeddings
        self.vector_store = vector_store
        
        # 텍스트 분할기 설정
        self.text_splitters = []
        for i in range(self.max_levels):
            if i < len(self.chunk_sizes):
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_sizes[i],
                    chunk_overlap=self.chunk_overlaps[i] if i < len(self.chunk_overlaps) else 0,
                    length_function=len,
                    is_separator_regex=False,
                )
                self.text_splitters.append(splitter)
        
        # 프롬프트 설정
        self._setup_prompts()
        
        logger.info("LangChain HierarchicalRAG 설정 완료")
    
    def _setup_prompts(self):
        """프롬프트 템플릿 설정"""
        
        # 요약 생성 프롬프트
        if self.language == "korean":
            summary_template = """다음 텍스트의 핵심 내용을 요약해주세요. 요약은 원문의 {compression_ratio:.0%} 길이로 작성하며, 주요 정보와 키워드를 포함해야 합니다.

원문:
{text}

요약:"""
            
            # 계층형 검색 프롬프트
            hierarchical_template = """계층형 문서 구조에서 검색된 정보를 바탕으로 질문에 답변해주세요.

문서 계층 정보:
{hierarchy_info}

검색된 내용:
{context}

질문: {question}

답변시 고려사항:
1. 문서의 계층 구조를 고려하여 정보를 종합하세요
2. 상위 레벨 요약과 하위 레벨 세부사항을 모두 활용하세요
3. 관련성이 높은 정보를 우선적으로 사용하세요

답변:"""
            
        else:
            summary_template = """Summarize the following text to {compression_ratio:.0%} of its original length, maintaining key information and keywords.

Text:
{text}

Summary:"""
            
            hierarchical_template = """Answer the question based on the hierarchical document structure and retrieved information.

Document Hierarchy:
{hierarchy_info}

Retrieved Content:
{context}

Question: {question}

Guidelines:
1. Synthesize information considering the document hierarchy
2. Use both high-level summaries and detailed content
3. Prioritize most relevant information

Answer:"""
        
        self.summary_prompt = PromptTemplate(
            template=summary_template,
            input_variables=["text", "compression_ratio"]
        )
        
        self.hierarchical_prompt = PromptTemplate(
            template=hierarchical_template,
            input_variables=["hierarchy_info", "context", "question"]
        )
    
    def _setup_custom_hierarchical_rag(self, llm, embeddings, vector_store):
        """커스텀 계층형 RAG 설정 (LangChain 없이)"""
        
        self.llm = llm
        self.embeddings = embeddings
        self.vector_store = vector_store
        
        # 간단한 텍스트 분할 함수들
        self._setup_custom_splitters()
        self._setup_prompts()
        
        logger.info("커스텀 HierarchicalRAG 설정 완료")
    
    def _setup_custom_splitters(self):
        """커스텀 텍스트 분할기 설정"""
        
        self.text_splitters = []
        
        for i in range(self.max_levels):
            chunk_size = self.chunk_sizes[i] if i < len(self.chunk_sizes) else 1000
            chunk_overlap = self.chunk_overlaps[i] if i < len(self.chunk_overlaps) else 100
            
            def create_splitter(size, overlap):
                def splitter(text):
                    chunks = []
                    start = 0
                    while start < len(text):
                        end = start + size
                        chunk = text[start:end]
                        chunks.append(chunk)
                        start = end - overlap
                        if start >= len(text):
                            break
                    return chunks
                return splitter
            
            self.text_splitters.append(create_splitter(chunk_size, chunk_overlap))
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """문서를 계층형 구조로 추가"""
        
        try:
            for doc in documents:
                self._process_document_hierarchically(doc)
            
            logger.info(f"계층형 문서 {len(documents)}개 추가 완료")
            return True
            
        except Exception as e:
            logger.error(f"계층형 문서 추가 오류: {e}")
            return False
    
    def _process_document_hierarchically(self, document: Dict[str, Any]) -> str:
        """문서를 계층형으로 처리"""
        
        doc_id = document.get("id", f"doc_{datetime.now().timestamp()}")
        content = document["content"]
        metadata = document.get("metadata", {})
        
        # 레벨 0: 원본 문서
        root_doc = HierarchicalDocument(
            id=doc_id,
            level=0,
            parent_id=None,
            content=content,
            summary=None,
            metadata=metadata,
            children_ids=[]
        )
        
        # 요약 생성 (선택적)
        if self.enable_summarization:
            root_doc.summary = self._generate_summary(content)
        
        self.document_hierarchy[doc_id] = root_doc
        self.level_indices[0].append(doc_id)
        
        # 하위 레벨 처리
        current_content = content
        current_parent_id = doc_id
        
        for level in range(1, self.max_levels):
            child_ids = self._split_and_create_children(
                current_content, level, current_parent_id, metadata
            )
            
            if child_ids:
                self.document_hierarchy[current_parent_id].children_ids = child_ids
                
                # 다음 레벨을 위한 준비
                if level < self.max_levels - 1:
                    # 하위 레벨의 모든 청크를 결합 (너무 클 경우 샘플링)
                    combined_content = ""
                    for child_id in child_ids[:10]:  # 최대 10개 청크만
                        child_doc = self.document_hierarchy[child_id]
                        combined_content += child_doc.content + "\n\n"
                    current_content = combined_content
        
        # 벡터 저장소에 추가
        self._add_to_vector_store(doc_id)
        
        return doc_id
    
    def _split_and_create_children(
        self,
        content: str,
        level: int,
        parent_id: str,
        base_metadata: Dict[str, Any]
    ) -> List[str]:
        """내용을 분할하고 자식 문서들 생성"""
        
        if level >= len(self.text_splitters):
            return []
        
        # 텍스트 분할
        if self.use_langchain and HAS_LANGCHAIN and level < len(self.text_splitters):
            chunks = self.text_splitters[level].split_text(content)
        else:
            chunks = self.text_splitters[level](content)
        
        child_ids = []
        
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:  # 너무 짧은 청크 제외
                continue
            
            child_id = f"{parent_id}_L{level}_{i}"
            
            # 메타데이터 설정
            child_metadata = base_metadata.copy()
            child_metadata.update({
                "level": level,
                "parent_id": parent_id,
                "chunk_index": i,
                "chunk_total": len(chunks)
            })
            
            # 자식 문서 생성
            child_doc = HierarchicalDocument(
                id=child_id,
                level=level,
                parent_id=parent_id,
                content=chunk,
                summary=None,
                metadata=child_metadata,
                children_ids=[]
            )
            
            # 요약 생성 (선택적, 특정 레벨에서만)
            if self.enable_summarization and level == 1:  # 섹션 레벨에서만
                child_doc.summary = self._generate_summary(chunk)
            
            self.document_hierarchy[child_id] = child_doc
            self.level_indices[level].append(child_id)
            child_ids.append(child_id)
        
        return child_ids
    
    def _generate_summary(self, text: str) -> str:
        """텍스트 요약 생성"""
        
        if len(text) < 200:  # 짧은 텍스트는 요약하지 않음
            return text
        
        try:
            prompt = self.summary_prompt.format(
                text=text,
                compression_ratio=self.summary_compression_ratio
            )
            
            if hasattr(self.llm, 'generate'):
                response = self.llm.generate(prompt)
                summary = response.content
            else:
                summary = str(self.llm(prompt))
            
            return summary.strip()
            
        except Exception as e:
            logger.warning(f"요약 생성 실패: {e}")
            # 간단한 요약 (첫 몇 문장)
            sentences = text.split('. ')
            max_sentences = max(1, int(len(sentences) * self.summary_compression_ratio))
            return '. '.join(sentences[:max_sentences]) + '.'
    
    def _add_to_vector_store(self, doc_id: str):
        """벡터 저장소에 문서 추가"""
        
        try:
            doc = self.document_hierarchy[doc_id]
            
            # 모든 레벨의 문서를 벡터 저장소에 추가
            self._add_doc_and_children_to_vector_store(doc)
            
        except Exception as e:
            logger.error(f"벡터 저장소 추가 오류 ({doc_id}): {e}")
    
    def _add_doc_and_children_to_vector_store(self, doc: HierarchicalDocument):
        """문서와 자식들을 벡터 저장소에 재귀적으로 추가"""
        
        try:
            # 현재 문서 추가
            if hasattr(self.vector_store, 'add_documents'):
                from ....core.data_processing.vector_store import VectorDocument
                
                # 요약이 있으면 요약도 함께 저장
                content_to_embed = doc.content
                if doc.summary:
                    content_to_embed = f"{doc.summary}\n\n{doc.content}"
                
                embedding = self.embeddings.embed_text(content_to_embed)
                
                vector_doc = VectorDocument(
                    id=doc.id,
                    content=content_to_embed,
                    embedding=embedding,
                    metadata=doc.metadata
                )
                
                self.vector_store.add_documents([vector_doc])
            
            # 자식 문서들도 추가
            for child_id in doc.children_ids:
                if child_id in self.document_hierarchy:
                    child_doc = self.document_hierarchy[child_id]
                    self._add_doc_and_children_to_vector_store(child_doc)
                    
        except Exception as e:
            logger.warning(f"문서 {doc.id} 벡터 저장소 추가 실패: {e}")
    
    def search(
        self,
        question: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        target_level: Optional[int] = None
    ) -> Dict[str, Any]:
        """계층형 검색 수행"""
        
        try:
            if self.search_strategy == "hierarchical":
                return self._hierarchical_search(question, k, filters, target_level)
            elif self.search_strategy == "flat":
                return self._flat_search(question, k, filters, target_level)
            elif self.search_strategy == "adaptive":
                return self._adaptive_search(question, k, filters)
            else:
                return self._hierarchical_search(question, k, filters, target_level)
                
        except Exception as e:
            logger.error(f"계층형 검색 오류: {e}")
            return {
                "question": question,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _hierarchical_search(
        self,
        question: str,
        k: int,
        filters: Optional[Dict[str, Any]],
        target_level: Optional[int]
    ) -> Dict[str, Any]:
        """계층형 검색 전략"""
        
        results_by_level = {}
        
        # 지정된 레벨이 있으면 해당 레벨만, 없으면 모든 레벨
        levels_to_search = [target_level] if target_level is not None else range(self.max_levels)
        
        for level in levels_to_search:
            level_results = self._search_at_level(question, k, filters, level)
            if level_results:
                results_by_level[level] = level_results
        
        # 결과 통합 및 랭킹
        integrated_results = self._integrate_hierarchical_results(results_by_level, k)
        
        return {
            "question": question,
            "search_strategy": "hierarchical",
            "results_by_level": results_by_level if self.return_hierarchy_info else None,
            "documents": integrated_results,
            "total_documents": len(integrated_results),
            "timestamp": datetime.now().isoformat()
        }
    
    def _search_at_level(
        self,
        question: str,
        k: int,
        filters: Optional[Dict[str, Any]],
        level: int
    ) -> List[Dict[str, Any]]:
        """특정 레벨에서 검색"""
        
        try:
            # 레벨 필터 추가
            level_filters = filters.copy() if filters else {}
            level_filters["level"] = level
            
            # 벡터 검색
            if hasattr(self.vector_store, 'search'):
                query_embedding = self.embeddings.embed_text(question)
                search_result = self.vector_store.search(
                    query_embedding, k=k, filters=level_filters
                )
                
                results = []
                for doc in search_result.documents:
                    doc_info = {
                        "id": doc.id,
                        "content": doc.content,
                        "metadata": doc.metadata,
                        "score": getattr(doc, 'score', 0.0),
                        "level": level
                    }
                    
                    # 계층 정보 추가
                    if doc.id in self.document_hierarchy:
                        hier_doc = self.document_hierarchy[doc.id]
                        doc_info["hierarchy"] = {
                            "parent_id": hier_doc.parent_id,
                            "children_ids": hier_doc.children_ids,
                            "summary": hier_doc.summary
                        }
                    
                    results.append(doc_info)
                
                return results
            
            return []
            
        except Exception as e:
            logger.warning(f"레벨 {level} 검색 실패: {e}")
            return []
    
    def _integrate_hierarchical_results(
        self,
        results_by_level: Dict[int, List[Dict[str, Any]]],
        k: int
    ) -> List[Dict[str, Any]]:
        """계층형 결과 통합"""
        
        # 레벨별 가중치 (상위 레벨일수록 높은 가중치)
        level_weights = {0: 1.0, 1: 0.8, 2: 0.6}
        
        all_results = []
        
        for level, results in results_by_level.items():
            weight = level_weights.get(level, 0.5)
            
            for result in results:
                # 스코어에 레벨 가중치 적용
                result["weighted_score"] = result.get("score", 0.0) * weight
                result["level_weight"] = weight
                all_results.append(result)
        
        # 가중 스코어로 정렬
        all_results.sort(key=lambda x: x["weighted_score"], reverse=True)
        
        # 중복 제거 (상위 레벨 우선)
        unique_results = []
        seen_content_hashes = set()
        
        for result in all_results:
            content_hash = hash(result["content"][:200])
            if content_hash not in seen_content_hashes:
                seen_content_hashes.add(content_hash)
                unique_results.append(result)
        
        return unique_results[:k]
    
    def _flat_search(
        self,
        question: str,
        k: int,
        filters: Optional[Dict[str, Any]],
        target_level: Optional[int]
    ) -> Dict[str, Any]:
        """평면적 검색 전략 (모든 레벨을 동등하게)"""
        
        try:
            search_filters = filters.copy() if filters else {}
            if target_level is not None:
                search_filters["level"] = target_level
            
            if hasattr(self.vector_store, 'search'):
                query_embedding = self.embeddings.embed_text(question)
                search_result = self.vector_store.search(
                    query_embedding, k=k, filters=search_filters
                )
                
                results = []
                for doc in search_result.documents:
                    doc_info = {
                        "id": doc.id,
                        "content": doc.content,
                        "metadata": doc.metadata,
                        "score": getattr(doc, 'score', 0.0)
                    }
                    
                    # 계층 정보 추가
                    if doc.id in self.document_hierarchy:
                        hier_doc = self.document_hierarchy[doc.id]
                        doc_info["level"] = hier_doc.level
                        doc_info["hierarchy"] = {
                            "parent_id": hier_doc.parent_id,
                            "children_ids": hier_doc.children_ids,
                            "summary": hier_doc.summary
                        }
                    
                    results.append(doc_info)
                
                return {
                    "question": question,
                    "search_strategy": "flat",
                    "documents": results,
                    "total_documents": len(results),
                    "timestamp": datetime.now().isoformat()
                }
            
            return {"question": question, "documents": [], "total_documents": 0}
            
        except Exception as e:
            logger.error(f"평면적 검색 오류: {e}")
            return {"question": question, "error": str(e)}
    
    def _adaptive_search(
        self,
        question: str,
        k: int,
        filters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """적응형 검색 전략 (질문 복잡도에 따라 레벨 선택)"""
        
        try:
            # 질문 복잡도 분석
            complexity = self._analyze_question_complexity(question)
            
            if complexity > 0.7:
                # 복잡한 질문: 상위 레벨부터 검색
                target_levels = [0, 1]
            elif complexity > 0.3:
                # 중간 복잡도: 중간 레벨 위주
                target_levels = [1, 2]
            else:
                # 단순한 질문: 하위 레벨에서 세부사항 검색
                target_levels = [2, 1]
            
            results_by_level = {}
            
            for level in target_levels:
                level_results = self._search_at_level(question, k//len(target_levels), filters, level)
                if level_results:
                    results_by_level[level] = level_results
            
            # 결과 통합
            integrated_results = self._integrate_hierarchical_results(results_by_level, k)
            
            return {
                "question": question,
                "search_strategy": "adaptive",
                "question_complexity": complexity,
                "target_levels": target_levels,
                "results_by_level": results_by_level if self.return_hierarchy_info else None,
                "documents": integrated_results,
                "total_documents": len(integrated_results),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"적응형 검색 오류: {e}")
            return {"question": question, "error": str(e)}
    
    def _analyze_question_complexity(self, question: str) -> float:
        """질문 복잡도 분석 (0.0 ~ 1.0)"""
        
        complexity_score = 0.0
        
        # 길이 기준
        if len(question) > 100:
            complexity_score += 0.2
        elif len(question) > 50:
            complexity_score += 0.1
        
        # 키워드 기준
        complex_keywords = [
            "왜", "어떻게", "비교", "분석", "설명", "차이", "관계", "영향",
            "why", "how", "compare", "analyze", "explain", "difference", "relationship", "impact"
        ]
        
        for keyword in complex_keywords:
            if keyword in question.lower():
                complexity_score += 0.15
        
        # 질문 개수 (복합 질문)
        question_marks = question.count("?")
        if question_marks > 1:
            complexity_score += 0.2
        
        return min(complexity_score, 1.0)
    
    def ask(
        self,
        question: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        target_level: Optional[int] = None
    ) -> Dict[str, Any]:
        """질문에 대한 답변 생성 (계층형 검색 + 답변 생성)"""
        
        # 1. 계층형 검색
        search_result = self.search(question, k, filters, target_level)
        
        if "error" in search_result:
            return search_result
        
        try:
            # 2. 검색된 문서들로부터 답변 생성
            documents = search_result["documents"]
            
            # 계층 정보 준비
            hierarchy_info = self._prepare_hierarchy_info(documents)
            
            # 컨텍스트 준비
            contexts = []
            for doc in documents:
                content = doc["content"]
                level = doc.get("level", 0)
                hierarchy = doc.get("hierarchy", {})
                
                # 요약이 있으면 포함
                if hierarchy.get("summary"):
                    content = f"[요약] {hierarchy['summary']}\n\n[상세] {content}"
                
                contexts.append(f"[Level {level}] {content}")
            
            combined_context = "\n\n".join(contexts)
            
            # 프롬프트 생성
            prompt = self.hierarchical_prompt.format(
                hierarchy_info=hierarchy_info,
                context=combined_context,
                question=question
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
                "search_strategy": search_result.get("search_strategy", "hierarchical"),
                "timestamp": datetime.now().isoformat()
            }
            
            if self.return_source_documents:
                result["source_documents"] = documents
            
            if self.return_hierarchy_info:
                result["hierarchy_info"] = hierarchy_info
                if "results_by_level" in search_result:
                    result["results_by_level"] = search_result["results_by_level"]
            
            return result
            
        except Exception as e:
            logger.error(f"계층형 답변 생성 오류: {e}")
            return {
                "question": question,
                "answer": "답변 생성 중 오류가 발생했습니다.",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _prepare_hierarchy_info(self, documents: List[Dict[str, Any]]) -> str:
        """계층 정보를 텍스트로 준비"""
        
        hierarchy_lines = []
        
        for doc in documents:
            doc_id = doc["id"]
            level = doc.get("level", 0)
            hierarchy = doc.get("hierarchy", {})
            
            parent_id = hierarchy.get("parent_id", "None")
            children_count = len(hierarchy.get("children_ids", []))
            
            hierarchy_lines.append(
                f"- Document {doc_id} (Level {level}): Parent={parent_id}, Children={children_count}"
            )
        
        return "\n".join(hierarchy_lines)
    
    def get_document_hierarchy(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """문서의 계층 구조 정보 반환"""
        
        if doc_id not in self.document_hierarchy:
            return None
        
        doc = self.document_hierarchy[doc_id]
        
        hierarchy_info = {
            "id": doc.id,
            "level": doc.level,
            "parent_id": doc.parent_id,
            "children_ids": doc.children_ids,
            "metadata": doc.metadata,
            "has_summary": doc.summary is not None
        }
        
        # 부모 정보
        if doc.parent_id and doc.parent_id in self.document_hierarchy:
            parent_doc = self.document_hierarchy[doc.parent_id]
            hierarchy_info["parent_info"] = {
                "id": parent_doc.id,
                "level": parent_doc.level,
                "summary": parent_doc.summary
            }
        
        # 자식 정보
        children_info = []
        for child_id in doc.children_ids:
            if child_id in self.document_hierarchy:
                child_doc = self.document_hierarchy[child_id]
                children_info.append({
                    "id": child_doc.id,
                    "level": child_doc.level,
                    "summary": child_doc.summary
                })
        
        hierarchy_info["children_info"] = children_info
        
        return hierarchy_info
    
    def get_config(self) -> Dict[str, Any]:
        """현재 설정 반환"""
        
        return {
            "language": self.language,
            "max_levels": self.max_levels,
            "chunk_sizes": self.chunk_sizes,
            "chunk_overlaps": self.chunk_overlaps,
            "enable_summarization": self.enable_summarization,
            "summary_compression_ratio": self.summary_compression_ratio,
            "search_strategy": self.search_strategy,
            "use_langchain": self.use_langchain,
            "has_langchain": HAS_LANGCHAIN,
            "return_source_documents": self.return_source_documents,
            "return_hierarchy_info": self.return_hierarchy_info,
            "total_documents": len(self.document_hierarchy),
            "documents_by_level": {level: len(docs) for level, docs in self.level_indices.items()},
            "llm_type": type(self.llm).__name__,
            "embeddings_type": type(self.embeddings).__name__,
            "vector_store_type": type(self.vector_store).__name__
        }