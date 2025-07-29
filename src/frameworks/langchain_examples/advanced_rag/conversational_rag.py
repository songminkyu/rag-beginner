"""
Conversational RAG with LangChain
LangChain을 사용한 대화형 RAG 시스템
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# LangChain imports (optional)
try:
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryBufferMemory
    from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
    from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
    from langchain.callbacks.manager import CallbackManagerForChainRun
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

logger = logging.getLogger(__name__)


class ConversationalRAG:
    """LangChain을 사용한 고급 대화형 RAG 시스템"""
    
    def __init__(
        self,
        llm,
        embeddings,
        vector_store,
        config: Optional[Dict[str, Any]] = None
    ):
        self.config = config or {}
        self.language = self.config.get("language", "korean")
        self.memory_type = self.config.get("memory_type", "window")  # window, summary
        self.window_size = self.config.get("window_size", 5)
        self.max_token_limit = self.config.get("max_token_limit", 2000)
        self.enable_context_compression = self.config.get("enable_context_compression", True)
        self.enable_chat_history_compression = self.config.get("enable_chat_history_compression", True)
        self.return_source_documents = self.config.get("return_source_documents", True)
        
        # LangChain 사용 여부
        self.use_langchain = HAS_LANGCHAIN and self.config.get("use_langchain", True)
        
        # 대화 세션 관리
        self.sessions = {}  # session_id -> conversation data
        self.current_session_id = None
        
        if self.use_langchain and HAS_LANGCHAIN:
            self._setup_langchain_conversational_rag(llm, embeddings, vector_store)
        else:
            self._setup_custom_conversational_rag(llm, embeddings, vector_store)
        
        logger.info(f"ConversationalRAG 초기화: LangChain={self.use_langchain}, Memory={self.memory_type}")
    
    def _setup_langchain_conversational_rag(self, llm, embeddings, vector_store):
        """LangChain을 사용한 대화형 RAG 설정"""
        
        self.llm = llm
        self.embeddings = embeddings
        self.vector_store = vector_store
        
        # 프롬프트 템플릿 설정
        self._setup_prompts()
        
        logger.info("LangChain ConversationalRAG 설정 완료")
    
    def _setup_prompts(self):
        """프롬프트 템플릿 설정"""
        
        if self.language == "korean":
            # 시스템 메시지
            system_template = """당신은 도움이 되는 AI 어시스턴트입니다. 다음 지침을 따라주세요:

1. 주어진 컨텍스트와 대화 기록을 활용하여 정확하고 도움이 되는 답변을 제공하세요
2. 컨텍스트에 없는 정보는 추측하지 마세요
3. 이전 대화 내용을 참고하여 일관성 있는 응답을 하세요
4. 사용자의 의도를 파악하고 맥락에 맞는 답변을 제공하세요
5. 필요시 추가 질문을 통해 명확화를 요청하세요

컨텍스트: {context}"""
            
            # 사용자 메시지
            human_template = """대화 기록:
{chat_history}

현재 질문: {question}

답변:"""
            
        else:
            # English prompts
            system_template = """You are a helpful AI assistant. Please follow these guidelines:

1. Provide accurate and helpful answers using the given context and chat history
2. Don't guess information that's not in the context
3. Maintain consistency with previous conversation
4. Understand user intent and provide contextually appropriate responses
5. Ask clarifying questions when needed

Context: {context}"""
            
            human_template = """Chat History:
{chat_history}

Current Question: {question}

Answer:"""
        
        # 프롬프트 템플릿 생성
        self.system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        self.human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        
        self.chat_prompt = ChatPromptTemplate.from_messages([
            self.system_message_prompt,
            self.human_message_prompt
        ])
        
        # Standalone question prompt (대화 기록을 고려한 질문 재구성)
        if self.language == "korean":
            standalone_template = """대화 기록과 후속 질문이 주어졌을 때, 대화 기록 없이도 이해할 수 있는 독립적인 질문으로 재구성해주세요.

대화 기록:
{chat_history}

후속 질문: {question}

독립적인 질문:"""
        else:
            standalone_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}

Follow Up Input: {question}

Standalone question:"""
        
        self.standalone_question_prompt = PromptTemplate(
            template=standalone_template,
            input_variables=["chat_history", "question"]
        )
    
    def _setup_custom_conversational_rag(self, llm, embeddings, vector_store):
        """커스텀 대화형 RAG 설정 (LangChain 없이)"""
        
        self.llm = llm
        self.embeddings = embeddings
        self.vector_store = vector_store
        
        logger.info("커스텀 ConversationalRAG 설정 완료")
    
    def start_conversation(self, session_id: Optional[str] = None) -> str:
        """새로운 대화 세션 시작"""
        
        if session_id is None:
            session_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_session_id = session_id
        
        # 세션 초기화
        if self.use_langchain and HAS_LANGCHAIN:
            if self.memory_type == "summary":
                memory = ConversationSummaryBufferMemory(
                    llm=self.llm,
                    max_token_limit=self.max_token_limit,
                    memory_key="chat_history",
                    return_messages=True,
                    output_key="answer"
                )
            else:
                memory = ConversationBufferWindowMemory(
                    k=self.window_size,
                    memory_key="chat_history",
                    return_messages=True,
                    output_key="answer"
                )
            
            # ConversationalRetrievalChain 생성
            chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vector_store.as_retriever(),
                memory=memory,
                combine_docs_chain_kwargs={"prompt": self.chat_prompt},
                condense_question_prompt=self.standalone_question_prompt,
                return_source_documents=self.return_source_documents,
                verbose=False
            )
            
            self.sessions[session_id] = {
                "chain": chain,
                "memory": memory,
                "start_time": datetime.now(),
                "message_count": 0
            }
        else:
            # 커스텀 메모리 구현
            self.sessions[session_id] = {
                "chat_history": [],
                "start_time": datetime.now(),
                "message_count": 0,
                "max_history": self.window_size * 2  # user + assistant pairs
            }
        
        logger.info(f"대화 세션 시작: {session_id}")
        return session_id
    
    def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """대화 메시지 처리"""
        
        # 세션 확인 및 설정
        if session_id is None:
            if self.current_session_id is None:
                session_id = self.start_conversation()
            else:
                session_id = self.current_session_id
        
        if session_id not in self.sessions:
            session_id = self.start_conversation(session_id)
        
        self.current_session_id = session_id
        
        if self.use_langchain and HAS_LANGCHAIN:
            return self._chat_langchain(message, session_id, k, filters)
        else:
            return self._chat_custom(message, session_id, k, filters)
    
    def _chat_langchain(
        self,
        message: str,
        session_id: str,
        k: int,
        filters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """LangChain을 사용한 대화"""
        
        try:
            session = self.sessions[session_id]
            chain = session["chain"]
            
            # 검색 설정
            chain.retriever.search_kwargs = {"k": k}
            if filters:
                chain.retriever.search_kwargs["filters"] = filters
            
            # 대화 실행
            result = chain({"question": message})
            
            # 세션 정보 업데이트
            session["message_count"] += 1
            
            response = {
                "session_id": session_id,
                "question": message,
                "answer": result["answer"],
                "message_count": session["message_count"],
                "timestamp": datetime.now().isoformat(),
                "method": "langchain"
            }
            
            if self.return_source_documents and "source_documents" in result:
                response["source_documents"] = [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in result["source_documents"]
                ]
            
            return response
            
        except Exception as e:
            logger.error(f"LangChain 대화 오류 (세션 {session_id}): {e}")
            return {
                "session_id": session_id,
                "question": message,
                "answer": "대화 처리 중 오류가 발생했습니다.",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "method": "langchain"
            }
    
    def _chat_custom(
        self,
        message: str,
        session_id: str,
        k: int,
        filters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """커스텀 대화 (LangChain 없이)"""
        
        try:
            session = self.sessions[session_id]
            chat_history = session["chat_history"]
            
            # 1. Standalone question 생성 (대화 기록 고려)
            standalone_question = self._generate_standalone_question(message, chat_history)
            
            # 2. 문서 검색
            if hasattr(self.vector_store, 'search'):
                query_embedding = self.embeddings.embed_text(standalone_question)
                search_result = self.vector_store.search(
                    query_embedding, k=k, filters=filters
                )
                retrieved_docs = search_result.documents
            else:
                retrieved_docs = []
            
            # 3. 컨텍스트 준비
            contexts = [doc.content for doc in retrieved_docs]
            combined_context = "\n\n".join(contexts)
            
            # 4. 대화 히스토리 준비
            history_text = self._format_chat_history(chat_history)
            
            # 5. 프롬프트 생성 및 답변 생성
            if self.language == "korean":
                prompt = f"""다음 컨텍스트와 대화 기록을 바탕으로 질문에 답변해주세요.

컨텍스트:
{combined_context}

대화 기록:
{history_text}

현재 질문: {message}

답변:"""
            else:
                prompt = f"""Answer the question based on the context and chat history.

Context:
{combined_context}

Chat History:
{history_text}

Current Question: {message}

Answer:"""
            
            # 6. 답변 생성
            if hasattr(self.llm, 'generate'):
                response = self.llm.generate(prompt)
                answer = response.content
            else:
                answer = str(self.llm(prompt))
            
            # 7. 메모리 업데이트
            chat_history.append({
                "role": "user",
                "content": message,
                "timestamp": datetime.now().isoformat()
            })
            chat_history.append({
                "role": "assistant",
                "content": answer,
                "timestamp": datetime.now().isoformat()
            })
            
            # 메모리 크기 제한
            if len(chat_history) > session["max_history"]:
                chat_history[:] = chat_history[-session["max_history"]:]
            
            # 세션 정보 업데이트
            session["message_count"] += 1
            
            response = {
                "session_id": session_id,
                "question": message,
                "answer": answer,
                "standalone_question": standalone_question,
                "message_count": session["message_count"],
                "timestamp": datetime.now().isoformat(),
                "method": "custom"
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
            logger.error(f"커스텀 대화 오류 (세션 {session_id}): {e}")
            return {
                "session_id": session_id,
                "question": message,
                "answer": "대화 처리 중 오류가 발생했습니다.",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "method": "custom"
            }
    
    def _generate_standalone_question(self, question: str, chat_history: List[Dict]) -> str:
        """대화 기록을 고려한 독립적인 질문 생성"""
        
        if not chat_history:
            return question
        
        # 최근 대화 기록만 사용 (토큰 제한)
        recent_history = chat_history[-6:]  # 최근 3턴
        history_text = self._format_chat_history(recent_history)
        
        if self.language == "korean":
            prompt = f"""대화 기록과 후속 질문이 주어졌을 때, 대화 기록 없이도 이해할 수 있는 독립적인 질문으로 재구성해주세요.

대화 기록:
{history_text}

후속 질문: {question}

독립적인 질문:"""
        else:
            prompt = f"""Given the conversation history and a follow-up question, rephrase it as a standalone question.

Chat History:
{history_text}

Follow Up Question: {question}

Standalone Question:"""
        
        try:
            if hasattr(self.llm, 'generate'):
                response = self.llm.generate(prompt)
                standalone_question = response.content.strip()
            else:
                standalone_question = str(self.llm(prompt)).strip()
            
            # 생성된 질문이 유효한지 확인
            if len(standalone_question) > 10 and standalone_question != question:
                return standalone_question
            else:
                return question
                
        except Exception as e:
            logger.warning(f"Standalone question 생성 실패: {e}")
            return question
    
    def _format_chat_history(self, chat_history: List[Dict]) -> str:
        """대화 기록을 텍스트로 포맷팅"""
        
        if not chat_history:
            return ""
        
        formatted_history = []
        for item in chat_history:
            role = "사용자" if item["role"] == "user" else "어시스턴트"
            if self.language != "korean":
                role = "User" if item["role"] == "user" else "Assistant"
            
            formatted_history.append(f"{role}: {item['content']}")
        
        return "\n".join(formatted_history)
    
    def get_conversation_history(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """대화 기록 반환"""
        
        if session_id is None:
            session_id = self.current_session_id
        
        if session_id not in self.sessions:
            return []
        
        session = self.sessions[session_id]
        
        if self.use_langchain and HAS_LANGCHAIN:
            memory = session["memory"]
            messages = memory.chat_memory.messages
            
            history = []
            for message in messages:
                if isinstance(message, HumanMessage):
                    history.append({
                        "role": "user",
                        "content": message.content,
                        "timestamp": datetime.now().isoformat()
                    })
                elif isinstance(message, AIMessage):
                    history.append({
                        "role": "assistant",
                        "content": message.content,
                        "timestamp": datetime.now().isoformat()
                    })
            
            return history
        else:
            return session["chat_history"]
    
    def end_conversation(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """대화 세션 종료"""
        
        if session_id is None:
            session_id = self.current_session_id
        
        if session_id not in self.sessions:
            return {"error": "세션을 찾을 수 없습니다."}
        
        session = self.sessions[session_id]
        
        # 세션 요약 정보
        summary = {
            "session_id": session_id,
            "start_time": session["start_time"].isoformat(),
            "end_time": datetime.now().isoformat(),
            "message_count": session["message_count"],
            "duration_minutes": (datetime.now() - session["start_time"]).total_seconds() / 60
        }
        
        # 세션 삭제
        del self.sessions[session_id]
        
        if self.current_session_id == session_id:
            self.current_session_id = None
        
        logger.info(f"대화 세션 종료: {session_id}")
        return summary
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """활성 세션 목록 반환"""
        
        sessions_info = []
        for session_id, session in self.sessions.items():
            sessions_info.append({
                "session_id": session_id,
                "start_time": session["start_time"].isoformat(),
                "message_count": session["message_count"],
                "is_current": session_id == self.current_session_id
            })
        
        return sessions_info
    
    def switch_session(self, session_id: str) -> bool:
        """세션 전환"""
        
        if session_id in self.sessions:
            self.current_session_id = session_id
            logger.info(f"세션 전환: {session_id}")
            return True
        else:
            logger.warning(f"존재하지 않는 세션: {session_id}")
            return False
    
    def save_conversation(self, session_id: Optional[str] = None, filepath: Optional[str] = None):
        """대화 내용을 파일로 저장"""
        
        import json
        
        if session_id is None:
            session_id = self.current_session_id
        
        if session_id not in self.sessions:
            raise ValueError("세션을 찾을 수 없습니다.")
        
        if filepath is None:
            filepath = f"conversation_{session_id}.json"
        
        session = self.sessions[session_id]
        history = self.get_conversation_history(session_id)
        
        conversation_data = {
            "session_id": session_id,
            "start_time": session["start_time"].isoformat(),
            "message_count": session["message_count"],
            "language": self.language,
            "memory_type": self.memory_type,
            "chat_history": history,
            "saved_at": datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"대화 내용 저장: {filepath}")
    
    def get_config(self) -> Dict[str, Any]:
        """현재 설정 반환"""
        
        return {
            "language": self.language,
            "memory_type": self.memory_type,
            "window_size": self.window_size,
            "max_token_limit": self.max_token_limit,
            "use_langchain": self.use_langchain,
            "has_langchain": HAS_LANGCHAIN,
            "enable_context_compression": self.enable_context_compression,
            "enable_chat_history_compression": self.enable_chat_history_compression,
            "return_source_documents": self.return_source_documents,
            "active_sessions": len(self.sessions),
            "current_session_id": self.current_session_id,
            "llm_type": type(self.llm).__name__,
            "embeddings_type": type(self.embeddings).__name__,
            "vector_store_type": type(self.vector_store).__name__
        }