"""
Document Chat with LangChain
LangChain을 사용한 문서 기반 채팅 시스템
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# LangChain imports (optional)
try:
    from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
    from langchain.prompts import PromptTemplate
    from langchain.schema import BaseMessage, HumanMessage, AIMessage
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

logger = logging.getLogger(__name__)


class DocumentChat:
    """LangChain을 사용한 문서 기반 채팅 시스템"""
    
    def __init__(
        self,
        llm,
        embeddings,
        vector_store,
        config: Optional[Dict[str, Any]] = None
    ):
        self.config = config or {}
        self.language = self.config.get("language", "korean")
        self.memory_type = self.config.get("memory_type", "buffer")  # buffer, summary
        self.max_token_limit = self.config.get("max_token_limit", 4000)
        self.return_source_documents = self.config.get("return_source_documents", True)
        
        # LangChain 사용 여부
        self.use_langchain = HAS_LANGCHAIN and self.config.get("use_langchain", True)
        
        # 채팅 히스토리
        self.chat_history = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.use_langchain and HAS_LANGCHAIN:
            self._setup_langchain_chat(llm, embeddings, vector_store)
        else:
            self._setup_custom_chat(llm, embeddings, vector_store)
        
        logger.info(f"DocumentChat 초기화: LangChain={self.use_langchain}, Memory={self.memory_type}")
    
    def _setup_langchain_chat(self, llm, embeddings, vector_store):
        """LangChain을 사용한 채팅 설정"""
        
        self.llm = llm
        self.embeddings = embeddings
        self.vector_store = vector_store
        
        # 메모리 설정
        if self.memory_type == "summary":
            self.memory = ConversationSummaryBufferMemory(
                llm=self.llm,
                max_token_limit=self.max_token_limit,
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
        else:
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
        
        # 프롬프트 템플릿
        if self.language == "korean":
            template = """당신은 도움이 되는 AI 어시스턴트입니다. 주어진 컨텍스트와 대화 기록을 바탕으로 질문에 답변해주세요.

컨텍스트: {context}

대화 기록:
{chat_history}

질문: {question}

답변시 다음 사항을 고려해주세요:
1. 컨텍스트의 정보를 우선적으로 활용하세요
2. 이전 대화 내용을 참고하여 일관성 있게 답변하세요
3. 확실하지 않은 정보는 추측하지 마세요

답변:"""
        else:
            template = """You are a helpful AI assistant. Answer the question based on the given context and chat history.

Context: {context}

Chat History:
{chat_history}

Question: {question}

Please consider the following when answering:
1. Prioritize information from the context
2. Maintain consistency with previous conversation
3. Don't guess if you're not certain

Answer:"""
        
        # ConversationalRetrievalChain 생성
        self.chat_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(),
            memory=self.memory,
            return_source_documents=self.return_source_documents,
            verbose=False
        )
        
        logger.info("LangChain ConversationalRetrievalChain 설정 완료")
    
    def _setup_custom_chat(self, llm, embeddings, vector_store):
        """커스텀 채팅 설정 (LangChain 없이)"""
        
        self.llm = llm
        self.embeddings = embeddings
        self.vector_store = vector_store
        
        # 간단한 메모리 구현
        self.memory = {
            "chat_history": [],
            "max_history": self.config.get("max_history", 10)
        }
        
        logger.info("커스텀 채팅 설정 완료")
    
    def chat(
        self,
        message: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """채팅 메시지 처리"""
        
        if self.use_langchain and HAS_LANGCHAIN:
            return self._chat_langchain(message, k, filters)
        else:
            return self._chat_custom(message, k, filters)
    
    def _chat_langchain(
        self,
        message: str,
        k: int,
        filters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """LangChain을 사용한 채팅"""
        
        try:
            # 검색 설정
            self.chat_chain.retriever.search_kwargs = {"k": k}
            if filters:
                self.chat_chain.retriever.search_kwargs["filters"] = filters
            
            # 채팅 실행
            result = self.chat_chain({
                "question": message,
                "chat_history": self.chat_history
            })
            
            # 채팅 히스토리 업데이트
            self.chat_history.extend([
                HumanMessage(content=message),
                AIMessage(content=result["answer"])
            ])
            
            response = {
                "question": message,
                "answer": result["answer"],
                "session_id": self.session_id,
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
            logger.error(f"LangChain 채팅 오류: {e}")
            return {
                "question": message,
                "answer": "채팅 처리 중 오류가 발생했습니다.",
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "method": "langchain"
            }
    
    def _chat_custom(
        self,
        message: str,
        k: int,
        filters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """커스텀 채팅 (LangChain 없이)"""
        
        try:
            # 1. 문서 검색
            if hasattr(self.vector_store, 'search'):
                query_embedding = self.embeddings.embed_text(message)
                search_result = self.vector_store.search(
                    query_embedding, k=k, filters=filters
                )
                retrieved_docs = search_result.documents
            else:
                retrieved_docs = []
            
            # 2. 컨텍스트 준비
            contexts = [doc.content for doc in retrieved_docs]
            combined_context = "\n\n".join(contexts)
            
            # 3. 대화 히스토리 준비
            history_text = ""
            if self.memory["chat_history"]:
                history_items = []
                for item in self.memory["chat_history"][-6:]:  # 최근 6개만
                    role = "사용자" if self.language == "korean" else "User"
                    assistant = "어시스턴트" if self.language == "korean" else "Assistant"
                    
                    if item["role"] == "user":
                        history_items.append(f"{role}: {item['content']}")
                    else:
                        history_items.append(f"{assistant}: {item['content']}")
                
                history_text = "\n".join(history_items)
            
            # 4. 프롬프트 생성
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
            
            # 5. 답변 생성
            if hasattr(self.llm, 'generate'):
                response = self.llm.generate(prompt)
                answer = response.content
            else:
                answer = str(self.llm(prompt))
            
            # 6. 메모리 업데이트
            self.memory["chat_history"].append({
                "role": "user",
                "content": message,
                "timestamp": datetime.now().isoformat()
            })
            self.memory["chat_history"].append({
                "role": "assistant", 
                "content": answer,
                "timestamp": datetime.now().isoformat()
            })
            
            # 메모리 크기 제한
            if len(self.memory["chat_history"]) > self.memory["max_history"] * 2:
                self.memory["chat_history"] = self.memory["chat_history"][-self.memory["max_history"]*2:]
            
            response = {
                "question": message,
                "answer": answer,
                "session_id": self.session_id,
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
            logger.error(f"커스텀 채팅 오류: {e}")
            return {
                "question": message,
                "answer": "채팅 처리 중 오류가 발생했습니다.",
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "method": "custom"
            }
    
    def get_chat_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """채팅 히스토리 반환"""
        
        if self.use_langchain and HAS_LANGCHAIN:
            # LangChain 메모리에서 가져오기
            history = []
            for message in self.chat_history:
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
        else:
            # 커스텀 메모리에서 가져오기
            history = self.memory["chat_history"]
        
        if limit:
            return history[-limit:]
        return history
    
    def clear_history(self):
        """채팅 히스토리 초기화"""
        
        if self.use_langchain and HAS_LANGCHAIN:
            self.memory.clear()
            self.chat_history = []
        else:
            self.memory["chat_history"] = []
        
        # 새 세션 ID 생성
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info("채팅 히스토리 초기화 완료")
    
    def save_conversation(self, filepath: str):
        """대화 내용을 파일로 저장"""
        
        import json
        
        conversation_data = {
            "session_id": self.session_id,
            "language": self.language,
            "memory_type": self.memory_type,
            "chat_history": self.get_chat_history(),
            "saved_at": datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"대화 내용 저장: {filepath}")
    
    def load_conversation(self, filepath: str):
        """파일에서 대화 내용 로드"""
        
        import json
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                conversation_data = json.load(f)
            
            self.session_id = conversation_data.get("session_id", self.session_id)
            loaded_history = conversation_data.get("chat_history", [])
            
            if self.use_langchain and HAS_LANGCHAIN:
                # LangChain 메모리에 로드
                self.chat_history = []
                for item in loaded_history:
                    if item["role"] == "user":
                        self.chat_history.append(HumanMessage(content=item["content"]))
                    else:
                        self.chat_history.append(AIMessage(content=item["content"]))
            else:
                # 커스텀 메모리에 로드
                self.memory["chat_history"] = loaded_history
            
            logger.info(f"대화 내용 로드 완료: {len(loaded_history)}개 메시지")
            
        except Exception as e:
            logger.error(f"대화 내용 로드 오류: {e}")
    
    def get_conversation_summary(self) -> str:
        """대화 요약 생성"""
        
        history = self.get_chat_history()
        
        if not history:
            return "대화 내용이 없습니다." if self.language == "korean" else "No conversation history."
        
        # 간단한 요약 생성
        user_messages = [msg for msg in history if msg["role"] == "user"]
        assistant_messages = [msg for msg in history if msg["role"] == "assistant"]
        
        if self.language == "korean":
            summary = f"""대화 요약:
- 세션 ID: {self.session_id}
- 전체 메시지: {len(history)}개
- 사용자 질문: {len(user_messages)}개
- 어시스턴트 응답: {len(assistant_messages)}개
- 최근 질문: {user_messages[-1]['content'][:100] if user_messages else '없음'}...
"""
        else:
            summary = f"""Conversation Summary:
- Session ID: {self.session_id}
- Total messages: {len(history)}
- User questions: {len(user_messages)}
- Assistant responses: {len(assistant_messages)}
- Latest question: {user_messages[-1]['content'][:100] if user_messages else 'None'}...
"""
        
        return summary
    
    def get_config(self) -> Dict[str, Any]:
        """현재 설정 반환"""
        
        return {
            "session_id": self.session_id,
            "language": self.language,
            "memory_type": self.memory_type,
            "max_token_limit": self.max_token_limit,
            "use_langchain": self.use_langchain,
            "return_source_documents": self.return_source_documents,
            "chat_history_length": len(self.get_chat_history())
        }