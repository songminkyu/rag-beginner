"""
Agentic RAG with LangChain
LangChain을 사용한 에이전트 기반 RAG 시스템
"""

import logging
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

# LangChain imports (optional)
try:
    from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
    from langchain.agents.agent import AgentOutputParser
    from langchain.agents.conversational.prompt import FORMAT_INSTRUCTIONS
    from langchain.chains import LLMChain
    from langchain.prompts import StringPromptTemplate
    from langchain.schema import AgentAction, AgentFinish, OutputParserException
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """에이전트 액션 타입"""
    SEARCH = "search"
    SUMMARIZE = "summarize"
    COMPARE = "compare"
    ANALYZE = "analyze"
    SYNTHESIZE = "synthesize"
    REFLECT = "reflect"


@dataclass
class AgentStep:
    """에이전트 실행 단계"""
    step_id: int
    action_type: ActionType
    input: str
    output: str
    confidence: float
    reasoning: str
    timestamp: datetime


class AgenticRAG:
    """LangChain을 사용한 에이전트 기반 RAG 시스템"""
    
    def __init__(
        self,
        llm,
        embeddings,
        vector_store,
        config: Optional[Dict[str, Any]] = None
    ):
        self.config = config or {}
        self.language = self.config.get("language", "korean")
        self.max_iterations = self.config.get("max_iterations", 5)
        self.enable_reflection = self.config.get("enable_reflection", True)
        self.enable_tool_selection = self.config.get("enable_tool_selection", True)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        self.return_agent_steps = self.config.get("return_agent_steps", True)
        self.return_source_documents = self.config.get("return_source_documents", True)
        
        # LangChain 사용 여부
        self.use_langchain = HAS_LANGCHAIN and self.config.get("use_langchain", True)
        
        # 에이전트 실행 기록
        self.execution_history = []
        
        if self.use_langchain and HAS_LANGCHAIN:
            self._setup_langchain_agentic_rag(llm, embeddings, vector_store)
        else:
            self._setup_custom_agentic_rag(llm, embeddings, vector_store)
        
        logger.info(f"AgenticRAG 초기화: LangChain={self.use_langchain}, MaxIter={self.max_iterations}")
    
    def _setup_langchain_agentic_rag(self, llm, embeddings, vector_store):
        """LangChain을 사용한 에이전트 RAG 설정"""
        
        self.llm = llm
        self.embeddings = embeddings
        self.vector_store = vector_store
        
        # 도구 설정
        self.tools = self._create_tools()
        
        # 프롬프트 설정
        self.prompt_template = self._create_agent_prompt()
        
        # 에이전트 설정
        self._setup_agent()
        
        logger.info("LangChain AgenticRAG 설정 완료")
    
    def _create_tools(self) -> List[Tool]:
        """에이전트 도구 생성"""
        
        tools = []
        
        # 검색 도구
        def search_tool(query: str) -> str:
            try:
                query_embedding = self.embeddings.embed_text(query)
                search_result = self.vector_store.search(query_embedding, k=5)
                
                results = []
                for doc in search_result.documents:
                    results.append(f"Document: {doc.content[:200]}...")
                
                return "\n".join(results) if results else "No documents found."
                
            except Exception as e:
                return f"Search error: {str(e)}"
        
        tools.append(Tool(
            name="search",
            description="Search for relevant documents in the knowledge base",
            func=search_tool
        ))
        
        # 요약 도구
        def summarize_tool(text: str) -> str:
            try:
                if self.language == "korean":
                    prompt = f"다음 텍스트를 간단히 요약해주세요:\n\n{text}\n\n요약:"
                else:
                    prompt = f"Please summarize the following text:\n\n{text}\n\nSummary:"
                
                if hasattr(self.llm, 'generate'):
                    response = self.llm.generate(prompt)
                    return response.content
                else:
                    return str(self.llm(prompt))
                    
            except Exception as e:
                return f"Summarization error: {str(e)}"
        
        tools.append(Tool(
            name="summarize",
            description="Summarize long text or multiple documents",
            func=summarize_tool
        ))
        
        # 비교 도구
        def compare_tool(items: str) -> str:
            try:
                if self.language == "korean":
                    prompt = f"다음 항목들을 비교하고 차이점을 설명해주세요:\n\n{items}\n\n비교 분석:"
                else:
                    prompt = f"Compare the following items and explain the differences:\n\n{items}\n\nComparison:"
                
                if hasattr(self.llm, 'generate'):
                    response = self.llm.generate(prompt)
                    return response.content
                else:
                    return str(self.llm(prompt))
                    
            except Exception as e:
                return f"Comparison error: {str(e)}"
        
        tools.append(Tool(
            name="compare",
            description="Compare multiple items or concepts",
            func=compare_tool
        ))
        
        # 분석 도구
        def analyze_tool(content: str) -> str:
            try:
                if self.language == "korean":
                    prompt = f"다음 내용을 분석하고 인사이트를 제공해주세요:\n\n{content}\n\n분석:"
                else:
                    prompt = f"Analyze the following content and provide insights:\n\n{content}\n\nAnalysis:"
                
                if hasattr(self.llm, 'generate'):
                    response = self.llm.generate(prompt)
                    return response.content
                else:
                    return str(self.llm(prompt))
                    
            except Exception as e:
                return f"Analysis error: {str(e)}"
        
        tools.append(Tool(
            name="analyze",
            description="Analyze content and provide insights",
            func=analyze_tool
        ))
        
        return tools
    
    def _create_agent_prompt(self) -> StringPromptTemplate:
        """에이전트 프롬프트 템플릿 생성"""
        
        if self.language == "korean":
            template = """당신은 지능형 RAG 에이전트입니다. 사용자의 질문에 답하기 위해 다음 도구들을 사용할 수 있습니다:

{tools}

도구 사용 방법:
```
Action: 도구_이름
Action Input: 도구에 전달할 입력
```

관찰 결과를 받으면 다음과 같이 생각하세요:
```
Thought: 결과에 대한 생각
```

최종 답변이 준비되면:
```
Final Answer: 최종 답변
```

이전 대화:
{chat_history}

질문: {input}
{agent_scratchpad}"""
        else:
            template = """You are an intelligent RAG agent. You can use the following tools to answer user questions:

{tools}

Use tools like this:
```
Action: tool_name
Action Input: input for the tool
```

When you receive observations, think like this:
```
Thought: your thoughts about the result
```

When ready with final answer:
```
Final Answer: your final answer
```

Previous conversation:
{chat_history}

Question: {input}
{agent_scratchpad}"""
        
        class CustomPromptTemplate(StringPromptTemplate):
            template: str
            tools: List[Tool]
            
            def format(self, **kwargs) -> str:
                # 도구 목록 생성
                tools_str = "\n".join([f"- {tool.name}: {tool.description}" for tool in self.tools])
                kwargs["tools"] = tools_str
                return self.template.format(**kwargs)
        
        return CustomPromptTemplate(
            template=template,
            tools=self.tools,
            input_variables=["input", "chat_history", "agent_scratchpad"]
        )
    
    def _setup_agent(self):
        """에이전트 설정"""
        
        # 출력 파서
        class CustomOutputParser(AgentOutputParser):
            def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
                # Final Answer 확인
                if "Final Answer:" in llm_output:
                    answer = llm_output.split("Final Answer:")[-1].strip()
                    return AgentFinish(
                        return_values={"output": answer},
                        log=llm_output,
                    )
                
                # Action 파싱
                if "Action:" in llm_output and "Action Input:" in llm_output:
                    try:
                        action_match = llm_output.split("Action:")[-1].split("Action Input:")[0].strip()
                        action_input_match = llm_output.split("Action Input:")[-1].strip()
                        
                        return AgentAction(
                            tool=action_match,
                            tool_input=action_input_match,
                            log=llm_output
                        )
                    except:
                        pass
                
                raise OutputParserException(f"Could not parse LLM output: `{llm_output}`")
        
        # LLM 체인
        llm_chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
        
        # 에이전트
        agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=CustomOutputParser(),
            stop=["\nObservation:"],
            allowed_tools=[tool.name for tool in self.tools]
        )
        
        # 에이전트 실행기
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            verbose=False,
            max_iterations=self.max_iterations
        )
    
    def _setup_custom_agentic_rag(self, llm, embeddings, vector_store):
        """커스텀 에이전트 RAG 설정 (LangChain 없이)"""
        
        self.llm = llm
        self.embeddings = embeddings
        self.vector_store = vector_store
        
        # 커스텀 도구 함수들
        self.custom_tools = {
            "search": self._custom_search_tool,
            "summarize": self._custom_summarize_tool,
            "compare": self._custom_compare_tool,
            "analyze": self._custom_analyze_tool
        }
        
        logger.info("커스텀 AgenticRAG 설정 완료")
    
    def _custom_search_tool(self, query: str) -> str:
        """커스텀 검색 도구"""
        try:
            if hasattr(self.vector_store, 'search'):
                query_embedding = self.embeddings.embed_text(query)
                search_result = self.vector_store.search(query_embedding, k=5)
                
                results = []
                for doc in search_result.documents:
                    results.append(f"Document: {doc.content[:200]}...")
                
                return "\n".join(results) if results else "No documents found."
            
            return "Search functionality not available."
            
        except Exception as e:
            return f"Search error: {str(e)}"
    
    def _custom_summarize_tool(self, text: str) -> str:
        """커스텀 요약 도구"""
        try:
            if self.language == "korean":
                prompt = f"다음 텍스트를 간단히 요약해주세요:\n\n{text}\n\n요약:"
            else:
                prompt = f"Please summarize the following text:\n\n{text}\n\nSummary:"
            
            if hasattr(self.llm, 'generate'):
                response = self.llm.generate(prompt)
                return response.content
            else:
                return str(self.llm(prompt))
                
        except Exception as e:
            return f"Summarization error: {str(e)}"
    
    def _custom_compare_tool(self, items: str) -> str:
        """커스텀 비교 도구"""
        try:
            if self.language == "korean":
                prompt = f"다음 항목들을 비교하고 차이점을 설명해주세요:\n\n{items}\n\n비교 분석:"
            else:
                prompt = f"Compare the following items and explain the differences:\n\n{items}\n\nComparison:"
            
            if hasattr(self.llm, 'generate'):
                response = self.llm.generate(prompt)
                return response.content
            else:
                return str(self.llm(prompt))
                
        except Exception as e:
            return f"Comparison error: {str(e)}"
    
    def _custom_analyze_tool(self, content: str) -> str:
        """커스텀 분석 도구"""
        try:
            if self.language == "korean":
                prompt = f"다음 내용을 분석하고 인사이트를 제공해주세요:\n\n{content}\n\n분석:"
            else:
                prompt = f"Analyze the following content and provide insights:\n\n{content}\n\nAnalysis:"
            
            if hasattr(self.llm, 'generate'):
                response = self.llm.generate(prompt)
                return response.content
            else:
                return str(self.llm(prompt))
                
        except Exception as e:
            return f"Analysis error: {str(e)}"
    
    def ask(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """에이전트를 사용한 질문 답변"""
        
        if self.use_langchain and HAS_LANGCHAIN:
            return self._ask_langchain(question, chat_history)
        else:
            return self._ask_custom(question, chat_history)
    
    def _ask_langchain(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]]
    ) -> Dict[str, Any]:
        """LangChain을 사용한 에이전트 질문 답변"""
        
        try:
            # 대화 기록 포맷팅
            history_str = ""
            if chat_history:
                for turn in chat_history[-3:]:  # 최근 3턴만
                    history_str += f"Human: {turn.get('human', '')}\nAI: {turn.get('ai', '')}\n"
            
            # 에이전트 실행
            result = self.agent_executor.run(
                input=question,
                chat_history=history_str
            )
            
            response = {
                "question": question,
                "answer": result,
                "method": "langchain_agent",
                "timestamp": datetime.now().isoformat()
            }
            
            return response
            
        except Exception as e:
            logger.error(f"LangChain 에이전트 오류: {e}")
            return {
                "question": question,
                "answer": "에이전트 처리 중 오류가 발생했습니다.",
                "error": str(e),
                "method": "langchain_agent",
                "timestamp": datetime.now().isoformat()
            }
    
    def _ask_custom(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]]
    ) -> Dict[str, Any]:
        """커스텀 에이전트 질문 답변"""
        
        try:
            # 에이전트 실행 단계 기록
            agent_steps = []
            
            # 초기 계획 수립
            plan = self._create_execution_plan(question)
            
            # 단계별 실행
            current_context = ""
            final_answer = ""
            
            for step_id, action_plan in enumerate(plan["steps"]):
                step = self._execute_agent_step(
                    step_id, action_plan, question, current_context
                )
                agent_steps.append(step)
                
                # 컨텍스트 업데이트
                current_context += f"\nStep {step_id}: {step.output}"
                
                # 신뢰도 확인
                if step.confidence >= self.confidence_threshold and step.action_type == ActionType.SYNTHESIZE:
                    final_answer = step.output
                    break
            
            # 최종 답변이 없으면 마지막 단계 결과 사용
            if not final_answer and agent_steps:
                final_answer = agent_steps[-1].output
            
            # 반성 단계 (선택적)
            if self.enable_reflection and len(agent_steps) > 1:
                reflection_step = self._reflect_on_execution(question, agent_steps, final_answer)
                agent_steps.append(reflection_step)
                
                # 반성 결과가 더 좋으면 업데이트
                if reflection_step.confidence > 0.8:
                    final_answer = reflection_step.output
            
            response = {
                "question": question,
                "answer": final_answer,
                "execution_plan": plan,
                "agent_steps": [self._step_to_dict(step) for step in agent_steps] if self.return_agent_steps else None,
                "total_steps": len(agent_steps),
                "method": "custom_agent",
                "timestamp": datetime.now().isoformat()
            }
            
            # 실행 기록 저장
            self.execution_history.append({
                "question": question,
                "steps": agent_steps,
                "final_answer": final_answer,
                "timestamp": datetime.now()
            })
            
            return response
            
        except Exception as e:
            logger.error(f"커스텀 에이전트 오류: {e}")
            return {
                "question": question,
                "answer": "에이전트 처리 중 오류가 발생했습니다.",
                "error": str(e),
                "method": "custom_agent",
                "timestamp": datetime.now().isoformat()
            }
    
    def _create_execution_plan(self, question: str) -> Dict[str, Any]:
        """실행 계획 수립"""
        
        # 질문 유형 분석
        question_type = self._analyze_question_type(question)
        
        # 질문 유형에 따른 실행 계획
        if question_type == "factual":
            steps = [
                {"action": ActionType.SEARCH, "description": "관련 문서 검색"},
                {"action": ActionType.SYNTHESIZE, "description": "정보 종합 및 답변 생성"}
            ]
        elif question_type == "analytical":
            steps = [
                {"action": ActionType.SEARCH, "description": "관련 문서 검색"},
                {"action": ActionType.ANALYZE, "description": "내용 분석"},
                {"action": ActionType.SYNTHESIZE, "description": "분석 결과 종합"}
            ]
        elif question_type == "comparative":
            steps = [
                {"action": ActionType.SEARCH, "description": "비교 대상 정보 검색"},
                {"action": ActionType.COMPARE, "description": "비교 분석 수행"},
                {"action": ActionType.SYNTHESIZE, "description": "비교 결과 종합"}
            ]
        elif question_type == "complex":
            steps = [
                {"action": ActionType.SEARCH, "description": "기본 정보 검색"},
                {"action": ActionType.ANALYZE, "description": "1차 분석"},
                {"action": ActionType.SEARCH, "description": "추가 정보 검색"},
                {"action": ActionType.SYNTHESIZE, "description": "최종 종합"}
            ]
        else:
            # 기본 계획
            steps = [
                {"action": ActionType.SEARCH, "description": "정보 검색"},
                {"action": ActionType.SYNTHESIZE, "description": "답변 생성"}
            ]
        
        return {
            "question_type": question_type,
            "steps": steps,
            "estimated_steps": len(steps)
        }
    
    def _analyze_question_type(self, question: str) -> str:
        """질문 유형 분석"""
        
        question_lower = question.lower()
        
        # 비교 질문
        compare_keywords = ["비교", "차이", "다른", "vs", "versus", "compare", "difference", "different"]
        if any(keyword in question_lower for keyword in compare_keywords):
            return "comparative"
        
        # 분석 질문
        analyze_keywords = ["분석", "왜", "어떻게", "이유", "원인", "analyze", "why", "how", "reason", "cause"]
        if any(keyword in question_lower for keyword in analyze_keywords):
            return "analytical"
        
        # 복잡한 질문 (여러 부분)
        if len(question.split("?")) > 2 or len(question.split("그리고")) > 1 or len(question.split("and")) > 2:
            return "complex"
        
        # 사실적 질문 (기본)
        return "factual"
    
    def _execute_agent_step(
        self,
        step_id: int,
        action_plan: Dict[str, Any],
        question: str,
        context: str
    ) -> AgentStep:
        """에이전트 단계 실행"""
        
        action_type = action_plan["action"]
        start_time = datetime.now()
        
        try:
            if action_type == ActionType.SEARCH:
                # 검색 쿼리 생성
                search_query = self._generate_search_query(question, context)
                output = self.custom_tools["search"](search_query)
                confidence = 0.8 if "Document:" in output else 0.3
                reasoning = f"검색 쿼리 '{search_query}'로 문서 검색 수행"
                
            elif action_type == ActionType.ANALYZE:
                output = self.custom_tools["analyze"](context)
                confidence = 0.7
                reasoning = "수집된 정보에 대한 분석 수행"
                
            elif action_type == ActionType.COMPARE:
                output = self.custom_tools["compare"](context)
                confidence = 0.7
                reasoning = "비교 분석 수행"
                
            elif action_type == ActionType.SUMMARIZE:
                output = self.custom_tools["summarize"](context)
                confidence = 0.6
                reasoning = "정보 요약 수행"
                
            elif action_type == ActionType.SYNTHESIZE:
                output = self._synthesize_final_answer(question, context)
                confidence = 0.9
                reasoning = "최종 답변 종합"
                
            else:
                output = "Unknown action type"
                confidence = 0.1
                reasoning = "알 수 없는 액션 타입"
            
            return AgentStep(
                step_id=step_id,
                action_type=action_type,
                input=question if step_id == 0 else context,
                output=output,
                confidence=confidence,
                reasoning=reasoning,
                timestamp=start_time
            )
            
        except Exception as e:
            logger.warning(f"Agent step {step_id} 실행 오류: {e}")
            return AgentStep(
                step_id=step_id,
                action_type=action_type,
                input=question if step_id == 0 else context,
                output=f"Step execution failed: {str(e)}",
                confidence=0.1,
                reasoning=f"실행 중 오류 발생: {str(e)}",
                timestamp=start_time
            )
    
    def _generate_search_query(self, question: str, context: str) -> str:
        """검색 쿼리 생성"""
        
        if not context:
            return question
        
        # 간단한 키워드 추출
        keywords = []
        for word in question.split():
            if len(word) > 2 and word.isalpha():
                keywords.append(word)
        
        return " ".join(keywords[:5])  # 상위 5개 키워드
    
    def _synthesize_final_answer(self, question: str, context: str) -> str:
        """최종 답변 종합"""
        
        try:
            if self.language == "korean":
                prompt = f"""다음 정보를 바탕으로 질문에 대한 최종 답변을 작성해주세요.

질문: {question}

수집된 정보:
{context}

답변시 고려사항:
1. 질문에 직접적으로 답변하세요
2. 수집된 정보를 논리적으로 정리하세요
3. 확실하지 않은 정보는 명시하세요

최종 답변:"""
            else:
                prompt = f"""Based on the following information, provide a final answer to the question.

Question: {question}

Collected Information:
{context}

Guidelines:
1. Answer the question directly
2. Organize the collected information logically
3. Indicate any uncertainties

Final Answer:"""
            
            if hasattr(self.llm, 'generate'):
                response = self.llm.generate(prompt)
                return response.content
            else:
                return str(self.llm(prompt))
                
        except Exception as e:
            return f"최종 답변 생성 중 오류가 발생했습니다: {str(e)}"
    
    def _reflect_on_execution(
        self,
        question: str,
        agent_steps: List[AgentStep],
        current_answer: str
    ) -> AgentStep:
        """실행 결과에 대한 반성"""
        
        try:
            steps_summary = "\n".join([
                f"Step {step.step_id}: {step.action_type.value} -> {step.output[:100]}..."
                for step in agent_steps
            ])
            
            if self.language == "korean":
                prompt = f"""다음 에이전트 실행 과정을 검토하고 개선된 답변을 제공해주세요.

원래 질문: {question}

실행 단계:
{steps_summary}

현재 답변: {current_answer}

검토 사항:
1. 모든 필요한 정보가 수집되었는가?
2. 답변이 질문에 충분히 대답하는가?
3. 논리적 오류나 누락된 부분이 있는가?

개선된 최종 답변:"""
            else:
                prompt = f"""Review the following agent execution process and provide an improved answer.

Original Question: {question}

Execution Steps:
{steps_summary}

Current Answer: {current_answer}

Review Points:
1. Was all necessary information collected?
2. Does the answer sufficiently address the question?
3. Are there any logical errors or missing parts?

Improved Final Answer:"""
            
            if hasattr(self.llm, 'generate'):
                response = self.llm.generate(prompt)
                improved_answer = response.content
            else:
                improved_answer = str(self.llm(prompt))
            
            # 개선 정도 평가 (간단한 휴리스틱)
            if len(improved_answer) > len(current_answer) * 1.2:
                confidence = 0.9
            elif improved_answer != current_answer:
                confidence = 0.8
            else:
                confidence = 0.6
            
            return AgentStep(
                step_id=len(agent_steps),
                action_type=ActionType.REFLECT,
                input=current_answer,
                output=improved_answer,
                confidence=confidence,
                reasoning="실행 과정 반성 및 답변 개선",
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.warning(f"반성 단계 오류: {e}")
            return AgentStep(
                step_id=len(agent_steps),
                action_type=ActionType.REFLECT,
                input=current_answer,
                output=current_answer,
                confidence=0.5,
                reasoning=f"반성 단계 실행 실패: {str(e)}",
                timestamp=datetime.now()
            )
    
    def _step_to_dict(self, step: AgentStep) -> Dict[str, Any]:
        """AgentStep을 딕셔너리로 변환"""
        
        return {
            "step_id": step.step_id,
            "action_type": step.action_type.value,
            "input": step.input[:200] + "..." if len(step.input) > 200 else step.input,
            "output": step.output[:500] + "..." if len(step.output) > 500 else step.output,
            "confidence": step.confidence,
            "reasoning": step.reasoning,
            "timestamp": step.timestamp.isoformat()
        }
    
    def get_execution_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """실행 기록 반환"""
        
        history = []
        for record in self.execution_history:
            history.append({
                "question": record["question"],
                "total_steps": len(record["steps"]),
                "final_answer": record["final_answer"][:200] + "..." if len(record["final_answer"]) > 200 else record["final_answer"],
                "timestamp": record["timestamp"].isoformat()
            })
        
        if limit:
            return history[-limit:]
        return history
    
    def analyze_performance(self) -> Dict[str, Any]:
        """에이전트 성능 분석"""
        
        if not self.execution_history:
            return {"message": "실행 기록이 없습니다."}
        
        total_executions = len(self.execution_history)
        total_steps = sum(len(record["steps"]) for record in self.execution_history)
        avg_steps = total_steps / total_executions
        
        # 평균 신뢰도
        all_confidences = []
        for record in self.execution_history:
            for step in record["steps"]:
                all_confidences.append(step.confidence)
        
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
        
        # 액션 타입 분포
        action_counts = {}
        for record in self.execution_history:
            for step in record["steps"]:
                action_type = step.action_type.value
                action_counts[action_type] = action_counts.get(action_type, 0) + 1
        
        return {
            "total_executions": total_executions,
            "total_steps": total_steps,
            "average_steps_per_execution": avg_steps,
            "average_confidence": avg_confidence,
            "action_type_distribution": action_counts,
            "performance_summary": {
                "efficiency": "high" if avg_steps <= 3 else "medium" if avg_steps <= 5 else "low",
                "reliability": "high" if avg_confidence >= 0.8 else "medium" if avg_confidence >= 0.6 else "low"
            }
        }
    
    def get_config(self) -> Dict[str, Any]:
        """현재 설정 반환"""
        
        return {
            "language": self.language,
            "max_iterations": self.max_iterations,
            "enable_reflection": self.enable_reflection,
            "enable_tool_selection": self.enable_tool_selection,
            "confidence_threshold": self.confidence_threshold,
            "use_langchain": self.use_langchain,
            "has_langchain": HAS_LANGCHAIN,
            "return_agent_steps": self.return_agent_steps,
            "return_source_documents": self.return_source_documents,
            "available_tools": list(self.custom_tools.keys()) if not self.use_langchain else [tool.name for tool in self.tools],
            "execution_history_size": len(self.execution_history),
            "llm_type": type(self.llm).__name__,
            "embeddings_type": type(self.embeddings).__name__,
            "vector_store_type": type(self.vector_store).__name__
        }