"""
RAG Evaluator Module
RAG 시스템의 종합적인 평가를 수행하는 모듈
"""

import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from .metrics import RAGMetrics, RetrievalMetrics, GenerationMetrics, MetricResult

logger = logging.getLogger(__name__)


@dataclass
class EvaluationDataset:
    """평가 데이터셋 클래스"""
    questions: List[str]
    ground_truth_answers: List[str]
    ground_truth_contexts: List[List[str]]
    metadata: Optional[Dict[str, Any]] = None
    
    def __len__(self) -> int:
        return len(self.questions)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "question": self.questions[idx],
            "ground_truth_answer": self.ground_truth_answers[idx],
            "ground_truth_contexts": self.ground_truth_contexts[idx],
            "metadata": self.metadata.get(str(idx), {}) if self.metadata else {}
        }
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationDataset':
        return cls(**data)
    
    def save(self, filepath: Union[str, Path]):
        """데이터셋을 JSON 파일로 저장"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"평가 데이터셋 저장: {filepath}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'EvaluationDataset':
        """JSON 파일에서 데이터셋 로드"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"평가 데이터셋 로드: {filepath}")
        return cls.from_dict(data)


@dataclass
class EvaluationResult:
    """평가 결과 클래스"""
    dataset_name: str
    total_questions: int
    evaluation_time: float
    retrieval_metrics: Dict[str, float]
    generation_metrics: Dict[str, float]
    rag_metrics: Dict[str, float]
    per_question_results: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None
    
    def get_summary(self) -> Dict[str, Any]:
        """평가 결과 요약"""
        
        summary = {
            "dataset": self.dataset_name,
            "total_questions": self.total_questions,
            "evaluation_time": self.evaluation_time,
            "average_scores": {
                **self.retrieval_metrics,
                **self.generation_metrics,
                **self.rag_metrics
            }
        }
        
        return summary
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def save(self, filepath: Union[str, Path]):
        """결과를 JSON 파일로 저장"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"평가 결과 저장: {filepath}")


class RAGEvaluator:
    """RAG 시스템 종합 평가기"""
    
    def __init__(
        self,
        retriever,
        generator,  # LLM provider
        config: Dict[str, Any]
    ):
        self.retriever = retriever
        self.generator = generator
        self.config = config
        
        # 평가 설정
        self.language = config.get("language", "korean")
        self.evaluation_metrics = config.get("evaluation_metrics", "all")
        self.k_values = config.get("k_values", [1, 3, 5, 10])
        self.include_per_question = config.get("include_per_question", True)
        
        # 메트릭 초기화
        self.rag_metrics = RAGMetrics(self.language, self.generator)
        self.retrieval_metrics = RetrievalMetrics()
        self.generation_metrics = GenerationMetrics(self.language)
        
        logger.info(f"RAG 평가기 초기화: {self.language}")
    
    def evaluate(
        self,
        dataset: EvaluationDataset,
        dataset_name: str = "unknown"
    ) -> EvaluationResult:
        """RAG 시스템 전체 평가"""
        
        start_time = time.time()
        logger.info(f"RAG 평가 시작: {dataset_name} ({len(dataset)}개 질문)")
        
        per_question_results = []
        all_retrieval_scores = {f"precision@{k}": [] for k in self.k_values}
        all_retrieval_scores.update({f"recall@{k}": [] for k in self.k_values})
        all_retrieval_scores.update({f"f1@{k}": [] for k in self.k_values})
        all_retrieval_scores.update({f"ndcg@{k}": [] for k in self.k_values})
        all_retrieval_scores.update({"mrr": [], "map": []})
        
        all_generation_scores = {
            "bleu_1": [], "bleu_2": [], "bleu_3": [], "bleu_4": [],
            "rouge1_f1": [], "rouge2_f1": [], "rougeL_f1": [],
            "semantic_similarity": []
        }
        
        all_rag_scores = {
            "faithfulness": [],
            "answer_relevancy": [],
            "context_precision": [],
            "context_recall": []
        }
        
        for i, item in enumerate(dataset):
            if i % 10 == 0:
                logger.info(f"평가 진행률: {i+1}/{len(dataset)}")
            
            try:
                result = self._evaluate_single_question(item)
                per_question_results.append(result)
                
                # 검색 메트릭 수집
                for metric, value in result["retrieval_metrics"].items():
                    if metric in all_retrieval_scores:
                        all_retrieval_scores[metric].append(value)
                
                # 생성 메트릭 수집
                for metric, value in result["generation_metrics"].items():
                    if metric in all_generation_scores:
                        all_generation_scores[metric].append(value)
                
                # RAG 메트릭 수집
                for metric, value in result["rag_metrics"].items():
                    if metric in all_rag_scores:
                        all_rag_scores[metric].append(value)
                        
            except Exception as e:
                logger.error(f"질문 {i} 평가 오류: {e}")
                # 빈 결과로 추가
                empty_result = {
                    "question_id": i,
                    "question": item["question"],
                    "error": str(e),
                    "retrieval_metrics": {},
                    "generation_metrics": {},
                    "rag_metrics": {}
                }
                per_question_results.append(empty_result)
        
        # 평균 점수 계산
        avg_retrieval_metrics = {
            metric: sum(scores) / len(scores) if scores else 0.0
            for metric, scores in all_retrieval_scores.items()
        }
        
        avg_generation_metrics = {
            metric: sum(scores) / len(scores) if scores else 0.0
            for metric, scores in all_generation_scores.items()
        }
        
        avg_rag_metrics = {
            metric: sum(scores) / len(scores) if scores else 0.0
            for metric, scores in all_rag_scores.items()
        }
        
        evaluation_time = time.time() - start_time
        
        result = EvaluationResult(
            dataset_name=dataset_name,
            total_questions=len(dataset),
            evaluation_time=evaluation_time,
            retrieval_metrics=avg_retrieval_metrics,
            generation_metrics=avg_generation_metrics,
            rag_metrics=avg_rag_metrics,
            per_question_results=per_question_results if self.include_per_question else [],
            metadata={
                "config": self.config,
                "k_values": self.k_values,
                "language": self.language
            }
        )
        
        logger.info(f"RAG 평가 완료: {evaluation_time:.2f}초")
        return result
    
    def _evaluate_single_question(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """단일 질문에 대한 평가"""
        
        question = item["question"]
        ground_truth_answer = item["ground_truth_answer"]
        ground_truth_contexts = item["ground_truth_contexts"]
        
        # 1. 검색 평가
        retrieval_result = self.retriever.retrieve(question, k=max(self.k_values))
        retrieved_docs = retrieval_result.documents
        retrieved_doc_ids = [doc.id for doc in retrieved_docs]
        retrieved_contexts = [doc.content for doc in retrieved_docs]
        
        # 2. 생성 평가
        # 검색된 컨텍스트로 답변 생성
        combined_context = "\n\n".join(retrieved_contexts[:5])  # 상위 5개 컨텍스트 사용
        rag_prompt = self.generator.format_rag_prompt(question, combined_context)
        generated_response = self.generator.generate(rag_prompt)
        generated_answer = generated_response.content
        
        # 3. 메트릭 계산
        retrieval_metrics = self._calculate_retrieval_metrics(
            ground_truth_contexts,
            retrieved_doc_ids,
            retrieved_contexts
        )
        
        generation_metrics = self._calculate_generation_metrics(
            ground_truth_answer,
            generated_answer
        )
        
        rag_metrics = self._calculate_rag_metrics(
            question,
            generated_answer,
            combined_context,
            ground_truth_answer,
            ground_truth_contexts,
            retrieved_contexts
        )
        
        return {
            "question_id": item.get("question_id", 0),
            "question": question,
            "ground_truth_answer": ground_truth_answer,
            "generated_answer": generated_answer,
            "retrieved_contexts": retrieved_contexts[:3],  # 상위 3개만 저장
            "retrieval_metrics": retrieval_metrics,
            "generation_metrics": generation_metrics,
            "rag_metrics": rag_metrics,
            "metadata": item.get("metadata", {})
        }
    
    def _calculate_retrieval_metrics(
        self,
        ground_truth_contexts: List[str],
        retrieved_doc_ids: List[str],
        retrieved_contexts: List[str]
    ) -> Dict[str, float]:
        """검색 메트릭 계산"""
        
        metrics = {}
        
        # 정답 컨텍스트를 ID로 변환 (간단히 인덱스 사용)
        # 실제로는 더 정교한 매칭이 필요
        relevant_doc_ids = [str(i) for i in range(len(ground_truth_contexts))]
        
        # 의미적 유사도를 기반으로 관련 문서 판단
        relevant_retrieved_ids = []
        for i, ret_context in enumerate(retrieved_contexts):
            for gt_context in ground_truth_contexts:
                similarity = self.generation_metrics.semantic_similarity(ret_context, gt_context)
                if similarity > 0.7:  # 임계값
                    relevant_retrieved_ids.append(retrieved_doc_ids[i])
                    break
        
        # K별 메트릭 계산
        for k in self.k_values:
            if k <= len(retrieved_doc_ids):
                metrics[f"precision@{k}"] = self.retrieval_metrics.precision_at_k(
                    relevant_retrieved_ids, retrieved_doc_ids, k
                )
                metrics[f"recall@{k}"] = self.retrieval_metrics.recall_at_k(
                    relevant_retrieved_ids, retrieved_doc_ids, k
                )
                metrics[f"f1@{k}"] = self.retrieval_metrics.f1_at_k(
                    relevant_retrieved_ids, retrieved_doc_ids, k
                )
                metrics[f"ndcg@{k}"] = self.retrieval_metrics.ndcg_at_k(
                    relevant_retrieved_ids, retrieved_doc_ids, k=k
                )
        
        # MRR 계산
        metrics["mrr"] = self.retrieval_metrics.mrr(
            relevant_retrieved_ids, retrieved_doc_ids
        )
        
        return metrics
    
    def _calculate_generation_metrics(
        self,
        ground_truth_answer: str,
        generated_answer: str
    ) -> Dict[str, float]:
        """생성 메트릭 계산"""
        
        metrics = {}
        
        # BLEU 점수
        bleu_scores = self.generation_metrics.bleu_score(
            ground_truth_answer, generated_answer
        )
        metrics.update(bleu_scores)
        
        # ROUGE 점수
        rouge_scores = self.generation_metrics.rouge_score(
            ground_truth_answer, generated_answer
        )
        metrics.update(rouge_scores)
        
        # 의미적 유사도
        metrics["semantic_similarity"] = self.generation_metrics.semantic_similarity(
            ground_truth_answer, generated_answer
        )
        
        return metrics
    
    def _calculate_rag_metrics(
        self,
        question: str,
        generated_answer: str,
        combined_context: str,
        ground_truth_answer: str,
        ground_truth_contexts: List[str],
        retrieved_contexts: List[str]
    ) -> Dict[str, float]:
        """RAG 특화 메트릭 계산"""
        
        metrics = {}
        
        # Faithfulness (충실성)
        metrics["faithfulness"] = self.rag_metrics.faithfulness(
            generated_answer, combined_context, question
        )
        
        # Answer Relevancy (답변 관련성)
        metrics["answer_relevancy"] = self.rag_metrics.answer_relevancy(
            generated_answer, question, combined_context
        )
        
        # Context Precision (컨텍스트 정밀도)
        metrics["context_precision"] = self.rag_metrics.context_precision(
            retrieved_contexts, ground_truth_contexts, question
        )
        
        # Context Recall (컨텍스트 재현율)
        metrics["context_recall"] = self.rag_metrics.context_recall(
            retrieved_contexts, ground_truth_contexts, question
        )
        
        return metrics
    
    def evaluate_retrieval_only(
        self,
        dataset: EvaluationDataset,
        dataset_name: str = "retrieval_only"
    ) -> EvaluationResult:
        """검색 시스템만 평가"""
        
        start_time = time.time()
        logger.info(f"검색 평가 시작: {dataset_name}")
        
        per_question_results = []
        all_scores = {f"precision@{k}": [] for k in self.k_values}
        all_scores.update({f"recall@{k}": [] for k in self.k_values})
        all_scores.update({f"f1@{k}": [] for k in self.k_values})
        all_scores.update({f"ndcg@{k}": [] for k in self.k_values})
        all_scores.update({"mrr": []})
        
        for i, item in enumerate(dataset):
            if i % 10 == 0:
                logger.info(f"검색 평가 진행률: {i+1}/{len(dataset)}")
            
            try:
                question = item["question"]
                ground_truth_contexts = item["ground_truth_contexts"]
                
                # 검색 수행
                retrieval_result = self.retriever.retrieve(question, k=max(self.k_values))
                retrieved_contexts = [doc.content for doc in retrieval_result.documents]
                retrieved_doc_ids = [doc.id for doc in retrieval_result.documents]
                
                # 메트릭 계산
                retrieval_metrics = self._calculate_retrieval_metrics(
                    ground_truth_contexts,
                    retrieved_doc_ids,
                    retrieved_contexts
                )
                
                # 점수 수집
                for metric, value in retrieval_metrics.items():
                    if metric in all_scores:
                        all_scores[metric].append(value)
                
                if self.include_per_question:
                    per_question_results.append({
                        "question_id": i,
                        "question": question,
                        "retrieval_metrics": retrieval_metrics,
                        "retrieved_contexts": retrieved_contexts[:3]
                    })
                    
            except Exception as e:
                logger.error(f"검색 평가 질문 {i} 오류: {e}")
        
        # 평균 점수 계산
        avg_scores = {
            metric: sum(scores) / len(scores) if scores else 0.0
            for metric, scores in all_scores.items()
        }
        
        evaluation_time = time.time() - start_time
        
        return EvaluationResult(
            dataset_name=dataset_name,
            total_questions=len(dataset),
            evaluation_time=evaluation_time,
            retrieval_metrics=avg_scores,
            generation_metrics={},
            rag_metrics={},
            per_question_results=per_question_results,
            metadata={"evaluation_type": "retrieval_only"}
        )


def create_evaluation_dataset(
    questions: List[str],
    ground_truth_answers: List[str],
    ground_truth_contexts: List[List[str]],
    metadata: Optional[Dict[str, Any]] = None
) -> EvaluationDataset:
    """평가 데이터셋 생성 헬퍼 함수"""
    
    if len(questions) != len(ground_truth_answers) != len(ground_truth_contexts):
        raise ValueError("질문, 답변, 컨텍스트의 개수가 일치하지 않습니다.")
    
    return EvaluationDataset(
        questions=questions,
        ground_truth_answers=ground_truth_answers,
        ground_truth_contexts=ground_truth_contexts,
        metadata=metadata
    )


def load_evaluation_dataset_from_csv(
    filepath: Union[str, Path],
    question_column: str = "question",
    answer_column: str = "answer",
    context_column: str = "context"
) -> EvaluationDataset:
    """CSV 파일에서 평가 데이터셋 로드"""
    
    try:
        import pandas as pd
        
        df = pd.read_csv(filepath)
        
        questions = df[question_column].tolist()
        answers = df[answer_column].tolist()
        
        # 컨텍스트가 여러 개인 경우 처리
        if context_column in df.columns:
            contexts = []
            for context_str in df[context_column]:
                if isinstance(context_str, str):
                    # 구분자로 분리 (예: "|" 또는 "\n")
                    context_list = [c.strip() for c in context_str.split("|")]
                    contexts.append(context_list)
                else:
                    contexts.append([])
        else:
            contexts = [[]] * len(questions)
        
        return EvaluationDataset(
            questions=questions,
            ground_truth_answers=answers,
            ground_truth_contexts=contexts
        )
        
    except ImportError:
        raise ImportError("CSV 로딩을 위해 pandas를 설치해주세요: pip install pandas")
    except Exception as e:
        logger.error(f"CSV 로딩 오류: {e}")
        raise e