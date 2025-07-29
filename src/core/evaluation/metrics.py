"""
Metrics Module
RAG 시스템 평가를 위한 다양한 메트릭 구현
"""

import re
import math
import logging
from collections import Counter
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass

# 외부 라이브러리 (optional imports)
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    import nltk
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False

try:
    from bert_score import score as bert_score
    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import konlpy
    from konlpy.tag import Okt
    HAS_KONLPY = True
except ImportError:
    HAS_KONLPY = False

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """메트릭 결과 표준화 클래스"""
    metric_name: str
    value: float
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "value": self.value,
            "metadata": self.metadata or {}
        }


class RetrievalMetrics:
    """검색 성능 평가 메트릭"""
    
    @staticmethod
    def precision_at_k(
        relevant_docs: List[str],
        retrieved_docs: List[str],
        k: int
    ) -> float:
        """Precision@K 계산"""
        
        if k == 0 or not retrieved_docs:
            return 0.0
        
        retrieved_k = retrieved_docs[:k]
        relevant_set = set(relevant_docs)
        
        relevant_retrieved = sum(1 for doc in retrieved_k if doc in relevant_set)
        
        return relevant_retrieved / len(retrieved_k)
    
    @staticmethod
    def recall_at_k(
        relevant_docs: List[str],
        retrieved_docs: List[str],
        k: int
    ) -> float:
        """Recall@K 계산"""
        
        if not relevant_docs:
            return 0.0
        
        retrieved_k = retrieved_docs[:k]
        relevant_set = set(relevant_docs)
        
        relevant_retrieved = sum(1 for doc in retrieved_k if doc in relevant_set)
        
        return relevant_retrieved / len(relevant_docs)
    
    @staticmethod
    def f1_at_k(
        relevant_docs: List[str],
        retrieved_docs: List[str],
        k: int
    ) -> float:
        """F1@K 계산"""
        
        precision = RetrievalMetrics.precision_at_k(relevant_docs, retrieved_docs, k)
        recall = RetrievalMetrics.recall_at_k(relevant_docs, retrieved_docs, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def ndcg_at_k(
        relevant_docs: List[str],
        retrieved_docs: List[str],
        relevance_scores: Optional[Dict[str, float]] = None,
        k: int = 10
    ) -> float:
        """NDCG@K (Normalized Discounted Cumulative Gain) 계산"""
        
        if k == 0 or not retrieved_docs:
            return 0.0
        
        retrieved_k = retrieved_docs[:k]
        relevant_set = set(relevant_docs)
        
        # 관련성 점수가 없으면 이진 관련성 사용 (관련: 1, 비관련: 0)
        if relevance_scores is None:
            relevance_scores = {doc: 1.0 for doc in relevant_docs}
        
        # DCG 계산
        dcg = 0.0
        for i, doc in enumerate(retrieved_k):
            if doc in relevant_set:
                relevance = relevance_scores.get(doc, 0.0)
                dcg += relevance / math.log2(i + 2)  # i+2 because log2(1) = 0
        
        # IDCG 계산 (이상적인 순서)
        ideal_relevances = sorted(
            [relevance_scores.get(doc, 0.0) for doc in relevant_docs],
            reverse=True
        )[:k]
        
        idcg = 0.0
        for i, relevance in enumerate(ideal_relevances):
            idcg += relevance / math.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def mrr(
        relevant_docs: List[str],
        retrieved_docs: List[str]
    ) -> float:
        """MRR (Mean Reciprocal Rank) 계산"""
        
        if not relevant_docs or not retrieved_docs:
            return 0.0
        
        relevant_set = set(relevant_docs)
        
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_set:
                return 1.0 / (i + 1)
        
        return 0.0
    
    @staticmethod
    def map_score(
        queries_relevant_docs: List[List[str]],
        queries_retrieved_docs: List[List[str]]
    ) -> float:
        """MAP (Mean Average Precision) 계산"""
        
        if len(queries_relevant_docs) != len(queries_retrieved_docs):
            raise ValueError("쿼리 개수가 일치하지 않습니다.")
        
        if not queries_relevant_docs:
            return 0.0
        
        ap_scores = []
        
        for relevant_docs, retrieved_docs in zip(queries_relevant_docs, queries_retrieved_docs):
            ap = RetrievalMetrics.average_precision(relevant_docs, retrieved_docs)
            ap_scores.append(ap)
        
        return sum(ap_scores) / len(ap_scores)
    
    @staticmethod
    def average_precision(
        relevant_docs: List[str],
        retrieved_docs: List[str]
    ) -> float:
        """Average Precision 계산"""
        
        if not relevant_docs or not retrieved_docs:
            return 0.0
        
        relevant_set = set(relevant_docs)
        
        precision_sum = 0.0
        relevant_count = 0
        
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_set:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i
        
        if relevant_count == 0:
            return 0.0
        
        return precision_sum / len(relevant_docs)


class GenerationMetrics:
    """텍스트 생성 품질 평가 메트릭"""
    
    def __init__(self, language: str = "korean"):
        self.language = language
        
        # 한국어 토크나이저 초기화
        self.korean_tokenizer = None
        if language == "korean" and HAS_KONLPY:
            try:
                self.korean_tokenizer = Okt()
                logger.info("한국어 토크나이저 (Okt) 초기화")
            except Exception as e:
                logger.warning(f"한국어 토크나이저 초기화 실패: {e}")
        
        # ROUGE scorer 초기화
        self.rouge_scorer = None
        if HAS_ROUGE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'], 
                use_stemmer=True
            )
        
        # 의미적 유사도를 위한 모델
        self.similarity_model = None
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                if language == "korean":
                    model_name = "jhgan/ko-sroberta-multitask"
                else:
                    model_name = "all-MiniLM-L6-v2"
                
                self.similarity_model = SentenceTransformer(model_name)
                logger.info(f"의미적 유사도 모델 로드: {model_name}")
            except Exception as e:
                logger.warning(f"의미적 유사도 모델 로드 실패: {e}")
    
    def tokenize_text(self, text: str) -> List[str]:
        """언어별 토큰화"""
        
        if self.language == "korean" and self.korean_tokenizer:
            try:
                return self.korean_tokenizer.morphs(text)
            except Exception:
                pass
        
        # 기본 토큰화
        if self.language == "korean":
            # 한글, 영문, 숫자 추출
            tokens = re.findall(r'[가-힣a-zA-Z0-9]+', text)
        else:
            # 영문, 숫자 추출
            tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())
        
        return tokens
    
    def bleu_score(
        self,
        reference: str,
        candidate: str,
        max_n: int = 4
    ) -> Dict[str, float]:
        """BLEU 점수 계산"""
        
        if not HAS_NLTK:
            logger.warning("NLTK가 없어 간단한 BLEU 구현 사용")
            return self._simple_bleu(reference, candidate, max_n)
        
        try:
            ref_tokens = self.tokenize_text(reference)
            cand_tokens = self.tokenize_text(candidate)
            
            if not ref_tokens or not cand_tokens:
                return {f"bleu_{i}": 0.0 for i in range(1, max_n + 1)}
            
            smoothing = SmoothingFunction().method1
            scores = {}
            
            for n in range(1, max_n + 1):
                weights = tuple([1.0/n] * n + [0.0] * (4-n))
                score = sentence_bleu(
                    [ref_tokens], 
                    cand_tokens, 
                    weights=weights,
                    smoothing_function=smoothing
                )
                scores[f"bleu_{n}"] = score
            
            return scores
            
        except Exception as e:
            logger.warning(f"BLEU 계산 오류: {e}")
            return self._simple_bleu(reference, candidate, max_n)
    
    def _simple_bleu(self, reference: str, candidate: str, max_n: int = 4) -> Dict[str, float]:
        """간단한 BLEU 구현"""
        
        ref_tokens = self.tokenize_text(reference)
        cand_tokens = self.tokenize_text(candidate)
        
        if not ref_tokens or not cand_tokens:
            return {f"bleu_{i}": 0.0 for i in range(1, max_n + 1)}
        
        scores = {}
        
        for n in range(1, max_n + 1):
            # n-gram 생성
            ref_ngrams = Counter([tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1)])
            cand_ngrams = Counter([tuple(cand_tokens[i:i+n]) for i in range(len(cand_tokens)-n+1)])
            
            # 매칭되는 n-gram 개수
            matches = sum((ref_ngrams & cand_ngrams).values())
            total = sum(cand_ngrams.values())
            
            if total == 0:
                scores[f"bleu_{n}"] = 0.0
            else:
                scores[f"bleu_{n}"] = matches / total
        
        return scores
    
    def rouge_score(
        self,
        reference: str,
        candidate: str
    ) -> Dict[str, float]:
        """ROUGE 점수 계산"""
        
        if not HAS_ROUGE:
            logger.warning("rouge-score 패키지가 없어 간단한 ROUGE 구현 사용")
            return self._simple_rouge(reference, candidate)
        
        try:
            scores = self.rouge_scorer.score(reference, candidate)
            
            result = {}
            for metric, score in scores.items():
                result[f"{metric}_precision"] = score.precision
                result[f"{metric}_recall"] = score.recall
                result[f"{metric}_f1"] = score.fmeasure
            
            return result
            
        except Exception as e:
            logger.warning(f"ROUGE 계산 오류: {e}")
            return self._simple_rouge(reference, candidate)
    
    def _simple_rouge(self, reference: str, candidate: str) -> Dict[str, float]:
        """간단한 ROUGE 구현"""
        
        ref_tokens = set(self.tokenize_text(reference))
        cand_tokens = set(self.tokenize_text(candidate))
        
        if not ref_tokens or not cand_tokens:
            return {
                "rouge1_precision": 0.0,
                "rouge1_recall": 0.0,
                "rouge1_f1": 0.0
            }
        
        # ROUGE-1 (unigram overlap)
        overlap = len(ref_tokens & cand_tokens)
        
        precision = overlap / len(cand_tokens) if cand_tokens else 0.0
        recall = overlap / len(ref_tokens) if ref_tokens else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "rouge1_precision": precision,
            "rouge1_recall": recall,
            "rouge1_f1": f1
        }
    
    def bert_score(
        self,
        references: List[str],
        candidates: List[str],
        model_type: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """BERTScore 계산"""
        
        if not HAS_BERTSCORE:
            logger.warning("bert-score 패키지가 없어 의미적 유사도로 대체")
            similarities = []
            for ref, cand in zip(references, candidates):
                sim = self.semantic_similarity(ref, cand)
                similarities.append(sim)
            
            return {
                "precision": similarities,
                "recall": similarities,
                "f1": similarities
            }
        
        try:
            if model_type is None:
                model_type = "klue/roberta-base" if self.language == "korean" else "bert-base-uncased"
            
            P, R, F1 = bert_score(
                candidates, 
                references, 
                model_type=model_type,
                verbose=False
            )
            
            return {
                "precision": P.tolist(),
                "recall": R.tolist(),
                "f1": F1.tolist()
            }
            
        except Exception as e:
            logger.warning(f"BERTScore 계산 오류: {e}")
            # 대체 구현
            similarities = []
            for ref, cand in zip(references, candidates):
                sim = self.semantic_similarity(ref, cand)
                similarities.append(sim)
            
            return {
                "precision": similarities,
                "recall": similarities,
                "f1": similarities
            }
    
    def semantic_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """의미적 유사도 계산"""
        
        if self.similarity_model:
            try:
                embeddings = self.similarity_model.encode([text1, text2])
                
                # 코사인 유사도 계산
                emb1, emb2 = embeddings[0], embeddings[1]
                
                if HAS_NUMPY:
                    cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                else:
                    dot_product = sum(a * b for a, b in zip(emb1, emb2))
                    norm1 = math.sqrt(sum(a * a for a in emb1))
                    norm2 = math.sqrt(sum(a * a for a in emb2))
                    cosine_sim = dot_product / (norm1 * norm2) if norm1 * norm2 > 0 else 0.0
                
                return float(cosine_sim)
                
            except Exception as e:
                logger.warning(f"의미적 유사도 계산 오류: {e}")
        
        # 간단한 토큰 기반 유사도 (Jaccard)
        tokens1 = set(self.tokenize_text(text1))
        tokens2 = set(self.tokenize_text(text2))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return intersection / union if union > 0 else 0.0


class RAGMetrics:
    """종단간 RAG 시스템 평가 메트릭"""
    
    def __init__(
        self,
        language: str = "korean",
        llm_evaluator = None
    ):
        self.language = language
        self.llm_evaluator = llm_evaluator
        self.generation_metrics = GenerationMetrics(language)
        
        logger.info(f"RAG 메트릭 초기화: {language}")
    
    def faithfulness(
        self,
        answer: str,
        context: str,
        question: Optional[str] = None
    ) -> float:
        """Faithfulness (충실성) - 답변이 제공된 컨텍스트에 얼마나 기반하고 있는지"""
        
        if self.llm_evaluator:
            return self._llm_faithfulness(answer, context, question)
        else:
            return self._simple_faithfulness(answer, context)
    
    def _llm_faithfulness(
        self,
        answer: str,
        context: str,
        question: Optional[str] = None
    ) -> float:
        """LLM을 사용한 충실성 평가"""
        
        prompt = f"""다음 답변이 주어진 컨텍스트에 얼마나 충실한지 0.0에서 1.0 사이의 점수로 평가해주세요.

컨텍스트: {context}

답변: {answer}

평가 기준:
1.0 - 답변의 모든 내용이 컨텍스트에서 직접 확인 가능
0.8 - 답변의 대부분이 컨텍스트에 기반하고 있음
0.6 - 답변의 일부가 컨텍스트에 기반하고 있음
0.4 - 답변이 컨텍스트와 약간 관련이 있음
0.2 - 답변이 컨텍스트와 거의 관련이 없음
0.0 - 답변이 컨텍스트와 전혀 관련이 없음

점수만 숫자로 답해주세요 (예: 0.8):"""
        
        try:
            response = self.llm_evaluator.generate(prompt)
            score = float(response.content.strip())
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"LLM 충실성 평가 오류: {e}")
            return self._simple_faithfulness(answer, context)
    
    def _simple_faithfulness(self, answer: str, context: str) -> float:
        """간단한 충실성 평가 (토큰 기반)"""
        
        answer_tokens = set(self.generation_metrics.tokenize_text(answer))
        context_tokens = set(self.generation_metrics.tokenize_text(context))
        
        if not answer_tokens:
            return 0.0
        
        # 답변 토큰 중 컨텍스트에 있는 비율
        overlap = len(answer_tokens & context_tokens)
        return overlap / len(answer_tokens)
    
    def answer_relevancy(
        self,
        answer: str,
        question: str,
        context: Optional[str] = None
    ) -> float:
        """Answer Relevancy (답변 관련성) - 답변이 질문에 얼마나 관련이 있는지"""
        
        if self.llm_evaluator:
            return self._llm_answer_relevancy(answer, question, context)
        else:
            return self._simple_answer_relevancy(answer, question)
    
    def _llm_answer_relevancy(
        self,
        answer: str,
        question: str,
        context: Optional[str] = None
    ) -> float:
        """LLM을 사용한 답변 관련성 평가"""
        
        context_text = f"\n컨텍스트: {context}" if context else ""
        
        prompt = f"""다음 답변이 질문에 얼마나 관련이 있는지 0.0에서 1.0 사이의 점수로 평가해주세요.

질문: {question}{context_text}

답변: {answer}

평가 기준:
1.0 - 답변이 질문에 완벽하게 대답함
0.8 - 답변이 질문에 잘 대답하지만 약간의 부족함이 있음
0.6 - 답변이 질문에 부분적으로 대답함
0.4 - 답변이 질문과 약간 관련이 있음
0.2 - 답변이 질문과 거의 관련이 없음
0.0 - 답변이 질문과 전혀 관련이 없음

점수만 숫자로 답해주세요 (예: 0.8):"""
        
        try:
            response = self.llm_evaluator.generate(prompt)
            score = float(response.content.strip())
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"LLM 답변 관련성 평가 오류: {e}")
            return self._simple_answer_relevancy(answer, question)
    
    def _simple_answer_relevancy(self, answer: str, question: str) -> float:
        """간단한 답변 관련성 평가 (의미적 유사도)"""
        
        return self.generation_metrics.semantic_similarity(answer, question)
    
    def context_precision(
        self,
        retrieved_contexts: List[str],
        ground_truth_contexts: List[str],
        question: str
    ) -> float:
        """Context Precision - 검색된 컨텍스트의 정밀도"""
        
        if not retrieved_contexts:
            return 0.0
        
        relevant_count = 0
        
        for context in retrieved_contexts:
            # 각 컨텍스트가 질문과 관련이 있는지 확인
            relevance = self._context_relevance(context, question, ground_truth_contexts)
            if relevance > 0.5:  # 임계값
                relevant_count += 1
        
        return relevant_count / len(retrieved_contexts)
    
    def context_recall(
        self,
        retrieved_contexts: List[str],
        ground_truth_contexts: List[str],
        question: str
    ) -> float:
        """Context Recall - 검색된 컨텍스트의 재현율"""
        
        if not ground_truth_contexts:
            return 0.0
        
        covered_count = 0
        
        for gt_context in ground_truth_contexts:
            # 각 정답 컨텍스트가 검색 결과에 포함되어 있는지 확인
            max_similarity = 0.0
            for ret_context in retrieved_contexts:
                similarity = self.generation_metrics.semantic_similarity(gt_context, ret_context)
                max_similarity = max(max_similarity, similarity)
            
            if max_similarity > 0.5:  # 임계값
                covered_count += 1
        
        return covered_count / len(ground_truth_contexts)
    
    def _context_relevance(
        self,
        context: str,
        question: str,
        ground_truth_contexts: List[str]
    ) -> float:
        """컨텍스트의 질문 관련성 평가"""
        
        # 질문과의 직접적 유사도
        question_similarity = self.generation_metrics.semantic_similarity(context, question)
        
        # 정답 컨텍스트들과의 유사도
        gt_similarities = [
            self.generation_metrics.semantic_similarity(context, gt)
            for gt in ground_truth_contexts
        ]
        
        max_gt_similarity = max(gt_similarities) if gt_similarities else 0.0
        
        # 두 유사도의 가중 평균
        return 0.7 * question_similarity + 0.3 * max_gt_similarity


# 헬퍼 함수들
def calculate_similarity(text1: str, text2: str, method: str = "semantic") -> float:
    """텍스트 간 유사도 계산"""
    
    metrics = GenerationMetrics()
    
    if method == "semantic":
        return metrics.semantic_similarity(text1, text2)
    elif method == "token":
        tokens1 = set(metrics.tokenize_text(text1))
        tokens2 = set(metrics.tokenize_text(text2))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return intersection / union if union > 0 else 0.0
    else:
        raise ValueError(f"지원하지 않는 유사도 방법: {method}")


def calculate_bleu_score(reference: str, candidate: str, language: str = "korean") -> Dict[str, float]:
    """BLEU 점수 계산 헬퍼 함수"""
    
    metrics = GenerationMetrics(language)
    return metrics.bleu_score(reference, candidate)


def calculate_rouge_score(reference: str, candidate: str, language: str = "korean") -> Dict[str, float]:
    """ROUGE 점수 계산 헬퍼 함수"""
    
    metrics = GenerationMetrics(language)
    return metrics.rouge_score(reference, candidate)


def calculate_bertscore(
    references: List[str],
    candidates: List[str],
    language: str = "korean"
) -> Dict[str, List[float]]:
    """BERTScore 계산 헬퍼 함수"""
    
    metrics = GenerationMetrics(language)
    return metrics.bert_score(references, candidates)