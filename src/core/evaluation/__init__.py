"""
Evaluation Package
RAG 시스템의 성능을 평가하는 패키지
"""

from .metrics import (
    RAGMetrics,
    RetrievalMetrics,
    GenerationMetrics,
    calculate_similarity,
    calculate_bleu_score,
    calculate_rouge_score,
    calculate_bertscore
)

from .rag_evaluator import (
    RAGEvaluator,
    EvaluationResult,
    EvaluationDataset,
    create_evaluation_dataset
)

from .benchmark import (
    RAGBenchmark,
    BenchmarkResult,
    BenchmarkSuite,
    create_benchmark_suite
)

__all__ = [
    # Metrics
    "RAGMetrics",
    "RetrievalMetrics", 
    "GenerationMetrics",
    "calculate_similarity",
    "calculate_bleu_score",
    "calculate_rouge_score",
    "calculate_bertscore",
    
    # Evaluation
    "RAGEvaluator",
    "EvaluationResult",
    "EvaluationDataset", 
    "create_evaluation_dataset",
    
    # Benchmark
    "RAGBenchmark",
    "BenchmarkResult",
    "BenchmarkSuite",
    "create_benchmark_suite",
]


def get_evaluation_info():
    """평가 시스템 정보 반환"""
    return {
        "metrics": {
            "retrieval": ["precision@k", "recall@k", "ndcg@k", "mrr", "map"],
            "generation": ["bleu", "rouge", "bertscore", "semantic_similarity"],
            "end_to_end": ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        },
        "evaluation_types": ["retrieval_only", "generation_only", "end_to_end"],
        "benchmark_datasets": ["korean_qa", "english_qa", "multilingual", "domain_specific"],
        "supported_languages": ["korean", "english", "multilingual"]
    }