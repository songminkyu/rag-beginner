"""
Benchmark Module
다양한 RAG 시스템을 비교 평가하는 벤치마크 도구
"""

import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from .rag_evaluator import RAGEvaluator, EvaluationDataset, EvaluationResult

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """벤치마크 결과 클래스"""
    benchmark_name: str
    systems: List[str]
    datasets: List[str]
    results: Dict[str, Dict[str, EvaluationResult]]  # system -> dataset -> result
    comparison_metrics: Dict[str, Dict[str, float]]  # metric -> system -> score
    rankings: Dict[str, List[str]]  # metric -> ranked_systems
    benchmark_time: float
    metadata: Optional[Dict[str, Any]] = None
    
    def get_best_system(self, metric: str) -> Optional[str]:
        """특정 메트릭에서 최고 성능 시스템 반환"""
        
        if metric not in self.rankings:
            return None
        
        return self.rankings[metric][0] if self.rankings[metric] else None
    
    def get_system_ranking(self, system: str, metric: str) -> Optional[int]:
        """특정 시스템의 메트릭별 순위 반환 (1부터 시작)"""
        
        if metric not in self.rankings or system not in self.rankings[metric]:
            return None
        
        return self.rankings[metric].index(system) + 1
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        
        # EvaluationResult를 dict로 변환
        results_dict = {}
        for system, dataset_results in self.results.items():
            results_dict[system] = {}
            for dataset, result in dataset_results.items():
                results_dict[system][dataset] = result.to_dict()
        
        return {
            "benchmark_name": self.benchmark_name,
            "systems": self.systems,
            "datasets": self.datasets,
            "results": results_dict,
            "comparison_metrics": self.comparison_metrics,
            "rankings": self.rankings,
            "benchmark_time": self.benchmark_time,
            "metadata": self.metadata
        }
    
    def save(self, filepath: Union[str, Path]):
        """벤치마크 결과를 JSON 파일로 저장"""
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"벤치마크 결과 저장: {filepath}")


class RAGBenchmark:
    """RAG 시스템 벤치마크"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.benchmark_name = config.get("benchmark_name", "RAG_Benchmark")
        self.parallel_execution = config.get("parallel_execution", False)
        self.max_workers = config.get("max_workers", 2)
        
        # 비교할 메트릭 목록
        self.comparison_metrics = config.get("comparison_metrics", [
            "precision@5", "recall@5", "ndcg@5", "mrr",
            "bleu_1", "rouge1_f1", "semantic_similarity",
            "faithfulness", "answer_relevancy", "context_precision", "context_recall"
        ])
        
        logger.info(f"RAG 벤치마크 초기화: {self.benchmark_name}")
    
    def run_benchmark(
        self,
        systems: Dict[str, RAGEvaluator],
        datasets: Dict[str, EvaluationDataset],
        save_individual_results: bool = True,
        results_dir: Optional[Union[str, Path]] = None
    ) -> BenchmarkResult:
        """벤치마크 실행"""
        
        start_time = time.time()
        logger.info(f"벤치마크 시작: {len(systems)}개 시스템, {len(datasets)}개 데이터셋")
        
        if results_dir:
            results_dir = Path(results_dir)
            results_dir.mkdir(parents=True, exist_ok=True)
        
        # 평가 실행
        all_results = {}
        
        if self.parallel_execution:
            all_results = self._run_parallel_evaluation(systems, datasets)
        else:
            all_results = self._run_sequential_evaluation(systems, datasets)
        
        # 개별 결과 저장
        if save_individual_results and results_dir:
            self._save_individual_results(all_results, results_dir)
        
        # 비교 메트릭 계산
        comparison_metrics = self._calculate_comparison_metrics(all_results)
        
        # 순위 계산
        rankings = self._calculate_rankings(comparison_metrics)
        
        benchmark_time = time.time() - start_time
        
        benchmark_result = BenchmarkResult(
            benchmark_name=self.benchmark_name,
            systems=list(systems.keys()),
            datasets=list(datasets.keys()),
            results=all_results,
            comparison_metrics=comparison_metrics,
            rankings=rankings,
            benchmark_time=benchmark_time,
            metadata={
                "config": self.config,
                "comparison_metrics": self.comparison_metrics,
                "parallel_execution": self.parallel_execution
            }
        )
        
        logger.info(f"벤치마크 완료: {benchmark_time:.2f}초")
        return benchmark_result
    
    def _run_sequential_evaluation(
        self,
        systems: Dict[str, RAGEvaluator],
        datasets: Dict[str, EvaluationDataset]
    ) -> Dict[str, Dict[str, EvaluationResult]]:
        """순차 평가 실행"""
        
        all_results = {}
        total_evaluations = len(systems) * len(datasets)
        current_eval = 0
        
        for system_name, evaluator in systems.items():
            all_results[system_name] = {}
            
            for dataset_name, dataset in datasets.items():
                current_eval += 1
                logger.info(f"평가 진행 ({current_eval}/{total_evaluations}): {system_name} - {dataset_name}")
                
                try:
                    result = evaluator.evaluate(dataset, dataset_name)
                    all_results[system_name][dataset_name] = result
                    
                except Exception as e:
                    logger.error(f"평가 실패 ({system_name} - {dataset_name}): {e}")
                    # 빈 결과로 대체
                    empty_result = EvaluationResult(
                        dataset_name=dataset_name,
                        total_questions=len(dataset),
                        evaluation_time=0.0,
                        retrieval_metrics={},
                        generation_metrics={},
                        rag_metrics={},
                        per_question_results=[],
                        metadata={"error": str(e)}
                    )
                    all_results[system_name][dataset_name] = empty_result
        
        return all_results
    
    def _run_parallel_evaluation(
        self,
        systems: Dict[str, RAGEvaluator],
        datasets: Dict[str, EvaluationDataset]
    ) -> Dict[str, Dict[str, EvaluationResult]]:
        """병렬 평가 실행"""
        
        all_results = {system_name: {} for system_name in systems.keys()}
        
        # 평가 작업 리스트 생성
        evaluation_tasks = []
        for system_name, evaluator in systems.items():
            for dataset_name, dataset in datasets.items():
                evaluation_tasks.append((system_name, evaluator, dataset_name, dataset))
        
        # 병렬 실행
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 작업 제출
            future_to_task = {}
            for task in evaluation_tasks:
                future = executor.submit(self._evaluate_single, *task)
                future_to_task[future] = task
            
            # 결과 수집
            completed = 0
            for future in as_completed(future_to_task):
                system_name, _, dataset_name, _ = future_to_task[future]
                completed += 1
                
                try:
                    result = future.result()
                    all_results[system_name][dataset_name] = result
                    logger.info(f"평가 완료 ({completed}/{len(evaluation_tasks)}): {system_name} - {dataset_name}")
                    
                except Exception as e:
                    logger.error(f"평가 실패 ({system_name} - {dataset_name}): {e}")
                    # 빈 결과로 대체
                    empty_result = EvaluationResult(
                        dataset_name=dataset_name,
                        total_questions=0,
                        evaluation_time=0.0,
                        retrieval_metrics={},
                        generation_metrics={},
                        rag_metrics={},
                        per_question_results=[],
                        metadata={"error": str(e)}
                    )
                    all_results[system_name][dataset_name] = empty_result
        
        return all_results
    
    def _evaluate_single(
        self,
        system_name: str,
        evaluator: RAGEvaluator,
        dataset_name: str,
        dataset: EvaluationDataset
    ) -> EvaluationResult:
        """단일 평가 실행 (병렬 처리용)"""
        
        return evaluator.evaluate(dataset, dataset_name)
    
    def _save_individual_results(
        self,
        all_results: Dict[str, Dict[str, EvaluationResult]],
        results_dir: Path
    ):
        """개별 결과 저장"""
        
        for system_name, dataset_results in all_results.items():
            system_dir = results_dir / system_name
            system_dir.mkdir(exist_ok=True)
            
            for dataset_name, result in dataset_results.items():
                result_file = system_dir / f"{dataset_name}_result.json"
                result.save(result_file)
    
    def _calculate_comparison_metrics(
        self,
        all_results: Dict[str, Dict[str, EvaluationResult]]
    ) -> Dict[str, Dict[str, float]]:
        """비교 메트릭 계산"""
        
        comparison_metrics = {}
        
        for metric_name in self.comparison_metrics:
            comparison_metrics[metric_name] = {}
            
            for system_name, dataset_results in all_results.items():
                scores = []
                
                for dataset_name, result in dataset_results.items():
                    # 메트릭 값 추출
                    score = self._extract_metric_value(result, metric_name)
                    if score is not None:
                        scores.append(score)
                
                # 평균 점수 계산
                if scores:
                    avg_score = sum(scores) / len(scores)
                    comparison_metrics[metric_name][system_name] = avg_score
                else:
                    comparison_metrics[metric_name][system_name] = 0.0
        
        return comparison_metrics
    
    def _extract_metric_value(
        self,
        result: EvaluationResult,
        metric_name: str
    ) -> Optional[float]:
        """평가 결과에서 특정 메트릭 값 추출"""
        
        # 검색 메트릭에서 찾기
        if metric_name in result.retrieval_metrics:
            return result.retrieval_metrics[metric_name]
        
        # 생성 메트릭에서 찾기
        if metric_name in result.generation_metrics:
            return result.generation_metrics[metric_name]
        
        # RAG 메트릭에서 찾기
        if metric_name in result.rag_metrics:
            return result.rag_metrics[metric_name]
        
        return None
    
    def _calculate_rankings(
        self,
        comparison_metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, List[str]]:
        """시스템 순위 계산"""
        
        rankings = {}
        
        for metric_name, system_scores in comparison_metrics.items():
            # 점수순으로 정렬 (내림차순)
            sorted_systems = sorted(
                system_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            rankings[metric_name] = [system_name for system_name, _ in sorted_systems]
        
        return rankings
    
    def generate_report(
        self,
        benchmark_result: BenchmarkResult,
        output_file: Optional[Union[str, Path]] = None
    ) -> str:
        """벤치마크 리포트 생성"""
        
        report_lines = []
        
        # 헤더
        report_lines.append(f"# {benchmark_result.benchmark_name} 벤치마크 리포트")
        report_lines.append("")
        report_lines.append(f"**벤치마크 시간**: {benchmark_result.benchmark_time:.2f}초")
        report_lines.append(f"**평가 시스템**: {len(benchmark_result.systems)}개")
        report_lines.append(f"**평가 데이터셋**: {len(benchmark_result.datasets)}개")
        report_lines.append("")
        
        # 전체 순위표
        report_lines.append("## 📊 종합 순위")
        report_lines.append("")
        
        # 주요 메트릭별 순위
        main_metrics = ["faithfulness", "answer_relevancy", "precision@5", "rouge1_f1"]
        
        for metric in main_metrics:
            if metric in benchmark_result.rankings:
                report_lines.append(f"### {metric}")
                for i, system in enumerate(benchmark_result.rankings[metric], 1):
                    score = benchmark_result.comparison_metrics[metric][system]
                    report_lines.append(f"{i}. **{system}**: {score:.4f}")
                report_lines.append("")
        
        # 상세 결과 테이블
        report_lines.append("## 📈 상세 결과")
        report_lines.append("")
        
        # 메트릭별 테이블
        for metric_name in self.comparison_metrics:
            if metric_name in benchmark_result.comparison_metrics:
                report_lines.append(f"### {metric_name}")
                report_lines.append("")
                
                # 테이블 헤더
                header = "| 시스템 | 점수 | 순위 |"
                separator = "|--------|------|------|"
                report_lines.append(header)
                report_lines.append(separator)
                
                # 테이블 내용
                for i, system in enumerate(benchmark_result.rankings[metric_name], 1):
                    score = benchmark_result.comparison_metrics[metric_name][system]
                    report_lines.append(f"| {system} | {score:.4f} | {i} |")
                
                report_lines.append("")
        
        # 시스템별 상세 분석
        report_lines.append("## 🔍 시스템별 분석")
        report_lines.append("")
        
        for system_name in benchmark_result.systems:
            report_lines.append(f"### {system_name}")
            report_lines.append("")
            
            # 강점과 약점 분석
            strengths = []
            weaknesses = []
            
            for metric_name in self.comparison_metrics:
                if metric_name in benchmark_result.rankings:
                    rank = benchmark_result.get_system_ranking(system_name, metric_name)
                    if rank and rank <= 2:  # 상위 2위 이내
                        strengths.append(f"{metric_name} ({rank}위)")
                    elif rank and rank >= len(benchmark_result.systems) - 1:  # 하위 2위 이내
                        weaknesses.append(f"{metric_name} ({rank}위)")
            
            if strengths:
                report_lines.append(f"**강점**: {', '.join(strengths)}")
            if weaknesses:
                report_lines.append(f"**약점**: {', '.join(weaknesses)}")
            
            report_lines.append("")
        
        # 전체 리포트 결합
        report = "\n".join(report_lines)
        
        # 파일 저장
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"벤치마크 리포트 저장: {output_file}")
        
        return report


class BenchmarkSuite:
    """벤치마크 스위트 - 여러 벤치마크를 관리"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.suite_name = config.get("suite_name", "RAG_Benchmark_Suite")
        self.benchmarks: List[RAGBenchmark] = []
        
        logger.info(f"벤치마크 스위트 초기화: {self.suite_name}")
    
    def add_benchmark(self, benchmark: RAGBenchmark):
        """벤치마크 추가"""
        self.benchmarks.append(benchmark)
        logger.info(f"벤치마크 추가: {benchmark.benchmark_name}")
    
    def run_all_benchmarks(
        self,
        systems: Dict[str, RAGEvaluator],
        datasets: Dict[str, EvaluationDataset],
        results_dir: Optional[Union[str, Path]] = None
    ) -> List[BenchmarkResult]:
        """모든 벤치마크 실행"""
        
        if results_dir:
            results_dir = Path(results_dir)
            results_dir.mkdir(parents=True, exist_ok=True)
        
        all_results = []
        
        for i, benchmark in enumerate(self.benchmarks, 1):
            logger.info(f"벤치마크 실행 ({i}/{len(self.benchmarks)}): {benchmark.benchmark_name}")
            
            # 벤치마크별 결과 디렉토리
            benchmark_results_dir = None
            if results_dir:
                benchmark_results_dir = results_dir / benchmark.benchmark_name
            
            try:
                result = benchmark.run_benchmark(
                    systems, 
                    datasets,
                    save_individual_results=True,
                    results_dir=benchmark_results_dir
                )
                all_results.append(result)
                
                # 벤치마크 결과 저장
                if results_dir:
                    result_file = results_dir / f"{benchmark.benchmark_name}_benchmark.json"
                    result.save(result_file)
                
            except Exception as e:
                logger.error(f"벤치마크 실행 실패 ({benchmark.benchmark_name}): {e}")
        
        logger.info(f"벤치마크 스위트 완료: {len(all_results)}개 벤치마크")
        return all_results


def create_benchmark_suite(
    benchmark_configs: List[Dict[str, Any]],
    suite_config: Optional[Dict[str, Any]] = None
) -> BenchmarkSuite:
    """벤치마크 스위트 생성 헬퍼 함수"""
    
    suite_config = suite_config or {}
    suite = BenchmarkSuite(suite_config)
    
    for config in benchmark_configs:
        benchmark = RAGBenchmark(config)
        suite.add_benchmark(benchmark)
    
    return suite


def create_standard_benchmark_suite() -> BenchmarkSuite:
    """표준 벤치마크 스위트 생성"""
    
    benchmark_configs = [
        {
            "benchmark_name": "retrieval_benchmark",
            "comparison_metrics": [
                "precision@1", "precision@3", "precision@5",
                "recall@1", "recall@3", "recall@5",
                "ndcg@5", "mrr"
            ]
        },
        {
            "benchmark_name": "generation_benchmark", 
            "comparison_metrics": [
                "bleu_1", "bleu_2", "bleu_4",
                "rouge1_f1", "rouge2_f1", "rougeL_f1",
                "semantic_similarity"
            ]
        },
        {
            "benchmark_name": "end_to_end_benchmark",
            "comparison_metrics": [
                "faithfulness", "answer_relevancy",
                "context_precision", "context_recall"
            ]
        }
    ]
    
    suite_config = {
        "suite_name": "Standard_RAG_Benchmark_Suite"
    }
    
    return create_benchmark_suite(benchmark_configs, suite_config)