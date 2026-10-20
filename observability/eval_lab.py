"""RAG Evaluation Lab with deterministic metrics, benchmark runner, and historical comparisons."""

import re
import io
import csv
import json
import time
import uuid
import logging
from typing import List, Dict, Any, Optional
import numpy as np

from generation.qa_chain import BookRAGPipeline
from observability.models import EvalRun, EvalBenchmarkItem
from observability.storage import storage
from observability.citation_verifier import citation_verifier

logger = logging.getLogger(__name__)


class EvaluationLab:
    """Benchmark runner computing deterministic retrieval and generation reliability metrics."""

    def __init__(self, pipeline: Optional[BookRAGPipeline] = None):
        self.pipeline = pipeline or BookRAGPipeline()
        self.storage = storage

    def run_benchmark(
        self,
        benchmark_name: str = "quantum_bench",
        strategy: str = "reranked",
        top_k: int = 5,
        max_items: Optional[int] = None
    ) -> EvalRun:
        """Execute a full benchmark evaluation and record metrics."""
        items = self.storage.load_benchmark(benchmark_name)
        if not items:
            raise ValueError(f"Benchmark '{benchmark_name}' not found or empty.")

        if max_items:
            items = items[:max_items]

        run_id = f"eval_{uuid.uuid4().hex[:8]}"
        latencies_ms: List[float] = []
        hits_count = 0
        reciprocal_ranks: List[float] = []
        precision_scores: List[float] = []
        citation_scores: List[float] = []
        groundedness_scores: List[float] = []
        abstention_correct = 0
        abstention_total = 0
        correctness_scores: List[float] = []
        stage_latencies_collector: Dict[str, List[float]] = {}
        error_count = 0

        per_query_results: List[Dict[str, Any]] = []

        for idx, item in enumerate(items, 1):
            t0 = time.time()
            is_error = False
            err_msg = None
            try:
                response = self.pipeline.query(
                    question=item.question,
                    top_k=top_k,
                    strategy=strategy
                )
            except Exception as e:
                is_error = True
                err_msg = str(e)
                error_count += 1
                latencies_ms.append(round((time.time() - t0) * 1000.0, 2))
                per_query_results.append({
                    "id": item.id,
                    "question": item.question,
                    "error": err_msg,
                    "is_hit": False,
                    "mrr": 0.0
                })
                continue

            dur_ms = round((time.time() - t0) * 1000.0, 2)
            latencies_ms.append(dur_ms)

            # Extract per-stage durations from trace if available
            if response.trace_id:
                tr = self.storage.get_trace(response.trace_id)
                if tr and tr.steps:
                    for stp in tr.steps:
                        if stp.name not in stage_latencies_collector:
                            stage_latencies_collector[stp.name] = []
                        stage_latencies_collector[stp.name].append(stp.duration_ms)

            # 1. Retrieval Metrics
            retrieved_pages = [int(h.get("metadata", {}).get("page", 0)) for h in response.retrieved_chunks]
            retrieved_cids = [h.get("chunk_id", "") for h in response.retrieved_chunks]
            exp_pages = set(item.expected_pages)
            exp_cids = set(item.expected_chunk_ids)

            # Hit rate & MRR
            first_rank = 0
            if exp_pages or exp_cids:
                for r, (p, cid) in enumerate(zip(retrieved_pages, retrieved_cids), 1):
                    if (exp_pages and p in exp_pages) or (exp_cids and cid in exp_cids):
                        first_rank = r
                        break

            is_hit = first_rank > 0
            if is_hit:
                hits_count += 1
                rr = 1.0 / first_rank
            else:
                rr = 0.0
            reciprocal_ranks.append(rr)

            # Precision@k
            if top_k > 0 and (exp_pages or exp_cids):
                matching_hits = sum(1 for p, cid in zip(retrieved_pages[:top_k], retrieved_cids[:top_k]) if (exp_pages and p in exp_pages) or (exp_cids and cid in exp_cids))
                precision_scores.append(matching_hits / top_k)

            # 2. Citation and Abstention Auditing
            audit = citation_verifier.audit_response(
                question=item.question,
                answer=response.answer,
                retrieved_chunks=response.retrieved_chunks,
                should_abstain=item.should_abstain
            )

            # Citation accuracy
            if audit.detected_citations:
                c_acc = len(audit.verified_citations) / len(audit.detected_citations)
                citation_scores.append(c_acc)

            # Groundedness estimate (token overlap with context)
            ans_tokens = set(re.findall(r"\b\w+\b", response.answer.lower()))
            ctx_tokens = set(re.findall(r"\b\w+\b", " ".join(c.get("content", "") for c in response.retrieved_chunks).lower()))
            if ans_tokens and ctx_tokens:
                overlap = len(ans_tokens.intersection(ctx_tokens)) / len(ans_tokens)
                groundedness_scores.append(min(1.0, overlap * 1.5))

            # Answer Correctness Metric
            if item.should_abstain:
                abstention_total += 1
                if audit.abstained_properly:
                    abstention_correct += 1
                    correctness_scores.append(1.0)
                else:
                    correctness_scores.append(0.0)
            elif item.expected_answer:
                exp_tokens = set(re.findall(r"\b\w+\b", item.expected_answer.lower()))
                if ans_tokens and exp_tokens:
                    c_overlap = len(ans_tokens.intersection(exp_tokens))
                    p_score = c_overlap / len(ans_tokens) if ans_tokens else 0.0
                    r_score = c_overlap / len(exp_tokens) if exp_tokens else 0.0
                    f1 = (2.0 * p_score * r_score) / (p_score + r_score) if (p_score + r_score) > 0 else 0.0
                    correctness_scores.append(f1)
                else:
                    correctness_scores.append(1.0 if is_hit else 0.0)
            else:
                correctness_scores.append(1.0 if is_hit else 0.0)

            per_query_results.append({
                "id": item.id,
                "question": item.question,
                "expected_pages": item.expected_pages,
                "expected_answer": item.expected_answer,
                "retrieved_pages": retrieved_pages,
                "is_hit": is_hit,
                "first_hit_rank": first_rank,
                "reciprocal_rank": rr,
                "answer_preview": response.answer[:120],
                "audit_verdict": audit.audit_verdict,
                "duration_ms": dur_ms,
                "should_abstain": item.should_abstain,
                "abstained_properly": audit.abstained_properly
            })

        # Calculate final aggregated metrics
        total = len(items)
        recall = (hits_count / total) * 100.0 if total > 0 else 0.0
        prec = (np.mean(precision_scores) * 100.0) if precision_scores else 0.0
        mean_mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
        cit_acc = (np.mean(citation_scores) * 100.0) if citation_scores else 100.0
        groundedness = (np.mean(groundedness_scores) * 100.0) if groundedness_scores else 0.0
        abst_acc = (abstention_correct / abstention_total * 100.0) if abstention_total > 0 else 100.0
        avg_correctness = (np.mean(correctness_scores) * 100.0) if correctness_scores else 0.0
        avg_lat = float(np.mean(latencies_ms)) if latencies_ms else 0.0
        p95_lat = float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0
        err_rate = (error_count / total * 100.0) if total > 0 else 0.0

        stage_averages = {
            stg: round(float(np.mean(vals)), 2)
            for stg, vals in stage_latencies_collector.items()
        }

        from config import settings
        eval_run = EvalRun(
            run_id=run_id,
            benchmark_name=benchmark_name,
            dataset_version="v2.0",
            configuration={
                "strategy": strategy,
                "top_k": top_k,
                "model": settings.llm_model,
                "provider": settings.llm_provider,
                "embedding_model": settings.embedding_model,
                "reranker_model": settings.reranker_model
            },
            total_queries=total,
            recall_at_k=round(recall, 2),
            precision_at_k=round(prec, 2),
            mrr=round(mean_mrr, 4),
            correctness_score=round(avg_correctness, 2),
            groundedness_score=round(groundedness, 2),
            citation_accuracy=round(cit_acc, 2),
            abstention_accuracy=round(abst_acc, 2),
            avg_latency_ms=round(avg_lat, 2),
            p95_latency_ms=round(p95_lat, 2),
            stage_latencies_avg_ms=stage_averages,
            error_rate=round(err_rate, 2),
            strategy=strategy,
            top_k=top_k,
            per_query_results=per_query_results
        )

        self.storage.save_eval_run(eval_run)
        logger.info("Evaluation run %s finished with Recall@%d: %.2f%%, MRR: %.4f, Correctness: %.2f%%", run_id, top_k, recall, mean_mrr, avg_correctness)
        return eval_run

    @staticmethod
    def export_benchmark_to_csv(items: List[EvalBenchmarkItem]) -> str:
        """Export benchmark items to CSV format."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["id", "question", "expected_pages", "topic", "should_abstain"])
        for item in items:
            writer.writerow([
                item.id,
                item.question,
                ";".join(map(str, item.expected_pages)),
                item.topic or "",
                str(item.should_abstain)
            ])
        return output.getvalue()

    @staticmethod
    def import_benchmark_from_csv(csv_text: str) -> List[EvalBenchmarkItem]:
        """Import benchmark items from CSV string."""
        f = io.StringIO(csv_text.strip())
        reader = csv.DictReader(f)
        items: List[EvalBenchmarkItem] = []
        for row in reader:
            p_str = row.get("expected_pages", "")
            pages = [int(p.strip()) for p in p_str.split(";") if p.strip().isdigit()]
            abst = row.get("should_abstain", "").lower() in ("true", "1", "yes")
            items.append(
                EvalBenchmarkItem(
                    id=row.get("id", f"bench_{len(items)+1:03d}"),
                    question=row.get("question", "").strip(),
                    expected_pages=pages,
                    topic=row.get("topic", "").strip(),
                    should_abstain=abst
                )
            )
        return items


eval_lab = EvaluationLab()
