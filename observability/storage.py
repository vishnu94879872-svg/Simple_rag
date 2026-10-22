"""Persistent storage for BookRAG Reliability Lab observability data."""

import json
import logging
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional

from config import settings
from observability.models import (
    IngestionJob,
    PipelineTrace,
    EvalRun,
    EvalBenchmarkItem,
    ChunkExperimentResult,
    FailureInvestigationReport,
)

logger = logging.getLogger(__name__)


def default_benchmark_items() -> List[EvalBenchmarkItem]:
    """Build the bundled quantum benchmark, including two abstention probes."""
    from eval import EVAL_PAIRS

    items: List[EvalBenchmarkItem] = []
    for i, pair in enumerate(EVAL_PAIRS, 1):
        items.append(
            EvalBenchmarkItem(
                id=f"bench_{i:03d}",
                question=pair["question"],
                expected_pages=[pair["expected_page"]],
                topic=pair.get("topic"),
                should_abstain=False,
            )
        )
    items.append(
        EvalBenchmarkItem(
            id=f"bench_{len(items) + 1:03d}",
            question="What is the capital city of Australia according to the quantum book?",
            expected_pages=[],
            topic="Out-of-Domain Abstention",
            should_abstain=True,
        )
    )
    items.append(
        EvalBenchmarkItem(
            id=f"bench_{len(items) + 1:03d}",
            question="How do black holes evaporate via Hawking radiation?",
            expected_pages=[],
            topic="Unindexed Physics Topic",
            should_abstain=True,
        )
    )
    return items


class ObservabilityStorage:
    """Thread-safe file-backed storage manager for observability records."""

    backend_name = "local"

    _instance = None
    _lock = threading.Lock()

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir or settings.observability_dir)
        self.jobs_dir = self.base_dir / "ingestion_jobs"
        self.traces_dir = self.base_dir / "traces"
        self.evals_dir = self.base_dir / "eval_runs"
        self.benchmarks_dir = self.base_dir / "benchmarks"
        self.experiments_dir = self.base_dir / "experiments"
        self.investigations_dir = self.base_dir / "investigations"

        self._ensure_dirs()
        self._seed_default_benchmark_if_needed()

    def _ensure_dirs(self) -> None:
        for d in [self.jobs_dir, self.traces_dir, self.evals_dir, self.benchmarks_dir, self.experiments_dir, self.investigations_dir]:
            d.mkdir(parents=True, exist_ok=True)

    # --- Ingestion Jobs ---
    def save_job(self, job: IngestionJob) -> None:
        with self._lock:
            path = self.jobs_dir / f"{job.job_id}.json"
            with open(path, "w", encoding="utf-8") as f:
                f.write(job.model_dump_json(indent=2))

    def get_job(self, job_id: str) -> Optional[IngestionJob]:
        path = self.jobs_dir / f"{job_id}.json"
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return IngestionJob.model_validate_json(f.read())
        except Exception as e:
            logger.error("Failed to read job %s: %s", job_id, e)
            return None

    def list_jobs(self, limit: int = 50) -> List[IngestionJob]:
        jobs = []
        paths = sorted(self.jobs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in paths[:limit]:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    jobs.append(IngestionJob.model_validate_json(f.read()))
            except Exception as e:
                logger.error("Error reading job file %s: %s", p, e)
        return jobs

    # --- Pipeline Traces ---
    def save_trace(self, trace: PipelineTrace) -> None:
        with self._lock:
            path = self.traces_dir / f"{trace.trace_id}.json"
            with open(path, "w", encoding="utf-8") as f:
                f.write(trace.model_dump_json(indent=2))

    def get_trace(self, trace_id: str) -> Optional[PipelineTrace]:
        path = self.traces_dir / f"{trace_id}.json"
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return PipelineTrace.model_validate_json(f.read())
        except Exception as e:
            logger.error("Failed to read trace %s: %s", trace_id, e)
            return None

    def list_traces(self, limit: int = 100) -> List[PipelineTrace]:
        traces = []
        paths = sorted(self.traces_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in paths[:limit]:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    traces.append(PipelineTrace.model_validate_json(f.read()))
            except Exception as e:
                logger.error("Error reading trace file %s: %s", p, e)
        return traces

    # --- Evaluation Runs ---
    def save_eval_run(self, run: EvalRun) -> None:
        with self._lock:
            path = self.evals_dir / f"{run.run_id}.json"
            with open(path, "w", encoding="utf-8") as f:
                f.write(run.model_dump_json(indent=2))

    def get_eval_run(self, run_id: str) -> Optional[EvalRun]:
        path = self.evals_dir / f"{run_id}.json"
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return EvalRun.model_validate_json(f.read())
        except Exception as e:
            logger.error("Failed to read eval run %s: %s", run_id, e)
            return None

    def list_eval_runs(self, limit: int = 50) -> List[EvalRun]:
        runs = []
        paths = sorted(self.evals_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in paths[:limit]:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    runs.append(EvalRun.model_validate_json(f.read()))
            except Exception as e:
                logger.error("Error reading eval run file %s: %s", p, e)
        return runs

    # --- Benchmarks ---
    def save_benchmark(self, name: str, items: List[EvalBenchmarkItem]) -> None:
        with self._lock:
            path = self.benchmarks_dir / f"{name}.json"
            data = [item.model_dump() for item in items]
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

    def load_benchmark(self, name: str = "quantum_bench") -> List[EvalBenchmarkItem]:
        path = self.benchmarks_dir / f"{name}.json"
        if not path.exists():
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return [EvalBenchmarkItem.model_validate(d) for d in data]
        except Exception as e:
            logger.error("Failed to read benchmark %s: %s", name, e)
            return []

    def list_benchmarks(self) -> List[str]:
        return [p.stem for p in self.benchmarks_dir.glob("*.json")]

    def _seed_default_benchmark_if_needed(self) -> None:
        bench_file = self.benchmarks_dir / "quantum_bench.json"
        if not bench_file.exists():
            self.save_benchmark("quantum_bench", default_benchmark_items())

    # --- Experiments ---
    def save_experiment(self, exp: ChunkExperimentResult) -> None:
        with self._lock:
            path = self.experiments_dir / f"{exp.experiment_id}.json"
            with open(path, "w", encoding="utf-8") as f:
                f.write(exp.model_dump_json(indent=2))

    def list_experiments(self, limit: int = 50) -> List[ChunkExperimentResult]:
        exps = []
        paths = sorted(self.experiments_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in paths[:limit]:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    exps.append(ChunkExperimentResult.model_validate_json(f.read()))
            except Exception as e:
                logger.error("Error reading experiment file %s: %s", p, e)
        return exps

    # --- Failure Investigations ---
    def save_investigation(self, report: FailureInvestigationReport) -> None:
        with self._lock:
            path = self.investigations_dir / f"{report.investigation_id}.json"
            with open(path, "w", encoding="utf-8") as f:
                f.write(report.model_dump_json(indent=2))

    def get_investigation(self, investigation_id: str) -> Optional[FailureInvestigationReport]:
        path = self.investigations_dir / f"{investigation_id}.json"
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return FailureInvestigationReport.model_validate_json(f.read())
        except Exception as e:
            logger.error("Error reading investigation file %s: %s", path, e)
            return None

    def list_investigations(self, limit: int = 50) -> List[FailureInvestigationReport]:
        reports = []
        paths = sorted(self.investigations_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in paths[:limit]:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    reports.append(FailureInvestigationReport.model_validate_json(f.read()))
            except Exception as e:
                logger.error("Error reading investigation file %s: %s", p, e)
        return reports

    def get_status(self) -> Dict[str, Any]:
        """Describe the active backend without performing any network IO."""
        return {
            "backend": self.backend_name,
            "configured": settings.supabase_configured,
            "connected": False,
            "degraded": False,
            "fallback_ops": 0,
            "last_error": getattr(self, "init_error", None),
            "last_error_at": None,
            "project_host": None,
            "table": settings.supabase_table,
            "local_dir": str(self.base_dir),
        }


def build_storage() -> ObservabilityStorage:
    """Select Supabase when configured and reachable, otherwise local files."""
    backend = settings.storage_backend
    use_supabase = backend == "supabase" or (backend == "auto" and settings.supabase_configured)
    if not use_supabase:
        return ObservabilityStorage()
    if not settings.supabase_configured:
        logger.warning("STORAGE_BACKEND=supabase but SUPABASE_URL/SUPABASE_KEY are unset; using local storage.")
        return ObservabilityStorage()

    try:
        from observability.supabase_storage import SupabaseObservabilityStorage

        return SupabaseObservabilityStorage()
    except Exception as exc:
        logger.exception("Supabase storage initialization failed; using local files.")
        store = ObservabilityStorage()
        store.init_error = f"{type(exc).__name__}: {exc}"
        return store


storage = build_storage()
