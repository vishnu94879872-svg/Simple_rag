"""Supabase-backed observability storage.

Records live in a single PostgREST table (default ``observability_records``)
keyed by ``(kind, record_id)`` with the full Pydantic document in ``payload``.
Writes and reads transparently fall back to the local file store when Supabase
is unreachable, and every fallback is logged plus surfaced through
``get_status()`` so the UI can show a degraded-state banner.
"""

import datetime
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, TypeVar
from urllib.parse import urlparse

from config import settings
from observability.models import (
    ChunkExperimentResult,
    EvalBenchmarkItem,
    EvalRun,
    FailureInvestigationReport,
    IngestionJob,
    PipelineTrace,
    SupportTicket,
)
from observability.storage import ObservabilityStorage, default_benchmark_items

logger = logging.getLogger(__name__)

T = TypeVar("T")

KIND_JOB = "job"
KIND_TRACE = "trace"
KIND_EVAL_RUN = "eval_run"
KIND_BENCHMARK = "benchmark"
KIND_EXPERIMENT = "experiment"
KIND_INVESTIGATION = "investigation"
KIND_SUPPORT_TICKET = "support_ticket"


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _is_transient_network_error(exc: Exception) -> bool:
    name = type(exc).__name__
    msg = str(exc).lower()
    return any(
        term in name.lower() or term in msg
        for term in (
            "remoteprotocolerror",
            "server disconnected",
            "connectionreset",
            "connecterror",
            "timeout",
            "broken pipe",
        )
    )


class SupabaseObservabilityStorage:
    """Postgres/Supabase implementation of the observability storage contract."""

    backend_name = "supabase"

    def __init__(
        self,
        url: Optional[str] = None,
        key: Optional[str] = None,
        table: Optional[str] = None,
        local_fallback: Optional[ObservabilityStorage] = None,
    ):
        self.url = (url or settings.supabase_url).rstrip("/")
        self.key = key or settings.supabase_key
        self.table = table or settings.supabase_table
        if not self.url or not self.key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY are required for Supabase storage")

        try:
            from supabase import create_client
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("The 'supabase' package is not installed. Run: pip install -r requirements.txt") from exc

        self._client = create_client(self.url, self.key)
        self._local = local_fallback or ObservabilityStorage()
        self._write_lock = threading.RLock()
        self._seed_lock = threading.Lock()
        self._seed_attempted = False
        self._connected: Optional[bool] = None
        self._degraded = False
        self._fallback_ops = 0
        self._last_error: Optional[str] = None
        self._last_error_at: Optional[str] = None

    def _execute_with_retry(self, fn: Callable[[], T], max_attempts: int = 3, initial_delay: float = 0.25) -> T:
        delay = initial_delay
        for attempt in range(max_attempts):
            try:
                return fn()
            except Exception as exc:
                if not _is_transient_network_error(exc) or attempt == max_attempts - 1:
                    raise
                logger.warning(
                    "Transient error on Supabase operation (%s: %s). Retrying (attempt %d/%d)...",
                    type(exc).__name__,
                    exc,
                    attempt + 1,
                    max_attempts,
                )
                time.sleep(delay)
                delay *= 2

    # ------------------------------------------------------------------ status
    def get_status(self) -> Dict[str, Any]:
        return {
            "backend": self.backend_name,
            "configured": True,
            "connected": self._connected,
            "degraded": self._degraded,
            "fallback_ops": self._fallback_ops,
            "last_error": self._last_error,
            "last_error_at": self._last_error_at,
            "project_host": urlparse(self.url).hostname,
            "table": self.table,
            "local_dir": str(self._local.base_dir),
        }

    def check_connection(self) -> Dict[str, Any]:
        """Ping PostgREST so Settings can verify credentials and schema."""
        try:
            self._execute_with_retry(
                lambda: self._client.table(self.table).select("record_id").limit(1).execute()
            )
            self._mark_healthy()
        except Exception as exc:
            self._connected = False
            self._mark_degraded("connection_check", exc)
        return self.get_status()

    def _mark_healthy(self) -> None:
        self._connected = True
        self._degraded = False
        self._last_error = None
        self._last_error_at = None

    def _mark_degraded(self, operation: str, exc: Exception) -> None:
        self._connected = False
        self._degraded = True
        self._fallback_ops += 1
        self._last_error = f"{operation}: {type(exc).__name__}: {str(exc)[:400]}"
        self._last_error_at = _utc_now()
        logger.error("Supabase %s failed; using local fallback. %s", operation, exc)

    def _call(self, operation: str, primary: Callable[[], T], fallback: Callable[[], T]) -> T:
        try:
            result = primary()
            self._mark_healthy()
            return result
        except Exception as exc:
            self._mark_degraded(operation, exc)
            return fallback()

    # ------------------------------------------------------------- primitives
    def _upsert(self, kind: str, record_id: str, payload: Dict[str, Any]) -> None:
        with self._write_lock:
            self._execute_with_retry(
                lambda: self._client.table(self.table).upsert(
                    {
                        "kind": kind,
                        "record_id": record_id,
                        "payload": payload,
                        "updated_at": _utc_now(),
                    },
                    on_conflict="kind,record_id",
                ).execute()
            )

    def _fetch(self, kind: str, record_id: str) -> Optional[Dict[str, Any]]:
        response = self._execute_with_retry(
            lambda: (
                self._client.table(self.table)
                .select("payload")
                .eq("kind", kind)
                .eq("record_id", record_id)
                .limit(1)
                .execute()
            )
        )
        rows = response.data or []
        return rows[0]["payload"] if rows else None

    def _list(self, kind: str, limit: int) -> List[Dict[str, Any]]:
        response = self._execute_with_retry(
            lambda: (
                self._client.table(self.table)
                .select("payload")
                .eq("kind", kind)
                .order("updated_at", desc=True)
                .limit(limit)
                .execute()
            )
        )
        return [row["payload"] for row in (response.data or [])]

    def _ensure_seed_benchmark(self) -> None:
        if self._seed_attempted:
            return
        with self._seed_lock:
            if self._seed_attempted:
                return
            names = self._list_benchmark_names()
            if not names:
                self._upsert(KIND_BENCHMARK, "quantum_bench", self._benchmark_payload("quantum_bench", default_benchmark_items()))
            self._seed_attempted = True

    def _list_benchmark_names(self) -> List[str]:
        response = self._execute_with_retry(
            lambda: (
                self._client.table(self.table)
                .select("record_id")
                .eq("kind", KIND_BENCHMARK)
                .order("updated_at", desc=True)
                .execute()
            )
        )
        return [row["record_id"] for row in (response.data or [])]

    @staticmethod
    def _benchmark_payload(name: str, items: List[EvalBenchmarkItem]) -> Dict[str, Any]:
        return {"name": name, "items": [item.model_dump(mode="json") for item in items]}

    # ------------------------------------------------------------------- jobs
    def save_job(self, job: IngestionJob) -> None:
        payload = job.model_dump(mode="json")
        self._call(
            "save_job",
            lambda: self._upsert(KIND_JOB, job.job_id, payload),
            lambda: self._local.save_job(job),
        )

    def get_job(self, job_id: str) -> Optional[IngestionJob]:
        def primary() -> Optional[IngestionJob]:
            payload = self._fetch(KIND_JOB, job_id)
            return IngestionJob.model_validate(payload) if payload else None

        return self._call("get_job", primary, lambda: self._local.get_job(job_id))

    def list_jobs(self, limit: int = 50) -> List[IngestionJob]:
        def primary() -> List[IngestionJob]:
            return [IngestionJob.model_validate(row) for row in self._list(KIND_JOB, limit)]

        return self._call("list_jobs", primary, lambda: self._local.list_jobs(limit))

    # ----------------------------------------------------------------- traces
    def save_trace(self, trace: PipelineTrace) -> None:
        payload = trace.model_dump(mode="json")
        self._call(
            "save_trace",
            lambda: self._upsert(KIND_TRACE, trace.trace_id, payload),
            lambda: self._local.save_trace(trace),
        )

    def get_trace(self, trace_id: str) -> Optional[PipelineTrace]:
        def primary() -> Optional[PipelineTrace]:
            payload = self._fetch(KIND_TRACE, trace_id)
            return PipelineTrace.model_validate(payload) if payload else None

        return self._call("get_trace", primary, lambda: self._local.get_trace(trace_id))

    def list_traces(self, limit: int = 100) -> List[PipelineTrace]:
        def primary() -> List[PipelineTrace]:
            return [PipelineTrace.model_validate(row) for row in self._list(KIND_TRACE, limit)]

        return self._call("list_traces", primary, lambda: self._local.list_traces(limit))

    # -------------------------------------------------------------- eval runs
    def save_eval_run(self, run: EvalRun) -> None:
        payload = run.model_dump(mode="json")
        self._call(
            "save_eval_run",
            lambda: self._upsert(KIND_EVAL_RUN, run.run_id, payload),
            lambda: self._local.save_eval_run(run),
        )

    def get_eval_run(self, run_id: str) -> Optional[EvalRun]:
        def primary() -> Optional[EvalRun]:
            payload = self._fetch(KIND_EVAL_RUN, run_id)
            return EvalRun.model_validate(payload) if payload else None

        return self._call("get_eval_run", primary, lambda: self._local.get_eval_run(run_id))

    def list_eval_runs(self, limit: int = 50) -> List[EvalRun]:
        def primary() -> List[EvalRun]:
            return [EvalRun.model_validate(row) for row in self._list(KIND_EVAL_RUN, limit)]

        return self._call("list_eval_runs", primary, lambda: self._local.list_eval_runs(limit))

    # -------------------------------------------------------------- benchmarks
    def save_benchmark(self, name: str, items: List[EvalBenchmarkItem]) -> None:
        payload = self._benchmark_payload(name, items)
        self._call(
            "save_benchmark",
            lambda: self._upsert(KIND_BENCHMARK, name, payload),
            lambda: self._local.save_benchmark(name, items),
        )

    def load_benchmark(self, name: str = "quantum_bench") -> List[EvalBenchmarkItem]:
        def primary() -> List[EvalBenchmarkItem]:
            payload = self._fetch(KIND_BENCHMARK, name)
            if not payload:
                return []
            return [EvalBenchmarkItem.model_validate(item) for item in payload.get("items", [])]

        return self._call("load_benchmark", primary, lambda: self._local.load_benchmark(name))

    def list_benchmarks(self) -> List[str]:
        def primary() -> List[str]:
            self._ensure_seed_benchmark()
            return self._list_benchmark_names()

        return self._call("list_benchmarks", primary, self._local.list_benchmarks)

    # ------------------------------------------------------------- experiments
    def save_experiment(self, exp: ChunkExperimentResult) -> None:
        payload = exp.model_dump(mode="json")
        self._call(
            "save_experiment",
            lambda: self._upsert(KIND_EXPERIMENT, exp.experiment_id, payload),
            lambda: self._local.save_experiment(exp),
        )

    def list_experiments(self, limit: int = 50) -> List[ChunkExperimentResult]:
        def primary() -> List[ChunkExperimentResult]:
            return [ChunkExperimentResult.model_validate(row) for row in self._list(KIND_EXPERIMENT, limit)]

        return self._call("list_experiments", primary, lambda: self._local.list_experiments(limit))

    # ----------------------------------------------------------- investigations
    def save_investigation(self, report: FailureInvestigationReport) -> None:
        payload = report.model_dump(mode="json")
        self._call(
            "save_investigation",
            lambda: self._upsert(KIND_INVESTIGATION, report.investigation_id, payload),
            lambda: self._local.save_investigation(report),
        )

    def get_investigation(self, investigation_id: str) -> Optional[FailureInvestigationReport]:
        def primary() -> Optional[FailureInvestigationReport]:
            payload = self._fetch(KIND_INVESTIGATION, investigation_id)
            return FailureInvestigationReport.model_validate(payload) if payload else None

        return self._call(
            "get_investigation",
            primary,
            lambda: self._local.get_investigation(investigation_id),
        )

    def list_investigations(self, limit: int = 50) -> List[FailureInvestigationReport]:
        def primary() -> List[FailureInvestigationReport]:
            return [FailureInvestigationReport.model_validate(row) for row in self._list(KIND_INVESTIGATION, limit)]

        return self._call("list_investigations", primary, lambda: self._local.list_investigations(limit))

    # -------------------------------------------------------- support tickets
    def save_support_ticket(self, ticket: SupportTicket) -> None:
        payload = ticket.model_dump(mode="json")
        self._call(
            "save_support_ticket",
            lambda: self._upsert(KIND_SUPPORT_TICKET, ticket.ticket_id, payload),
            lambda: self._local.save_support_ticket(ticket),
        )

    def get_support_ticket(self, ticket_id: str) -> Optional[SupportTicket]:
        def primary() -> Optional[SupportTicket]:
            payload = self._fetch(KIND_SUPPORT_TICKET, ticket_id)
            return SupportTicket.model_validate(payload) if payload else None

        return self._call(
            "get_support_ticket",
            primary,
            lambda: self._local.get_support_ticket(ticket_id),
        )

    def list_support_tickets(self, limit: int = 50) -> List[SupportTicket]:
        def primary() -> List[SupportTicket]:
            return [SupportTicket.model_validate(row) for row in self._list(KIND_SUPPORT_TICKET, limit)]

        return self._call("list_support_tickets", primary, lambda: self._local.list_support_tickets(limit))

    # ---------------------------------------------------------------- backfill
    def push_local_records(self) -> Dict[str, int]:
        """Copy every locally persisted record into Supabase. Idempotent upserts."""
        counts = {"jobs": 0, "traces": 0, "eval_runs": 0, "benchmarks": 0, "experiments": 0, "investigations": 0, "tickets": 0}
        try:
            for job in self._local.list_jobs(limit=10000):
                self._upsert(KIND_JOB, job.job_id, job.model_dump(mode="json"))
                counts["jobs"] += 1
            for trace in self._local.list_traces(limit=10000):
                self._upsert(KIND_TRACE, trace.trace_id, trace.model_dump(mode="json"))
                counts["traces"] += 1
            for run in self._local.list_eval_runs(limit=10000):
                self._upsert(KIND_EVAL_RUN, run.run_id, run.model_dump(mode="json"))
                counts["eval_runs"] += 1
            for name in self._local.list_benchmarks():
                items = self._local.load_benchmark(name)
                self._upsert(KIND_BENCHMARK, name, self._benchmark_payload(name, items))
                counts["benchmarks"] += 1
            for exp in self._local.list_experiments(limit=10000):
                self._upsert(KIND_EXPERIMENT, exp.experiment_id, exp.model_dump(mode="json"))
                counts["experiments"] += 1
            for report in self._local.list_investigations(limit=10000):
                self._upsert(KIND_INVESTIGATION, report.investigation_id, report.model_dump(mode="json"))
                counts["investigations"] += 1
            for ticket in self._local.list_support_tickets(limit=10000):
                self._upsert(KIND_SUPPORT_TICKET, ticket.ticket_id, ticket.model_dump(mode="json"))
                counts["tickets"] += 1
        except Exception as exc:
            done = sum(counts.values())
            self._mark_degraded("push_local_records", exc)
            raise RuntimeError(f"Backfill stopped after {done} records: {exc}") from exc

        self._mark_healthy()
        return counts
