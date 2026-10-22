"""Unit tests for the Supabase storage adapter using an in-memory fake client."""

import sys
import types
from types import SimpleNamespace

import pytest

from observability.models import EvalRun, IngestionJob, PipelineTrace, StageStatus
from observability.storage import ObservabilityStorage


class FakeQuery:
    def __init__(self, client):
        self.client = client
        self.op = None
        self.filters = []
        self.order_field = None
        self.order_desc = False
        self.max_rows = None
        self.data = None
        self.columns = "*"

    def table(self, name):
        self.client.table_name = name
        return self

    def upsert(self, data, on_conflict=None):
        self.op = "upsert"
        self.data = data
        return self

    def select(self, columns="*"):
        self.op = "select"
        self.columns = columns
        return self

    def eq(self, column, value):
        self.filters.append((column, value))
        return self

    def order(self, field, desc=False):
        self.order_field = field
        self.order_desc = desc
        return self

    def limit(self, count):
        self.max_rows = count
        return self

    def execute(self):
        if self.client.fail_next:
            self.client.fail_next = False
            raise ConnectionError("simulated PostgREST outage")

        if self.op == "upsert":
            key = (self.data["kind"], self.data["record_id"])
            self.client.rows[key] = dict(self.data)
            return SimpleNamespace(data=[self.data])

        rows = list(self.client.rows.values())
        for column, value in self.filters:
            rows = [row for row in rows if row.get(column) == value]
        if self.order_field:
            rows.sort(key=lambda row: row.get(self.order_field, ""), reverse=self.order_desc)
        if self.max_rows is not None:
            rows = rows[: self.max_rows]
        if self.columns == "record_id":
            return SimpleNamespace(data=[{"record_id": row["record_id"]} for row in rows])
        return SimpleNamespace(data=[{"payload": row["payload"]} for row in rows])


class FakeClient:
    def __init__(self):
        self.rows = {}
        self.fail_next = False
        self.table_name = None

    def table(self, name):
        self.table_name = name
        return FakeQuery(self)


@pytest.fixture()
def fake_supabase(monkeypatch):
    client = FakeClient()
    module = types.ModuleType("supabase")
    module.create_client = lambda url, key: client
    monkeypatch.setitem(sys.modules, "supabase", module)
    return client


@pytest.fixture()
def store(fake_supabase, tmp_path):
    from observability.supabase_storage import SupabaseObservabilityStorage

    return SupabaseObservabilityStorage(
        url="https://demo.supabase.co",
        key="service-role-key",
        local_fallback=ObservabilityStorage(base_dir=tmp_path / "local"),
    )


def test_roundtrip_and_listing(store):
    trace = PipelineTrace(trace_id="trc_remote_1", query="What is a qubit?", final_answer="A qubit.")
    store.save_trace(trace)

    loaded = store.get_trace("trc_remote_1")
    assert loaded is not None
    assert loaded.query == "What is a qubit?"
    assert [t.trace_id for t in store.list_traces()] == ["trc_remote_1"]
    assert store.get_status()["backend"] == "supabase"
    assert store.get_status()["connected"] is True


def test_benchmark_seed_and_roundtrip(store):
    assert store.list_benchmarks() == ["quantum_bench"]
    items = store.load_benchmark("quantum_bench")
    assert len(items) == 22

    modified = items[:2]
    store.save_benchmark("custom_bench", modified)
    assert set(store.list_benchmarks()) == {"quantum_bench", "custom_bench"}
    assert len(store.load_benchmark("custom_bench")) == 2


def test_fallback_on_outage(store, fake_supabase):
    trace = PipelineTrace(trace_id="trc_fallback", query="fallback?", final_answer="local")
    fake_supabase.fail_next = True

    store.save_trace(trace)

    status = store.get_status()
    assert status["degraded"] is True
    assert status["fallback_ops"] == 1
    assert "simulated PostgREST outage" in status["last_error"]
    assert store._local.get_trace("trc_fallback") is not None
    assert ("trace", "trc_fallback") not in fake_supabase.rows


def test_read_fallback_when_remote_empty_and_unreachable(store, fake_supabase):
    local_trace = PipelineTrace(trace_id="trc_local", query="local only", final_answer="ok")
    store._local.save_trace(local_trace)
    fake_supabase.fail_next = True

    traces = store.list_traces()

    assert [t.trace_id for t in traces] == ["trc_local"]
    assert store.get_status()["degraded"] is True


def test_backfill_local_records(store):
    store._local.save_job(IngestionJob(job_id="job_local", document_id="doc", filename="a.pdf", overall_status=StageStatus.COMPLETED))
    store._local.save_eval_run(EvalRun(run_id="run_local", benchmark_name="quantum_bench"))

    counts = store.push_local_records()

    assert counts["jobs"] == 1
    assert counts["eval_runs"] == 1
    assert counts["benchmarks"] == 1
    assert store.get_status()["degraded"] is False
    assert ("job", "job_local") in store._client.rows
    assert ("eval_run", "run_local") in store._client.rows


def test_connection_check_reports_failure(store, fake_supabase):
    fake_supabase.fail_next = True
    status = store.check_connection()
    assert status["connected"] is False
    assert status["degraded"] is True
