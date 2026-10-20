"""Pipeline execution tracer with correlation ID, secret masking, and privacy policies."""

import time
import uuid
import re
import logging
from typing import Dict, Any, Optional, List
from contextlib import contextmanager

from config import settings
from observability.models import PipelineTrace, TraceStep
from observability.storage import storage

logger = logging.getLogger(__name__)

# Sensitive pattern regexes to redact
SECRET_PATTERNS = [
    re.compile(r"sk-[a-zA-Z0-9_\-]{20,}", re.IGNORECASE),
    re.compile(r"Bearer\s+[a-zA-Z0-9_\-\.]{20,}", re.IGNORECASE),
    re.compile(r"api[-_]?key\s*[:=]\s*['\"]?[a-zA-Z0-9_\-]{16,}['\"]?", re.IGNORECASE),
]

# Estimated token pricing for known models (per 1,000,000 tokens)
MODEL_PRICING = {
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    "gpt-4o": {"prompt": 2.50, "completion": 10.00},
    "gpt-3.5-turbo": {"prompt": 0.50, "completion": 1.50},
}


def mask_secrets(text: Optional[str]) -> Optional[str]:
    """Redact API keys, tokens, and passwords from string representations."""
    if not text:
        return text
    masked = text
    for pattern in SECRET_PATTERNS:
        masked = pattern.sub("[REDACTED_SECRET]", masked)
    return masked


class PipelineTracer:
    """Manages an active trace for an end-to-end question answering pipeline run."""

    def __init__(
        self,
        query: str,
        trace_id: Optional[str] = None,
        strategy: str = "reranked",
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        log_prompts: bool = True
    ):
        self.trace_id = trace_id or f"trc_{uuid.uuid4().hex[:10]}"
        self.log_prompts = log_prompts
        self.start_time = time.time()
        self.trace = PipelineTrace(
            trace_id=self.trace_id,
            query=mask_secrets(query) or "",
            strategy=strategy,
            top_k=top_k,
            filters=filters,
            model=settings.llm_model,
            provider=settings.llm_provider
        )
        self.step_start_times: Dict[str, float] = {}

    def start_step(self, step_name: str, input_summary: Optional[Dict[str, Any]] = None) -> None:
        """Mark the beginning of a named execution step."""
        self.step_start_times[step_name] = time.time()
        logger.debug("[%s] Started step '%s'", self.trace_id, step_name)

    def finish_step(
        self,
        step_name: str,
        status: str = "success",
        output_summary: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        input_summary: Optional[Dict[str, Any]] = None
    ) -> None:
        """Mark the completion of a named step and record its metrics."""
        start_t = self.step_start_times.pop(step_name, time.time())
        duration_ms = round((time.time() - start_t) * 1000.0, 2)

        safe_in = {k: mask_secrets(str(v)) for k, v in (input_summary or {}).items()}
        safe_out = {k: mask_secrets(str(v)) for k, v in (output_summary or {}).items()}

        step = TraceStep(
            step_id=f"step_{len(self.trace.steps) + 1:02d}",
            name=step_name,
            duration_ms=duration_ms,
            status=status,
            input_summary=safe_in,
            output_summary=safe_out,
            error=mask_secrets(error) if error else None
        )
        self.trace.steps.append(step)
        logger.debug("[%s] Finished step '%s' in %.2fms (status=%s)", self.trace_id, step_name, duration_ms, status)

    @contextmanager
    def step_context(self, step_name: str, input_summary: Optional[Dict[str, Any]] = None):
        """Context manager for tracing an atomic operation."""
        self.start_step(step_name, input_summary)
        out_summary: Dict[str, Any] = {}
        try:
            yield out_summary
            self.finish_step(step_name, status="success", output_summary=out_summary, input_summary=input_summary)
        except Exception as e:
            self.finish_step(
                step_name,
                status="failed",
                error=str(e),
                output_summary=out_summary,
                input_summary=input_summary
            )
            raise

    def set_prompt_and_context(self, prompt: str, context: str) -> None:
        """Store prompt and context if user privacy policy permits."""
        if self.log_prompts:
            self.trace.prompt_preview = mask_secrets(prompt)
            self.trace.context_preview = mask_secrets(context)
        else:
            self.trace.prompt_preview = "[REDACTED_BY_PRIVACY_POLICY]"
            self.trace.context_preview = f"[{len(context)} characters redacted by privacy policy]"

    def set_token_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Record token counts and compute cost estimate if model pricing is known."""
        total = prompt_tokens + completion_tokens
        self.trace.token_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total
        }

        pricing = MODEL_PRICING.get(self.trace.model)
        if pricing:
            p_cost = (prompt_tokens / 1_000_000.0) * pricing["prompt"]
            c_cost = (completion_tokens / 1_000_000.0) * pricing["completion"]
            self.trace.estimated_cost_usd = round(p_cost + c_cost, 6)

    def finalize(
        self,
        final_answer: str,
        citations: List[Dict[str, Any]],
        retrieved_chunk_ids: List[str],
        rewritten_query: str = "",
        status: str = "success"
    ) -> PipelineTrace:
        """Finalize the trace, compute total elapsed time, and persist to storage."""
        self.trace.total_duration_ms = round((time.time() - self.start_time) * 1000.0, 2)
        self.trace.final_answer = mask_secrets(final_answer) or ""
        self.trace.citations = citations
        self.trace.retrieved_chunk_ids = retrieved_chunk_ids
        self.trace.rewritten_query = rewritten_query
        self.trace.status = status

        try:
            storage.save_trace(self.trace)
        except Exception as e:
            logger.warning("Failed to auto-save pipeline trace: %s", e)

        return self.trace
