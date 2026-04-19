from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

from prometheus_client import Counter, Gauge, Histogram


HTTP_REQUESTS_TOTAL = Counter(
    "medgraphqa_http_requests_total",
    "Total HTTP requests.",
    ["method", "path", "status"],
)
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "medgraphqa_http_request_duration_seconds",
    "HTTP request duration in seconds.",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120),
)
HTTP_REQUESTS_IN_PROGRESS = Gauge(
    "medgraphqa_http_requests_in_progress",
    "HTTP requests currently in progress.",
    ["method", "path"],
)

CHAT_REQUESTS_TOTAL = Counter(
    "medgraphqa_chat_requests_total",
    "Total chat requests.",
    ["mode", "status"],
)
CHAT_REQUEST_DURATION_SECONDS = Histogram(
    "medgraphqa_chat_request_duration_seconds",
    "Chat request duration in seconds.",
    ["mode"],
    buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 40, 60, 90, 120, 180, 300),
)
CHAT_ACTIVE_STREAMS = Gauge(
    "medgraphqa_chat_active_streams",
    "Currently active chat SSE streams.",
)
CHAT_ANSWER_LENGTH = Histogram(
    "medgraphqa_chat_answer_length_chars",
    "Final assistant answer length in characters.",
    ["mode"],
    buckets=(0, 20, 50, 100, 200, 400, 800, 1600, 3200),
)
CHAT_ENTITY_COUNT = Histogram(
    "medgraphqa_chat_entity_count",
    "Number of normalized entities in a chat response.",
    ["mode"],
    buckets=(0, 1, 2, 3, 5, 8, 13, 21, 34),
)
CHAT_EVIDENCE_COUNT = Histogram(
    "medgraphqa_chat_evidence_count",
    "Number of KG evidence rows in a chat response.",
    ["mode"],
    buckets=(0, 1, 2, 3, 5, 8, 13, 21),
)

OPERATION_DURATION_SECONDS = Histogram(
    "medgraphqa_operation_duration_seconds",
    "Internal operation duration in seconds.",
    ["operation", "status"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120),
)
OPERATION_ERRORS_TOTAL = Counter(
    "medgraphqa_operation_errors_total",
    "Internal operation errors.",
    ["operation", "error_type"],
)

CHAT_NODE_DURATION_SECONDS = Histogram(
    "medgraphqa_chat_node_duration_seconds",
    "LangGraph node duration in seconds.",
    ["node", "status"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120),
)
CHAT_NODE_ERRORS_TOTAL = Counter(
    "medgraphqa_chat_node_errors_total",
    "LangGraph node errors.",
    ["node", "error_type"],
)

LLM_REQUESTS_TOTAL = Counter(
    "medgraphqa_llm_requests_total",
    "LLM requests.",
    ["provider", "model", "operation", "status"],
)
LLM_DURATION_SECONDS = Histogram(
    "medgraphqa_llm_duration_seconds",
    "LLM call duration in seconds.",
    ["provider", "model", "operation"],
    buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 40, 60, 90, 120, 180),
)
LLM_TIMEOUTS_TOTAL = Counter(
    "medgraphqa_llm_timeouts_total",
    "LLM timeout errors.",
    ["provider", "model", "operation"],
)
LLM_PROMPT_LENGTH = Histogram(
    "medgraphqa_llm_prompt_length_chars",
    "LLM prompt length in characters.",
    ["provider", "model", "operation"],
    buckets=(0, 100, 250, 500, 1000, 2000, 4000, 8000, 16000),
)
LLM_ANSWER_LENGTH = Histogram(
    "medgraphqa_llm_answer_length_chars",
    "LLM answer length in characters.",
    ["provider", "model", "operation"],
    buckets=(0, 20, 50, 100, 200, 400, 800, 1600, 3200),
)
LLM_TOKENS_TOTAL = Counter(
    "medgraphqa_llm_tokens_total",
    "LLM streamed token count.",
    ["provider", "model", "operation", "type"],
)

ENTITY_SEARCH_DURATION_SECONDS = Histogram(
    "medgraphqa_entity_search_duration_seconds",
    "Entity search operation duration in seconds.",
    ["stage", "status"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)
ENTITY_SEARCH_CANDIDATES = Histogram(
    "medgraphqa_entity_search_candidates_count",
    "Entity search candidate count.",
    ["stage"],
    buckets=(0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233),
)

EMBEDDING_DURATION_SECONDS = Histogram(
    "medgraphqa_embedding_duration_seconds",
    "Embedding request duration in seconds.",
    ["provider", "model", "status"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 40, 60),
)
EMBEDDING_ERRORS_TOTAL = Counter(
    "medgraphqa_embedding_errors_total",
    "Embedding errors.",
    ["provider", "model", "error_type"],
)

KG_QUERY_DURATION_SECONDS = Histogram(
    "medgraphqa_kg_query_duration_seconds",
    "Knowledge graph query duration in seconds.",
    ["operation", "status"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
)
KG_QUERY_ROWS = Histogram(
    "medgraphqa_kg_query_rows_count",
    "Knowledge graph query returned row count.",
    ["operation"],
    buckets=(0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144),
)
KG_QUERY_ERRORS_TOTAL = Counter(
    "medgraphqa_kg_query_errors_total",
    "Knowledge graph query errors.",
    ["operation", "error_type"],
)

SAFETY_INPUT_DECISIONS_TOTAL = Counter(
    "medgraphqa_safety_input_decisions_total",
    "Input safety decisions.",
    ["category", "action", "severity"],
)
SAFETY_OUTPUT_DECISIONS_TOTAL = Counter(
    "medgraphqa_safety_output_decisions_total",
    "Output safety decisions.",
    ["safe", "action"],
)
SAFETY_HITS_TOTAL = Counter(
    "medgraphqa_safety_hits_total",
    "Safety guardrail hits.",
    ["stage", "code"],
)

MEMORY_EXTRACT_CANDIDATES_TOTAL = Counter(
    "medgraphqa_memory_extract_candidates_total",
    "Long-term memory extract candidates.",
    ["memory_type"],
)
MEMORY_SAVED_TOTAL = Counter(
    "medgraphqa_memory_saved_total",
    "Saved long-term memories.",
    ["memory_type", "status"],
)
MEMORY_LOADED_TOTAL = Counter(
    "medgraphqa_memory_loaded_total",
    "Loaded long-term memories.",
    ["memory_type"],
)


def normalize_path(path: str) -> str:
    if path.startswith("/api/chat/sessions/") and path.endswith("/messages"):
        return "/api/chat/sessions/{session_id}/messages"
    if path.startswith("/api/chat/sessions/"):
        return "/api/chat/sessions/{session_id}"
    if path.startswith("/api/memories/"):
        return "/api/memories/{memory_id}"
    return path


def record_http_request(method: str, path: str, status: int, duration_seconds: float) -> None:
    route = normalize_path(path)
    HTTP_REQUESTS_TOTAL.labels(method=method, path=route, status=str(status)).inc()
    HTTP_REQUEST_DURATION_SECONDS.labels(method=method, path=route).observe(duration_seconds)


@contextmanager
def track_http_in_progress(method: str, path: str) -> Iterator[None]:
    route = normalize_path(path)
    gauge = HTTP_REQUESTS_IN_PROGRESS.labels(method=method, path=route)
    gauge.inc()
    try:
        yield
    finally:
        gauge.dec()


@contextmanager
def track_chat_request(mode: str) -> Iterator[dict]:
    started = time.perf_counter()
    CHAT_ACTIVE_STREAMS.inc() if mode == "stream" else None
    result: dict = {}
    try:
        yield result
    except Exception:
        _record_chat(mode, "error", started, result)
        raise
    else:
        _record_chat(mode, "ok", started, result)
    finally:
        CHAT_ACTIVE_STREAMS.dec() if mode == "stream" else None


def _record_chat(mode: str, status: str, started: float, result: dict) -> None:
    duration = time.perf_counter() - started
    CHAT_REQUESTS_TOTAL.labels(mode=mode, status=status).inc()
    CHAT_REQUEST_DURATION_SECONDS.labels(mode=mode).observe(duration)
    CHAT_ANSWER_LENGTH.labels(mode=mode).observe(float(result.get("answer_len") or 0))
    CHAT_ENTITY_COUNT.labels(mode=mode).observe(float(result.get("entity_count") or 0))
    CHAT_EVIDENCE_COUNT.labels(mode=mode).observe(float(result.get("evidence_count") or 0))


def observe_operation(
    operation: str,
    status: str,
    duration_seconds: float,
    *,
    fields: dict,
    result: dict | None = None,
    error_type: str | None = None,
) -> None:
    result = result or {}
    OPERATION_DURATION_SECONDS.labels(operation=operation, status=status).observe(duration_seconds)
    if error_type:
        OPERATION_ERRORS_TOTAL.labels(operation=operation, error_type=error_type).inc()
    if operation == "chat.node":
        _observe_chat_node(status, duration_seconds, fields, error_type)
    elif operation.startswith("llm."):
        _observe_llm(operation, status, duration_seconds, fields, result, error_type)
    elif operation.startswith("entity_search."):
        _observe_entity_search(operation, status, duration_seconds, result)
    elif operation.startswith("embedding."):
        _observe_embedding(status, duration_seconds, fields, error_type)
    elif operation.startswith("kg."):
        _observe_kg(operation, status, duration_seconds, result, error_type)


def _observe_chat_node(
    status: str,
    duration_seconds: float,
    fields: dict,
    error_type: str | None,
) -> None:
    node = str(fields.get("node") or "-")
    CHAT_NODE_DURATION_SECONDS.labels(node=node, status=status).observe(duration_seconds)
    if error_type:
        CHAT_NODE_ERRORS_TOTAL.labels(node=node, error_type=error_type).inc()


def _observe_llm(
    operation: str,
    status: str,
    duration_seconds: float,
    fields: dict,
    result: dict,
    error_type: str | None,
) -> None:
    provider = str(fields.get("provider") or "-")
    model = str(fields.get("model") or "-")
    op = operation.removeprefix("llm.")
    LLM_REQUESTS_TOTAL.labels(provider=provider, model=model, operation=op, status=status).inc()
    LLM_DURATION_SECONDS.labels(provider=provider, model=model, operation=op).observe(duration_seconds)
    LLM_PROMPT_LENGTH.labels(provider=provider, model=model, operation=op).observe(
        float(fields.get("prompt_len") or 0)
    )
    LLM_ANSWER_LENGTH.labels(provider=provider, model=model, operation=op).observe(
        float(result.get("answer_len") or 0)
    )
    token_count = result.get("token_count")
    if token_count:
        LLM_TOKENS_TOTAL.labels(provider=provider, model=model, operation=op, type="completion").inc(
            float(token_count)
        )
    if error_type and "timeout" in error_type.lower():
        LLM_TIMEOUTS_TOTAL.labels(provider=provider, model=model, operation=op).inc()


def _observe_entity_search(
    operation: str,
    status: str,
    duration_seconds: float,
    result: dict,
) -> None:
    stage = operation.removeprefix("entity_search.")
    ENTITY_SEARCH_DURATION_SECONDS.labels(stage=stage, status=status).observe(duration_seconds)
    for key in ("candidate_count", "selected_count", "ranked_count", "elastic_count", "vector_count"):
        if key in result:
            ENTITY_SEARCH_CANDIDATES.labels(stage=f"{stage}.{key}").observe(float(result.get(key) or 0))


def _observe_embedding(
    status: str,
    duration_seconds: float,
    fields: dict,
    error_type: str | None,
) -> None:
    provider = str(fields.get("provider") or "-")
    model = str(fields.get("model") or "-")
    EMBEDDING_DURATION_SECONDS.labels(provider=provider, model=model, status=status).observe(duration_seconds)
    if error_type:
        EMBEDDING_ERRORS_TOTAL.labels(provider=provider, model=model, error_type=error_type).inc()


def _observe_kg(
    operation: str,
    status: str,
    duration_seconds: float,
    result: dict,
    error_type: str | None,
) -> None:
    op = operation.removeprefix("kg.")
    KG_QUERY_DURATION_SECONDS.labels(operation=op, status=status).observe(duration_seconds)
    if "row_count" in result:
        KG_QUERY_ROWS.labels(operation=op).observe(float(result.get("row_count") or 0))
    if error_type:
        KG_QUERY_ERRORS_TOTAL.labels(operation=op, error_type=error_type).inc()


def record_safety_input(category: str, action: str, severity: str, hit_codes: list[str]) -> None:
    SAFETY_INPUT_DECISIONS_TOTAL.labels(
        category=category,
        action=action,
        severity=severity,
    ).inc()
    for code in hit_codes:
        SAFETY_HITS_TOTAL.labels(stage="input", code=code).inc()


def record_safety_output(safe: bool, action: str, hit_codes: list[str]) -> None:
    SAFETY_OUTPUT_DECISIONS_TOTAL.labels(safe=str(safe).lower(), action=action).inc()
    for code in hit_codes:
        SAFETY_HITS_TOTAL.labels(stage="output", code=code).inc()

