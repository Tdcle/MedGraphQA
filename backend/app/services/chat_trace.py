import json
import logging
import time
from datetime import datetime, timezone

from app.core.request_context import get_request_id
from app.services.chat_response_builder import intent_labels


trace_logger = logging.getLogger("medgraphqa.chat_trace")


def truncate_text(value: str, max_chars: int) -> str:
    normalized = value.replace("\r", "\\r").replace("\n", "\\n")
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[:max_chars]}...<truncated chars={len(normalized)}>"


def emit_chat_trace(
    *,
    enabled: bool,
    max_chars: int,
    llm_provider: str,
    llm_model: str,
    state: dict,
) -> None:
    if not enabled:
        return

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "request_id": get_request_id(),
        "event": "chat_turn",
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "query": truncate_text(state["query"], max_chars),
        "effective_query": truncate_text(
            state.get("effective_query", state["query"]), max_chars
        ),
        "conversation_id": state.get("conversation_id"),
        "entities": [item.to_log_dict() for item in state.get("entities", [])],
        "intents": state.get("intents", []),
        "intent_labels": intent_labels(state.get("intents", [])),
        "resolved_intent_labels": state.get("used_intents")
        or intent_labels(state.get("intents", [])),
        "kg_evidence": [
            truncate_text(item, max_chars) for item in state.get("evidence", [])
        ],
        "prompt": truncate_text(state.get("prompt", ""), max_chars),
        "answer": truncate_text(state.get("answer", ""), max_chars),
        "fallback_reason": state.get("fallback_reason"),
        "llm_error": state.get("llm_error"),
        "safety": {
            "input": state.get("input_safety", {}),
            "output": state.get("output_safety", {}),
        },
        "stats": {
            "query_len": len(state["query"]),
            "prompt_len": len(state.get("prompt", "")),
            "answer_len": len(state.get("answer", "")),
            "evidence_count": len(state.get("evidence", [])),
            "total_duration_ms": round(
                (time.perf_counter() - state["started_at"]) * 1000, 2
            ),
            "llm_duration_ms": round(state["llm_duration_ms"], 2)
            if state.get("llm_duration_ms") is not None
            else None,
        },
    }
    trace_logger.info(json.dumps(record, ensure_ascii=False))
