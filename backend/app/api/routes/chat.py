import json
import logging
import queue
import time
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.api.deps import get_container, get_current_user
from app.core.container import ServiceContainer
from app.schemas.chat import (
    ChatMessageItem,
    ChatRequest,
    ChatResponse,
    ChatSessionSummary,
)
from app.services.auth_service import SessionUser
from app.services.operation_log import log_operation

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger("medgraphqa.chat")


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/ask", response_model=ChatResponse)
def ask(
    payload: ChatRequest,
    container: ServiceContainer = Depends(get_container),
    current_user: SessionUser = Depends(get_current_user),
):
    try:
        return container.chat_service.ask(
            query=payload.query,
            user_id=current_user.username,
            conversation_id=payload.conversation_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/ask/stream")
def ask_stream(
    payload: ChatRequest,
    container: ServiceContainer = Depends(get_container),
    current_user: SessionUser = Depends(get_current_user),
):
    def generate():
        events: queue.Queue[tuple[str, object]] = queue.Queue()

        def on_status(message: str) -> None:
            events.put(("status", {"message": message}))

        def on_token(token: str) -> None:
            events.put(("token", {"text": token}))

        def run_worker() -> None:
            try:
                response = container.chat_service.ask_stream(
                    payload.query,
                    current_user.username,
                    payload.conversation_id,
                    on_status,
                    on_token,
                )
                events.put(("_result", response))
            except ValueError as exc:
                events.put(("_value_error", str(exc)))
            except Exception:
                logger.exception("operation=chat.ask_stream_worker status=error")
                events.put(("_worker_error", "请求失败，请稍后重试"))

        with log_operation(
            logger,
            "chat.ask_stream",
            query_len=len(payload.query),
            conversation_id=payload.conversation_id,
        ) as result:
            with ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(run_worker)
                response = None
                while True:
                    try:
                        event, data = events.get(timeout=15)
                    except queue.Empty:
                        yield _sse_event("ping", {"ts": time.time()})
                        continue

                    if event == "_result":
                        response = data
                        break
                    if event == "_value_error":
                        result["error"] = str(data)
                        yield _sse_event("error", {"message": str(data)})
                        return
                    if event == "_worker_error":
                        result["error"] = "worker_exception"
                        yield _sse_event("error", {"message": str(data)})
                        return

                    yield _sse_event(event, data)

                while not events.empty():
                    event, data = events.get_nowait()
                    if event.startswith("_"):
                        continue
                    yield _sse_event(event, data)

            result["answer_len"] = len(response.answer)
            result["entity_count"] = len(response.entities)
            yield _sse_event("final", response.model_dump(mode="json"))
            yield _sse_event("done", {"message": "done"})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/sessions", response_model=list[ChatSessionSummary])
def list_sessions(
    container: ServiceContainer = Depends(get_container),
    current_user: SessionUser = Depends(get_current_user),
):
    rows = container.entity_repository.list_chat_sessions(current_user.username)
    return [
        ChatSessionSummary(
            conversation_id=str(row["id"]),
            title=row.get("title"),
            updated_at=row["updated_at"].isoformat(),
            awaiting_user_clarification=bool(row.get("awaiting_user_clarification")),
            last_answer=row.get("last_answer"),
        )
        for row in rows
    ]


@router.post("/sessions", response_model=ChatSessionSummary)
def create_session(
    container: ServiceContainer = Depends(get_container),
    current_user: SessionUser = Depends(get_current_user),
):
    row = container.entity_repository.create_chat_session(
        user_id=current_user.username,
        title="新对话",
    )
    return ChatSessionSummary(
        conversation_id=str(row["id"]),
        title=row.get("title"),
        updated_at=row["updated_at"].isoformat(),
        awaiting_user_clarification=False,
        last_answer=None,
    )


@router.delete("/sessions/{conversation_id}")
def delete_session(
    conversation_id: str,
    container: ServiceContainer = Depends(get_container),
    current_user: SessionUser = Depends(get_current_user),
):
    try:
        deleted = container.entity_repository.delete_chat_session(
            current_user.username,
            conversation_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="会话ID无效") from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"message": "删除成功"}


@router.get("/sessions/{conversation_id}/messages", response_model=list[ChatMessageItem])
def list_messages(
    conversation_id: str,
    container: ServiceContainer = Depends(get_container),
    current_user: SessionUser = Depends(get_current_user),
):
    try:
        rows = container.entity_repository.get_chat_messages(
            current_user.username, conversation_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="会话不存在") from exc
    return [
        ChatMessageItem(
            id=int(row["id"]),
            role=str(row["role"]),
            content=str(row["content"]),
            created_at=row["created_at"].isoformat(),
            metadata=dict(row.get("metadata") or {}),
        )
        for row in rows
    ]
