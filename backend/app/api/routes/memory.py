from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_container, get_current_user
from app.core.container import ServiceContainer
from app.schemas.memory import MemoryItem, MemoryUpdateRequest
from app.services.auth_service import SessionUser
from app.services.memory_repository import UserMemory


router = APIRouter(prefix="/memories", tags=["memories"])


def _to_item(memory: UserMemory) -> MemoryItem:
    return MemoryItem(
        id=int(memory.id or 0),
        memory_type=memory.memory_type,
        memory_key=memory.memory_key,
        value=memory.value,
        text=memory.text,
        source=memory.source,
        confidence=memory.confidence,
        status=memory.status,
        expires_at=memory.expires_at,
        metadata=memory.metadata or {},
    )


@router.get("", response_model=list[MemoryItem])
def list_memories(
    container: ServiceContainer = Depends(get_container),
    current_user: SessionUser = Depends(get_current_user),
):
    return [
        _to_item(item)
        for item in container.chat_service.memory_service.list_memories(
            current_user.username
        )
    ]


@router.post("/{memory_id}/activate", response_model=MemoryItem)
def activate_memory(
    memory_id: int,
    container: ServiceContainer = Depends(get_container),
    current_user: SessionUser = Depends(get_current_user),
):
    memory = container.chat_service.memory_service.activate_memory(
        current_user.username,
        memory_id,
    )
    if not memory:
        raise HTTPException(status_code=404, detail="记忆不存在")
    return _to_item(memory)


@router.put("/{memory_id}", response_model=MemoryItem)
def update_memory(
    memory_id: int,
    payload: MemoryUpdateRequest,
    container: ServiceContainer = Depends(get_container),
    current_user: SessionUser = Depends(get_current_user),
):
    try:
        memory = container.chat_service.memory_service.update_memory(
            current_user.username,
            memory_id,
            payload.text,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not memory:
        raise HTTPException(status_code=404, detail="记忆不存在")
    return _to_item(memory)


@router.delete("/{memory_id}")
def delete_memory(
    memory_id: int,
    container: ServiceContainer = Depends(get_container),
    current_user: SessionUser = Depends(get_current_user),
):
    memory = container.chat_service.memory_service.delete_memory(
        current_user.username,
        memory_id,
    )
    if not memory:
        raise HTTPException(status_code=404, detail="记忆不存在")
    return {"message": "删除成功"}
