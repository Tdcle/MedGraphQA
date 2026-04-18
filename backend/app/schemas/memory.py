from datetime import datetime

from pydantic import BaseModel, Field


class MemoryItem(BaseModel):
    id: int
    memory_type: str
    memory_key: str
    value: dict
    text: str
    source: str
    confidence: float
    status: str
    expires_at: datetime | None = None
    metadata: dict = Field(default_factory=dict)


class MemoryUpdateRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
