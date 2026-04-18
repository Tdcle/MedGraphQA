from typing import List

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = None


class ResolvedEntity(BaseModel):
    mention: str
    entity_type: str
    canonical_name: str
    matched_alias: str
    alias_type: str
    confidence: float
    score: float
    match_method: str
    source: str


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    intents: List[str]
    entities: List[ResolvedEntity]
    evidence: List[str]
    awaiting_user_clarification: bool = False


class ChatSessionSummary(BaseModel):
    conversation_id: str
    title: str | None = None
    updated_at: str
    awaiting_user_clarification: bool = False
    last_answer: str | None = None


class ChatMessageItem(BaseModel):
    id: int
    role: str
    content: str
    created_at: str
    metadata: dict = Field(default_factory=dict)
