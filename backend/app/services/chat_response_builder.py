from typing import List

from app.schemas.chat import ChatResponse, ResolvedEntity
from app.services.entity_search import EntityCandidate
from app.services.intent_service import INTENT_LABELS


def fallback_answer(query: str, evidence: List[str]) -> str:
    if not evidence:
        return "根据已知信息无法回答该问题。"
    joined = "\n".join(f"- {item}" for item in evidence)
    return f"问题：{query}\n已检索到的知识如下：\n{joined}"


def intent_labels(intents: List[str]) -> List[str]:
    return [INTENT_LABELS.get(intent, intent) for intent in intents]


def serialize_entities(entities: List[EntityCandidate]) -> List[ResolvedEntity]:
    return [
        ResolvedEntity(
            mention=item.mention,
            entity_type=item.entity_type,
            canonical_name=item.canonical_name,
            matched_alias=item.matched_alias,
            alias_type=item.alias_type,
            confidence=round(float(item.confidence), 4),
            score=round(float(item.score), 4),
            match_method=item.match_method,
            source=item.source,
        )
        for item in entities
    ]


def build_chat_response(state: dict, awaiting: bool) -> ChatResponse:
    intents = state.get("used_intents") or intent_labels(state["intents"])
    return ChatResponse(
        conversation_id=state["conversation_id"],
        answer=state["answer"],
        intents=intents,
        entities=serialize_entities(state.get("entities", [])),
        evidence=state.get("evidence", []),
        awaiting_user_clarification=awaiting,
    )
