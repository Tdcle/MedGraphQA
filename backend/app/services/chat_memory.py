from typing import List

from app.services.disease_resolution import DiseaseResolutionResult
from app.services.entity_search import EntityCandidate


def entity_to_memory(item: EntityCandidate) -> dict:
    return item.to_log_dict()


def entity_from_memory(data: dict) -> EntityCandidate | None:
    try:
        return EntityCandidate(
            entity_id=int(data.get("entity_id") or 0),
            alias_id=int(data["alias_id"]) if data.get("alias_id") is not None else None,
            canonical_name=str(data["canonical_name"]),
            entity_type=str(data["entity_type"]),
            matched_alias=str(data.get("matched_alias") or data["canonical_name"]),
            normalized_alias=str(data.get("normalized_alias") or data["canonical_name"]),
            alias_type=str(data.get("alias_type") or "memory"),
            confidence=float(data.get("confidence") or 0.0),
            source=str(data.get("source") or "memory"),
            mention=str(data.get("mention") or data["canonical_name"]),
            match_method=str(data.get("match_method") or "memory"),
            score=float(data.get("score") or 0.0),
        )
    except (KeyError, TypeError, ValueError):
        return None


def entities_from_state(state: dict) -> list[EntityCandidate]:
    result: list[EntityCandidate] = []
    for item in state.get("known_entities", []):
        if isinstance(item, dict):
            candidate = entity_from_memory(item)
            if candidate:
                result.append(candidate)
    return result


def merge_entities(
    previous: List[EntityCandidate], current: List[EntityCandidate]
) -> List[EntityCandidate]:
    merged: dict[tuple[str, str], EntityCandidate] = {}
    current_ids = {id(item) for item in current}
    for item in previous + current:
        key = (item.entity_type, item.canonical_name)
        existing = merged.get(key)
        if existing is None:
            merged[key] = item
            continue
        if id(item) in current_ids and id(existing) not in current_ids:
            merged[key] = item
        elif item.confidence > existing.confidence:
            merged[key] = item

    return sorted(
        merged.values(),
        key=lambda item: (
            item.entity_type != "疾病",
            item.entity_type != "疾病症状",
            -float(item.confidence),
            -float(item.score),
        ),
    )


def merge_intents(state: dict, current_intents: List[str]) -> List[str]:
    pending = state.get("pending_intents") or []
    if state.get("awaiting_user_clarification") and pending:
        merged = list(pending)
        for intent in current_intents:
            if intent not in merged and intent != "disease_desc":
                merged.append(intent)
        return merged[:5]
    return current_intents


def effective_query(query: str, state: dict) -> str:
    pending_query = str(state.get("pending_query") or "").strip()
    if state.get("awaiting_user_clarification") and pending_query:
        return f"{pending_query}\n用户补充信息：{query}"
    return query


def build_next_state(
    previous_state: dict,
    query: str,
    intents: List[str],
    entities: List[EntityCandidate],
    answer: str,
    follow_up_answer: str | None,
    disease_resolution: DiseaseResolutionResult | None,
) -> dict:
    known_symptoms = []
    for item in entities:
        if item.entity_type == "疾病症状" and item.canonical_name not in known_symptoms:
            known_symptoms.append(item.canonical_name)

    known_disease = None
    for item in entities:
        if item.entity_type == "疾病":
            known_disease = entity_to_memory(item)
            break

    awaiting = bool(follow_up_answer)
    previous_follow_up_turns = int(previous_state.get("follow_up_turns") or 0)
    follow_up_turns = previous_follow_up_turns + 1 if awaiting else 0
    pending_query = previous_state.get("pending_query")
    if awaiting and not pending_query:
        pending_query = query
    if not awaiting:
        pending_query = None

    return {
        "pending_intents": intents if awaiting else [],
        "pending_query": pending_query,
        "known_entities": [entity_to_memory(item) for item in entities],
        "known_symptoms": known_symptoms,
        "negated_symptoms": previous_state.get("negated_symptoms", []),
        "known_disease": known_disease,
        "candidate_diseases": [
            item.to_log_dict() for item in disease_resolution.candidates
        ]
        if disease_resolution
        else previous_state.get("candidate_diseases", []),
        "last_follow_up_question": follow_up_answer,
        "follow_up_turns": follow_up_turns,
        "awaiting_user_clarification": awaiting,
        "last_answer": answer,
    }
