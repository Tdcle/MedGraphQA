import logging
import re
from typing import Sequence

from app.services.entity_search import EntityCandidate, normalize_entity_text
from app.services.memory_repository import MemoryRepository, UserMemory
from app.services.metrics import (
    MEMORY_EXTRACT_CANDIDATES_TOTAL,
    MEMORY_LOADED_TOTAL,
    MEMORY_SAVED_TOTAL,
)


logger = logging.getLogger("medgraphqa.memory")

_AGE_RE = re.compile(r"(?:我|本人)?\s*(\d{1,3})\s*岁")
_ALLERGY_RE = re.compile(r"(?:我|本人)?(?:对)?([\u4e00-\u9fffA-Za-z0-9]{2,20})(?:过敏|会过敏)")
_MEDICATION_RE = re.compile(
    r"(?:我|本人)?(?:正在|一直|长期|现在|目前|平时|每天|每日|规律|按医嘱)(?:在)?"
    r"(?:吃|服用|使用|用)([\u4e00-\u9fffA-Za-z0-9、，,]{2,40})(?:药)?"
)
_NEGATION_MARKERS = ("没有", "没", "无", "未", "否认", "不是", "不")
_PREGNANCY_NEGATION_RE = re.compile(r"(?:没有|没|无|未|否认|不是).{0,8}(?:怀孕|妊娠|孕)")
_QUESTION_OR_GENERIC_RE = re.compile(
    r"(什么|哪种|哪类|多少|几|怎么|如何|能不能|可不可以|需要|应该|合适|吗|么|是否)"
)

_CHRONIC_DISEASES = [
    "高血压",
    "糖尿病",
    "冠心病",
    "哮喘",
    "慢阻肺",
    "慢性胃炎",
    "胃炎",
    "乙肝",
    "肾病",
    "肝病",
]

_PREGNANCY_PATTERNS = [
    "我怀孕",
    "本人怀孕",
    "怀孕了",
    "孕期",
    "妊娠",
    "孕妇",
]

_SHORT_PREFERENCE_PATTERNS = [
    "以后回答简短",
    "回答简短",
    "说简单点",
    "简单说",
]


class MemoryService:
    def __init__(self, repository: MemoryRepository) -> None:
        self.repository = repository

    def load_for_chat(self, user_id: str, query: str, intents: Sequence[str]) -> list[UserMemory]:
        memory_types = [
            "allergy",
            "pregnancy",
            "chronic_disease",
            "medication",
            "profile",
        ]
        if "drug_producer" in intents or any("drug" in intent for intent in intents):
            memory_types.extend(["contraindication"])
        memories = self.repository.list_active(
            user_id=user_id,
            memory_types=memory_types,
            limit=20,
        )
        preferences = self.repository.list_active(
            user_id=user_id,
            memory_types=["preference"],
            limit=5,
        )
        result = memories + preferences
        for memory in result:
            MEMORY_LOADED_TOTAL.labels(memory_type=memory.memory_type).inc()
        return result

    def list_memories(self, user_id: str) -> list[UserMemory]:
        return self.repository.list_for_user(user_id=user_id)

    def activate_memory(self, user_id: str, memory_id: int) -> UserMemory | None:
        return self.repository.set_status(user_id, memory_id, "active")

    def delete_memory(self, user_id: str, memory_id: int) -> UserMemory | None:
        return self.repository.set_status(user_id, memory_id, "deleted")

    def update_memory(self, user_id: str, memory_id: int, text: str) -> UserMemory | None:
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("memory text cannot be empty")
        return self.repository.update(
            user_id=user_id,
            memory_id=memory_id,
            text=clean_text,
            value={"text": clean_text},
        )

    def format_for_prompt(self, memories: Sequence[UserMemory]) -> str:
        lines: list[str] = []
        seen: set[str] = set()
        for memory in memories:
            text = memory.to_prompt_text()
            if not text or text in seen:
                continue
            seen.add(text)
            lines.append(f"- {text}")
        return "\n".join(lines)

    def extract_and_save(
        self,
        user_id: str,
        query: str,
        entities: Sequence[EntityCandidate],
    ) -> list[UserMemory]:
        candidates = self._extract_candidates(user_id, query, entities)
        for candidate in candidates:
            MEMORY_EXTRACT_CANDIDATES_TOTAL.labels(memory_type=candidate.memory_type).inc()
        saved: list[UserMemory] = []
        for candidate in candidates:
            try:
                memory = self.repository.upsert(candidate)
                MEMORY_SAVED_TOTAL.labels(
                    memory_type=memory.memory_type,
                    status=memory.status,
                ).inc()
                saved.append(memory)
            except Exception:
                logger.exception(
                    "failed to save user memory type=%s key=%s",
                    candidate.memory_type,
                    candidate.memory_key,
                )
        return saved

    def _extract_candidates(
        self,
        user_id: str,
        query: str,
        entities: Sequence[EntityCandidate],
    ) -> list[UserMemory]:
        memories: list[UserMemory] = []
        memories.extend(self._extract_profile(user_id, query))
        memories.extend(self._extract_allergies(user_id, query))
        memories.extend(self._extract_chronic_diseases(user_id, query))
        memories.extend(self._extract_medications(user_id, query))
        memories.extend(self._extract_pregnancy(user_id, query))
        memories.extend(self._extract_preferences(user_id, query))
        return self._dedupe(memories)

    def _extract_profile(self, user_id: str, query: str) -> list[UserMemory]:
        memories: list[UserMemory] = []
        match = _AGE_RE.search(query)
        if match:
            age = int(match.group(1))
            if 0 < age < 120:
                memories.append(
                    self._memory(
                        user_id,
                        "profile",
                        "age",
                        {"age": age},
                        f"年龄：{age}岁。",
                    )
                )
        if re.search(r"(?:我是|本人是|性别[:：]?|[，,]\s*)男(?:性)?(?:[，,。.!！?？\s]|$)", query):
            memories.append(
                self._memory(user_id, "profile", "sex", {"sex": "男"}, "性别：男。")
            )
        if re.search(r"(?:我是|本人是|性别[:：]?|[，,]\s*)女(?:性)?(?:[，,。.!！?？\s]|$)", query):
            memories.append(
                self._memory(user_id, "profile", "sex", {"sex": "女"}, "性别：女。")
            )
        return memories

    def _extract_allergies(self, user_id: str, query: str) -> list[UserMemory]:
        memories: list[UserMemory] = []
        for match in _ALLERGY_RE.finditer(query):
            if self._is_negated_context(query, match.start()):
                continue
            name = self._clean_name(match.group(1))
            if not self._valid_memory_name(name):
                continue
            key = normalize_entity_text(name)
            memories.append(
                self._memory(
                    user_id,
                    "allergy",
                    key,
                    {"name": name},
                    f"对{name}过敏。",
                )
            )
        return memories

    def _extract_chronic_diseases(self, user_id: str, query: str) -> list[UserMemory]:
        memories: list[UserMemory] = []
        for disease in _CHRONIC_DISEASES:
            if disease not in query:
                continue
            if self._is_negated_context(query, query.find(disease)):
                continue
            if not re.search(rf"(?:我|本人)?(?:有|患有|得了|确诊|病史).{{0,6}}{re.escape(disease)}|{re.escape(disease)}病史", query):
                continue
            key = normalize_entity_text(disease)
            memories.append(
                self._memory(
                    user_id,
                    "chronic_disease",
                    key,
                    {"name": disease, "diagnosed": True},
                    f"有{disease}病史。",
                )
            )
        return memories

    def _extract_medications(self, user_id: str, query: str) -> list[UserMemory]:
        memories: list[UserMemory] = []
        for match in _MEDICATION_RE.finditer(query):
            raw = self._clean_name(match.group(1))
            if not self._valid_memory_name(raw):
                continue
            for name in re.split(r"[、，,]", raw):
                drug = self._clean_name(name)
                if not self._valid_memory_name(drug):
                    continue
                key = normalize_entity_text(drug)
                memories.append(
                    self._memory(
                        user_id,
                        "medication",
                        key,
                        {"name": drug},
                        f"正在服用{drug}。",
                    )
                )
        return memories

    def _extract_pregnancy(self, user_id: str, query: str) -> list[UserMemory]:
        if _PREGNANCY_NEGATION_RE.search(query):
            return []
        if not any(pattern in query for pattern in _PREGNANCY_PATTERNS):
            return []
        return [
            self._memory(
                user_id,
                "pregnancy",
                "pregnant",
                {"pregnant": True},
                "处于孕期或可能怀孕。",
            )
        ]

    def _extract_preferences(self, user_id: str, query: str) -> list[UserMemory]:
        if not any(pattern in query for pattern in _SHORT_PREFERENCE_PATTERNS):
            return []
        return [
            self._memory(
                user_id,
                "preference",
                "answer_style",
                {"style": "concise"},
                "偏好简短直接的回答。",
            )
        ]

    @staticmethod
    def _memory(
        user_id: str,
        memory_type: str,
        memory_key: str,
        value: dict,
        text: str,
        confidence: float = 1.0,
    ) -> UserMemory:
        return UserMemory(
            id=None,
            user_id=user_id,
            memory_type=memory_type,
            memory_key=memory_key,
            value=value,
            text=text,
            source="rule",
            confidence=confidence,
            status="pending",
            metadata={"extractor": "rule_v1"},
        )

    @staticmethod
    def _clean_name(value: str) -> str:
        return re.sub(r"[，,。.!！?？\s]+$", "", value.strip())

    @staticmethod
    def _valid_memory_name(value: str) -> bool:
        if len(value) < 2 or len(value) > 30:
            return False
        if any(marker in value for marker in _NEGATION_MARKERS):
            return False
        if _QUESTION_OR_GENERIC_RE.search(value):
            return False
        return value not in {
            "这个",
            "那个",
            "药物",
            "药",
            "东西",
            "过敏",
            "什么药",
            "什么",
            "一些",
            "一点",
        }

    @staticmethod
    def _is_negated_context(query: str, start: int) -> bool:
        if start < 0:
            return False
        prefix = query[:start]
        last_boundary = max(prefix.rfind(item) for item in ("，", ",", "。", "；", ";", "！", "!", "？", "?"))
        window = prefix[last_boundary + 1:start]
        return any(marker in window for marker in _NEGATION_MARKERS)

    @staticmethod
    def _dedupe(memories: Sequence[UserMemory]) -> list[UserMemory]:
        result: dict[tuple[str, str], UserMemory] = {}
        for memory in memories:
            result[(memory.memory_type, memory.memory_key)] = memory
        return list(result.values())
