import logging
import re
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Sequence

from app.services.entity_search import EntityCandidate, normalize_entity_text
from app.services.memory_repository import MemoryRepository, UserMemory


logger = logging.getLogger("medgraphqa.memory")

_AGE_RE = re.compile(r"(?:我|本人)?\s*(\d{1,3})\s*岁")
_ALLERGY_RE = re.compile(r"(?:我|本人)?(?:对)?([\u4e00-\u9fffA-Za-z0-9]{2,20})(?:过敏|会过敏)")
_MEDICATION_RE = re.compile(
    r"(?:正在|一直|长期|现在)?(?:吃|服用|用)([\u4e00-\u9fffA-Za-z0-9、，,]{2,40})(?:药)?"
)
_NEGATION_RE = re.compile(r"(?:没有|无|未|否认).{0,8}(?:过敏|高血压|糖尿病|哮喘|冠心病|怀孕)")

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

_SYMPTOM_CONTAINS = {
    "上腹部疼痛": "腹痛",
    "下腹部疼痛": "腹痛",
    "腹部疼痛": "腹痛",
    "腹痛": "腹痛",
    "拉肚子": "腹泻",
}


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
            "recent_symptom",
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
        return memories + preferences

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
        if _NEGATION_RE.search(query):
            return []

        candidates = self._extract_candidates(user_id, query, entities)
        saved: list[UserMemory] = []
        for candidate in candidates:
            try:
                saved.append(self.repository.upsert(candidate))
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
        memories.extend(self._extract_recent_symptoms(user_id, query, entities))
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
        if re.search(r"(?:我是|本人是|性别[:：]?|，|,)?男(?:性)?", query):
            memories.append(
                self._memory(user_id, "profile", "sex", {"sex": "男"}, "性别：男。")
            )
        if re.search(r"(?:我是|本人是|性别[:：]?|，|,)?女(?:性)?", query):
            memories.append(
                self._memory(user_id, "profile", "sex", {"sex": "女"}, "性别：女。")
            )
        return memories

    def _extract_allergies(self, user_id: str, query: str) -> list[UserMemory]:
        memories: list[UserMemory] = []
        for match in _ALLERGY_RE.finditer(query):
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

    def _extract_recent_symptoms(
        self,
        user_id: str,
        query: str,
        entities: Sequence[EntityCandidate],
    ) -> list[UserMemory]:
        if not re.search(r"最近|这几天|近\d+天|今天|昨天|持续", query):
            return []
        symptoms = self._select_recent_symptoms(entities)
        memories: list[UserMemory] = []
        expires_at = datetime.now(timezone.utc) + timedelta(days=14)
        for symptom in symptoms:
            key = normalize_entity_text(symptom)
            memories.append(
                replace(
                    self._memory(
                        user_id,
                        "recent_symptom",
                        key,
                        {"name": symptom},
                        f"近期出现：{symptom}。",
                        confidence=0.85,
                    ),
                    expires_at=expires_at,
                )
            )
        return memories

    def _select_recent_symptoms(
        self,
        entities: Sequence[EntityCandidate],
    ) -> list[str]:
        candidates = [
            item
            for item in entities
            if item.entity_type == "疾病症状"
            and (
                "postgres_exact" in item.match_method
                or item.match_method == "elasticsearch"
                or item.match_method.startswith("rrf:")
            )
            and item.match_method != "elasticsearch_vector"
            and item.confidence >= 0.8
        ]
        if not candidates:
            return []

        best_by_name: dict[str, EntityCandidate] = {}
        for item in candidates:
            normalized = self._normalize_symptom_name(item.canonical_name)
            current = best_by_name.get(normalized)
            if current is None or item.score > current.score:
                best_by_name[normalized] = item

        ranked = sorted(
            best_by_name.items(),
            key=lambda pair: (
                0 if "postgres_exact" in pair[1].match_method else 1,
                -pair[1].score,
                len(pair[0]),
            ),
        )
        return [name for name, _ in ranked[:1]]

    @staticmethod
    def _normalize_symptom_name(name: str) -> str:
        for source, target in _SYMPTOM_CONTAINS.items():
            if source == name or source in name:
                return target
        return name

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
        return value not in {"这个", "那个", "药物", "药", "东西", "过敏"}

    @staticmethod
    def _dedupe(memories: Sequence[UserMemory]) -> list[UserMemory]:
        result: dict[tuple[str, str], UserMemory] = {}
        for memory in memories:
            result[(memory.memory_type, memory.memory_key)] = memory
        return list(result.values())
