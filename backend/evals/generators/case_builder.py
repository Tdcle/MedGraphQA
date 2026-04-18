from __future__ import annotations

import hashlib
import random
import re
from dataclasses import asdict
from typing import Any

from .kg_client import AliasRecord, DiseaseProfile


NEGATION_WORDS = ["没有", "不", "暂时没有", "无"]
INTENTS = {
    "cure": "disease_cure_way",
    "check": "disease_check",
    "drugs": "disease_drugs",
    "desc": "disease_desc",
}


class EvalCaseBuilder:
    def __init__(
        self,
        profiles: list[DiseaseProfile],
        aliases: dict[str, list[AliasRecord]],
        alias_samples: list[AliasRecord],
        seed: int = 42,
    ) -> None:
        self.profiles = profiles
        self.aliases = aliases
        self.alias_samples = alias_samples
        self.random = random.Random(seed)

    def build_core(self, limit: int) -> dict[str, list[dict[str, Any]]]:
        per_group = max(20, limit)
        return {
            "core_single_turn": self.core_single_turn_cases(per_group),
            "core_multi_turn": self.multi_turn_cases(per_group),
        }

    def build_all(self, limit: int) -> dict[str, list[dict[str, Any]]]:
        per_group = max(20, limit // 5)
        return {
            **self.build_core(per_group),
            "entity_normalization": self.entity_normalization_cases(per_group),
            "clinical_context": self.clinical_context_cases(per_group),
            "disease_resolution": self.disease_resolution_cases(per_group),
            "answer_grounding": self.answer_grounding_cases(per_group),
        }

    def core_single_turn_cases(self, limit: int) -> list[dict[str, Any]]:
        cases: list[dict[str, Any]] = []
        for profile in self.profiles:
            positive = self._select_non_conflicting_terms(profile.symptoms, 3)
            if len(positive) < 3:
                continue
            negative = self._select_non_conflicting_terms(
                profile.symptoms,
                2,
                blocked=positive,
            )
            negative_text = ""
            if negative:
                negative_text = "，没有" + "，也没有".join(negative)
            query = f"我最近{positive[0]}，还有{positive[1]}和{positive[2]}{negative_text}，怎么办"
            cases.append(
                self._case(
                    "core_single",
                    {
                        "query": query,
                        "target": {
                            "expected_positive_symptoms": positive,
                            "expected_negated_symptoms": negative,
                            "expected_possible_diseases": [profile.name],
                            "top_k": 5,
                        },
                        "metrics": [
                            "positive_symptom_recall",
                            "negated_symptom_filter",
                            "disease_top5_recall",
                        ],
                        "source": "neo4j.disease_symptoms",
                    },
                )
            )
            if len(cases) >= limit:
                break
        return cases

    def entity_normalization_cases(self, limit: int) -> list[dict[str, Any]]:
        cases: list[dict[str, Any]] = []
        for record in self.alias_samples[: limit * 2]:
            query = self._entity_query(record)
            cases.append(
                self._case(
                    "entity",
                    {
                        "query": query,
                        "target": {
                            "canonical_name": record.canonical_name,
                            "entity_type": record.entity_type,
                            "matched_alias": record.alias,
                        },
                        "metrics": ["entity_recall@1", "entity_type_accuracy"],
                        "source": "postgres.entity_alias",
                    },
                )
            )
            if len(cases) >= limit:
                break
        return cases

    def clinical_context_cases(self, limit: int) -> list[dict[str, Any]]:
        cases: list[dict[str, Any]] = []
        for profile in self.profiles:
            positive = self._select_non_conflicting_terms(profile.symptoms, 2)
            negative = self._select_non_conflicting_terms(
                profile.symptoms,
                2,
                blocked=positive,
            )
            if len(positive) < 2 or len(negative) < 2:
                continue
            query = (
                f"今天早上开始{positive[0]}，还有点{positive[1]}，"
                f"{NEGATION_WORDS[0]}{negative[0]}，也{NEGATION_WORDS[1]}{negative[1]}"
            )
            cases.append(
                self._case(
                    "clinical",
                    {
                        "query": query,
                        "target": {
                            "symptoms": positive,
                            "negated_symptoms": negative,
                            "duration": "今天早上",
                        },
                        "metrics": [
                            "clinical_symptom_f1",
                            "clinical_negation_accuracy",
                            "duration_accuracy",
                        ],
                        "source": "neo4j.disease_symptoms",
                    },
                )
            )
            if len(cases) >= limit:
                break
        return cases

    def disease_resolution_cases(self, limit: int) -> list[dict[str, Any]]:
        cases: list[dict[str, Any]] = []
        for profile in self.profiles:
            symptoms = self._select_non_conflicting_terms(profile.symptoms, 3)
            if len(symptoms) < 3:
                continue
            query = f"我最近{symptoms[0]}，还有{symptoms[1]}和{symptoms[2]}，怎么办"
            cases.append(
                self._case(
                    "disease",
                    {
                        "query": query,
                        "target": {
                            "expected_disease_in_top_k": [profile.name],
                            "expected_symptoms": symptoms,
                            "top_k": 5,
                        },
                        "metrics": ["disease_top5_recall", "symptom_normalization_recall"],
                        "source": "neo4j.disease_symptoms",
                    },
                )
            )
            if len(cases) >= limit:
                break
        return cases

    def multi_turn_cases(self, limit: int) -> list[dict[str, Any]]:
        cases: list[dict[str, Any]] = []
        for profile in self.profiles:
            positive = self._select_non_conflicting_terms(profile.symptoms, 4)
            negative = self._select_non_conflicting_terms(
                profile.symptoms,
                1,
                blocked=positive,
            )
            if len(positive) < 4 or len(negative) < 1:
                continue
            turns = [
                f"早上起来{positive[0]}，有点{positive[1]}",
                f"还有{positive[2]}，暂时没有{negative[0]}",
                f"{positive[3]}比较明显，持续一天了",
            ]
            cases.append(
                self._case(
                    "multi_turn",
                    {
                        "turns": turns,
                        "target": {
                            "expected_positive_symptoms": [
                                positive[0],
                                positive[1],
                                positive[2],
                                positive[3],
                            ],
                            "expected_negated_symptoms": negative,
                            "expected_possible_diseases": [profile.name],
                            "max_follow_up_turns": 2,
                            "expected_final_decision": [
                                "answer_direct",
                                "answer_inferred",
                                "answer_possible",
                            ],
                        },
                        "metrics": [
                            "multi_turn_context_merge",
                            "negated_symptom_filter",
                            "follow_up_turn_limit",
                            "disease_top5_recall",
                        ],
                        "source": "neo4j.disease_symptoms",
                    },
                )
            )
            if len(cases) >= limit:
                break
        return cases

    def answer_grounding_cases(self, limit: int) -> list[dict[str, Any]]:
        cases: list[dict[str, Any]] = []
        for profile in self.profiles:
            alias = self._display_name(profile.name)
            if profile.cure_methods:
                cases.append(
                    self._case(
                        "answer",
                        {
                            "query": f"{alias}怎么治疗？",
                            "target": {
                                "expected_disease": profile.name,
                                "expected_intents": [INTENTS["cure"]],
                                "must_use_evidence": profile.cure_methods,
                            },
                            "metrics": ["answer_groundedness", "intent_accuracy"],
                            "source": "neo4j.treatment",
                        },
                    )
                )
            if len(cases) >= limit:
                break
            if profile.checks:
                cases.append(
                    self._case(
                        "answer",
                        {
                            "query": f"怀疑{alias}需要做什么检查？",
                            "target": {
                                "expected_disease": profile.name,
                                "expected_intents": [INTENTS["check"]],
                                "must_use_evidence": profile.checks,
                            },
                            "metrics": ["answer_groundedness", "intent_accuracy"],
                            "source": "neo4j.checks",
                        },
                    )
                )
            if len(cases) >= limit:
                break
        return cases

    def _display_name(self, canonical_name: str) -> str:
        records = self.aliases.get(canonical_name) or []
        preferred = [
            item.alias
            for item in records
            if item.alias != canonical_name and item.alias_type in {"colloquial", "synonym", "manual"}
        ]
        if preferred:
            return preferred[0]
        return canonical_name

    @staticmethod
    def _entity_query(record: AliasRecord) -> str:
        if record.entity_type == "疾病症状":
            return f"我有点{record.alias}，这是什么情况"
        if record.entity_type == "疾病":
            return f"{record.alias}应该怎么办"
        if record.entity_type == "药品":
            return f"{record.alias}是谁生产的"
        return f"请帮我识别{record.alias}"

    @classmethod
    def _select_non_conflicting_terms(
        cls,
        terms: list[str],
        count: int,
        blocked: list[str] | None = None,
    ) -> list[str]:
        selected: list[str] = []
        blocked = blocked or []
        for term in terms:
            term = str(term or "").strip()
            if not term:
                continue
            if cls._has_conflict(term, selected) or cls._has_conflict(term, blocked):
                continue
            selected.append(term)
            if len(selected) >= count:
                break
        return selected

    @classmethod
    def _has_conflict(cls, term: str, others: list[str]) -> bool:
        return any(cls._is_conflicting_term(term, other) for other in others)

    @classmethod
    def _is_conflicting_term(cls, left: str, right: str) -> bool:
        left_norm = cls._normalize_for_conflict(left)
        right_norm = cls._normalize_for_conflict(right)
        if not left_norm or not right_norm:
            return False
        if left_norm == right_norm:
            return True
        shorter, longer = sorted([left_norm, right_norm], key=len)
        return len(shorter) >= 2 and shorter in longer

    @staticmethod
    def _normalize_for_conflict(value: str) -> str:
        return re.sub(r"[\s，,。；;、（）()【】\\[\\]“”\"'：:]+", "", str(value or ""))

    @staticmethod
    def _case(prefix: str, payload: dict[str, Any]) -> dict[str, Any]:
        raw = repr(sorted(payload.items())).encode("utf-8", errors="ignore")
        digest = hashlib.sha1(raw).hexdigest()[:12]
        return {
            "id": f"{prefix}_{digest}",
            **payload,
        }


def serialize_profile(profile: DiseaseProfile) -> dict[str, Any]:
    return asdict(profile)
