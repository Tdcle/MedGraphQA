from dataclasses import dataclass
import logging
from typing import List

from app.services.disease_resolution import DiseaseResolutionResult, DiseaseResolver
from app.services.entity_search import EntityCandidate
from app.services.intent_service import INTENT_LABELS
from app.services.kg_service import GraphService


logger = logging.getLogger("medgraphqa.knowledge")

ATTR_INTENT_MAP = {
    "disease_desc": "疾病简介",
    "disease_cause": "疾病病因",
    "disease_prevent": "预防措施",
    "disease_cycle": "治疗周期",
    "disease_prob": "治愈概率",
    "disease_population": "疾病易感人群",
}

REL_INTENT_MAP = {
    "disease_drugs": ("疾病使用药品", "药品"),
    "disease_do_eat": ("疾病宜吃食物", "食物"),
    "disease_not_eat": ("疾病忌吃食物", "食物"),
    "disease_check": ("疾病所需检查", "检查项目"),
    "disease_department": ("疾病所属科目", "科目"),
    "disease_symptom": ("疾病的症状", "疾病症状"),
    "disease_cure_way": ("治疗的方法", "治疗方法"),
    "disease_acompany": ("疾病并发疾病", "疾病"),
}


@dataclass
class KnowledgeResult:
    evidence: list[str]
    used_intents: list[str]
    entities: list[EntityCandidate]
    follow_up_answer: str | None
    disease_resolution: DiseaseResolutionResult | None


class MedicalKnowledgeGatherer:
    def __init__(
        self,
        graph_service: GraphService,
        disease_resolver: DiseaseResolver,
    ) -> None:
        self.graph_service = graph_service
        self.disease_resolver = disease_resolver

    def gather(
        self,
        effective_query: str,
        intents: List[str],
        entities: List[EntityCandidate],
        follow_up_turns: int = 0,
        max_follow_up_turns: int = 2,
        possible_confidence_threshold: float = 0.55,
        possible_candidate_limit: int = 3,
        negated_symptoms: List[str] | None = None,
    ) -> KnowledgeResult:
        evidence: list[str] = []
        used_intents: list[str] = []
        follow_up_answer = None
        disease_resolution = None
        updated_entities = list(entities)
        symptoms = [
            item for item in updated_entities if item.entity_type == "疾病症状"
        ]
        symptom_names = [item.canonical_name for item in symptoms]
        disease = self._select_disease(
            [item for item in updated_entities if item.entity_type == "疾病"],
            symptom_names,
        )
        drug = self._select_best_entity(
            [item for item in updated_entities if item.entity_type == "药品"]
        )
        logger.info(
            "operation=knowledge.select_entities status=ok disease=%s symptom_count=%s drug=%s disease_entity_count=%s",
            disease.canonical_name if disease else None,
            len(symptoms),
            drug.canonical_name if drug else None,
            len([item for item in updated_entities if item.entity_type == "疾病"]),
        )
        disease_name = disease.canonical_name if disease else None
        drug_name = drug.canonical_name if drug else None

        if needs_disease(intents):
            disease_resolution = self.disease_resolver.resolve(
                query=effective_query,
                disease=disease,
                symptoms=symptoms,
                negated_symptoms=negated_symptoms or [],
            )
            evidence.extend(disease_resolution.evidence)
            if (
                disease_resolution.decision == "ask_follow_up"
                and follow_up_turns >= max_follow_up_turns
            ):
                possible = self._possible_candidates(
                    disease_resolution,
                    threshold=possible_confidence_threshold,
                    limit=possible_candidate_limit,
                )
                if not possible:
                    possible = disease_resolution.candidates[:possible_candidate_limit]
                if possible:
                    disease_resolution = self._mark_possible_answer(
                        disease_resolution,
                        possible,
                        follow_up_turns,
                    )
                    evidence[:] = disease_resolution.evidence
                    self._append_possible_disease_entities(
                        possible,
                        symptoms,
                        updated_entities,
                    )
                    self._append_possible_disease_evidence(
                        possible,
                        intents,
                        evidence,
                        used_intents,
                    )
                    disease_name = None
                else:
                    follow_up_answer = disease_resolution.follow_up_question
            else:
                disease_name = self._disease_name_from_resolution(
                    disease_resolution,
                    symptoms,
                    updated_entities,
                )
                if disease_resolution.decision == "ask_follow_up":
                    follow_up_answer = disease_resolution.follow_up_question

        if disease_name:
            self._append_disease_evidence(
                disease_name,
                intents,
                evidence,
                used_intents,
            )

        if "drug_producer" in intents and drug_name:
            producers = self.graph_service.get_producer_by_drug(drug_name)
            if producers:
                used_intents.append(INTENT_LABELS["drug_producer"])
                evidence.append(f"{drug_name}的生产商：{'、'.join(producers[:20])}")

        return KnowledgeResult(
            evidence=evidence,
            used_intents=used_intents,
            entities=updated_entities,
            follow_up_answer=follow_up_answer,
            disease_resolution=disease_resolution,
        )

    @staticmethod
    def _possible_candidates(
        resolution: DiseaseResolutionResult,
        threshold: float,
        limit: int,
    ):
        candidates = [
            item
            for item in resolution.candidates
            if float(item.confidence or 0.0) >= threshold
        ]
        return candidates[:limit]

    @staticmethod
    def _mark_possible_answer(
        resolution: DiseaseResolutionResult,
        possible,
        follow_up_turns: int,
    ) -> DiseaseResolutionResult:
        summary = "、".join(
            f"{item.disease}({item.confidence:.2f})" for item in possible
        )
        return DiseaseResolutionResult(
            decision="answer_possible",
            disease_name=possible[0].disease,
            confidence=possible[0].confidence,
            candidates=resolution.candidates,
            evidence=resolution.evidence
            + [
                (
                    f"已完成{follow_up_turns}轮追问，仍无法唯一确定疾病。"
                    f"以下候选达到可能疾病阈值，仅作为可能性参考：{summary}。"
                )
            ],
            follow_up_question=None,
        )

    @staticmethod
    def _append_possible_disease_entities(
        possible,
        symptoms: List[EntityCandidate],
        entities: List[EntityCandidate],
    ) -> None:
        existing = {
            (item.entity_type, item.canonical_name)
            for item in entities
        }
        for item in possible:
            key = ("疾病", item.disease)
            if key in existing:
                continue
            entities.append(
                EntityCandidate(
                    entity_id=0,
                    alias_id=None,
                    canonical_name=item.disease,
                    entity_type="疾病",
                    matched_alias="、".join(sym.canonical_name for sym in symptoms),
                    normalized_alias="、".join(sym.normalized_alias for sym in symptoms),
                    alias_type="possible_disease",
                    confidence=item.confidence,
                    source="neo4j",
                    mention="、".join(sym.mention for sym in symptoms),
                    match_method="possible_disease",
                    score=item.confidence,
                )
            )
            existing.add(key)

    def _append_possible_disease_evidence(
        self,
        possible,
        intents: List[str],
        evidence: List[str],
        used_intents: List[str],
    ) -> None:
        for candidate in possible:
            self._append_disease_evidence(
                disease_name=candidate.disease,
                intents=intents,
                evidence=evidence,
                used_intents=used_intents,
                prefix="可能疾病",
            )

    def _select_disease(
        self,
        diseases: List[EntityCandidate],
        symptom_names: List[str],
    ) -> EntityCandidate | None:
        reliable = [
            item
            for item in diseases
            if not self.disease_resolver._is_unreliable_disease_match(
                item,
                symptom_names,
            )
        ]
        candidates = reliable or [
            item for item in diseases if item.canonical_name not in symptom_names
        ]
        return self._select_best_entity(candidates)

    @staticmethod
    def _select_best_entity(
        entities: List[EntityCandidate],
    ) -> EntityCandidate | None:
        if not entities:
            return None
        return sorted(
            entities,
            key=lambda item: (
                "postgres_exact" not in item.match_method,
                item.match_method == "elasticsearch_vector",
                -float(item.confidence or 0.0),
                -float(item.score or 0.0),
            ),
        )[0]

    def _disease_name_from_resolution(
        self,
        resolution: DiseaseResolutionResult,
        symptoms: List[EntityCandidate],
        entities: List[EntityCandidate],
    ) -> str | None:
        if resolution.decision == "answer_direct":
            return resolution.disease_name
        if resolution.decision != "answer_inferred" or not resolution.disease_name:
            return None

        entities.append(
            EntityCandidate(
                entity_id=0,
                alias_id=None,
                canonical_name=resolution.disease_name,
                entity_type="疾病",
                matched_alias="、".join(item.canonical_name for item in symptoms),
                normalized_alias="、".join(
                    item.normalized_alias for item in symptoms
                ),
                alias_type="disease_resolution",
                confidence=resolution.confidence,
                source="neo4j",
                mention="、".join(item.mention for item in symptoms),
                match_method="disease_resolution",
                score=resolution.confidence,
            )
        )
        return resolution.disease_name

    def _append_disease_evidence(
        self,
        disease_name: str,
        intents: List[str],
        evidence: List[str],
        used_intents: List[str],
        prefix: str | None = None,
    ) -> None:
        for intent in intents:
            if intent in ATTR_INTENT_MAP:
                field_name = ATTR_INTENT_MAP[intent]
                value = self.graph_service.get_disease_attribute(
                    disease_name,
                    field_name,
                )
                if value:
                    if INTENT_LABELS[intent] not in used_intents:
                        used_intents.append(INTENT_LABELS[intent])
                    title = f"{disease_name}的{field_name}"
                    if prefix:
                        title = f"{prefix}“{disease_name}”的{field_name}"
                    evidence.append(f"{title}：{value}")
                continue

            if intent in REL_INTENT_MAP:
                relation, target = REL_INTENT_MAP[intent]
                rows = self.graph_service.get_related_entities(
                    disease_name,
                    relation,
                    target,
                )
                if rows:
                    if INTENT_LABELS[intent] not in used_intents:
                        used_intents.append(INTENT_LABELS[intent])
                    title = f"{disease_name}的{target}"
                    if prefix:
                        title = f"{prefix}“{disease_name}”的{target}"
                    evidence.append(f"{title}：{'、'.join(rows[:15])}")


def needs_disease(intents: List[str]) -> bool:
    return any(intent in ATTR_INTENT_MAP or intent in REL_INTENT_MAP for intent in intents)
