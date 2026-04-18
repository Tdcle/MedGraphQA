import logging
import math
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

from app.services.entity_search import EntityCandidate, PostgresEntityRepository
from app.services.kg_service import GraphService


logger = logging.getLogger("medgraphqa.disease_resolution")


@dataclass
class DiseaseCandidateScore:
    disease: str
    confidence: float
    matched_symptoms: list[str]
    disease_symptoms: list[str]
    symptom_coverage: float
    symptom_specificity: float
    evidence_count_score: float
    disease_symptom_count: int

    def to_log_dict(self) -> dict:
        data = asdict(self)
        data["confidence"] = round(float(self.confidence), 4)
        data["symptom_coverage"] = round(float(self.symptom_coverage), 4)
        data["symptom_specificity"] = round(float(self.symptom_specificity), 4)
        data["evidence_count_score"] = round(float(self.evidence_count_score), 4)
        data["disease_symptoms"] = self.disease_symptoms[:30]
        return data


@dataclass
class DiseaseResolutionResult:
    decision: str
    disease_name: str | None
    confidence: float
    candidates: list[DiseaseCandidateScore]
    evidence: list[str]
    follow_up_question: str | None = None


class DiseaseResolver:
    def __init__(
        self,
        graph_service: GraphService,
        repository: PostgresEntityRepository,
        confidence_threshold: float,
        top_gap_threshold: float,
        min_symptoms_for_inference: int,
        ask_follow_up_when_below_threshold: bool,
        candidate_limit: int,
    ) -> None:
        self.graph_service = graph_service
        self.repository = repository
        self.confidence_threshold = confidence_threshold
        self.top_gap_threshold = top_gap_threshold
        self.min_symptoms_for_inference = max(1, min_symptoms_for_inference)
        self.ask_follow_up_when_below_threshold = ask_follow_up_when_below_threshold
        self.candidate_limit = candidate_limit

    def resolve(
        self,
        query: str,
        disease: EntityCandidate | None,
        symptoms: Sequence[EntityCandidate],
    ) -> DiseaseResolutionResult:
        symptom_names = self._unique_names(item.canonical_name for item in symptoms)
        weak_disease_evidence: list[str] = []

        if disease and not self._is_unreliable_disease_match(disease, symptom_names):
            confidence = float(disease.confidence or 0.0)
            if confidence >= self.confidence_threshold:
                result = DiseaseResolutionResult(
                    decision="answer_direct",
                    disease_name=disease.canonical_name,
                    confidence=confidence,
                    candidates=[],
                    evidence=[
                        f"已直接识别到疾病“{disease.canonical_name}”，实体匹配置信度为{confidence:.2f}。"
                    ],
                )
                self._log(query, symptom_names, result)
                return result

            weak_disease_evidence.append(
                f"识别到可能疾病“{disease.canonical_name}”，但实体匹配置信度为{confidence:.2f}，低于阈值{self.confidence_threshold:.2f}。"
            )
            if not symptom_names:
                result = DiseaseResolutionResult(
                    decision="ask_follow_up",
                    disease_name=None,
                    confidence=confidence,
                    candidates=[],
                    evidence=weak_disease_evidence,
                    follow_up_question=(
                        f"我识别到你可能在问“{disease.canonical_name}”，但还不够确定。"
                        "请补充你是否已经由医生诊断为该疾病，或描述主要症状、持续时间和检查结果。"
                    ),
                )
                self._log(query, symptom_names, result)
                return result

        if not symptom_names:
            result = DiseaseResolutionResult(
                decision="no_disease_evidence",
                disease_name=None,
                confidence=0.0,
                candidates=[],
                evidence=weak_disease_evidence,
            )
            self._log(query, symptom_names, result)
            return result

        try:
            candidates = self._score_candidates(symptom_names)
        except Exception:
            logger.exception("failed to score disease candidates")
            result = DiseaseResolutionResult(
                decision="ask_follow_up",
                disease_name=None,
                confidence=0.0,
                candidates=[],
                evidence=weak_disease_evidence + [f"已识别到症状：{'、'.join(symptom_names)}。"],
                follow_up_question=self._build_follow_up_question(symptom_names, []),
            )
            self._log(query, symptom_names, result)
            return result

        if not candidates:
            result = DiseaseResolutionResult(
                decision="ask_follow_up",
                disease_name=None,
                confidence=0.0,
                candidates=[],
                evidence=weak_disease_evidence + [f"已识别到症状：{'、'.join(symptom_names)}，但知识图谱中没有找到足够的疾病候选。"],
                follow_up_question=self._build_follow_up_question(symptom_names, []),
            )
            self._log(query, symptom_names, result)
            return result

        top = candidates[0]
        second_confidence = candidates[1].confidence if len(candidates) > 1 else 0.0
        top_gap = top.confidence - second_confidence
        evidence = weak_disease_evidence + [
            (
                f"已识别到症状：{'、'.join(symptom_names)}。"
                f"疾病候选Top{len(candidates)}：{self._format_candidate_summary(candidates)}。"
            )
        ]

        if (
            len(symptom_names) >= self.min_symptoms_for_inference
            and top.confidence >= self.confidence_threshold
            and top_gap >= self.top_gap_threshold
        ):
            result = DiseaseResolutionResult(
                decision="answer_inferred",
                disease_name=top.disease,
                confidence=top.confidence,
                candidates=candidates,
                evidence=evidence
                + [
                    f"候选疾病“{top.disease}”置信度为{top.confidence:.2f}，且与第二候选差值为{top_gap:.2f}，达到查询阈值。"
                ],
            )
            self._log(query, symptom_names, result)
            return result

        if self.ask_follow_up_when_below_threshold:
            reasons: list[str] = []
            if len(symptom_names) < self.min_symptoms_for_inference:
                reasons.append(
                    f"有效症状数为{len(symptom_names)}，少于最低要求{self.min_symptoms_for_inference}个"
                )
            if top.confidence < self.confidence_threshold:
                reasons.append(
                    f"最高候选“{top.disease}”置信度{top.confidence:.2f}低于阈值{self.confidence_threshold:.2f}"
                )
            if top_gap < self.top_gap_threshold:
                reasons.append(
                    f"Top1与Top2差值{top_gap:.2f}低于阈值{self.top_gap_threshold:.2f}"
                )

            result = DiseaseResolutionResult(
                decision="ask_follow_up",
                disease_name=None,
                confidence=top.confidence,
                candidates=candidates,
                evidence=evidence + ["暂不直接确定疾病：" + "；".join(reasons) + "。"],
                follow_up_question=self._build_follow_up_question(symptom_names, candidates),
            )
            self._log(query, symptom_names, result)
            return result

        result = DiseaseResolutionResult(
            decision="no_confident_disease",
            disease_name=None,
            confidence=top.confidence,
            candidates=candidates,
            evidence=evidence,
        )
        self._log(query, symptom_names, result)
        return result

    @staticmethod
    def _is_unreliable_disease_match(
        disease: EntityCandidate,
        symptom_names: Sequence[str],
    ) -> bool:
        if disease.canonical_name in symptom_names:
            return True
        if disease.canonical_name in {"腹痛", "腹泻", "恶心", "呕吐", "发热", "发烧", "咳嗽"}:
            return True
        if "postgres_exact" in disease.match_method:
            return False
        if disease.match_method == "elasticsearch":
            mention = disease.mention or ""
            alias = disease.matched_alias or disease.canonical_name
            return alias not in mention
        if disease.match_method == "elasticsearch_vector":
            return True
        if disease.match_method.startswith("rrf:") and "postgres_exact" not in disease.match_method:
            return True
        return False

    def _score_candidates(self, symptom_names: Sequence[str]) -> list[DiseaseCandidateScore]:
        rows = self.graph_service.get_disease_candidates_by_symptoms(
            symptom_names, limit=max(self.candidate_limit, 2)
        )
        total_disease_count = max(self.graph_service.count_diseases(), 1)
        symptom_disease_counts = self.graph_service.get_symptom_disease_counts(symptom_names)

        scored: list[DiseaseCandidateScore] = []
        for row in rows:
            matched_symptoms = self._unique_names(row.get("matched_symptoms") or [])
            disease_symptoms = self._unique_names(row.get("disease_symptoms") or [])
            symptom_coverage = len(matched_symptoms) / max(len(symptom_names), 1)
            symptom_specificity = self._average_specificity(
                matched_symptoms, symptom_disease_counts, total_disease_count
            )
            evidence_count_score = min(len(symptom_names) / 3, 1.0)
            confidence = (
                0.45 * symptom_coverage
                + 0.35 * symptom_specificity
                + 0.20 * evidence_count_score
            )
            scored.append(
                DiseaseCandidateScore(
                    disease=str(row["disease"]),
                    confidence=confidence,
                    matched_symptoms=matched_symptoms,
                    disease_symptoms=disease_symptoms,
                    symptom_coverage=symptom_coverage,
                    symptom_specificity=symptom_specificity,
                    evidence_count_score=evidence_count_score,
                    disease_symptom_count=int(row.get("disease_symptom_count") or 0),
                )
            )

        return sorted(
            scored,
            key=lambda item: (
                -item.confidence,
                -len(item.matched_symptoms),
                item.disease_symptom_count,
                item.disease,
            ),
        )[: self.candidate_limit]

    @staticmethod
    def _average_specificity(
        symptoms: Sequence[str],
        symptom_disease_counts: dict[str, int],
        total_disease_count: int,
    ) -> float:
        if not symptoms:
            return 0.0
        max_idf = math.log(total_disease_count + 1)
        if max_idf <= 0:
            return 0.0
        scores: list[float] = []
        for symptom in symptoms:
            disease_count = max(symptom_disease_counts.get(symptom, 0), 1)
            idf = math.log((total_disease_count + 1) / (disease_count + 1))
            scores.append(max(0.0, min(idf / max_idf, 1.0)))
        return sum(scores) / len(scores)

    @staticmethod
    def _unique_names(values: Iterable[str]) -> list[str]:
        result: list[str] = []
        for value in values:
            text = str(value).strip()
            if text and text not in result:
                result.append(text)
        return result

    @staticmethod
    def _format_candidate_summary(candidates: Sequence[DiseaseCandidateScore]) -> str:
        return "、".join(
            f"{item.disease}({item.confidence:.2f})" for item in candidates[:5]
        )

    @staticmethod
    def _build_follow_up_question(
        symptoms: Sequence[str], candidates: Sequence[DiseaseCandidateScore]
    ) -> str:
        symptom_text = "、".join(symptoms)
        return (
            f"我识别到你描述的是“{symptom_text}”。"
            "仅凭这些信息还不能确定具体疾病。请补充更能区分病因的信息，"
            "例如症状变化、伴随不适、可能诱因、接触史、既往病史或已有检查结果。"
            "如果症状明显加重或出现严重不适，请及时就医或急诊。"
        )

    def _log(
        self,
        query: str,
        symptoms: Sequence[str],
        result: DiseaseResolutionResult,
    ) -> None:
        self.repository.log_disease_resolution(
            query=query,
            symptoms=symptoms,
            candidates=[item.to_log_dict() for item in result.candidates],
            selected_disease=result.disease_name,
            confidence=result.confidence,
            decision=result.decision,
            follow_up_question=result.follow_up_question,
        )
