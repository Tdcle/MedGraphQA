import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Iterable, Sequence

from app.services.entity_search import EntityCandidate, PostgresEntityRepository
from app.services.kg_service import GraphService


logger = logging.getLogger("medgraphqa.disease_resolution")

_COMMON_DISEASE_PRIORS = {
    "感冒": 1.0,
    "流行性感冒": 0.95,
    "急性上呼吸道感染": 0.95,
    "鼻炎": 0.95,
    "急性鼻炎": 0.9,
    "过敏性鼻炎": 0.9,
    "胃炎": 0.95,
    "急性胃炎": 0.95,
    "慢性胃炎": 0.85,
    "肠胃炎": 0.9,
    "急性胃肠炎": 0.95,
    "胃肠炎": 0.9,
    "咽炎": 0.9,
    "急性咽炎": 0.9,
    "慢性咽炎": 0.85,
    "扁桃体炎": 0.9,
    "肺炎": 0.85,
    "支气管炎": 0.85,
    "肾炎": 0.85,
    "肾病": 0.8,
    "牙病": 0.8,
    "牙龈炎": 0.85,
    "肛裂": 0.85,
    "肛瘘": 0.8,
    "痔": 0.85,
    "胆石": 0.75,
    "胆囊炎": 0.8,
    "烧伤": 0.85,
    "便血": 0.75,
    "咯血": 0.75,
}
_COMMON_DISEASE_KEYWORDS = {
    "感冒": 0.85,
    "鼻炎": 0.82,
    "胃炎": 0.82,
    "咽炎": 0.8,
    "扁桃体炎": 0.8,
    "肺炎": 0.75,
    "支气管炎": 0.75,
    "肾炎": 0.75,
    "牙龈炎": 0.75,
    "胆囊炎": 0.72,
}
_RARER_DISEASE_MARKERS = (
    "综合征",
    "嗜酸",
    "细胞增多",
    "非变态反应",
    "非变应性",
    "小儿",
    "老年",
    "妊娠",
    "新生儿",
    "婴幼儿",
    "寄生虫",
    "绦虫",
    "蛔虫",
    "滴虫",
    "阿米巴",
    "中毒",
    "肿瘤",
    "癌",
    "肉瘤",
    "白血病",
)
_MIN_PRIOR_FOR_SYMPTOM_INFERENCE = 0.5
_SYMPTOM_LIKE_DISEASE_NAMES = {
    "水肿",
    "肿胀",
    "黄疸",
    "咽部异物",
    "咽喉痛",
    "喉咙痛",
    "腹痛",
    "腹泻",
    "恶心",
    "呕吐",
    "发热",
    "发烧",
    "咳嗽",
    "便血",
    "咯血",
    "头痛",
    "头晕",
}


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
    weighted_symptom_coverage: float = 0.0
    disease_symptom_coverage: float = 0.0
    disease_prior: float = 0.55
    negated_symptom_penalty: float = 0.0
    score_components: dict[str, float] = field(default_factory=dict)

    def to_log_dict(self) -> dict:
        data = asdict(self)
        data["confidence"] = round(float(self.confidence), 4)
        data["symptom_coverage"] = round(float(self.symptom_coverage), 4)
        data["symptom_specificity"] = round(float(self.symptom_specificity), 4)
        data["evidence_count_score"] = round(float(self.evidence_count_score), 4)
        data["weighted_symptom_coverage"] = round(float(self.weighted_symptom_coverage), 4)
        data["disease_symptom_coverage"] = round(float(self.disease_symptom_coverage), 4)
        data["disease_prior"] = round(float(self.disease_prior), 4)
        data["negated_symptom_penalty"] = round(float(self.negated_symptom_penalty), 4)
        data["score_components"] = {
            key: round(float(value), 4) for key, value in self.score_components.items()
        }
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
        negated_symptoms: Sequence[str] | None = None,
    ) -> DiseaseResolutionResult:
        symptom_names = self._unique_names(item.canonical_name for item in symptoms)
        negated_symptom_names = self._unique_names(negated_symptoms or [])
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
            candidates = self._score_candidates(symptom_names, negated_symptom_names)
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
            and top.disease_prior >= _MIN_PRIOR_FOR_SYMPTOM_INFERENCE
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
            if top.disease_prior < _MIN_PRIOR_FOR_SYMPTOM_INFERENCE:
                reasons.append(
                    f"最高候选“{top.disease}”常见性先验{top.disease_prior:.2f}低于直接推断要求{_MIN_PRIOR_FOR_SYMPTOM_INFERENCE:.2f}"
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
        if any(
            disease.canonical_name
            and disease.canonical_name in symptom
            and disease.canonical_name != symptom
            for symptom in symptom_names
        ):
            return True
        if disease.canonical_name in _SYMPTOM_LIKE_DISEASE_NAMES:
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

    def _score_candidates(
        self,
        symptom_names: Sequence[str],
        negated_symptoms: Sequence[str] | None = None,
    ) -> list[DiseaseCandidateScore]:
        candidate_pool_limit = max(self.candidate_limit * 12, 50)
        rows = self.graph_service.get_disease_candidates_by_symptoms(
            symptom_names, limit=candidate_pool_limit
        )
        total_disease_count = max(self.graph_service.count_diseases(), 1)
        all_scored_symptoms = self._unique_names(list(symptom_names) + list(negated_symptoms or []))
        symptom_disease_counts = self.graph_service.get_symptom_disease_counts(all_scored_symptoms)
        user_weights = {
            symptom: self._symptom_weight(
                symptom,
                symptom_disease_counts,
                total_disease_count,
            )
            for symptom in symptom_names
        }
        total_user_weight = sum(user_weights.values()) or 1.0

        scored: list[DiseaseCandidateScore] = []
        for row in rows:
            matched_symptoms = self._unique_names(row.get("matched_symptoms") or [])
            disease_symptoms = self._unique_names(row.get("disease_symptoms") or [])
            symptom_coverage = len(matched_symptoms) / max(len(symptom_names), 1)
            weighted_symptom_coverage = (
                sum(user_weights.get(symptom, 0.0) for symptom in matched_symptoms)
                / total_user_weight
            )
            symptom_specificity = self._top_specificity(
                matched_symptoms, symptom_disease_counts, total_disease_count
            )
            evidence_count_score = min(len(matched_symptoms) / 3, 1.0)
            disease_symptom_coverage = len(matched_symptoms) / max(
                int(row.get("disease_symptom_count") or len(disease_symptoms) or 1),
                1,
            )
            disease_prior = self._disease_prior(str(row["disease"]))
            negated_penalty = self._negated_symptom_penalty(
                negated_symptoms or [],
                disease_symptoms,
                symptom_disease_counts,
                total_disease_count,
            )
            score_components = {
                "weighted_symptom_coverage": 0.38 * weighted_symptom_coverage,
                "symptom_specificity": 0.22 * symptom_specificity,
                "evidence_count_score": 0.16 * evidence_count_score,
                "disease_symptom_coverage": 0.09 * disease_symptom_coverage,
                "disease_prior": 0.15 * disease_prior,
                "negated_symptom_penalty": -0.08 * negated_penalty,
            }
            confidence = self._clamp01(sum(score_components.values()))
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
                    weighted_symptom_coverage=weighted_symptom_coverage,
                    disease_symptom_coverage=disease_symptom_coverage,
                    disease_prior=disease_prior,
                    negated_symptom_penalty=negated_penalty,
                    score_components=score_components,
                )
            )

        return sorted(
            scored,
            key=lambda item: (
                -item.confidence,
                -len(item.matched_symptoms),
                -item.disease_prior,
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

    @classmethod
    def _top_specificity(
        cls,
        symptoms: Sequence[str],
        symptom_disease_counts: dict[str, int],
        total_disease_count: int,
    ) -> float:
        scores = [
            cls._symptom_weight(symptom, symptom_disease_counts, total_disease_count)
            for symptom in symptoms
        ]
        if not scores:
            return 0.0
        top_scores = sorted(scores, reverse=True)[:3]
        return sum(top_scores) / len(top_scores)

    @staticmethod
    def _symptom_weight(
        symptom: str,
        symptom_disease_counts: dict[str, int],
        total_disease_count: int,
    ) -> float:
        max_idf = math.log(total_disease_count + 1)
        if max_idf <= 0:
            return 0.2
        disease_count = max(symptom_disease_counts.get(symptom, 0), 1)
        idf = math.log((total_disease_count + 1) / (disease_count + 1))
        specificity = max(0.0, min(idf / max_idf, 1.0))
        length_bonus = min(max(len(symptom) - 2, 0) / 20, 0.2)
        return max(0.15, min(specificity + length_bonus, 1.0))

    @staticmethod
    def _disease_prior(disease: str) -> float:
        if disease in _COMMON_DISEASE_PRIORS:
            return _COMMON_DISEASE_PRIORS[disease]
        prior = 0.55
        for keyword, value in _COMMON_DISEASE_KEYWORDS.items():
            if keyword in disease:
                prior = max(prior, value)
        if any(marker in disease for marker in _RARER_DISEASE_MARKERS):
            prior = min(prior, 0.42)
        return prior

    @classmethod
    def _negated_symptom_penalty(
        cls,
        negated_symptoms: Sequence[str],
        disease_symptoms: Sequence[str],
        symptom_disease_counts: dict[str, int],
        total_disease_count: int,
    ) -> float:
        if not negated_symptoms or not disease_symptoms:
            return 0.0
        penalty = 0.0
        for negated in negated_symptoms:
            if not any(cls._term_matches(negated, symptom) for symptom in disease_symptoms):
                continue
            penalty += cls._symptom_weight(
                negated,
                symptom_disease_counts,
                total_disease_count,
            )
        return min(penalty / max(len(disease_symptoms), 6), 0.45)

    @staticmethod
    def _term_matches(left: str, right: str) -> bool:
        left = str(left or "").strip()
        right = str(right or "").strip()
        if not left or not right:
            return False
        if left == right:
            return True
        if len(left) >= 3 and len(right) >= 3:
            return left in right or right in left
        return False

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(float(value), 1.0))

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
