import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Sequence

from app.services.entity_search import EntityCandidate
from app.services.llm_service import DashScopeService, OllamaService
from app.services.operation_log import log_operation


logger = logging.getLogger("medgraphqa.clinical_context")

_GENERIC_SYMPTOM_NAMES = {"痛", "疼", "疼痛", "不适", "难受", "不舒服"}
_NON_ENTITY_TERMS = {
    "早上",
    "上午",
    "中午",
    "下午",
    "晚上",
    "今天",
    "昨天",
    "有点",
    "轻微",
    "严重",
    "一整天",
    "一天",
    "两天",
    "持续",
    "加重",
    "缓解",
}
_NEGATION_PREFIXES = ("没有", "没", "不", "无", "否认", "未见", "暂时没有", "也没有")
_NEGATION_STOPWORDS = {
    "其他",
    "其它",
    "之类",
    "情况",
    "症状",
    "不适",
    "不良反应",
    "反应",
    "类似症状",
    "类似",
    "类似发作",
    "吃药",
    "用药",
    "服药",
    "舒服",
}
_COMPLEX_CLINICAL_MARKERS = (
    "但是",
    "不过",
    "然而",
    "后来",
    "之前",
    "现在",
    "同时",
    "伴随",
    "伴有",
    "除了",
    "反复",
    "时好时坏",
)
_DURATION_PATTERN = re.compile(
    r"(今天早上|早上|上午|中午|下午|晚上|昨天|今天|一整天|一天|两天|三天|"
    r"\d+\s*[天日周月年小时分钟]|半天|刚刚|刚才|最近|近期)"
)
_SEVERITY_WORDS = (
    "轻微",
    "有点",
    "稍微",
    "轻度",
    "明显",
    "比较明显",
    "严重",
    "剧烈",
)
_PROGRESSION_WORDS = (
    "没有加重",
    "没加重",
    "不加重",
    "加重",
    "缓解",
    "减轻",
    "持续",
    "反复",
    "越来越",
)
_MEDICATION_STATUS_PATTERNS = (
    r"(没有|没|未)(吃药|用药|服药)",
    r"(暂时)?还?(没有|没)(吃药|用药|服药)",
    r"(吃了|用了|服用).{0,8}(药|药物)",
)
_DIET_STATUS_PATTERNS = (
    r"饮食(正常|清淡|不正常)",
    r"(吃饭|胃口|食欲)(正常|还行|不好|变差)",
)
_SIMILAR_HISTORY_PATTERNS = (
    r"(近期|最近|之前|以前)?(没有|没|无).{0,4}(类似|相似)(症状|发作|情况)?",
    r"(近期|最近|之前|以前)?有.{0,4}(类似|相似)(症状|发作|情况)?",
)
_NEGATION_BOUNDARY_CHARS = set("，,。；;、 \n\t我也并且但又还暂目前现在")
_POSITIVE_SEGMENT_PREFIX = re.compile(
    r"^(我最近|我|最近|近期|早上起来|今天早上起来|今天|早上|还有|伴有|伴随|并且|同时|有点|主要是|出现)"
)
_POSITIVE_SEGMENT_SUFFIX = re.compile(
    r"(怎么办|怎么处理|怎么治疗|比较明显|很明显|有点|轻微|严重|持续.*|已经.*|了)$"
)


@dataclass
class SymptomMention:
    name: str
    body_part: str | None = None
    severity: str | None = None
    duration: str | None = None
    progression: str | None = None
    quality: str | None = None
    frequency: str | None = None


@dataclass
class ClinicalContext:
    symptoms: list[SymptomMention] = field(default_factory=list)
    negated_symptoms: list[str] = field(default_factory=list)
    red_flags: list[str] = field(default_factory=list)
    known_diseases: list[str] = field(default_factory=list)
    medications: list[str] = field(default_factory=list)
    medication_status: str | None = None
    diet_status: str | None = None
    similar_history: str | None = None
    allergies: list[str] = field(default_factory=list)
    pregnancy: bool | None = None
    user_goal: str | None = None
    missing_info: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class ClinicalContextService:
    def __init__(
        self,
        llm_service: DashScopeService | OllamaService | None,
        enabled: bool,
    ) -> None:
        self.llm_service = llm_service
        self.enabled = enabled
        self._stats = self._empty_stats()

    def reset_stats(self) -> None:
        self._stats = self._empty_stats()

    def stats_snapshot(self) -> dict[str, int]:
        return dict(self._stats)

    @staticmethod
    def _empty_stats() -> dict[str, int]:
        return {
            "extract_total": 0,
            "rules_only_count": 0,
            "llm_call_count": 0,
            "llm_success_count": 0,
            "llm_fallback_count": 0,
            "json_parse_error_count": 0,
            "extract_error_count": 0,
            "disabled_count": 0,
        }

    def _record_stat(self, key: str, amount: int = 1) -> None:
        self._stats[key] = self._stats.get(key, 0) + amount

    def extract(
        self,
        query: str,
        previous_context: dict | None,
        entities: Sequence[EntityCandidate],
        entity_hints: Sequence[str] | None = None,
    ) -> ClinicalContext:
        entity_hints = list(entity_hints or [])
        rule_context = self._rule_context_from_text(query, entity_hints, entities)
        self._record_stat("extract_total")
        if not self.enabled or not self.llm_service:
            self._record_stat("disabled_count")
            return self._merge_context(previous_context, rule_context)
        raw = ""
        with log_operation(
            logger,
            "clinical_context.extract",
            query_len=len(query),
            entity_hint_count=len(entity_hints),
        ) as result:
            result["rule_symptom_count"] = len(rule_context.symptoms)
            result["rule_negated_count"] = len(rule_context.negated_symptoms)
            if self._should_skip_llm(query, previous_context, entity_hints, rule_context):
                result["gate"] = "rules_only"
                self._record_stat("rules_only_count")
                return self._merge_context(previous_context, rule_context)
            try:
                prompt = self._build_prompt(query, previous_context, entities, entity_hints)
                result["prompt_len"] = len(prompt)
                result["gate"] = "llm"
                self._record_stat("llm_call_count")
                if hasattr(self.llm_service, "generate_json"):
                    raw = self.llm_service.generate_json(prompt)
                else:
                    raw = self.llm_service.generate(prompt)
                result["raw_len"] = len(raw)
                parsed_context = self._parse_context(raw)
                self._drop_negations_conflicting_with_rule_positives(
                    parsed_context,
                    rule_context,
                )
                context = self._merge_context(rule_context.to_dict(), parsed_context)
                result["symptom_count"] = len(context.symptoms)
                self._record_stat("llm_success_count")
                return self._merge_context(previous_context, context)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "operation=clinical_context.parse_json status=error error=%s raw=%r",
                    exc,
                    raw[:1000],
                )
                result["fallback"] = "json_parse_error"
                self._record_stat("llm_fallback_count")
                self._record_stat("json_parse_error_count")
                fallback = self._merge_context(
                    previous_context,
                    rule_context,
                )
                return self._merge_context(
                    fallback.to_dict(),
                    self._fallback_context_from_text(query, entity_hints, entities),
                )
            except Exception:
                logger.exception("operation=clinical_context.extract_inner status=error")
                result["fallback"] = "extract_error"
                self._record_stat("llm_fallback_count")
                self._record_stat("extract_error_count")
                fallback = self._merge_context(
                    previous_context,
                    rule_context,
                )
                return self._merge_context(
                    fallback.to_dict(),
                    self._fallback_context_from_text(query, entity_hints, entities),
                )

    def symptom_terms(self, context: ClinicalContext) -> list[str]:
        result: list[str] = []
        for symptom in context.symptoms:
            terms = []
            if symptom.body_part and symptom.name:
                terms.append(f"{symptom.body_part}{symptom.name}")
            if symptom.body_part and symptom.quality:
                terms.append(f"{symptom.body_part}{symptom.quality}")
            if symptom.name and symptom.name not in _GENERIC_SYMPTOM_NAMES:
                terms.append(symptom.name)
            if (
                symptom.quality
                and symptom.quality not in _GENERIC_SYMPTOM_NAMES
                and symptom.quality not in _NON_ENTITY_TERMS
            ):
                terms.append(symptom.quality)
            for term in terms:
                if term and term not in _NON_ENTITY_TERMS and term not in result:
                    result.append(term)
        return result[:8]

    @staticmethod
    def _build_prompt(
        query: str,
        previous_context: dict | None,
        entities: Sequence[EntityCandidate],
        entity_hints: Sequence[str] | None = None,
    ) -> str:
        previous = json.dumps(previous_context or {}, ensure_ascii=False)
        candidate_hints = [
            f"{item.entity_type}:{item.canonical_name}"
            for item in entities[:8]
            if item.entity_type in {"疾病症状", "疾病"}
        ]
        plain_hints = [item for item in (entity_hints or []) if item]
        all_hints = plain_hints + candidate_hints
        hints = "、".join(dict.fromkeys(all_hints)) or "无"
        return (
            "你是医疗对话结构化抽取器。只抽取用户明确说出的事实，不诊断，不补全。\n"
            "只输出一个紧凑 JSON 对象，不要 Markdown，不要解释，不要换行缩进。\n"
            "字段：symptoms, negated_symptoms, red_flags, known_diseases, medications, medication_status, diet_status, similar_history, allergies, pregnancy, user_goal, missing_info。\n"
            "symptoms 元素字段：name, body_part, severity, duration, progression, quality, frequency。\n"
            "negated_symptoms 记录用户明确否认的症状，例如“不发热”输出[\"发热\"]；否认的症状不要放入 symptoms。"
            "medication_status 记录是否用药，例如“没有吃药”。diet_status 记录饮食情况，例如“饮食正常”。"
            "similar_history 记录近期是否有类似发作，例如“近期没有类似症状”。\n"
            "示例：上腹部轻微疼痛一天 -> "
            '{"symptoms":[{"name":"疼痛","body_part":"上腹部","severity":"轻微","duration":"一天","progression":null,"quality":"疼痛","frequency":null}],'
            '"negated_symptoms":[],"red_flags":[],"known_diseases":[],"medications":[],"medication_status":null,"diet_status":null,"similar_history":null,"allergies":[],"pregnancy":null,"user_goal":null,"missing_info":[]}\n'
            "示例：不发热，没有其他不良反应，没有吃药，饮食正常 -> "
            '{"symptoms":[],"negated_symptoms":["发热","其他不良反应"],"red_flags":[],"known_diseases":[],"medications":[],"medication_status":"没有吃药","diet_status":"饮食正常","similar_history":null,"allergies":[],"pregnancy":null,"user_goal":null,"missing_info":[]}\n'
            "示例：头痛，没有流鼻涕，也没有浑身忽冷忽热 -> "
            '{"symptoms":[{"name":"头痛","body_part":null,"severity":null,"duration":null,"progression":null,"quality":null,"frequency":null}],"negated_symptoms":["流鼻涕","浑身忽冷忽热"],"red_flags":[],"known_diseases":[],"medications":[],"medication_status":null,"diet_status":null,"similar_history":null,"allergies":[],"pregnancy":null,"user_goal":null,"missing_info":[]}\n'
            f"上一轮已知上下文：{previous}\n"
            f"实体提示：{hints}\n"
            f"用户本轮输入：{query}"
        )

    def _parse_context(self, raw: str) -> ClinicalContext:
        data = self._extract_json(raw)
        negated_symptoms = self._string_list(data.get("negated_symptoms"))
        symptoms = []
        for item in data.get("symptoms") or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            if self._is_negated_term(name, negated_symptoms):
                continue
            symptoms.append(
                SymptomMention(
                    name=name,
                    body_part=self._clean_optional(item.get("body_part")),
                    severity=self._clean_optional(item.get("severity")),
                    duration=self._clean_optional(item.get("duration")),
                    progression=self._clean_optional(item.get("progression")),
                    quality=self._clean_optional(item.get("quality")),
                    frequency=self._clean_optional(item.get("frequency")),
                )
            )
        return ClinicalContext(
            symptoms=symptoms,
            negated_symptoms=negated_symptoms,
            red_flags=self._string_list(data.get("red_flags")),
            known_diseases=self._string_list(data.get("known_diseases")),
            medications=self._string_list(data.get("medications")),
            medication_status=self._clean_optional(data.get("medication_status")),
            diet_status=self._clean_optional(data.get("diet_status")),
            similar_history=self._clean_optional(data.get("similar_history")),
            allergies=self._string_list(data.get("allergies")),
            pregnancy=data.get("pregnancy") if isinstance(data.get("pregnancy"), bool) else None,
            user_goal=self._clean_optional(data.get("user_goal")),
            missing_info=self._string_list(data.get("missing_info")),
        )

    @staticmethod
    def _extract_json(raw: str) -> dict:
        text = raw.strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end >= start:
            text = text[start : end + 1]
        data = ClinicalContextService._loads_json_with_repair(text)
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _loads_json_with_repair(text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            repaired = text
            repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
            repaired = re.sub(
                r'((?::)\s*(?:"[^"\\]*(?:\\.[^"\\]*)*"|null|true|false|-?\d+(?:\.\d+)?))\s+("[-\w\u4e00-\u9fff]+"\s*:)',
                r"\1, \2",
                repaired,
            )
            if repaired != text:
                return json.loads(repaired)
            raise

    def _merge_context(
        self,
        previous_context: dict | None,
        current: ClinicalContext,
    ) -> ClinicalContext:
        previous = self._context_from_dict(previous_context or {})
        symptoms = self._merge_symptoms(previous.symptoms, current.symptoms)
        negated_symptoms = self._merge_list(previous.negated_symptoms, current.negated_symptoms)
        symptoms = [
            item for item in symptoms if not self._is_negated_term(item.name, negated_symptoms)
        ]
        return ClinicalContext(
            symptoms=symptoms,
            negated_symptoms=negated_symptoms,
            red_flags=self._merge_list(previous.red_flags, current.red_flags),
            known_diseases=self._merge_list(previous.known_diseases, current.known_diseases),
            medications=self._merge_list(previous.medications, current.medications),
            medication_status=current.medication_status or previous.medication_status,
            diet_status=current.diet_status or previous.diet_status,
            similar_history=current.similar_history or previous.similar_history,
            allergies=self._merge_list(previous.allergies, current.allergies),
            pregnancy=current.pregnancy if current.pregnancy is not None else previous.pregnancy,
            user_goal=current.user_goal or previous.user_goal,
            missing_info=current.missing_info or previous.missing_info,
        )

    def _drop_negations_conflicting_with_rule_positives(
        self,
        parsed_context: ClinicalContext,
        rule_context: ClinicalContext,
    ) -> None:
        if not parsed_context.negated_symptoms or not rule_context.symptoms:
            return
        positives = [item.name for item in rule_context.symptoms if item.name]
        parsed_context.negated_symptoms = [
            item
            for item in parsed_context.negated_symptoms
            if not any(self._same_text(item, positive) for positive in positives)
        ]

    def _context_from_dict(self, data: dict) -> ClinicalContext:
        negated_symptoms = self._string_list(data.get("negated_symptoms"))
        symptoms = []
        for item in data.get("symptoms") or []:
            if (
                isinstance(item, dict)
                and item.get("name")
                and not self._is_negated_term(str(item.get("name")), negated_symptoms)
            ):
                symptoms.append(
                    SymptomMention(
                        name=str(item.get("name")),
                        body_part=self._clean_optional(item.get("body_part")),
                        severity=self._clean_optional(item.get("severity")),
                        duration=self._clean_optional(item.get("duration")),
                        progression=self._clean_optional(item.get("progression")),
                        quality=self._clean_optional(item.get("quality")),
                        frequency=self._clean_optional(item.get("frequency")),
                    )
                )
        return ClinicalContext(
            symptoms=symptoms,
            negated_symptoms=negated_symptoms,
            red_flags=self._string_list(data.get("red_flags")),
            known_diseases=self._string_list(data.get("known_diseases")),
            medications=self._string_list(data.get("medications")),
            medication_status=self._clean_optional(data.get("medication_status")),
            diet_status=self._clean_optional(data.get("diet_status")),
            similar_history=self._clean_optional(data.get("similar_history")),
            allergies=self._string_list(data.get("allergies")),
            pregnancy=data.get("pregnancy") if isinstance(data.get("pregnancy"), bool) else None,
            user_goal=self._clean_optional(data.get("user_goal")),
            missing_info=self._string_list(data.get("missing_info")),
        )

    @staticmethod
    def _context_from_entities(entities: Sequence[EntityCandidate]) -> ClinicalContext:
        symptoms = []
        for item in entities:
            if item.entity_type != "疾病症状":
                continue
            if any(existing.name == item.canonical_name for existing in symptoms):
                continue
            symptoms.append(SymptomMention(name=item.canonical_name))
        return ClinicalContext(symptoms=symptoms)

    @staticmethod
    def _merge_symptoms(
        previous: Sequence[SymptomMention],
        current: Sequence[SymptomMention],
    ) -> list[SymptomMention]:
        result: dict[str, SymptomMention] = {item.name: item for item in previous}
        for item in current:
            old = result.get(item.name)
            if not old:
                result[item.name] = item
                continue
            result[item.name] = SymptomMention(
                name=item.name,
                body_part=item.body_part or old.body_part,
                severity=item.severity or old.severity,
                duration=item.duration or old.duration,
                progression=item.progression or old.progression,
                quality=item.quality or old.quality,
                frequency=item.frequency or old.frequency,
            )
        return list(result.values())

    @staticmethod
    def _merge_list(previous: Sequence[str], current: Sequence[str]) -> list[str]:
        result: list[str] = []
        for item in list(previous) + list(current):
            if item and item not in result:
                result.append(item)
        return result

    def _rule_context_from_text(
        self,
        query: str,
        entity_hints: Sequence[str] | None,
        entities: Sequence[EntityCandidate] | None = None,
    ) -> ClinicalContext:
        hints = self._clean_entity_hints(entity_hints or [])
        negated = self._extract_negated_terms_from_text(query, hints)
        duration = self._first_match(query, _DURATION_PATTERN)
        severity = self._first_contained(query, _SEVERITY_WORDS)
        progression = self._first_contained(query, _PROGRESSION_WORDS)
        symptoms: list[SymptomMention] = []
        entity_context = self._context_from_entities(entities or [])
        for item in entity_context.symptoms:
            if not self._is_negated_term(item.name, negated):
                self._append_symptom_prefer_specific(symptoms, item)
        for hint in hints:
            if self._is_negated_term(hint, negated):
                continue
            self._append_symptom_prefer_specific(
                symptoms,
                SymptomMention(
                    name=hint,
                    severity=severity,
                    duration=duration,
                    progression=progression,
                ),
            )
        for phrase in self._extract_positive_phrases_from_text(query, negated):
            self._append_symptom_prefer_specific(
                symptoms,
                SymptomMention(
                    name=phrase,
                    severity=severity,
                    duration=duration,
                    progression=progression,
                ),
            )
        return ClinicalContext(
            symptoms=symptoms,
            negated_symptoms=negated,
            medication_status=self._first_regex_text(query, _MEDICATION_STATUS_PATTERNS),
            diet_status=self._first_regex_text(query, _DIET_STATUS_PATTERNS),
            similar_history=self._first_regex_text(query, _SIMILAR_HISTORY_PATTERNS),
            user_goal=self._extract_user_goal(query),
        )

    @classmethod
    def _fallback_context_from_text(
        cls,
        query: str,
        entity_hints: Sequence[str] | None,
        entities: Sequence[EntityCandidate] | None = None,
    ) -> ClinicalContext:
        service = cls(llm_service=None, enabled=False)
        return service._rule_context_from_text(query, entity_hints, entities)

    def _should_skip_llm(
        self,
        query: str,
        previous_context: dict | None,
        entity_hints: Sequence[str],
        rule_context: ClinicalContext,
    ) -> bool:
        has_rule_signal = any(
            [
                rule_context.symptoms,
                rule_context.negated_symptoms,
                rule_context.medication_status,
                rule_context.diet_status,
                rule_context.similar_history,
                rule_context.user_goal,
            ]
        )
        if not has_rule_signal:
            return False
        if any(marker in query for marker in _COMPLEX_CLINICAL_MARKERS):
            return False
        if len(query) > 28:
            return False
        clause_count = len([item for item in re.split(r"[，,。；;、\s]+", query) if item])
        if clause_count > 3:
            return False
        if previous_context and len(query) > 18 and entity_hints:
            return False
        return True

    @staticmethod
    def _clean_entity_hints(entity_hints: Sequence[str]) -> list[str]:
        result: list[str] = []
        for item in entity_hints:
            text = str(item or "").strip()
            if len(text) < 2 or text in _NON_ENTITY_TERMS:
                continue
            if text not in result:
                result.append(text)
        return result[:8]

    @classmethod
    def _append_symptom_prefer_specific(
        cls,
        symptoms: list[SymptomMention],
        item: SymptomMention,
    ) -> None:
        name = str(item.name or "").strip()
        if not name:
            return
        for index, existing in enumerate(list(symptoms)):
            existing_name = str(existing.name or "").strip()
            if not existing_name:
                continue
            if existing_name == name:
                symptoms[index] = cls._merge_symptom_detail(existing, item)
                return
            if cls._same_text(existing_name, name):
                if len(name) > len(existing_name):
                    symptoms[index] = cls._merge_symptom_detail(
                        SymptomMention(name=name),
                        existing,
                    )
                    symptoms[index] = cls._merge_symptom_detail(symptoms[index], item)
                return
        symptoms.append(item)

    @staticmethod
    def _merge_symptom_detail(
        base: SymptomMention,
        extra: SymptomMention,
    ) -> SymptomMention:
        return SymptomMention(
            name=base.name,
            body_part=base.body_part or extra.body_part,
            severity=base.severity or extra.severity,
            duration=base.duration or extra.duration,
            progression=base.progression or extra.progression,
            quality=base.quality or extra.quality,
            frequency=base.frequency or extra.frequency,
        )

    @staticmethod
    def _first_match(query: str, pattern: re.Pattern) -> str | None:
        match = pattern.search(query)
        return match.group(0).strip() if match else None

    @staticmethod
    def _first_contained(query: str, words: Sequence[str]) -> str | None:
        return next((word for word in words if word in query), None)

    @staticmethod
    def _first_regex_text(query: str, patterns: Sequence[str]) -> str | None:
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                return match.group(0).strip()
        return None

    @staticmethod
    def _extract_user_goal(query: str) -> str | None:
        if any(word in query for word in ("怎么办", "怎么治", "治疗", "吃什么药")):
            return "治疗建议"
        if any(word in query for word in ("检查", "查什么")):
            return "检查建议"
        return None

    @classmethod
    def _extract_positive_phrases_from_text(
        cls,
        query: str,
        negated: Sequence[str],
    ) -> list[str]:
        result: list[str] = []
        for segment in re.split(r"[，,。；;、]", query):
            segment = segment.strip()
            if not segment:
                continue
            if cls._segment_has_negation(segment):
                continue
            segment = _POSITIVE_SEGMENT_PREFIX.sub("", segment).strip()
            segment = _DURATION_PATTERN.sub("", segment)
            segment = _POSITIVE_SEGMENT_PREFIX.sub("", segment).strip()
            segment = _POSITIVE_SEGMENT_SUFFIX.sub("", segment).strip()
            for word in _SEVERITY_WORDS + _PROGRESSION_WORDS:
                segment = segment.replace(word, "")
            segment = segment.strip()
            if not segment:
                continue
            for phrase in re.split(r"(?:还有|以及|和)", segment):
                phrase = phrase.strip()
                if cls._valid_positive_phrase(phrase) and not cls._is_negated_term(phrase, negated):
                    cls._append_unique(result, phrase)
        return result[:8]

    @staticmethod
    def _segment_has_negation(segment: str) -> bool:
        for match in re.finditer(r"(没有|没|不|无|否认|未见|暂时没有|也没有)", segment):
            marker = match.group(1)
            before = segment[match.start() - 1] if match.start() > 0 else ""
            if marker in {"不", "没", "无"} and before and before not in _NEGATION_BOUNDARY_CHARS:
                continue
            return True
        return False

    @staticmethod
    def _valid_positive_phrase(value: str) -> bool:
        if len(value) < 2 or len(value) > 18:
            return False
        if value in _NON_ENTITY_TERMS or value in _GENERIC_SYMPTOM_NAMES:
            return False
        if any(marker in value for marker in ("怎么办", "怎么治", "是否", "有没有")):
            return False
        return True

    @classmethod
    def _extract_negated_terms_from_text(
        cls,
        query: str,
        entity_hints: Sequence[str],
    ) -> list[str]:
        result: list[str] = []
        hints = sorted(
            {item.strip() for item in entity_hints if item and len(item.strip()) >= 2},
            key=len,
            reverse=True,
        )
        for hint in hints:
            for match in re.finditer(re.escape(hint), query):
                prefix = query[max(0, match.start() - 8) : match.start()]
                separator_positions = [prefix.rfind(item) for item in "，,。；;、\n\t "]
                last_separator = max(separator_positions)
                if last_separator >= 0:
                    prefix = prefix[last_separator + 1 :]
                if any(marker in prefix for marker in _NEGATION_PREFIXES):
                    cls._append_unique(result, hint)
        for match in re.finditer(
            r"(没有|没|不|无|否认|未见|暂时没有|也没有)([^，,。；;、\s]{1,16})",
            query,
        ):
            marker = match.group(1)
            before = query[match.start() - 1] if match.start() > 0 else ""
            if marker in {"不", "没", "无"} and before and before not in _NEGATION_BOUNDARY_CHARS:
                continue
            phrase = match.group(2).strip()
            phrase = re.sub(r"^(明显|任何|其他|其它|出现|伴随|伴有)", "", phrase).strip()
            phrase = re.sub(r"(之类的?|等|情况|症状)$", "", phrase).strip()
            if cls._valid_negated_phrase(phrase):
                cls._append_unique(result, phrase)
        return result[:8]

    @staticmethod
    def _valid_negated_phrase(value: str) -> bool:
        if len(value) < 2:
            return False
        if value in _NEGATION_STOPWORDS or value in _NON_ENTITY_TERMS:
            return False
        return True

    @staticmethod
    def _append_unique(items: list[str], value: str) -> None:
        if value and value not in items:
            items.append(value)

    @staticmethod
    def _is_negated_term(term: str, negated: Sequence[str]) -> bool:
        term = str(term or "").strip()
        if not term:
            return False
        return any(
            item
            and (
                item == term
                or (
                    len(item) >= 3
                    and len(term) >= 3
                    and (
                        item in term
                        or term in item
                    )
                )
            )
            for item in negated
        )

    @staticmethod
    def _same_text(left: str, right: str) -> bool:
        left = str(left or "").strip()
        right = str(right or "").strip()
        return bool(left and right and (left == right or left in right or right in left))

    @staticmethod
    def _string_list(value) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    @staticmethod
    def _clean_optional(value) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None
