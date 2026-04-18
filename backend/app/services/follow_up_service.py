import json
import logging
import re
from collections import Counter
from typing import Any, Sequence

from app.services.disease_resolution import DiseaseCandidateScore
from app.services.follow_up_planner import FollowUpQuestionPlanner, FollowUpQuestionSlot
from app.services.llm_service import DashScopeService, OllamaService
from app.services.operation_log import log_operation


logger = logging.getLogger("medgraphqa.follow_up")


class FollowUpQuestionService:
    def __init__(self, llm_service: DashScopeService | OllamaService) -> None:
        self.llm_service = llm_service
        self.planner = FollowUpQuestionPlanner()

    def build(self, state: dict[str, Any]) -> str | None:
        resolution = state.get("disease_resolution")
        candidates: Sequence[DiseaseCandidateScore] = (
            getattr(resolution, "candidates", []) if resolution else []
        )
        if not candidates:
            return None

        context = state.get("clinical_context") or {}
        known = self._known_context(state, context)
        profiles = self._candidate_profiles(candidates, known)
        slots = self.planner.question_slots(candidates, known)
        if not profiles or not slots:
            return None

        fallback_answer = self.planner.deterministic_answer(slots)
        prompt = self._build_prompt(state, context, known, profiles, slots)
        with log_operation(
            logger,
            "follow_up.generate",
            candidate_count=len(profiles),
            selected_question_count=len(slots),
            selected_questions="|".join(slot.question for slot in slots),
            prompt_len=len(prompt),
        ) as result:
            try:
                answer = self.llm_service.generate(prompt)
            except Exception:
                logger.exception("operation=follow_up.generate_inner status=error")
                result["fallback"] = "llm_error"
                result["answer_len"] = len(fallback_answer)
                return fallback_answer
            answer = self._clean_answer(answer)
            result["answer_len"] = len(answer)
            if not answer:
                result["fallback"] = "llm_empty_output"
                result["answer_len"] = len(fallback_answer)
                return fallback_answer
            return answer

    def _build_prompt(
        self,
        state: dict[str, Any],
        context: dict[str, Any],
        known: dict[str, list[str] | str | None],
        profiles: list[dict[str, Any]],
        slots: list[FollowUpQuestionSlot],
    ) -> str:
        payload = {
            "user_query": state.get("effective_query") or state.get("query"),
            "clinical_context": context,
            "known_positive": known["positive"],
            "known_negative": known["negative"],
            "medication_status": known["medication_status"],
            "diet_status": known["diet_status"],
            "similar_history": known["similar_history"],
            "candidate_diseases": profiles,
            "selected_questions": [slot.to_prompt_dict() for slot in slots],
            "knowledge_evidence": state.get("evidence", [])[:3],
        }
        return (
            "你是医疗问答助手的追问生成器。你不能诊断，也不要给治疗建议。\n"
            "任务：根据系统已经选出的 selected_questions，生成能区分候选疾病的追问。\n"
            "要求：\n"
            "1. 必须围绕 selected_questions 追问，可以合并表达，但不要新增无关问题。\n"
            "2. 不要重复询问用户已经说明或否认的信息。\n"
            "3. 不要照搬固定模板，不要一次问太多，最多提出3个问题。\n"
            "4. 语气自然，先说明“目前还不能确定具体疾病”，再追问关键信息。\n"
            "5. 如当前症状可能需要及时就医，只保留一句简短风险提醒。\n"
            "6. 只输出给用户看的中文回答，不要输出JSON、Markdown或候选疾病内部评分。\n\n"
            "输入数据：\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )

    def _candidate_profiles(
        self,
        candidates: Sequence[DiseaseCandidateScore],
        known: dict[str, list[str] | str | None],
    ) -> list[dict[str, Any]]:
        top = list(candidates[:5])
        known_positive = set(known["positive"] or [])
        known_negative = set(known["negative"] or [])
        all_symptoms = [
            symptom
            for candidate in top
            for symptom in candidate.disease_symptoms
            if self.planner.usable_symptom(symptom, known_positive, known_negative)
        ]
        counts = Counter(all_symptoms)
        profile_list: list[dict[str, Any]] = []
        for candidate in top:
            discriminative = [
                symptom
                for symptom in candidate.disease_symptoms
                if self.planner.usable_symptom(symptom, known_positive, known_negative)
                and counts.get(symptom, 0) < len(top)
            ]
            discriminative = sorted(
                dict.fromkeys(discriminative),
                key=lambda item: (counts.get(item, 0), len(item)),
            )[:8]
            common_missing = [
                symptom
                for symptom in candidate.disease_symptoms
                if self.planner.usable_symptom(symptom, known_positive, known_negative)
            ][:8]
            profile_list.append(
                {
                    "disease": candidate.disease,
                    "confidence": round(float(candidate.confidence), 4),
                    "matched_symptoms": candidate.matched_symptoms,
                    "distinguishing_symptoms": discriminative or common_missing,
                }
            )
        return profile_list

    def _known_context(
        self,
        state: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, list[str] | str | None]:
        positive: list[str] = []
        for item in state.get("entities", []):
            if getattr(item, "entity_type", "") == "疾病症状":
                self._append_unique(positive, getattr(item, "canonical_name", ""))
        for item in context.get("symptoms") or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            body_part = str(item.get("body_part") or "").strip()
            quality = str(item.get("quality") or "").strip()
            self._append_unique(positive, name)
            if body_part and name:
                self._append_unique(positive, f"{body_part}{name}")
            if body_part and quality:
                self._append_unique(positive, f"{body_part}{quality}")

        negative = [
            str(item).strip()
            for item in context.get("negated_symptoms", [])
            if str(item).strip()
        ]
        return {
            "positive": positive,
            "negative": negative,
            "medication_status": context.get("medication_status"),
            "diet_status": context.get("diet_status"),
            "similar_history": context.get("similar_history"),
        }

    @staticmethod
    def _append_unique(items: list[str], value: str) -> None:
        text = str(value or "").strip()
        if text and text not in items:
            items.append(text)

    @staticmethod
    def _clean_answer(answer: str) -> str:
        text = str(answer or "").strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        text = re.sub(r"^```(?:\w+)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
        return text
