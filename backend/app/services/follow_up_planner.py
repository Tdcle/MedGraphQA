from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Sequence

from app.services.disease_resolution import DiseaseCandidateScore


_GENERIC_SYMPTOMS = {
    "疼",
    "痛",
    "疼痛",
    "不适",
    "难受",
    "不舒服",
    "症状",
    "体征",
}

_TERM_ALIASES = {
    "喉咙痛": "咽痛",
    "咽喉痛": "咽痛",
    "咽部疼痛": "咽痛",
    "咽痛": "咽痛",
    "发烧": "发热",
    "发热": "发热",
    "低热": "发热",
    "高热": "发热",
    "流涕": "流鼻涕",
    "流鼻涕": "流鼻涕",
    "清水鼻涕": "流鼻涕",
    "鼻涕": "流鼻涕",
    "喷嚏": "打喷嚏",
    "打喷嚏": "打喷嚏",
    "气短": "呼吸困难",
    "喘": "呼吸困难",
    "呼吸困难": "呼吸困难",
    "咳血": "咯血",
    "咯血": "咯血",
    "拉肚子": "腹泻",
    "腹泻": "腹泻",
}


@dataclass
class FollowUpQuestionSlot:
    question: str
    score: float
    symptoms: list[str] = field(default_factory=list)
    diseases: list[str] = field(default_factory=list)

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "score": round(float(self.score), 4),
            "source_symptoms": self.symptoms[:8],
            "related_candidate_diseases": self.diseases[:5],
        }


class FollowUpQuestionPlanner:
    def question_slots(
        self,
        candidates: Sequence[DiseaseCandidateScore],
        known: dict[str, list[str] | str | None],
    ) -> list[FollowUpQuestionSlot]:
        top = list(candidates[:5])
        if not top:
            return []

        known_positive = set(known["positive"] or [])
        known_negative = set(known["negative"] or [])
        symptom_counts: Counter[str] = Counter()
        symptom_mass: Counter[str] = Counter()
        symptom_diseases: dict[str, list[str]] = {}
        total_confidence = sum(max(float(item.confidence), 0.0) for item in top) or 1.0

        for candidate in top:
            usable_symptoms = [
                symptom
                for symptom in dict.fromkeys(candidate.disease_symptoms)
                if self.usable_symptom(symptom, known_positive, known_negative)
            ]
            for symptom in usable_symptoms:
                symptom_counts[symptom] += 1
                symptom_mass[symptom] += max(float(candidate.confidence), 0.0)
                self._append_unique(
                    symptom_diseases.setdefault(symptom, []),
                    candidate.disease,
                )

        grouped: dict[str, FollowUpQuestionSlot] = {}
        for symptom, count in symptom_counts.items():
            question = self.question_for_symptom(symptom, known_positive, known_negative)
            if not question:
                continue
            frequency = count / max(len(top), 1)
            rarity = 1.0 - ((count - 1) / max(len(top) - 1, 1))
            balance = 1.0 - abs(frequency - 0.5) * 2
            discriminative = max(rarity, balance, 0.0)
            mass = symptom_mass[symptom] / total_confidence
            specificity = min(max(len(symptom) - 1, 1) / 12, 1.0)
            top_bonus = 0.12 if symptom in top[0].disease_symptoms else 0.0
            score = (
                0.42 * mass
                + 0.32 * discriminative
                + 0.14 * specificity
                + top_bonus
            )

            slot = grouped.get(question)
            if not slot:
                grouped[question] = FollowUpQuestionSlot(
                    question=question,
                    score=score,
                    symptoms=[symptom],
                    diseases=list(symptom_diseases.get(symptom, [])),
                )
                continue
            slot.score = max(slot.score, score)
            self._append_unique(slot.symptoms, symptom)
            for disease in symptom_diseases.get(symptom, []):
                self._append_unique(slot.diseases, disease)

        return sorted(
            grouped.values(),
            key=lambda item: (-item.score, item.question),
        )[:3]

    @staticmethod
    def deterministic_answer(slots: list[FollowUpQuestionSlot]) -> str:
        questions = [slot.question.rstrip("。？?") for slot in slots[:3]]
        if not questions:
            return "目前还不能确定具体疾病，请补充症状变化、伴随不适、诱因或既往病史。"
        question_text = "；".join(
            f"{index}. {question}" for index, question in enumerate(questions, start=1)
        )
        return (
            "目前还不能确定具体疾病。为了区分几种可能情况，请补充："
            f"{question_text}。"
            "如果出现高热不退、呼吸困难、胸痛、意识异常或出血等情况，请及时就医。"
        )

    @classmethod
    def usable_symptom(
        cls,
        symptom: str,
        known_positive: set[str],
        known_negative: set[str],
    ) -> bool:
        text = str(symptom or "").strip()
        if not text or text in _GENERIC_SYMPTOMS:
            return False
        if any(cls._terms_overlap(text, known) for known in known_positive if known):
            return False
        if any(cls._terms_overlap(text, known) for known in known_negative if known):
            return False
        return True

    def question_for_symptom(
        self,
        symptom: str,
        known_positive: set[str],
        known_negative: set[str],
    ) -> str | None:
        text = str(symptom or "").strip()
        if not self.usable_symptom(text, known_positive, known_negative):
            return None

        groups = [
            (
                ("发热", "发烧", "低热", "高热", "寒战", "畏寒"),
                [
                    ("发热或体温升高", ("发热", "发烧", "低热", "高热")),
                    ("畏寒或寒战", ("畏寒", "寒战")),
                ],
                "是否还有{items}，体温大概多少、是否持续不退",
            ),
            (
                ("鼻", "喷嚏", "流涕", "流鼻涕", "鼻塞", "鼻痒"),
                [
                    ("清水样或黄绿色鼻涕", ("流鼻涕", "流涕", "鼻涕", "脓涕")),
                    ("鼻痒", ("鼻痒",)),
                    ("频繁打喷嚏", ("喷嚏", "打喷嚏")),
                    ("鼻塞加重", ("鼻塞",)),
                ],
                "是否还有{items}，这些情况是否持续或加重",
            ),
            (
                ("咽", "喉", "吞咽", "声音嘶哑", "异物感"),
                [
                    ("咽喉干痛", ("咽干", "喉咙干", "咽喉干燥", "咽喉痛")),
                    ("吞咽痛", ("吞咽痛", "吞咽困难")),
                    ("异物感", ("异物感",)),
                    ("声音嘶哑", ("声音嘶哑", "嘶哑")),
                ],
                "是否还有{items}",
            ),
            (
                ("咳", "痰", "胸痛", "呼吸困难", "气短", "喘", "咯血"),
                [
                    ("咳嗽", ("咳嗽",)),
                    ("咳痰及痰色变化", ("咳痰", "痰", "黄痰", "白痰")),
                    ("胸痛", ("胸痛",)),
                    ("呼吸困难或气短", ("呼吸困难", "气短", "喘")),
                    ("咯血", ("咯血", "咳血")),
                ],
                "是否还有{items}",
            ),
            (
                ("腹", "胃", "恶心", "呕吐", "腹泻", "便", "黑便", "食欲"),
                [
                    ("腹部不适的具体部位和性质", ("腹痛", "腹部疼痛", "腹部不适")),
                    ("恶心或呕吐", ("恶心", "呕吐")),
                    ("腹泻", ("腹泻", "拉肚子")),
                    ("黑便或便血", ("黑便", "便血")),
                    ("食欲明显下降", ("食欲", "食欲不振", "食欲减退")),
                ],
                "是否还有{items}",
            ),
            (
                ("尿", "排尿", "水肿", "泡沫尿", "少尿", "无尿"),
                [
                    ("尿痛或尿频", ("尿痛", "尿频")),
                    ("尿量变化", ("少尿", "无尿", "尿量")),
                    ("尿色异常或泡沫尿", ("尿色", "泡沫尿", "尿泡沫")),
                    ("水肿加重", ("水肿",)),
                ],
                "是否还有{items}",
            ),
            (
                ("皮疹", "瘙痒", "红肿", "水疱", "脓肿", "溃疡"),
                [
                    ("皮疹", ("皮疹",)),
                    ("瘙痒", ("瘙痒",)),
                    ("局部红肿", ("红肿",)),
                    ("水疱", ("水疱",)),
                    ("脓肿或溃疡变化", ("脓肿", "溃疡")),
                ],
                "是否还有{items}",
            ),
            (
                ("头痛", "头晕", "意识", "乏力", "麻木", "抽搐"),
                [
                    ("头痛或头晕", ("头痛", "头晕")),
                    ("明显乏力", ("乏力", "无力")),
                    ("麻木或抽搐", ("麻木", "抽搐")),
                    ("意识异常", ("意识", "意识障碍", "意识异常")),
                ],
                "是否还有{items}",
            ),
        ]
        for keywords, options, template in groups:
            if any(keyword in text for keyword in keywords):
                option_labels = self._unknown_option_labels(
                    options,
                    known_positive,
                    known_negative,
                )
                if not option_labels:
                    return None
                return template.format(items="、".join(option_labels[:3]))
        return f"是否还有{text}，它出现多久了、是否持续加重"

    @classmethod
    def _unknown_option_labels(
        cls,
        options: list[tuple[str, tuple[str, ...]]],
        known_positive: set[str],
        known_negative: set[str],
    ) -> list[str]:
        known = known_positive | known_negative
        labels: list[str] = []
        for label, keywords in options:
            if any(
                cls._terms_overlap(keyword, known_item)
                for keyword in keywords
                for known_item in known
            ):
                continue
            labels.append(label)
        return labels

    @staticmethod
    def _terms_overlap(left: str, right: str) -> bool:
        left = str(left or "").strip()
        right = str(right or "").strip()
        if not left or not right:
            return False
        left_normalized = _TERM_ALIASES.get(left, left)
        right_normalized = _TERM_ALIASES.get(right, right)
        if left_normalized == right_normalized:
            return True
        if left == right:
            return True
        if len(left) >= 2 and len(right) >= 2:
            return left in right or right in left
        return False

    @staticmethod
    def _append_unique(items: list[str], value: str) -> None:
        text = str(value or "").strip()
        if text and text not in items and text not in _GENERIC_SYMPTOMS:
            items.append(text)
