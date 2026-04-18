from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class SafetyRuleHit:
    code: str
    severity: str
    message: str
    matches: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "matches": self.matches[:10],
        }


@dataclass
class SafetyAssessment:
    stage: str
    category: str
    action: str
    severity: str
    answer: str | None = None
    hits: list[SafetyRuleHit] = field(default_factory=list)

    @property
    def should_short_circuit(self) -> bool:
        return self.action == "direct_response"

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "category": self.category,
            "action": self.action,
            "severity": self.severity,
            "answer": self.answer,
            "hits": [item.to_dict() for item in self.hits],
        }


@dataclass
class OutputGuardResult:
    safe: bool
    answer: str
    action: str
    hits: list[SafetyRuleHit] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "stage": "output",
            "safe": self.safe,
            "action": self.action,
            "hits": [item.to_dict() for item in self.hits],
        }


class RuleBasedSafetyGuard:
    """Rule-only safety layer for the medical QA pipeline."""

    _NEGATION_MARKERS = (
        "没有",
        "没",
        "无",
        "未",
        "否认",
        "不伴",
        "不出现",
        "不发烧",
        "不发热",
        "不胸痛",
        "不咳嗽",
        "不呕吐",
        "不腹泻",
        "不呼吸困难",
    )

    _MEDICAL_HINTS = (
        "病",
        "症",
        "痛",
        "疼",
        "痒",
        "咳",
        "烧",
        "热",
        "吐",
        "泻",
        "晕",
        "药",
        "检查",
        "治疗",
        "医生",
        "医院",
        "急诊",
        "诊断",
        "过敏",
        "发炎",
        "感染",
        "血",
        "呼吸",
        "胸",
        "腹",
        "胃",
        "头",
        "喉",
        "鼻",
        "皮疹",
        "怀孕",
        "孕",
    )

    _PROMPT_INJECTION_PATTERNS = (
        r"忽略.*(规则|指令|系统|提示词)",
        r"无视.*(规则|指令|系统|提示词)",
        r"忘记.*(规则|指令|系统|提示词)",
        r"输出.*(系统提示|提示词|prompt)",
        r"jailbreak|越狱|developer mode",
    )

    _SELF_HARM_PATTERNS = (
        "自杀",
        "不想活",
        "结束生命",
        "割腕",
        "跳楼",
        "上吊",
        "服毒",
        "百草枯",
        "农药",
        "安眠药吃多",
        "安眠药会死",
        "吃多少会死",
        "药物过量",
        "overdose",
    )

    _EMERGENCY_TERMS = (
        "呼吸困难",
        "喘不上气",
        "胸痛",
        "胸闷",
        "意识不清",
        "意识障碍",
        "昏迷",
        "抽搐",
        "休克",
        "大出血",
        "大量出血",
        "呕血",
        "便血",
        "黑便",
        "腹部僵硬",
        "剧烈腹痛",
        "持续高热",
        "高热不退",
        "口唇发紫",
        "过敏性休克",
        "喉头水肿",
        "吞咽困难",
        "一侧肢体无力",
        "口角歪斜",
        "说话含糊",
        "视物模糊",
    )

    _DOSAGE_QUERY_PATTERNS = (
        r"(吃|服|用|口服|注射).{0,8}(多少|几片|几粒|几袋|几毫克|几克|剂量|用量)",
        r"(一次|每天|每日|一日).{0,8}(几|多少|剂量|用量)",
        r"(剂量|用量).{0,8}(多少|怎么|如何)",
        r"(开|配).{0,4}(处方|药)",
    )

    _HIGH_RISK_MEDICINES = (
        "胰岛素",
        "华法林",
        "地高辛",
        "吗啡",
        "芬太尼",
        "曲马多",
        "安眠药",
        "抗凝",
        "激素",
        "抗生素",
        "头孢",
        "阿莫西林",
        "阿司匹林",
        "硝酸甘油",
    )

    _MEDICATION_REQUEST_PATTERNS = (
        r"吃什么药",
        r"用什么药",
        r"买什么药",
        r"能不能吃",
        r"可不可以吃",
        r"需要吃药",
    )

    _UNSAFE_OUTPUT_PATTERNS = (
        (
            "unsafe_definitive_diagnosis",
            "block",
            r"((你|您)(已经)?得了|(你|您).{0,4}(就是|肯定是|一定是|确诊为|已经确诊|患有))",
            "输出包含确定性诊断表述",
        ),
        (
            "unsafe_avoid_care",
            "block",
            r"(不用|无需|不需要).{0,4}(就医|看医生|去医院|检查|复诊)",
            "输出包含不当规避就医表述",
        ),
        (
            "unsafe_guarantee",
            "block",
            r"(保证|一定|肯定).{0,8}(治愈|治好|根治|没事|能好|会好)",
            "输出包含保证疗效表述",
        ),
        (
            "unsafe_medication_adjustment",
            "block",
            r"(自行|直接).{0,8}(停药|加量|减量|换药|服用|吃药)",
            "输出包含自行调整或使用药物表述",
        ),
        (
            "unsafe_dosage",
            "block",
            r"(一次|每日|每天|一日|口服|服用|注射).{0,12}\d+(\.\d+)?\s*(mg|毫克|g|克|片|粒|袋|ml|毫升)",
            "输出包含具体用药剂量",
        ),
    )

    def assess_input(self, query: str) -> SafetyAssessment:
        text = self._normalize(query)
        hits: list[SafetyRuleHit] = []

        prompt_injection = self._regex_matches(text, self._PROMPT_INJECTION_PATTERNS)
        if prompt_injection:
            hits.append(
                SafetyRuleHit(
                    code="prompt_injection",
                    severity="warn",
                    message="输入包含试图覆盖系统规则的内容",
                    matches=prompt_injection,
                )
            )

        self_harm = self._term_matches(text, self._SELF_HARM_PATTERNS)
        if self_harm:
            hits.append(
                SafetyRuleHit(
                    code="self_harm_or_poisoning",
                    severity="urgent",
                    message="输入包含自伤、中毒或药物过量风险",
                    matches=self_harm,
                )
            )
            return SafetyAssessment(
                stage="input",
                category="self_harm_or_poisoning",
                action="direct_response",
                severity="urgent",
                answer=self._self_harm_answer(),
                hits=hits,
            )

        emergency = self._unnegated_term_matches(text, self._EMERGENCY_TERMS)
        if emergency:
            hits.append(
                SafetyRuleHit(
                    code="red_flag_symptom",
                    severity="urgent",
                    message="输入包含需要优先就医的危险信号",
                    matches=emergency,
                )
            )
            return SafetyAssessment(
                stage="input",
                category="emergency",
                action="direct_response",
                severity="urgent",
                answer=self._emergency_answer(emergency),
                hits=hits,
            )

        dosage = self._regex_matches(text, self._DOSAGE_QUERY_PATTERNS)
        high_risk = self._term_matches(text, self._HIGH_RISK_MEDICINES)
        if dosage or (high_risk and self._looks_like_medication_question(text)):
            hits.append(
                SafetyRuleHit(
                    code="medication_dosage_or_prescription",
                    severity="block",
                    message="输入请求具体用药剂量、处方或高风险药物使用建议",
                    matches=(dosage + high_risk)[:10],
                )
            )
            return SafetyAssessment(
                stage="input",
                category="medication_dosage",
                action="direct_response",
                severity="block",
                answer=self._medication_boundary_answer(),
                hits=hits,
            )

        medication_request = self._regex_matches(text, self._MEDICATION_REQUEST_PATTERNS)
        if medication_request:
            hits.append(
                SafetyRuleHit(
                    code="medication_general_request",
                    severity="warn",
                    message="输入涉及药物建议，需要限制到通用信息",
                    matches=medication_request,
                )
            )

        if not self._looks_medical(text):
            hits.append(
                SafetyRuleHit(
                    code="non_medical",
                    severity="block",
                    message="输入不是医疗健康相关问题",
                    matches=[],
                )
            )
            return SafetyAssessment(
                stage="input",
                category="non_medical",
                action="direct_response",
                severity="block",
                answer="我只能回答医疗健康相关的问题。请描述你的症状、疾病、检查、用药或健康咨询需求。",
                hits=hits,
            )

        severity = "warn" if hits else "none"
        return SafetyAssessment(
            stage="input",
            category="medical",
            action="continue",
            severity=severity,
            hits=hits,
        )

    def prompt_constraints(self, assessment: dict | SafetyAssessment | None) -> str:
        category = self._assessment_category(assessment)
        lines = [
            "安全边界：",
            "- 不能把候选疾病表述为确诊；只能说“可能”“需医生评估”。",
            "- 不能提供具体处方、药物剂量、服药频次，不能建议用户自行停药、加量或换药。",
            "- 不能保证疗效，不能建议用户不用就医或不用检查。",
            "- 如信息不足，优先追问或说明无法根据已知信息回答。",
            "- 如出现胸痛、呼吸困难、意识异常、大量出血、呕血/便血、剧烈腹痛、持续高热等危险信号，应建议及时就医或急诊。",
        ]
        if category == "medical":
            lines.append("- 对药物相关问题，仅提供通用原则，并建议咨询医生或药师。")
        return "\n".join(lines)

    def guard_output(
        self,
        answer: str,
        *,
        query: str,
        evidence: Sequence[str],
        input_assessment: dict | SafetyAssessment | None = None,
    ) -> OutputGuardResult:
        text = self._normalize(answer)
        hits: list[SafetyRuleHit] = []
        for code, severity, pattern, message in self._UNSAFE_OUTPUT_PATTERNS:
            matches = self._regex_matches(text, [pattern])
            if matches:
                hits.append(
                    SafetyRuleHit(
                        code=code,
                        severity=severity,
                        message=message,
                        matches=matches,
                    )
                )

        if not hits:
            return OutputGuardResult(safe=True, answer=answer, action="pass")

        safe_answer = self._safe_rewrite(
            query=query,
            evidence=evidence,
            hits=hits,
            input_assessment=input_assessment,
        )
        return OutputGuardResult(
            safe=False,
            answer=safe_answer,
            action="rewrite",
            hits=hits,
        )

    @staticmethod
    def _normalize(value: str) -> str:
        return re.sub(r"\s+", "", str(value or "")).lower()

    def _looks_medical(self, text: str) -> bool:
        if any(item in text for item in self._MEDICAL_HINTS):
            return True
        return bool(self._looks_like_medication_question(text))

    def _looks_like_medication_question(self, text: str) -> bool:
        return bool(
            self._regex_matches(text, self._MEDICATION_REQUEST_PATTERNS)
            or any(item in text for item in self._HIGH_RISK_MEDICINES)
            or "剂量" in text
            or "用量" in text
        )

    @staticmethod
    def _regex_matches(text: str, patterns: Sequence[str]) -> list[str]:
        matches: list[str] = []
        for pattern in patterns:
            for item in re.finditer(pattern, text, flags=re.IGNORECASE):
                matched = item.group(0)
                if matched and matched not in matches:
                    matches.append(matched)
        return matches

    @staticmethod
    def _term_matches(text: str, terms: Sequence[str]) -> list[str]:
        matches: list[str] = []
        for term in terms:
            normalized = RuleBasedSafetyGuard._normalize(term)
            if normalized and normalized in text and term not in matches:
                matches.append(term)
        return matches

    def _unnegated_term_matches(self, text: str, terms: Sequence[str]) -> list[str]:
        matches: list[str] = []
        for term in terms:
            normalized = self._normalize(term)
            if not normalized:
                continue
            start = text.find(normalized)
            while start >= 0:
                if not self._is_negated(text, start):
                    matches.append(term)
                    break
                start = text.find(normalized, start + len(normalized))
        return matches

    def _is_negated(self, text: str, start: int) -> bool:
        window = text[max(0, start - 10):start]
        return any(marker in window for marker in self._NEGATION_MARKERS)

    @staticmethod
    def _assessment_category(assessment: dict | SafetyAssessment | None) -> str:
        if isinstance(assessment, SafetyAssessment):
            return assessment.category
        if isinstance(assessment, dict):
            return str(assessment.get("category") or "")
        return ""

    @staticmethod
    def _emergency_answer(matches: Sequence[str]) -> str:
        matched = "、".join(dict.fromkeys(str(item) for item in matches if item)) or "危险信号"
        return (
            f"你提到“{matched}”，这类情况可能需要优先排除急症。"
            "请尽快前往急诊或拨打当地急救电话；如果症状正在加重、伴随意识异常、明显出血、持续高热或呼吸困难，不要等待线上问答。"
            "在就医前，尽量让身边的人陪同，并准备好既往病史、用药和过敏史信息。"
        )

    @staticmethod
    def _self_harm_answer() -> str:
        return (
            "你描述的情况可能存在自伤、中毒或药物过量风险。请立刻联系身边可信任的人陪你，并尽快拨打当地急救电话或前往急诊。"
            "如果已经服用了过量药物、农药或不明物质，不要自行催吐，也不要继续独处；请带上药物包装或相关信息给医护人员。"
        )

    @staticmethod
    def _medication_boundary_answer() -> str:
        return (
            "我不能提供具体处方、药物剂量、服药频次，或指导你自行使用高风险药物。"
            "用药需要结合年龄、体重、肝肾功能、过敏史、正在使用的药物和具体诊断。"
            "请咨询医生或药师；如果已经误服、过量服药，或出现呼吸困难、意识异常、严重过敏等情况，请及时就医或急诊。"
        )

    @staticmethod
    def _safe_rewrite(
        *,
        query: str,
        evidence: Sequence[str],
        hits: Sequence[SafetyRuleHit],
        input_assessment: dict | SafetyAssessment | None,
    ) -> str:
        hit_codes = {item.code for item in hits}
        if any("dosage" in code or "medication" in code for code in hit_codes):
            return RuleBasedSafetyGuard._medication_boundary_answer()

        lines = [
            "根据当前信息，我不能给出确诊、保证疗效，或建议你避免就医。",
            "下面内容仅基于已检索到的知识，供你和医生沟通时参考，不能替代面诊判断。",
        ]
        usable_evidence = [str(item).strip() for item in evidence if str(item).strip()]
        if usable_evidence:
            lines.append("已检索到的知识：")
            lines.extend(f"- {item}" for item in usable_evidence[:5])
        else:
            lines.append("根据已知信息无法回答该问题。")
        lines.append("如果症状明显加重，或出现胸痛、呼吸困难、意识异常、呕血/便血、剧烈腹痛、持续高热等情况，请及时就医或急诊。")
        return "\n".join(lines)
