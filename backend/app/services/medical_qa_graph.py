import logging
import time
from typing import List, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.core.request_context import get_request_id
from app.schemas.chat import ChatResponse
from app.services import chat_memory
from app.services.clinical_context import ClinicalContext, ClinicalContextService
from app.services.chat_response_builder import build_chat_response, fallback_answer
from app.services.chat_trace import emit_chat_trace
from app.services.disease_resolution import DiseaseResolutionResult, DiseaseResolver
from app.services.entity_search import EntityCandidate, EntityNormalizer
from app.services.follow_up_service import FollowUpQuestionService
from app.services.intent_service import IntentRuleEngine
from app.services.kg_service import GraphService
from app.services.llm_service import DashScopeService, OllamaService
from app.services.medical_qa_knowledge import MedicalKnowledgeGatherer
from app.services.memory_repository import UserMemory
from app.services.memory_service import MemoryService
from app.services.operation_log import log_operation
from app.services.safety_guardrails import RuleBasedSafetyGuard


logger = logging.getLogger("medgraphqa.chat")

_NEGATED_TERM_ALIASES = {
    "发烧": "发热",
    "发热": "发热",
    "胸口痛": "胸痛",
    "胸部疼痛": "胸痛",
    "胸疼": "胸痛",
    "喘不上气": "呼吸困难",
    "气短": "呼吸困难",
    "呼吸费力": "呼吸困难",
}


class MedicalQAState(TypedDict, total=False):
    query: str
    user_id: str
    conversation_id: str | None
    memory_state: dict
    long_term_memories: List[UserMemory]
    memory_context: str
    memory_writes: List[UserMemory]
    effective_query: str
    current_intents: List[str]
    intents: List[str]
    ner_terms: List[str]
    current_entities: List[EntityCandidate]
    memory_entities: List[EntityCandidate]
    preliminary_entities: List[EntityCandidate]
    clinical_context: dict
    entities: List[EntityCandidate]
    evidence: List[str]
    used_intents: List[str]
    follow_up_answer: str | None
    disease_resolution: DiseaseResolutionResult | None
    input_safety: dict
    output_safety: dict
    safety_short_circuit: bool
    prompt: str
    answer: str
    fallback_reason: str | None
    llm_error: str | None
    llm_duration_ms: float | None
    started_at: float
    response: ChatResponse


class MedicalQAGraph:
    def __init__(
        self,
        entity_normalizer: EntityNormalizer,
        intent_engine: IntentRuleEngine,
        graph_service: GraphService,
        disease_resolver: DiseaseResolver,
        memory_service: MemoryService,
        clinical_context_service: ClinicalContextService,
        llm_service: DashScopeService | OllamaService,
        chat_trace_enabled: bool,
        chat_trace_max_chars: int,
        llm_provider: str,
        llm_model: str,
        disease_max_follow_up_turns: int = 2,
        disease_possible_confidence_threshold: float = 0.55,
        disease_possible_candidate_limit: int = 3,
    ) -> None:
        self.entity_normalizer = entity_normalizer
        self.intent_engine = intent_engine
        self.knowledge = MedicalKnowledgeGatherer(
            graph_service=graph_service,
            disease_resolver=disease_resolver,
        )
        self.memory_service = memory_service
        self.clinical_context_service = clinical_context_service
        self.llm_service = llm_service
        self.follow_up_service = FollowUpQuestionService(llm_service)
        self.safety_guard = RuleBasedSafetyGuard()
        self.chat_trace_enabled = chat_trace_enabled
        self.chat_trace_max_chars = max(200, chat_trace_max_chars)
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.disease_max_follow_up_turns = max(0, disease_max_follow_up_turns)
        self.disease_possible_confidence_threshold = disease_possible_confidence_threshold
        self.disease_possible_candidate_limit = max(1, disease_possible_candidate_limit)
        self.app = self._compile()

    def invoke(
        self,
        query: str,
        user_id: str,
        conversation_id: str | None,
    ) -> ChatResponse:
        state = self.app.invoke(
            self._initial_state(query, user_id, conversation_id)
        )
        return state["response"]

    def invoke_stream(
        self,
        query: str,
        user_id: str,
        conversation_id: str | None,
        on_status=None,
        on_token=None,
    ) -> ChatResponse:
        state: MedicalQAState = self._initial_state(query, user_id, conversation_id)
        for label, node in [
            ("正在载入会话", self._load_session),
            ("正在进行安全检查", self._evaluate_input_safety),
        ]:
            if on_status:
                on_status(label)
            state.update(self._run_node(label, node, state))

        if self._route_after_input_safety(state) == "safety_response":
            state.update(self._run_node("构建安全回答", self._build_safety_response, state))
        else:
            for label, node in [
                ("正在识别意图", self._detect_intents),
                ("正在用RoBERTa识别实体片段", self._extract_entity_mentions),
                ("正在抽取临床上下文", self._extract_clinical_context),
                ("正在识别实体", self._normalize_entities),
                ("正在读取长期记忆", self._load_long_term_memory),
                ("正在检索知识图谱", self._gather_knowledge),
            ]:
                if on_status:
                    on_status(label)
                state.update(self._run_node(label, node, state))

            route = self._route_after_knowledge(state)
            logger.info("operation=chat.route_after_knowledge status=ok route=%s", route)
            if route == "follow_up":
                state.update(self._run_node("构建追问", self._build_follow_up, state))
            elif route == "no_evidence":
                state.update(self._run_node("构建无证据回答", self._build_no_evidence, state))
            else:
                if on_status:
                    on_status("正在生成回答")
                state.update(
                    self._run_node(
                        "生成模型回答",
                        lambda current: self._generate_answer_stream(current, on_token=on_token),
                        state,
                    )
                )

        state.update(self._run_node("应用安全护栏", self._apply_output_guardrails, state))
        if on_token and state.get("answer"):
            on_token(state["answer"])
        state.update(self._run_node("抽取长期记忆", self._extract_memory, state))
        state.update(self._run_node("持久化对话", self._persist, state))
        return state["response"]

    @staticmethod
    def _initial_state(
        query: str,
        user_id: str,
        conversation_id: str | None,
    ) -> MedicalQAState:
        return {
            "query": query,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "started_at": time.perf_counter(),
            "fallback_reason": None,
            "llm_error": None,
            "llm_duration_ms": None,
        }

    @staticmethod
    def _run_node(label: str, node, state: MedicalQAState) -> dict:
        with log_operation(
            logger,
            "chat.node",
            node=label,
            query_len=len(state.get("query", "")),
        ) as result:
            output = node(state)
            if isinstance(output, dict):
                result["output_keys"] = ",".join(output.keys())
            return output

    def _compile(self):
        workflow = StateGraph(MedicalQAState)
        workflow.add_node("load_session", lambda state: self._run_node("载入会话", self._load_session, state))
        workflow.add_node("evaluate_input_safety", lambda state: self._run_node("安全检查", self._evaluate_input_safety, state))
        workflow.add_node("build_safety_response", lambda state: self._run_node("构建安全回答", self._build_safety_response, state))
        workflow.add_node("load_long_term_memory", lambda state: self._run_node("读取长期记忆", self._load_long_term_memory, state))
        workflow.add_node("detect_intents", lambda state: self._run_node("识别意图", self._detect_intents, state))
        workflow.add_node("extract_entity_mentions", lambda state: self._run_node("用RoBERTa识别实体片段", self._extract_entity_mentions, state))
        workflow.add_node("extract_clinical_context", lambda state: self._run_node("抽取临床上下文", self._extract_clinical_context, state))
        workflow.add_node("normalize_entities", lambda state: self._run_node("识别实体", self._normalize_entities, state))
        workflow.add_node("gather_knowledge", lambda state: self._run_node("检索知识图谱", self._gather_knowledge, state))
        workflow.add_node("build_follow_up", lambda state: self._run_node("构建追问", self._build_follow_up, state))
        workflow.add_node("build_no_evidence", lambda state: self._run_node("构建无证据回答", self._build_no_evidence, state))
        workflow.add_node("generate_answer", lambda state: self._run_node("生成模型回答", self._generate_answer, state))
        workflow.add_node("apply_output_guardrails", lambda state: self._run_node("应用安全护栏", self._apply_output_guardrails, state))
        workflow.add_node("extract_memory", lambda state: self._run_node("抽取长期记忆", self._extract_memory, state))
        workflow.add_node("persist", lambda state: self._run_node("持久化对话", self._persist, state))

        workflow.add_edge(START, "load_session")
        workflow.add_edge("load_session", "evaluate_input_safety")
        workflow.add_conditional_edges(
            "evaluate_input_safety",
            self._route_after_input_safety,
            {
                "safety_response": "build_safety_response",
                "continue": "detect_intents",
            },
        )
        workflow.add_edge("detect_intents", "extract_entity_mentions")
        workflow.add_edge("extract_entity_mentions", "extract_clinical_context")
        workflow.add_edge("extract_clinical_context", "normalize_entities")
        workflow.add_edge("normalize_entities", "load_long_term_memory")
        workflow.add_edge("load_long_term_memory", "gather_knowledge")
        workflow.add_conditional_edges(
            "gather_knowledge",
            self._route_after_knowledge,
            {
                "follow_up": "build_follow_up",
                "no_evidence": "build_no_evidence",
                "answer": "generate_answer",
            },
        )
        workflow.add_edge("build_safety_response", "apply_output_guardrails")
        workflow.add_edge("build_follow_up", "apply_output_guardrails")
        workflow.add_edge("build_no_evidence", "apply_output_guardrails")
        workflow.add_edge("generate_answer", "apply_output_guardrails")
        workflow.add_edge("apply_output_guardrails", "extract_memory")
        workflow.add_edge("extract_memory", "persist")
        workflow.add_edge("persist", END)
        return workflow.compile()

    def _load_session(self, state: MedicalQAState) -> dict:
        repo = self.entity_normalizer.repository
        conversation_id = repo.ensure_chat_session(
            user_id=state["user_id"],
            conversation_id=state.get("conversation_id"),
            title=state["query"],
        )
        memory_state = repo.get_chat_state(state["user_id"], conversation_id)
        repo.add_chat_message(
            user_id=state["user_id"],
            conversation_id=conversation_id,
            role="user",
            content=state["query"],
            metadata={"request_id": get_request_id()},
        )
        return {
            "conversation_id": conversation_id,
            "memory_state": memory_state,
            "effective_query": chat_memory.effective_query(
                state["query"],
                memory_state,
            ),
        }

    def _evaluate_input_safety(self, state: MedicalQAState) -> dict:
        assessment = self.safety_guard.assess_input(state["query"])
        hit_codes = ",".join(item.code for item in assessment.hits)
        logger.info(
            "operation=safety.input status=ok category=%s action=%s severity=%s hits=%s",
            assessment.category,
            assessment.action,
            assessment.severity,
            hit_codes or "-",
        )
        return {
            "input_safety": assessment.to_dict(),
            "safety_short_circuit": assessment.should_short_circuit,
        }

    @staticmethod
    def _route_after_input_safety(
        state: MedicalQAState,
    ) -> Literal["safety_response", "continue"]:
        if state.get("safety_short_circuit"):
            return "safety_response"
        return "continue"

    def _build_safety_response(self, state: MedicalQAState) -> dict:
        safety = state.get("input_safety") or {}
        answer = str(safety.get("answer") or "该问题暂时不能直接回答。")
        return {
            "answer": answer,
            "fallback_reason": f"safety_{safety.get('category', 'guardrail')}",
            "intents": [],
            "current_intents": [],
            "used_intents": ["安全提醒"],
            "entities": [],
            "evidence": [],
            "disease_resolution": None,
        }

    def _detect_intents(self, state: MedicalQAState) -> dict:
        current = self.intent_engine.detect(state["query"])
        intents = chat_memory.merge_intents(state.get("memory_state", {}), current)
        return {"current_intents": current, "intents": intents}

    def _extract_entity_mentions(self, state: MedicalQAState) -> dict:
        expected_types = self.entity_normalizer.expected_types(state["intents"])
        terms = self.entity_normalizer.extract_mention_terms(
            query=state["query"],
            expected_types=expected_types,
        )
        return {"ner_terms": terms}

    def _normalize_entities(self, state: MedicalQAState) -> dict:
        context = self._context_from_state(state)
        terms = self._filter_negated_terms(
            list(state.get("ner_terms", [])),
            context.negated_symptoms,
        )
        for term in self.clinical_context_service.symptom_terms(context):
            if (
                term
                and term not in terms
                and not self._is_negated_text(term, context.negated_symptoms)
            ):
                terms.append(term)
        for item in state.get("preliminary_entities", []):
            for term in [item.mention, item.matched_alias, item.canonical_name]:
                if (
                    term
                    and term not in terms
                    and not self._is_negated_text(term, context.negated_symptoms)
                ):
                    terms.append(term)
        if terms:
            current = self.entity_normalizer.resolve_terms(
                terms=terms,
                query=state["query"],
                intents=state["intents"],
                allow_vector=True,
            )
        else:
            current = self.entity_normalizer.resolve(
                query=state["query"],
                intents=state["intents"],
            )
        if not current and state.get("preliminary_entities"):
            current = state["preliminary_entities"]
        elif current and state.get("preliminary_entities"):
            current = chat_memory.merge_entities(state["preliminary_entities"], current)
        current = self._filter_negated_entities(current, context.negated_symptoms)
        memory_entities = chat_memory.entities_from_state(state.get("memory_state", {}))
        memory_entities = self._filter_negated_entities(
            memory_entities,
            context.negated_symptoms,
        )
        entities = chat_memory.merge_entities(memory_entities, current)
        entities = self._filter_negated_entities(entities, context.negated_symptoms)
        return {
            "current_entities": current,
            "memory_entities": memory_entities,
            "entities": entities,
        }

    @staticmethod
    def _filter_negated_terms(terms: list[str], negated: list[str]) -> list[str]:
        result: list[str] = []
        for term in terms:
            if term and not MedicalQAGraph._is_negated_text(term, negated):
                result.append(term)
        return result

    @staticmethod
    def _filter_negated_entities(
        entities: list[EntityCandidate],
        negated: list[str],
    ) -> list[EntityCandidate]:
        if not negated:
            return entities
        return [
            item
            for item in entities
            if item.entity_type != "疾病症状"
            or not any(
                MedicalQAGraph._same_clinical_term(value, neg)
                for value in [
                    item.canonical_name,
                    item.matched_alias,
                    item.normalized_alias,
                ]
                for neg in negated
            )
        ]

    @staticmethod
    def _is_negated_text(text: str, negated: list[str]) -> bool:
        return any(
            MedicalQAGraph._same_clinical_term(text, neg)
            for neg in negated
        )

    @staticmethod
    def _same_clinical_term(left: str, right: str) -> bool:
        left = MedicalQAGraph._canonical_clinical_term(left)
        right = MedicalQAGraph._canonical_clinical_term(right)
        if not left or not right:
            return False
        if left == right:
            return True
        return len(left) >= 2 and len(right) >= 2 and (left in right or right in left)

    @staticmethod
    def _canonical_clinical_term(value: str) -> str:
        text = str(value or "").strip()
        return _NEGATED_TERM_ALIASES.get(text, text)

    def _extract_clinical_context(self, state: MedicalQAState) -> dict:
        context = self.clinical_context_service.extract(
            query=state["query"],
            previous_context=state.get("memory_state", {}).get("clinical_context", {}),
            entities=[],
            entity_hints=state.get("ner_terms", []),
        )
        return {
            "preliminary_entities": [],
            "clinical_context": context.to_dict(),
        }

    def _load_long_term_memory(self, state: MedicalQAState) -> dict:
        memories = self.memory_service.load_for_chat(
            user_id=state["user_id"],
            query=state["query"],
            intents=state.get("intents", []),
        )
        return {
            "long_term_memories": memories,
            "memory_context": self.memory_service.format_for_prompt(memories),
        }

    def _gather_knowledge(self, state: MedicalQAState) -> dict:
        result = self.knowledge.gather(
            effective_query=state["effective_query"],
            intents=state["intents"],
            entities=state.get("entities", []),
            follow_up_turns=int(
                state.get("memory_state", {}).get("follow_up_turns") or 0
            ),
            max_follow_up_turns=self.disease_max_follow_up_turns,
            possible_confidence_threshold=self.disease_possible_confidence_threshold,
            possible_candidate_limit=self.disease_possible_candidate_limit,
            negated_symptoms=[
                str(item).strip()
                for item in (state.get("clinical_context") or {}).get("negated_symptoms", [])
                if str(item).strip()
            ],
        )
        return {
            "evidence": result.evidence,
            "used_intents": result.used_intents,
            "follow_up_answer": result.follow_up_answer,
            "disease_resolution": result.disease_resolution,
            "entities": result.entities,
        }

    def _route_after_knowledge(
        self,
        state: MedicalQAState,
    ) -> Literal["follow_up", "no_evidence", "answer"]:
        if state.get("follow_up_answer"):
            return "follow_up"
        if not state.get("evidence"):
            return "no_evidence"
        return "answer"

    def _build_follow_up(self, state: MedicalQAState) -> dict:
        resolution = state.get("disease_resolution")
        candidates = [item.disease for item in resolution.candidates] if resolution else []
        clinical_answer = self.follow_up_service.build(state)
        if not clinical_answer:
            clinical_answer = self._clinical_follow_up(state, candidates)
        answer = clinical_answer or state["follow_up_answer"] or "请补充更多信息。"
        return {
            "answer": answer,
            "fallback_reason": "disease_confidence_below_threshold",
        }

    def _build_no_evidence(self, state: MedicalQAState) -> dict:
        return {
            "answer": "根据已知信息无法回答该问题。",
            "fallback_reason": "no_evidence",
        }

    def _generate_answer(self, state: MedicalQAState) -> dict:
        prompt = self._build_answer_prompt(state)
        started = time.perf_counter()
        try:
            answer = self.llm_service.generate(prompt=prompt)
            llm_duration_ms = (time.perf_counter() - started) * 1000
        except Exception as exc:
            logger.exception("llm generation failed; using fallback answer")
            return {
                "prompt": prompt,
                "answer": fallback_answer(state["query"], state.get("evidence", [])),
                "fallback_reason": "llm_exception",
                "llm_error": f"{type(exc).__name__}: {exc}",
                "llm_duration_ms": (time.perf_counter() - started) * 1000,
            }

        if not answer:
            return {
                "prompt": prompt,
                "answer": fallback_answer(state["query"], state.get("evidence", [])),
                "fallback_reason": "llm_empty_output",
                "llm_duration_ms": llm_duration_ms,
            }
        return {"prompt": prompt, "answer": answer, "llm_duration_ms": llm_duration_ms}

    def _generate_answer_stream(self, state: MedicalQAState, on_token=None) -> dict:
        prompt = self._build_answer_prompt(state)
        started = time.perf_counter()
        if not hasattr(self.llm_service, "generate_stream"):
            return self._generate_answer(state)
        try:
            chunks: list[str] = []
            for token in self.llm_service.generate_stream(prompt):
                chunks.append(token)
            answer = "".join(chunks).strip()
            llm_duration_ms = (time.perf_counter() - started) * 1000
        except Exception as exc:
            logger.exception("streaming llm generation failed; using fallback answer")
            return {
                "prompt": prompt,
                "answer": fallback_answer(state["query"], state.get("evidence", [])),
                "fallback_reason": "llm_exception",
                "llm_error": f"{type(exc).__name__}: {exc}",
                "llm_duration_ms": (time.perf_counter() - started) * 1000,
            }
        if not answer:
            return {
                "prompt": prompt,
                "answer": fallback_answer(state["query"], state.get("evidence", [])),
                "fallback_reason": "llm_empty_output",
                "llm_duration_ms": llm_duration_ms,
            }
        return {"prompt": prompt, "answer": answer, "llm_duration_ms": llm_duration_ms}

    def _apply_output_guardrails(self, state: MedicalQAState) -> dict:
        result = self.safety_guard.guard_output(
            state.get("answer", ""),
            query=state["query"],
            evidence=state.get("evidence", []),
            input_assessment=state.get("input_safety", {}),
        )
        logger.info(
            "operation=safety.output status=ok action=%s safe=%s hits=%s",
            result.action,
            result.safe,
            ",".join(item.code for item in result.hits) or "-",
        )
        output = {"output_safety": result.to_dict()}
        if not result.safe:
            output["answer"] = result.answer
            output["fallback_reason"] = state.get("fallback_reason") or "safety_guardrail"
        return output

    def _build_answer_prompt(self, state: MedicalQAState) -> str:
        memory_prompt = self._memory_prompt(state.get("memory_context", ""))
        safety_prompt = self.safety_guard.prompt_constraints(state.get("input_safety", {}))
        return (
            "你是医疗问答助手。仅使用给定知识回答用户问题，禁止编造。"
            "如果知识不足，直接回答“根据已知信息无法回答该问题”。\n\n"
            f"{safety_prompt}\n\n"
            f"{memory_prompt}"
            f"用户问题：{state['effective_query']}\n\n"
            "知识：\n" + "\n".join(f"- {line}" for line in state.get("evidence", []))
        )

    def _extract_memory(self, state: MedicalQAState) -> dict:
        if state.get("safety_short_circuit"):
            return {"memory_writes": []}
        try:
            memories = self.memory_service.extract_and_save(
                user_id=state["user_id"],
                query=state["query"],
                entities=state.get("entities", []),
            )
            return {"memory_writes": memories}
        except Exception:
            logger.exception("long-term memory extraction failed")
            return {"memory_writes": []}

    def _persist(self, state: MedicalQAState) -> dict:
        repo = self.entity_normalizer.repository
        awaiting = state.get("fallback_reason") == "disease_confidence_below_threshold"
        next_state = chat_memory.build_next_state(
            previous_state=state.get("memory_state", {}),
            query=state["effective_query"],
            intents=state.get("intents", []),
            entities=state.get("entities", []),
            answer=state["answer"],
            follow_up_answer=state["answer"] if awaiting else None,
            disease_resolution=state.get("disease_resolution"),
        )
        next_state["clinical_context"] = state.get("clinical_context", {})
        repo.save_chat_state(state["user_id"], state["conversation_id"], next_state)
        repo.add_chat_message(
            user_id=state["user_id"],
            conversation_id=state["conversation_id"],
            role="assistant",
            content=state["answer"],
            metadata=self._message_metadata(state, awaiting),
        )
        self._emit_trace(state)
        return {"response": build_chat_response(state, awaiting)}

    def _message_metadata(self, state: MedicalQAState, awaiting: bool) -> dict:
        return {
            "request_id": get_request_id(),
            "intents": state.get("intents", []),
            "entities": [item.to_log_dict() for item in state.get("entities", [])],
            "evidence": state.get("evidence", []),
            "fallback_reason": state.get("fallback_reason"),
            "awaiting_user_clarification": awaiting,
            "long_term_memories": [
                item.to_log_dict() for item in state.get("long_term_memories", [])
            ],
            "memory_writes": [
                item.to_log_dict() for item in state.get("memory_writes", [])
            ],
            "clinical_context": state.get("clinical_context", {}),
            "safety": {
                "input": state.get("input_safety", {}),
                "output": state.get("output_safety", {}),
            },
        }

    @staticmethod
    def _context_from_state(state: MedicalQAState) -> ClinicalContext:
        service = ClinicalContextService(llm_service=None, enabled=False)
        return service._context_from_dict(state.get("clinical_context", {}))

    @staticmethod
    def _clinical_follow_up(state: MedicalQAState, candidates: List[str]) -> str | None:
        context = state.get("clinical_context") or {}
        symptoms = context.get("symptoms") or []
        if not symptoms:
            return None

        symptom_parts = []
        for item in symptoms[:5]:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            details = [
                item.get("body_part"),
                item.get("severity"),
                item.get("duration"),
                item.get("progression"),
                item.get("quality"),
            ]
            detail_text = "、".join(str(x) for x in details if x)
            if detail_text:
                symptom_parts.append(f"{item['name']}({detail_text})")
            else:
                symptom_parts.append(str(item["name"]))
        if not symptom_parts:
            return None

        missing = [
            str(item)
            for item in context.get("missing_info", [])
            if str(item).strip()
        ]
        red_flags = [
            str(item)
            for item in context.get("red_flags", [])
            if str(item).strip()
        ]
        negated_symptoms = [
            str(item)
            for item in context.get("negated_symptoms", [])
            if str(item).strip()
        ]
        red_flag_text = ""
        if red_flags:
            red_flag_text = "你还提到了需要注意的信息：" + "、".join(red_flags[:3]) + "。"
        negated_text = ""
        if negated_symptoms:
            negated_text = "已记录你否认了：" + "、".join(negated_symptoms[:6]) + "。"
        missing_questions = MedicalQAGraph._follow_up_missing_questions(
            symptoms=symptoms,
            llm_missing=missing,
            negated_symptoms=negated_symptoms,
            medication_status=context.get("medication_status"),
            diet_status=context.get("diet_status"),
            similar_history=context.get("similar_history"),
        )
        missing_text = "还需要补充：" + "；".join(missing_questions) + "。"
        return (
            f"我已记录你的描述：{'、'.join(symptom_parts)}。"
            f"{red_flag_text}"
            f"{negated_text}"
            "仅凭现有信息还不能确定具体疾病。"
            f"{missing_text}"
            "如果症状剧烈、持续加重，或伴随高热、呕血/便血、意识异常、腹部僵硬，请及时就医或急诊。"
        )

    @staticmethod
    def _follow_up_missing_questions(
        symptoms: list,
        llm_missing: list[str],
        negated_symptoms: list[str] | None = None,
        medication_status: str | None = None,
        diet_status: str | None = None,
        similar_history: str | None = None,
    ) -> list[str]:
        has_duration = False
        has_location = False
        has_severity = False
        names: set[str] = set()
        body_parts: set[str] = set()
        for item in symptoms:
            if not isinstance(item, dict):
                continue
            names.add(str(item.get("name") or ""))
            body_parts.add(str(item.get("body_part") or ""))
            has_duration = has_duration or bool(item.get("duration"))
            has_location = has_location or bool(item.get("body_part"))
            has_severity = has_severity or bool(item.get("severity"))

        questions: list[str] = []
        if not has_duration:
            questions.append("症状持续多久了，是否反复出现")
        if not has_location:
            questions.append("疼痛或不适的具体部位")
        if not has_severity:
            questions.append("程度是轻微、中等还是剧烈")

        negated = set(negated_symptoms or [])
        has_generic_no_other = any(
            item in negated for item in ["其他不良反应", "其他不适", "其他症状"]
        )
        accompaniment = []
        accompaniment_pool = MedicalQAGraph._accompaniment_pool(names, body_parts)
        for name in accompaniment_pool:
            if name not in names and name not in negated and not has_generic_no_other:
                accompaniment.append(name)
        if accompaniment:
            questions.append("是否伴随" + "、".join(accompaniment[:6]))

        history_parts = []
        if not diet_status:
            history_parts.append("近期饮食")
        if not medication_status:
            history_parts.append("用药")
        history_parts.append("既往病史")
        if not similar_history:
            history_parts.append("是否有类似发作")
        if history_parts:
            questions.append("、".join(history_parts))

        for item in llm_missing:
            text = str(item).strip()
            if MedicalQAGraph._is_answered_missing_item(
                text=text,
                negated=negated,
                medication_status=medication_status,
                diet_status=diet_status,
                similar_history=similar_history,
            ):
                continue
            if text and all(text not in existing for existing in questions):
                questions.append(text)
            if len(questions) >= 4:
                break
        return questions[:4]

    @staticmethod
    def _accompaniment_pool(names: set[str], body_parts: set[str]) -> list[str]:
        text = "".join(names | body_parts)
        if any(key in text for key in ["喉", "咽", "鼻"]):
            return ["发热", "咳嗽", "流鼻涕", "吞咽困难", "呼吸困难", "皮疹"]
        if any(key in text for key in ["腹", "胃", "肚"]):
            return ["发热", "呕吐", "腹泻", "便血", "黑便", "尿痛", "皮疹"]
        return ["发热", "咳嗽", "呕吐", "腹泻", "皮疹"]

    @staticmethod
    def _is_answered_missing_item(
        text: str,
        negated: set[str],
        medication_status: str | None,
        diet_status: str | None,
        similar_history: str | None,
    ) -> bool:
        if not text:
            return True
        if any(item and item in text for item in negated):
            return True
        if medication_status and any(key in text for key in ["用药", "吃药", "服药", "药"]):
            return True
        if diet_status and any(key in text for key in ["饮食", "吃了", "食物"]):
            return True
        if similar_history and any(key in text for key in ["类似", "反复", "既往", "以前"]):
            return True
        return False

    @staticmethod
    def _memory_prompt(memory_context: str) -> str:
        if not memory_context.strip():
            return ""
        return (
            "已知用户背景（仅作为风险提示和追问依据，不得据此直接诊断）：\n"
            f"{memory_context}\n\n"
        )

    def _emit_trace(self, state: MedicalQAState) -> None:
        emit_chat_trace(
            enabled=self.chat_trace_enabled,
            max_chars=self.chat_trace_max_chars,
            llm_provider=self.llm_provider,
            llm_model=self.llm_model,
            state=state,
        )
