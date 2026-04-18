from app.schemas.chat import ChatResponse
from app.services.clinical_context import ClinicalContextService
from app.services.disease_resolution import DiseaseResolver
from app.services.entity_search import EntityNormalizer
from app.services.intent_service import IntentRuleEngine
from app.services.kg_service import GraphService
from app.services.llm_service import DashScopeService, OllamaService
from app.services.medical_qa_graph import MedicalQAGraph
from app.services.memory_service import MemoryService


class ChatService:
    def __init__(
        self,
        entity_normalizer: EntityNormalizer,
        intent_engine: IntentRuleEngine,
        graph_service: GraphService,
        disease_resolver: DiseaseResolver,
        memory_service: MemoryService,
        clinical_context_service: ClinicalContextService,
        llm_service: DashScopeService | OllamaService,
        chat_trace_enabled: bool = False,
        chat_trace_max_chars: int = 4000,
        llm_provider: str = "unknown",
        llm_model: str = "unknown",
        disease_max_follow_up_turns: int = 2,
        disease_possible_confidence_threshold: float = 0.55,
        disease_possible_candidate_limit: int = 3,
    ) -> None:
        self.graph = MedicalQAGraph(
            entity_normalizer=entity_normalizer,
            intent_engine=intent_engine,
            graph_service=graph_service,
            disease_resolver=disease_resolver,
            memory_service=memory_service,
            clinical_context_service=clinical_context_service,
            llm_service=llm_service,
            chat_trace_enabled=chat_trace_enabled,
            chat_trace_max_chars=chat_trace_max_chars,
            llm_provider=llm_provider,
            llm_model=llm_model,
            disease_max_follow_up_turns=disease_max_follow_up_turns,
            disease_possible_confidence_threshold=disease_possible_confidence_threshold,
            disease_possible_candidate_limit=disease_possible_candidate_limit,
        )
        self.memory_service = memory_service

    def ask(
        self,
        query: str,
        user_id: str,
        conversation_id: str | None = None,
    ) -> ChatResponse:
        return self.graph.invoke(
            query=query,
            user_id=user_id,
            conversation_id=conversation_id,
        )

    def ask_stream(
        self,
        query: str,
        user_id: str,
        conversation_id: str | None = None,
        on_status=None,
        on_token=None,
    ) -> ChatResponse:
        return self.graph.invoke_stream(
            query=query,
            user_id=user_id,
            conversation_id=conversation_id,
            on_status=on_status,
            on_token=on_token,
        )
