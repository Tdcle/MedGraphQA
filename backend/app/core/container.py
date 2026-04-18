from dataclasses import dataclass

from app.core.config import Settings
from app.services.auth_repository import AuthRepository
from app.services.auth_service import AuthService
from app.services.chat_service import ChatService
from app.services.clinical_context import ClinicalContextService
from app.services.disease_resolution import DiseaseResolver
from app.services.entity_search import (
    ElasticsearchEntityIndex,
    EntityNormalizer,
    PostgresEntityRepository,
)
from app.services.entity_ner import NullEntityMentionExtractor, RobertaRnnEntityMentionExtractor
from app.services.embedding_service import EntityEmbeddingService
from app.services.intent_service import IntentRuleEngine
from app.services.kg_service import GraphService
from app.services.llm_service import DashScopeService, OllamaService
from app.services.memory_repository import MemoryRepository
from app.services.memory_service import MemoryService


@dataclass
class ServiceContainer:
    settings: Settings
    auth_service: AuthService
    auth_repository: AuthRepository
    memory_repository: MemoryRepository
    chat_service: ChatService
    graph_service: GraphService
    entity_repository: PostgresEntityRepository
    entity_search_index: ElasticsearchEntityIndex


def build_container(settings: Settings) -> ServiceContainer:
    auth_repository = AuthRepository(
        dsn=settings.postgres_dsn,
        min_size=settings.postgres_pool_min_size,
        max_size=settings.postgres_pool_max_size,
        timeout_seconds=settings.postgres_pool_timeout_seconds,
    )
    auth_service = AuthService(
        auth_repository=auth_repository, ttl_minutes=settings.token_ttl_minutes
    )
    memory_repository = MemoryRepository(
        dsn=settings.postgres_dsn,
        min_size=settings.postgres_pool_min_size,
        max_size=settings.postgres_pool_max_size,
        timeout_seconds=settings.postgres_pool_timeout_seconds,
    )
    memory_service = MemoryService(repository=memory_repository)

    entity_repository = PostgresEntityRepository(
        dsn=settings.postgres_dsn,
        min_size=settings.postgres_pool_min_size,
        max_size=settings.postgres_pool_max_size,
        timeout_seconds=settings.postgres_pool_timeout_seconds,
    )
    entity_search_index = ElasticsearchEntityIndex(
        hosts=settings.elasticsearch_hosts,
        index_name=settings.elasticsearch_index,
        username=settings.elasticsearch_username,
        password=settings.elasticsearch_password,
        request_timeout_seconds=settings.elasticsearch_request_timeout_seconds,
    )
    embedding_service = EntityEmbeddingService(
        provider=settings.embedding_provider,
        model=settings.embedding_model,
        api_base=settings.embedding_api_base,
        api_key=settings.embedding_api_key,
        timeout_seconds=settings.embedding_timeout_seconds,
        enabled=settings.entity_search_vector_enabled,
    )
    if settings.entity_ner_enabled:
        mention_extractor = RobertaRnnEntityMentionExtractor(
            enabled=True,
            model_path=settings.entity_ner_model_path,
            pretrained_model=settings.entity_ner_pretrained_model,
            labels=settings.entity_ner_labels,
            max_length=settings.entity_ner_max_length,
            device=settings.entity_ner_device,
            rnn_hidden_size=settings.entity_ner_rnn_hidden_size,
            rnn_num_layers=settings.entity_ner_rnn_num_layers,
            confidence_threshold=settings.entity_ner_confidence_threshold,
        )
    else:
        mention_extractor = NullEntityMentionExtractor()
    entity_normalizer = EntityNormalizer(
        repository=entity_repository,
        search_index=entity_search_index,
        embedding_service=embedding_service,
        max_entities=settings.entity_search_max_entities,
        exact_terms_limit=settings.entity_search_exact_terms_limit,
        elastic_terms_limit=settings.entity_search_elastic_terms_limit,
        elastic_top_k=settings.entity_search_elastic_top_k,
        elastic_min_score=settings.entity_search_elastic_min_score,
        vector_enabled=settings.entity_search_vector_enabled,
        vector_top_k=settings.entity_search_vector_top_k,
        vector_min_score=settings.entity_search_vector_min_score,
        rrf_k=settings.entity_search_rrf_k,
        exact_rrf_weight=settings.entity_search_exact_rrf_weight,
        elastic_rrf_weight=settings.entity_search_elastic_rrf_weight,
        vector_rrf_weight=settings.entity_search_vector_rrf_weight,
        mention_extractor=mention_extractor,
    )
    intent_engine = IntentRuleEngine()
    graph_service = GraphService(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )
    disease_resolver = DiseaseResolver(
        graph_service=graph_service,
        repository=entity_repository,
        confidence_threshold=settings.disease_confidence_threshold,
        top_gap_threshold=settings.disease_top_gap_threshold,
        min_symptoms_for_inference=settings.disease_min_symptoms_for_inference,
        ask_follow_up_when_below_threshold=settings.disease_ask_follow_up_when_below_threshold,
        candidate_limit=settings.disease_candidate_limit,
    )
    if settings.llm_provider == "ollama":
        llm_service = OllamaService(
            api_base=settings.ollama_api_base,
            model=settings.ollama_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout_seconds=settings.llm_timeout_seconds,
            enable_thinking=settings.llm_enable_thinking,
        )
    elif settings.llm_provider == "dashscope":
        llm_service = DashScopeService(
            api_base=settings.dashscope_api_base,
            model=settings.dashscope_model,
            api_key=settings.dashscope_api_key,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout_seconds=settings.llm_timeout_seconds,
            enable_thinking=settings.llm_enable_thinking,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")

    if settings.clinical_context_provider == "dashscope":
        clinical_context_llm = DashScopeService(
            api_base=settings.clinical_context_dashscope_api_base,
            model=settings.clinical_context_model,
            api_key=settings.clinical_context_api_key,
            temperature=settings.clinical_context_temperature,
            max_tokens=settings.clinical_context_max_tokens,
            timeout_seconds=settings.clinical_context_timeout_seconds,
            enable_thinking=settings.clinical_context_enable_thinking,
        )
    elif settings.clinical_context_provider == "ollama":
        clinical_context_llm = OllamaService(
            api_base=settings.ollama_api_base,
            model=settings.clinical_context_model,
            temperature=settings.clinical_context_temperature,
            max_tokens=settings.clinical_context_max_tokens,
            timeout_seconds=settings.clinical_context_timeout_seconds,
            enable_thinking=settings.clinical_context_enable_thinking,
        )
    else:
        clinical_context_llm = None
    clinical_context_service = ClinicalContextService(
        llm_service=clinical_context_llm,
        enabled=settings.clinical_context_enabled,
    )
    chat_service = ChatService(
        entity_normalizer=entity_normalizer,
        intent_engine=intent_engine,
        graph_service=graph_service,
        disease_resolver=disease_resolver,
        memory_service=memory_service,
        clinical_context_service=clinical_context_service,
        llm_service=llm_service,
        chat_trace_enabled=settings.chat_trace_enabled,
        chat_trace_max_chars=settings.chat_trace_max_chars,
        llm_provider=settings.llm_provider,
        llm_model=settings.ollama_model
        if settings.llm_provider == "ollama"
        else settings.dashscope_model,
        disease_max_follow_up_turns=settings.disease_max_follow_up_turns,
        disease_possible_confidence_threshold=settings.disease_possible_confidence_threshold,
        disease_possible_candidate_limit=settings.disease_possible_candidate_limit,
    )
    return ServiceContainer(
        settings=settings,
        auth_service=auth_service,
        auth_repository=auth_repository,
        memory_repository=memory_repository,
        chat_service=chat_service,
        graph_service=graph_service,
        entity_repository=entity_repository,
        entity_search_index=entity_search_index,
    )
