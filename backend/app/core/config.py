import os
import json
from pathlib import Path

from dotenv import load_dotenv


_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(dotenv_path=_BACKEND_ROOT / ".env", override=False)
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=False)


def _load_app_config() -> dict:
    config_path = Path(os.getenv("APP_CONFIG_FILE", str(_BACKEND_ROOT / "config.json")))
    if not config_path.is_absolute():
        config_path = _BACKEND_ROOT / config_path
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_path(value: str, base: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _optional_bool(value) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


class Settings:
    def __init__(self) -> None:
        backend_root = _BACKEND_ROOT
        project_root = _PROJECT_ROOT
        config = _load_app_config()

        self.project_root = project_root
        self.backend_root = backend_root

        api_config = config.get("api", {})
        self.api_title = api_config.get("title", "MedGraphQA API")
        self.api_version = api_config.get("version", "1.0.0")
        self.debug = bool(api_config.get("debug", False))

        neo4j_config = config.get("neo4j", {})
        self.neo4j_uri = neo4j_config.get("uri", "bolt://localhost:7687")
        self.neo4j_user = neo4j_config.get("user", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "neo4j")
        self.neo4j_database = neo4j_config.get("database", "neo4j")

        postgres_config = config.get("postgres", {})
        self.postgres_dsn = os.getenv(
            "POSTGRES_DSN",
            postgres_config.get(
                "dsn", "postgresql://postgres:postgres@localhost:5432/medgraphqa"
            ),
        )
        self.postgres_pool_min_size = int(postgres_config.get("pool_min_size", 1))
        self.postgres_pool_max_size = int(postgres_config.get("pool_max_size", 5))
        self.postgres_pool_timeout_seconds = int(
            postgres_config.get("pool_timeout_seconds", 5)
        )

        elastic_config = config.get("elasticsearch", {})
        elastic_hosts_env = os.getenv("ELASTICSEARCH_HOSTS", "")
        self.elasticsearch_hosts = (
            [x.strip() for x in elastic_hosts_env.split(",") if x.strip()]
            if elastic_hosts_env
            else elastic_config.get("hosts", ["http://localhost:9200"])
        )
        self.elasticsearch_index = elastic_config.get("index", "medical_entity_alias")
        self.elasticsearch_username = os.getenv(
            "ELASTICSEARCH_USERNAME", elastic_config.get("username", "")
        )
        self.elasticsearch_password = os.getenv(
            "ELASTICSEARCH_PASSWORD", elastic_config.get("password", "")
        )
        self.elasticsearch_request_timeout_seconds = int(
            elastic_config.get("request_timeout_seconds", 5)
        )

        entity_search_config = config.get("entity_search", {})
        self.entity_search_max_entities = int(entity_search_config.get("max_entities", 5))
        self.entity_search_exact_terms_limit = int(
            entity_search_config.get("exact_terms_limit", 256)
        )
        self.entity_search_elastic_terms_limit = int(
            entity_search_config.get("elastic_terms_limit", 24)
        )
        self.entity_search_elastic_top_k = int(
            entity_search_config.get("elastic_top_k", 8)
        )
        self.entity_search_elastic_min_score = float(
            entity_search_config.get("elastic_min_score", 2.0)
        )
        self.entity_search_vector_enabled = bool(
            entity_search_config.get("vector_enabled", False)
        )
        self.entity_search_vector_top_k = int(entity_search_config.get("vector_top_k", 12))
        self.entity_search_vector_min_score = float(
            entity_search_config.get("vector_min_score", 1.2)
        )
        self.entity_search_rrf_k = int(entity_search_config.get("rrf_k", 60))
        self.entity_search_exact_rrf_weight = float(
            entity_search_config.get("exact_rrf_weight", 1.6)
        )
        self.entity_search_elastic_rrf_weight = float(
            entity_search_config.get("elastic_rrf_weight", 1.0)
        )
        self.entity_search_vector_rrf_weight = float(
            entity_search_config.get("vector_rrf_weight", 1.2)
        )

        entity_ner_config = config.get("entity_ner", {})
        self.entity_ner_enabled = bool(entity_ner_config.get("enabled", False))
        self.entity_ner_model_path = _resolve_path(
            entity_ner_config.get(
                "model_path",
                "../model/best_roberta_rnn_model_ent_aug.pt",
            ),
            backend_root,
        )
        entity_ner_pretrained_model = str(
            entity_ner_config.get("pretrained_model", "../model/chinese-roberta-wwm-ext")
        )
        pretrained_path = Path(entity_ner_pretrained_model)
        if pretrained_path.is_absolute():
            self.entity_ner_pretrained_model = str(pretrained_path)
        elif any(sep in entity_ner_pretrained_model for sep in ["/", "\\"]):
            self.entity_ner_pretrained_model = str((backend_root / pretrained_path).resolve())
        else:
            self.entity_ner_pretrained_model = entity_ner_pretrained_model
        self.entity_ner_max_length = int(entity_ner_config.get("max_length", 128))
        self.entity_ner_device = str(entity_ner_config.get("device", "cpu"))
        self.entity_ner_confidence_threshold = float(
            entity_ner_config.get("confidence_threshold", 0.4)
        )
        self.entity_ner_rnn_hidden_size = int(
            entity_ner_config.get("rnn_hidden_size", entity_ner_config.get("gru_hidden_size", 128))
        )
        self.entity_ner_rnn_num_layers = int(
            entity_ner_config.get("rnn_num_layers", entity_ner_config.get("gru_num_layers", 2))
        )
        self.entity_ner_labels = entity_ner_config.get("labels")

        embedding_config = config.get("embedding", {})
        self.embedding_provider = str(embedding_config.get("provider", "dashscope")).lower()
        self.embedding_model = str(embedding_config.get("model", "text-embedding-v3"))
        self.embedding_api_base = str(
            embedding_config.get(
                "api_base",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
        )
        self.embedding_dimension = int(embedding_config.get("dimension", 1024))
        self.embedding_timeout_seconds = int(embedding_config.get("timeout_seconds", 30))
        self.embedding_api_key = os.getenv(
            "EMBEDDING_API_KEY",
            os.getenv("DASHSCOPE_API_KEY", ""),
        )

        intent_config = config.get("intent", {})
        self.intent_engine = str(intent_config.get("engine", "rule")).lower()
        self.intent_max_intents = int(intent_config.get("max_intents", 5))
        self.intent_rule_confidence_threshold = float(
            intent_config.get("rule_confidence_threshold", 0.75)
        )
        self.intent_llm_confidence_threshold = float(
            intent_config.get("llm_confidence_threshold", 0.7)
        )
        self.intent_use_llm_when_rule_uncertain = bool(
            intent_config.get("use_llm_when_rule_uncertain", True)
        )
        self.intent_use_rule_when_llm_fails = bool(
            intent_config.get("use_rule_when_llm_fails", True)
        )
        intent_llm_config = intent_config.get("llm", {})
        self.intent_llm_provider = str(intent_llm_config.get("provider", "dashscope")).lower()
        self.intent_dashscope_api_base = intent_llm_config.get(
            "api_base", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.intent_dashscope_model = intent_llm_config.get("model", "qwen-plus")
        self.intent_llm_temperature = float(intent_llm_config.get("temperature", 0.0))
        self.intent_llm_max_tokens = int(intent_llm_config.get("max_tokens", 256))
        self.intent_llm_timeout_seconds = int(intent_llm_config.get("timeout_seconds", 12))
        self.intent_llm_enable_thinking = _optional_bool(
            intent_llm_config.get("enable_thinking", False)
        )
        self.intent_dashscope_api_key = os.getenv(
            "INTENT_DASHSCOPE_API_KEY", os.getenv("DASHSCOPE_API_KEY", "")
        )

        clinical_context_config = config.get("clinical_context", {})
        self.clinical_context_enabled = bool(
            clinical_context_config.get("enabled", True)
        )
        self.clinical_context_provider = str(
            clinical_context_config.get("provider", "dashscope")
        ).lower()
        self.clinical_context_dashscope_api_base = clinical_context_config.get(
            "api_base", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.clinical_context_model = clinical_context_config.get(
            "model", "qwen3.5-flash"
        )
        self.clinical_context_temperature = float(
            clinical_context_config.get("temperature", 0.0)
        )
        self.clinical_context_max_tokens = int(
            clinical_context_config.get("max_tokens", 512)
        )
        self.clinical_context_timeout_seconds = int(
            clinical_context_config.get("timeout_seconds", 30)
        )
        self.clinical_context_enable_thinking = _optional_bool(
            clinical_context_config.get("enable_thinking", False)
        )
        self.clinical_context_api_key = os.getenv(
            "CLINICAL_CONTEXT_API_KEY", os.getenv("DASHSCOPE_API_KEY", "")
        )

        disease_resolution_config = config.get("disease_resolution", {})
        self.disease_confidence_threshold = float(
            disease_resolution_config.get("confidence_threshold", 0.85)
        )
        self.disease_possible_confidence_threshold = float(
            disease_resolution_config.get("possible_confidence_threshold", 0.55)
        )
        self.disease_top_gap_threshold = float(
            disease_resolution_config.get("top_gap_threshold", 0.15)
        )
        self.disease_min_symptoms_for_inference = int(
            disease_resolution_config.get("min_symptoms_for_inference", 2)
        )
        self.disease_ask_follow_up_when_below_threshold = bool(
            disease_resolution_config.get("ask_follow_up_when_below_threshold", True)
        )
        self.disease_max_follow_up_turns = int(
            disease_resolution_config.get("max_follow_up_turns", 3)
        )
        self.disease_possible_candidate_limit = int(
            disease_resolution_config.get("possible_candidate_limit", 3)
        )
        self.disease_candidate_limit = int(
            disease_resolution_config.get("candidate_limit", 5)
        )

        llm_config = config.get("llm", {})
        self.llm_provider = llm_config.get("provider", "dashscope").lower()
        self.dashscope_api_base = llm_config.get(
            "api_base", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.dashscope_model = llm_config.get("model", "qwen-plus")
        ollama_config = llm_config.get("ollama", {})
        self.ollama_api_base = ollama_config.get("api_base", "http://localhost:11434")
        self.ollama_model = ollama_config.get("model", "qwen3:8b")
        self.llm_temperature = float(llm_config.get("temperature", 0.2))
        self.llm_max_tokens = int(llm_config.get("max_tokens", 1024))
        self.llm_timeout_seconds = int(llm_config.get("timeout_seconds", 90))
        self.llm_enable_thinking = _optional_bool(llm_config.get("enable_thinking"))
        self.dashscope_api_key = os.getenv("DASHSCOPE_API_KEY", "")

        paths_config = config.get("paths", {})
        self.entity_dir = _resolve_path(
            paths_config.get("entity_dir", "../data/ent_aug"), backend_root
        )

        auth_config = config.get("auth", {})
        self.secret_key = os.getenv("SECRET_KEY", "change-this-in-production")
        self.token_ttl_minutes = int(auth_config.get("token_ttl_minutes", 480))

        logging_config = config.get("logging", {})
        self.log_level = str(logging_config.get("level", "INFO")).upper()
        self.log_dir = _resolve_path(logging_config.get("directory", "logs"), backend_root)
        self.log_file_enabled = bool(logging_config.get("file_enabled", True))
        self.log_retention_days = int(logging_config.get("retention_days", 14))
        self.access_log_enabled = bool(logging_config.get("access_log_enabled", True))
        self.chat_trace_enabled = bool(logging_config.get("chat_trace_enabled", False))
        self.chat_trace_file = str(logging_config.get("chat_trace_file", "chat_trace.jsonl"))
        self.chat_trace_max_chars = int(logging_config.get("chat_trace_max_chars", 4000))

        cors_config = config.get("cors", {})
        self.cors_origins = cors_config.get(
            "origins", ["http://localhost:5173", "http://127.0.0.1:5173"]
        )


settings = Settings()
