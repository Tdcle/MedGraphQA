import logging
import re
import threading
import unicodedata
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
from typing import Iterable, List, Sequence

from elasticsearch import Elasticsearch
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from app.services.embedding_service import EntityEmbeddingService
from app.services.entity_ner import NullEntityMentionExtractor, RobertaRnnEntityMentionExtractor
from app.services.operation_log import log_operation


logger = logging.getLogger("medgraphqa.entity_search")


_NORMALIZE_REMOVE_RE = re.compile(r"[\s\u3000,，.。?？!！;；:：、\"'“”‘’()（）\[\]【】{}<>《》/\\|+-]+")
_TEXT_CHUNK_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]+")

_STOP_TERMS = {
    "我",
    "我们",
    "本人",
    "最近",
    "现在",
    "有点",
    "感觉",
    "觉得",
    "请问",
    "一下",
    "这个",
    "那个",
    "什么",
    "怎么",
    "怎么办",
    "为什么",
    "可以",
    "能不能",
    "是不是",
    "需要",
    "应该",
    "要",
    "吃",
    "药",
    "吃药",
    "吃什么药",
    "治疗",
    "检查",
    "挂号",
    "挂什么科",
    "哪个科",
}

_STOP_PHRASES = sorted(_STOP_TERMS, key=len, reverse=True)

_INTENT_EXPECTED_TYPES = {
    "drug_producer": ["药品"],
    "disease_symptom": ["疾病"],
    "disease_drugs": ["疾病", "疾病症状"],
    "disease_do_eat": ["疾病", "疾病症状"],
    "disease_not_eat": ["疾病", "疾病症状"],
    "disease_check": ["疾病", "疾病症状"],
    "disease_department": ["疾病", "疾病症状"],
    "disease_cure_way": ["疾病", "疾病症状"],
    "disease_acompany": ["疾病"],
    "disease_desc": ["疾病", "疾病症状"],
    "disease_cause": ["疾病"],
    "disease_prevent": ["疾病"],
    "disease_cycle": ["疾病"],
    "disease_prob": ["疾病"],
    "disease_population": ["疾病"],
}

_ALIAS_TYPE_WEIGHT = {
    "canonical": 6.0,
    "manual": 5.5,
    "synonym": 5.0,
    "colloquial": 5.0,
    "abbr": 4.5,
    "generated": 3.0,
    "typo": 2.0,
}

_MULTI_ENTITY_TYPES = {"疾病症状"}
_SYMPTOM_EQUIVALENTS = {
    "上腹部疼痛": "腹痛",
    "腹部疼痛": "腹痛",
    "中上腹持续性疼痛": "腹痛",
    "下腹部疼痛": "腹痛",
    "拉肚子": "腹泻",
}


@dataclass
class EntityCandidate:
    entity_id: int
    alias_id: int | None
    canonical_name: str
    entity_type: str
    matched_alias: str
    normalized_alias: str
    alias_type: str
    confidence: float
    source: str
    mention: str
    match_method: str
    score: float

    def to_log_dict(self) -> dict:
        data = asdict(self)
        data["score"] = round(float(self.score), 4)
        data["confidence"] = round(float(self.confidence), 4)
        return data


def normalize_entity_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value or "").casefold()
    return _NORMALIZE_REMOVE_RE.sub("", normalized)


def expected_types_for_intents(intents: Sequence[str]) -> list[str]:
    expected: list[str] = []
    for intent in intents:
        for entity_type in _INTENT_EXPECTED_TYPES.get(intent, []):
            if entity_type not in expected:
                expected.append(entity_type)
    return expected


def _is_noise_term(term: str) -> bool:
    normalized = normalize_entity_text(term)
    if len(normalized) < 2:
        return True
    if normalized in _STOP_TERMS:
        return True
    if normalized.isdigit():
        return True
    return False


def _remove_stop_phrases(text: str) -> str:
    result = text
    for phrase in _STOP_PHRASES:
        result = result.replace(phrase, " ")
    return result


def generate_search_terms(query: str, limit: int) -> list[str]:
    text = unicodedata.normalize("NFKC", query or "").strip()
    if not text:
        return []

    terms: OrderedDict[str, str] = OrderedDict()

    def add(term: str) -> None:
        clean = term.strip()
        normalized = normalize_entity_text(clean)
        if not normalized or _is_noise_term(clean) or normalized in terms:
            return
        terms[normalized] = clean

    add(text)
    for chunk in _TEXT_CHUNK_RE.findall(_remove_stop_phrases(text)):
        add(chunk)

    for chunk in _TEXT_CHUNK_RE.findall(text[:120]):
        max_len = min(10, len(chunk))
        for size in range(max_len, 1, -1):
            for start in range(0, len(chunk) - size + 1):
                add(chunk[start : start + size])
                if len(terms) >= limit:
                    return list(terms.values())

    return list(terms.values())[:limit]


class PostgresEntityRepository:
    def __init__(
        self,
        dsn: str,
        min_size: int,
        max_size: int,
        timeout_seconds: int,
    ) -> None:
        self._pool = ConnectionPool(
            conninfo=dsn,
            min_size=min_size,
            max_size=max_size,
            timeout=timeout_seconds,
            open=False,
        )
        self._open_lock = threading.Lock()
        self._opened = False

    def _ensure_open(self) -> None:
        if self._opened:
            return
        with self._open_lock:
            if not self._opened:
                self._pool.open(wait=False)
                self._opened = True

    @staticmethod
    def _normalize_uuid(value: str) -> str:
        try:
            return str(uuid.UUID(value))
        except ValueError as exc:
            raise ValueError("invalid conversation id") from exc

    def close(self) -> None:
        if self._opened:
            self._pool.close()
            self._opened = False

    def ping(self) -> bool:
        try:
            self._ensure_open()
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            return True
        except Exception:
            logger.exception("postgres health check failed")
            return False

    def find_exact_aliases(
        self,
        terms: Iterable[str],
        expected_types: Sequence[str],
        limit: int,
    ) -> list[EntityCandidate]:
        normalized_to_mention: OrderedDict[str, str] = OrderedDict()
        for term in terms:
            normalized = normalize_entity_text(term)
            if normalized and normalized not in normalized_to_mention:
                normalized_to_mention[normalized] = term
        if not normalized_to_mention:
            return []

        type_filter = ""
        params: list[object] = [list(normalized_to_mention.keys())]
        if expected_types:
            type_filter = "AND t.name = ANY(%s)"
            params.append(list(expected_types))
        params.append(limit)

        sql = f"""
            SELECT
                e.id AS entity_id,
                a.id AS alias_id,
                e.name AS canonical_name,
                t.name AS entity_type,
                a.alias AS matched_alias,
                a.normalized_alias,
                a.alias_type,
                a.confidence,
                a.source
            FROM entity_alias a
            JOIN kg_entity e ON e.id = a.entity_id
            JOIN entity_type t ON t.id = e.type_id
            WHERE a.is_active = TRUE
              AND e.is_active = TRUE
              AND a.normalized_alias = ANY(%s)
              {type_filter}
            ORDER BY
                char_length(a.normalized_alias) DESC,
                a.confidence DESC,
                e.name ASC
            LIMIT %s
        """

        self._ensure_open()
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        candidates: list[EntityCandidate] = []
        for row in rows:
            normalized_alias = str(row["normalized_alias"])
            candidates.append(
                EntityCandidate(
                    entity_id=int(row["entity_id"]),
                    alias_id=int(row["alias_id"]),
                    canonical_name=str(row["canonical_name"]),
                    entity_type=str(row["entity_type"]),
                    matched_alias=str(row["matched_alias"]),
                    normalized_alias=normalized_alias,
                    alias_type=str(row["alias_type"]),
                    confidence=float(row["confidence"]),
                    source=str(row["source"]),
                    mention=normalized_to_mention.get(normalized_alias, str(row["matched_alias"])),
                    match_method="postgres_exact",
                    score=0.0,
                )
            )
        return candidates

    def log_search(
        self,
        query: str,
        mention: str | None,
        expected_types: Sequence[str],
        candidates: Sequence[EntityCandidate],
        selected: EntityCandidate | None,
    ) -> None:
        try:
            self._ensure_open()
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO entity_search_log (
                            query,
                            mention,
                            expected_types,
                            candidates,
                            selected_entity_id,
                            selected_name,
                            selected_type,
                            score,
                            matched_alias,
                            match_method,
                            is_accepted
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            query,
                            mention,
                            list(expected_types),
                            Jsonb([candidate.to_log_dict() for candidate in candidates[:20]]),
                            selected.entity_id if selected else None,
                            selected.canonical_name if selected else None,
                            selected.entity_type if selected else None,
                            selected.score if selected else None,
                            selected.matched_alias if selected else None,
                            selected.match_method if selected else None,
                            selected is not None,
                        ),
                    )
        except Exception:
            logger.exception("failed to write entity search log")

    def log_disease_resolution(
        self,
        query: str,
        symptoms: Sequence[str],
        candidates: Sequence[dict],
        selected_disease: str | None,
        confidence: float | None,
        decision: str,
        follow_up_question: str | None,
    ) -> None:
        try:
            self._ensure_open()
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO disease_candidate_score_log (
                            query,
                            symptoms,
                            candidates,
                            selected_disease,
                            confidence,
                            decision,
                            follow_up_question
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            query,
                            Jsonb(list(symptoms)),
                            Jsonb(list(candidates)),
                            selected_disease,
                            confidence,
                            decision,
                            follow_up_question,
                        ),
                    )
        except Exception:
            logger.exception("failed to write disease resolution log")

    def ensure_chat_session(
        self,
        user_id: str,
        conversation_id: str | None,
        title: str,
    ) -> str:
        self._ensure_open()
        if conversation_id:
            session_id = self._normalize_uuid(conversation_id)
        else:
            session_id = str(uuid.uuid4())
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT id, user_id, title FROM chat_session WHERE id = %s AND is_active = TRUE",
                    (session_id,),
                )
                row = cur.fetchone()
                if row:
                    if str(row["user_id"]) != user_id:
                        raise ValueError("conversation does not belong to current user")
                    current_title = (row.get("title") or "").strip()
                    new_title = title.strip()[:255]
                    if new_title and current_title in {"", "新对话"}:
                        cur.execute(
                            """
                            UPDATE chat_session
                            SET title = %s, updated_at = now()
                            WHERE id = %s
                            """,
                            (new_title, session_id),
                        )
                    else:
                        cur.execute(
                            "UPDATE chat_session SET updated_at = now() WHERE id = %s",
                            (session_id,),
                        )
                    return session_id

                cur.execute(
                    """
                    INSERT INTO chat_session (id, user_id, title, updated_at)
                    VALUES (%s, %s, %s, now())
                    """,
                    (session_id, user_id, title[:255]),
                )
        return session_id

    def create_chat_session(self, user_id: str, title: str = "新对话") -> dict:
        self._ensure_open()
        session_id = str(uuid.uuid4())
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    INSERT INTO chat_session (id, user_id, title, updated_at)
                    VALUES (%s, %s, %s, now())
                    RETURNING id, title, created_at, updated_at
                    """,
                    (session_id, user_id, title[:255]),
                )
                row = cur.fetchone()
        return dict(row)

    def delete_chat_session(self, user_id: str, conversation_id: str) -> bool:
        self._ensure_open()
        conversation_id = self._normalize_uuid(conversation_id)
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE chat_session
                    SET is_active = FALSE, updated_at = now()
                    WHERE id = %s AND user_id = %s AND is_active = TRUE
                    """,
                    (conversation_id, user_id),
                )
                return cur.rowcount > 0

    def get_chat_state(self, user_id: str, conversation_id: str) -> dict:
        self._ensure_open()
        conversation_id = self._normalize_uuid(conversation_id)
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT st.state
                    FROM chat_session s
                    LEFT JOIN chat_session_state st ON st.session_id = s.id
                    WHERE s.id = %s AND s.user_id = %s AND s.is_active = TRUE
                    """,
                    (conversation_id, user_id),
                )
                row = cur.fetchone()
        if not row or not row.get("state"):
            return {}
        return dict(row["state"])

    def save_chat_state(
        self,
        user_id: str,
        conversation_id: str,
        state: dict,
    ) -> None:
        self._ensure_open()
        conversation_id = self._normalize_uuid(conversation_id)
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 1
                    FROM chat_session
                    WHERE id = %s AND user_id = %s AND is_active = TRUE
                    """,
                    (conversation_id, user_id),
                )
                if not cur.fetchone():
                    raise ValueError("conversation does not belong to current user")
                cur.execute(
                    """
                    INSERT INTO chat_session_state (session_id, state, updated_at)
                    VALUES (%s, %s, now())
                    ON CONFLICT (session_id) DO UPDATE SET
                        state = EXCLUDED.state,
                        updated_at = now()
                    """,
                    (conversation_id, Jsonb(state)),
                )
                cur.execute(
                    "UPDATE chat_session SET updated_at = now() WHERE id = %s",
                    (conversation_id,),
                )

    def add_chat_message(
        self,
        user_id: str,
        conversation_id: str,
        role: str,
        content: str,
        metadata: dict | None = None,
    ) -> None:
        self._ensure_open()
        conversation_id = self._normalize_uuid(conversation_id)
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 1
                    FROM chat_session
                    WHERE id = %s AND user_id = %s AND is_active = TRUE
                    """,
                    (conversation_id, user_id),
                )
                if not cur.fetchone():
                    raise ValueError("conversation does not belong to current user")
                cur.execute(
                    """
                    INSERT INTO chat_message (session_id, role, content, metadata)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (conversation_id, role, content, Jsonb(metadata or {})),
                )

    def list_chat_sessions(self, user_id: str, limit: int = 50) -> list[dict]:
        self._ensure_open()
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT
                        s.id,
                        s.title,
                        s.created_at,
                        s.updated_at,
                        st.state->>'last_answer' AS last_answer,
                        COALESCE((st.state->>'awaiting_user_clarification')::boolean, FALSE)
                            AS awaiting_user_clarification
                    FROM chat_session s
                    LEFT JOIN chat_session_state st ON st.session_id = s.id
                    WHERE s.user_id = %s AND s.is_active = TRUE
                    ORDER BY s.updated_at DESC
                    LIMIT %s
                    """,
                    (user_id, limit),
                )
                rows = cur.fetchall()
        return [dict(row) for row in rows]

    def get_chat_messages(self, user_id: str, conversation_id: str) -> list[dict]:
        self._ensure_open()
        conversation_id = self._normalize_uuid(conversation_id)
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT 1
                    FROM chat_session
                    WHERE id = %s AND user_id = %s AND is_active = TRUE
                    """,
                    (conversation_id, user_id),
                )
                if not cur.fetchone():
                    raise ValueError("conversation does not belong to current user")
                cur.execute(
                    """
                    SELECT id, role, content, metadata, created_at
                    FROM chat_message
                    WHERE session_id = %s
                    ORDER BY created_at ASC, id ASC
                    """,
                    (conversation_id,),
                )
                rows = cur.fetchall()
        return [dict(row) for row in rows]


class ElasticsearchEntityIndex:
    def __init__(
        self,
        hosts: Sequence[str],
        index_name: str,
        username: str,
        password: str,
        request_timeout_seconds: int,
    ) -> None:
        basic_auth = (username, password) if username and password else None
        self.index_name = index_name
        self._client = Elasticsearch(
            list(hosts),
            basic_auth=basic_auth,
            request_timeout=request_timeout_seconds,
        )

    def ping(self) -> bool:
        try:
            return bool(self._client.ping())
        except Exception:
            logger.exception("elasticsearch health check failed")
            return False

    def search(
        self,
        term: str,
        expected_types: Sequence[str],
        size: int,
        min_score: float,
    ) -> list[EntityCandidate]:
        body = self._lexical_body(term, expected_types, size, min_score)
        response = self._client.search(index=self.index_name, body=body)
        return self._candidates_from_hits(
            response.get("hits", {}).get("hits", []),
            mention=term,
            match_method="elasticsearch",
        )

    def search_many(
        self,
        terms: Sequence[str],
        expected_types: Sequence[str],
        size: int,
        min_score: float,
    ) -> tuple[list[EntityCandidate], int, int]:
        clean_terms = [term for term in terms if term]
        if not clean_terms:
            return [], 0, 0
        searches: list[dict] = []
        for term in clean_terms:
            searches.append({})
            searches.append(self._lexical_body(term, expected_types, size, min_score))
        response = self._client.msearch(index=self.index_name, searches=searches)
        candidates: list[EntityCandidate] = []
        success_count = 0
        failed_count = 0
        for term, item in zip(clean_terms, response.get("responses", [])):
            if item.get("error"):
                failed_count += 1
                logger.warning(
                    "operation=entity_search.elasticsearch_msearch_item status=error term=%s error=%s",
                    term,
                    item.get("error"),
                )
                continue
            success_count += 1
            candidates.extend(
                self._candidates_from_hits(
                    item.get("hits", {}).get("hits", []),
                    mention=term,
                    match_method="elasticsearch",
                )
            )
        return candidates, success_count, failed_count

    @staticmethod
    def _candidates_from_hits(
        hits: Sequence[dict],
        mention: str,
        match_method: str,
    ) -> list[EntityCandidate]:
        candidates: list[EntityCandidate] = []
        for hit in hits:
            source = hit.get("_source", {})
            candidates.append(
                EntityCandidate(
                    entity_id=int(source["entity_id"]),
                    alias_id=int(source["alias_id"]) if source.get("alias_id") is not None else None,
                    canonical_name=str(source["canonical_name"]),
                    entity_type=str(source["entity_type"]),
                    matched_alias=str(source["alias"]),
                    normalized_alias=str(
                        source.get("normalized_alias") or normalize_entity_text(str(source["alias"]))
                    ),
                    alias_type=str(source.get("alias_type") or "unknown"),
                    confidence=float(source.get("confidence") or 0.0),
                    source=str(source.get("source") or "elasticsearch"),
                    mention=mention,
                    match_method=match_method,
                    score=float(hit.get("_score") or 0.0),
                )
            )
        return candidates

    @staticmethod
    def _lexical_body(
        term: str,
        expected_types: Sequence[str],
        size: int,
        min_score: float,
    ) -> dict:
        filters: list[dict] = [{"term": {"is_active": True}}]
        if expected_types:
            filters.append({"terms": {"entity_type": list(expected_types)}})

        return {
            "size": size,
            "min_score": min_score,
            "_source": [
                "alias_id",
                "entity_id",
                "alias",
                "canonical_name",
                "entity_type",
                "alias_type",
                "confidence",
                "source",
                "normalized_alias",
            ],
            "query": {
                "bool": {
                    "filter": filters,
                    "should": [
                        {"term": {"alias.keyword": {"value": term, "boost": 30}}},
                        {"term": {"canonical_name.keyword": {"value": term, "boost": 28}}},
                        {"match_phrase": {"alias": {"query": term, "boost": 12}}},
                        {"match_phrase": {"canonical_name": {"query": term, "boost": 10}}},
                        {
                            "multi_match": {
                                "query": term,
                                "type": "best_fields",
                                "fields": [
                                    "alias^5",
                                    "canonical_name^4",
                                    "search_text^2",
                                    "pinyin",
                                ],
                            }
                        },
                    ],
                    "minimum_should_match": 1,
                }
            },
        }

    def search_vector(
        self,
        query_vector: Sequence[float],
        mention: str,
        expected_types: Sequence[str],
        size: int,
        min_score: float,
    ) -> list[EntityCandidate]:
        if not query_vector:
            return []

        filters: list[dict] = [
            {"term": {"is_active": True}},
            {"exists": {"field": "embedding"}},
        ]
        if expected_types:
            filters.append({"terms": {"entity_type": list(expected_types)}})

        body = {
            "size": size,
            "min_score": min_score,
            "_source": [
                "alias_id",
                "entity_id",
                "alias",
                "canonical_name",
                "entity_type",
                "alias_type",
                "confidence",
                "source",
                "normalized_alias",
            ],
            "query": {
                "script_score": {
                    "query": {"bool": {"filter": filters}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": list(query_vector)},
                    },
                }
            },
        }

        response = self._client.search(index=self.index_name, body=body)
        candidates: list[EntityCandidate] = []
        for hit in response.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            candidates.append(
                EntityCandidate(
                    entity_id=int(source["entity_id"]),
                    alias_id=int(source["alias_id"]) if source.get("alias_id") is not None else None,
                    canonical_name=str(source["canonical_name"]),
                    entity_type=str(source["entity_type"]),
                    matched_alias=str(source["alias"]),
                    normalized_alias=str(
                        source.get("normalized_alias") or normalize_entity_text(str(source["alias"]))
                    ),
                    alias_type=str(source.get("alias_type") or "unknown"),
                    confidence=float(source.get("confidence") or 0.0),
                    source=str(source.get("source") or "elasticsearch"),
                    mention=mention,
                    match_method="elasticsearch_vector",
                    score=float(hit.get("_score") or 0.0),
                )
            )
        return candidates


class EntityNormalizer:
    def __init__(
        self,
        repository: PostgresEntityRepository,
        search_index: ElasticsearchEntityIndex,
        embedding_service: EntityEmbeddingService,
        max_entities: int,
        exact_terms_limit: int,
        elastic_terms_limit: int,
        elastic_top_k: int,
        elastic_min_score: float,
        vector_enabled: bool,
        vector_top_k: int,
        vector_min_score: float,
        rrf_k: int,
        exact_rrf_weight: float,
        elastic_rrf_weight: float,
        vector_rrf_weight: float,
        mention_extractor: RobertaRnnEntityMentionExtractor | NullEntityMentionExtractor | None = None,
    ) -> None:
        self.repository = repository
        self.search_index = search_index
        self.embedding_service = embedding_service
        self.max_entities = max_entities
        self.exact_terms_limit = exact_terms_limit
        self.elastic_terms_limit = elastic_terms_limit
        self.elastic_top_k = elastic_top_k
        self.elastic_min_score = elastic_min_score
        self.vector_enabled = vector_enabled
        self.vector_top_k = vector_top_k
        self.vector_min_score = vector_min_score
        self.rrf_k = rrf_k
        self.exact_rrf_weight = exact_rrf_weight
        self.elastic_rrf_weight = elastic_rrf_weight
        self.vector_rrf_weight = vector_rrf_weight
        self.mention_extractor = mention_extractor or NullEntityMentionExtractor()

    def resolve(self, query: str, intents: Sequence[str]) -> list[EntityCandidate]:
        expected_types = expected_types_for_intents(intents)
        with log_operation(
            logger,
            "entity_search.resolve",
            query_len=len(query),
            expected_types=",".join(expected_types),
        ) as result:
            mention_terms = self.mention_extractor.extract_terms(query, expected_types)
            result["ner_term_count"] = len(mention_terms)
            if mention_terms:
                candidates = self.resolve_terms(
                    mention_terms,
                    query=query,
                    intents=intents,
                    allow_vector=False,
                )
                result["candidate_count"] = len(candidates)
                result["source"] = "ner_terms"
                if candidates:
                    return candidates
            candidates = self.resolve_terms([query], query=query, intents=intents, allow_vector=True)
            result["candidate_count"] = len(candidates)
            result["source"] = "query_fallback"
            return candidates

    @staticmethod
    def expected_types(intents: Sequence[str]) -> list[str]:
        return expected_types_for_intents(intents)

    def extract_mention_terms(
        self,
        query: str,
        expected_types: Sequence[str],
    ) -> list[str]:
        return self.mention_extractor.extract_terms(query, expected_types)

    def resolve_terms(
        self,
        terms: Sequence[str],
        query: str,
        intents: Sequence[str],
        allow_vector: bool = False,
    ) -> list[EntityCandidate]:
        with log_operation(
            logger,
            "entity_search.resolve_terms",
            term_count=len(terms),
            allow_vector=allow_vector,
        ) as result:
            expected_types = expected_types_for_intents(intents)
            exact_terms = self._terms_for_lookup(terms, self.exact_terms_limit)
            exact_candidates: list[EntityCandidate] = []
            elastic_candidates: list[EntityCandidate] = []
            vector_candidates: list[EntityCandidate] = []

            try:
                exact_candidates = self.repository.find_exact_aliases(
                    terms=exact_terms,
                    expected_types=expected_types,
                    limit=max(self.max_entities * 10, 20),
                )
            except Exception:
                logger.exception("operation=entity_search.postgres_exact status=error")

            elastic_terms = self._terms_for_lookup(terms, self.elastic_terms_limit)
            elastic_success_count = 0
            elastic_failed_count = 0
            try:
                with log_operation(
                    logger,
                    "entity_search.elasticsearch_batch",
                    request_count=len(elastic_terms),
                ) as elastic_result:
                    (
                        elastic_candidates,
                        elastic_success_count,
                        elastic_failed_count,
                    ) = self.search_index.search_many(
                        terms=elastic_terms,
                        expected_types=expected_types,
                        size=self.elastic_top_k,
                        min_score=self.elastic_min_score,
                    )
                    elastic_result["success_count"] = elastic_success_count
                    elastic_result["failed_count"] = elastic_failed_count
                    elastic_result["candidate_count"] = len(elastic_candidates)
            except Exception:
                elastic_failed_count = len(elastic_terms)
                logger.exception("operation=entity_search.elasticsearch_batch status=error")

            if self.vector_enabled and allow_vector:
                try:
                    query_vector = self.embedding_service.embed_one(query)
                    if query_vector:
                        vector_candidates = self.search_index.search_vector(
                            query_vector=query_vector,
                            mention=query,
                            expected_types=expected_types,
                            size=self.vector_top_k,
                            min_score=self.vector_min_score,
                        )
                except Exception:
                    logger.exception("operation=entity_search.elasticsearch_vector status=error")

            candidates = self._rrf_fuse(
                [
                    (exact_candidates, self.exact_rrf_weight),
                    (elastic_candidates, self.elastic_rrf_weight),
                    (vector_candidates, self.vector_rrf_weight),
                ]
            )

            ranked = self._rank_candidates(candidates, expected_types)
            ranked = self._drop_unreliable_cross_type_diseases(ranked)
            ranked = self._filter_noisy_candidates(ranked)
            ranked = self._prefer_specific_symptoms(ranked)
            ranked = self._drop_symptom_substring_diseases(ranked)
            selected = self._select_entities(ranked)
            selected = self._prefer_specific_symptoms(selected)
            result["exact_count"] = len(exact_candidates)
            result["elastic_count"] = len(elastic_candidates)
            result["elastic_request_count"] = len(elastic_terms)
            result["elastic_success_count"] = elastic_success_count
            result["elastic_failed_count"] = elastic_failed_count
            result["vector_count"] = len(vector_candidates)
            result["ranked_count"] = len(ranked)
            result["selected_count"] = len(selected)

            try:
                self.repository.log_search(
                    query=query,
                    mention=selected[0].mention if selected else None,
                    expected_types=expected_types,
                    candidates=ranked,
                    selected=selected[0] if selected else None,
                )
            except Exception:
                logger.exception("operation=entity_search.log_search status=error")

            return selected

    @staticmethod
    def _terms_for_lookup(terms: Sequence[str], limit: int) -> list[str]:
        result: list[str] = []
        for term in terms:
            for item in generate_search_terms(term, limit):
                if item not in result:
                    result.append(item)
                if len(result) >= limit:
                    return result
        return result

    def _rrf_fuse(
        self,
        ranked_lists: Sequence[tuple[Sequence[EntityCandidate], float]],
    ) -> list[EntityCandidate]:
        by_entity: dict[tuple[int, str], EntityCandidate] = {}
        scores: dict[tuple[int, str], float] = {}
        methods: dict[tuple[int, str], list[str]] = {}

        for candidates, weight in ranked_lists:
            seen_in_list: set[tuple[int, str]] = set()
            for rank, candidate in enumerate(candidates, 1):
                key = (candidate.entity_id, candidate.entity_type)
                if key in seen_in_list:
                    continue
                seen_in_list.add(key)
                scores[key] = scores.get(key, 0.0) + weight / (self.rrf_k + rank)
                methods.setdefault(key, [])
                if candidate.match_method not in methods[key]:
                    methods[key].append(candidate.match_method)

                current = by_entity.get(key)
                if current is None or self._prefer_candidate(candidate, current):
                    by_entity[key] = replace(candidate)

        fused: list[EntityCandidate] = []
        for key, candidate in by_entity.items():
            candidate.score = scores[key] * 1000
            if len(methods.get(key, [])) > 1:
                candidate.match_method = "rrf:" + "+".join(methods[key])
            fused.append(candidate)
        return sorted(fused, key=lambda item: -item.score)

    @staticmethod
    def _prefer_candidate(candidate: EntityCandidate, current: EntityCandidate) -> bool:
        if candidate.match_method == "postgres_exact" and current.match_method != "postgres_exact":
            return True
        if candidate.confidence != current.confidence:
            return candidate.confidence > current.confidence
        return candidate.score > current.score

    def _rank_candidates(
        self,
        candidates: Sequence[EntityCandidate],
        expected_types: Sequence[str],
    ) -> list[EntityCandidate]:
        ranked_by_entity: dict[tuple[int, str], EntityCandidate] = {}
        for candidate in candidates:
            alias_weight = _ALIAS_TYPE_WEIGHT.get(candidate.alias_type, 1.0)
            type_weight = 4.0 if candidate.entity_type in expected_types else 0.0
            length_weight = min(len(candidate.normalized_alias), 12) / 3
            method_weight = 50.0 if "postgres_exact" in candidate.match_method else 0.0
            candidate.score = (
                float(candidate.score)
                + method_weight
                + alias_weight
                + type_weight
                + length_weight
                + float(candidate.confidence) * 10
            )
            key = (candidate.entity_id, candidate.entity_type)
            current = ranked_by_entity.get(key)
            if current is None or candidate.score > current.score:
                ranked_by_entity[key] = candidate

        return sorted(
            ranked_by_entity.values(),
            key=lambda item: (
                -item.score,
                -len(item.normalized_alias),
                item.entity_type,
                item.canonical_name,
            ),
        )

    @staticmethod
    def _drop_unreliable_cross_type_diseases(
        candidates: Sequence[EntityCandidate],
    ) -> list[EntityCandidate]:
        symptom_names = {
            item.canonical_name
            for item in candidates
            if item.entity_type == "疾病症状"
        }
        if not symptom_names:
            return list(candidates)
        result: list[EntityCandidate] = []
        for item in candidates:
            if (
                item.entity_type == "疾病"
                and item.canonical_name in symptom_names
                and "postgres_exact" not in item.match_method
            ):
                continue
            result.append(item)
        return result

    def _select_entities(self, candidates: Sequence[EntityCandidate]) -> list[EntityCandidate]:
        selected: list[EntityCandidate] = []
        used_single_types: set[str] = set()
        used_entities: set[tuple[int, str]] = set()
        used_symptom_groups: set[str] = set()
        for candidate in candidates:
            entity_key = (candidate.entity_id, candidate.entity_type)
            if entity_key in used_entities:
                continue
            if candidate.entity_type == "疾病症状":
                symptom_group = self._symptom_group(candidate.canonical_name)
                if symptom_group in used_symptom_groups:
                    continue
            if (
                candidate.entity_type not in _MULTI_ENTITY_TYPES
                and candidate.entity_type in used_single_types
            ):
                continue
            selected.append(candidate)
            used_entities.add(entity_key)
            if candidate.entity_type == "疾病症状":
                used_symptom_groups.add(self._symptom_group(candidate.canonical_name))
            if candidate.entity_type not in _MULTI_ENTITY_TYPES:
                used_single_types.add(candidate.entity_type)
            if len(selected) >= self.max_entities:
                break
        return selected

    def _prefer_specific_symptoms(
        self,
        candidates: Sequence[EntityCandidate],
    ) -> list[EntityCandidate]:
        symptoms = [
            item for item in candidates if item.entity_type == "疾病症状"
        ]
        if len(symptoms) < 2:
            return list(candidates)

        dropped: set[tuple[int, str, str]] = set()
        for candidate in symptoms:
            candidate_name = normalize_entity_text(candidate.canonical_name)
            if not candidate_name:
                continue
            for other in symptoms:
                if other is candidate:
                    continue
                other_name = normalize_entity_text(other.canonical_name)
                if not self._is_more_specific_symptom(
                    specific=other_name,
                    generic=candidate_name,
                ):
                    continue
                dropped.add(
                    (
                        candidate.entity_id,
                        candidate.entity_type,
                        candidate.canonical_name,
                    )
                )
                break

        if not dropped:
            return list(candidates)

        return [
            item
            for item in candidates
            if (
                item.entity_id,
                item.entity_type,
                item.canonical_name,
            )
            not in dropped
        ]

    @staticmethod
    def _is_more_specific_symptom(specific: str, generic: str) -> bool:
        if not specific or not generic or specific == generic:
            return False
        if len(generic) < 2 or len(specific) - len(generic) < 2:
            return False
        return generic in specific

    def _drop_symptom_substring_diseases(
        self,
        candidates: Sequence[EntityCandidate],
    ) -> list[EntityCandidate]:
        symptom_names = [
            normalize_entity_text(item.canonical_name)
            for item in candidates
            if item.entity_type == "疾病症状"
        ]
        if not symptom_names:
            return list(candidates)

        result: list[EntityCandidate] = []
        for item in candidates:
            if item.entity_type != "疾病":
                result.append(item)
                continue
            disease_name = normalize_entity_text(item.canonical_name)
            if not disease_name:
                result.append(item)
                continue
            has_more_specific_symptom = any(
                self._is_more_specific_symptom(symptom, disease_name)
                for symptom in symptom_names
            )
            mention = normalize_entity_text(item.mention)
            matched_alias = normalize_entity_text(item.matched_alias)
            explicit_disease_mention = mention == disease_name or matched_alias == mention
            if has_more_specific_symptom and not explicit_disease_mention:
                continue
            result.append(item)
        return result

    def _filter_noisy_candidates(
        self,
        candidates: Sequence[EntityCandidate],
    ) -> list[EntityCandidate]:
        reliable_symptoms = {
            self._symptom_group(item.canonical_name)
            for item in candidates
            if item.entity_type == "疾病症状" and self._is_reliable_mention(item)
        }
        result: list[EntityCandidate] = []
        for item in candidates:
            if item.entity_type == "疾病症状":
                if item.match_method == "elasticsearch_vector":
                    continue
                if not self._is_reliable_mention(item):
                    continue
                group = self._symptom_group(item.canonical_name)
                if group in reliable_symptoms or "postgres_exact" in item.match_method:
                    result.append(item)
                continue

            if item.entity_type == "疾病":
                if item.match_method == "elasticsearch_vector":
                    continue
                if item.match_method == "elasticsearch" and not self._is_reliable_mention(item):
                    continue
                if self._symptom_group(item.canonical_name) in reliable_symptoms:
                    continue
            result.append(item)
        return result

    @staticmethod
    def _is_reliable_mention(candidate: EntityCandidate) -> bool:
        if "postgres_exact" in candidate.match_method:
            return True
        if candidate.match_method.startswith("rrf:") and "elasticsearch_vector" not in candidate.match_method:
            return True
        if candidate.match_method == "elasticsearch":
            mention = normalize_entity_text(candidate.mention)
            alias = normalize_entity_text(candidate.matched_alias)
            return bool(alias and alias in mention)
        return False

    @staticmethod
    def _symptom_group(name: str) -> str:
        for source, target in _SYMPTOM_EQUIVALENTS.items():
            if source == name or source in name:
                return target
        return name
