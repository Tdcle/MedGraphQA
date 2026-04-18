import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import psycopg
from elasticsearch import Elasticsearch, helpers
from psycopg.rows import dict_row
from dotenv import load_dotenv


BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent
load_dotenv(dotenv_path=BACKEND_ROOT / ".env", override=False)
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.entity_search import normalize_entity_text  # noqa: E402
from app.core.security import hash_password  # noqa: E402
from app.services.embedding_service import EntityEmbeddingService  # noqa: E402


ENTITY_TYPES = [
    {
        "code": "disease",
        "name": "疾病",
        "neo4j_label": "疾病",
        "source_file": "疾病.txt",
        "priority": 10,
    },
    {
        "code": "symptom",
        "name": "疾病症状",
        "neo4j_label": "疾病症状",
        "source_file": "疾病症状.txt",
        "priority": 20,
    },
    {
        "code": "drug",
        "name": "药品",
        "neo4j_label": "药品",
        "source_file": "药品.txt",
        "priority": 30,
    },
    {
        "code": "producer",
        "name": "药品商",
        "neo4j_label": "药品商",
        "source_file": "药品商.txt",
        "priority": 40,
    },
    {
        "code": "check",
        "name": "检查项目",
        "neo4j_label": "检查项目",
        "source_file": "检查项目.txt",
        "priority": 50,
    },
    {
        "code": "treatment",
        "name": "治疗方法",
        "neo4j_label": "治疗方法",
        "source_file": "治疗方法.txt",
        "priority": 60,
    },
    {
        "code": "department",
        "name": "科目",
        "neo4j_label": "科目",
        "source_file": "科目.txt",
        "priority": 70,
    },
    {
        "code": "food",
        "name": "食物",
        "neo4j_label": "食物",
        "source_file": "食物.txt",
        "priority": 80,
    },
]


INDEX_BODY = {
    "settings": {},
    "mappings": {
        "properties": {
            "alias_id": {"type": "long"},
            "entity_id": {"type": "long"},
            "alias": {
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_smart",
                "fields": {"keyword": {"type": "keyword"}},
            },
            "normalized_alias": {"type": "keyword"},
            "canonical_name": {
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_smart",
                "fields": {"keyword": {"type": "keyword"}},
            },
            "entity_type": {"type": "keyword"},
            "alias_type": {"type": "keyword"},
            "confidence": {"type": "float"},
            "source": {"type": "keyword"},
            "search_text": {
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_smart",
            },
            "pinyin": {"type": "text"},
            "is_active": {"type": "boolean"},
            "updated_at": {"type": "date"},
        }
    },
}


def build_index_body(vector_dimension: int | None = None) -> dict:
    body = json.loads(json.dumps(INDEX_BODY))
    if vector_dimension:
        body["mappings"]["properties"]["embedding"] = {
            "type": "dense_vector",
            "dims": vector_dimension,
            "index": True,
            "similarity": "cosine",
        }
    return body


def load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def postgres_dsn(args: argparse.Namespace, config: dict) -> str:
    return (
        args.dsn
        or os.getenv("POSTGRES_DSN")
        or config.get("postgres", {}).get(
            "dsn", "postgresql://postgres:postgres@localhost:5432/medgraphqa"
        )
    )


def elastic_hosts(args: argparse.Namespace, config: dict) -> list[str]:
    if args.es_hosts:
        return [item.strip() for item in args.es_hosts.split(",") if item.strip()]
    if os.getenv("ELASTICSEARCH_HOSTS"):
        return [item.strip() for item in os.getenv("ELASTICSEARCH_HOSTS", "").split(",") if item.strip()]
    return config.get("elasticsearch", {}).get("hosts", ["http://localhost:9200"])


def elastic_index(args: argparse.Namespace, config: dict) -> str:
    return args.es_index or config.get("elasticsearch", {}).get("index", "medical_entity_alias")


def elastic_client(args: argparse.Namespace, config: dict) -> Elasticsearch:
    username = args.es_username or os.getenv("ELASTICSEARCH_USERNAME") or config.get("elasticsearch", {}).get("username", "")
    password = args.es_password or os.getenv("ELASTICSEARCH_PASSWORD") or config.get("elasticsearch", {}).get("password", "")
    basic_auth = (username, password) if username and password else None
    return Elasticsearch(elastic_hosts(args, config), basic_auth=basic_auth, request_timeout=30)


def ensure_ik_analyzers(client: Elasticsearch) -> None:
    try:
        client.indices.analyze(body={"analyzer": "ik_smart", "text": "test"})
        client.indices.analyze(body={"analyzer": "ik_max_word", "text": "test"})
    except Exception as exc:
        raise RuntimeError(
            "Elasticsearch IK analyzers are unavailable. Install analysis-ik in the "
            "Elasticsearch container, restart it, then recreate the index."
        ) from exc


def vector_enabled(args: argparse.Namespace, config: dict) -> bool:
    if getattr(args, "with_embeddings", False):
        return True
    return bool(config.get("entity_search", {}).get("vector_enabled", False))


def embedding_dimension(args: argparse.Namespace, config: dict) -> int:
    return int(
        args.embedding_dimension
        or config.get("embedding", {}).get("dimension", 1024)
    )


def embedding_service(args: argparse.Namespace, config: dict) -> EntityEmbeddingService:
    embedding_config = config.get("embedding", {})
    provider = args.embedding_provider or embedding_config.get("provider", "dashscope")
    api_base = args.embedding_api_base or embedding_config.get(
        "api_base", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    model = args.embedding_model or embedding_config.get("model", "text-embedding-v3")
    api_key = os.getenv("EMBEDDING_API_KEY", os.getenv("DASHSCOPE_API_KEY", ""))
    timeout_seconds = int(embedding_config.get("timeout_seconds", 30))
    return EntityEmbeddingService(
        provider=provider,
        model=model,
        api_base=api_base,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        enabled=True,
    )


def put_vector_mapping(client: Elasticsearch, index_name: str, dimension: int) -> None:
    client.indices.put_mapping(
        index=index_name,
        body={
            "properties": {
                "embedding": {
                    "type": "dense_vector",
                    "dims": dimension,
                    "index": True,
                    "similarity": "cosine",
                }
            }
        },
    )


def init_db(args: argparse.Namespace, config: dict) -> None:
    schema_path = args.schema or BACKEND_ROOT / "db" / "entity_search_schema.sql"
    sql = Path(schema_path).read_text(encoding="utf-8")
    with psycopg.connect(postgres_dsn(args, config)) as conn:
        conn.execute(sql)
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM app_user")
            user_count = int(cur.fetchone()[0])
            if user_count == 0:
                username = os.getenv("DEFAULT_ADMIN_USERNAME", "admin")
                password = os.getenv("DEFAULT_ADMIN_PASSWORD", "admin123")
                cur.execute(
                    """
                    INSERT INTO app_user (username, password_hash, is_admin)
                    VALUES (%s, %s, TRUE)
                    ON CONFLICT (username) DO NOTHING
                    """,
                    (username, hash_password(password)),
                )
                print(f"Created default admin user: {username}")
    print("PostgreSQL schema is ready.")


def upsert_entity_type(cur: psycopg.Cursor, item: dict) -> int:
    cur.execute(
        """
        INSERT INTO entity_type (code, name, neo4j_label, source_file, priority)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (code) DO UPDATE SET
            name = EXCLUDED.name,
            neo4j_label = EXCLUDED.neo4j_label,
            source_file = EXCLUDED.source_file,
            priority = EXCLUDED.priority
        RETURNING id
        """,
        (
            item["code"],
            item["name"],
            item["neo4j_label"],
            item["source_file"],
            item["priority"],
        ),
    )
    return int(cur.fetchone()[0])


def upsert_entity(
    cur: psycopg.Cursor,
    type_id: int,
    name: str,
    source_file: str,
    source_line: int | None,
    source: str,
) -> int:
    cur.execute(
        """
        INSERT INTO kg_entity (
            type_id,
            name,
            normalized_name,
            source,
            source_file,
            source_line,
            is_active,
            updated_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, TRUE, now())
        ON CONFLICT (type_id, name) DO UPDATE SET
            normalized_name = EXCLUDED.normalized_name,
            source = EXCLUDED.source,
            source_file = EXCLUDED.source_file,
            source_line = EXCLUDED.source_line,
            is_active = TRUE,
            updated_at = now()
        RETURNING id
        """,
        (
            type_id,
            name,
            normalize_entity_text(name),
            source,
            source_file,
            source_line,
        ),
    )
    return int(cur.fetchone()[0])


def upsert_alias(
    cur: psycopg.Cursor,
    entity_id: int,
    alias: str,
    alias_type: str,
    confidence: float,
    source: str,
) -> None:
    cur.execute(
        """
        INSERT INTO entity_alias (
            entity_id,
            alias,
            normalized_alias,
            alias_type,
            confidence,
            source,
            is_active,
            updated_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, TRUE, now())
        ON CONFLICT (entity_id, normalized_alias) DO UPDATE SET
            alias = EXCLUDED.alias,
            alias_type = EXCLUDED.alias_type,
            confidence = EXCLUDED.confidence,
            source = EXCLUDED.source,
            is_active = TRUE,
            updated_at = now()
        """,
        (
            entity_id,
            alias,
            normalize_entity_text(alias),
            alias_type,
            confidence,
            source,
        ),
    )


def iter_entity_lines(path: Path) -> Iterable[tuple[int, str]]:
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        name = line.strip().split(" ")[0].strip()
        if len(name) >= 2:
            yield line_no, name


def import_entities(args: argparse.Namespace, config: dict) -> None:
    entity_dir = Path(args.entity_dir or PROJECT_ROOT / "data" / "ent_aug")
    alias_file = Path(args.alias_file or PROJECT_ROOT / "data" / "entity_aliases.csv")

    with psycopg.connect(postgres_dsn(args, config)) as conn:
        with conn.cursor() as cur:
            if args.clear:
                cur.execute(
                    "TRUNCATE disease_candidate_score_log, entity_search_log, entity_alias, kg_entity, entity_type RESTART IDENTITY CASCADE"
                )

            type_ids: dict[str, int] = {}
            for item in ENTITY_TYPES:
                type_ids[item["name"]] = upsert_entity_type(cur, item)

            imported = 0
            for item in ENTITY_TYPES:
                path = entity_dir / item["source_file"]
                if not path.exists():
                    print(f"Skip missing file: {path}")
                    continue
                type_id = type_ids[item["name"]]
                for line_no, name in iter_entity_lines(path):
                    entity_id = upsert_entity(
                        cur,
                        type_id=type_id,
                        name=name,
                        source_file=item["source_file"],
                        source_line=line_no,
                        source="ent_aug",
                    )
                    upsert_alias(
                        cur,
                        entity_id=entity_id,
                        alias=name,
                        alias_type="canonical",
                        confidence=1.0,
                        source="ent_aug",
                    )
                    imported += 1

            alias_imported = 0
            if alias_file.exists():
                with alias_file.open("r", encoding="utf-8", newline="") as f:
                    for row in csv.DictReader(f):
                        entity_type = row["entity_type"].strip()
                        canonical_name = row["canonical_name"].strip()
                        alias = row["alias"].strip()
                        if not entity_type or not canonical_name or not alias:
                            continue
                        cur.execute(
                            """
                            SELECT e.id
                            FROM kg_entity e
                            JOIN entity_type t ON t.id = e.type_id
                            WHERE t.name = %s AND e.name = %s
                            """,
                            (entity_type, canonical_name),
                        )
                        found = cur.fetchone()
                        if not found:
                            print(f"Skip alias with missing canonical entity: {entity_type},{canonical_name},{alias}")
                            continue
                        upsert_alias(
                            cur,
                            entity_id=int(found[0]),
                            alias=alias,
                            alias_type=row.get("alias_type", "manual").strip() or "manual",
                            confidence=float(row.get("confidence") or 1.0),
                            source=row.get("source", "manual").strip() or "manual",
                        )
                        alias_imported += 1

    print(f"Imported {imported} canonical entities and {alias_imported} aliases.")


def create_index(args: argparse.Namespace, config: dict) -> None:
    client = elastic_client(args, config)
    index_name = elastic_index(args, config)
    dimension = embedding_dimension(args, config) if vector_enabled(args, config) else None
    if args.recreate or not client.indices.exists(index=index_name):
        ensure_ik_analyzers(client)
    if args.recreate and client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)
    if not client.indices.exists(index=index_name):
        client.indices.create(index=index_name, body=build_index_body(dimension))
    elif dimension:
        put_vector_mapping(client, index_name, dimension)
    print(f"Elasticsearch index is ready: {index_name}")


def iter_alias_sources(args: argparse.Namespace, config: dict) -> Iterable[dict]:
    sql = """
        SELECT
            a.id AS alias_id,
            e.id AS entity_id,
            a.alias,
            a.normalized_alias,
            a.alias_type,
            a.confidence,
            a.source,
            e.name AS canonical_name,
            t.name AS entity_type,
            a.is_active,
            a.updated_at
        FROM entity_alias a
        JOIN kg_entity e ON e.id = a.entity_id
        JOIN entity_type t ON t.id = e.type_id
        WHERE e.is_active = TRUE
    """
    with psycopg.connect(postgres_dsn(args, config), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            for row in cur:
                yield {
                    "alias_id": row["alias_id"],
                    "entity_id": row["entity_id"],
                    "alias": row["alias"],
                    "normalized_alias": row["normalized_alias"],
                    "canonical_name": row["canonical_name"],
                    "entity_type": row["entity_type"],
                    "alias_type": row["alias_type"],
                    "confidence": float(row["confidence"]),
                    "source": row["source"],
                    "search_text": f"{row['alias']} {row['canonical_name']}",
                    "pinyin": "",
                    "is_active": row["is_active"],
                    "updated_at": row["updated_at"].isoformat(),
                }


def count_alias_sources(args: argparse.Namespace, config: dict) -> int:
    sql = """
        SELECT count(*)
        FROM entity_alias a
        JOIN kg_entity e ON e.id = a.entity_id
        WHERE e.is_active = TRUE
    """
    with psycopg.connect(postgres_dsn(args, config)) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            return int(cur.fetchone()[0])


def format_progress(label: str, current: int, total: int, start_time: float) -> str:
    elapsed = time.monotonic() - start_time
    rate = current / elapsed if elapsed > 0 else 0.0
    if total > 0:
        percent = current / total * 100
        return f"{label}: {current}/{total} ({percent:.1f}%) elapsed={elapsed:.1f}s rate={rate:.1f}/s"
    return f"{label}: {current} elapsed={elapsed:.1f}s rate={rate:.1f}/s"


def alias_documents(
    args: argparse.Namespace,
    config: dict,
    progress_callback=None,
) -> Iterable[dict]:
    embedder = embedding_service(args, config) if vector_enabled(args, config) else None
    batch: list[dict] = []

    def emit(rows: list[dict]) -> Iterable[dict]:
        if embedder:
            texts = [
                f"{row['alias']} {row['canonical_name']} {row['entity_type']}"
                for row in rows
            ]
            vectors = embedder.embed_batch(texts)
            if len(vectors) != len(rows):
                raise ValueError(
                    f"embedding count mismatch: rows={len(rows)} vectors={len(vectors)}"
                )
            for row, vector in zip(rows, vectors):
                row["embedding"] = vector

        for row in rows:
            alias_id = row.pop("alias_id")
            if progress_callback:
                progress_callback()
            yield {"_id": alias_id, "_source": row}

    for source in iter_alias_sources(args, config):
        batch.append(source)
        if len(batch) >= args.embedding_batch_size:
            yield from emit(batch)
            batch = []
    if batch:
        yield from emit(batch)


def sync_index(args: argparse.Namespace, config: dict) -> None:
    client = elastic_client(args, config)
    index_name = elastic_index(args, config)
    total = count_alias_sources(args, config)
    progress_interval = max(int(getattr(args, "progress_interval", 1000) or 0), 0)
    start_time = time.monotonic()
    print(f"Syncing {total} alias documents to Elasticsearch index: {index_name}", flush=True)
    if not client.indices.exists(index=index_name):
        ensure_ik_analyzers(client)
        dimension = embedding_dimension(args, config) if vector_enabled(args, config) else None
        client.indices.create(index=index_name, body=build_index_body(dimension))
    elif vector_enabled(args, config):
        put_vector_mapping(client, index_name, embedding_dimension(args, config))

    prepared = 0
    indexed = 0
    success = 0
    errors = 0

    def report(label: str, current: int, force: bool = False) -> None:
        if progress_interval <= 0 and not force:
            return
        if force or current % progress_interval == 0 or (total > 0 and current >= total):
            print(format_progress(label, current, total, start_time), flush=True)

    def on_document_prepared() -> None:
        nonlocal prepared
        prepared += 1
        report("Prepared", prepared)

    def actions() -> Iterable[dict]:
        for doc in alias_documents(args, config, progress_callback=on_document_prepared):
            yield {
                "_op_type": "index",
                "_index": index_name,
                **doc,
            }

    for ok, _item in helpers.streaming_bulk(
        client,
        actions(),
        chunk_size=1000,
        raise_on_error=False,
        raise_on_exception=False,
    ):
        indexed += 1
        if ok:
            success += 1
        else:
            errors += 1
        report("Indexed", indexed)

    client.indices.refresh(index=index_name)
    report("Prepared", prepared, force=True)
    report("Indexed", indexed, force=True)
    print(f"Indexed {success} alias documents. errors={errors}")


def rebuild(args: argparse.Namespace, config: dict) -> None:
    init_db(args, config)
    args.clear = True
    import_entities(args, config)
    args.recreate = True
    create_index(args, config)
    sync_index(args, config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage MedGraphQA entity search data")
    parser.add_argument("--config", default=str(BACKEND_ROOT / "config.json"))
    parser.add_argument("--dsn")
    parser.add_argument("--es-hosts")
    parser.add_argument("--es-index")
    parser.add_argument("--es-username")
    parser.add_argument("--es-password")
    parser.add_argument("--schema", type=Path)
    parser.add_argument("--entity-dir")
    parser.add_argument("--alias-file")
    parser.add_argument("--with-embeddings", action="store_true")
    parser.add_argument("--embedding-provider")
    parser.add_argument("--embedding-model")
    parser.add_argument("--embedding-api-base")
    parser.add_argument("--embedding-dimension", type=int)
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--progress-interval", type=int, default=1000)

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init-db")

    import_parser = subparsers.add_parser("import-entities")
    import_parser.add_argument("--clear", action="store_true")

    index_parser = subparsers.add_parser("create-index")
    index_parser.add_argument("--recreate", action="store_true")

    sync_parser = subparsers.add_parser("sync-index")
    sync_parser.add_argument("--progress-interval", type=int, default=1000)

    rebuild_parser = subparsers.add_parser("rebuild")
    rebuild_parser.add_argument("--clear", action="store_true")
    rebuild_parser.add_argument("--recreate", action="store_true")
    rebuild_parser.add_argument("--progress-interval", type=int, default=1000)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(Path(args.config))

    if args.command == "init-db":
        init_db(args, config)
    elif args.command == "import-entities":
        import_entities(args, config)
    elif args.command == "create-index":
        create_index(args, config)
    elif args.command == "sync-index":
        sync_index(args, config)
    elif args.command == "rebuild":
        rebuild(args, config)
    else:
        parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
