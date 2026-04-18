CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS entity_type (
    id bigserial PRIMARY KEY,
    code varchar(64) NOT NULL UNIQUE,
    name varchar(64) NOT NULL UNIQUE,
    neo4j_label varchar(64) NOT NULL,
    source_file varchar(255),
    priority integer NOT NULL DEFAULT 100,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS kg_entity (
    id bigserial PRIMARY KEY,
    type_id bigint NOT NULL REFERENCES entity_type(id) ON DELETE RESTRICT,
    name varchar(255) NOT NULL,
    normalized_name varchar(255) NOT NULL,
    neo4j_node_key varchar(512),
    source varchar(64) NOT NULL DEFAULT 'ent_aug',
    source_file varchar(255),
    source_line integer,
    is_active boolean NOT NULL DEFAULT TRUE,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (type_id, name)
);

CREATE TABLE IF NOT EXISTS entity_alias (
    id bigserial PRIMARY KEY,
    entity_id bigint NOT NULL REFERENCES kg_entity(id) ON DELETE CASCADE,
    alias varchar(255) NOT NULL,
    normalized_alias varchar(255) NOT NULL,
    alias_type varchar(32) NOT NULL,
    confidence numeric(5, 4) NOT NULL DEFAULT 1.0,
    source varchar(64) NOT NULL,
    is_active boolean NOT NULL DEFAULT TRUE,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CHECK (confidence >= 0 AND confidence <= 1),
    UNIQUE (entity_id, normalized_alias)
);

CREATE TABLE IF NOT EXISTS entity_search_log (
    id bigserial PRIMARY KEY,
    query text NOT NULL,
    mention varchar(255),
    expected_types text[] NOT NULL DEFAULT '{}',
    candidates jsonb NOT NULL DEFAULT '[]'::jsonb,
    selected_entity_id bigint REFERENCES kg_entity(id) ON DELETE SET NULL,
    selected_name varchar(255),
    selected_type varchar(64),
    score numeric(12, 4),
    matched_alias varchar(255),
    match_method varchar(64),
    is_accepted boolean,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS disease_candidate_score_log (
    id bigserial PRIMARY KEY,
    query text NOT NULL,
    symptoms jsonb NOT NULL DEFAULT '[]'::jsonb,
    candidates jsonb NOT NULL DEFAULT '[]'::jsonb,
    selected_disease varchar(255),
    confidence numeric(12, 4),
    decision varchar(32) NOT NULL,
    follow_up_question text,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_kg_entity_type_normalized
    ON kg_entity (type_id, normalized_name);

CREATE INDEX IF NOT EXISTS idx_entity_alias_normalized
    ON entity_alias (normalized_alias);

CREATE INDEX IF NOT EXISTS idx_entity_alias_trgm
    ON entity_alias USING gin (alias gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_entity_search_log_created_at
    ON entity_search_log (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_disease_candidate_score_log_created_at
    ON disease_candidate_score_log (created_at DESC);

CREATE TABLE IF NOT EXISTS chat_session (
    id uuid PRIMARY KEY,
    user_id varchar(64) NOT NULL,
    title varchar(255),
    is_active boolean NOT NULL DEFAULT TRUE,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chat_message (
    id bigserial PRIMARY KEY,
    session_id uuid NOT NULL REFERENCES chat_session(id) ON DELETE CASCADE,
    role varchar(32) NOT NULL,
    content text NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chat_session_state (
    session_id uuid PRIMARY KEY REFERENCES chat_session(id) ON DELETE CASCADE,
    state jsonb NOT NULL DEFAULT '{}'::jsonb,
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chat_session_user_updated
    ON chat_session (user_id, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_chat_message_session_created
    ON chat_message (session_id, created_at);

CREATE TABLE IF NOT EXISTS app_user (
    username varchar(64) PRIMARY KEY,
    password_hash text NOT NULL,
    is_admin boolean NOT NULL DEFAULT FALSE,
    is_active boolean NOT NULL DEFAULT TRUE,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS auth_session (
    token_hash char(64) PRIMARY KEY,
    username varchar(64) NOT NULL REFERENCES app_user(username) ON DELETE CASCADE,
    expire_at timestamptz NOT NULL,
    revoked_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    last_seen_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_auth_session_username
    ON auth_session (username, expire_at DESC);

CREATE INDEX IF NOT EXISTS idx_auth_session_expire_at
    ON auth_session (expire_at);

CREATE TABLE IF NOT EXISTS user_memory (
    id bigserial PRIMARY KEY,
    user_id varchar(64) NOT NULL REFERENCES app_user(username) ON DELETE CASCADE,
    memory_type varchar(64) NOT NULL,
    memory_key varchar(128) NOT NULL,
    value jsonb NOT NULL DEFAULT '{}'::jsonb,
    text text NOT NULL,
    source varchar(64) NOT NULL DEFAULT 'rule',
    confidence numeric(5, 4) NOT NULL DEFAULT 1.0,
    status varchar(32) NOT NULL DEFAULT 'active',
    first_seen_at timestamptz NOT NULL DEFAULT now(),
    last_seen_at timestamptz NOT NULL DEFAULT now(),
    expires_at timestamptz,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    CHECK (confidence >= 0 AND confidence <= 1),
    UNIQUE (user_id, memory_type, memory_key)
);

CREATE INDEX IF NOT EXISTS idx_user_memory_user_status
    ON user_memory (user_id, status, last_seen_at DESC);

CREATE INDEX IF NOT EXISTS idx_user_memory_expires_at
    ON user_memory (expires_at);
