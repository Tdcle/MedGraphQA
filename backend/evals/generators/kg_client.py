from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import psycopg
import py2neo
from psycopg.rows import dict_row


@dataclass
class DiseaseProfile:
    name: str
    symptoms: list[str] = field(default_factory=list)
    cure_methods: list[str] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)
    drugs: list[str] = field(default_factory=list)
    departments: list[str] = field(default_factory=list)
    description: str | None = None


@dataclass
class AliasRecord:
    canonical_name: str
    entity_type: str
    alias: str
    alias_type: str
    confidence: float


COMMON_DISEASE_PRIORITY = [
    "感冒",
    "流行性感冒",
    "急性上呼吸道感染",
    "急性咽炎",
    "慢性咽炎",
    "鼻炎",
    "急性胃炎",
    "胃炎",
    "急性胃肠炎",
    "腹泻",
    "肺炎",
    "支气管炎",
    "高血压",
    "糖尿病",
    "偏头痛",
]

SYMPTOM_KEYWORDS = {
    "痛",
    "疼",
    "热",
    "烧",
    "咳",
    "痰",
    "鼻",
    "涕",
    "喷嚏",
    "咽",
    "喉",
    "哑",
    "泻",
    "便",
    "呕",
    "恶心",
    "胀",
    "晕",
    "乏力",
    "无力",
    "寒",
    "汗",
    "皮疹",
    "瘙痒",
    "出血",
    "呼吸",
    "胸闷",
    "心悸",
    "水肿",
    "食欲",
    "减退",
    "困难",
    "干",
    "麻",
    "肿",
    "黄疸",
    "尿",
    "抽搐",
    "意识",
}

BAD_SYMPTOM_KEYWORDS = {
    "要利琴",
    "衣玉品",
    "查见",
    "不明原因",
    "水样带黏",
    "伴有",
    "梗死",
    "心包炎",
    "癌",
    "瘤",
    "综合征",
    "感染",
    "疾病",
    "试验",
    "检查",
}

LOW_PRIORITY_DISEASE_KEYWORDS = {
    "恶性",
    "肿瘤",
    "癌",
    "转移",
    "综合征",
    "先天",
    "罕见",
}

SYMPTOM_LIKE_DISEASE_NAMES = {
    "腹痛",
    "腹泻",
    "咳嗽",
    "发热",
    "发烧",
    "头痛",
    "胃痛",
    "咽痛",
    "咽喉痛",
    "鼻塞",
}


class EvalDataSource:
    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        neo4j_database: str,
        postgres_dsn: str,
    ) -> None:
        self.graph = py2neo.Graph(
            neo4j_uri,
            user=neo4j_user,
            password=neo4j_password,
            name=neo4j_database,
        )
        self.postgres_dsn = postgres_dsn

    def fetch_disease_profiles(self, limit: int) -> list[DiseaseProfile]:
        rows = self.graph.run(
            """
            MATCH (d:疾病)-[:`疾病的症状`]->(s:疾病症状)
            WITH d, collect(DISTINCT s.名称) AS symptoms
            WHERE size(symptoms) >= 3
            OPTIONAL MATCH (d)-[:`治疗的方法`]->(t:治疗方法)
            WITH d, symptoms, collect(DISTINCT t.名称) AS cure_methods
            OPTIONAL MATCH (d)-[:`疾病所需检查`]->(c:检查项目)
            WITH d, symptoms, cure_methods, collect(DISTINCT c.名称) AS checks
            OPTIONAL MATCH (d)-[:`疾病使用药品`]->(m:药品)
            WITH d, symptoms, cure_methods, checks, collect(DISTINCT m.名称) AS drugs
            OPTIONAL MATCH (d)-[:`疾病所属科目`]->(dep:科目)
            RETURN
                d.名称 AS disease,
                d.`疾病简介` AS description,
                symptoms,
                cure_methods,
                checks,
                drugs,
                collect(DISTINCT dep.名称) AS departments
            LIMIT $limit
            """,
            limit=max(limit * 50, 5000),
        ).data()
        profiles = [
            DiseaseProfile(
                name=str(row["disease"]),
                description=self._clean_text(row.get("description")),
                symptoms=self._clean_symptoms(row.get("symptoms"), 14),
                cure_methods=self._clean_list(row.get("cure_methods"), 8),
                checks=self._clean_list(row.get("checks"), 8),
                drugs=self._clean_list(row.get("drugs"), 8),
                departments=self._clean_list(row.get("departments"), 5),
            )
            for row in rows
            if row.get("disease")
        ]
        profiles = [
            item
            for item in profiles
            if len(item.symptoms) >= 3 and self._is_quality_disease(item.name)
        ]
        return self._prioritize_common_profiles(profiles)[:limit]

    def fetch_aliases(self, names: Iterable[str] | None = None) -> dict[str, list[AliasRecord]]:
        params: list[object] = []
        filter_sql = ""
        if names:
            params.append(list(dict.fromkeys(names)))
            filter_sql = "AND e.name = ANY(%s)"
        sql = f"""
            SELECT
                e.name AS canonical_name,
                t.name AS entity_type,
                a.alias,
                a.alias_type,
                a.confidence
            FROM entity_alias a
            JOIN kg_entity e ON e.id = a.entity_id
            JOIN entity_type t ON t.id = e.type_id
            WHERE a.is_active = TRUE
              AND e.is_active = TRUE
              {filter_sql}
            ORDER BY e.name, a.confidence DESC, length(a.alias) DESC
        """
        aliases: dict[str, list[AliasRecord]] = {}
        try:
            with psycopg.connect(self.postgres_dsn, row_factory=dict_row) as conn:
                rows = conn.execute(sql, params).fetchall()
        except Exception:
            return aliases
        for row in rows:
            record = AliasRecord(
                canonical_name=str(row["canonical_name"]),
                entity_type=str(row["entity_type"]),
                alias=str(row["alias"]),
                alias_type=str(row["alias_type"]),
                confidence=float(row["confidence"]),
            )
            aliases.setdefault(record.canonical_name, []).append(record)
        return aliases

    def fetch_alias_samples(self, limit: int) -> list[AliasRecord]:
        sql = """
            SELECT
                e.name AS canonical_name,
                t.name AS entity_type,
                a.alias,
                a.alias_type,
                a.confidence
            FROM entity_alias a
            JOIN kg_entity e ON e.id = a.entity_id
            JOIN entity_type t ON t.id = e.type_id
            WHERE a.is_active = TRUE
              AND e.is_active = TRUE
              AND a.alias <> e.name
              AND length(a.alias) >= 2
            ORDER BY a.confidence DESC, random()
            LIMIT %s
        """
        try:
            with psycopg.connect(self.postgres_dsn, row_factory=dict_row) as conn:
                rows = conn.execute(sql, (limit,)).fetchall()
        except Exception:
            return []
        return [
            AliasRecord(
                canonical_name=str(row["canonical_name"]),
                entity_type=str(row["entity_type"]),
                alias=str(row["alias"]),
                alias_type=str(row["alias_type"]),
                confidence=float(row["confidence"]),
            )
            for row in rows
        ]

    @staticmethod
    def _clean_list(value, limit: int) -> list[str]:
        if not value:
            return []
        result: list[str] = []
        for item in value:
            text = str(item or "").strip()
            if text and text not in result:
                result.append(text)
            if len(result) >= limit:
                break
        return result

    @classmethod
    def _clean_symptoms(cls, value, limit: int) -> list[str]:
        if not value:
            return []
        result: list[str] = []
        for item in value:
            text = str(item or "").strip()
            if not cls._is_quality_symptom(text):
                continue
            if text not in result:
                result.append(text)
            if len(result) >= limit:
                break
        return result

    @staticmethod
    def _is_quality_symptom(text: str) -> bool:
        if not text:
            return False
        if any(mark in text for mark in "，,。；;、/\\"):
            return False
        if len(text) < 2 or len(text) > 12:
            return False
        if any(item in text for item in BAD_SYMPTOM_KEYWORDS):
            return False
        return any(item in text for item in SYMPTOM_KEYWORDS)

    @staticmethod
    def _is_quality_disease(name: str) -> bool:
        text = str(name or "").strip()
        if not text or text in SYMPTOM_LIKE_DISEASE_NAMES:
            return False
        if len(text) > 16:
            return False
        return True

    @staticmethod
    def _clean_text(value) -> str | None:
        text = str(value or "").strip()
        return text or None

    @staticmethod
    def _prioritize_common_profiles(
        profiles: list[DiseaseProfile],
    ) -> list[DiseaseProfile]:
        priority = {name: idx for idx, name in enumerate(COMMON_DISEASE_PRIORITY)}
        return sorted(
            profiles,
            key=lambda item: (
                priority.get(item.name, 999),
                EvalDataSource._disease_noise_penalty(item.name),
                len(item.name),
                -len(item.symptoms),
                -(len(item.cure_methods) + len(item.checks) + len(item.drugs)),
                item.name,
            ),
        )

    @staticmethod
    def _disease_noise_penalty(name: str) -> int:
        return 1 if any(item in name for item in LOW_PRIORITY_DISEASE_KEYWORDS) else 0
