import logging
from typing import List, Sequence

import py2neo

from app.services.operation_log import log_operation


logger = logging.getLogger("medgraphqa.kg")


class GraphService:
    _ATTR_FIELDS = {
        "疾病简介",
        "疾病病因",
        "预防措施",
        "治疗周期",
        "治愈概率",
        "疾病易感人群",
    }

    _RELATION_TARGETS = {
        ("疾病使用药品", "药品"),
        ("疾病宜吃食物", "食物"),
        ("疾病忌吃食物", "食物"),
        ("疾病所需检查", "检查项目"),
        ("疾病所属科目", "科目"),
        ("疾病的症状", "疾病症状"),
        ("治疗的方法", "治疗方法"),
        ("疾病并发疾病", "疾病"),
    }

    def __init__(self, uri: str, user: str, password: str, database: str):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self._graph: py2neo.Graph | None = None

    @property
    def graph(self) -> py2neo.Graph:
        if self._graph is None:
            logger.info(
                "connecting neo4j uri=%s database=%s user=%s",
                self.uri,
                self.database,
                self.user,
            )
            self._graph = py2neo.Graph(
                self.uri, user=self.user, password=self.password, name=self.database
            )
        return self._graph

    def ping(self) -> bool:
        try:
            self.graph.run("RETURN 1 AS ok").data()
            return True
        except Exception:
            logger.exception("neo4j health check failed")
            return False

    def get_disease_attribute(self, disease: str, field_name: str) -> str | None:
        if field_name not in self._ATTR_FIELDS:
            raise ValueError("不支持的属性查询")
        with log_operation(
            logger,
            "kg.get_disease_attribute",
            disease=disease,
            field=field_name,
        ) as result:
            cypher = f"MATCH (a:疾病 {{名称:$name}}) RETURN a.`{field_name}` AS value LIMIT 1"
            data = self.graph.run(cypher, name=disease).data()
            result["row_count"] = len(data)
            if not data:
                return None
            value = data[0].get("value")
            result["value_len"] = len(str(value or ""))
            return value

    def get_related_entities(
        self, disease: str, relation: str, target_label: str
    ) -> List[str]:
        if (relation, target_label) not in self._RELATION_TARGETS:
            raise ValueError("不支持的关系查询")
        with log_operation(
            logger,
            "kg.get_related_entities",
            disease=disease,
            relation=relation,
            target_label=target_label,
        ) as result:
            cypher = (
                f"MATCH (a:疾病 {{名称:$name}})-[:`{relation}`]->(b:{target_label}) "
                "RETURN b.名称 AS name"
            )
            rows = self.graph.run(cypher, name=disease).data()
            names = [row["name"] for row in rows if row.get("name")]
            result["row_count"] = len(names)
            return names

    def get_diseases_by_symptom(self, symptom: str) -> List[str]:
        cypher = (
            "MATCH (a:疾病)-[:`疾病的症状`]->(b:疾病症状 {名称:$name}) "
            "RETURN a.名称 AS name"
        )
        with log_operation(logger, "kg.get_diseases_by_symptom", symptom=symptom) as result:
            rows = self.graph.run(cypher, name=symptom).data()
            names = [row["name"] for row in rows if row.get("name")]
            result["row_count"] = len(names)
            return names

    def count_diseases(self) -> int:
        with log_operation(logger, "kg.count_diseases") as result:
            rows = self.graph.run("MATCH (a:疾病) RETURN count(a) AS total").data()
            total = int(rows[0].get("total") or 0) if rows else 0
            result["total"] = total
            return total

    def get_symptom_disease_counts(self, symptoms: Sequence[str]) -> dict[str, int]:
        names = [item for item in symptoms if item]
        if not names:
            return {}
        cypher = (
            "MATCH (a:疾病)-[:`疾病的症状`]->(b:疾病症状) "
            "WHERE b.名称 IN $names "
            "RETURN b.名称 AS symptom, count(DISTINCT a) AS disease_count"
        )
        with log_operation(
            logger,
            "kg.get_symptom_disease_counts",
            symptom_count=len(names),
        ) as result:
            rows = self.graph.run(cypher, names=names).data()
            counts = {
                str(row["symptom"]): int(row.get("disease_count") or 0)
                for row in rows
                if row.get("symptom")
            }
            result["row_count"] = len(counts)
            return counts

    def get_disease_candidates_by_symptoms(
        self, symptoms: Sequence[str], limit: int
    ) -> list[dict]:
        names = [item for item in symptoms if item]
        if not names:
            return []
        cypher = """
        MATCH (d:疾病)-[:`疾病的症状`]->(s:疾病症状)
        WHERE s.名称 IN $symptoms
        WITH d, collect(DISTINCT s.名称) AS matched_symptoms, count(DISTINCT s) AS matched_count
        MATCH (d)-[:`疾病的症状`]->(all_symptom:疾病症状)
        WITH
            d,
            matched_symptoms,
            matched_count,
            collect(DISTINCT all_symptom.名称) AS disease_symptoms
        RETURN
            d.名称 AS disease,
            matched_symptoms,
            matched_count,
            disease_symptoms,
            size(disease_symptoms) AS disease_symptom_count
        ORDER BY matched_count DESC, disease_symptom_count ASC, disease ASC
        LIMIT $limit
        """
        with log_operation(
            logger,
            "kg.get_disease_candidates_by_symptoms",
            symptom_count=len(names),
            limit=limit,
        ) as result:
            rows = self.graph.run(cypher, symptoms=names, limit=limit).data()
            candidates = [
                {
                    "disease": row.get("disease"),
                    "matched_symptoms": row.get("matched_symptoms") or [],
                    "matched_count": int(row.get("matched_count") or 0),
                    "disease_symptoms": row.get("disease_symptoms") or [],
                    "disease_symptom_count": int(row.get("disease_symptom_count") or 0),
                }
                for row in rows
                if row.get("disease")
            ]
            result["row_count"] = len(candidates)
            return candidates

    def get_producer_by_drug(self, drug: str) -> List[str]:
        cypher = (
            "MATCH (a:药品商)-[:`生产`]->(b:药品 {名称:$name}) "
            "RETURN a.名称 AS name"
        )
        with log_operation(logger, "kg.get_producer_by_drug", drug=drug) as result:
            rows = self.graph.run(cypher, name=drug).data()
            names = [row["name"] for row in rows if row.get("name")]
            result["row_count"] = len(names)
            return names
