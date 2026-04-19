from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Sequence

from app.core.config import settings
from app.services.kg_service import GraphService
from app.services.operation_log import log_operation
from mcp_servers.kg.schemas import (
    ATTR_FIELDS,
    ENTITY_LABELS,
    RELATION_TARGETS,
    clean_attribute,
    clean_limit,
    clean_name,
    clean_names,
    clean_relation,
    error,
    ok,
)


logger = logging.getLogger("medgraphqa.mcp.kg")


class KGMcpTools:
    def __init__(self, graph_service: GraphService):
        self.graph_service = graph_service

    @classmethod
    def from_settings(cls) -> "KGMcpTools":
        graph_service = GraphService(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password,
            database=settings.neo4j_database,
        )
        return cls(graph_service)

    def schema(self) -> dict:
        return ok(
            {
                "labels": list(ENTITY_LABELS),
                "disease_attribute_fields": list(ATTR_FIELDS),
                "disease_relation_targets": [
                    {"relation": relation, "target_label": target}
                    for relation, target in RELATION_TARGETS.items()
                ],
                "limits": {
                    "max_name_length": 80,
                    "max_symptoms": 20,
                    "max_limit": 50,
                },
                "security": {
                    "mode": "read_only",
                    "raw_cypher": False,
                    "write_operations": False,
                },
            }
        )

    def ping(self) -> dict:
        return self._safe_call(
            "mcp.kg.ping",
            lambda: ok(
                {
                    "connected": self.graph_service.ping(),
                    "neo4j_uri": self.graph_service.uri,
                    "database": self.graph_service.database,
                }
            ),
        )

    def get_disease_profile(self, disease_name: str) -> dict:
        def run() -> dict:
            disease = clean_name(disease_name, "disease_name")
            attributes: dict[str, str | None] = {}
            for field_name in ATTR_FIELDS:
                attributes[field_name] = self.graph_service.get_disease_attribute(
                    disease,
                    field_name,
                )

            relations: dict[str, list[str]] = {}
            for relation, target_label in RELATION_TARGETS.items():
                relations[relation] = self.graph_service.get_related_entities(
                    disease,
                    relation,
                    target_label,
                )

            has_data = any(attributes.values()) or any(relations.values())
            return ok(
                {
                    "disease": disease,
                    "exists": has_data,
                    "attributes": attributes,
                    "relations": relations,
                }
            )

        return self._safe_call("mcp.kg.get_disease_profile", run)

    def get_disease_attribute(self, disease_name: str, field_name: str) -> dict:
        def run() -> dict:
            disease = clean_name(disease_name, "disease_name")
            field = clean_attribute(field_name)
            value = self.graph_service.get_disease_attribute(disease, field)
            return ok(
                {
                    "disease": disease,
                    "field_name": field,
                    "value": value,
                    "exists": value is not None and str(value).strip() != "",
                }
            )

        return self._safe_call("mcp.kg.get_disease_attribute", run)

    def get_related_entities(
        self,
        disease_name: str,
        relation: str,
        target_label: str | None = None,
    ) -> dict:
        def run() -> dict:
            disease = clean_name(disease_name, "disease_name")
            rel, target = clean_relation(relation, target_label)
            names = self.graph_service.get_related_entities(disease, rel, target)
            return ok(
                {
                    "disease": disease,
                    "relation": rel,
                    "target_label": target,
                    "names": names,
                    "count": len(names),
                }
            )

        return self._safe_call("mcp.kg.get_related_entities", run)

    def get_diseases_by_symptom(self, symptom: str, limit: int = 50) -> dict:
        def run() -> dict:
            name = clean_name(symptom, "symptom")
            capped_limit = clean_limit(limit, default=50)
            diseases = self.graph_service.get_diseases_by_symptom(name)
            return ok(
                {
                    "symptom": name,
                    "diseases": diseases[:capped_limit],
                    "count": len(diseases),
                    "returned_count": min(len(diseases), capped_limit),
                    "limit": capped_limit,
                }
            )

        return self._safe_call("mcp.kg.get_diseases_by_symptom", run)

    def get_disease_candidates_by_symptoms(
        self,
        symptoms: Sequence[str],
        limit: int = 10,
    ) -> dict:
        def run() -> dict:
            names = clean_names(symptoms, "symptoms")
            capped_limit = clean_limit(limit, default=10)
            candidates = self.graph_service.get_disease_candidates_by_symptoms(
                names,
                capped_limit,
            )
            return ok(
                {
                    "symptoms": names,
                    "candidates": candidates,
                    "count": len(candidates),
                    "limit": capped_limit,
                }
            )

        return self._safe_call("mcp.kg.get_disease_candidates_by_symptoms", run)

    def get_symptom_disease_counts(self, symptoms: Sequence[str]) -> dict:
        def run() -> dict:
            names = clean_names(symptoms, "symptoms")
            counts = self.graph_service.get_symptom_disease_counts(names)
            return ok({"symptoms": names, "counts": counts})

        return self._safe_call("mcp.kg.get_symptom_disease_counts", run)

    def get_producer_by_drug(self, drug_name: str, limit: int = 50) -> dict:
        def run() -> dict:
            drug = clean_name(drug_name, "drug_name")
            capped_limit = clean_limit(limit, default=50)
            producers = self.graph_service.get_producer_by_drug(drug)
            return ok(
                {
                    "drug": drug,
                    "producers": producers[:capped_limit],
                    "count": len(producers),
                    "returned_count": min(len(producers), capped_limit),
                    "limit": capped_limit,
                }
            )

        return self._safe_call("mcp.kg.get_producer_by_drug", run)

    def inspect_entity(self, name: str, sample_limit: int = 20) -> dict:
        def run() -> dict:
            entity_name = clean_name(name, "name")
            capped_limit = clean_limit(sample_limit, default=20)
            nodes = self._inspect_nodes(entity_name)
            outgoing = self._relation_samples(entity_name, "out", capped_limit)
            incoming = self._relation_samples(entity_name, "in", capped_limit)
            labels = {
                label
                for node in nodes
                for label in node.get("labels", [])
            }
            possible_issue = None
            if len(labels) > 1:
                possible_issue = "same_name_multi_label"
            return ok(
                {
                    "name": entity_name,
                    "nodes": nodes,
                    "node_count": len(nodes),
                    "labels": sorted(labels),
                    "possible_issue": possible_issue,
                    "relation_samples": {
                        "outgoing": outgoing,
                        "incoming": incoming,
                    },
                }
            )

        return self._safe_call("mcp.kg.inspect_entity", run)

    def _inspect_nodes(self, name: str) -> list[dict[str, Any]]:
        cypher = """
        MATCH (n {名称:$name})
        WITH n, labels(n) AS labels, properties(n) AS properties
        CALL {
            WITH n
            MATCH (n)-[r]->()
            RETURN count(r) AS out_degree
        }
        CALL {
            WITH n
            MATCH ()-[r]->(n)
            RETURN count(r) AS in_degree
        }
        RETURN labels, properties, out_degree, in_degree
        ORDER BY labels
        """
        rows = self.graph_service.graph.run(cypher, name=name).data()
        result: list[dict[str, Any]] = []
        for row in rows:
            properties = dict(row.get("properties") or {})
            result.append(
                {
                    "labels": row.get("labels") or [],
                    "properties": properties,
                    "out_degree": int(row.get("out_degree") or 0),
                    "in_degree": int(row.get("in_degree") or 0),
                }
            )
        return result

    def _relation_samples(self, name: str, direction: str, limit: int) -> list[dict]:
        if direction == "out":
            cypher = """
            MATCH (n {名称:$name})-[r]->(m)
            RETURN type(r) AS relation, labels(m) AS labels, m.名称 AS name
            ORDER BY relation, name
            LIMIT $limit
            """
        else:
            cypher = """
            MATCH (m)-[r]->(n {名称:$name})
            RETURN type(r) AS relation, labels(m) AS labels, m.名称 AS name
            ORDER BY relation, name
            LIMIT $limit
            """
        rows = self.graph_service.graph.run(cypher, name=name, limit=limit).data()
        return [
            {
                "relation": row.get("relation"),
                "labels": row.get("labels") or [],
                "name": row.get("name"),
            }
            for row in rows
            if row.get("relation")
        ]

    def _safe_call(self, operation: str, fn: Callable[[], dict]) -> dict:
        with log_operation(logger, operation) as result:
            try:
                payload = fn()
                result["ok"] = payload.get("ok")
                data = payload.get("data")
                if isinstance(data, dict):
                    result["data_keys"] = ",".join(data.keys())
                return payload
            except ValueError as exc:
                result["ok"] = False
                result["error_code"] = "validation_error"
                return error("validation_error", str(exc))
            except Exception as exc:
                logger.exception("operation=%s status=error", operation)
                result["ok"] = False
                result["error_code"] = type(exc).__name__
                return error(
                    "kg_query_failed",
                    "知识图谱查询失败，请检查 Neo4j 服务和参数。",
                    error_type=type(exc).__name__,
                )
