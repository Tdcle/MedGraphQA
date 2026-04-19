from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from mcp_servers.kg.kg_tools import KGMcpTools  # noqa: E402


def create_server(tools: KGMcpTools | None = None, host: str = "127.0.0.1", port: int = 8001) -> FastMCP:
    kg_tools = tools or KGMcpTools.from_settings()
    mcp = FastMCP(
        "MedGraphQA KG",
        instructions=(
            "Read-only MCP server for MedGraphQA's Neo4j medical knowledge graph. "
            "Use these tools to inspect diseases, symptoms, relations, and KG-derived candidates. "
            "Do not treat returned candidates as a confirmed diagnosis."
        ),
        json_response=True,
        host=host,
        port=port,
    )

    @mcp.tool()
    def kg_ping() -> dict:
        """Check whether the configured Neo4j knowledge graph is reachable."""
        return kg_tools.ping()

    @mcp.tool()
    def kg_get_disease_profile(disease_name: str) -> dict:
        """Return whitelisted attributes and relations for a disease."""
        return kg_tools.get_disease_profile(disease_name)

    @mcp.tool()
    def kg_get_disease_attribute(disease_name: str, field_name: str) -> dict:
        """Return one whitelisted disease attribute."""
        return kg_tools.get_disease_attribute(disease_name, field_name)

    @mcp.tool()
    def kg_get_related_entities(
        disease_name: str,
        relation: str,
        target_label: str | None = None,
    ) -> dict:
        """Return related entities for a whitelisted disease relation."""
        return kg_tools.get_related_entities(disease_name, relation, target_label)

    @mcp.tool()
    def kg_get_diseases_by_symptom(symptom: str, limit: int = 50) -> dict:
        """Return diseases connected to a symptom through 疾病的症状."""
        return kg_tools.get_diseases_by_symptom(symptom, limit)

    @mcp.tool()
    def kg_get_disease_candidates_by_symptoms(
        symptoms: list[str],
        limit: int = 10,
    ) -> dict:
        """Return KG disease candidates ranked by matched symptom count."""
        return kg_tools.get_disease_candidates_by_symptoms(symptoms, limit)

    @mcp.tool()
    def kg_get_symptom_disease_counts(symptoms: list[str]) -> dict:
        """Return how many diseases are connected to each symptom."""
        return kg_tools.get_symptom_disease_counts(symptoms)

    @mcp.tool()
    def kg_get_producer_by_drug(drug_name: str, limit: int = 50) -> dict:
        """Return producers connected to a drug through 生产."""
        return kg_tools.get_producer_by_drug(drug_name, limit)

    @mcp.tool()
    def kg_inspect_entity(name: str, sample_limit: int = 20) -> dict:
        """Inspect labels, degrees, and relation samples for an entity name."""
        return kg_tools.inspect_entity(name, sample_limit)

    @mcp.resource("medgraphqa://kg/schema")
    def kg_schema_resource() -> str:
        """Return KG labels, whitelisted fields, relations, and safety limits."""
        return _json_resource(kg_tools.schema())

    @mcp.resource("medgraphqa://kg/disease/{name}")
    def kg_disease_resource(name: str) -> str:
        """Return a disease profile as a JSON resource."""
        return _json_resource(kg_tools.get_disease_profile(name))

    @mcp.resource("medgraphqa://kg/symptom/{name}/diseases")
    def kg_symptom_diseases_resource(name: str) -> str:
        """Return diseases connected to a symptom as a JSON resource."""
        return _json_resource(kg_tools.get_diseases_by_symptom(name))

    return mcp


def _json_resource(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MedGraphQA KG MCP server.")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http", "http", "sse"],
        default="stdio",
        help="MCP transport. 'http' is an alias for streamable-http.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP host.")
    parser.add_argument("--port", type=int, default=8001, help="HTTP port.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transport = "streamable-http" if args.transport == "http" else args.transport
    mcp = create_server(host=args.host, port=args.port)
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
