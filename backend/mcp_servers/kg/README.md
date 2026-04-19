# MedGraphQA KG MCP Server

This MCP server exposes read-only tools and resources for the MedGraphQA Neo4j medical knowledge graph.

It is intended for agent/debug workflows, not for direct patient-facing diagnosis. Returned disease candidates are KG matches only and must not be treated as confirmed diagnoses.

## Install

Install backend dependencies:

```powershell
pip install -r backend\requirements.txt
```

The MCP dependency is declared as:

```text
mcp[cli]>=1.0.0
```

## Configuration

The server reuses the existing MedGraphQA backend configuration:

- `backend/config.json`
- `backend/.env`
- project root `.env`
- `NEO4J_PASSWORD`

It reads these values through `app.core.config.settings`:

- `neo4j.uri`
- `neo4j.user`
- `neo4j.database`
- `NEO4J_PASSWORD`

## Run

Stdio transport:

```powershell
python backend\mcp_servers\kg\server.py --transport stdio
```

Streamable HTTP transport:

```powershell
python backend\mcp_servers\kg\server.py --transport http --host 127.0.0.1 --port 8001
```

`http` is an alias for `streamable-http`.

## Smoke Test

The smoke test calls the same KG tool wrapper directly, without going through the MCP protocol:

```powershell
python backend\scripts\test_kg_mcp_tools.py
```

Custom input:

```powershell
python backend\scripts\test_kg_mcp_tools.py --disease 感冒 --symptom 腹痛 --symptoms 咽痛 鼻塞 流鼻涕
```

## Tools

### `kg_ping`

Checks whether Neo4j is reachable.

### `kg_get_disease_profile`

Returns whitelisted attributes and relations for a disease.

Input:

```json
{"disease_name": "感冒"}
```

### `kg_get_disease_attribute`

Returns one disease attribute.

Supported fields:

- `疾病简介`
- `疾病病因`
- `预防措施`
- `治疗周期`
- `治愈概率`
- `疾病易感人群`

### `kg_get_related_entities`

Returns related entities for a whitelisted disease relation.

Supported relation targets:

- `疾病使用药品 -> 药品`
- `疾病宜吃食物 -> 食物`
- `疾病忌吃食物 -> 食物`
- `疾病所需检查 -> 检查项目`
- `疾病所属科目 -> 科目`
- `疾病的症状 -> 疾病症状`
- `治疗的方法 -> 治疗方法`
- `疾病并发疾病 -> 疾病`

### `kg_get_diseases_by_symptom`

Returns diseases connected to a symptom through `疾病的症状`.

### `kg_get_disease_candidates_by_symptoms`

Returns KG disease candidates ranked by matched symptom count.

Input:

```json
{
  "symptoms": ["咽痛", "鼻塞", "流鼻涕"],
  "limit": 10
}
```

### `kg_get_symptom_disease_counts`

Returns how many diseases are connected to each symptom. This helps estimate symptom specificity.

### `kg_get_producer_by_drug`

Returns producers connected to a drug through `生产`.

### `kg_inspect_entity`

Inspects labels, degrees, and relation samples for an entity name. This is useful for finding KG type conflicts such as the same name appearing as both `疾病` and `疾病症状`.

## Resources

### `medgraphqa://kg/schema`

Returns KG labels, whitelisted fields, relation targets, and limits.

### `medgraphqa://kg/disease/{name}`

Returns a disease profile as JSON.

### `medgraphqa://kg/symptom/{name}/diseases`

Returns diseases connected to a symptom as JSON.

## Safety Boundaries

First version constraints:

- Read-only.
- No raw Cypher tool.
- No write operations.
- Attribute and relation queries are allowlisted.
- Entity names are limited to 80 characters.
- Symptom lists are limited to 20 items.
- `limit` is capped at 50.
- Errors are returned as structured JSON instead of exposing tracebacks.

Do not add a general `run_cypher` tool without a separate security design.
