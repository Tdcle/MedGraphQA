# MedGraphQA

MedGraphQA 是一个面向医疗知识图谱的智能问答系统。项目不是让大模型自由生成医疗答案，而是以 Neo4j 医疗知识图谱为事实来源，通过实体标准化、临床上下文抽取、疾病候选推断、多轮追问、安全护栏、长期记忆、评测和监控，把医疗问答流程拆成可解释、可评测、可观测的工程链路。

> 说明：本项目用于医疗知识图谱问答和工程实践展示，不提供医学诊断服务，不能替代医生面诊。

## 核心能力

- 口语实体标准化：将“肚子疼”“拉肚子”“喉咙痛”等用户表达映射到 KG 标准实体。
- 混合检索：Postgres 别名精确匹配 + Elasticsearch 全文检索 + 向量召回 + RRF 融合。
- RoBERTa NER：加载 `best_roberta_rnn_model_ent_aug.pt` 抽取用户问题中的实体片段。
- 临床上下文抽取：识别症状、否定症状、时间、程度、进展、已知疾病等信息。
- 多轮追问：当症状不足以确定疾病候选时，根据候选疾病差异生成追问。
- KG 问答：从 Neo4j 查询疾病简介、病因、治疗方法、药品、检查项目、科目、并发症等。
- 长期记忆：保存用户确认后的稳定信息，例如过敏史、慢性病史、长期用药、孕期、偏好。
- 安全护栏：规则拦截急症、自伤/中毒、prompt injection、明确非医疗问题，并限制输出中的确诊式和保证式表达。
- SSE 流式交互：前端显示处理中状态和回答内容。
- 评测体系：生成 KG 派生评测集，评估实体、症状、否定症状、疾病 TopK 召回、追问轮数和安全护栏。
- 监控体系：Prometheus + Grafana 监控 HTTP、Chat、LLM、Embedding、ES、KG、Safety、Memory 等指标。
- MCP Server：以 MCP 工具形式暴露只读 KG 查询能力。

## 架构概览

```text
Vue 3 + Arco Design
  |
  | HTTP / SSE
  v
FastAPI Backend
  |
  |-- Auth / Session / Memory
  |-- LangGraph QA Pipeline
  |-- Safety Guardrails
  |-- Metrics
  |
  |-- Postgres
  |     |-- 用户、会话、实体别名、长期记忆
  |
  |-- Elasticsearch + Kibana
  |     |-- medical_entity_alias
  |     |-- 全文检索、向量检索
  |
  |-- Neo4j
  |     |-- 医疗知识图谱
  |
  |-- Ollama / DashScope
        |-- 回答生成
        |-- 临床上下文结构化抽取
        |-- 追问生成

Prometheus / Grafana
  |-- /metrics
  |-- 性能、错误率、模型耗时、检索耗时
```

## 技术栈

后端：

- FastAPI
- LangGraph
- Pydantic
- psycopg / psycopg_pool
- Elasticsearch Python Client
- py2neo
- prometheus_client
- MCP Python SDK

前端：

- Vue 3
- Arco Design Vue
- Vite
- Axios
- Fetch ReadableStream / SSE

模型与检索：

- RoBERTa + RNN NER 微调模型
- Ollama 本地模型，例如 `qwen3:4b`、`qwen3:8b`
- Ollama embedding 模型，例如 `bge-m3`
- DashScope OpenAI 兼容接口，可选

数据存储：

- Neo4j
- Postgres
- Elasticsearch
- Kibana

监控：

- Prometheus
- Grafana

## 目录结构

```text
MedGraphQA/
├─ backend/
│  ├─ app/
│  │  ├─ api/routes/          # FastAPI 路由
│  │  ├─ core/                # 配置、容器、日志、request_id
│  │  ├─ middleware/          # 请求日志中间件
│  │  ├─ schemas/             # 请求/响应模型
│  │  └─ services/            # 问答、检索、KG、记忆、护栏、评测辅助服务
│  ├─ evals/                  # 核心评测、安全评测、数据集生成
│  ├─ mcp_servers/            # MCP Server
│  ├─ scripts/                # 图谱导入、实体检索初始化、连接测试
│  ├─ config.json             # 非敏感配置
│  ├─ requirements.txt
│  └─ .env.example
├─ frontend/
│  ├─ src/
│  │  ├─ api/                 # Axios client
│  │  ├─ assets/              # 用户/助手头像
│  │  ├─ router/              # 路由守卫
│  │  ├─ stores/              # token/user 本地存储
│  │  └─ views/               # 登录、注册、聊天页面
│  ├─ package.json
│  └─ vite.config.js
├─ data/
│  └─ ent_aug/                # KG 实体词典
├─ model/
│  ├─ best_roberta_rnn_model_ent_aug.pt
│  └─ chinese-roberta-wwm-ext/
├─ monitoring/
│  ├─ prometheus/
│  └─ grafana/
├─ docker/
├─ docker-compose.search.yml
└─ README.md
```

## 环境要求

- Python 3.10+
- Node.js 18+
- Docker / Docker Compose
- Neo4j 5.x
- Ollama
- 可选：DashScope API Key

推荐在项目根目录执行命令：

```powershell
cd D:\PythonProject\MedGraphQA
```

## 依赖服务启动

Postgres、Elasticsearch、Kibana、Prometheus、Grafana 已写入 `docker-compose.search.yml`。

```powershell
docker compose -f docker-compose.search.yml up -d postgres elasticsearch kibana
```

如需监控：

```powershell
docker compose -f docker-compose.search.yml up -d prometheus grafana
```

默认端口：

- Postgres: `5432`
- Elasticsearch: `9200`
- Kibana: `5601`
- Prometheus: `9090`
- Grafana: `3001`

Neo4j 需要单独启动，默认配置为：

- URI: `bolt://localhost:7687`
- Database: `neo4j`
- User: `neo4j`

Ollama 默认地址：

```text
http://localhost:11434
```

常用模型示例：

```powershell
ollama pull qwen3:8b
ollama pull qwen3:4b
ollama pull bge-m3
```

## 后端配置

复制环境变量模板：

```powershell
Copy-Item backend\.env.example backend\.env
```

`backend/.env` 用于存敏感信息：

```dotenv
NEO4J_PASSWORD=你的Neo4j密码
DASHSCOPE_API_KEY=sk-your-dashscope-api-key
EMBEDDING_API_KEY=
INTENT_DASHSCOPE_API_KEY=
CLINICAL_CONTEXT_API_KEY=
SECRET_KEY=请替换为随机长字符串
DEFAULT_ADMIN_USERNAME=admin
DEFAULT_ADMIN_PASSWORD=admin123
POSTGRES_DSN=postgresql://postgres:postgres@localhost:5432/medgraphqa
ELASTICSEARCH_HOSTS=http://localhost:9200
ELASTICSEARCH_USERNAME=
ELASTICSEARCH_PASSWORD=
```

`backend/config.json` 用于非敏感配置，例如：

- Neo4j 地址、数据库名。
- Postgres 连接池。
- Elasticsearch index。
- RoBERTa NER 模型路径。
- embedding provider/model。
- clinical context 模型。
- LLM provider/model。
- 疾病置信度阈值。
- 日志和 CORS。

当前默认：

- 主回答模型：Ollama `qwen3:8b`
- 临床上下文模型：Ollama `qwen3:4b`
- embedding：Ollama `bge-m3`
- 向量检索：开启
- clinical context thinking：关闭

## 初始化实体检索数据

创建 Postgres 表、导入 `data/ent_aug` 实体词、创建 ES index 并同步索引：

```powershell
python backend\scripts\manage_entity_search.py --with-embeddings rebuild --clear --recreate
```

如果暂时不想生成 embedding，可去掉 `--with-embeddings`：

```powershell
python backend\scripts\manage_entity_search.py rebuild --clear --recreate
```

分步执行：

```powershell
python backend\scripts\manage_entity_search.py init-db
python backend\scripts\manage_entity_search.py import-entities --clear
python backend\scripts\manage_entity_search.py create-index --recreate
python backend\scripts\manage_entity_search.py --with-embeddings sync-index
```

## 导入 Neo4j 知识图谱

如果需要从 `data/medical_new_2.json` 重建图谱：

```powershell
python backend\scripts\import_graph.py --uri bolt://localhost:7687 --user neo4j --password <你的Neo4j密码> --database neo4j --clear
```

## 启动后端

安装依赖：

```powershell
cd backend
pip install -r requirements.txt
```

启动：

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

后端地址：

```text
http://localhost:8000
```

常用接口：

- `GET /api/health`
- `POST /api/auth/register`
- `POST /api/auth/login`
- `POST /api/auth/logout`
- `GET /api/auth/me`
- `POST /api/chat/ask`
- `POST /api/chat/ask/stream`
- `GET /api/chat/sessions`
- `GET /api/memories`
- `GET /metrics`

## 启动前端

```powershell
cd frontend
npm install
npm run dev
```

前端地址：

```text
http://localhost:5173
```

前端默认 API base：

```dotenv
VITE_API_BASE=/api
```

开发时如果没有代理，也可以在 `frontend/.env` 中改成：

```dotenv
VITE_API_BASE=http://localhost:8000/api
```

## 默认账号

`manage_entity_search.py init-db` 会根据环境变量创建默认管理员。

默认值：

- 用户名：`admin`
- 密码：`admin123`

建议首次启动后修改默认密码，并在非本地环境中替换 `SECRET_KEY`。

## 问答主流程

简化链路：

```text
用户输入
  |
  |-- 输入安全护栏
  |-- 会话状态载入
  |-- 意图识别
  |-- RoBERTa NER 抽取实体片段
  |-- 临床上下文抽取
  |-- 实体标准化
  |     |-- Postgres 精确匹配
  |     |-- Elasticsearch 全文检索
  |     |-- Elasticsearch 向量检索
  |     |-- RRF 融合
  |-- 长期记忆读取
  |-- Neo4j KG 查询
  |-- 疾病候选置信度判断
  |     |-- 置信度不足：追问
  |     |-- 置信度足够：生成回答
  |-- 输出安全护栏
  |-- 长期记忆候选抽取
  |-- 对话持久化
```

## 评测

生成核心评测数据：

```powershell
python backend\evals\generate_eval_dataset.py --output backend\evals\datasets --limit 100
```

运行核心评测：

```powershell
python backend\evals\run_core_eval.py --dataset-dir backend\evals\datasets
```

小样本调试：

```powershell
python backend\evals\run_core_eval.py --dataset-dir backend\evals\datasets --max-cases 10
```

运行安全护栏评测：

```powershell
python backend\evals\run_safety_eval.py
```

报告输出：

```text
backend/evals/runs/core_eval_*.json
backend/evals/runs/core_eval_history.jsonl
backend/evals/runs/safety_eval_*.json
backend/evals/runs/safety_eval_history.jsonl
```

核心指标：

- `disease_top5_recall`
- `positive_symptom_recall`
- `clinical_negation_recall`
- `negated_symptom_false_positive_rate`
- `follow_up_over_limit_count`
- `clinical_context.llm_call_rate`

## 监控

后端暴露 Prometheus 指标：

```text
GET /metrics
```

启动 Prometheus 和 Grafana：

```powershell
docker compose -f docker-compose.search.yml up -d prometheus grafana
```

访问：

- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3001`

Grafana 默认账号：

- 用户名：`admin`
- 密码：`admin`

第一版监控覆盖：

- HTTP 请求量、错误率、耗时。
- Chat 主流程耗时。
- LangGraph node 耗时。
- LLM generate / generate_json / stream 耗时。
- embedding 耗时。
- entity search 召回统计。
- KG 查询耗时。
- safety guardrail 命中。
- memory 读取和写入。

## KG MCP Server

项目提供只读 KG 查询 MCP Server。

stdio 模式：

```powershell
python backend\mcp_servers\kg\server.py --transport stdio
```

HTTP 模式：

```powershell
python backend\mcp_servers\kg\server.py --transport http --host 127.0.0.1 --port 8001
```

Smoke test：

```powershell
python backend\scripts\test_kg_mcp_tools.py
```

MCP 工具示例：

- `kg_ping`
- `kg_get_disease_profile`
- `kg_get_disease_attribute`
- `kg_get_related_entities`
- `kg_get_diseases_by_symptom`
- `kg_get_disease_candidates_by_symptoms`
- `kg_get_symptom_disease_counts`
- `kg_get_producer_by_drug`
- `kg_inspect_entity`

## 常用测试命令

测试 DashScope 连接：

```powershell
python backend\scripts\test_dashscope_connection.py --clinical-context
```

测试后端编译：

```powershell
python -m compileall backend\app
```

构建前端：

```powershell
cd frontend
npm run build
```

## 安全边界

- 系统不能替代医生诊断。
- 候选疾病不是确诊结果。
- 普通用药问题可以回答 KG 中的相关药品信息，但不应给出处方、剂量或保证疗效。
- 出现胸痛、呼吸困难、意识异常、严重过敏、大出血、呕血/便血、剧烈腹痛、持续高热等危险信号时，应优先就医或急诊。
- 开发环境可开启 chat trace，生产环境应关闭或做脱敏处理。

## 已知短板

- 当前评测数据主要来自 KG 弱监督，不等同于临床金标准。
- 疾病常见性先验仍有工程启发式成分。
- KG 数据质量会显著影响疾病候选排序。
- 前端目前偏工程演示，普通用户视图仍可继续优化。
- 本地 Ollama 模型速度受硬件影响较大。
- 意图识别配置中预留了 hybrid/LLM 项，当前主链路仍以规则意图识别为主。
