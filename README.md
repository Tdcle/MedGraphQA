# MedGraphQA (FastAPI + Vue)

本项目已重构为前后端分离架构：

- 后端：FastAPI
- 前端：Vue 3 + Arco Design
- 知识库：Neo4j
- 生成模型服务：Ollama 或 千问 API（DashScope OpenAI 兼容模式）

## 目录结构

```text
MedGraphQA/
├─ backend/
│  ├─ app/
│  │  ├─ api/routes/      # 路由
│  │  ├─ core/            # 配置与容器
│  │  ├─ schemas/         # 请求/响应模型
│  │  └─ services/        # 认证、实体识别、意图识别、图谱查询、问答
│  ├─ scripts/import_graph.py
│  ├─ config.json
│  ├─ requirements.txt
│  └─ .env.example
├─ frontend/
│  ├─ src/
│  │  ├─ api/             # axios 客户端
│  │  ├─ router/          # 路由守卫
│  │  ├─ stores/          # 本地 token/user 存储
│  │  └─ views/           # 登录页、聊天页
│  ├─ package.json
│  └─ vite.config.js
└─ data/                  # 原始数据与实体词典
```

## 环境要求

- Python 3.10+
- Node.js 18+
- Neo4j 5.x
- Ollama（默认）或阿里云百炼 / DashScope API Key

## 配置

非敏感配置写在 `backend/config.json`，例如：

- Neo4j 地址、用户名、数据库名
- LLM provider、模型名、温度、最大输出长度
- 日志级别、日志目录、保留天数
- CORS、token 过期时间、本地路径

隐私配置写在 `backend/.env`：

```dotenv
NEO4J_PASSWORD=你的Neo4j密码
DASHSCOPE_API_KEY=你的DashScope API Key（仅 provider=dashscope 时需要）
SECRET_KEY=请替换为随机长字符串
```

默认使用 Ollama：

```json
"llm": {
  "provider": "ollama",
  "ollama": {
    "api_base": "http://localhost:11434",
    "model": "qwen3:8b"
  }
}
```

切换到千问 API 时，把 `provider` 改成 `dashscope`，并设置 `DASHSCOPE_API_KEY`。

日志默认写入 `backend/logs/app.log` 和 `backend/logs/access.log`，同时输出到控制台。每个请求都会带 `X-Request-ID` 响应头，便于按请求排查问题。

## 后端启动

```bash
cd backend
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

后端默认地址：`http://localhost:8000`

### 主要接口

- `POST /api/auth/register` 注册
- `POST /api/auth/login` 登录
- `GET /api/auth/me` 当前用户
- `POST /api/chat/ask` 问答（需登录）
- `GET /api/health` 健康检查

## 前端启动

```bash
cd frontend
npm install
npm run dev
```

前端默认地址：`http://localhost:5173`

## 图谱导入

如果需要从 `data/medical_new_2.json` 重建图谱：

```bash
cd backend
python scripts/import_graph.py --uri bolt://localhost:7687 --user neo4j --password <你的密码> --database neo4j --clear
```

## 默认账号

首次启动后端会自动创建默认管理员：

- 用户名：`admin`
- 密码：`admin123`

建议登录后自行改造为更安全的认证流程。
