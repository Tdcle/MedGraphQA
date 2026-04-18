# MedGraphQA Eval Harness

这个目录用于把 MedGraphQA 从“能跑”推进到“可评测、可回放、可比较”。

当前优先级：

1. 构建固定评测集，覆盖实体标准化、临床上下文、疾病候选、追问策略和最终回答。
2. 保存每轮对话的结构化 trace，支持失败样例回放。
3. 把追问策略、疾病置信度阈值、可能疾病阈值变成可评测 policy。
4. 做 KG 数据质量检查，重点清理“疾病/症状”跨类型污染。
5. 接入报告输出，逐步对齐 Prometheus/Grafana 或 CI gate。

## 数据生成

第一版先只生成两个核心数据集，避免链路过重：

```powershell
cd D:\PythonProject\MedGraphQA
python backend\evals\generate_eval_dataset.py --output backend\evals\datasets --limit 100
```

默认会连接当前 `backend/config.json` 和 `.env` 中配置的 Neo4j/Postgres。

默认生成的数据集：

- `core_single_turn.jsonl`: 单轮核心评测，覆盖正向症状、否定症状过滤、疾病 TopK 召回。
- `core_multi_turn.jsonl`: 多轮核心评测，覆盖上下文合并、否定症状过滤、追问上限、可能疾病收敛。
- `manifest.json`: 本次生成统计和来源说明。

如果需要更细的辅助数据集，再加 `--full`：

```powershell
python backend\evals\generate_eval_dataset.py --output backend\evals\datasets --limit 100 --full
```

`--full` 会额外生成：

- `entity_normalization.jsonl`
- `clinical_context.jsonl`
- `disease_resolution.jsonl`
- `answer_grounding.jsonl`

默认生成前会清理旧的 JSONL 文件，避免核心集和历史细分集混在一起。需要保留旧文件时使用 `--keep-existing`。

这些 JSONL 是评测输入，不是医学建议。

## 核心评测

快速 smoke test，不调用临床上下文 LLM：

```powershell
python backend\evals\run_core_eval.py --dataset-dir backend\evals\datasets --max-cases 5 --skip-clinical-llm
```

正式核心评测：

```powershell
python backend\evals\run_core_eval.py --dataset-dir backend\evals\datasets
```

正式评测会调用当前配置的临床上下文抽取模型，耗时取决于 `clinical_context` 配置。建议先用 `--max-cases 10` 小批量验证。

输出报告位于：

```text
backend/evals/runs/core_eval_*.json
```

每次成功评测后还会追加一行历史记录：

```text
backend/evals/runs/core_eval_history.jsonl
```

该文件用于观察指标趋势，包含本次耗时、报告路径、临床上下文模型配置、单轮/多轮核心指标。需要写到其他位置时：

```powershell
python backend\evals\run_core_eval.py --dataset-dir backend\evals\datasets --history-file backend\evals\runs\my_history.jsonl
```

核心指标：

- `positive_symptom_recall`: 正向症状召回。
- `clinical_negation_recall`: 否定症状抽取召回。
- `negated_symptom_false_positive_rate`: 被否认的症状被错误当作正向症状的比例。
- `disease_top5_recall`: 目标疾病是否进入 Top5 候选。
- `follow_up_over_limit_count`: 多轮对话是否超过追问上限。

## 安全护栏评测

规则性护栏评测不调用 KG、NER 或 LLM，只验证 `RuleBasedSafetyGuard` 的输入拦截和输出重写逻辑。

```powershell
python backend\evals\run_safety_eval.py
```

默认读取：

```text
backend/evals/datasets/safety_guardrails.jsonl
```

输出报告位于：

```text
backend/evals/runs/safety_eval_*.json
```

每次成功评测后会追加历史记录：

```text
backend/evals/runs/safety_eval_history.jsonl
```

常用调试命令：

```powershell
python backend\evals\run_safety_eval.py --max-cases 10
python backend\evals\run_safety_eval.py --no-progress
python backend\evals\run_safety_eval.py --dataset backend\evals\datasets\safety_guardrails.jsonl
```

安全护栏核心指标：

- `input_category_accuracy`: 输入安全分类准确率。
- `action_accuracy`: `continue` / `direct_response` 动作准确率。
- `expected_hit_code_recall`: 期望触发规则的召回率。
- `forbidden_hit_violation_rate`: 不应触发规则的误触发比例。
- `direct_response_recall`: 应直接安全回答的样例召回率。
- `continue_pass_rate`: 应继续主流程的样例放行率。
- `unsafe_output_recall`: 不安全输出被识别的比例。
- `safe_output_pass_rate`: 安全输出被放行的比例。
- `rewrite_success_rate`: 不安全输出重写后通过二次安全检查的比例。
