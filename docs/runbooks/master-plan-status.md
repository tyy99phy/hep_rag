# hep_rag 主计划状态与夜间执行说明

本文档把 `.omx/plans/2026-04-01-hep-rag-master-plan.md` 映射到当前仓库实现，便于夜间执行时快速判断：

- 当前已经稳定存在什么
- 哪些 Step 仍未落地
- 下一步应优先推进哪里
- 现阶段用哪些命令做最小可重复验证

## 1. 当前已验证基线

当前仓库仍然是 **relational-first 的 SQLite substrate**，核心能力集中在：

- `works / work_ids / citations / documents / chunks / work_families`
- 在线检索与元数据入库
- PDF 下载与 MinerU 全文解析
- BM25 / 向量 / 图结构派生层
- `query` / `ask` 两条 paper/chunk 检索链路

这与 master plan 的大方向一致，但仍然停留在 **paper/chunk retrieval substrate**，尚未进入 `pdg/result/method` typed-object 阶段。

## 2. 当前代码质量结论（对应主计划）

### 2.1 Step 1 尚未完成：默认流程仍然是全量 rebuild 驱动

当前 `ingest-online` 与 `reparse-pdfs` 结束后，默认会直接重建搜索、向量与图层：

- `src/hep_rag_v2/pipeline.py:1271-1302`
- `src/hep_rag_v2/pipeline.py:1431-1462`

这说明仓库还没有落地 master plan 所要求的：

- `dirty_objects`
- `maintenance_jobs`
- `sync-search / sync-vectors / sync-graph`
- 按 scope / collection / updated_since 的增量同步

### 2.2 Step 2 尚未开始：agent-facing shell 仍缺统一抽象

当前仓库尚不存在以下计划文件：

- `src/hep_rag_v2/retrieval_adapter.py`
- `src/hep_rag_v2/evidence.py`
- `src/hep_rag_v2/tools/registry.py`
- `src/hep_rag_v2/service/facade.py`
- `src/hep_rag_v2/service/factory.py`

`retrieve()` 与 `ask()` 仍直接围绕 `works/chunks` 工作，尚未形成统一 retrieval/evidence/service assembly contract。

### 2.3 Step 3 尚未开始：cold metadata benchmark 还没有基准执行面

当前仓库中不存在 master plan 指定的 benchmark 入口：

- `src/hep_rag_v2/loadtest.py`
- `scripts/run_cms_lhc_1000.py`
- `scripts/run_scale_benchmark.py`
- `tests/test_loadtest.py`

这意味着目前还不能对 `10k / 50k / 100k metadata-only lane` 做一条命令复现的基准验证。

### 2.4 Step 4-10 也仍未落地，应在 Step 1-3 稳定后再推进

当前仓库也还没有以下主计划模块：

- PDG spine：`pdg.py` / `providers/pdg.py`
- result layer：`results.py` / `result_extract.py`
- method layer：`methods.py` / `method_extract.py`
- answer shell：`answer_planner.py` / `answer_reporter.py`
- validation / cache：`validation.py` / `cache.py`

因此当前最安全的结论是：

> **仓库主干仍是 paper/chunk-first 的检索系统，尚未切换到 PDG/result/method-first 的 typed retrieval substrate。**

## 3. Step 状态表

| Step | 目标 | 当前状态 | 仓库证据 | 建议优先级 |
| --- | --- | --- | --- | --- |
| Step 1 | 增量维护 substrate | 未完成 | `pipeline.py` 默认全量 rebuild | 最高 |
| Step 2 | 统一 agent-facing shell | 未开始 | 相关文件缺失 | 最高 |
| Step 3 | cold metadata benchmark | 未开始 | benchmark 脚本/测试缺失 | 高 |
| Step 4 | 最小 PDG spine | 未开始 | PDG 模块缺失 | 中 |
| Step 5 | typed result objects | 未开始 | result 模块缺失 | 中 |
| Step 6 | typed method objects | 未开始 | method 模块缺失 | 中 |
| Step 7 | hot-layer evidence | 未开始 | 仍是全文默认深加工导向 | 中低 |
| Step 8 | planner/executor/reporter | 未开始 | `ask()` 仍是 retrieve + prompt | 中低 |
| Step 9 | consistency + cache | 未开始 | validation/cache 缺失 | 低 |
| Step 10 | HEP-native eval | 未开始 | eval 目录与 runbook 缺失 | 低 |

## 4. 当前推荐的最小安全推进顺序

### 4.1 先完成 Step 1

先把 ingest / reparse 从“默认全量 rebuild”改为“写 dirty marks + 显式 sync”：

1. schema 增加 maintenance tables
2. pipeline 只登记 dirty scope
3. CLI 增加 `sync-search` / `sync-vectors` / `sync-graph`
4. 补覆盖 ingest/reparse/sync 行为的测试

### 4.2 再完成 Step 2

在 substrate 不再频繁变形后，再抽统一 agent-facing shell：

1. 定义统一 retrieval result 数据结构
2. 抽 evidence registry / provenance dedupe
3. 抽 service facade / factory
4. 让 API / pipeline 通过 facade 装配

### 4.3 然后做 Step 3

等 Step 1-2 接口稳定后，再补 benchmark 面：

1. 生成 metadata-only 数据集
2. 固化 `10k / 50k / 100k` 三档命令
3. 把输出写到 workspace exports
4. 把结果沉淀到 runbook

## 5. 当前可重复验证命令

这些命令验证的是 **当前基线仍然工作**，而不是表示 master plan 已完成：

```bash
python -m compileall src tests
pytest
pytest tests/test_bootstrap.py tests/test_vector.py
```

如果后续开始实施 Step 1-3，建议新增并持续维护以下验证面：

- Step 1：ingest/reparse 后 dirty mark 与 sync command 行为
- Step 2：typed retrieval/evidence/facade contract 测试
- Step 3：benchmark 命令可复现性与产物落盘检查

## 6. 夜间执行摘要模板

夜间执行结束时，建议按下面三段式输出，避免把“规划中”误报成“已完成”：

1. **已完成项**：明确到文件/命令/测试
2. **延期项**：说明停在哪个 Step、为什么暂缓
3. **验证证据**：列出命令、结果与失败/跳过原因

推荐摘要句式：

> 已完成：……  
> 延期：……  
> 验证：`命令 -> 结果`
