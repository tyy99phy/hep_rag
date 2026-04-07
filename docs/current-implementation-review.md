# 当前版本实现总览与 Review 指南

> 对齐版本：`origin/main` / `8f53e9d`  
> 对齐日期：2026-04-07  
> 文档目的：把当前仓库**已经实现并已落到远端主分支**的功能、代码入口、测试证据、边界与风险集中总结，方便后续做人工 review。

---

## 0. 一句话结论

当前版本已经从“单纯抓论文 + 检索”推进到了一个**可运行的 metadata-first / selective-fulltext / typed-retrieval substrate**：

- 可以从 InspireHEP 检索并入库论文元数据；
- 可以下载 PDF、调用 MinerU、把解析结果 materialize 成 sections / blocks / chunks；
- 可以做 BM25、向量、混合检索；
- 可以输出 typed retrieval shell 和 evidence registry；
- 可以跑基础 API / Web UI；
- 可以跑 scale benchmark 和“弱模型接不接数据库”的 benchmark scaffolding；
- 已经补上了 PDG 这一波次最关键的两个入口：
  1. **官方完整 PDG PDF 的稳定下载 URL**；
  2. **本地 PDG MinerU bundle 的直接导入入口**。

但它**还不是最终版的 PDG/work 双层高能物理知识图谱**。当前更准确的定位是：

- `PDG/work` 新架构的**骨架和约束已经开始落地**；
- `work/result/method` 默认层的最小结构化实现已经存在；
- 但离“完整 PDG spine + work 深层 typed graph + query harness + 全量真实数据压测”还有明显距离。

---

## 1. 最近这一波到底改了什么

如果只看当前主干上最关键的一串提交，主线大致是：

1. `de7c8a8`  
   把检索底座、typed retrieval shell、evaluation scaffolding、规模 benchmark 脚手架推到可用状态。
2. `59fb2dc` / `f677bc0`  
   清理公开文档，把内部 runbooks 从用户面移开，只保留用户真正会用到的测试说明。
3. `e42a241`  
   新增 `docs/pdg-work-implementation.md`，把 PDG/work 波次的硬约束、落点和验收方向写清楚。
4. `44cb1f3`  
   把 build / retrieval / embedding / profile 相关配置显式化，减少 silent fallback。
5. `c642361`  
   让 PDG archival ingest 能先通起来，即使完整结构抽取还没全部接上。
6. `13c52bd`  
   修掉 PDG 入口最后两个关键问题：
   - `import-pdg` 不再只接受 `--edition`，现在也支持**直接导入本地 PDG MinerU bundle**；
   - 默认 PDG 下载地址改成官方完整 PDF：`https://pdg.lbl.gov/<year>/download/db<year>.pdf`，不再指向已经失效的 intro PDF 路径。

换句话说，**当前主分支的重点不是“做了一个华丽的新 graph 产品”**，而是把后续真正做大规模 HEP RAG 所必须的几个底座先落稳：

- 数据进库路径；
- 显式配置与模式；
- typed retrieval / evidence shell；
- PDG 接入骨架；
- benchmark 脚手架；
- 面向用户的可运行入口。

---

## 2. 当前版本的系统能力总表

### 2.1 在线论文侧

当前可以：

- 从 InspireHEP 做在线检索；
- 做 query rewrite；
- 做 family-aware 合并 / 去重；
- metadata 入库到 SQLite；
- 下载 PDF；
- 调 MinerU 解析；
- materialize 为 document / section / block / chunk；
- 建 BM25 / vector / graph 索引；
- 做 query / ask；
- 从 Web/API 走一遍同样流程。

对应主入口：

- `src/hep_rag_v2/pipeline.py`
- `src/hep_rag_v2/metadata.py`
- `src/hep_rag_v2/fulltext/*`
- `src/hep_rag_v2/search.py`
- `src/hep_rag_v2/vector/*`
- `src/hep_rag_v2/graph.py`
- `src/hep_rag_v2/cli/*`
- `src/hep_rag_v2/service/*`

### 2.2 检索 / 问答侧

当前可以：

- 输出 work 命中；
- 输出 chunk 命中；
- 输出 typed retrieval metadata；
- 输出 evidence registry；
- 让 `ask` 在 evidence 之上生成答案；
- 通过 facade / tool registry 暴露统一服务接口。

对应主入口：

- `src/hep_rag_v2/retrieval_adapter.py`
- `src/hep_rag_v2/evidence.py`
- `src/hep_rag_v2/service/facade.py`
- `src/hep_rag_v2/service/factory.py`

### 2.3 PDG 侧

当前已经实现两条路径：

1. **archival ingest 路径**  
   输入 `edition + 本地 PDF / 远端下载`，先把 PDG 作为一个 canonical work 注册进主库。
2. **local parsed source 路径**  
   输入本地 PDG MinerU bundle，直接生成 `pdg_sources / pdg_sections`，产出 section capsule。

对应主入口：

- `src/hep_rag_v2/providers/pdg.py`
- `src/hep_rag_v2/pdg.py`
- `src/hep_rag_v2/pipeline.py` 中的 `import_pdg(...)`
- `src/hep_rag_v2/cli/_parser.py`
- `src/hep_rag_v2/cli/ingest.py`

### 2.4 结构层 / 默认层

当前已经有一个最小可运行结构层：

- 新表 `work_capsules`
- 对非综述文章抽 `result_signature` / `method_signature`
- 如果缺必需签名，不是静默丢掉，而是标成 `needs_attention`
- 综述文章允许 `review_relaxed`

对应实现：

- `src/hep_rag_v2/structure.py`
- `src/hep_rag_v2/schema.sql`
- `tests/test_pdg_structure.py`

### 2.5 benchmark 侧

当前已经有两套脚手架：

1. **规模 benchmark**：测 10k / 50k / 100k metadata-only 数据量下的建库与检索延迟；
2. **RAG 增益 benchmark**：比较弱模型在 `llm_only / llm_plus_retrieve / llm_plus_retrieve_and_structure` 三种场景下的表现。

对应实现：

- `src/hep_rag_v2/loadtest.py`
- `scripts/run_scale_benchmark.py`
- `src/hep_rag_v2/benchmark_suite.py`
- `scripts/run_rag_effect_benchmark.py`
- `tests/test_loadtest.py`
- `tests/test_benchmark_suite.py`

---

## 3. 代码结构分块 review

## 3.1 配置与模式层

### 关键文件

- `src/hep_rag_v2/config.py`
- `config.example.yaml`
- `tests/test_config_runtime.py`

### 当前实现了什么

`config.py` 现在已经不只是“读配置文件”，而是承担了这几个作用：

- 提供完整默认配置；
- 支持 `build` / `retrieval` 显式 mode；
- 支持 `profiles.structure` / `profiles.embedding` / `profiles.pdg`；
- 提供多套 embedding profile；
- 显式把 `allow_silent_fallback` 设为 `False`；
- 支持在无 config 路径时按 workspace 生成默认配置。

### 当前默认 profile 值

`DEFAULT_EMBEDDING_PROFILES` 里已经有三档：

- `bootstrap`：`hash-idf-v1`，CPU，最轻；
- `semantic_small_local`：`BAAI/bge-small-en-v1.5`，CUDA；
- `semantic_prod_local`：`BAAI/bge-base-en-v1.5`，CUDA。

### 这意味着什么

这部分已经明显体现出此前讨论过的方向：

- **结构构建模型**和**embedding 模型**分工；
- 用 profile 显式切换，而不是到处暗藏 fallback；
- 为单卡 3090 本地 embedding 预留了实际入口。

### 仍然没做到的地方

- 结构构建 backend 目前只是配置上显式化了，还没形成完整的生产级 structured extraction pipeline；
- `hash-idf-v1` 仍然保留在 bootstrap profile 里，没有被完全淘汰；
- 还没有把 “semantic_only / structure_only / hybrid” 的全部运行分叉做成完整 end-to-end 产品路径。

---

## 3.2 在线 ingest / PDF / MinerU 主流程

### 关键文件

- `src/hep_rag_v2/pipeline.py`
- `src/hep_rag_v2/cli/ingest.py`
- `src/hep_rag_v2/cli/_parser.py`
- `src/hep_rag_v2/fulltext/*`
- `tests/test_incremental_reparse.py`
- `tests/test_mineru_api.py`
- `tests/test_online_search.py`

### 当前实现了什么

这是仓库最成熟的一段：

- `fetch-papers`：只做在线检索预览；
- `ingest-online`：做 search → metadata ingest → PDF 下载 → MinerU 解析 → 索引 / 图同步；
- `reparse-pdfs`：对本地缓存 PDF 做增量重解析；
- `import-mineru`：手工导入已有 MinerU 输出。

### 当前的实际特点

- 并不是所有文章都必须立刻走 fulltext；
- 支持 metadata-first，再 selective fulltext；
- 这与“100k 级 paper 不要一开始就把全文都拖进重流程”这个担忧是对齐的。

### 风险/边界

- 真正 100k 级真实 PDF + 真 MinerU 成本，目前还没有在这个 session 里完成实测；
- 当前仓库更适合先验证 metadata-first + selective fulltext 的分层可行性，而不是直接宣称能吃下全部真实 HEP 语料。

---

## 3.3 typed retrieval / evidence shell

### 关键文件

- `src/hep_rag_v2/retrieval_adapter.py`
- `src/hep_rag_v2/evidence.py`
- `src/hep_rag_v2/service/facade.py`
- `src/hep_rag_v2/service/factory.py`
- `tests/test_retrieval_adapter.py`
- `tests/test_evidence.py`
- `tests/test_retrieval_shell.py`

### 当前实现了什么

这里是这次版本很重要的一点：**把检索输出变成可组合、可审计的 typed shell，而不是只扔一堆散 JSON。**

`retrieval_adapter.py` 现在有：

- `TypedRetrievalMetadata`
- `TypedRetrievalResult`
- `adapt_work_hit(...)`
- `adapt_chunk_hit(...)`
- `normalize_retrieval_payload(...)`
- `build_retrieval_shell(...)`

`evidence.py` 里有：

- `EvidenceEntry`
- `EvidenceRegistry`
- work / chunk 分型注册；
- 统一 citation id / evidence key。

`service/facade.py` 则把 pipeline 的原始 payload 再 enrich 成：

- `typed_retrieval`
- `evidence_registry`
- `results`

### 这部分的价值

这是后续做真正 database-augmented reasoning 的必要一步，因为它至少先保证：

- 上层 LLM 看到的是结构化证据；
- work 命中和 chunk 命中不再混在一起；
- 未来你要做 ablation（纯语义 / 纯结构 / 混合）时，有一致的壳可以接。

### 当前还没做到的地方

- 还没有真正把 PDG section / work capsule / result/method graph 全并进这个 retrieval shell；
- 还没有做更高级的 global / local / drill-down 三段式 query harness；
- 现在更像是“typed evidence substrate”而不是“完整 GraphRAG agent”。

---

## 3.4 work 结构层：`work_capsules`

### 关键文件

- `src/hep_rag_v2/structure.py`
- `src/hep_rag_v2/schema.sql`
- `tests/test_pdg_structure.py`

### 当前实现了什么

新增了 `work_capsules` 表，并实现了 `build_work_structures(...)`：

- 对选定 work 读取标题、摘要、最多 12 个可检索 chunks；
- 用启发式规则提取 result signatures；
- 用启发式规则提取 method signatures；
- 对非 review work 强制检查两类签名；
- 缺失时状态置为 `needs_attention`，并给 `missing_required_signatures`；
- review work 允许 `review_relaxed`。

### 已有状态语义

- `ready`
- `needs_attention`
- `review_relaxed`

### 为什么这很重要

这正好对应了你之前强调过的产品约束：

> 默认层的 result / method signature 对每篇非综述 work 都必须有，否则就没有意义。

当前实现虽然还是 heuristic v1，但至少：

- 没有静默跳过；
- 失败可见；
- review 有明确例外；
- 数据结构已经落到主 schema 里。

### 当前不足

- 还是 pattern-based heuristic，不是 LLM 级结构构建；
- 还没有拆出 typed `result objects` / `method objects`；
- 还没有连到 PDG spine 上；
- 还没有把 anomaly 流程真正接进 inspect/audit/用户侧运营流程。

---

## 3.5 PDG：archival ingest 与 section capsule

### 关键文件

- `src/hep_rag_v2/providers/pdg.py`
- `src/hep_rag_v2/pdg.py`
- `src/hep_rag_v2/pipeline.py`
- `src/hep_rag_v2/cli/_parser.py`
- `src/hep_rag_v2/cli/ingest.py`
- `tests/test_pdg_import_pipeline.py`
- `tests/test_pdg_structure.py`

### 当前实现了什么

#### A. archival ingest

`pipeline.import_pdg(...)` 会：

- 根据 `edition` 解析 canonical PDG reference；
- 初始化 collection / workspace；
- 往 `works` 表写一条 canonical PDG work；
- 把 PDF 暂存到 workspace；
- 在 `documents` 表里注册一条 `parser_name='pdg'` 的 archival document；
- 当前 parse_status 主要是 `pdf_ready` / `awaiting_pdf`。

#### B. PDG provider

`providers/pdg.py` 现在会：

- 规范化 `edition` / `slug`；
- 生成 canonical id：`pdg-<year>-review-of-particle-physics`；
- 生成 landing URL；
- **生成官方完整 PDF URL：** `https://pdg.lbl.gov/<year>/download/db<year>.pdf`；
- 支持本地 PDF copy / 远端下载 / 缓存复用。

#### C. local parsed source import

`pdg.import_pdg_source(...)` 会：

- 导入本地 MinerU bundle；
- 读取 manifest 与 content list；
- 产出 `annotated_blocks`；
- 按 heading 聚合 section；
- 生成 `capsule_text`；
- 写入 `pdg_sources` 与 `pdg_sections`。

#### D. CLI 双入口

`import-pdg` 现在支持两类调用：

1. edition-driven archival import：

```bash
hep-rag import-pdg --edition 2024 --pdf /path/to/pdg.pdf
```

2. local parsed-source import：

```bash
hep-rag import-pdg --source /path/to/pdg_bundle --source-id pdg-2024 --title "PDG 2024"
```

### 最近一次 PDG 修复的意义

这次 `13c52bd` 的价值非常直接：

- 把一个实际上会 404 的默认 PDG URL 修成了真实可访问的完整 PDF；
- 把 CLI 从“只能注册 PDF stub”推进到了“也能直接吃已解析 bundle”。

### 已有验证

- 单元测试验证 `resolve_pdg_reference()` 返回稳定 canonical metadata；
- 单元测试验证 local PDF import 会把 archival record 注册进 `works/documents`；
- 单元测试验证 local bundle import 会写入 `pdg_sources/pdg_sections`；
- 真实网络请求验证：`https://pdg.lbl.gov/2024/download/db2024.pdf` 返回 `200 application/pdf`。

### 当前边界

- 还**没有在本 session 真正跑完整 PDG 全文 + 真 MinerU + 全量 structure**；
- `pdg_sections` 现在是 section capsule，不是完整 PDG typed graph；
- 还没把 PDG section 检索、PDG↔work 联结、PDG global/local routing 接进最终 retrieval path。

---

## 3.6 benchmark 与规模可行性脚手架

### 关键文件

- `src/hep_rag_v2/loadtest.py`
- `scripts/run_scale_benchmark.py`
- `src/hep_rag_v2/benchmark_suite.py`
- `scripts/run_rag_effect_benchmark.py`
- `tests/fixtures/rag_effect_benchmark_cases.json`
- `tests/test_loadtest.py`
- `tests/test_benchmark_suite.py`

### 当前实现了什么

#### A. scale benchmark

`loadtest.py` 提供：

- `10k` / `50k` / `100k` tier；
- synthetic metadata hit 生成；
- BM25 / vector index build；
- query latency (`p50`, `p95`)；
- SQLite 大小、峰值内存、导出结果。

这部分主要是在回答：

> 如果先走 metadata-first，数据库在 10k / 50k / 100k work 量级会不会立刻变笨重？

它还不是“真实语料压测”，但已经给了一个统一的 harness。

#### B. RAG-effect benchmark

`benchmark_suite.py` 里定义了三种场景：

- `llm_only`
- `llm_plus_retrieve`
- `llm_plus_retrieve_and_structure`

这与我们前面讨论的 ablation 基本一致：

- 不接数据库；
- 只接检索；
- 接检索 + 结构化 shell。

### 当前不足

- 还没有补出你提到的“纯结构图谱不加 embedding”的完整执行路径；
- 还没有做真实弱模型自动打分；
- 目前更像 manifest / fixture / harness，而不是完整 benchmark 平台。

---

## 3.7 API / Web UI / service facade

### 关键文件

- `src/hep_rag_v2/service/facade.py`
- `src/hep_rag_v2/service/factory.py`
- `src/hep_rag_v2/api/*`
- `README.md`
- `docs/testing.md`

### 当前实现了什么

目前面向用户的交互面已经比较完整：

- CLI：适合本地操作与脚本化测试；
- FastAPI：适合服务化调用；
- 简单 Web UI：适合做用户端 smoke test。

README 里已经明确列出用户路径：

- `fetch-papers`
- `ingest-online`
- `import-pdg`
- `reparse-pdfs`
- `query`
- `ask`
- `hep-rag-api`

### 当前的意义

这意味着当前版本不是“只有内核代码”，而是已经到了一个用户可以：

- clone；
- 配 config；
- 导入一批 work；
- 看 retrieve；
- 看 ask；
- 看 Web UI；
- 跑 benchmark。

### 当前不足

- UI 仍然是最小可用，不是产品级前端；
- 还没有专门为 PDG/work 新架构设计用户 query harness；
- 还没有暴露完整的结构化 graph query 体验。

---

## 4. 数据库层面新增/关键表

当前主 schema 里值得重点 review 的新增/关键表有：

### 4.1 已有主干表

- `works`
- `work_ids`
- `collection_works`
- `documents`
- `document_sections`
- `blocks`
- `chunks`
- 图谱与向量相关表

### 4.2 本波次值得重点看的新表

#### `work_capsules`

用途：

- 放 work 级最小结构化表示；
- 挂 result / method signature；
- 挂 anomaly 与 status。

#### `pdg_sources`

用途：

- 登记某个 PDG 源（source_id、manifest、parsed_dir、block/capsule 数量）。

#### `pdg_sections`

用途：

- 存 PDG section 级 capsule；
- 支持 page range / title / path_text / raw_text / capsule_text。

### 当前数据库设计含义

这说明现在的方向已经不是“把全文硬塞进一个大向量库”，而是逐渐朝：

- work 级轻量层；
- PDG section 级轻量层；
- chunk 级 drill-down 层；

三层拆分迈进。

---

## 5. 当前用户真正能怎么用

## 5.1 最短可运行路径

### A. 初始化

```bash
hep-rag init-config --config ./hep-rag.yaml --workspace ./workspace
```

### B. 预览在线候选

```bash
hep-rag fetch-papers "same sign WW CMS" --config ./hep-rag.yaml --limit 5
```

### C. metadata-only ingest

```bash
hep-rag ingest-online "same sign WW CMS" \
  --config ./hep-rag.yaml \
  --limit 20 \
  --download-limit 0 \
  --parse-limit 0
```

### D. selective fulltext

```bash
hep-rag ingest-online "same sign WW CMS" \
  --config ./hep-rag.yaml \
  --limit 20 \
  --download-limit 8 \
  --parse-limit 4
```

### E. 检索 / 问答

```bash
hep-rag query "CMS VBS SSWW latest result" --config ./hep-rag.yaml --limit 8
hep-rag ask "总结 CMS VBS SSWW 的最新结果" --config ./hep-rag.yaml --mode survey
```

### F. PDG 路径

#### 用本地 PDF 注册 archival stub

```bash
hep-rag import-pdg \
  --config ./hep-rag.yaml \
  --collection pdg \
  --edition 2024 \
  --pdf /path/to/pdg-2024.pdf
```

#### 用本地 MinerU bundle 直接导入 PDG section capsule

```bash
hep-rag import-pdg \
  --source /path/to/pdg_bundle \
  --source-id pdg-2024 \
  --title "PDG 2024"
```

### G. Web/API

```bash
hep-rag-api --config ./hep-rag.yaml --host 127.0.0.1 --port 8000
```

然后打开：

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/docs`

---

## 6. 当前版本已经验证过什么

## 6.1 本 session 已重新验证的事实

### Git / 版本

- 已确认本文对应文档提交 `8f53e9d` 已 push 到 `origin/main`；
- 本文重点总结的最新核心 PDG 功能修复提交为 `13c52bd`，也已包含在当前远端主分支中。

### 测试

- `pytest -q` 通过；
- 结果：**113 passed**。

### PDG 定点验证

- `pytest -q tests/test_pdg_import_pipeline.py tests/test_pdg_structure.py` 通过；
- 结果：**9 passed**。

### 真实网络验证

- 对 `https://pdg.lbl.gov/2024/download/db2024.pdf` 发起真实 GET；
- 返回：`200 application/pdf`。

## 6.2 自动化测试重点覆盖了什么

### 配置与运行时

- `tests/test_config_runtime.py`

覆盖：

- config 创建/加载；
- workspace 切换；
- explicit mode/profile；
- embedding profile 解析；
- PDF candidate 优先级；
- runtime CLI 行为。

### PDG / structure

- `tests/test_pdg_import_pipeline.py`
- `tests/test_pdg_structure.py`

覆盖：

- PDG canonical metadata；
- local PDF archival import；
- local PDG bundle import；
- work signature policy；
- review 例外；
- anomaly 标记。

### retrieval / evidence

- `tests/test_retrieval_adapter.py`
- `tests/test_evidence.py`
- `tests/test_retrieval_shell.py`

覆盖：

- work/chunk 命中适配；
- evidence registry；
- typed shell 输出。

### benchmark

- `tests/test_loadtest.py`
- `tests/test_benchmark_suite.py`

覆盖：

- scale tier；
- dry-run；
- benchmark manifest 与 scenario matrix。

---

## 7. 当前明确还没完成的部分

这部分建议在 review 时特别注意，不要把“已有骨架”误判成“已完工产品”。

### 7.1 PDG 还不是完整 spine

当前只做到了：

- archival stub；
- local parsed source import；
- section capsule 持久化。

还没做到：

- 全量 PDG 真实导入的端到端耗时评估；
- PDG 全局图谱构建；
- PDG ↔ work typed relation 自动写入；
- PDG 驱动的 global retrieval 路由。

### 7.2 structure layer 还是最小启发式版

当前 `result/method` 提取是 heuristic v1，不是成熟结构抽取系统。它解决的是：

- “默认层不能静默跳过”这个产品约束；

但还没解决：

- 高质量 typed extraction；
- 复杂实验 paper 的细粒度对象图；
- 跨 article 的 method transfer graph。

### 7.3 benchmark 还是 scaffold，不是终局评测平台

当前已经有：

- tier benchmark；
- rag-effect manifest；

但还没有：

- 自动打分闭环；
- 真实弱模型批量跑分；
- semantic_only / structure_only / hybrid 的完整线上可切换实现矩阵。

### 7.4 没有在本 session 做真实完整 PDG 全流程观察

这点必须明确：

- 当前我修通了入口、验证了 URL、验证了单测；
- 但还**没有在这次 session 里完全作为观察者跑完整 PDG 真 PDF → 真 MinerU → 真结构化 → 真入库耗时**。

所以如果后续要 review “真实 PDG 导入行为是否顺畅”，还需要单独跑一次现实验。

---

## 8. 对未来 review 最值得看的文件顺序

如果你想最高效 review，我建议按下面顺序看：

### 第一组：先看产品入口与边界

1. `README.md`
2. `docs/testing.md`
3. `docs/pdg-work-implementation.md`

目标：

- 看当前用户路径是什么；
- 看当前版本明确承诺了什么；
- 看 PDG/work 波次的硬约束是什么。

### 第二组：看配置与模式

4. `src/hep_rag_v2/config.py`
5. `config.example.yaml`
6. `tests/test_config_runtime.py`

目标：

- 看 fallback 是否被压缩成显式 mode/profile；
- 看 embedding profile 是否可切换；
- 看 3090 本地 embedding 路径是否清楚。

### 第三组：看检索底座

7. `src/hep_rag_v2/retrieval_adapter.py`
8. `src/hep_rag_v2/evidence.py`
9. `src/hep_rag_v2/service/facade.py`
10. `tests/test_retrieval_adapter.py`
11. `tests/test_evidence.py`
12. `tests/test_retrieval_shell.py`

目标：

- 看 typed retrieval shell 是否足够干净；
- 看 evidence registry 是否适合作为上层 agent 的统一接口。

### 第四组：看 PDG 与结构层

13. `src/hep_rag_v2/providers/pdg.py`
14. `src/hep_rag_v2/pdg.py`
15. `src/hep_rag_v2/structure.py`
16. `tests/test_pdg_import_pipeline.py`
17. `tests/test_pdg_structure.py`

目标：

- 看 PDG 入口是否稳；
- 看 section capsule 的粒度是否合理；
- 看默认层签名门控是否符合你最初要求。

### 第五组：看 benchmark

18. `src/hep_rag_v2/loadtest.py`
19. `src/hep_rag_v2/benchmark_suite.py`
20. `scripts/run_scale_benchmark.py`
21. `scripts/run_rag_effect_benchmark.py`
22. `tests/test_loadtest.py`
23. `tests/test_benchmark_suite.py`

目标：

- 看 100k 级可行性验证是否有了统一入口；
- 看未来“数据库到底有没有帮助弱模型”是否有了比较框架。

---

## 9. 最后总结：当前版本的真实定位

我认为当前主干最准确的定位不是：

- “已经完成最终 HEP GraphRAG 产品”；也不是
- “只是一些零散 patch”。

而是：

> **已经搭出一个值得继续投资的、relational-first、typed-substrate 优先的 HEP RAG v2 基座。**

它最重要的价值在于：

1. **避免了一开始就做大而笨重的全量图系统；**
2. **把 metadata-first、work-level、chunk drill-down 分层先立住；**
3. **开始把 PDG 作为未来 spine 接进来，但还没假装自己已经做完；**
4. **把弱模型 + 数据库增益的 benchmark 思路落成了代码脚手架；**
5. **把 silent fallback 往 explicit mode/profile 的方向推进了。**

如果下一步继续推进，我认为最自然的主线会是：

- 真跑一次完整 PDG 导入观察；
- 把 PDG section、work capsule、chunk drill-down 串成真正的 local/global/query harness；
- 把 structure-only / semantic-only / hybrid 三条 ablation 彻底做实；
- 再往 typed result/method objects 与 PDG↔work 联结推进。

---

## 10. 附：本次文档对齐时的实际证据

- 对齐远端提交：`8f53e9d`
- 当前分支状态：`main...origin/main`（写本文档前已 clean）
- 已重新执行：
  - `pytest -q` → `113 passed in 11.16s`
  - `pytest -q tests/test_pdg_import_pipeline.py tests/test_pdg_structure.py` → `9 passed in 1.88s`
- 已重新确认：
  - `https://pdg.lbl.gov/2024/download/db2024.pdf` → `200 application/pdf`
