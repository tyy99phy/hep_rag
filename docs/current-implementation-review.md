# 当前版本实现总览与改动审查（对齐 `origin/main@13c52bd`）

> 文档目的：把当前仓库**已经落地**的能力、最近一轮**实际改动**、关键文件职责、验证证据、未完成项与 review 建议集中写清，方便后续人工 review。
>
> 对齐基线：`origin/main` 最新提交 `13c52bd`（2026-04-07 本次已 push）。

---

## 0. 一句话结论

当前版本已经从“单纯的 metadata / PDF / chunk 检索框架”推进到一个**带有显式配置档位、PDG 导入骨架、PDG MinerU 结构化入口、work capsule 默认层、typed retrieval shell、benchmark 脚手架**的 HEP RAG substrate。

但它**还不是最终形态的 PDG spine + work/result/method 全图谱系统**。更准确地说：

- **数据库/检索/服务外壳**已经具备继续扩展的基础；
- **PDG 接入**已经有了两个入口（官方整本 PDF archival ingest、MinerU bundle 直接结构化导入）；
- **默认层结构化**已经实现了“非综述文章必须有 result/method signature，否则显式标异常”的第一版门控；
- **benchmark / ablation** 已有最小脚手架；
- 但**真正的大规模 PDG 全量解析、paper 结构抽取流水线、global/local 分层检索编排、生产级 typed graph** 还没有完全做完。

---

## 1. 当前版本面向用户已经能做什么

### 1.1 基础工作流

用户现在可以直接用 CLI / API 完成下面几类事情：

1. 初始化配置与 workspace
2. 在线检索 InspireHEP 候选论文
3. 执行 online ingest（metadata / PDF / MinerU / index / graph）
4. 对已有 PDF 做增量 MinerU 重解析
5. 做本地 BM25 / vector / hybrid 检索
6. 做 retrieval + LLM 问答
7. 启动 FastAPI + Web Console 做用户端操作
8. 跑 metadata-only scale benchmark
9. 生成“弱模型接数据库前后效果” benchmark manifest
10. 导入 PDG：
   - **路径 A：**以 official PDF 为入口，先注册 archival ingest stub
   - **路径 B：**以本地 MinerU bundle 为入口，直接生成 `pdg_sections` 胶囊结构

### 1.2 当前版本最重要的新能力

相对更早的版本，这一轮最关键的新增/强化点是：

- 引入了**显式 mode/profile** 配置约定，而不是靠隐式 fallback；
- 给出了一套**本地 embedding profile**，默认强调 3090 友好的本地小模型语义向量路径；
- 新增 **PDG provider + import pipeline**；
- 新增 **PDG 结构化表**：`pdg_sources` / `pdg_sections`；
- 新增 **work capsule 默认层**：`work_capsules`；
- 把“**非综述文章缺 result/method signature 不得静默跳过**”落成了真实代码和测试；
- 增补了 review 文档与测试文档，让用户入口更整洁；
- 修复了 PDG 官方 PDF 默认 URL，改为当前可用的 `db{year}.pdf`。

---

## 2. 这一轮实际改了哪些文件

从 `f677bc0` 到当前 `13c52bd`，本波次新增/修改并已推送的主要文件有：

### 文档层

- `README.md`
- `docs/testing.md`
- `docs/pdg-work-implementation.md`
- `config.example.yaml`

### 配置与 CLI 层

- `src/hep_rag_v2/config.py`
- `src/hep_rag_v2/cli/__init__.py`
- `src/hep_rag_v2/cli/_parser.py`
- `src/hep_rag_v2/cli/ingest.py`

### PDG / structure / pipeline 层

- `src/hep_rag_v2/providers/__init__.py`
- `src/hep_rag_v2/providers/pdg.py`
- `src/hep_rag_v2/pipeline.py`
- `src/hep_rag_v2/pdg.py`
- `src/hep_rag_v2/structure.py`
- `src/hep_rag_v2/schema.sql`

### 测试层

- `tests/test_config_runtime.py`
- `tests/test_pdg_import_pipeline.py`
- `tests/test_pdg_structure.py`

---

## 3. 关键实现分块说明

## 3.1 配置：从“隐式 fallback”转向“显式 mode/profile”

### 相关文件

- `src/hep_rag_v2/config.py`
- `config.example.yaml`
- `tests/test_config_runtime.py`

### 当前实现了什么

`config.py` 现在明确给出了以下配置骨架：

- `modes.build`
- `modes.retrieval`
- `profiles.structure`
- `profiles.embedding`
- `profiles.pdg`
- `build.structure_backend`
- `build.embedding_source`
- `build.allow_silent_fallback`
- `embedding.allow_silent_fallback`
- `structure.allow_silent_fallback`
- `embedding_profiles.bootstrap`
- `embedding_profiles.semantic_small_local`
- `embedding_profiles.semantic_prod_local`

对应的设计意思非常清楚：

1. **build mode / retrieval mode 显式化**；
2. **embedding profile 显式化**；
3. 不再鼓励“出了问题偷偷 fallback”；
4. 默认把“结构构建”和“语义 embedding”看成两套职责。

### 当前默认值的实际含义

当前默认配置里：

- `build.structure_backend = api_llm`
- `build.embedding_source = local_profile`
- `profiles.embedding = bootstrap`
- `embedding_profiles.semantic_small_local = sentence-transformers:BAAI/bge-small-en-v1.5`
- `embedding_profiles.semantic_prod_local = sentence-transformers:BAAI/bge-base-en-v1.5`

也就是说，代码层已经支持你后续走这样的路线：

- **结构抽取**：API 模型
- **语义 embedding**：本地轻量模型

这与你前面提出的“构建数据库结构可以用 API，小语义向量要本地化”是对齐的。

### review 时重点看什么

- `resolve_mode()` 是否真的让模式选择显式化；
- `resolve_embedding_profile()` 是否彻底禁止 profile 名称拼错后的静默回退；
- `config.example.yaml` 的默认值是否与你未来真正想推荐给用户的默认安装流程一致。

### 当前边界

虽然配置层已经把“反 fallback”方向搭好了，但**还没有形成完整的 profile 驱动执行矩阵**。也就是说：

- 配置契约已经更清楚；
- 但很多 downstream 流程还没有完全用这些 mode/profile 做细粒度编排。

---

## 3.2 PDG archival ingest：先把 PDG 当成一个稳定 canonical work

### 相关文件

- `src/hep_rag_v2/providers/pdg.py`
- `src/hep_rag_v2/pipeline.py`
- `src/hep_rag_v2/cli/_parser.py`
- `src/hep_rag_v2/cli/ingest.py`
- `tests/test_pdg_import_pipeline.py`

### 当前实现了什么

这一层解决的是：

> “先把 PDG 这本东西作为一个可追踪、可缓存、可继续往下解析的 canonical source 放进系统。”

主要实现包括：

1. `resolve_pdg_reference()`
   - 生成稳定的 `canonical_source = pdg`
   - 生成稳定的 `canonical_id = pdg-{year}-review-of-particle-physics`
   - 生成标题、landing_url、pdf_url 等元数据

2. `stage_pdg_pdf()`
   - 支持本地 PDF copy/stage
   - 支持远程下载
   - 支持复用已有缓存
   - 带 PDF 内容合法性检查

3. `pipeline.import_pdg()`
   - 初始化 collection / workspace
   - 把 PDG 作为 `works` 中的一个 archival work 注册
   - 在 `documents` 中创建 parser=`pdg` 的文档记录
   - 把 parse status 标成 `pdf_ready` 或 `awaiting_pdf`

### 这次修复了什么关键问题

本次最新 push 的 `13c52bd` 做了两个非常关键的修正：

#### (1) `import-pdg` 不再只能走 `--edition`

现在 `import-pdg` 有两条路径：

- `--edition [--pdf | --download]`：archival ingest 入口
- `--source --source-id --title`：本地 MinerU bundle 直接导入 PDG 结构层

这使得你既能：

- 从官方整本 PDF 开始走标准入口；
- 也能直接拿已有 MinerU 产物做结构化导入测试。

#### (2) 默认 PDF URL 改成当前官方可用路径

已从旧的 intro review PDF 路径切到：

- `https://pdg.lbl.gov/{year}/download/db{year}.pdf`

这个改动很重要，因为旧的 `rev-intro` 路径在当前站点布局下已经不稳定，2024 年对应路径甚至会 404。

### 当前用户可用命令

#### 路径 A：official PDF -> archival ingest

```bash
hep-rag import-pdg \
  --config ./hep-rag.yaml \
  --collection pdg \
  --edition 2024 \
  --pdf /path/to/db2024.pdf
```

或：

```bash
hep-rag import-pdg \
  --config ./hep-rag.yaml \
  --collection pdg \
  --edition 2024 \
  --download
```

#### 路径 B：MinerU bundle -> 直接结构化导入

```bash
hep-rag import-pdg \
  --source /path/to/pdg_bundle \
  --source-id pdg-2024 \
  --title "PDG 2024"
```

### 当前边界

这条 archival ingest 流程目前**还没有自动把真实 PDG PDF 送进 MinerU 并完成整本结构化**。它现在完成的是：

- canonical work 注册
- PDF 落地/缓存
- 文档记录初始化

它是**稳定骨架**，不是最终整本 PDG 自动结构化流水线。

---

## 3.3 PDG MinerU bundle -> `pdg_sections`：PDG 结构化入口已落地

### 相关文件

- `src/hep_rag_v2/pdg.py`
- `src/hep_rag_v2/schema.sql`
- `tests/test_pdg_structure.py`

### 当前实现了什么

`pdg.py` 新增了一套专门针对 PDG 结构化结果的 schema 与导入逻辑：

#### 新表

- `pdg_sources`
- `pdg_sections`

它们的角色是：

- `pdg_sources`：记录一次 PDG 来源、manifest、parsed_dir、block_count、capsule_count
- `pdg_sections`：记录 PDG 的 section 级胶囊，包括：
  - `title`
  - `path_text`
  - `section_kind`
  - `level`
  - `page_start/page_end`
  - `raw_text`
  - `capsule_text`

#### 导入函数

`import_pdg_source()` 当前会：

1. 把本地 MinerU bundle 导入到 workspace 下的 `parsed/pdg/<source_id>`；
2. 读取 manifest 与 content list；
3. 把 MinerU block 解析成 annotated blocks；
4. 按 heading 收集 section；
5. 为每个 section 生成 capsule text；
6. 写入 `pdg_sources` 与 `pdg_sections`。

### 这层实现的意义

这正是你前面一直在说的“PDG 不应该只当成一个粗糙摘要，也不能直接把整本原文当默认搜索单位”之间的中间层。

当前版本已经给出一个**section capsule 级别**的 PDG 结构层，它比整本摘要细、比原始 chunk 粗，适合作为后续：

- PDG spine
- global/community summary
- 与 work capsule 对接

的基础材料。

### 当前边界

目前 `pdg_sections` 还只是**章节胶囊层**，还没有进一步抽成：

- PDG concept / entity / relation graph
- PDG topic -> work linkage
- PDG spine 上的 typed semantic nodes

所以这是一层很有价值的 substrate，但不是最终图谱本体。

---

## 3.4 `work_capsules`：默认层门控已经从想法变成代码

### 相关文件

- `src/hep_rag_v2/structure.py`
- `src/hep_rag_v2/schema.sql`
- `tests/test_pdg_structure.py`
- `docs/pdg-work-implementation.md`

### 当前实现了什么

`structure.py` 新增了 `work_capsules` 表和 `build_work_structures()` 流程。

#### 新表 `work_capsules`

每个 `work_id` 最多对应一个 capsule，主要字段包括：

- `profile`
- `builder`
- `is_review`
- `status`
- `capsule_text`
- `result_signature_json`
- `method_signature_json`
- `anomaly_code`
- `anomaly_detail`

#### 当前的结构化逻辑

对于选中的 work：

1. 取 title / abstract / 前若干可检索 chunk；
2. 组成 `text_blob`；
3. 用 heuristic pattern 抽 `result signatures`；
4. 用 heuristic pattern 抽 `method signatures`；
5. 判定是不是 review；
6. 如果是非综述且缺少任一类 signature，则显式标为 `needs_attention`；
7. 否则标为 `ready`；
8. 生成 capsule_text 并 upsert 到 `work_capsules`。

### 当前已经落地的硬约束

这一点尤其重要：

> **非综述文章如果抽不出 result signature 或 method signature，不会被静默跳过，而会被显式标异常。**

这是你之前反复强调的要求，这一轮已经在代码里真正实现了：

- review -> `review_relaxed`
- 缺签名 -> `needs_attention`
- 满足要求 -> `ready`

### 当前抽取方式的性质

这套抽取现在还是 **heuristic-v1**，也就是关键字/模式驱动，而不是 API LLM 驱动。

优点：

- 快
- 可测试
- 不依赖外部模型
- 先把数据库门控规则落地

缺点：

- 召回/精度有限
- 不足以覆盖真实复杂 paper 的 method/result 描述
- 离你想要的“高能物理结果与方法网络”还有距离

### 当前边界

- 还没有 result object / method object / relation table
- 还没有把 `work_capsules` 真正纳入 retrieval planner
- 还没有 API LLM 结构抽取版本
- 还没有基于 figure/table/formula 的 richer signature 抽取

所以这部分更像是：

> **默认层门禁和结构骨架已经落地，但 typed graph 本体仍在后续。**

---

## 3.5 retrieval / evidence shell / benchmark：现阶段可 review 的成熟部分

### 相关文件

- `src/hep_rag_v2/retrieval_adapter.py`
- `src/hep_rag_v2/evidence.py`
- `src/hep_rag_v2/service/facade.py`
- `src/hep_rag_v2/service/factory.py`
- `src/hep_rag_v2/benchmark_suite.py`
- `src/hep_rag_v2/loadtest.py`
- `scripts/run_scale_benchmark.py`
- `scripts/run_rag_effect_benchmark.py`
- `tests/test_retrieval_adapter.py`
- `tests/test_retrieval_shell.py`
- `tests/test_evidence.py`
- `tests/test_benchmark_suite.py`

### 当前实现了什么

#### typed retrieval shell

现在 retrieval 返回结果不只是松散 JSON，而是有一层 typed normalization：

- `TypedRetrievalMetadata`
- `TypedRetrievalResult`
- `adapt_work_hit()`
- `adapt_chunk_hit()`
- `normalize_retrieval_payload()`
- `build_retrieval_shell()`

这层会统一 work/chunk 两种命中，让输出更适合作为后续结构化 answer 或外部 agent tool 调用的中间格式。

#### evidence registry

`EvidenceRegistry` 支持：

- evidence 注册
- citation id 分配
- work/chunk 分 lane 管理
- 重复证据去重

它的作用是把“检索结果”进一步包装成“可以稳定引用的证据对象”。

#### service facade / tool registry

`HepRagServiceFacade` 当前已经把：

- `retrieve`
- `ask`
- `fetch_papers`
- `ingest_online`
- `reparse_pdfs`
- `workspace_status`
- `show_graph`
- `show_document`
- `audit_document`

统一包装成服务层接口。

这使 CLI、API、后续 agent tool registry 可以共享一套中间服务层，而不是各自直连底层 pipeline。

#### benchmark 脚手架

目前已经有两类 benchmark：

1. **scale benchmark**
   - `10k / 50k / 100k`
   - 测 metadata-only ingest、BM25、vector 索引与查询延迟

2. **RAG effect benchmark**
   - `llm_only`
   - `llm_plus_retrieve`
   - `llm_plus_retrieve_and_structure`

第二类 benchmark 非常贴合你后面想做的事情：

> 用较弱模型，比较接入数据库前后性能差别，衡量数据库真正提供了多少价值。

### 当前边界

- ablation 现在是 **manifest scaffolding**，不是完整自动评测流水线；
- `structure_only / semantic_only / hybrid` 的运行时切换思想已经有了方向，但还没有在完整 query planner 中彻底跑通；
- `work_capsules` / `pdg_sections` 还没有真正接进 `TypedRetrievalResult` 的主 lane。

---

## 3.6 API / Web Console：用户端测试入口已经比较完整

### 相关文件

- `src/hep_rag_v2/api/*`
- `src/hep_rag_v2/service/facade.py`
- `tests/test_service_api.py`
- `README.md`
- `docs/testing.md`

### 当前实现了什么

从文档和测试看，当前 Web/API 层已经支持：

- `/` Web Console
- `/docs` Swagger
- `/health`
- `/auth/status`
- `/retrieve`
- `/ask`
- `/fetch-papers`
- `/jobs/ingest-online`
- `/jobs/reparse-pdfs`
- `/jobs/{job_id}`
- `/jobs/{job_id}/events`

并且异步 job 事件会落到单独的 API DB，不和主业务 DB 强耦合。

### 这层对 review 的价值

这一层说明当前仓库不是纯研究脚本，而已经有一个比较清楚的**用户交互面**，适合：

- 做本地 demo
- 做用户测试
- 做数据导入与运行日志观察
- 做 API 集成验证

### 当前边界

- Web Console 仍然是“最小可用控制台”，不是成熟产品 UI；
- 结构化检索层、PDG spine、global/local planner 还没有在 UI 层显式体现；
- 更适合技术验证，不适合现在就作为 polished 产品前端。

---

## 4. 文档层现在是什么状态

## 4.1 `README.md`

作用：

- 用户入口总览
- 快速开始
- CLI 命令索引
- Web/API 使用说明
- 指向 `docs/testing.md` 与 `docs/pdg-work-implementation.md`

当前风格已经比之前更整洁，至少没有把夜间 runbook 之类内部操作塞给用户。

## 4.2 `docs/testing.md`

作用：

- 给用户一条比较清楚的测试路径：
  - metadata-only
  - selective fulltext
  - query / ask
  - Web/API
  - benchmark

同时明确写了：

- 当前适合测什么
- 当前还不适合直接测什么

这是对外比较合适的文档层次。

## 4.3 `docs/pdg-work-implementation.md`

作用：

- 不是用户教程，而是**当前这一波 PDG/work 实施约束说明**
- 把你在讨论中反复强调的约束写进一个稳定文档：
  - 显式 mode/profile
  - 非综述强制 result/method signature
  - 失败要显式标记
  - embedding 与结构模型解耦

这个文档对未来 review 和继续实现都很重要，因为它把“为什么这么做”固定下来了。

---

## 5. 当前版本最值得你 review 的 8 个问题

### 5.1 PDG official PDF URL 是否稳定

当前已经修到了 `db{year}.pdf`，并实测 2024 路径返回 `200 application/pdf`。

这比上一版可靠得多，但它依旧依赖 PDG 官网目录结构未来不变。

### 5.2 `import-pdg` 双路径契约是否足够清楚

代码已经支持：

- archival import
- local bundle import

但 README 里对 `--source --source-id --title` 这条路径还没有展开写清。
这不是代码 bug，但属于**文档覆盖略落后于能力**。

### 5.3 `work_capsules` 的 heuristic 抽取是否只是临时版

答案是：是的。

它现在主要解决“门控与显式异常”问题，而不是最终抽取质量问题。

### 5.4 `work_capsules` 是否已经真正接入主检索

还没有完全接进去。

它已经存在于数据库里，也有清晰状态模型，但还未成为 retrieval planner 的主 lane。

### 5.5 `pdg_sections` 是否已经和 paper/work 打通

也还没有。

当前 PDG 和 work 在数据库层都已有入口，但“PDG spine + work capture 联动”的关系层还没有真正写出来。

### 5.6 fallback 是否真的基本被拿掉了

配置层已经非常明确地往“显式模式，不要 silent fallback”推进；
但在更大范围的 pipeline 编排层，还没有完全做成严格模式矩阵。

### 5.7 benchmark 是否已经足够用于最终科研评估

还不够。

现在更像是：

- 有比较合理的脚手架
- 有你后面想做的评估方向
- 但还没有完整 automated harness + scoring pipeline

### 5.8 是否已经适合直接导入 100k paper 做生产级验证

还不能直接下这个结论。

因为当前虽然已有：

- 10k/50k/100k scale benchmark scaffold
- metadata-only 压测路径

但对真正的：

- MinerU 成本
- chunk/asset 保留策略
- 结构抽取耗时
- 增量更新策略
- PDG + work 复合检索延迟

还没有完整实测闭环。

---

## 6. 当前已验证证据

以下是这次对齐远端前后实际跑过的验证：

### 6.1 全量测试

```bash
pytest -q
```

结果：

- `113 passed`

### 6.2 PDG 相关定向测试

```bash
pytest -q tests/test_pdg_import_pipeline.py tests/test_pdg_structure.py
```

结果：

- `9 passed`

### 6.3 官方 PDG PDF 路径在线检查

已实际请求：

- `https://pdg.lbl.gov/2024/download/db2024.pdf`

结果：

- `200`
- `content-type: application/pdf`

### 6.4 本次 push 结果

已执行并成功：

```bash
git push origin main
```

远端已对齐到：

- `origin/main@13c52bd`

---

## 7. 当前仍然明显未完成 / 暂缓的部分

下面这些是 review 时必须明确区分为“未来工作”的，而不是误以为已完成：

### 7.1 PDG spine 还没有完成

当前只有：

- PDG archival work
- PDG section capsules

还没有：

- PDG concept graph
- PDG relation graph
- PDG -> work typed linkage layer

### 7.2 work/result/method 图谱本体还没有完成

当前只有：

- `work_capsules`
- `result_signature_json`
- `method_signature_json`

还没有：

- 独立 typed nodes / edges
- 跨 work 方法迁移链路
- 可供 global reasoning 的高层方法图

### 7.3 full end-to-end PDG real PDF parse 还没在这次 session 实测

代码入口已经准备得比之前好很多，但这次 session 没有把**整本真实 PDG**完整喂进 MinerU 再跑完全结构化，所以下面这些还没有真实数据：

- 真正耗时
- 真正 section 数量
- 真正表规模
- 真正数据库膨胀情况

### 7.4 global/local 检索编排还没真正产品化

你前面讨论的：

- local：work seed -> 扩散 -> drill-down chunk
- global：community / summary / PDG spine / work capsule 层级搜索

这些思想已经体现在规划和部分 substrate 上，但还未完整落实为一个成熟 planner。

---

## 8. 建议的 review 顺序

如果你准备系统 review，建议按下面顺序看：

### 第一组：看“方向有没有偏”

1. `docs/pdg-work-implementation.md`
2. `config.example.yaml`
3. `src/hep_rag_v2/config.py`

### 第二组：看“PDG 接入是不是靠谱”

4. `src/hep_rag_v2/providers/pdg.py`
5. `src/hep_rag_v2/pipeline.py`（`import_pdg` 部分）
6. `src/hep_rag_v2/cli/_parser.py`
7. `src/hep_rag_v2/cli/ingest.py`
8. `src/hep_rag_v2/pdg.py`
9. `tests/test_pdg_import_pipeline.py`
10. `tests/test_pdg_structure.py`

### 第三组：看“默认层规则是否真的落地”

11. `src/hep_rag_v2/structure.py`
12. `src/hep_rag_v2/schema.sql`

### 第四组：看“这个项目现在是不是已经有用户面和评测面”

13. `README.md`
14. `docs/testing.md`
15. `src/hep_rag_v2/retrieval_adapter.py`
16. `src/hep_rag_v2/evidence.py`
17. `src/hep_rag_v2/service/facade.py`
18. `src/hep_rag_v2/benchmark_suite.py`
19. `src/hep_rag_v2/loadtest.py`
20. `tests/test_benchmark_suite.py`

---

## 9. 我对当前版本的总体判断

如果把目标分成三个层级：

### 层级 A：小而美、可跑、可测、可继续长

当前版本**已经达到了这个层级**。

它已经不是散乱脚本，而是：

- 有配置契约
- 有 CLI
- 有 API
- 有 DB schema
- 有 typed retrieval shell
- 有 benchmark scaffold
- 有 PDG/work 演化方向的代码落点

### 层级 B：PDG + work 双层知识底座

当前版本**已经把底座的第一段打好了**，但还没有真正完成。

已完成的是：

- PDG archival/source layer
- PDG section capsule layer
- work capsule default layer

未完成的是：

- PDG spine 与 work capsule 的有机耦合
- global/local 分层 planner
- typed result/method graph

### 层级 C：面向高能物理创新发现的高阶 GraphRAG

当前版本**还远未到这一层**。

但和 trivial graphRAG + 本地小模型 demo 相比，它已经更接近一个真正能往“高能物理知识底座”方向生长的 substrate。

---

## 10. 最后总结

如果你问“当前这版到底做成了什么”，最准确的回答是：

> 它已经把 **HEP RAG 的工程骨架、PDG 导入入口、默认层结构门控、typed retrieval shell、benchmark 脚手架** 做成了一个可 review、可继续迭代的版本；
> 但它还没有把 **PDG spine + work/result/method typed graph + global/local planner** 彻底做完。

如果你问“这一版最大的价值是什么”，我会说是：

1. **方向已经从‘堆 fallback 的抓取器’转向‘显式模式的知识底座’**；
2. **PDG 与 work 的两个入口都已经落代码**；
3. **默认层不再允许非综述 work 静默缺失 result/method 签名**；
4. **后续无论做 benchmark、做 PDG 全量导入、做 global/local planner，都已经有真实代码支点可以接。**

