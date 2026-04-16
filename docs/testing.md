# 测试方法

这份文档只保留用户真正会用到的测试路径，并补充当前 `structure-upstream` 波次必须满足的最小验证规则。

如果你在推进当前结构化推理 substrate，请同时查看：

- [`docs/pdg-work-implementation.md`](./pdg-work-implementation.md)：当前波次的结构优先规则、状态语义和验收清单
- [`docs/hep-core-object-contracts.md`](./hep-core-object-contracts.md)：冻结的对象合同与允许状态集合

## 当前波次：structure 先于 downstream lanes

当前仓库不再把 `results` / `methods` / `transfer` 视为彼此独立的自由抽取器，而是把 `structure` 当作默认上游判断来源。推进这一波次时，至少要保证：

- ingest / reparse 默认路径先跑 `build_work_structures()`，再跑 `results` / `methods` / `transfer`
- 顶层状态只使用合同允许值：`ready` / `partial` / `needs_review` / `failed`（不再保留 `review_relaxed` / `needs_attention` 兼容语义）
- 非综述文章缺少必需签名时，不允许静默跳过；必须留下合同级可见状态
- README、测试文档和实现文档统一把仓库描述为 reasoning substrate，而不是仅仅“若干独立 producer 的集合”

建议在当前波次至少补跑：

```bash
pytest -q tests/test_thinking_extraction.py tests/test_bootstrap.py tests/test_config_runtime.py
pytest -q tests/test_object_contracts.py tests/test_pdg_structure.py
```

如果工作触及检索/服务/基准适配层，再补：

```bash
pytest -q tests/test_retrieval_adapter.py tests/test_service_api.py tests/test_benchmark_suite.py
```

## 1. 最小导入测试

### 初始化

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[api,langchain]"
hep-rag init-config --config ./hep-rag.yaml --workspace ./workspace
```

### 先预览候选文章

```bash
hep-rag fetch-papers "same sign WW CMS" \
  --config ./hep-rag.yaml \
  --limit 5
```

### 做 metadata-only 导入

```bash
hep-rag ingest-online "same sign WW CMS" \
  --config ./hep-rag.yaml \
  --limit 20 \
  --download-limit 0 \
  --parse-limit 0
```

这一步主要验证：

- 在线检索是否正常
- 元数据是否成功入库
- work-level 检索是否可用
- 默认 ingest 路径没有因为结构层接线而退化

## 1.5 PDG corpus 导入冒烟

如果你正在推进 PDG 主干导入，建议优先验证 `website` 或 `full` artifact，而不是先走 booklet PDF：

```bash
hep-rag import-pdg \
  --config ./hep-rag.yaml \
  --collection pdg \
  --edition 2024 \
  --artifact website \
  --download
```

这一步当前主要验证：

- PDG 官方 artifact 元数据是否能解析成稳定 canonical id
- website zip 是否能落到 workspace，并被导入为 `pdg_sections`
- website 内嵌的 review / listing / table PDFs 是否会被注册成正式 parse candidates
- physics substrate 是否能从 PDG corpus 主干正常构建

当前框架已经不再引入 standalone `book_pdf` / `booklet_pdf`。PDG 的活跃 PDF 语料只来自 website bundle 内嵌的 `reviews` / `tables` / `listings`。

```bash
hep-rag import-pdg \
  --config ./hep-rag.yaml \
  --collection pdg \
  --edition 2024 \
  --artifact website \
  --download
```

预期行为：

- `rpp-2024.zip` 会落到 `workspace/data/raw/pdg/website/`
- website 内嵌 PDF 会被注册到 `workspace/data/pdfs/pdg/`
- 对应 `documents.parse_status` 会变成 `pdf_ready`

## 2. selective fulltext / structure 测试

如果 metadata-only 没问题，再测全文热层与结构层衔接。

先在 `hep-rag.yaml` 中确认 `mineru.enabled=true`，然后执行：

```bash
hep-rag ingest-online "same sign WW CMS" \
  --config ./hep-rag.yaml \
  --limit 20 \
  --download-limit 8 \
  --parse-limit 4
```

这一步主要验证：

- PDF 下载
- MinerU 解析
- chunk 层索引
- 结构层在 downstream producers 之前完成
- 证据 drill-down

如果要验证本地已有 PDF 的增量结构刷新：

```bash
hep-rag reparse-pdfs --config ./hep-rag.yaml --collection default
```

关注点不是“有没有跑完一个命令”，而是：结构层是否在 reparse 后重新成为结果/方法/迁移三条下游 lane 的共同上游。

## 3. 导入后检查

### 查看 workspace 状态

```bash
hep-rag status --config ./hep-rag.yaml
```

重点看：

- `works`
- `documents`
- `chunks`
- 搜索/向量索引计数
- 结构相关衍生物是否开始可见

### 只做检索

```bash
hep-rag query "CMS VBS SSWW latest result" \
  --config ./hep-rag.yaml \
  --limit 8
```

### 做问答

```bash
hep-rag ask "总结 CMS VBS SSWW 的最新结果" \
  --config ./hep-rag.yaml \
  --mode survey
```

## 4. Web / API 用户端测试

### 启动服务

```bash
hep-rag-api --config ./hep-rag.yaml --host 127.0.0.1 --port 8000
```

打开：

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/docs`

### 推荐测试顺序

1. `Fetch Papers`
2. `Retrieve`
3. `Ask`
4. `Start Ingest Job`
5. `Refresh Workspace`

### 重点观察

- `Fetch Papers`：召回是否合理
- `Retrieve`：works/chunks/structure 证据是否与 query 对齐
- `Ask`：答案是否带证据、有没有明显幻觉
- `Start Ingest Job`：日志是否持续刷新、结构层是否先于下游 lane 完成

## 5. 基准测试

### metadata-only scale benchmark

```bash
python scripts/run_scale_benchmark.py \
  --tier 10k \
  --workspace-root /tmp/hep-rag-bench-10k
```

也可以切到：

- `50k`
- `100k --dry-run`

### 弱模型 + 数据库增益 benchmark

先生成 benchmark manifest：

```bash
python scripts/run_rag_effect_benchmark.py \
  --model-label weak-model \
  --print-cases
```

这一步会生成后续评测所需的：

- 问题集
- 对照场景
- manifest

当前默认的对照场景是：

1. `llm_only`
2. `llm_plus_retrieve`
3. `llm_plus_retrieve_and_structure`
4. `thinking_engine_trace`

其中 `thinking_engine_trace` 会额外检查 reasoning substrate 是否仍能输出可回放的思考摘要；`llm_plus_retrieve_and_structure` 则用来观察结构层接入后，对弱模型问答的实际增益。

如果只是确认 manifest 内容，也可以直接跑 CLI：

```bash
hep-rag benchmark-manifest --model-label weak-model
```

## 6. 最小验收清单

如果你只想快速判断当前系统能不能用，至少过这几个点：

- `fetch-papers` 返回合理候选
- metadata-only ingest 成功
- `query` 返回相关 works/chunks
- `ask` 能生成答案
- Web 控制台交互正常
- `python -m pytest -q` 全绿

如果你在推进 structure-upstream 波次，再额外确认：

- `tests/test_pdg_structure.py` 覆盖 article / review 的签名策略
- `tests/test_thinking_extraction.py` 证明 structure 与 downstream lanes 仍保持一致
- 顶层状态没有漂移出 `ready` / `partial` / `needs_review` / `failed`

## 7. 当前边界

当前仓库适合测试：

- metadata-first retrieval substrate
- selective fulltext hot lane
- structure-governed downstream extraction
- typed retrieval / evidence shell
- benchmark 脚手架

当前仓库还不适合直接测试：

- 完整 PDG spine
- 全量基准金标数据
- 已冻结但全面商品化的 reasoning/object production pipeline

对象合同已经在 [`docs/hep-core-object-contracts.md`](./hep-core-object-contracts.md) 冻结；本波次的重点不是再发明新的状态语义，而是让 `structure -> results/methods/transfer` 的默认上游关系在实现、测试和文档里都真正成立。
