# hep\_rag\_v2

配置驱动的高能物理论文检索与结构化 reasoning substrate。从 InspireHEP 在线检索论文元数据，下载 PDF 并解析全文，构建引用图谱与检索索引，并在当前波次中把 `structure` 作为 `results` / `methods` / `transfer` 的上游判断来源。

## 测试方法

如果你想快速试用、导入文章或做 benchmark，建议看：

- [`docs/testing.md`](docs/testing.md)
- [`docs/pdg-work-implementation.md`](docs/pdg-work-implementation.md)（当前 structure-upstream 波次的落地约束、状态语义与验收清单）

## 当前仓库姿态

当前仓库不只是“检索 + 问答”工具链，也在收敛为一个 HEP reasoning substrate：只要存在 `structure` 输出，它就是 `results` / `methods` / `transfer` 默认消费的语义来源。当前允许的顶层状态以 [`docs/hep-core-object-contracts.md`](docs/hep-core-object-contracts.md) 为准：`ready` / `partial` / `needs_review` / `failed`；历史兼容值 `review_relaxed` / `needs_attention` 已退出运行时语义。

## 架构

```
InspireHEP API
    │
    ▼
多 query 在线检索 + family-aware 合并 / 去重
    │
    ▼
元数据入库 (works, citations, authors, topics, work families)
    │
    ▼
PDF 下载 (并行 ThreadPoolExecutor)
    │
    ▼
MinerU 全文解析 → sections → blocks → chunks
    │
    ├── BM25 全文索引 (works / chunks / formulas / assets)
    ├── 向量索引 (hash-idf-v1 / sentence-transformers)
    ├── 图结构边 (引文、书目耦合、共被引、向量相似度)
    └── structure 上游判断层
            │
            ├── results 抽取
            ├── methods 抽取
            └── transfer 候选生成
                    │
                    ▼
            work / chunk 双层混合检索 + LLM 问答
```

## 安装

```bash
git clone https://github.com/tyy99phy/hep_rag.git
cd hep_rag
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# 可选：本地嵌入模型
pip install -e ".[embeddings]"

# 可选：本地 LLM 推理
pip install -e ".[local-llm]"

# 可选：Chroma 向量数据库
pip install -e ".[vectorstore]"

# 可选：FastAPI 服务层
pip install -e ".[api]"

# 可选：LangChain 外壳
pip install -e ".[langchain]"
```

## 快速开始

```bash
# 1. 初始化配置和工作区
hep-rag init-config --config ./hep-rag.yaml --workspace ./workspace

# 2. 编辑 hep-rag.yaml，填入 MinerU / LLM 凭证（见下方"配置"一节）

# 3. 预览检索结果
hep-rag fetch-papers "rare decay eta to four muons CMS" \
  --config ./hep-rag.yaml --limit 5

# 4. 一键入库（元数据 + PDF 下载 + MinerU 解析 + structure → downstream extractions + 索引 / 图谱）
hep-rag ingest-online "rare decay eta to four muons CMS" \
  --config ./hep-rag.yaml --limit 10 --download-limit 10 --parse-limit 10

# 4b. 导入 PDG 主干语料
# 默认只走 website 路线：
#   - 导入 pdg_sections
#   - 注册 600+ 内嵌 review/listing/table PDFs
hep-rag import-pdg --config ./hep-rag.yaml --collection pdg --edition 2024 --download

# 4b-alt. 如需同时归档 SQLite（不引入 standalone book PDF）
hep-rag import-pdg --config ./hep-rag.yaml --collection pdg --edition 2024 --artifact full --download

# 4c. 让 PDG website 内嵌 PDF 进入 MinerU / structure 链路
hep-rag reparse-pdfs --config ./hep-rag.yaml --collection pdg --limit 4

# 4c-alt. 只解析 website 注册的 PDG PDF，不碰其他 parser
hep-rag reparse-pdfs --config ./hep-rag.yaml --collection pdg --parser-name pdg_website_pdf

# 5. 检索（不调用 LLM）
hep-rag query "eta meson rare decay branching fraction" \
  --config ./hep-rag.yaml --limit 8

# 6. 问答（检索 + LLM 生成）
hep-rag ask "总结 eta -> 4mu 的最新实验结果" \
  --config ./hep-rag.yaml --mode survey

# 7. 启动交互式 Web/API 控制台
hep-rag-api --config ./hep-rag.yaml --host 127.0.0.1 --port 8000
# 然后打开 http://127.0.0.1:8000/ 或 http://127.0.0.1:8000/docs
```

## 配置

`hep-rag init-config` 会生成默认配置文件，关键字段：

```yaml
online:
  max_parallelism: 4                # 在线多 query 检索的最大并行数

download:
  max_download_workers: 4           # PDF 下载并行数

mineru:
  enabled: true                      # 开启全文解析
  api_base: https://mineru.net/api/v4
  api_token: "你的 token"
  oversize_strategy: split           # PDF 超过页数限制时自动分片后再提交 MinerU
  max_pages_per_pdf: 200             # 当前 MinerU API 单次解析页数上限

pdg:
  default_artifact: website          # 默认只导入 website corpus
  sqlite_variant: all                # 下载带历史 Summary Table 数据的 SQLite 版本
  register_embedded_pdfs: true       # 将 website 内嵌 review/listing/table PDFs 注册为正式 parse candidates

embedding:
  model: hash-idf-v1                 # 内置无依赖模型，或填 sentence-transformers 模型名
  allow_silent_fallback: false       # 推荐保持 false；GPU 不可用时直接失败
  runtime:
    device: cuda
    batch_size: 64
    huggingface:
      endpoint: ""                   # 例如 https://hf-mirror.com
      cache_dir: ""                  # 例如 ~/.cache/huggingface
      local_files_only: false
      token: ""

retrieval:
  max_parallelism: 2                # 本地混合检索中 BM25 / 向量的最大并行数

api:
  auth_token: ""                     # 为空表示不鉴权；设置后支持 Bearer / X-API-Key
  enable_ui: true                    # 是否开放内置 Web Console
  job_max_workers: 2
  job_max_events: 1000
```

### LLM 后端

**方式一：OpenAI 兼容接口**（vLLM / Ollama / 任何 OpenAI API 格式的服务）

```yaml
llm:
  enabled: true
  backend: openai_compatible
  api_base: "http://127.0.0.1:8000/v1"  # vLLM / Ollama 等服务地址
  api_key: "EMPTY"                       # 本地服务通常填 EMPTY
  model: "Qwen/Qwen3-32B"
  chat_path: /chat/completions           # 默认值，一般不用改
  temperature: 0.2
  max_tokens: 1200
  timeout_sec: 120
  extra_headers: {}                      # 需要自定义 header 时填写
```

**方式二：本地 HuggingFace 模型**（需要 `pip install -e ".[local-llm]"`）

```yaml
llm:
  enabled: true
  backend: local_transformers
  local_model_path: "/path/to/your/model"
  device: cpu                            # 或 cuda
  torch_dtype: auto
  trust_remote_code: false
```

完整字段说明见 [`config.example.yaml`](config.example.yaml)。

如果你在国内网络环境下使用 `sentence-transformers`，建议显式配置：

```yaml
embedding:
  runtime:
    huggingface:
      endpoint: "https://hf-mirror.com"
      cache_dir: "~/.cache/huggingface"
```

如果 `embedding.runtime.device: cuda`，框架现在会先做 CUDA 预检查；一旦本机 `torch` 和 NVIDIA driver 不兼容，会直接失败并提示修复，不再默默回退到 CPU。

`pypdf` 现在是基础依赖的一部分，用于在框架内部处理超页数 PDF 的自动分页；不需要再在仓库外面写脚本手工 split。

`fetch-papers` / `ingest-online` 在在线检索阶段会先做多 query 改写，再对命中结果做 family-aware 去重后截取 top-N；同一 work 的 note / preprint / article 等相关版本会保留在返回结果的 `related_versions` 中。返回结果里还会带 `local_summary` 和 `local_status`，用于标记本地是否已有该 work、PDF 是否已缓存、MinerU 是否已经 materialize。

## Web / API

FastAPI 服务默认提供：

- `/`：最小 Web 控制台，适合临时用户测试
- `/docs`：Swagger/OpenAPI 交互文档
- `/auth/status`、`/health`：公开状态接口
- `/retrieve`、`/ask`、`/fetch-papers`：同步接口
- `/jobs/ingest-online`、`/jobs/reparse-pdfs`：异步任务接口
- `/jobs/{job_id}`、`/jobs/{job_id}/events`：任务状态与进度事件

如果设置了 `api.auth_token`，除 `/`、`/ui`、`/docs`、`/openapi.json`、`/health`、`/auth/status` 之外的接口都需要带 token。内置 Web Console 支持直接输入 token；异步 job 元数据和事件会持久化到独立的 `workspace/db/hep_rag_api.db`，因此服务重启后仍可查询历史任务，同时避免和主业务库写事务争锁。

### Web UI 使用

先启动服务：

```bash
hep-rag-api --config ./hep-rag.yaml --host 127.0.0.1 --port 8000
```

然后打开 `http://127.0.0.1:8000/`。页面主要分成两块：

- 左侧是输入区和任务日志
- 右侧是结果区，支持 `Display` / `Raw` 两种模式

推荐的使用顺序：

1. 先看顶部 `Workspace` 卡片，确认当前服务绑定的是你期望的 workspace。
2. 如果你设置了 `api.auth_token`，先把 token 填到 `API Token` 输入框；未设置时可以留空。
3. 在 `Query` 里输入主题，例如 `CMS VBS SSWW`，再按需填写 `Collection`、`Limit`、`Target`、`Ask Mode`、`Download Limit`、`Parse Limit`、`Max Parallel`。
4. 点击 `Fetch Papers` 预览在线搜索结果。这个步骤不会改本地数据库，适合先检查 query 改写和召回是否合理。
5. 点击 `Retrieve` 只做本地检索，返回 works / chunks 证据，适合检查当前数据库里能不能回答问题。
6. 点击 `Ask` 会先检索再调用配置好的 LLM，返回最终答案和引用证据。
7. 点击 `Start Ingest Job` 会提交异步在线入库任务；左侧日志区会持续显示进度事件，例如搜索、下载、MinerU 解析、建索引、建图。
8. ingest job 成功后，页面会自动刷新 workspace 统计；如果想手动确认当前库状态，可以再点 `Refresh Workspace`。

几个按钮的含义：

- `Fetch Papers`：在线搜索预览，不写库
- `Retrieve`：只查本地库，不调 LLM
- `Ask`：本地检索 + LLM 生成
- `Start Ingest Job`：在线搜索 + 元数据入库 + PDF 下载 + MinerU 解析 + 索引 / 图谱构建
- `Refresh Workspace`：刷新 works、documents、chunks、citations 统计
- `Clear Log`：清空当前页面上的任务日志显示

页面右侧结果区支持两种模式：`Display` 适合直接阅读问答结果和证据卡片，`Raw` 保留完整 JSON，便于观察真实 API 行为；如果要逐个调试接口参数，可以打开 `/docs` 用 Swagger UI 交互调用。

## CLI 命令一览

| 命令 | 说明 |
|------|------|
| `init-config` | 生成默认配置文件和工作区目录 |
| `init` | 初始化数据库 |
| `fetch-papers` | 在线搜索 InspireHEP，预览候选论文 |
| `ingest-online` | 搜索 + 下载 + 解析 + 结构判断 + 下游抽取 + 建索引（全流程） |
| `reparse-pdfs` | 仅对本地已有 PDF 重新提交 MinerU，并刷新 structure/results/methods/transfer |
| `ingest-metadata` | 仅导入元数据（不下载 PDF） |
| `import-mineru` | 手动导入 MinerU 解析结果 |
| `import-pdg` | 导入 PDG 官方 artifact，并将 website corpus 写入 `pdg_sections` / physics substrate |
| `enrich-inspire-metadata` | 补全引文、摘要等字段 |
| `build-search-index` | 重建 BM25 全文索引 |
| `build-vector-index` | 重建向量索引 |
| `build-graph` | 重建图结构边 |
| `search-bm25` | BM25 关键词检索 |
| `search-vector` | 向量语义检索 |
| `search-hybrid` | 混合检索（自动路由 work/chunk 级别） |
| `query` | 检索证据（可消费 structure/works/chunks，不调用 LLM） |
| `ask` | 检索 + LLM 问答（消费结构化证据外壳） |
| `benchmark-manifest` | 导出 RAG effect / thinking-engine benchmark manifest |
| `show-document` | 查看论文解析结果 |
| `audit-document` | 审查解析质量 |
| `show-graph` | 查看图谱邻居 |
| `status` | 工作区统计 |
| `collections` | 列出所有 collection |
| `bootstrap-legacy-corpus` | 从旧版数据库迁移 |

每个命令加 `--help` 查看详细参数。

## 项目结构

```
hep_rag/
├── pyproject.toml
├── config.example.yaml          # 完整配置模板
├── db/schema.sql                # SQLite schema（参考副本）
│
├── src/hep_rag_v2/
│   ├── schema.sql               # 打包在内的 schema
│   ├── config.py                # 配置加载与合并
│   ├── paths.py                 # 工作区路径管理 (WorkspacePaths)
│   ├── db.py                    # SQLite 连接与初始化
│   ├── metadata.py              # InspireHEP 元数据入库
│   ├── pipeline.py              # 高级工作流（ingest / retrieve / ask）
│   ├── graph.py                 # 图结构边构建
│   ├── query.py                 # 查询改写
│   ├── records.py               # 数据记录类型
│   ├── textnorm.py              # 文本归一化 (CJK 分词、LaTeX 清洗)
│   │
│   ├── fulltext/                # 全文处理
│   │   ├── parser.py            #   MinerU 输出导入
│   │   ├── document.py          #   文档结构化 (sections → blocks)
│   │   └── chunks.py            #   分块 (chunking)
│   │
│   ├── vector/                  # 向量检索
│   │   ├── embedding.py         #   嵌入模型 (hash-idf / sentence-transformers)
│   │   ├── index.py             #   索引构建
│   │   ├── search.py            #   向量 / 混合检索
│   │   └── chroma.py            #   Chroma 向量数据库集成
│   │
│   ├── cli/                     # 命令行接口
│   │   ├── _common.py           #   共享工具函数
│   │   ├── _parser.py           #   argparse 定义
│   │   ├── workspace.py         #   工作区管理命令
│   │   ├── ingest.py            #   数据入库命令
│   │   ├── search.py            #   检索命令
│   │   └── inspect.py           #   审查 / 展示命令
│   │
│   ├── service/                 # 服务层 payload 组装
│   │   ├── workspace.py         #   工作区状态接口
│   │   └── inspect.py           #   文档 / 图谱展示接口
│   │
│   ├── integrations/            # 外部框架适配
│   │   └── langchain_adapter.py #   LangChain retriever / chat / tools
│   │
│   ├── api/                     # FastAPI 服务与内置控制台
│   │   ├── app.py               #   FastAPI 应用入口
│   │   ├── jobs.py              #   异步 job 管理与事件持久化
│   │   └── static/index.html    #   最小 Web Console
│   │
│   └── providers/               # 外部服务适配
│       ├── inspire.py           #   InspireHEP API
│       ├── mineru_api.py        #   MinerU 解析 API
│       ├── openai_compatible.py #   OpenAI 兼容 LLM
│       └── local_transformers.py#   本地 HuggingFace 模型
│
└── tests/
    ├── conftest.py              # pytest fixtures
    ├── test_bootstrap.py        # 入库 / 解析 / 索引 / 图谱集成测试
    ├── test_config_runtime.py   # 配置加载测试
    └── test_vector.py           # 向量检索 / 混合路由测试
```

## 数据库

主业务 SQLite 数据库是 `workspace/db/hep_rag_v2.db`，WAL 模式，26 张表：

- **元数据**: works, work\_ids, authors, collaborations, venues, topics
- **引文网络**: citations, collection\_works
- **全文**: documents, document\_sections, blocks, formulas, assets, chunks
- **图结构**: similarity\_edges, bibliographic\_coupling\_edges, co\_citation\_edges
- **嵌入**: work\_embeddings, chunk\_embeddings
- **运行记录**: collections, ingest\_runs, graph\_build\_runs

API 服务的异步任务状态和事件单独存放在 `workspace/db/hep_rag_api.db`：

- **API 任务**: api\_jobs, api\_job\_events

## 工作区目录

```
workspace/
├── db/hep_rag_v2.db       # 主业务数据库
├── db/hep_rag_api.db      # FastAPI job / event 数据库
├── collections/            # collection 配置 JSON
├── data/
│   ├── raw/inspire/        # InspireHEP 原始响应
│   ├── pdfs/               # 下载的 PDF
│   └── parsed/             # MinerU 解析结果
├── indexes/                # BM25 + 向量索引
└── exports/                # 导出文件
```

## 运行测试

```bash
pip install pytest
python -m pytest tests/ -v
```

更完整的试用与测试流程见：

- [`docs/testing.md`](docs/testing.md)

## License

MIT
