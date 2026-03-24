# hep-rag-v2

一个面向 HEP 论文场景的、配置驱动的文献图谱与检索框架。

它的目标不是“让 LLM 全程指挥开发环境”，而是让最终用户通过一份 `config` 和几条固定命令，就可以完成：

- 在线检索论文
- 下载 PDF
- 调用 MinerU 解析全文
- 建立 `work / document / section / block / chunk` 结构
- 建立 citation / similarity graph
- 进行结构化检索
- 可选地调用 OpenAI-compatible 模型做综述/回答/idea 提炼

仓库默认不附带任何预加载论文、任何私有 API key、任何项目专属 endpoint。
真正使用时，用户需要自己在 `config` 中填写自己的 token、自己的远程接口，或者自己的本地模型路径。

默认设计原则：

- 图谱、chunking、embedding、图构建尽量走算法路径
- LLM 只放在最终可选的 `ask` 层，不参与基础建库
- 数据工作区和代码仓库分离
- 不预加载任何本地论文

## 1. 当前仓库已经具备什么

底层核心仍然是现在这套图谱和全文结构：

- metadata graph: `collections / works / authors / topics / citations`
- document graph: `documents / document_sections / blocks / formulas / assets / chunks`
- search: SQLite FTS5 BM25
- vector: `hash-idf-v1` 默认本地向量，也支持 `st:<sentence-transformers-model>`
- graph edges: bibliographic coupling / co-citation / similarity
- retrieval: BM25 + vector + graph expansion 的 hybrid 检索

新加的用户层能力：

- `init-config`: 初始化配置和空 workspace
- `fetch-papers`: 在线搜论文，只看候选结果
- `ingest-online`: 在线搜论文、下载 PDF、可选送 MinerU、再建索引
- `query`: 不走 LLM，只返回结构化检索证据
- `ask`: 在 `query` 基础上调用 OpenAI-compatible 模型生成回答

## 2. 安装

```bash
pip install -e .
```

如果你要用真实 dense embedding：

```bash
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .[embeddings]
```

如果你要用 Chroma 作为可选向量存储：

```bash
pip install -e .[vectorstore]
```

如果你要直接加载本地 Transformers 模型做 `ask`：

```bash
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .[local-llm]
```

## 3. 快速开始

先初始化一份配置和一个空 workspace：

```bash
hep-rag init-config --config ./hep-rag.yaml --workspace ./workspace
```

然后编辑 `hep-rag.yaml`，填入：

- `mineru.api_token`
- 如果走远程接口：
  - `llm.backend: openai_compatible`
  - `llm.api_base`
  - `llm.api_key`
  - `llm.model`
- 如果走本地直连模型：
  - `llm.backend: local_transformers`
  - `llm.local_model_path`
  - `llm.device`

如果你暂时只做算法检索，不做 MinerU 或 LLM，可以把：

- `mineru.enabled: false`
- `llm.enabled: false`

保持默认即可。

先在线看候选论文：

```bash
hep-rag fetch-papers "Higgs boson light pseudoscalars four photons" --config ./hep-rag.yaml --limit 5
```

再把候选论文拉下来并入库：

```bash
hep-rag ingest-online "Higgs boson light pseudoscalars four photons" \
  --config ./hep-rag.yaml \
  --limit 10
```

只做检索，不调大模型：

```bash
hep-rag query "综述一下 H -> aa 相关工作" --config ./hep-rag.yaml --limit 8
```

如果配置了 OpenAI-compatible endpoint：

```bash
hep-rag ask "综述一下 H -> aa 相关工作" --config ./hep-rag.yaml --mode survey
```

## 4. 配置文件

仓库里提供了一个示例文件：

- `config.example.yaml`

核心配置项：

- `workspace.root`
  代码仓库之外的数据工作区，里面会放 `db/ data/ indexes/ exports/`
- `collection.name`
  当前默认 collection 名称
- `online.*`
  INSPIRE 在线检索参数
- `download.*`
  PDF 下载超时、重试、SSL 校验
- `mineru.*`
  MinerU 官方 API 配置
- `embedding.*`
  向量模型与索引构建策略
- `llm.*`
  问答后端配置，支持两种模式：
  - `openai_compatible`
  - `local_transformers`

两种 LLM 接入方式：

- `openai_compatible`
  适合 OpenAI API、兼容 OpenAI 的云端服务、以及本地 OpenAI-compatible server
- `local_transformers`
  适合直接加载本地 HuggingFace/Transformers 模型，不需要任何外部 API

例如：

- 本地 vLLM / SGLang / LM Studio 可以走 `openai_compatible`
- 本地下载好的 Qwen / Llama / Mistral 权重可以走 `local_transformers`

## 5. 输出约定

### `fetch-papers`

返回候选论文列表，每条至少包含：

- 标题
- 年份
- `inspire / arxiv / doi`
- 可尝试的 PDF URL 列表

### `ingest-online`

返回一次完整导入的摘要：

- metadata 命中数
- 创建/更新了多少 `works`
- 下载了多少 PDF
- MinerU 成功/失败了多少篇
- 是否重建了 search / vector / graph
- 当前 workspace 的快照统计

### `query`

返回结构化证据，不做 LLM 合成：

- `works`: 论文级候选
- `evidence_chunks`: 支撑片段
- `routing`: 这次查询被路由到 `works` 还是 `chunks`

这适合：

- 人工 review
- 后续接你自己的前端
- 给别的 agent / script 做二次处理

### `ask`

返回：

- `answer`: 模型生成的回答
- `evidence.works`
- `evidence.chunks`

也就是说，回答层不是黑盒，证据仍然保留。

`ask` 目前支持两种后端：

1. `llm.backend: openai_compatible`
2. `llm.backend: local_transformers`

## 6. 这次封装和现成小脚本的区别

用户提到的这个脚本：

- `https://github.com/PKUfudawei/graphRAG/blob/master/scripts/get_papers.py`

它的思路比较直接：

- 先从 INSPIRE 拉 metadata
- 再优先 DOI，失败就用 arXiv PDF 下载

这个思路能用，但不够稳。现在这里的下载链路做了几件更适合长期用的事：

- 优先使用 INSPIRE `documents` / `files` 里已经给出的 PDF 线索
- 再回退到 arXiv PDF
- DOI 放到更后面，只作为兜底
- 下载后做 `content-type` / `%PDF-` 检查，避免把 HTML landing page 当 PDF 收下
- 下载、解析、入库、建索引放进同一条可复用管线里

也就是说，这里不是只把 `get_papers.py` 抄过来，而是把它升级成了“可持续维护的 ingest pipeline”。

## 7. MinerU 接入说明

当前实现对接的是 MinerU 官方批量 API 流程：

1. 调 `file-urls/batch` 申请上传地址
2. 把本地 PDF `PUT` 到返回的 URL
3. 轮询 `extract-results/batch/{batch_id}`
4. 下载 `full_zip_url`
5. 导入并 materialize 成本项目的结构化文档

这条链路和官方文档一致，适合“先在线下载 PDF，再本地上传给 MinerU”的工作流。

## 8. LLM 在哪一层介入

默认不介入基础建库。

也就是：

- chunking: 算法
- embedding: 算法 / 本地模型
- graph build: 算法
- retrieval: 算法
- answer / survey / idea: 可选 LLM

这和本项目一开始的目标一致：先把“可扩展的科研文献图谱底座”做好，再把 LLM 放在最后一公里。

## 9. 适合直接推到 GitHub 吗

可以。

当前仓库已经满足：

- 可以直接作为代码仓库发布
- 不依赖我们的私有 API
- 支持远程 OpenAI-compatible 接口
- 支持直接加载本地 Transformers 模型
- 数据工作区和代码仓库分离
- `.gitignore` 已排除数据库、索引、导入数据、构建产物

所以初始化 git 后，正常 `git add .` 不会把本地语料和索引一起提交。

## 10. 已保留的底层开发命令

原来的开发向命令还都在：

```bash
hepv2 build-search-index --target all
hepv2 build-vector-index --target all
hepv2 build-graph --target all --collection cms_rare_decay
hepv2 search-hybrid "综述一下 H -> aa 相关工作" --target auto --collection cms_rare_decay
```

所以现在是两层入口并存：

- `hep-rag ...` 给最终用户
- `hepv2 ...` 给开发和调试

## 11. 参考

- MinerU 官方 API 文档: https://mineru.net/doc/docs/
- MinerU 输出文件说明: https://opendatalab.github.io/MinerU/reference/output_files/
- INSPIRE literature API: https://inspirehep.net/info/hep/api
- PKUfudawei/graphRAG `get_papers.py`: https://github.com/PKUfudawei/graphRAG/blob/master/scripts/get_papers.py
