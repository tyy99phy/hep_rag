# 用户导入与前端测试 Runbook

## 目标

这份 runbook 面向两类需求：

1. **怎样把文章导入当前新底座**
2. **怎样从用户视角测试 Web / API / RAG 行为**

当前仓库已经有新的 retrieval shell、maintenance substrate 和 benchmark 脚手架，但还**不是**完整的 `PDG/result/method` 终态架构。所以这份 runbook 的重点是：

- 验证当前底座是否稳定
- 验证文章导入、检索、问答、前端交互是否正常

## 1. 初始化环境

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[api,langchain]"
```

如果你需要本地 embedding 或本地模型，再按需安装：

```bash
pip install -e ".[embeddings]"
pip install -e ".[local-llm]"
```

## 2. 初始化配置和工作区

```bash
hep-rag init-config --config ./hep-rag.yaml --workspace ./workspace
```

然后编辑 `hep-rag.yaml`，至少确认：

- `collection.name`
- `embedding.model`
- `llm.*`（如果你要测 `ask`）
- `mineru.*`（如果你要测全文解析）

## 3. 先做 metadata-only 导入

这是最推荐的第一步，因为它最能稳定验证当前底座。

### 3.1 先预览候选文章

```bash
hep-rag fetch-papers "same sign WW CMS" \
  --config ./hep-rag.yaml \
  --limit 5
```

看点：

- 召回的主题是否对
- 有没有明显跑偏到无关主题

### 3.2 只导入 metadata，不下载 PDF，不跑 MinerU

```bash
hep-rag ingest-online "same sign WW CMS" \
  --config ./hep-rag.yaml \
  --limit 20 \
  --download-limit 0 \
  --parse-limit 0
```

这一步主要验证：

- 在线检索
- 元数据入库
- work-level 检索与索引

## 4. 再做 selective fulltext 导入

如果 metadata-only 没问题，再测试全文热层。

先确认 `mineru.enabled=true` 且 token 配好，然后执行：

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
- evidence drill-down

## 5. 导入后如何检查

### 5.1 看 workspace 总状态

```bash
hep-rag status --config ./hep-rag.yaml
```

重点看：

- `works`
- `documents`
- `chunks`
- 搜索/向量索引计数

### 5.2 做纯检索测试

```bash
hep-rag query "CMS VBS SSWW latest result" \
  --config ./hep-rag.yaml \
  --limit 8
```

看点：

- 返回是不是相关 papers / chunks
- 有没有明显把 CMS / ATLAS / 不同物理过程混掉

### 5.3 做问答测试

```bash
hep-rag ask "总结 CMS VBS SSWW 的最新结果" \
  --config ./hep-rag.yaml \
  --mode survey
```

看点：

- 答案是否依赖具体证据
- 是否能把主题说清楚
- 有没有明显幻觉

## 6. 用户端 Web / API 测试

### 6.1 启动服务

```bash
hep-rag-api --config ./hep-rag.yaml --host 127.0.0.1 --port 8000
```

打开：

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/docs`

### 6.2 最小用户测试顺序

推荐按这个顺序：

1. `Fetch Papers`
2. `Retrieve`
3. `Ask`
4. `Start Ingest Job`
5. `Refresh Workspace`

### 6.3 页面侧重点

#### A. `Fetch Papers`

看：

- 预览结果是否合理
- 结果标题/来源链接是否正常

#### B. `Retrieve`

看：

- works / chunks 是否和 query 对齐
- 前端结果区是否显示正常

#### C. `Ask`

看：

- 答案是否生成成功
- 是否带证据引用
- Raw / Display 两种视图是否都正常

#### D. `Start Ingest Job`

看：

- 左侧日志是否持续刷新
- job 完成后 workspace 状态是否变化

## 7. 推荐的用户端测试问题

### 结果类

- `总结 CMS VBS SSWW 的最新结果`
- `比较 CMS 和 ATLAS 在某个相同过程上的近期结果`
- `expected 和 observed limit 有什么区别`

### 方法类

- `哪些分析使用了 profile likelihood`
- `哪些分析依赖 jet substructure`

### 证据类

- `给我看这个结果对应的具体证据`
- `这个结论来自哪篇文章`

## 8. 当前最有价值的 benchmark 入口

### A. metadata-only scale benchmark

```bash
python scripts/run_scale_benchmark.py \
  --tier 10k \
  --workspace-root /tmp/hep-rag-bench-10k
```

### B. 弱模型 + 数据库增益 benchmark manifest

```bash
python scripts/run_rag_effect_benchmark.py \
  --model-label weak-model \
  --print-cases
```

这一步不会真正跑模型，只会生成后续评测所需的：

- case 集
- scenario 集
- manifest

## 9. 一份最小验收清单

如果你只想快速判断“当前系统能不能用”，至少过这几个点：

- `fetch-papers` 能返回合理候选
- metadata-only ingest 能成功
- `query` 返回相关 works/chunks
- `ask` 能生成答案
- Web 控制台能正常交互
- `python -m pytest -q` 全绿

## 10. 当前诚实说明

当前仓库适合测试：

- metadata-first retrieval substrate
- selective fulltext hot lane
- typed retrieval / evidence shell
- 基础用户端交互

当前仓库还不适合直接测试：

- 完整 PDG spine
- typed result objects
- typed method transfer objects

这些是后续架构演化方向，不是当前已完整交付面。
