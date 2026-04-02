# 测试方法

这份文档只保留用户真正会用到的测试路径，不放内部主计划或夜间执行说明。

如果你在推进当前 `PDG/work` 双层图谱实现波次，请同时查看 [`docs/pdg-work-implementation.md`](./pdg-work-implementation.md)；那份文档定义了这条开发线的显式约束、异常处理要求和验收清单。

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

## 1.5 PDG archival ingest 骨架冒烟

如果你正在推进 PDG/work 结构化导入，而手头已经有本地 PDG PDF，可以先验证骨架入口：

```bash
hep-rag import-pdg \
  --config ./hep-rag.yaml \
  --collection pdg \
  --edition 2024 \
  --pdf /path/to/pdg-2024.pdf
```

这一步当前主要验证：

- PDG edition 元数据是否能解析成稳定 canonical id
- 本地 PDF 是否能进入 workspace PDF 区
- archival ingest stub / 后续 MinerU 接口路径是否稳定

## 2. selective fulltext 测试

如果 metadata-only 没问题，再测全文热层。

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
- 证据 drill-down

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
- `Retrieve`：works/chunks 是否与 query 对齐
- `Ask`：答案是否带证据、有没有明显幻觉
- `Start Ingest Job`：日志是否持续刷新、完成后统计是否变化

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

也就是专门用来比较：

> 同一个较弱模型，在“不接数据库”和“接数据库”时，表现差了多少。

## 6. 最小验收清单

如果你只想快速判断当前系统能不能用，至少过这几个点：

- `fetch-papers` 返回合理候选
- metadata-only ingest 成功
- `query` 返回相关 works/chunks
- `ask` 能生成答案
- Web 控制台交互正常
- `python -m pytest -q` 全绿

## 7. 当前边界

当前仓库适合测试：

- metadata-first retrieval substrate
- selective fulltext hot lane
- typed retrieval / evidence shell
- 基础用户端交互
- benchmark 脚手架

当前仓库还不适合直接测试：

- 完整 PDG spine
- typed result objects
- typed method transfer objects

这些仍然属于后续架构演化方向。
