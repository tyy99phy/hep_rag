# Step 3：metadata-only scale benchmark runbook

## 目标

在不改 schema / pipeline / CLI 主逻辑的前提下，提供一套可复现的 Step 3 benchmark harness：

- 支持 `10k / 50k / 100k` metadata-only tier
- 记录 ingest、work-search、work-vector 的基线耗时
- 将结果导出到 workspace `exports/benchmarks/*.json`
- `100k` 默认允许先做 `--dry-run` 计划校验，不要求当前 lane 真实跑完

## 推荐命令

### 10k

```bash
python scripts/run_scale_benchmark.py \
  --tier 10k \
  --workspace-root /tmp/hep-rag-bench-10k
```

### 50k

```bash
python scripts/run_scale_benchmark.py \
  --tier 50k \
  --workspace-root /tmp/hep-rag-bench-50k
```

### 100k（先 dry-run）

```bash
python scripts/run_scale_benchmark.py \
  --tier 100k \
  --workspace-root /tmp/hep-rag-bench-100k \
  --dry-run
```

### 100k（如果机器资源允许，再执行实际 run）

```bash
python scripts/run_scale_benchmark.py \
  --tier 100k \
  --workspace-root /tmp/hep-rag-bench-100k \
  --query-repeats 1
```

## 导出内容

每次运行都会输出 JSON，默认写到：

```text
<workspace-root>/exports/benchmarks/<tier>-metadata-benchmark.json
```

关键字段：

- `ingest.work_count`
- `ingest.works_per_second`
- `search.build_seconds`
- `search.latency_ms.p50 / p95`
- `vectors.build_seconds`
- `vectors.latency_ms.p50 / p95`
- `sqlite_size_mb`
- `memory_peak_mb`

## 说明

- 该 harness 目前聚焦 **metadata-only** lane，因此只覆盖 `works` 数据层，不构造 fulltext chunk 热层。
- search/vector 均使用现有 repo 能直接调用的内部能力；不引入新的运行时依赖。
- `100k` tier 的首选动作是先 `--dry-run` 校验导出路径、查询集与 workspace 规划，再决定是否真实执行。
