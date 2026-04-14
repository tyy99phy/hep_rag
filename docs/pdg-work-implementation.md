# structure-upstream 实施波次审查与落地约束

这份文档面向当前 `structure-upstream` 收敛波次。目标不是重新发散规划，而是把当前仓库已经确认的规则、上游/下游职责和验收方式固定下来，供实现 lane、验证 lane 与集成 lane 对齐。

## 1. 当前波次的核心判断

当前仓库应被描述为一个 **reasoning substrate**：

- `structure` 是上游判断层
- `results` / `methods` / `transfer` 是消费结构化判断的下游生产层
- 对象合同（尤其状态语义）是这条链路的公共边界

因此，本波次的目标不是“给三个 producer 各自补一点启发式”，而是把 `structure` 变成真正的 truth source。

## 2. 必须保持的硬约束

### 2.1 结构优先

默认 ingest / reparse 路径必须先运行 `build_work_structures()`，再运行：

1. `results`
2. `methods`
3. `transfer`

如果下游 lane 仍然各自独立决定 article-vs-review 策略，说明这条波次还没完成。

### 2.2 状态只允许合同值

顶层状态必须限制在：

- `ready`
- `partial`
- `needs_review`
- `failed`

不允许再产生或传播非合同状态，例如：

- `review_relaxed`
- `needs_attention`
- 任何只在局部 producer 中自造、但对象合同未批准的顶层值

### 2.3 article / review 的统一策略

对 **非综述 article**：

- 必须具备结构层要求的必需签名
- 如果必需签名缺失，必须保留显式合同状态（通常是 `needs_review`，必要时才是 `failed`）
- 不允许静默丢弃

对 **review**：

- 可以放宽必需签名要求
- 但宽松不等于自造状态；仍必须落在合同状态集合内

### 2.4 文档叙事必须同步

README、测试文档与实现文档都必须反映同一件事：

> 这个仓库正在收敛为一个 structure-governed reasoning substrate。

如果文档仍把 `results` / `methods` / `transfer` 写成松散并列、互不共享判断来源的 producer 集合，会误导后续实现者继续复制局部策略。

## 3. 与当前仓库的直接映射

### 3.1 已有稳定支点

- `src/hep_rag_v2/structure.py`
  - 已提供 `build_work_structures()` 入口
  - 已有 article / review / required-signature 相关测试
- `src/hep_rag_v2/object_contracts.py`
  - 已冻结允许状态集合
- `src/hep_rag_v2/pipeline.py`
  - 是默认 ingest / reparse 编排的真实落点
- `docs/hep-core-object-contracts.md`
  - 是状态与对象形状的规范来源，而不是参考意见

### 3.2 本波次需要完成的收敛

1. pipeline 默认顺序先跑 `structure`
2. downstream lanes 消费结构层语义，不再独立决定 article-vs-review 规则
3. 顶层状态对齐对象合同
4. README / docs / tests 都使用同一套叙事

## 4. 推荐验证矩阵

### 4.1 结构与状态语义

至少覆盖：

1. article + required signatures 完整 → `ready`
2. article + 缺少必需签名 → `needs_review`（或有意选择的合同级失败路径）
3. review + 缺少必需签名 → 允许宽松，但状态仍为合同允许值
4. 不出现 `review_relaxed` / `needs_attention` 之类的顶层漂移

优先关注：

```bash
pytest -q tests/test_pdg_structure.py tests/test_object_contracts.py
```

### 4.2 编排顺序与下游一致性

至少覆盖：

1. ingest / reparse 默认路径先跑 structure
2. `results` / `methods` / `transfer` 与 structure 输出保持一致
3. 下游 lane 不再绕过结构层各自做 policy 判断

优先关注：

```bash
pytest -q tests/test_thinking_extraction.py tests/test_bootstrap.py tests/test_config_runtime.py
```

### 4.3 额外回归

如果本波次触及服务、检索或 benchmark 适配层，再补：

```bash
pytest -q tests/test_retrieval_adapter.py tests/test_service_api.py tests/test_benchmark_suite.py
```

## 5. 文档 lane 的完成标准

文档 lane 不负责替别的 lane 写实现细节，但必须做到：

- README 用简洁语言说明 structure-first substrate 当前阶段
- `docs/testing.md` 给出用户可执行的路径，并明确当前波次的验证命令
- 本文档固定结构优先、合同状态、上下游职责与验收口径

## 6. 结论

当前仓库已经不缺“再来一份抽象规划”，真正缺的是三件事同时成立：

1. `structure` 成为默认上游 truth source
2. downstream lanes 共享同一套合同状态与判断语义
3. 文档、测试、实现三者讲的是同一个系统

只要这三件事接上，本仓库就会更接近一个可审查、可复验、可继续扩展的 HEP reasoning substrate，而不是一组彼此松散耦合的抽取脚本。
