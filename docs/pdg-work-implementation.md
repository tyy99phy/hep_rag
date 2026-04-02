# PDG/work 实施波次审查与落地约束

这份文档面向当前 `PDG/work` 双层图谱实现波次，目标不是再做泛泛规划，而是把已经确认的设计约束落到代码入口、配置接口和验证清单上，方便实现 lane、验证 lane 和集成 lane 对齐。

## 1. 现状审查（基于当前仓库代码）

### 1.1 已有可复用基础

- `src/hep_rag_v2/config.py`
  - 已有稳定的配置合并与运行时加载入口。
  - 未限制未知字段，因此可以安全承载后续 `pdg` / `profile` 类配置，不会阻塞现有 CLI。
- `src/hep_rag_v2/metadata.py`
  - 已有文献类型判定能力，能稳定区分 `article` / `review` / `note` / `proceedings`。
  - 这意味着“默认层对非综述文章强制抽取 result/method signature、综述可例外”的门控条件已经有可靠落点。
- `src/hep_rag_v2/pipeline.py`
  - 现有 ingest 主流程已负责在线检索、去重、下载、解析、建索引、建图，适合作为 PDG import pipeline 的骨架入口。
  - 当前缺的是“结构抽取与异常登记”这一段，而不是重写整条 ingest 链路。
- `src/hep_rag_v2/service/inspect.py`
  - 已有 `needs_manual_review` 这样的显式审查状态。
  - 后续如果非综述文章抽不出必需签名，建议沿用这一类显式状态，而不是静默跳过。
- `src/hep_rag_v2/vector/embedding.py`
  - 已支持内置 hash / hash-idf 与 `st:` / `sentence-transformers:` 前缀的本地 embedding 模型接入。
  - 这为“结构构建模型”和“本地 embedding 模型”分离提供了现成接入点。
- `src/hep_rag_v2/providers/local_transformers.py`
  - 已支持本地 Transformers 推理，含 `device` / `torch_dtype` / `trust_remote_code` 等参数。
  - 这正是 3090 友好部署的可复用入口。

### 1.2 当前明显缺口

- 还没有显式 `mode/profile` 配置契约，后续容易退化成隐式分支逻辑。
- 还没有面向 `work/result/method` 默认层的必需签名抽取入口。
- 还没有“抽取失败必须显式标异常/待补”的统一状态模型与验收测试。
- 用户文档仍以当前稳定 CLI 为主，尚未把本波次实现约束整理成可执行的开发/验证说明。

## 2. 本波次必须保持的硬约束

### 2.1 显式 mode/profile

后续实现必须把运行模式写成显式配置，不允许靠“是否开启某个 flag”来隐式推断层级。

建议新增的配置段如下（当前仓库可以先写入配置文件，未接线字段不会破坏现有运行）：

```yaml
pdg:
  mode: default
  profile: work_result_method
  require_result_signature: true
  require_method_signature: true
  allow_review_without_signatures: true
  on_missing_signature: needs_manual_review
```

约束说明：

- `mode`：决定导入/检索走哪条结构层。
- `profile`：决定默认层强制项，不要把规则散落在代码里。
- `on_missing_signature`：必须是显式异常策略，不能默默忽略。

### 2.2 默认层的强制签名规则

对每篇**非综述**文章：

- 必须产出 `result signature`
- 必须产出 `method signature`

如果任一签名无法稳定抽出，系统必须执行下列二选一之一：

1. 标记为异常；或
2. 标记为待人工补录 / 待复审

无论采用哪种状态，都必须满足：

- 在数据库或可检查的结构化产物里可见
- 在 inspect / audit / 验收日志里可见
- 不允许仅因为抽取失败就把文章从默认层静默丢掉

### 2.3 结构构建模型与 embedding 模型解耦

本波次必须继续坚持两套模型分工：

- **结构构建模型**：负责 PDG/work/result/method 抽取
- **embedding 模型**：负责本地向量检索

不要把“能生成结构的模型”直接当作默认 embedding 模型；两者生命周期、资源占用和失败模式不同。

## 3. 推荐的代码落点（避免重复造轮子）

### 3.1 配置入口

优先沿用：

- `src/hep_rag_v2/config.py`
- `config.example.yaml`

建议把 PDG/work 波次新增配置统一挂在 `pdg:` 下，而不是把多个布尔开关撒到 `ingest:` / `retrieval:` / `llm:` 顶层。

### 3.2 导入骨架

优先沿用：

- `src/hep_rag_v2/pipeline.py`

推荐新增的职责切分：

1. online hit / metadata 入库
2. 文献类型判定（article vs review）
3. 默认层签名抽取
4. 抽取失败显式登记
5. 成功后再进入 work/result/method/PDG 写入
6. 最后再触发 search / vector / graph 同步

这样可以把“必需签名校验”放在结构写入之前，避免后面出现半成品 work capsule。

### 3.3 文献类型与异常状态

优先沿用：

- `src/hep_rag_v2/metadata.py` 中已有的 `article/review` 判定
- `src/hep_rag_v2/service/inspect.py` 中已有的 `needs_manual_review`

建议统一语义：

- `review`：允许没有 result/method signature
- `article`：默认必须有 result/method signature
- `needs_manual_review`：抽取不稳定或缺关键签名时的默认兜底状态

## 4. 3090 友好的本地 embedding / LLM 接入建议

### 4.1 embedding 侧

当前仓库已经支持：

```yaml
embedding:
  model: st:/path/to/local-or-hf-sentence-transformer
```

或者：

```yaml
embedding:
  model: sentence-transformers:/path/to/local-or-hf-sentence-transformer
```

注意：

- `src/hep_rag_v2/vector/embedding.py` 只接受 `st:` 或 `sentence-transformers:` 前缀；文档和示例不要漏掉前缀。
- 如果只是为了 3090 上稳定跑通本地检索，优先保持 embedding 模型小而稳，不要和结构抽取 LLM 绑死。

### 4.2 本地结构模型 / 问答模型侧

当前仓库已经支持：

```yaml
llm:
  enabled: true
  backend: local_transformers
  local_model_path: /path/to/your/model
  device: auto
  torch_dtype: fp16
```

在 3090 环境下，建议优先尝试：

- `device: auto`
- `torch_dtype: fp16`（若模型更适合 `bf16` 再单独切换）

这样能直接复用 `src/hep_rag_v2/providers/local_transformers.py` 的现有加载路径，不需要额外引入新依赖层。

## 5. 验证 lane 必须盯住的检查项

### 5.1 当前仓库通用检查

每次改动后至少跑：

```bash
python -m pytest -q
python -m compileall src tests
```

如果环境里装了 `ruff`，再补：

```bash
ruff check src tests
```

### 5.2 本波次专用验收标准

在 PDG/work 默认层真正接通后，验证 lane 至少要覆盖以下场景：

1. **非综述文章 + 两类签名齐全**
   - 成功进入默认层
   - work/result/method 结构可见
2. **非综述文章 + 缺少 result signature**
   - 不得静默跳过
   - 必须产生显式异常或待补状态
3. **非综述文章 + 缺少 method signature**
   - 不得静默跳过
   - 必须产生显式异常或待补状态
4. **综述文章**
   - 允许不强制要求 result/method signature
   - 但仍应保留 review 身份与可检索元数据
5. **本地 embedding 模型切换**
   - `hash-idf-v1` 与 `st:` 模型至少各过一次索引构建/检索冒烟
6. **3090 本地部署**
   - 本地 `local_transformers` 路径至少过一次加载与最小问答冒烟

### 5.3 建议新增测试名（供实现 lane 对齐）

下面这些测试名不是空想设计，而是为了保证实现完成后能被持续回归：

- `tests/test_pdg_config.py`
- `tests/test_pdg_signature_policy.py`
- `tests/test_pdg_import_pipeline.py`
- `tests/test_pdg_review_exceptions.py`
- `tests/test_local_embedding_profiles.py`

## 6. 对文档与集成 lane 的直接建议

- README 只保留用户入口和文档链接，不要把实现细节全塞进主页。
- `docs/testing.md` 继续保留用户测试路径，但应明确：PDG/work 波次约束与验收标准以本文件为准。
- 一旦默认层签名抽取代码落地，优先更新本文件和对应测试，而不是再开一份新的“临时计划”。

## 7. 结论

当前仓库已经具备四个关键支点：

1. 稳定的配置入口
2. 可复用的 ingest 骨架
3. 已有的 article/review 判定
4. 本地 embedding / 本地 Transformers 接入点

因此，本波次真正需要补的是：**显式配置契约、必需签名抽取、失败显式登记、验证闭环**。只要这四件事接上，PDG/work 双层图谱就能从“计划”进入“可运行实现”。
