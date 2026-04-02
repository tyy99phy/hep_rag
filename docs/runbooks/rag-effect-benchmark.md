# 弱模型 + 外接数据库增益 Benchmark Runbook

## 目标

这套 benchmark 不直接测“大模型绝对能力”，而是测：

> **同一个较弱模型，在“裸答”与“接入数据库/检索证据后”之间，表现提升了多少。**

因此它更适合回答：

- 数据库到底有没有帮助？
- 数据库主要帮在哪些问题类型上？
- 弱模型在哪些任务上仍然救不回来？

## 对照实验矩阵

每个模型至少跑三种场景：

1. `llm_only`
   - 不接数据库
   - 直接让模型回答
2. `llm_plus_retrieve`
   - 接数据库
   - 提供检索出来的 work/chunk 证据
3. `llm_plus_retrieve_and_structure`
   - 接数据库
   - 提供 typed retrieval shell + evidence registry

推荐记录维度：

- `model_label`
- `scenario`
- `case_id`
- `raw_answer`
- `score_total`
- `score_breakdown`
- `notes`

## 问题类型

当前样例集覆盖：

- `latest_result_summary`
- `expected_vs_observed`
- `cross_experiment_comparison`
- `method_reuse`
- `method_transfer`
- `pdg_lookup_future`

这些类型正好对应后续 master plan 的核心价值：

- 结果汇总
- expected/observed 不混淆
- 跨实验比较
- 方法复用
- 方法迁移
- canonical PDG + paper evidence 联动

## 指标建议

### 1. 事实命中

看答案是否覆盖 `must_mention` 中的关键要素。

### 2. 混淆惩罚

看答案是否触发 `must_not_confuse` 中的错误模式，例如：

- expected / observed 混淆
- CMS / ATLAS 混淆
- 不同 channel 当成同一对象

### 3. 证据锚定

看答案是否体现了数据库带来的证据优势：

- 是否引用具体 work/chunk/result
- 是否说明来源或支撑点

### 4. 结构化增益

比较：

- `llm_only` → `llm_plus_retrieve`
- `llm_plus_retrieve` → `llm_plus_retrieve_and_structure`

如果第三档明显更稳，说明 retrieval shell / evidence registry 是有效的。

## 样例集位置

```text
tests/fixtures/rag_effect_benchmark_cases.json
```

## 生成 benchmark manifest

```bash
python scripts/run_rag_effect_benchmark.py \
  --model-label weak-model \
  --print-cases
```

默认输出到：

```text
.omx/benchmarks/<model-label>-rag-effect-manifest.json
```

## 推荐执行顺序

1. 先固定一版弱模型
2. 跑 `llm_only`
3. 跑 `llm_plus_retrieve`
4. 跑 `llm_plus_retrieve_and_structure`
5. 对每个 case 记录结果与评分
6. 汇总哪些类别最受数据库帮助

## 当前状态

当前仓库只落了：

- benchmark case fixture
- scenario manifest 生成脚手架
- runbook

还**没有**落地：

- 真实模型调用执行器
- 自动评分器
- 最终报表聚合

这正是后续可以逐步补的部分。
