# hep_rag

这个仓库提供一条标准流程：在线检索 HEP 论文，下载 PDF，调用 MinerU 解析，建立检索库，然后做查询和问答。

## 1. 安装

```bash
git clone https://github.com/tyy99phy/hep_rag.git
cd hep_rag
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e .
```

## 2. 初始化配置和工作区

```bash
hep-rag init-config --config ./hep-rag.yaml --workspace ./workspace
```

这一步会生成：

- `hep-rag.yaml`
- `./workspace/`

## 3. 修改配置

打开 `hep-rag.yaml`，只先改下面这几项，其他保持默认：

```yaml
mineru:
  enabled: true
  api_token: "你的 MinerU token"

llm:
  enabled: true
  backend: openai_compatible
  api_base: "你的 OpenAI 兼容接口，例如 https://your-endpoint/v1"
  api_key: "你的 API key"
  model: "你要调用的模型名"
```

## 4. 先搜索候选论文

```bash
hep-rag fetch-papers "Higgs boson light pseudoscalars four photons" \
  --config ./hep-rag.yaml \
  --limit 5
```

先确认返回结果里有你想要的论文，再继续下一步。

## 5. 下载、解析、入库

```bash
hep-rag ingest-online "Higgs boson light pseudoscalars four photons" \
  --config ./hep-rag.yaml \
  --limit 5 \
  --download-limit 5 \
  --parse-limit 5
```

这一步会自动完成：

- 在线检索元数据
- 下载 PDF
- 调用 MinerU 解析
- 建立搜索索引
- 建立向量索引
- 建立图结构边

第一次建议先用小一点的 `limit` 跑通流程。

## 6. 检索

```bash
hep-rag query "总结一下 H -> aa 的主要研究路线" \
  --config ./hep-rag.yaml \
  --limit 8
```

这个命令不调用大模型，只返回检索证据。

## 7. 问答

```bash
hep-rag ask "总结一下 H -> aa 的主要研究路线" \
  --config ./hep-rag.yaml \
  --mode survey
```

这个命令会先检索，再调用你配置的模型生成回答。

## 8. 输出在哪里

- `workspace/db/hep_rag_v2.db`：主数据库
- `workspace/data/pdfs/`：下载的 PDF
- `workspace/data/parsed/`：MinerU 解析结果
- `workspace/indexes/`：检索索引
- `workspace/exports/`：导出结果

## 9. 本地模型

如果你不用远程 OpenAI 兼容接口，而是直接加载本地模型，再执行：

```bash
python3 -m pip install -e .[local-llm]
```

然后把 `hep-rag.yaml` 里的：

```yaml
llm:
  enabled: true
  backend: local_transformers
  local_model_path: "/path/to/your/model"
  device: cpu
```

填好即可。
