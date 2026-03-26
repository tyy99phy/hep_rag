from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import yaml

from hep_rag_v2 import paths


DEFAULT_CONFIG_NAME = "hep-rag.yaml"
DEFAULT_INSPIRE_FIELDS = [
    "titles",
    "abstracts",
    "arxiv_eprints",
    "authors",
    "citation_count",
    "collaborations",
    "control_number",
    "dois",
    "documents",
    "earliest_date",
    "files",
    "keywords",
    "preprint_date",
    "publication_info",
    "references",
]

DEFAULT_CONFIG: dict[str, Any] = {
    "workspace": {
        "root": "./workspace",
    },
    "collection": {
        "name": "default",
        "label": "Default HEP collection",
        "notes": "Online-ingested HEP literature graph",
    },
    "online": {
        "provider": "inspirehep",
        "published_only": False,
        "query_suffix": "",
        "page_size": 25,
        "timeout_sec": 60,
        "retries": 3,
        "sleep_sec": 0.2,
        "fields": DEFAULT_INSPIRE_FIELDS,
    },
    "query_rewrite": {
        "enabled": True,
        "max_queries": 4,
        "per_query_limit": 15,
        "temperature": 0.0,
        "max_tokens": 320,
    },
    "download": {
        "timeout_sec": 120,
        "retries": 3,
        "verify_ssl": True,
        "max_download_workers": 4,
    },
    "mineru": {
        "enabled": False,
        "api_base": "https://mineru.net/api/v4",
        "api_token": "",
        "model_version": "pipeline",
        "is_ocr": False,
        "enable_formula": True,
        "enable_table": True,
        "language": "en",
        "poll_interval_sec": 10,
        "max_wait_sec": 1800,
        "timeout_sec": 120,
    },
    "embedding": {
        "model": "hash-idf-v1",
        "dim": 768,
        "build_after_ingest": True,
        "chroma": {
            "enabled": False,
            "dir": None,
        },
    },
    "ingest": {
        "chunk_size": 2400,
        "overlap_blocks": 1,
        "section_parent_char_limit": 12000,
    },
    "retrieval": {
        "target": "auto",
        "limit": 8,
        "graph_expand": None,
        "seed_limit": 5,
        "chunk_limit": 12,
        "answer_evidence_limit": 6,
    },
    "api": {
        "auth_token": "",
        "enable_ui": True,
        "job_max_workers": 2,
        "job_max_events": 1000,
    },
    "llm": {
        "enabled": False,
        "backend": "openai_compatible",
        "api_base": "http://127.0.0.1:8000/v1",
        "api_key": "EMPTY",
        "model": "Qwen/Qwen3-32B",
        "chat_path": "/chat/completions",
        "temperature": 0.2,
        "max_tokens": 1200,
        "timeout_sec": 120,
        "extra_headers": {},
        "local_model_path": "",
        "device": "cpu",
        "torch_dtype": "auto",
        "trust_remote_code": False,
    },
}


def default_config(*, workspace_root: str | Path | None = None) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if workspace_root is not None:
        config["workspace"]["root"] = str(Path(workspace_root).expanduser())
    return config


def resolve_config_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path).expanduser().resolve()

    env_path = os.environ.get("HEP_RAG_CONFIG")
    if env_path:
        return Path(env_path).expanduser().resolve()

    cwd = Path.cwd()
    for candidate in (
        cwd / "hep-rag.yaml",
        cwd / "hep-rag.yml",
        cwd / "hep-rag.json",
    ):
        if candidate.exists():
            return candidate.resolve()
    return (cwd / DEFAULT_CONFIG_NAME).resolve()


def load_config(path: str | Path | None = None) -> tuple[Path, dict[str, Any]]:
    config_path = resolve_config_path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    suffix = config_path.suffix.lower()
    raw_text = config_path.read_text(encoding="utf-8")
    if suffix == ".json":
        loaded = json.loads(raw_text)
    else:
        loaded = yaml.safe_load(raw_text)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")

    merged = default_config()
    _deep_update(merged, loaded)
    return config_path, merged


def save_config(config: dict[str, Any], path: str | Path | None = None, *, overwrite: bool = False) -> Path:
    config_path = resolve_config_path(path)
    if config_path.exists() and not overwrite:
        raise FileExistsError(f"Config already exists: {config_path}")
    config_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = config_path.suffix.lower()
    if suffix == ".json":
        content = json.dumps(config, ensure_ascii=False, indent=2) + "\n"
    else:
        content = yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
    config_path.write_text(content, encoding="utf-8")
    return config_path


def apply_runtime_config(
    *,
    config_path: str | Path | None = None,
    workspace_root: str | Path | None = None,
    create_default: bool = False,
) -> tuple[Path, dict[str, Any]]:
    resolved_path = resolve_config_path(config_path)
    if resolved_path.exists():
        loaded_path, config = load_config(resolved_path)
    else:
        if not create_default:
            raise FileNotFoundError(f"Config not found: {resolved_path}")
        config = default_config(workspace_root=workspace_root or (resolved_path.parent / "workspace"))
        loaded_path = save_config(config, resolved_path, overwrite=False)

    root = _resolve_workspace_root(
        config=config,
        config_path=loaded_path,
        workspace_root=workspace_root,
    )
    paths.set_workspace_root(root)
    config["workspace"]["root"] = str(root)
    return loaded_path, config


def runtime_collection_config(config: dict[str, Any], *, name: str | None = None) -> dict[str, Any]:
    collection = config.get("collection") or {}
    collection_name = str(name or collection.get("name") or "default").strip() or "default"
    fields = list(config.get("online", {}).get("fields") or DEFAULT_INSPIRE_FIELDS)
    return {
        "name": collection_name,
        "label": str(collection.get("label") or collection_name).strip() or collection_name,
        "notes": str(collection.get("notes") or "").strip() or None,
        "source_priority": ["inspirehep", "mineru"],
        "queries": {"inspire": []},
        "fields": fields,
    }


def _resolve_workspace_root(
    *,
    config: dict[str, Any],
    config_path: Path,
    workspace_root: str | Path | None,
) -> Path:
    if workspace_root is not None:
        return Path(workspace_root).expanduser().resolve()

    configured = str((config.get("workspace") or {}).get("root") or "").strip()
    if not configured:
        return (config_path.parent / "workspace").resolve()

    root = Path(configured).expanduser()
    if not root.is_absolute():
        root = (config_path.parent / root).resolve()
    return root


def _deep_update(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
