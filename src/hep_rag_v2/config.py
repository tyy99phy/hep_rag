from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import yaml

from hep_rag_v2 import paths


DEFAULT_CONFIG_NAME = "hep-rag.yaml"
DEFAULT_EMBEDDING_PROFILES: dict[str, dict[str, Any]] = {
    "bootstrap": {
        "model": "hash-idf-v1",
        "dim": 768,
        "runtime": {
            "device": "cpu",
            "batch_size": 32,
            "huggingface": {
                "endpoint": "",
                "cache_dir": "",
                "local_files_only": False,
                "token": "",
            },
        },
    },
    "semantic_small_local": {
        "model": "sentence-transformers:BAAI/bge-small-en-v1.5",
        "dim": 384,
        "runtime": {
            "device": "cuda",
            "batch_size": 64,
            "huggingface": {
                "endpoint": "",
                "cache_dir": "",
                "local_files_only": False,
                "token": "",
            },
        },
    },
    "semantic_prod_local": {
        "model": "sentence-transformers:BAAI/bge-base-en-v1.5",
        "dim": 768,
        "runtime": {
            "device": "cuda",
            "batch_size": 32,
            "huggingface": {
                "endpoint": "",
                "cache_dir": "",
                "local_files_only": False,
                "token": "",
            },
        },
    },
}
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
    "modes": {
        "build": "full",
        "retrieval": "hybrid",
    },
    "profiles": {
        "structure": "default",
        "embedding": "bootstrap",
        "pdg": "default",
    },
    "workspace": {
        "root": "./workspace",
    },
    "build": {
        "mode": "full",
        "structure_backend": "api_llm",
        "embedding_source": "local_profile",
        "allow_silent_fallback": False,
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
        "max_parallelism": 4,
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
        "oversize_strategy": "split",
        "max_pages_per_pdf": 200,
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
        "profile": "bootstrap",
        "build_after_ingest": True,
        "allow_silent_fallback": False,
        "runtime": {
            "device": "cpu",
            "batch_size": 32,
            "huggingface": {
                "endpoint": "",
                "cache_dir": "",
                "local_files_only": False,
                "token": "",
            },
        },
        "chroma": {
            "enabled": False,
            "dir": None,
        },
    },
    "embedding_profiles": copy.deepcopy(DEFAULT_EMBEDDING_PROFILES),
    "ingest": {
        "chunk_size": 2400,
        "overlap_blocks": 1,
        "section_parent_char_limit": 12000,
    },
    "structure": {
        "require_default_signatures": True,
        "result_limit": 8,
        "method_limit": 8,
        "profile": "default",
        "builder": "heuristic-v1",
        "backend": "api_llm",
        "allow_silent_fallback": False,
    },
    "pdg": {
        "profile": "default",
        "source_id": "pdg",
        "title": "Particle Data Group",
        "default_artifact": "full",
        "sqlite_variant": "all",
        "register_embedded_pdfs": True,
        "max_capsule_chars": 1200,
    },
    "retrieval": {
        "target": "auto",
        "mode": "hybrid",
        "limit": 8,
        "max_parallelism": 2,
        "graph_expand": None,
        "seed_limit": 5,
        "chunk_limit": 12,
        "answer_evidence_limit": 6,
    },
    "reasoning": {
        "trace_persistence": "structured_summary",
        "store_raw_fragments": False,
        "max_steps": 24,
        "session_timeout_sec": 900,
    },
    "ideas": {
        "max_candidates": 8,
        "max_evidence_links": 6,
        "score_axes": [
            "physics_novelty",
            "method_novelty",
            "feasibility",
            "evidence_coverage",
            "consensus",
        ],
    },
    "transfer": {
        "max_candidates": 8,
        "max_edges_per_candidate": 4,
        "min_score": 0.0,
        "require_evidence": True,
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
        "answer_strategy": "auto",
        "map_reduce_enabled": True,
        "map_reduce_max_communities": 3,
        "map_reduce_child_limit": 3,
        "map_max_tokens": 500,
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


def resolve_mode(config: dict[str, Any], kind: str, *, default: str) -> str:
    modes = config.get("modes") or {}
    legacy = config.get(kind) or {}
    value = str(modes.get(kind) or legacy.get("mode") or default).strip()
    return value or default


def resolve_embedding_profile(config: dict[str, Any], *, name: str | None = None) -> dict[str, Any]:
    profiles = copy.deepcopy(DEFAULT_EMBEDDING_PROFILES)
    custom_profiles = config.get("embedding_profiles") or {}
    _deep_update(profiles, custom_profiles)

    embedding = config.get("embedding") or {}
    selected = (
        name
        or str((config.get("profiles") or {}).get("embedding") or "").strip()
        or str(embedding.get("profile") or "").strip()
        or "bootstrap"
    )
    payload = profiles.get(selected)
    if payload is None:
        raise ValueError(f"Unknown embedding profile: {selected}")
    resolved = copy.deepcopy(payload)
    resolved["name"] = selected
    return resolved


def resolve_embedding_settings(
    config: dict[str, Any],
    *,
    model: str | None = None,
    dim: int | None = None,
) -> dict[str, Any]:
    embedding_cfg = copy.deepcopy(config.get("embedding") or {})
    selected_name = (
        str((config.get("profiles") or {}).get("embedding") or "").strip()
        or str(embedding_cfg.get("profile") or "").strip()
        or None
    )

    resolved: dict[str, Any] = {}
    if selected_name:
        resolved = resolve_embedding_profile(config, name=selected_name)
        legacy_overlay = _deep_diff(embedding_cfg, DEFAULT_CONFIG.get("embedding") or {})
        legacy_overlay.pop("model", None)
        legacy_overlay.pop("dim", None)
        legacy_overlay.pop("profile", None)
        _deep_update(resolved, legacy_overlay)
    else:
        _deep_update(resolved, embedding_cfg)

    if model is not None:
        resolved["model"] = str(model)
    if dim is not None:
        resolved["dim"] = int(dim)

    runtime = copy.deepcopy(resolved.get("runtime") or {})
    runtime["device"] = str(runtime.get("device") or "cpu").strip() or "cpu"
    runtime["batch_size"] = max(1, int(runtime.get("batch_size") or 32))

    huggingface = copy.deepcopy(runtime.get("huggingface") or {})
    env_endpoint = str(os.environ.get("HF_ENDPOINT") or os.environ.get("HUGGINGFACE_HUB_ENDPOINT") or "").strip()
    env_cache_dir = str(os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE") or "").strip()
    env_token = str(os.environ.get("HF_TOKEN") or "").strip()

    huggingface["endpoint"] = str(huggingface.get("endpoint") or env_endpoint or "").strip()
    huggingface["cache_dir"] = str(huggingface.get("cache_dir") or env_cache_dir or "").strip()
    huggingface["local_files_only"] = bool(huggingface.get("local_files_only", False))
    huggingface["token"] = str(huggingface.get("token") or env_token or "").strip()

    runtime["huggingface"] = huggingface
    resolved["runtime"] = runtime
    resolved["allow_silent_fallback"] = bool(resolved.get("allow_silent_fallback", False))
    return resolved


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


def _deep_diff(current: dict[str, Any], default: dict[str, Any]) -> dict[str, Any]:
    diff: dict[str, Any] = {}
    for key, value in current.items():
        default_value = default.get(key)
        if isinstance(value, dict) and isinstance(default_value, dict):
            nested = _deep_diff(value, default_value)
            if nested:
                diff[key] = nested
            continue
        if value != default_value:
            diff[key] = copy.deepcopy(value)
    return diff
