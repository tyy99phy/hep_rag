from __future__ import annotations

import argparse
import json
from pathlib import Path

from hep_rag_v2 import paths
from hep_rag_v2.config import apply_runtime_config, default_config, resolve_config_path, save_config
from hep_rag_v2.db import ensure_db
from hep_rag_v2.pipeline import initialize_workspace
from hep_rag_v2.service.workspace import workspace_status_payload


def _maybe_apply_workspace_config(args: argparse.Namespace) -> None:
    config_path = getattr(args, "config", None)
    workspace_root = getattr(args, "workspace", None)

    if config_path is not None:
        apply_runtime_config(config_path=config_path, workspace_root=workspace_root)
        return

    resolved_default = resolve_config_path(None)
    if resolved_default.exists():
        apply_runtime_config(config_path=resolved_default, workspace_root=workspace_root)
        return

    if workspace_root:
        paths.set_workspace_root(Path(workspace_root).expanduser().resolve())


def cmd_init(args: argparse.Namespace) -> None:
    _maybe_apply_workspace_config(args)
    ensure_db()
    print(f"Initialized DB at {paths.DB_PATH}")


def cmd_collections(args: argparse.Namespace) -> None:
    _maybe_apply_workspace_config(args)
    for path in sorted(paths.COLLECTIONS_DIR.glob("*.json")):
        print(path.stem)


def cmd_status(args: argparse.Namespace) -> None:
    _maybe_apply_workspace_config(args)
    payload = workspace_status_payload()
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def cmd_init_config(args: argparse.Namespace) -> None:
    try:
        config_path = resolve_config_path(args.config)
        workspace_root = Path(args.workspace).expanduser().resolve() if args.workspace else (config_path.parent / "workspace")
        config = default_config(workspace_root=workspace_root)
        if args.collection:
            config["collection"]["name"] = args.collection
            config["collection"]["label"] = args.collection
        save_config(config, config_path, overwrite=args.force)
        _, loaded = apply_runtime_config(config_path=config_path, workspace_root=workspace_root)
        summary = initialize_workspace(loaded, collection_name=args.collection)
        summary["config_path"] = str(config_path)
    except Exception as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(summary, ensure_ascii=False, indent=2))
