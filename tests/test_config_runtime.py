from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hep_rag_v2 import paths
from hep_rag_v2.cli import build_parser
from hep_rag_v2.config import apply_runtime_config, default_config, runtime_collection_config, save_config
from hep_rag_v2.providers.local_transformers import LocalTransformersClient
from hep_rag_v2.providers.inspire import build_search_query, list_pdf_candidates


class ConfigRuntimeTests(unittest.TestCase):
    def test_apply_runtime_config_can_create_default_and_switch_workspace(self) -> None:
        original_root = paths.workspace_root()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                config_path = tmp_path / "hep-rag.yaml"
                workspace_root = tmp_path / "custom-workspace"

                loaded_path, config = apply_runtime_config(
                    config_path=config_path,
                    workspace_root=workspace_root,
                    create_default=True,
                )

                self.assertEqual(loaded_path, config_path.resolve())
                self.assertTrue(config_path.exists())
                self.assertEqual(paths.workspace_root(), workspace_root.resolve())
                self.assertEqual(Path(config["workspace"]["root"]), workspace_root.resolve())
        finally:
            paths.set_workspace_root(original_root)

    def test_runtime_collection_config_respects_override_name(self) -> None:
        config = {
            "collection": {
                "name": "default_name",
                "label": "Default Label",
                "notes": "notes",
            },
            "online": {
                "fields": ["titles", "abstracts"],
            },
        }
        payload = runtime_collection_config(config, name="override_name")
        self.assertEqual(payload["name"], "override_name")
        self.assertEqual(payload["label"], "Default Label")
        self.assertEqual(payload["fields"], ["titles", "abstracts"])

    def test_default_config_disables_published_filter_and_enables_query_rewrite(self) -> None:
        config = default_config()
        self.assertFalse(config["online"]["published_only"])
        self.assertTrue(config["query_rewrite"]["enabled"])
        self.assertEqual(config["query_rewrite"]["max_queries"], 4)
        self.assertIn("references", config["online"]["fields"])
        self.assertEqual(config["api"]["job_max_workers"], 2)
        self.assertEqual(config["api"]["job_max_events"], 1000)
        self.assertTrue(config["api"]["enable_ui"])

    def test_build_search_query_only_appends_published_filter_when_requested(self) -> None:
        self.assertEqual(
            build_search_query('collaboration:"CMS"', published_only=False),
            'collaboration:"CMS"',
        )
        self.assertEqual(
            build_search_query('collaboration:"CMS"', published_only=True),
            'collaboration:"CMS" and collection:Published',
        )
        self.assertEqual(
            build_search_query('collaboration:"CMS" and collection:Published', published_only=True),
            'collaboration:"CMS" and collection:Published',
        )

    def test_list_pdf_candidates_prefers_direct_pdf_urls(self) -> None:
        hit = {
            "metadata": {
                "documents": [{"url": "https://example.org/paper.pdf"}],
                "files": [{"url": "https://example.org/backup.pdf"}],
                "arxiv_eprints": [{"value": "2501.01234"}],
                "dois": [{"value": "10.1000/example"}],
            }
        }
        candidates = list_pdf_candidates(hit)
        self.assertEqual(candidates[0]["url"], "https://example.org/paper.pdf")
        self.assertEqual(candidates[1]["url"], "https://example.org/backup.pdf")
        self.assertEqual(candidates[2]["url"], "https://arxiv.org/pdf/2501.01234.pdf")
        self.assertEqual(candidates[3]["url"], "https://doi.org/10.1000/example")

    def test_list_pdf_candidates_can_fallback_from_doi_to_arxiv(self) -> None:
        hit = {
            "metadata": {
                "dois": [{"value": "10.1000/example"}],
            }
        }
        with mock.patch("hep_rag_v2.providers.inspire.doi_to_arxiv", return_value="1711.04330v2") as patched:
            candidates = list_pdf_candidates(hit, resolve_arxiv_from_doi=True)
        patched.assert_called_once_with("10.1000/example", timeout=10, retries=3)
        self.assertEqual(candidates[0]["url"], "https://arxiv.org/pdf/1711.04330v2.pdf")
        self.assertEqual(candidates[1]["url"], "https://doi.org/10.1000/example")

    def test_local_transformers_client_requires_model_path(self) -> None:
        with self.assertRaises(ValueError):
            LocalTransformersClient(model_name_or_path="")


class CliRuntimeTests(unittest.TestCase):
    def test_status_honors_explicit_config_path(self) -> None:
        original = paths.workspace_root()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                workspace_root = tmp_path / "runtime"
                config_path = tmp_path / "hep-rag.yaml"
                save_config(default_config(workspace_root=workspace_root), config_path)

                parser = build_parser()
                args = parser.parse_args(["status", "--config", str(config_path)])
                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    args.func(args)

                payload = json.loads(buffer.getvalue())
                self.assertEqual(payload["snapshot"]["collections"], 0)
                self.assertTrue((workspace_root / "db" / "hep_rag_v2.db").exists())
                self.assertEqual(paths.workspace_root(), workspace_root.resolve())
        finally:
            paths.set_workspace_root(original)

    def test_collections_auto_loads_cwd_config(self) -> None:
        original_root = paths.workspace_root()
        original_cwd = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                workspace_root = tmp_path / "runtime"
                config_path = tmp_path / "hep-rag.yaml"
                save_config(default_config(workspace_root=workspace_root), config_path)
                collections_dir = workspace_root / "collections"
                collections_dir.mkdir(parents=True, exist_ok=True)
                (collections_dir / "demo.json").write_text('{"name": "demo"}\n', encoding="utf-8")

                os.chdir(tmp_path)
                parser = build_parser()
                args = parser.parse_args(["collections"])
                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    args.func(args)

                self.assertEqual(buffer.getvalue().strip(), "demo")
                self.assertEqual(paths.workspace_root(), workspace_root.resolve())
        finally:
            os.chdir(original_cwd)
            paths.set_workspace_root(original_root)

    def test_init_accepts_workspace_override_without_config(self) -> None:
        original = paths.workspace_root()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                workspace_root = tmp_path / "custom-workspace"

                parser = build_parser()
                args = parser.parse_args(["init", "--workspace", str(workspace_root)])
                args.func(args)

                self.assertTrue((workspace_root / "db" / "hep_rag_v2.db").exists())
                self.assertEqual(paths.workspace_root(), workspace_root.resolve())
        finally:
            paths.set_workspace_root(original)

    def test_init_config_writes_collection_json(self) -> None:
        original = paths.workspace_root()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                workspace_root = tmp_path / "runtime"
                config_path = tmp_path / "hep-rag.yaml"

                parser = build_parser()
                args = parser.parse_args(
                    [
                        "init-config",
                        "--config",
                        str(config_path),
                        "--workspace",
                        str(workspace_root),
                    ]
                )
                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    args.func(args)

                payload = json.loads(buffer.getvalue())
                collection_config = workspace_root / "collections" / "default.json"
                self.assertTrue(collection_config.exists())
                self.assertEqual(payload["collection"]["config_path"], str(collection_config))
        finally:
            paths.set_workspace_root(original)


if __name__ == "__main__":
    unittest.main()
