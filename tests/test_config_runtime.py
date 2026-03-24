from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hep_rag_v2 import paths
from hep_rag_v2.config import apply_runtime_config, runtime_collection_config
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

    def test_build_search_query_appends_published_filter_once(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
