from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hep_rag_v2.cli import build_parser  # noqa: E402
from hep_rag_v2.config import default_config  # noqa: E402
from hep_rag_v2.smoke import DEFAULT_SMOKE_CORPORA, load_smoke_queries, run_metadata_smoke  # noqa: E402


class SmokeMetadataTests(unittest.TestCase):
    def test_parser_declares_smoke_metadata_command_with_expected_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["smoke-metadata"])

        self.assertEqual(args.corpus, "cms_atlas_2k")
        self.assertEqual(args.limit, 2000)
        self.assertEqual(args.download_limit, 0)
        self.assertEqual(args.parse_limit, 0)
        self.assertTrue(args.build_search)
        self.assertTrue(args.build_vectors)
        self.assertFalse(args.build_graph)

    def test_load_smoke_queries_supports_mapping_and_string_entries(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "queries.yaml"
            path.write_text(
                """
queries:
  - query: CMS VBS SSWW latest result
    target: works
    limit: 8
  - ATLAS Higgs combination
""".strip()
                + "\n",
                encoding="utf-8",
            )

            queries = load_smoke_queries(path)

        self.assertEqual(len(queries), 2)
        self.assertEqual(queries[0]["query"], "CMS VBS SSWW latest result")
        self.assertEqual(queries[0]["target"], "works")
        self.assertEqual(queries[0]["limit"], 8)
        self.assertEqual(queries[1]["query"], "ATLAS Higgs combination")
        self.assertNotIn("target", queries[1])

    def test_run_metadata_smoke_runs_ingest_sync_query_and_exports_report(self) -> None:
        config = default_config()
        corpus = "cms_atlas_2k"
        preset = DEFAULT_SMOKE_CORPORA[corpus]
        validation_count = len(preset["validation_queries"])

        retrieval_payload = {
            "collection": corpus,
            "search_scope": {"key": corpus, "label": corpus, "collection_name": corpus},
            "requested_target": "works",
            "routing": {"target": "works", "graph_expand": 0, "reasons": ["manual_target"]},
            "model": "hash-idf-v1",
            "community_summaries": [],
            "ontology_summaries": [],
            "works": [{"work_id": 1, "raw_title": "Test work", "rank": 1}],
            "evidence_chunks": [],
        }

        with tempfile.TemporaryDirectory() as td:
            export_path = Path(td) / "smoke-report.json"
            conn = mock.MagicMock()
            context = mock.MagicMock()
            context.__enter__.return_value = conn
            context.__exit__.return_value = False

            with (
                mock.patch("hep_rag_v2.smoke.ensure_db"),
                mock.patch("hep_rag_v2.smoke.connect", return_value=context),
                mock.patch(
                    "hep_rag_v2.smoke.ingest_online",
                    return_value={"metadata": {"created": 2000, "updated": 0}},
                ) as ingest,
                mock.patch(
                    "hep_rag_v2.smoke.rebuild_search_indices",
                    return_value={"works": 2000, "chunks": 0},
                ) as rebuild_search,
                mock.patch(
                    "hep_rag_v2.smoke.search_index_counts",
                    return_value={"work_search": 2000, "chunk_search": 0},
                ),
                mock.patch("hep_rag_v2.smoke.configure_embedding_runtime") as configure_runtime,
                mock.patch(
                    "hep_rag_v2.smoke.rebuild_vector_indices",
                    return_value={"works": 2000, "chunks": 0},
                ) as rebuild_vectors,
                mock.patch(
                    "hep_rag_v2.smoke.vector_index_counts",
                    return_value={"work_embeddings": 2000, "chunk_embeddings": 0},
                ),
                mock.patch(
                    "hep_rag_v2.smoke.retrieve",
                    side_effect=[dict(retrieval_payload) for _ in range(validation_count)],
                ) as retrieve_mock,
                mock.patch(
                    "hep_rag_v2.smoke.workspace_status_payload",
                    return_value={"snapshot": {"works": 2000}, "collections": [], "search_scopes": []},
                ),
            ):
                report = run_metadata_smoke(
                    config,
                    corpus=corpus,
                    limit=2000,
                    export_report=export_path,
                )

            self.assertTrue(export_path.exists())
            exported = json.loads(export_path.read_text(encoding="utf-8"))

        ingest.assert_called_once()
        self.assertEqual(ingest.call_args.kwargs["query"], preset["ingest_query"])
        self.assertEqual(ingest.call_args.kwargs["collection_name"], corpus)
        rebuild_search.assert_called_once()
        rebuild_vectors.assert_called_once()
        configure_runtime.assert_called_once()
        self.assertEqual(retrieve_mock.call_count, validation_count)
        self.assertEqual(report["corpus"], corpus)
        self.assertEqual(report["collection"], corpus)
        self.assertEqual(report["steps"]["ingest"]["summary"]["metadata"]["created"], 2000)
        self.assertEqual(len(report["steps"]["queries"]["items"]), validation_count)
        self.assertIn("p50", report["steps"]["queries"]["latency_ms"])
        self.assertEqual(exported["workspace_status"]["snapshot"]["works"], 2000)


if __name__ == "__main__":
    unittest.main()
