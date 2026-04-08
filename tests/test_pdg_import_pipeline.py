from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from hep_rag_v2 import cli, db, paths
from hep_rag_v2.config import default_config, save_config
from hep_rag_v2.pipeline import import_pdg
from hep_rag_v2.providers.pdg import resolve_pdg_reference


@contextlib.contextmanager
def _patch_workspace(tmp: Path):
    original = paths.workspace_root()
    try:
        paths.set_workspace_root(tmp)
        yield
    finally:
        paths.set_workspace_root(original)


class PdgImportPipelineTests(unittest.TestCase):
    def test_resolve_pdg_reference_exposes_stable_pdf_metadata(self) -> None:
        ref = resolve_pdg_reference(edition="2024")

        self.assertEqual(ref["canonical_source"], "pdg")
        self.assertEqual(ref["edition"], "2024")
        self.assertEqual(ref["slug"], "review-of-particle-physics")
        self.assertTrue(ref["canonical_id"].startswith("pdg-2024-"))
        self.assertEqual(ref["pdf_name"], "db2024.pdf")
        self.assertIn("/2024/download/db2024.pdf", ref["pdf_url"])
        self.assertTrue(ref["pdf_url"].endswith(".pdf"))
        self.assertIn("pdg.lbl.gov", ref["pdf_url"])

    def test_import_pdg_with_local_pdf_registers_archival_ingest_stub(self) -> None:
        original_root = paths.workspace_root()
        try:
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                pdf_source = tmp / "pdg-2024.pdf"
                pdf_source.write_bytes(b"%PDF-1.4\n% fake pdg pdf\n")

                with _patch_workspace(tmp):
                    config = default_config(workspace_root=tmp)
                    payload = import_pdg(
                        config,
                        edition="2024",
                        pdf_path=pdf_source,
                        collection_name="pdg",
                    )

                    self.assertEqual(payload["reference"]["canonical_id"], "pdg-2024-review-of-particle-physics")
                    self.assertEqual(payload["pdf"]["state"], "copied")
                    self.assertTrue(Path(payload["pdf"]["path"]).exists())
                    self.assertEqual(payload["document"]["parse_status"], "pdf_ready")
                    self.assertTrue(payload["document"]["parsed_dir"].endswith("pdg/pdg-2024-review-of-particle-physics"))
                    self.assertTrue(payload["document"]["manifest_path"].endswith("manifest.json"))

                    with db.connect() as conn:
                        work = conn.execute(
                            "SELECT canonical_source, canonical_id, title, primary_pdf_url FROM works"
                        ).fetchone()
                        self.assertIsNotNone(work)
                        self.assertEqual(work["canonical_source"], "pdg")
                        self.assertEqual(work["canonical_id"], "pdg-2024-review-of-particle-physics")
                        self.assertEqual(work["primary_pdf_url"], payload["reference"]["pdf_url"])

                        doc = conn.execute(
                            "SELECT parser_name, parse_status FROM documents WHERE work_id = 1"
                        ).fetchone()
                        self.assertEqual(doc["parser_name"], "pdg")
                        self.assertEqual(doc["parse_status"], "pdf_ready")
        finally:
            paths.set_workspace_root(original_root)

    def test_cli_import_pdg_prints_json_payload(self) -> None:
        original_root = paths.workspace_root()
        try:
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                workspace_root = tmp / "workspace"
                config_path = tmp / "hep-rag.yaml"
                pdf_source = tmp / "pdg-cli.pdf"
                pdf_source.write_bytes(b"%PDF-1.4\n% cli pdg pdf\n")
                save_config(default_config(workspace_root=workspace_root), config_path, overwrite=True)

                parser = cli.build_parser()
                args = parser.parse_args(
                    [
                        "import-pdg",
                        "--config",
                        str(config_path),
                        "--collection",
                        "pdg",
                        "--edition",
                        "2024",
                        "--pdf",
                        str(pdf_source),
                    ]
                )

                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    args.func(args)
                payload = json.loads(out.getvalue())
                self.assertEqual(payload["reference"]["edition"], "2024")
                self.assertEqual(payload["pdf"]["state"], "copied")
                self.assertEqual(payload["document"]["parse_status"], "pdf_ready")
        finally:
            paths.set_workspace_root(original_root)


    def test_pdg_source_reimport_is_idempotent(self) -> None:
        original_root = paths.workspace_root()
        try:
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                pdf_source = tmp / "pdg-2024.pdf"
                pdf_source.write_bytes(b"%PDF-1.4\n% fake pdg pdf\n")

                with _patch_workspace(tmp):
                    config = default_config(workspace_root=tmp)
                    payload1 = import_pdg(config, edition="2024", pdf_path=pdf_source, collection_name="pdg")
                    payload2 = import_pdg(config, edition="2024", pdf_path=pdf_source, collection_name="pdg")

                    self.assertEqual(payload1["work"]["work_id"], payload2["work"]["work_id"])
                    with db.connect() as conn:
                        work_count = conn.execute("SELECT COUNT(*) as c FROM works").fetchone()["c"]
                        self.assertEqual(work_count, 1)
        finally:
            paths.set_workspace_root(original_root)

    def test_pdg_section_import_transaction_rollback_on_failure(self) -> None:
        from hep_rag_v2.pdg import ensure_pdg_schema, import_pdg_source
        from unittest import mock

        original_root = paths.workspace_root()
        try:
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                with _patch_workspace(tmp):
                    db.ensure_db()
                    with db.connect() as conn:
                        ensure_pdg_schema(conn)
                        conn.execute(
                            "INSERT INTO pdg_sources (source_id, title, block_count, capsule_count) VALUES (?, ?, 0, 0)",
                            ("test-src", "Test Source"),
                        )

                    with mock.patch(
                        "hep_rag_v2.pdg.import_mineru_source",
                        side_effect=RuntimeError("simulated parse failure"),
                    ):
                        with self.assertRaises(RuntimeError):
                            with db.connect() as conn:
                                import_pdg_source(
                                    conn,
                                    source_path="/nonexistent",
                                    source_id="test-src",
                                    title="Test Source",
                                )

                    with db.connect() as conn:
                        row = conn.execute(
                            "SELECT source_id FROM pdg_sources WHERE source_id = ?",
                            ("test-src",),
                        ).fetchone()
                        self.assertIsNotNone(row, "Original source row should survive failed re-import")
        finally:
            paths.set_workspace_root(original_root)


if __name__ == "__main__":
    unittest.main()
