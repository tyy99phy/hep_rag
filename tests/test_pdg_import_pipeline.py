from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from hep_rag_v2 import cli, db, paths
from hep_rag_v2.config import default_config, save_config
from hep_rag_v2.pipeline import import_pdg
from hep_rag_v2.providers.pdg import resolve_pdg_reference, resolve_pdg_references


@contextlib.contextmanager
def _patch_workspace(tmp: Path):
    original = paths.workspace_root()
    try:
        paths.set_workspace_root(tmp)
        yield
    finally:
        paths.set_workspace_root(original)


class PdgImportPipelineTests(unittest.TestCase):
    def test_resolve_pdg_reference_defaults_to_website_corpus_metadata(self) -> None:
        ref = resolve_pdg_reference(edition="2024")

        self.assertEqual(ref["canonical_source"], "pdg")
        self.assertEqual(ref["edition"], "2024")
        self.assertEqual(ref["artifact_kind"], "website")
        self.assertEqual(ref["slug"], "website")
        self.assertEqual(ref["file_name"], "rpp-2024.zip")
        self.assertIn("/2024/download/rpp-2024.zip", ref["download_url"])

    def test_resolve_pdg_references_full_profile_expands_official_assets(self) -> None:
        refs = resolve_pdg_references(edition="2024", artifact="full", sqlite_variant="all")

        self.assertEqual([item["artifact_kind"] for item in refs], ["website", "sqlite"])
        self.assertTrue(any(str(item.get("file_name")).endswith(".sqlite") for item in refs))
        self.assertTrue(any(str(item.get("file_name")).endswith(".zip") for item in refs))

    def test_import_pdg_with_local_website_bundle_materializes_sections(self) -> None:
        original_root = paths.workspace_root()
        try:
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                source_zip = _make_pdg_website_zip(tmp)

                with _patch_workspace(tmp):
                    config = default_config(workspace_root=tmp)
                    payload = import_pdg(
                        config,
                        edition="2024",
                        artifact="website",
                        source_path=source_zip,
                        collection_name="pdg",
                    )

                    self.assertEqual(payload["artifact"], "website")
                    self.assertIsNotNone(payload["website_import"])
                    self.assertIsNotNone(payload["registered_embedded_pdfs"])
                    self.assertEqual(len(payload["staged_artifacts"]), 1)
                    self.assertEqual(payload["staged_artifacts"][0]["artifact"]["state"], "copied")
                    self.assertGreater(payload["website_import"]["capsule_count"], 0)
                    self.assertEqual(payload["registered_embedded_pdfs"]["registered"], 2)
                    self.assertEqual(payload["cleanup"]["removed_artifacts"], 0)
                    self.assertEqual(payload["cleanup"]["removed_works"], 0)

                    with db.connect() as conn:
                        source_row = conn.execute(
                            "SELECT source_id, block_count, capsule_count FROM pdg_sources"
                        ).fetchone()
                        self.assertIsNotNone(source_row)
                        self.assertEqual(source_row["source_id"], "pdg-2024-website")
                        self.assertGreater(int(source_row["block_count"]), 0)
                        self.assertGreater(int(source_row["capsule_count"]), 0)

                        artifact_row = conn.execute(
                            "SELECT artifact_kind, local_path FROM pdg_artifacts WHERE source_id = ?",
                            ("pdg-2024-website",),
                        ).fetchone()
                        self.assertEqual(artifact_row["artifact_kind"], "website")
                        self.assertTrue(str(artifact_row["local_path"]).endswith("rpp-2024.zip"))
                        work_count = conn.execute("SELECT COUNT(*) as c FROM works").fetchone()["c"]
                        self.assertEqual(work_count, 2)
                        work_rows = conn.execute(
                            "SELECT canonical_id FROM works ORDER BY canonical_id"
                        ).fetchall()
                        canonical_ids = [str(row["canonical_id"]) for row in work_rows]
                        self.assertTrue(all(not item.endswith(".pdf") for item in canonical_ids))
                        sample_paths = [
                            str(item["path"])
                            for item in payload["registered_embedded_pdfs"]["sample"]
                        ]
                        self.assertTrue(all(path.endswith(".pdf") and not path.endswith(".pdf.pdf") for path in sample_paths))

                        section_titles = [
                            str(row["title"])
                            for row in conn.execute(
                                "SELECT title FROM pdg_sections WHERE source_id = ? ORDER BY order_index",
                                ("pdg-2024-website",),
                            ).fetchall()
                        ]
                        self.assertIn("Higgs Boson and Properties", section_titles)
        finally:
            paths.set_workspace_root(original_root)

    def test_cli_import_pdg_prints_json_payload(self) -> None:
        original_root = paths.workspace_root()
        try:
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                workspace_root = tmp / "workspace"
                config_path = tmp / "hep-rag.yaml"
                source_zip = _make_pdg_website_zip(tmp)
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
                        "--artifact",
                        "website",
                        "--source",
                        str(source_zip),
                    ]
                )

                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    args.func(args)
                payload = json.loads(out.getvalue())
                self.assertEqual(payload["artifact"], "website")
                self.assertIsNotNone(payload["website_import"])
                self.assertEqual(payload["registered_embedded_pdfs"]["registered"], 2)
                self.assertEqual(payload["staged_artifacts"][0]["artifact"]["state"], "copied")
        finally:
            paths.set_workspace_root(original_root)

    def test_pdg_website_import_is_idempotent(self) -> None:
        original_root = paths.workspace_root()
        try:
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                source_zip = _make_pdg_website_zip(tmp)

                with _patch_workspace(tmp):
                    config = default_config(workspace_root=tmp)
                    payload1 = import_pdg(config, edition="2024", artifact="website", source_path=source_zip, collection_name="pdg")
                    payload2 = import_pdg(config, edition="2024", artifact="website", source_path=source_zip, collection_name="pdg")

                    self.assertEqual(payload1["website_import"]["source_id"], payload2["website_import"]["source_id"])
                    with db.connect() as conn:
                        source_count = conn.execute("SELECT COUNT(*) as c FROM pdg_sources").fetchone()["c"]
                        artifact_count = conn.execute("SELECT COUNT(*) as c FROM pdg_artifacts WHERE source_id = 'pdg-2024-website'").fetchone()["c"]
                        work_count = conn.execute("SELECT COUNT(*) as c FROM works").fetchone()["c"]
                        self.assertEqual(source_count, 1)
                        self.assertEqual(artifact_count, 1)
                        self.assertEqual(work_count, 2)
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

    def test_import_pdg_prunes_deprecated_book_pdf_records(self) -> None:
        original_root = paths.workspace_root()
        try:
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                source_zip = _make_pdg_website_zip(tmp)

                with _patch_workspace(tmp):
                    config = default_config(workspace_root=tmp)
                    db.ensure_db()
                    from hep_rag_v2.metadata import upsert_collection
                    from hep_rag_v2.pdg import ensure_pdg_schema

                    with db.connect() as conn:
                        ensure_pdg_schema(conn)
                        collection_id = upsert_collection(conn, {"name": "pdg", "label": "PDG"})
                        conn.execute(
                            """
                            INSERT INTO works (canonical_source, canonical_id, title, year)
                            VALUES ('pdg', 'pdg-2024-book-pdf', 'Legacy PDG Book', 2024)
                            """
                        )
                        conn.execute(
                            "INSERT INTO work_ids (id_type, id_value, work_id, is_primary) VALUES ('pdg', 'pdg-2024-book-pdf', 1, 1)"
                        )
                        conn.execute(
                            "INSERT INTO collection_works (collection_id, work_id) VALUES (?, 1)",
                            (collection_id,),
                        )
                        parsed_dir = paths.PARSED_DIR / "pdg" / "pdg-2024-book-pdf"
                        parsed_dir.mkdir(parents=True, exist_ok=True)
                        (parsed_dir / "manifest.json").write_text("{}", encoding="utf-8")
                        pdf_path = paths.PDF_DIR / "pdg" / "pdg-2024-book-pdf.pdf"
                        pdf_path.parent.mkdir(parents=True, exist_ok=True)
                        pdf_path.write_bytes(b"%PDF-1.4\n")
                        raw_path = paths.RAW_DIR / "pdg" / "book_pdf" / "PhysRevD.110.030001.pdf"
                        raw_path.parent.mkdir(parents=True, exist_ok=True)
                        raw_path.write_bytes(b"%PDF-1.4\n")
                        conn.execute(
                            """
                            INSERT INTO documents (
                              work_id, parser_name, parser_version, parse_status, parsed_dir, manifest_path
                            ) VALUES (1, 'pdg_book_pdf', '2024', 'pdf_ready', ?, ?)
                            """,
                            (str(parsed_dir), str(parsed_dir / "manifest.json")),
                        )
                        conn.execute(
                            """
                            INSERT INTO pdg_artifacts (
                              source_id, artifact_kind, edition, title, local_path, file_name
                            ) VALUES ('pdg-2024-book-pdf', 'book_pdf', '2024', 'Legacy PDG Book', ?, 'PhysRevD.110.030001.pdf')
                            """,
                            (str(raw_path),),
                        )

                    payload = import_pdg(
                        config,
                        edition="2024",
                        artifact="website",
                        source_path=source_zip,
                        collection_name="pdg",
                    )

                    self.assertEqual(payload["cleanup"]["removed_artifacts"], 1)
                    self.assertEqual(payload["cleanup"]["removed_works"], 1)

                    with db.connect() as conn:
                        artifact_row = conn.execute(
                            "SELECT 1 FROM pdg_artifacts WHERE artifact_kind = 'book_pdf'"
                        ).fetchone()
                        self.assertIsNone(artifact_row)
                        work_row = conn.execute(
                            "SELECT 1 FROM works WHERE canonical_id = 'pdg-2024-book-pdf'"
                        ).fetchone()
                        self.assertIsNone(work_row)
                    self.assertFalse(raw_path.exists())
                    self.assertFalse(pdf_path.exists())
                    self.assertFalse(parsed_dir.exists())
        finally:
            paths.set_workspace_root(original_root)


def _make_pdg_website_zip(tmp: Path) -> Path:
    zip_path = tmp / "rpp-2024.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(
            "rpp-2024/index.html",
            """
            <html><head><title>Particle Data Group</title></head>
            <body>
              <div id="details">
                <h1>Review of Particle Physics 2024</h1>
                <p>PDG root overview for the 2024 edition.</p>
              </div>
            </body></html>
            """,
        )
        zf.writestr(
            "rpp-2024/reviews/higgs.html",
            """
            <html><head><title>Higgs Boson and Properties</title></head>
            <body>
              <div id="details">
                <h1>Higgs Boson and Properties</h1>
                <p>The Higgs review summarizes couplings and decay modes.</p>
                <h2>Standard Model Higgs</h2>
                <p>Mass and width values are discussed in detail.</p>
              </div>
            </body></html>
            """,
        )
        zf.writestr("rpp-2024/reviews/rpp2024-rev-higgs-boson.pdf", b"%PDF-1.4\n% fake review pdf\n")
        zf.writestr(
            "rpp-2024/tables/summary_tables.html",
            """
            <html><head><title>Summary Tables</title></head>
            <body>
              <div id="details">
                <h1>Summary Tables</h1>
                <p>Summary tables provide compact numerical values.</p>
              </div>
            </body></html>
            """,
        )
        zf.writestr(
            "rpp-2024/listings/z_boson.html",
            """
            <html><head><title>Z Boson Listing</title></head>
            <body>
              <div id="details">
                <h1>Z Boson Listing</h1>
                <p>The Z boson listing contains identifiers and decay information.</p>
              </div>
            </body></html>
            """,
        )
        zf.writestr("rpp-2024/listings/rpp2024-list-z-boson.pdf", b"%PDF-1.4\n% fake listing pdf\n")
    return zip_path


if __name__ == "__main__":
    unittest.main()
