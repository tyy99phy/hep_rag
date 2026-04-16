from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

from pypdf import PdfReader, PdfWriter

from hep_rag_v2 import db, paths
from hep_rag_v2.config import default_config, runtime_collection_config
from hep_rag_v2.fulltext import load_content_list, load_manifest
from hep_rag_v2.metadata import upsert_collection
from hep_rag_v2.pipeline import _select_reparse_candidates
from hep_rag_v2.parse import _parse_with_mineru


class IncrementalReparseTests(unittest.TestCase):
    def test_select_reparse_candidates_prefers_incomplete_local_pdfs(self) -> None:
        original_root = paths.workspace_root()
        try:
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                paths.set_workspace_root(tmp)
                db.ensure_db()

                config = default_config(workspace_root=tmp)
                collection = runtime_collection_config(config, name="default")
                with db.connect() as conn:
                    collection_id = upsert_collection(conn, collection)
                    conn.execute(
                        """
                        INSERT INTO works (canonical_source, canonical_id, title)
                        VALUES ('inspire', '1001', 'Needs parse'), ('inspire', '1002', 'Already materialized')
                        """
                    )
                    conn.execute(
                        "INSERT INTO collection_works (collection_id, work_id) VALUES (?, 1), (?, 2)",
                        (collection_id, collection_id),
                    )
                    pdf_dir = paths.PDF_DIR / "default"
                    pdf_dir.mkdir(parents=True, exist_ok=True)
                    (pdf_dir / "1001.pdf").write_bytes(b"%PDF-1.4\n")
                    (pdf_dir / "1002.pdf").write_bytes(b"%PDF-1.4\n")
                    parsed_dir = paths.PARSED_DIR / "default" / "1002"
                    parsed_dir.mkdir(parents=True, exist_ok=True)
                    manifest = parsed_dir / "manifest.json"
                    manifest.write_text("{}", encoding="utf-8")
                    conn.execute(
                        """
                        INSERT INTO documents (
                          work_id, parser_name, parser_version, parse_status, parsed_dir, manifest_path
                        ) VALUES (?, 'mineru', 'v2-contract', 'failed', ?, NULL)
                        """,
                        (1, str(paths.PARSED_DIR / "default" / "1001")),
                    )
                    conn.execute(
                        """
                        INSERT INTO documents (
                          work_id, parser_name, parser_version, parse_status, parsed_dir, manifest_path
                        ) VALUES (?, 'mineru', 'v2-contract', 'materialized', ?, ?)
                        """,
                        (2, str(parsed_dir), str(manifest)),
                    )

                    candidates = _select_reparse_candidates(
                        conn,
                        collection_name="default",
                        limit=None,
                        work_ids=None,
                        parser_name=None,
                        replace_existing=False,
                    )

                self.assertEqual(len(candidates), 1)
                self.assertEqual(candidates[0]["work_id"], 1)
                self.assertEqual(candidates[0]["parse_status"], "failed")
        finally:
            paths.set_workspace_root(original_root)

    def test_select_reparse_candidates_can_filter_by_parser_name(self) -> None:
        original_root = paths.workspace_root()
        try:
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                paths.set_workspace_root(tmp)
                db.ensure_db()

                config = default_config(workspace_root=tmp)
                collection = runtime_collection_config(config, name="pdg")
                with db.connect() as conn:
                    collection_id = upsert_collection(conn, collection)
                    conn.execute(
                        """
                        INSERT INTO works (canonical_source, canonical_id, title)
                        VALUES ('pdg', 'pdg-2024-book-pdf', 'PDG Book'),
                               ('pdg', 'pdg-2024-listings-z-boson', 'Z Boson Listing')
                        """
                    )
                    conn.execute(
                        "INSERT INTO collection_works (collection_id, work_id) VALUES (?, 1), (?, 2)",
                        (collection_id, collection_id),
                    )
                    pdf_dir = paths.PDF_DIR / "pdg"
                    pdf_dir.mkdir(parents=True, exist_ok=True)
                    (pdf_dir / "pdg-2024-book-pdf.pdf").write_bytes(b"%PDF-1.4\n")
                    (pdf_dir / "pdg-2024-listings-z-boson.pdf").write_bytes(b"%PDF-1.4\n")
                    conn.execute(
                        """
                        INSERT INTO documents (
                          work_id, parser_name, parser_version, parse_status, parsed_dir, manifest_path
                        ) VALUES (?, 'pdg_book_pdf', '2024', 'pdf_ready', ?, NULL)
                        """,
                        (1, str(paths.PARSED_DIR / "pdg" / "pdg-2024-book-pdf")),
                    )
                    conn.execute(
                        """
                        INSERT INTO documents (
                          work_id, parser_name, parser_version, parse_status, parsed_dir, manifest_path
                        ) VALUES (?, 'pdg_website_pdf', '2024', 'pdf_ready', ?, NULL)
                        """,
                        (2, str(paths.PARSED_DIR / "pdg" / "pdg-2024-listings-z-boson")),
                    )

                    candidates = _select_reparse_candidates(
                        conn,
                        collection_name="pdg",
                        limit=None,
                        work_ids=None,
                        parser_name="pdg_book_pdf",
                        replace_existing=False,
                    )

                self.assertEqual(len(candidates), 1)
                self.assertEqual(candidates[0]["work_id"], 1)
                self.assertEqual(candidates[0]["parser_name"], "pdg_book_pdf")
        finally:
            paths.set_workspace_root(original_root)

    def test_parse_with_mineru_splits_large_pdf_and_merges_parts(self) -> None:
        original_root = paths.workspace_root()
        try:
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                paths.set_workspace_root(tmp)
                db.ensure_db()

                config = default_config(workspace_root=tmp)
                config["mineru"]["max_pages_per_pdf"] = 2
                config["mineru"]["oversize_strategy"] = "split"
                collection = runtime_collection_config(config, name="default")
                pdf_path = tmp / "oversize.pdf"
                _write_pdf(pdf_path, pages=5)
                fake_client = _FakeMinerUClient()

                with db.connect() as conn:
                    collection_id = upsert_collection(conn, collection)
                    conn.execute(
                        """
                        INSERT INTO works (canonical_source, canonical_id, title)
                        VALUES ('inspire', '1001', 'Oversize PDF')
                        """
                    )
                    conn.execute(
                        "INSERT INTO work_ids (id_type, id_value, work_id, is_primary) VALUES ('inspire', '1001', 1, 1)"
                    )
                    conn.execute(
                        "INSERT INTO collection_works (collection_id, work_id) VALUES (?, 1)",
                        (collection_id,),
                    )

                    result = _parse_with_mineru(
                        conn,
                        config=config,
                        client=fake_client,
                        work_id=1,
                        pdf_path=pdf_path,
                        collection_name="default",
                        replace_existing=False,
                    )

                    manifest = load_manifest(Path(result["parsed_dir"]) / "manifest.json")
                    items = load_content_list(Path(str(manifest["content_list_path"])))
                    pages = [int(item["page_idx"]) for item in items if item.get("text", "").startswith("Section")]
                    document = conn.execute(
                        "SELECT parse_status, parse_error FROM documents WHERE work_id = 1"
                    ).fetchone()

                self.assertEqual([entry["pages"] for entry in fake_client.submissions], [2, 2, 1])
                self.assertEqual(pages, [1, 3, 5])
                self.assertEqual(result["split_pdf_pages"], 5)
                self.assertEqual(result["split_page_limit"], 2)
                self.assertEqual(document["parse_status"], "materialized")
                self.assertIsNone(document["parse_error"])
        finally:
            paths.set_workspace_root(original_root)


def _write_pdf(path: Path, *, pages: int) -> None:
    writer = PdfWriter()
    for _ in range(max(1, int(pages))):
        writer.add_blank_page(width=200, height=200)
    with path.open("wb") as handle:
        writer.write(handle)


class _FakeMinerUClient:
    def __init__(self) -> None:
        self.submissions: list[dict[str, int | str]] = []
        self._page_counts: dict[str, int] = {}

    def submit_local_pdf(self, pdf_path: Path, *, data_id: str | None = None, progress=None):
        pages = len(PdfReader(str(pdf_path)).pages)
        batch_id = f"batch-{len(self.submissions) + 1}"
        self.submissions.append({"batch_id": batch_id, "pages": pages, "data_id": data_id or ""})
        self._page_counts[batch_id] = pages
        if progress is not None:
            progress(f"accepted {pdf_path.name}")
        return SimpleNamespace(
            batch_id=batch_id,
            data_id=data_id,
            state="done",
            full_zip_url=f"https://example.com/{batch_id}.zip",
            error_message=None,
        )

    def download_result_zip(self, task, *, output_path: Path, progress=None):
        part_index = int(str(task.batch_id).split("-")[-1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output_path, "w") as zf:
            zf.writestr(
                f"part_{part_index}_content_list.json",
                json.dumps(
                    [
                        {
                            "type": "text",
                            "text_level": 1,
                            "text": f"Section {part_index}",
                            "page_idx": 1,
                        },
                        {
                            "type": "text",
                            "text": f"Body {part_index}",
                            "page_idx": 1,
                        },
                        {
                            "type": "image",
                            "image_path": "assets/figure.png",
                            "image_caption": [f"Figure {part_index}"],
                            "page_idx": 1,
                        },
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
            )
            zf.writestr(f"part_{part_index}_full.md", f"# Part {part_index}\n")
            zf.writestr("assets/figure.png", b"fake")
        if progress is not None:
            progress(f"saved {output_path.name}")
        return output_path


if __name__ == "__main__":
    unittest.main()
