from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from hep_rag_v2 import db, paths
from hep_rag_v2.config import default_config, runtime_collection_config
from hep_rag_v2.metadata import upsert_collection
from hep_rag_v2.pipeline import _select_reparse_candidates


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
                        replace_existing=False,
                    )

                self.assertEqual(len(candidates), 1)
                self.assertEqual(candidates[0]["work_id"], 1)
                self.assertEqual(candidates[0]["parse_status"], "failed")
        finally:
            paths.set_workspace_root(original_root)


if __name__ == "__main__":
    unittest.main()
