from __future__ import annotations

import contextlib
import io
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hep_rag_v2 import cli, db, paths
from hep_rag_v2.fulltext import build_chunks, import_mineru_source, materialize_mineru_document
from hep_rag_v2.graph import graph_neighbors, rebuild_graph_edges
from hep_rag_v2.metadata import backfill_unresolved_citations, upsert_collection, upsert_work_from_hit
from hep_rag_v2.search import (
    rebuild_search_indices,
    search_assets_bm25,
    search_chunks_bm25,
    search_formulas_bm25,
    search_works_bm25,
)
from hep_rag_v2.textnorm import normalize_display_text, normalize_search_text


@contextlib.contextmanager
def _patch_workspace(tmp: Path):
    original = paths.workspace_root()
    try:
        paths.set_workspace_root(tmp)
        yield
    finally:
        paths.set_workspace_root(original)


class TestBootstrap(unittest.TestCase):
    def test_ensure_db_initializes_schema_and_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            with _patch_workspace(tmp):
                db.ensure_db()

            db_path = tmp / "db" / "hep_rag_v2.db"
            self.assertTrue(db_path.exists())
            self.assertTrue((tmp / "collections").exists())
            self.assertTrue((tmp / "data" / "raw" / "inspire").exists())
            self.assertTrue((tmp / "data" / "parsed").exists())
            self.assertTrue((tmp / "indexes").exists())
            self.assertTrue((tmp / "exports").exists())

            with contextlib.closing(sqlite3.connect(db_path)) as conn:
                names = {
                    row[0]
                    for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                }
            for required in (
                "collections",
                "ingest_runs",
                "works",
                "work_ids",
                "authors",
                "citations",
                "documents",
                "chunks",
                "result_objects",
                "method_objects",
                "transfer_candidates",
                "reasoning_sessions",
                "reasoning_steps",
                "idea_candidates",
                "idea_scores",
                "idea_evidence_links",
            ):
                self.assertIn(required, names)

    def test_cli_bootstrap_legacy_corpus_imports_metadata_and_documents(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            with _patch_workspace(tmp):
                collections_dir = tmp / "collections"
                collections_dir.mkdir(parents=True, exist_ok=True)
                (collections_dir / "cms_rare_decay.json").write_text(
                    json.dumps(
                        {
                            "name": "cms_rare_decay",
                            "label": "CMS rare decays",
                            "source_priority": ["inspirehep", "arxiv"],
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                legacy_db = tmp / "legacy.db"
                with contextlib.closing(sqlite3.connect(legacy_db)) as legacy_conn:
                    legacy_conn.execute(
                        """
                        CREATE TABLE papers (
                          paper_id INTEGER PRIMARY KEY AUTOINCREMENT,
                          collection TEXT NOT NULL,
                          inspire_id TEXT,
                          arxiv_id TEXT,
                          doi TEXT,
                          title TEXT,
                          abstract TEXT,
                          year INTEGER,
                          collaboration TEXT,
                          experiments_json TEXT,
                          authors_json TEXT,
                          keywords_json TEXT,
                          citation_count INTEGER,
                          source_url TEXT,
                          pdf_url TEXT,
                          local_pdf_path TEXT,
                          local_txt_path TEXT,
                          raw_metadata_json TEXT,
                          created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                          updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                        )
                        """
                    )
                    legacy_conn.executemany(
                        """
                        INSERT INTO papers (
                          collection, inspire_id, arxiv_id, doi, title, abstract, year, collaboration,
                          citation_count, source_url, raw_metadata_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            (
                                "cms_rare_decay",
                                "901",
                                "2501.00011",
                                None,
                                "Bootstrap source paper",
                                "A source-paper abstract.",
                                2025,
                                "CMS",
                                3,
                                "https://inspirehep.net/literature/901",
                                json.dumps(
                                    {
                                        "control_number": 901,
                                        "titles": [{"title": "Bootstrap source paper"}],
                                        "abstracts": [{"value": "A source-paper abstract."}],
                                        "publication_info": [{"year": 2025}],
                                        "arxiv_eprints": [{"value": "2501.00011"}],
                                        "references": [{"control_number": 902}],
                                    },
                                    ensure_ascii=False,
                                ),
                            ),
                            (
                                "cms_rare_decay",
                                "902",
                                "2501.00012",
                                None,
                                "Bootstrap target paper",
                                "A target-paper abstract.",
                                2024,
                                "CMS",
                                1,
                                "https://inspirehep.net/literature/902",
                                json.dumps(
                                    {
                                        "control_number": 902,
                                        "titles": [{"title": "Bootstrap target paper"}],
                                        "abstracts": [{"value": "A target-paper abstract."}],
                                        "publication_info": [{"year": 2024}],
                                        "arxiv_eprints": [{"value": "2501.00012"}],
                                    },
                                    ensure_ascii=False,
                                ),
                            ),
                        ],
                    )
                    legacy_conn.commit()

                parsed_root = tmp / "legacy_parsed" / "cms_rare_decay" / "2501.00011"
                raw_dir = parsed_root / "raw"
                raw_dir.mkdir(parents=True)
                (raw_dir / "paper_full.md").write_text("# Bootstrap source paper\n", encoding="utf-8")
                (raw_dir / "paper_content_list.json").write_text(
                    json.dumps(
                        [
                            {"type": "text", "text_level": 1, "text": "Bootstrap source paper", "page_idx": 1},
                            {"type": "text", "text_level": 1, "text": "Abstract", "page_idx": 1},
                            {"type": "text", "text": "We summarize the source measurement [1].", "page_idx": 1},
                            {"type": "text", "text_level": 1, "text": "1 Method", "page_idx": 2},
                            {"type": "text", "text": "The profile likelihood fit constrains the signal yield.", "page_idx": 2},
                            {"type": "text", "text_level": 1, "text": "References", "page_idx": 3},
                            {"type": "text", "text": "[1] Bootstrap target paper.", "page_idx": 3},
                        ],
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                (parsed_root / "manifest.json").write_text(
                    json.dumps(
                        {
                            "engine": "mineru",
                            "raw_dir": str(raw_dir),
                            "full_md_path": str(raw_dir / "paper_full.md"),
                            "content_list_path": str(raw_dir / "paper_content_list.json"),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                db.ensure_db()
                parser = cli.build_parser()
                args = parser.parse_args(
                    [
                        "bootstrap-legacy-corpus",
                        "--legacy-db",
                        str(legacy_db),
                        "--parsed-root",
                        str(tmp / "legacy_parsed" / "cms_rare_decay"),
                        "--collection",
                        "cms_rare_decay",
                        "--audit-limit",
                        "2",
                        "--min-shared",
                        "1",
                    ]
                )

                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    args.func(args)
                payload = json.loads(out.getvalue())

                self.assertEqual(payload["metadata"]["legacy_papers"], 2)
                self.assertEqual(payload["metadata"]["created"], 2)
                self.assertEqual(payload["metadata"]["resolved_citations"], 1)
                self.assertEqual(payload["import"]["matched_manifests"], 1)
                self.assertEqual(payload["import"]["materialized"], 1)
                self.assertEqual(payload["import"]["ready_documents"], 1)
                self.assertEqual(payload["import"]["not_ready_documents"], 0)
                self.assertEqual(payload["status"]["works"], 2)
                self.assertEqual(payload["status"]["documents"], 1)
                self.assertGreaterEqual(payload["status"]["chunks"], 2)
                self.assertEqual(payload["search"]["works"], 2)
                self.assertGreaterEqual(payload["search"]["chunks"], 2)
                self.assertIsNotNone(payload["graph"])

                with db.connect() as conn:
                    work_ids = conn.execute(
                        "SELECT canonical_id FROM works ORDER BY canonical_id"
                    ).fetchall()
                    self.assertEqual([row[0] for row in work_ids], ["901", "902"])
                    doc = conn.execute(
                        """
                        SELECT d.document_id, w.canonical_id
                        FROM documents d
                        JOIN works w ON w.work_id = d.work_id
                        """
                    ).fetchone()
                    self.assertEqual(doc["canonical_id"], "901")

    def test_cli_enrich_inspire_metadata_backfills_citations_and_graph(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            collections_dir = tmp / "collections"
            collections_dir.mkdir()
            (collections_dir / "cms_rare_decay.json").write_text(
                json.dumps(
                    {
                        "name": "cms_rare_decay",
                        "label": "CMS rare decays",
                        "source_priority": ["inspirehep", "arxiv"],
                        "fields": ["control_number", "titles", "abstracts", "references"],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            with _patch_workspace(tmp):
                db.ensure_db()
                with db.connect() as conn:
                    collection_id = upsert_collection(conn, {"name": "cms_rare_decay"})
                    for control_number, title in [
                        (7001, "Shared target"),
                        (7002, "Source A"),
                        (7003, "Source B"),
                    ]:
                        upsert_work_from_hit(
                            conn,
                            collection_id=collection_id,
                            hit={
                                "metadata": {
                                    "control_number": control_number,
                                    "titles": [{"title": title}],
                                }
                            },
                        )
                    conn.commit()

                payload_by_id = {
                    "7001": {
                        "links": {"self": "https://inspirehep.net/literature/7001"},
                        "metadata": {
                            "control_number": 7001,
                            "titles": [{"title": "Shared target"}],
                            "abstracts": [{"value": "Target abstract."}],
                            "references": [],
                        },
                    },
                    "7002": {
                        "links": {"self": "https://inspirehep.net/literature/7002"},
                        "metadata": {
                            "control_number": 7002,
                            "titles": [{"title": "Source A"}],
                            "abstracts": [{"value": "Source A abstract."}],
                            "references": [{"control_number": 7001}],
                        },
                    },
                    "7003": {
                        "links": {"self": "https://inspirehep.net/literature/7003"},
                        "metadata": {
                            "control_number": 7003,
                            "titles": [{"title": "Source B"}],
                            "abstracts": [{"value": "Source B abstract."}],
                            "references": [{"control_number": 7001}],
                        },
                    },
                }

                def fake_http_get_json(url: str, *, timeout: int = 60, retries: int = 3) -> dict[str, Any]:
                    del timeout, retries
                    work_id = url.split("/literature/", 1)[1].split("?", 1)[0]
                    return payload_by_id[work_id]

                out = io.StringIO()
                with mock.patch("hep_rag_v2.cli.ingest.http_get_json", side_effect=fake_http_get_json):
                    with contextlib.redirect_stdout(out):
                        cli.cmd_enrich_inspire_metadata(
                            SimpleNamespace(
                                collection="cms_rare_decay",
                                limit=None,
                                force=False,
                                timeout=60,
                                retries=3,
                                sleep=0.0,
                                skip_search=False,
                                skip_graph=False,
                                min_shared=1,
                            )
                        )

                payload = json.loads(out.getvalue())
                self.assertEqual(payload["targets"], 3)
                self.assertEqual(payload["fetched"], 3)
                self.assertEqual(payload["citations_written"], 2)
                self.assertEqual(payload["resolved_citations"], 0)
                self.assertEqual(payload["status"]["citations"], 2)
                self.assertEqual(payload["status"]["resolved_citations"], 2)
                self.assertEqual(payload["status"]["bibliographic_coupling_edges"], 1)
                self.assertEqual(payload["search"]["works"], 3)

                with db.connect() as conn:
                    neighbor = graph_neighbors(
                        conn,
                        work_id=conn.execute("SELECT work_id FROM works WHERE canonical_id = '7002'").fetchone()[0],
                        edge_kind="bibliographic-coupling",
                        collection="cms_rare_decay",
                        limit=5,
                    )
                    self.assertEqual(len(neighbor), 1)
                    self.assertEqual(neighbor[0]["canonical_id"], "7003")


class TestMetadataGraph(unittest.TestCase):
    def test_upsert_work_materializes_graph_rows(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            with _patch_workspace(tmp):
                db.ensure_db()
                with db.connect() as conn:
                    collection_id = upsert_collection(
                        conn,
                        {
                            "name": "cms_rare_decay",
                            "label": "CMS rare decays",
                            "source_priority": ["inspirehep"],
                        },
                    )
                    hit = {
                        "links": {"self": "https://inspirehep.net/literature/123456"},
                        "metadata": {
                            "control_number": 123456,
                            "titles": [{"title": "Measurement of a rare decay"}],
                            "abstracts": [{"value": "We study a rare CMS decay mode."}],
                            "authors": [
                                {"full_name": "Alice Example", "affiliations": [{"value": "CERN"}]},
                                {"full_name": "Bob Example", "affiliations": [{"value": "MIT"}]},
                            ],
                            "collaborations": [{"value": "CMS"}],
                            "publication_info": [{"year": 2024, "journal_title": "JHEP"}],
                            "arxiv_eprints": [{"value": "2401.01234"}],
                            "dois": [{"value": "10.1000/example"}],
                            "citation_count": 12,
                            "keywords": [{"value": "rare decay"}, {"value": "FCNC"}],
                            "inspire_categories": [{"term": "Phenomenology of High Energy Physics"}],
                            "accelerator_experiments": [{"experiment": "CMS"}],
                            "documents": [{"url": "https://arxiv.org/pdf/2401.01234.pdf"}],
                            "references": [
                                {"control_number": 987654},
                            ],
                        },
                    }
                    stats = upsert_work_from_hit(conn, collection_id=collection_id, hit=hit)
                    conn.commit()

                    self.assertEqual(stats["created"], 1)
                    self.assertEqual(conn.execute("SELECT COUNT(*) FROM works").fetchone()[0], 1)
                    self.assertEqual(conn.execute("SELECT COUNT(*) FROM work_ids").fetchone()[0], 3)
                    self.assertEqual(conn.execute("SELECT COUNT(*) FROM authors").fetchone()[0], 2)
                    self.assertEqual(conn.execute("SELECT COUNT(*) FROM work_authors").fetchone()[0], 2)
                    self.assertEqual(conn.execute("SELECT COUNT(*) FROM collaborations").fetchone()[0], 1)
                    self.assertEqual(conn.execute("SELECT COUNT(*) FROM work_venues").fetchone()[0], 1)
                    self.assertEqual(conn.execute("SELECT COUNT(*) FROM work_topics").fetchone()[0], 4)
                    citation = conn.execute(
                        "SELECT dst_source, dst_external_id, resolution_status FROM citations"
                    ).fetchone()
                    self.assertEqual(dict(citation), {
                        "dst_source": "inspire",
                        "dst_external_id": "987654",
                        "resolution_status": "unresolved",
                    })

    def test_backfill_resolves_citations_after_target_arrives(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            with _patch_workspace(tmp):
                db.ensure_db()
                with db.connect() as conn:
                    collection_id = upsert_collection(conn, {"name": "cms_rare_decay"})

                    src_hit = {
                        "metadata": {
                            "control_number": 111,
                            "titles": [{"title": "Source paper"}],
                            "references": [{"control_number": 222}],
                        }
                    }
                    dst_hit = {
                        "metadata": {
                            "control_number": 222,
                            "titles": [{"title": "Target paper"}],
                        }
                    }

                    upsert_work_from_hit(conn, collection_id=collection_id, hit=src_hit)
                    resolved_before = backfill_unresolved_citations(conn)
                    upsert_work_from_hit(conn, collection_id=collection_id, hit=dst_hit)
                    resolved_after = backfill_unresolved_citations(conn)
                    conn.commit()

                    self.assertEqual(resolved_before, 0)
                    self.assertEqual(resolved_after, 1)
                    row = conn.execute(
                        "SELECT dst_work_id, resolution_status FROM citations WHERE dst_external_id = '222'"
                    ).fetchone()
                    self.assertIsNotNone(row["dst_work_id"])
                    self.assertEqual(row["resolution_status"], "resolved")


class TestFullTextMaterialization(unittest.TestCase):
    def test_materialize_mineru_document_strips_inline_citations_and_excludes_bibliography(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            with _patch_workspace(tmp):
                db.ensure_db()
                with db.connect() as conn:
                    collection_id = upsert_collection(conn, {"name": "cms_rare_decay"})
                    hit = {
                        "metadata": {
                            "control_number": 333,
                            "titles": [{"title": "Search for a rare decay"}],
                            "arxiv_eprints": [{"value": "2401.00001"}],
                        }
                    }
                    target_hit = {
                        "metadata": {
                            "control_number": 334,
                            "titles": [{"title": "First paper on the topic"}],
                            "dois": [{"value": "10.1000/first-paper"}],
                        }
                    }
                    upsert_work_from_hit(conn, collection_id=collection_id, hit=hit)
                    upsert_work_from_hit(conn, collection_id=collection_id, hit=target_hit)
                    work_id = conn.execute("SELECT work_id FROM works WHERE canonical_id = '333'").fetchone()[0]
                    conn.commit()

                    bundle_dir = tmp / "bundle"
                    bundle_dir.mkdir()
                    (bundle_dir / "paper_full.md").write_text("# Search for a rare decay\n", encoding="utf-8")
                    (bundle_dir / "paper_content_list.json").write_text(
                        json.dumps(
                            [
                                {"type": "text", "text_level": 1, "text": "Search for a rare decay", "page_idx": 1},
                                {"type": "text", "text": "The CMS Collaboration", "page_idx": 1},
                                {"type": "text", "text_level": 1, "text": "Abstract", "page_idx": 1},
                                {"type": "text", "text": "We follow Refs. [1, 2] and measure the branching fraction.", "page_idx": 1},
                                {"type": "text", "text_level": 1, "text": "1 Introduction", "page_idx": 2},
                                {"type": "text", "text": "The rare decay [3-5] is studied with a profile likelihood.", "page_idx": 2},
                                {"type": "equation", "latex": r"E = mc^2", "page_idx": 2},
                                {"type": "image", "caption": "Observed limit [6]", "image_path": "figures/fig1.png", "page_idx": 3},
                                {"type": "text", "text_level": 1, "text": "References", "page_idx": 4},
                                {"type": "text", "text": "[1] First paper on the topic. https://doi.org/10.1000/first-paper.", "page_idx": 4},
                            ],
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )

                    dest_dir = paths.PARSED_DIR / "cms_rare_decay" / "2401.00001"
                    import_mineru_source(source_path=bundle_dir, dest_dir=dest_dir)
                    summary = materialize_mineru_document(
                        conn,
                        work_id=int(work_id),
                        manifest_path=dest_dir / "manifest.json",
                    )
                    conn.commit()

                    self.assertEqual(summary["block_roles"]["abstract"], 2)
                    self.assertEqual(summary["block_roles"]["bibliography"], 2)
                    self.assertGreaterEqual(summary["chunk_roles"]["section_child"], 1)
                    self.assertEqual(summary["chunk_roles"]["abstract_chunk"], 1)
                    self.assertEqual(summary["chunk_roles"]["formula_window"], 1)
                    self.assertEqual(summary["chunk_roles"]["asset_window"], 1)
                    self.assertEqual(summary["citations_written"], 1)
                    self.assertEqual(summary["citations_resolved"], 1)
                    self.assertEqual(summary["bibliography_entries"], 1)

                    abstract_texts = [
                        row["clean_text"]
                        for row in conn.execute(
                            """
                            SELECT clean_text
                            FROM blocks
                            WHERE document_id = ?
                              AND block_role = 'abstract'
                              AND clean_text IS NOT NULL
                            ORDER BY block_id
                            """,
                            (summary["document_id"],),
                        ).fetchall()
                    ]
                    body_texts = [
                        row["clean_text"]
                        for row in conn.execute(
                            """
                            SELECT clean_text
                            FROM blocks
                            WHERE document_id = ?
                              AND block_role = 'body'
                              AND clean_text IS NOT NULL
                            ORDER BY block_id
                            """,
                            (summary["document_id"],),
                        ).fetchall()
                    ]
                    self.assertTrue(any("[1, 2]" not in text and "branching fraction" in text for text in abstract_texts))
                    self.assertTrue(any("Refs." not in text and "branching fraction" in text for text in abstract_texts))
                    self.assertTrue(any("[3-5]" not in text and "profile likelihood" in text for text in body_texts))
                    biblio_row = conn.execute(
                        """
                        SELECT is_retrievable, exclusion_reason
                        FROM blocks
                        WHERE document_id = ?
                          AND block_role = 'bibliography'
                        ORDER BY block_id DESC
                        LIMIT 1
                        """,
                        (summary["document_id"],),
                    ).fetchone()
                    self.assertEqual((int(biblio_row["is_retrievable"]), biblio_row["exclusion_reason"]), (0, "bibliography"))
                    chunk_texts = [
                        row[0]
                        for row in conn.execute(
                            "SELECT clean_text FROM chunks WHERE document_id = ? ORDER BY chunk_id",
                            (summary["document_id"],),
                        ).fetchall()
                    ]
                    section_child_texts = [
                        row[0]
                        for row in conn.execute(
                            """
                            SELECT clean_text
                            FROM chunks
                            WHERE document_id = ?
                              AND chunk_role = 'section_child'
                            ORDER BY chunk_id
                            """,
                            (summary["document_id"],),
                        ).fetchall()
                    ]
                    formula_window_texts = [
                        row[0]
                        for row in conn.execute(
                            """
                            SELECT clean_text
                            FROM chunks
                            WHERE document_id = ?
                              AND chunk_role = 'formula_window'
                            ORDER BY chunk_id
                            """,
                            (summary["document_id"],),
                        ).fetchall()
                    ]
                    self.assertTrue(any("branching fraction" in text for text in chunk_texts))
                    self.assertFalse(any("First paper on the topic" in text for text in chunk_texts))
                    self.assertTrue(any("Equation:" in text for text in section_child_texts))
                    self.assertFalse(any("E = mc^2" in text for text in formula_window_texts))
                    self.assertEqual(
                        conn.execute("SELECT COUNT(*) FROM formulas WHERE document_id = ?", (summary["document_id"],)).fetchone()[0],
                        1,
                    )
                    citation_row = conn.execute(
                        """
                        SELECT dst_source, dst_external_id, dst_work_id, resolution_status, raw_json
                        FROM citations
                        WHERE src_work_id = ?
                        """,
                        (int(work_id),),
                    ).fetchone()
                    self.assertEqual(citation_row["dst_source"], "doi")
                    self.assertEqual(citation_row["dst_external_id"], "10.1000/first-paper")
                    self.assertIsNotNone(citation_row["dst_work_id"])
                    self.assertEqual(citation_row["resolution_status"], "resolved")
                    self.assertIn("mineru_bibliography", citation_row["raw_json"])

    def test_cli_import_mineru_and_show_document(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            with _patch_workspace(tmp):
                db.ensure_db()
                with db.connect() as conn:
                    collection_id = upsert_collection(conn, {"name": "cms_rare_decay"})
                    hit = {
                        "metadata": {
                            "control_number": 444,
                            "titles": [{"title": "Rare decay note"}],
                            "arxiv_eprints": [{"value": "2401.00002"}],
                        }
                    }
                    upsert_work_from_hit(conn, collection_id=collection_id, hit=hit)
                    conn.commit()

                bundle_dir = tmp / "bundle_cli"
                bundle_dir.mkdir()
                (bundle_dir / "paper_full.md").write_text("# Rare decay note\n", encoding="utf-8")
                (bundle_dir / "paper_content_list.json").write_text(
                    json.dumps(
                        [
                            {"type": "text", "text_level": 1, "text": "Rare decay note", "page_idx": 1},
                            {"type": "text", "text_level": 1, "text": "Abstract", "page_idx": 1},
                            {"type": "text", "text": "An overview of the search [1].", "page_idx": 1},
                            {"type": "text", "text_level": 1, "text": "1 Method", "page_idx": 2},
                            {"type": "text", "text": "We fit the signal model with templates.", "page_idx": 2},
                        ],
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )

                import_out = io.StringIO()
                with contextlib.redirect_stdout(import_out):
                    cli.cmd_import_mineru(
                        SimpleNamespace(
                            source=str(bundle_dir),
                            collection="cms_rare_decay",
                            work_id=None,
                            id_type="arxiv",
                            id_value="2401.00002",
                            replace=False,
                            chunk_size=2400,
                            overlap_blocks=1,
                            section_parent_char_limit=12000,
                        )
                    )
                import_payload = json.loads(import_out.getvalue())
                self.assertEqual(import_payload["chunk_roles"]["abstract_chunk"], 1)

                show_out = io.StringIO()
                with contextlib.redirect_stdout(show_out):
                    cli.cmd_show_document(
                        SimpleNamespace(
                            work_id=None,
                            id_type="arxiv",
                            id_value="2401.00002",
                            limit=10,
                        )
                    )
                show_payload = json.loads(show_out.getvalue())
                self.assertEqual(show_payload["document"]["parse_status"], "materialized")
                self.assertIn("abstract", show_payload["block_roles"])
                self.assertIn("section_child", show_payload["chunk_roles"])

                with db.connect() as conn:
                    rebuild_search_indices(conn, target="all")
                    conn.commit()

                audit_out = io.StringIO()
                with contextlib.redirect_stdout(audit_out):
                    cli.cmd_audit_document(
                        SimpleNamespace(
                            work_id=None,
                            id_type="arxiv",
                            id_value="2401.00002",
                            limit=10,
                        )
                    )
                audit_payload = json.loads(audit_out.getvalue())
                self.assertTrue(audit_payload["ready"])
                self.assertEqual(audit_payload["recommendation"], "ready_for_next_phase")
                self.assertTrue(audit_payload["readiness_checks"]["chunk_index_complete"])

    def test_audit_document_ignores_valid_section_references_and_inequalities(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            with _patch_workspace(tmp):
                db.ensure_db()
                with db.connect() as conn:
                    collection_id = upsert_collection(conn, {"name": "cms_rare_decay"})
                    hit = {
                        "metadata": {
                            "control_number": 445,
                            "titles": [{"title": "Valid section reference note"}],
                            "arxiv_eprints": [{"value": "2401.00003"}],
                        }
                    }
                    upsert_work_from_hit(conn, collection_id=collection_id, hit=hit)
                    conn.commit()

                bundle_dir = tmp / "bundle_valid_audit"
                bundle_dir.mkdir()
                (bundle_dir / "paper_full.md").write_text("# Valid section reference note\n", encoding="utf-8")
                (bundle_dir / "paper_content_list.json").write_text(
                    json.dumps(
                        [
                            {"type": "text", "text_level": 1, "text": "Valid section reference note", "page_idx": 1},
                            {"type": "text", "text_level": 1, "text": "Abstract", "page_idx": 1},
                            {"type": "text", "text": "The treatment is detailed in Section 9.2.", "page_idx": 1},
                            {"type": "text", "text_level": 1, "text": "1 Selection", "page_idx": 2},
                            {"type": "text", "text": "We require m ℓ ^ + ℓ ^ - > 40 GeV and the detector description can be found in.", "page_idx": 2},
                        ],
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )

                import_out = io.StringIO()
                with contextlib.redirect_stdout(import_out):
                    cli.cmd_import_mineru(
                        SimpleNamespace(
                            source=str(bundle_dir),
                            collection="cms_rare_decay",
                            work_id=None,
                            id_type="arxiv",
                            id_value="2401.00003",
                            replace=False,
                            chunk_size=2400,
                            overlap_blocks=1,
                            section_parent_char_limit=12000,
                        )
                    )

                with db.connect() as conn:
                    rebuild_search_indices(conn, target="all")
                    conn.commit()

                audit_out = io.StringIO()
                with contextlib.redirect_stdout(audit_out):
                    cli.cmd_audit_document(
                        SimpleNamespace(
                            work_id=None,
                            id_type="arxiv",
                            id_value="2401.00003",
                            limit=10,
                        )
                    )
                audit_payload = json.loads(audit_out.getvalue())
                self.assertTrue(audit_payload["ready"])
                self.assertEqual(audit_payload["noise"]["retrievable_chunks"]["pattern_counts"]["orphan_citation_phrase"], 0)
                self.assertEqual(audit_payload["noise"]["retrievable_chunks"]["pattern_counts"]["arrow_split"], 0)


class TestSearchIndex(unittest.TestCase):
    def test_build_chunks_repairs_continuation_blocks_before_packing(self) -> None:
        blocks = [
            {
                "block_id": 1,
                "section_id": 10,
                "section_kind": "body",
                "section_hint": "Method",
                "section_path": "Paper / Method",
                "block_type": "text",
                "page": 1,
                "order_index": 1,
                "raw_text": "We use the following observables:",
                "clean_text": "We use the following observables:",
                "block_role": "body",
                "is_heading": False,
                "is_retrievable": True,
                "flags": {"citation_markers_removed": 0},
            },
            {
                "block_id": 2,
                "section_id": 10,
                "section_kind": "body",
                "section_hint": "Method",
                "section_path": "Paper / Method",
                "block_type": "text",
                "page": 1,
                "order_index": 2,
                "raw_text": "trimuon pT, vertex chi2, and isolation.",
                "clean_text": "trimuon pT, vertex chi2, and isolation.",
                "block_role": "body",
                "is_heading": False,
                "is_retrievable": True,
                "flags": {"citation_markers_removed": 0},
            },
            {
                "block_id": 3,
                "section_id": 10,
                "section_kind": "body",
                "section_hint": "Method",
                "section_path": "Paper / Method",
                "block_type": "text",
                "page": 1,
                "order_index": 3,
                "raw_text": "The final fit is performed with templates.",
                "clean_text": "The final fit is performed with templates.",
                "block_role": "body",
                "is_heading": False,
                "is_retrievable": True,
                "flags": {"citation_markers_removed": 0},
            },
        ]

        chunks = build_chunks(
            blocks,
            chunk_size=50,
            overlap_blocks=0,
            section_parent_char_limit=1000,
        )
        section_children = [row for row in chunks if row["chunk_role"] == "section_child"]
        self.assertEqual(len(section_children), 2)
        self.assertIn("observables: trimuon pT", section_children[0]["clean_text"])
        self.assertNotIn("observables:\n\ntrimuon", section_children[0]["clean_text"])

    def test_build_chunks_splits_long_text_by_sentence_budget(self) -> None:
        blocks = [
            {
                "block_id": 1,
                "section_id": 11,
                "section_kind": "body",
                "section_hint": "Results",
                "section_path": "Paper / Results",
                "block_type": "text",
                "page": 1,
                "order_index": 1,
                "raw_text": (
                    "The first sentence summarizes the setup. "
                    "The second sentence explains the observable in more detail. "
                    "The third sentence reports the final result."
                ),
                "clean_text": (
                    "The first sentence summarizes the setup. "
                    "The second sentence explains the observable in more detail. "
                    "The third sentence reports the final result."
                ),
                "block_role": "body",
                "is_heading": False,
                "is_retrievable": True,
                "flags": {"citation_markers_removed": 0},
            },
        ]

        chunks = build_chunks(
            blocks,
            chunk_size=70,
            overlap_blocks=0,
            section_parent_char_limit=1000,
        )
        section_children = [row for row in chunks if row["chunk_role"] == "section_child"]
        self.assertGreaterEqual(len(section_children), 2)
        self.assertTrue(section_children[0]["clean_text"].endswith("."))
        self.assertIn("The third sentence reports the final result.", section_children[-1]["clean_text"])

    def test_build_chunks_inserts_equation_bridge_placeholder(self) -> None:
        blocks = [
            {
                "block_id": 1,
                "section_id": 12,
                "section_kind": "body",
                "section_hint": "Method",
                "section_path": "Paper / Method",
                "block_type": "text",
                "page": 2,
                "order_index": 1,
                "raw_text": "The expected yield is related to the branching fraction by",
                "clean_text": "The expected yield is related to the branching fraction by",
                "block_role": "body",
                "is_heading": False,
                "is_retrievable": True,
                "flags": {"citation_markers_removed": 0},
            },
            {
                "block_id": 2,
                "section_id": 12,
                "section_kind": "body",
                "section_hint": "Method",
                "section_path": "Paper / Method",
                "block_type": "equation",
                "page": 2,
                "order_index": 2,
                "raw_text": "N_sig = epsilon * B",
                "clean_text": "N sig = ϵ B",
                "block_role": "body",
                "is_heading": False,
                "is_retrievable": True,
                "flags": {"citation_markers_removed": 0},
            },
            {
                "block_id": 3,
                "section_id": 12,
                "section_kind": "body",
                "section_hint": "Method",
                "section_path": "Paper / Method",
                "block_type": "text",
                "page": 2,
                "order_index": 3,
                "raw_text": "where epsilon denotes the efficiency.",
                "clean_text": "where epsilon denotes the efficiency.",
                "block_role": "body",
                "is_heading": False,
                "is_retrievable": True,
                "flags": {"citation_markers_removed": 0},
            },
        ]

        chunks = build_chunks(
            blocks,
            chunk_size=200,
            overlap_blocks=0,
            section_parent_char_limit=1000,
        )
        section_child = [row for row in chunks if row["chunk_role"] == "section_child"][0]
        self.assertIn("Equation:", section_child["clean_text"])
        self.assertIn("where epsilon denotes the efficiency.", section_child["clean_text"])

    def test_build_chunks_equation_placeholder_keeps_arrow_tokens(self) -> None:
        blocks = [
            {
                "block_id": 1,
                "section_id": 13,
                "section_kind": "body",
                "section_hint": "Method",
                "section_path": "Paper / Method",
                "block_type": "text",
                "page": 2,
                "order_index": 1,
                "raw_text": "The normalization is defined as",
                "clean_text": "The normalization is defined as",
                "block_role": "body",
                "is_heading": False,
                "is_retrievable": True,
                "flags": {"citation_markers_removed": 0},
            },
            {
                "block_id": 2,
                "section_id": 13,
                "section_kind": "body",
                "section_hint": "Method",
                "section_path": "Paper / Method",
                "block_type": "equation",
                "page": 2,
                "order_index": 2,
                "raw_text": "L_mu(B) = B(B -> tau + X)",
                "clean_text": "L μ (B) = B (B -> τ + X)",
                "block_role": "body",
                "is_heading": False,
                "is_retrievable": True,
                "flags": {"citation_markers_removed": 0},
            },
            {
                "block_id": 3,
                "section_id": 13,
                "section_kind": "body",
                "section_hint": "Method",
                "section_path": "Paper / Method",
                "block_type": "text",
                "page": 2,
                "order_index": 3,
                "raw_text": "The control sample constrains the yield.",
                "clean_text": "The control sample constrains the yield.",
                "block_role": "body",
                "is_heading": False,
                "is_retrievable": True,
                "flags": {"citation_markers_removed": 0},
            },
        ]

        chunks = build_chunks(
            blocks,
            chunk_size=200,
            overlap_blocks=0,
            section_parent_char_limit=1000,
        )
        section_child = [row for row in chunks if row["chunk_role"] == "section_child"][0]
        self.assertIn("B -> τ + X", section_child["clean_text"])
        self.assertNotIn("B - >", section_child["clean_text"])

    def test_build_chunks_equation_placeholder_linearizes_fractions(self) -> None:
        blocks = [
            {
                "block_id": 1,
                "section_id": 14,
                "section_kind": "body",
                "section_hint": "Method",
                "section_path": "Paper / Method",
                "block_type": "text",
                "page": 2,
                "order_index": 1,
                "raw_text": "The branching fraction is related to the signal normalization by",
                "clean_text": "The branching fraction is related to the signal normalization by",
                "block_role": "body",
                "is_heading": False,
                "is_retrievable": True,
                "flags": {"citation_markers_removed": 0},
            },
            {
                "block_id": 2,
                "section_id": 14,
                "section_kind": "body",
                "section_hint": "Method",
                "section_path": "Paper / Method",
                "block_type": "equation",
                "page": 2,
                "order_index": 2,
                "raw_text": (
                    r"$$\mathcal{B}(\tau \to 3 \mu) = \frac{N_{\mathrm{sig}}}{\epsilon}"
                    r"\frac{\mathcal{A}_{3\mu(\mathrm{W})}}{\mathcal{A}_{\mu\mu\pi}}$$"
                ),
                "clean_text": "B (τ -> 3 μ) = frac N sig ϵ frac A 3 μ (W) A μ μ π",
                "block_role": "body",
                "is_heading": False,
                "is_retrievable": True,
                "flags": {"citation_markers_removed": 0},
            },
            {
                "block_id": 3,
                "section_id": 14,
                "section_kind": "body",
                "section_hint": "Method",
                "section_path": "Paper / Method",
                "block_type": "text",
                "page": 2,
                "order_index": 3,
                "raw_text": "The efficiency is measured in data.",
                "clean_text": "The efficiency is measured in data.",
                "block_role": "body",
                "is_heading": False,
                "is_retrievable": True,
                "flags": {"citation_markers_removed": 0},
            },
        ]

        chunks = build_chunks(
            blocks,
            chunk_size=220,
            overlap_blocks=0,
            section_parent_char_limit=1000,
        )
        section_child = [row for row in chunks if row["chunk_role"] == "section_child"][0]
        self.assertIn("Equation:", section_child["clean_text"])
        self.assertIn("/", section_child["clean_text"])
        self.assertNotRegex(section_child["clean_text"], r"\bfrac\b")
        self.assertNotIn("- >", section_child["clean_text"])

    def test_build_chunks_skips_bullet_display_lists_misclassified_as_equations(self) -> None:
        blocks = [
            {
                "block_id": 1,
                "section_id": 15,
                "section_kind": "body",
                "section_hint": "Channels",
                "section_path": "Paper / Channels",
                "block_type": "text",
                "page": 2,
                "order_index": 1,
                "raw_text": "We consider the following channels:",
                "clean_text": "We consider the following channels:",
                "block_role": "body",
                "is_heading": False,
                "is_retrievable": True,
                "flags": {"citation_markers_removed": 0},
            },
            {
                "block_id": 2,
                "section_id": 15,
                "section_kind": "body",
                "section_hint": "Channels",
                "section_path": "Paper / Channels",
                "block_type": "equation",
                "page": 2,
                "order_index": 2,
                "raw_text": r"$$\begin{array}{rl} & \bullet H \to WW \\ & \bullet H \to \gamma\gamma \end{array}$$",
                "clean_text": "bullet H -> WW, bullet H -> γγ",
                "block_role": "body",
                "is_heading": False,
                "is_retrievable": True,
                "flags": {"citation_markers_removed": 0},
            },
            {
                "block_id": 3,
                "section_id": 15,
                "section_kind": "body",
                "section_hint": "Channels",
                "section_path": "Paper / Channels",
                "block_type": "text",
                "page": 2,
                "order_index": 3,
                "raw_text": "The final states are combined statistically.",
                "clean_text": "The final states are combined statistically.",
                "block_role": "body",
                "is_heading": False,
                "is_retrievable": True,
                "flags": {"citation_markers_removed": 0},
            },
        ]

        chunks = build_chunks(
            blocks,
            chunk_size=240,
            overlap_blocks=0,
            section_parent_char_limit=1000,
        )
        section_child = [row for row in chunks if row["chunk_role"] == "section_child"][0]
        self.assertNotIn("Equation:", section_child["clean_text"])
        self.assertIn("We consider the following channels:", section_child["clean_text"])
        self.assertIn("The final states are combined statistically.", section_child["clean_text"])

    def test_normalize_display_text_tightens_tex_math_spacing(self) -> None:
        text = (
            r"$1 3 \mathrm { T e V }$ and $9 7. 7 \mathrm { f b } ^ { - 1 }$ and "
            r"$1 0 ^ { - 8 }$ with $\tau \to 3 \mu$"
        )
        normalized = normalize_display_text(text)
        self.assertIn("13 TeV", normalized)
        self.assertIn("97.7 fb^-1", normalized)
        self.assertIn("10^-8", normalized)
        self.assertIn("τ -> 3 μ", normalized)

    def test_normalize_display_text_repairs_mineru_noise_patterns(self) -> None:
        text = (
            r"$5 9 . 7 \mathrm { f b } ^ { - 1 } .$ , respectively; "
            r"$1 0 G e V _ { , }$ and $| \eta | < 2 . 4 ,$ . "
            r"$\mathnormal { J } / \psi$ data in $\boldsymbol { \mathrm { W } }$ boson events. "
            r"$\sigma ( \mathrm { p p }  W + X )$, $\mathrm { B }  \tau + X$, "
            r"$\mathrm { B }  \mathrm { D } _ { \mathrm { s } } ^ { + } + \mathrm { X }$, "
            r"$\sigma ( \mathrm { p p }  \mathrm { D } _ { \mathrm { s } } ^ { + } + \mathrm { X } )$, "
            r"$L m ( \mu ^ { + } \mu ^ { - } \pi ^ { + } ) / p ,$ and "
            r"$\Delta R \ : < \ : 0 . 8$ with the lowest- pT muon."
        )
        normalized = normalize_display_text(text)
        self.assertIn("59.7 fb^-1, respectively", normalized)
        self.assertIn("10 GeV,", normalized)
        self.assertIn("| η | < 2.4", normalized)
        self.assertIn("J/ψ", normalized)
        self.assertNotIn("boldsymbol", normalized)
        self.assertIn("W boson events", normalized)
        self.assertIn("σ (pp -> W + X)", normalized)
        self.assertIn("B -> τ + X", normalized)
        self.assertIn("B -> Ds ^ + + X", normalized)
        self.assertIn("σ (pp -> Ds ^ + + X)", normalized)
        self.assertIn("L m (μ ^ + μ ^ - π ^ +) / p,", normalized)
        self.assertIn("Delta R < 0.8", normalized)
        self.assertIn("lowest-pT", normalized)

    def test_normalize_display_text_keeps_comparison_operators_and_cleans_array_artifacts(self) -> None:
        text = (
            r"The selection requires $| \eta | < 2 . 4$ and $p _ { \mathrm { T } } > 3 . 5 \mathrm { G e V }$. "
            r"The transverse mass is $\begin{array} { r } { m _ { \mathrm { T } } = \sqrt { 2 p _ { \mathrm { T } } ^ { \tau } p _ { \mathrm { T } } ^ { \mathrm { m i s s } } } , } \end{array}$."
        )
        normalized = normalize_display_text(text)
        self.assertIn("| η | < 2.4", normalized)
        self.assertIn("pT > 3.5 GeV", normalized)
        self.assertIn("mT = sqrt", normalized)
        self.assertNotIn("begin array", normalized)
        self.assertNotIn("end array", normalized)

    def test_normalize_display_text_removes_orphaned_citation_phrases_and_double_periods(self) -> None:
        text = "The signal normalization strategy, detailed in Ref., is summarized here. Any muon must have pT > 3.5 GeV . ."
        normalized = normalize_display_text(text)
        self.assertIn("The signal normalization strategy is summarized here.", normalized)
        self.assertNotIn("detailed in,", normalized)
        self.assertNotIn("detailed in Ref.", normalized)
        self.assertNotIn("..", normalized)

    def test_normalize_display_text_removes_control_chars_and_linearizes_literal_frac(self) -> None:
        text = "The exact procedure is detailed in.\x00 The isolation is rI = frac pT (B) pT (B) + sum trkpT."
        normalized = normalize_display_text(text)
        self.assertNotIn("\x00", normalized)
        self.assertNotIn("detailed in.", normalized)
        self.assertNotRegex(normalized, r"\bfrac\b")
        self.assertIn("/", normalized)

    def test_normalize_search_text_handles_mathml_and_latex(self) -> None:
        text = (
            'Rare decay <math><mi>η</mi><mo>→</mo><msup><mi>μ</mi><mo>+</mo></msup></math> '
            'at $\\sqrt{s}$ = 1 3 TeV and 101 fb$^{-1}$.'
        )
        normalized = normalize_search_text(text)
        self.assertIn("eta", normalized)
        self.assertIn("mu", normalized)
        self.assertIn("sqrt", normalized)
        self.assertIn("13", normalized)
        self.assertIn("101", normalized)

    def test_build_search_index_and_query_works_and_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            with _patch_workspace(tmp):
                db.ensure_db()
                with db.connect() as conn:
                    collection_id = upsert_collection(conn, {"name": "cms_rare_decay"})
                    rare_hit = {
                        "metadata": {
                            "control_number": 501,
                            "titles": [{"title": "Observation of the rare decay of the $\\eta$ meson to four muons"}],
                            "abstracts": [{
                                "value": (
                                    'A search for the rare <math><mi>η</mi><mo>→</mo><msup><mi>μ</mi><mo>+</mo></msup>'
                                    "</math> mode with 101 fb$^{-1}$ at $\\sqrt{s}$ = 1 3 TeV."
                                )
                            }],
                            "arxiv_eprints": [{"value": "2501.00001"}],
                            "keywords": [{"value": "rare decay"}],
                        }
                    }
                    higgs_hit = {
                        "metadata": {
                            "control_number": 502,
                            "titles": [{"title": "Search for a Higgs boson decay"}],
                            "abstracts": [{"value": "We study a Higgs final state."}],
                            "arxiv_eprints": [{"value": "2501.00002"}],
                            "keywords": [{"value": "Higgs"}],
                        }
                    }
                    upsert_work_from_hit(conn, collection_id=collection_id, hit=rare_hit)
                    upsert_work_from_hit(conn, collection_id=collection_id, hit=higgs_hit)
                    work_id = conn.execute(
                        "SELECT work_id FROM works WHERE canonical_id = '501'"
                    ).fetchone()[0]

                    bundle_dir = tmp / "bundle_search"
                    bundle_dir.mkdir()
                    (bundle_dir / "paper_full.md").write_text("# Rare decay note\n", encoding="utf-8")
                    (bundle_dir / "paper_content_list.json").write_text(
                        json.dumps(
                            [
                                {"type": "text", "text_level": 1, "text": "Rare decay note", "page_idx": 1},
                                {"type": "text", "text_level": 1, "text": "Abstract", "page_idx": 1},
                                {"type": "text", "text": "A rare decay search with profile likelihood.", "page_idx": 1},
                                {"type": "text", "text_level": 1, "text": "1 Method", "page_idx": 2},
                                {"type": "text", "text": "We fit the signal model with a profile likelihood and templates.", "page_idx": 2},
                                {"type": "text", "text_level": 1, "text": "References", "page_idx": 3},
                                {"type": "text", "text": "[1] Bibliography entry.", "page_idx": 3},
                            ],
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )
                    dest_dir = paths.PARSED_DIR / "cms_rare_decay" / "2501.00001"
                    import_mineru_source(source_path=bundle_dir, dest_dir=dest_dir)
                    materialize_mineru_document(
                        conn,
                        work_id=int(work_id),
                        manifest_path=dest_dir / "manifest.json",
                    )

                    summary = rebuild_search_indices(conn, target="all")
                    conn.commit()

                    self.assertEqual(summary["works"], 2)
                    self.assertGreaterEqual(summary["chunks"], 2)

                    work_results = search_works_bm25(conn, query="eta muons 101 fb 13 TeV", limit=5)
                    self.assertGreaterEqual(len(work_results), 1)
                    self.assertEqual(work_results[0]["work_id"], int(work_id))
                    self.assertIn("eta", work_results[0]["indexed_abstract"].lower())

                    chunk_results = search_chunks_bm25(conn, query="profile likelihood templates", limit=5)
                    self.assertGreaterEqual(len(chunk_results), 1)
                    self.assertIn("profile likelihood", chunk_results[0]["clean_text"].lower())
                    self.assertNotIn("bibliography entry", chunk_results[0]["clean_text"].lower())
                    self.assertNotEqual(chunk_results[0]["chunk_role"], "formula_window")

    def test_build_structure_search_index_and_query_formulas_and_assets(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            with _patch_workspace(tmp):
                db.ensure_db()
                with db.connect() as conn:
                    collection_id = upsert_collection(conn, {"name": "cms_rare_decay"})
                    hit = {
                        "metadata": {
                            "control_number": 503,
                            "titles": [{"title": "Tau branching analysis"}],
                            "abstracts": [{"value": "A branching-fraction analysis with formula and figure support."}],
                            "arxiv_eprints": [{"value": "2501.00004"}],
                        }
                    }
                    upsert_work_from_hit(conn, collection_id=collection_id, hit=hit)
                    work_id = conn.execute(
                        "SELECT work_id FROM works WHERE canonical_id = '503'"
                    ).fetchone()[0]

                    bundle_dir = tmp / "bundle_structure"
                    bundle_dir.mkdir()
                    (bundle_dir / "paper_full.md").write_text("# Tau branching analysis\n", encoding="utf-8")
                    (bundle_dir / "paper_content_list.json").write_text(
                        json.dumps(
                            [
                                {"type": "text", "text_level": 1, "text": "Tau branching analysis", "page_idx": 1},
                                {"type": "text", "text_level": 1, "text": "Abstract", "page_idx": 1},
                                {"type": "text", "text": "We estimate the branching fraction with a simple normalization formula.", "page_idx": 1},
                                {"type": "text", "text_level": 1, "text": "1 Method", "page_idx": 2},
                                {"type": "text", "text": "The branching fraction is constrained by the signal yield and efficiency.", "page_idx": 2},
                                {"type": "equation", "text": r"$$\mathcal{B}(\tau \to 3 \mu) = \frac{N_{\mathrm{sig}}}{\epsilon}$$", "page_idx": 2},
                                {"type": "text", "text": "The normalization uses the detector efficiency and signal count.", "page_idx": 2},
                                {"type": "image", "caption": "Observed limit for tau to 3 mu as a function of efficiency", "image_path": "figures/limit.png", "page_idx": 3},
                            ],
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )
                    dest_dir = paths.PARSED_DIR / "cms_rare_decay" / "2501.00004"
                    import_mineru_source(source_path=bundle_dir, dest_dir=dest_dir)
                    materialize_mineru_document(
                        conn,
                        work_id=int(work_id),
                        manifest_path=dest_dir / "manifest.json",
                    )

                    summary = rebuild_search_indices(conn, target="all")
                    conn.commit()

                    self.assertEqual(summary["formulas"], 1)
                    self.assertEqual(summary["assets"], 1)

                    formula_results = search_formulas_bm25(conn, query="tau mu efficiency signal", limit=5)
                    self.assertGreaterEqual(len(formula_results), 1)
                    self.assertEqual(formula_results[0]["work_id"], int(work_id))
                    self.assertIn(r"\tau", formula_results[0]["raw_latex"])
                    self.assertIn("branching fraction", formula_results[0]["display_context"].lower())

                    asset_results = search_assets_bm25(conn, query="observed limit efficiency tau mu", limit=5)
                    self.assertGreaterEqual(len(asset_results), 1)
                    self.assertEqual(asset_results[0]["work_id"], int(work_id))
                    self.assertIn("observed limit", asset_results[0]["caption"].lower())
                    self.assertIn("efficiency", asset_results[0]["display_context"].lower())

    def test_cli_build_search_index_and_search_bm25(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            with _patch_workspace(tmp):
                db.ensure_db()
                with db.connect() as conn:
                    collection_id = upsert_collection(conn, {"name": "cms_rare_decay"})
                    hit = {
                        "metadata": {
                            "control_number": 601,
                            "titles": [{"title": "Rare decay overview"}],
                            "abstracts": [{"value": "A concise overview of a rare decay analysis."}],
                            "arxiv_eprints": [{"value": "2501.00003"}],
                        }
                    }
                    upsert_work_from_hit(conn, collection_id=collection_id, hit=hit)
                    conn.commit()

                build_out = io.StringIO()
                with contextlib.redirect_stdout(build_out):
                    cli.cmd_build_search_index(SimpleNamespace(target="works"))
                build_payload = json.loads(build_out.getvalue())
                self.assertEqual(build_payload["works"], 1)
                self.assertEqual(build_payload["work_search"], 1)

                search_out = io.StringIO()
                with contextlib.redirect_stdout(search_out):
                    cli.cmd_search_bm25(
                        SimpleNamespace(
                            query="rare decay overview",
                            target="works",
                            collection=None,
                            limit=5,
                        )
                    )
                search_payload = json.loads(search_out.getvalue())
                self.assertEqual(search_payload[0]["canonical_id"], "601")


class TestGraphBuild(unittest.TestCase):
    def test_rebuild_graph_edges_materializes_bibliographic_coupling_and_co_citation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            with _patch_workspace(tmp):
                db.ensure_db()
                with db.connect() as conn:
                    collection_id = upsert_collection(conn, {"name": "cms_rare_decay"})
                    for control_number, title, refs in [
                        (201, "Target A", []),
                        (202, "Target B", []),
                        (101, "Source A", [{"control_number": 201}, {"control_number": 202}]),
                        (102, "Source B", [{"control_number": 201}, {"control_number": 202}]),
                    ]:
                        upsert_work_from_hit(
                            conn,
                            collection_id=collection_id,
                            hit={
                                "metadata": {
                                    "control_number": control_number,
                                    "titles": [{"title": title}],
                                    "references": refs,
                                }
                            },
                        )
                    backfill_unresolved_citations(conn)

                    summary = rebuild_graph_edges(conn, target="all", collection="cms_rare_decay", min_shared=2)
                    conn.commit()

                    self.assertEqual(summary["bibliographic_coupling_edges"], 1)
                    self.assertEqual(summary["co_citation_edges"], 1)

                    bc_row = conn.execute(
                        """
                        SELECT src_work_id, dst_work_id, shared_reference_count, score
                        FROM bibliographic_coupling_edges
                        """
                    ).fetchone()
                    self.assertEqual(int(bc_row["shared_reference_count"]), 2)
                    self.assertAlmostEqual(float(bc_row["score"]), 1.0)

                    cc_row = conn.execute(
                        """
                        SELECT src_work_id, dst_work_id, shared_citer_count, score
                        FROM co_citation_edges
                        """
                    ).fetchone()
                    self.assertEqual(int(cc_row["shared_citer_count"]), 2)
                    self.assertAlmostEqual(float(cc_row["score"]), 1.0)

                    runs = conn.execute(
                        "SELECT COUNT(*) FROM graph_build_runs WHERE status = 'completed'"
                    ).fetchone()[0]
                    self.assertEqual(runs, 2)

                    source_a = conn.execute(
                        "SELECT work_id FROM works WHERE canonical_id = '101'"
                    ).fetchone()[0]
                    target_a = conn.execute(
                        "SELECT work_id FROM works WHERE canonical_id = '201'"
                    ).fetchone()[0]

                    bc_neighbors = graph_neighbors(
                        conn,
                        work_id=int(source_a),
                        edge_kind="bibliographic-coupling",
                        collection="cms_rare_decay",
                        limit=5,
                    )
                    self.assertEqual(len(bc_neighbors), 1)
                    self.assertEqual(bc_neighbors[0]["canonical_id"], "102")
                    self.assertEqual(bc_neighbors[0]["shared_count"], 2)

                    cc_neighbors = graph_neighbors(
                        conn,
                        work_id=int(target_a),
                        edge_kind="co-citation",
                        collection="cms_rare_decay",
                        limit=5,
                    )
                    self.assertEqual(len(cc_neighbors), 1)
                    self.assertEqual(cc_neighbors[0]["canonical_id"], "202")
                    self.assertEqual(cc_neighbors[0]["shared_count"], 2)

    def test_cli_show_graph_reports_neighbors(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            with _patch_workspace(tmp):
                db.ensure_db()
                with db.connect() as conn:
                    collection_id = upsert_collection(conn, {"name": "cms_rare_decay"})
                    for control_number, title, refs in [
                        (301, "Ref target", []),
                        (401, "Graph source A", [{"control_number": 301}]),
                        (402, "Graph source B", [{"control_number": 301}]),
                    ]:
                        upsert_work_from_hit(
                            conn,
                            collection_id=collection_id,
                            hit={
                                "metadata": {
                                    "control_number": control_number,
                                    "titles": [{"title": title}],
                                    "arxiv_eprints": [{"value": f"25{control_number}.00001"}],
                                    "references": refs,
                                }
                            },
                        )
                    backfill_unresolved_citations(conn)
                    rebuild_graph_edges(conn, target="bibliographic-coupling", collection="cms_rare_decay", min_shared=1)
                    conn.commit()

                show_out = io.StringIO()
                with contextlib.redirect_stdout(show_out):
                    cli.cmd_show_graph(
                        SimpleNamespace(
                            work_id=None,
                            id_type="arxiv",
                            id_value="25401.00001",
                            edge_kind="bibliographic-coupling",
                            collection="cms_rare_decay",
                            limit=5,
                        )
                    )
                payload = json.loads(show_out.getvalue())
                self.assertEqual(payload["work"]["canonical_id"], "401")
                self.assertEqual(len(payload["neighbors"]), 1)
                self.assertEqual(payload["neighbors"][0]["canonical_id"], "402")
