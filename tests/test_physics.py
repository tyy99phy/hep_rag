from __future__ import annotations

import contextlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hep_rag_v2 import db, paths  # noqa: E402
from hep_rag_v2.community import rebuild_community_summaries, search_community_summaries  # noqa: E402
from hep_rag_v2.metadata import upsert_collection, upsert_work_from_hit  # noqa: E402
from hep_rag_v2.methods import ensure_method_schema  # noqa: E402
from hep_rag_v2.ontology import rebuild_ontology_summaries, search_ontology_summaries  # noqa: E402
from hep_rag_v2.pdg import import_pdg_source  # noqa: E402
from hep_rag_v2.physics import build_physics_substrate  # noqa: E402
from hep_rag_v2.results import ensure_result_schema  # noqa: E402


@contextlib.contextmanager
def _patch_workspace(tmp: Path):
    original = paths.workspace_root()
    try:
        paths.set_workspace_root(tmp)
        yield
    finally:
        paths.set_workspace_root(original)


class PhysicsSubstrateTests(unittest.TestCase):
    def test_import_pdg_source_builds_physics_concepts_and_groundings(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            with _patch_workspace(tmp):
                bundle_dir = _make_pdg_bundle(tmp)
                db.ensure_db()
                with db.connect() as conn:
                    collection_id = upsert_collection(conn, {"name": "default"})
                    work_id = _insert_work(
                        conn,
                        collection_id=collection_id,
                        control_number=9301,
                        title="CMS measurement of B meson branching fractions with CKM constraints",
                        abstract="CMS measures B meson branching fractions and constrains the CKM matrix.",
                    )
                    _seed_result_and_chunk(conn, collection_id=collection_id, work_id=work_id)

                    summary = import_pdg_source(
                        conn,
                        source_path=bundle_dir,
                        source_id="pdg-physics",
                        title="PDG 2026",
                    )

                    concept_rows = conn.execute(
                        "SELECT label, concept_kind FROM physics_concepts ORDER BY label"
                    ).fetchall()
                    work_rows = conn.execute(
                        """
                        SELECT pc.label
                        FROM work_physics_groundings wpg
                        JOIN physics_concepts pc ON pc.physics_concept_id = wpg.physics_concept_id
                        WHERE wpg.work_id = ?
                        ORDER BY wpg.confidence DESC, pc.label
                        """,
                        (work_id,),
                    ).fetchall()
                    result_count = int(
                        conn.execute(
                            """
                            SELECT COUNT(*)
                            FROM result_physics_groundings rpg
                            JOIN result_objects ro ON ro.result_object_id = rpg.result_object_id
                            WHERE ro.work_id = ?
                            """,
                            (work_id,),
                        ).fetchone()[0]
                    )
                    chunk_count = int(
                        conn.execute(
                            """
                            SELECT COUNT(*)
                            FROM chunk_physics_groundings cpg
                            JOIN chunks c ON c.chunk_id = cpg.chunk_id
                            WHERE c.work_id = ?
                            """,
                            (work_id,),
                        ).fetchone()[0]
                    )

                labels = [str(row["label"]) for row in concept_rows]
                self.assertEqual(summary["source_id"], "pdg-physics")
                self.assertGreaterEqual(summary["physics"]["concepts"]["total"], 2)
                self.assertTrue(any("B meson" in label or "B Meson" in label for label in labels))
                self.assertTrue(any("CKM Matrix" in label for label in labels))
                self.assertGreaterEqual(len(work_rows), 1)
                self.assertTrue(any("B meson" in str(row["label"]) or "B Meson" in str(row["label"]) or "CKM Matrix" in str(row["label"]) for row in work_rows))
                self.assertGreaterEqual(result_count, 1)
                self.assertGreaterEqual(chunk_count, 1)

    def test_ontology_and_community_summaries_expose_physics_backbone(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            with _patch_workspace(tmp):
                bundle_dir = _make_pdg_bundle(tmp)
                db.ensure_db()
                with db.connect() as conn:
                    collection_id = upsert_collection(conn, {"name": "default"})
                    work_a = _insert_work(
                        conn,
                        collection_id=collection_id,
                        control_number=9302,
                        title="CMS B meson branching-fraction measurement",
                        abstract="CMS reports a B meson branching fraction measurement with CKM interpretation.",
                    )
                    work_b = _insert_work(
                        conn,
                        collection_id=collection_id,
                        control_number=9303,
                        title="Updated CMS study of B meson lifetimes and CKM matrix constraints",
                        abstract="CMS updates B meson lifetime measurements and improves CKM matrix constraints.",
                    )
                    _seed_result_and_chunk(conn, collection_id=collection_id, work_id=work_a)
                    _seed_result_and_chunk(conn, collection_id=collection_id, work_id=work_b)
                    ensure_method_schema(conn)
                    conn.executemany(
                        """
                        INSERT INTO method_objects (work_id, collection_id, object_key, name, method_family, summary_text, status)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            (work_a, collection_id, "method:a", "profile likelihood fit", "statistical_fit", "CKM fit for B mesons", "ready"),
                            (work_b, collection_id, "method:b", "profile likelihood fit", "statistical_fit", "B meson likelihood fit", "ready"),
                        ],
                    )
                    conn.execute("INSERT INTO collaborations (name) VALUES ('CMS')")
                    cms_id = int(conn.execute("SELECT collaboration_id FROM collaborations WHERE name = 'CMS'").fetchone()[0])
                    conn.executemany(
                        "INSERT INTO work_collaborations (work_id, collaboration_id) VALUES (?, ?)",
                        [(work_a, cms_id), (work_b, cms_id)],
                    )
                    conn.execute(
                        """
                        INSERT INTO similarity_edges (src_work_id, dst_work_id, metric, score)
                        VALUES (?, ?, ?, ?)
                        """,
                        (min(work_a, work_b), max(work_a, work_b), "cosine::hash-idf-v1", 0.64),
                    )

                    import_pdg_source(
                        conn,
                        source_path=bundle_dir,
                        source_id="pdg-community",
                        title="PDG 2026",
                    )
                    build_physics_substrate(conn, collection="default", work_ids=[work_a, work_b])
                    ontology_summary = rebuild_ontology_summaries(conn, collection="default")
                    ontology_hits = search_ontology_summaries(
                        conn,
                        query="B mesons CKM matrix",
                        collection="default",
                        limit=6,
                    )
                    community_summary = rebuild_community_summaries(conn, collection="default")
                    community_hits = search_community_summaries(
                        conn,
                        query="CMS B meson CKM measurement",
                        collection="default",
                        limit=4,
                    )

                self.assertIn("physics_concept", ontology_summary["by_kind"])
                self.assertTrue(any(item["facet_kind"] == "physics_concept" for item in ontology_hits))
                self.assertGreaterEqual(community_summary["total"], 1)
                self.assertTrue(community_hits)
                self.assertTrue(
                    any(
                        "B meson" in str(label) or "B Meson" in str(label) or "CKM Matrix" in str(label)
                        for item in community_hits
                        for label in (item.get("metadata", {}) or {}).get("physics_concepts", [])
                    )
                )


def _make_pdg_bundle(tmp: Path) -> Path:
    bundle_dir = tmp / "pdg_bundle"
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "pdg_full.md").write_text("# PDG\n", encoding="utf-8")
    (bundle_dir / "pdg_content_list.json").write_text(
        json.dumps(
            [
                {"type": "text", "text_level": 1, "text": "PDG Review of B Mesons", "page_idx": 1},
                {"type": "text", "text_level": 1, "text": "1 Properties", "page_idx": 1},
                {"type": "text", "text": "The B meson masses and lifetimes are precision observables.", "page_idx": 1},
                {"type": "text", "text_level": 1, "text": "2 CKM Matrix", "page_idx": 2},
                {"type": "text", "text": "The CKM matrix governs quark flavor transitions in B meson decays.", "page_idx": 2},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return bundle_dir


def _insert_work(conn, *, collection_id: int, control_number: int, title: str, abstract: str) -> int:
    hit = {
        "metadata": {
            "control_number": control_number,
            "titles": [{"title": title}],
            "abstracts": [{"value": abstract}],
            "publication_info": [{"year": 2026}],
        }
    }
    upsert_work_from_hit(conn, collection_id=collection_id, hit=hit)
    row = conn.execute(
        "SELECT work_id FROM works WHERE canonical_source = 'inspire' AND canonical_id = ?",
        (str(control_number),),
    ).fetchone()
    assert row is not None
    return int(row["work_id"])


def _seed_result_and_chunk(conn, *, collection_id: int, work_id: int) -> None:
    ensure_result_schema(conn)
    conn.execute(
        """
        INSERT INTO result_objects (work_id, collection_id, object_key, label, result_kind, summary_text, status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            work_id,
            collection_id,
            f"result:{work_id}",
            "B meson branching fraction result",
            "measurement",
            "Measurement of B meson branching fractions with CKM interpretation.",
            "ready",
        ),
    )
    conn.execute(
        """
        INSERT INTO documents (work_id, parser_name, parser_version, parse_status, parsed_dir, manifest_path)
        VALUES (?, 'test', '1', 'materialized', 'tmp', 'tmp/manifest.json')
        ON CONFLICT(work_id) DO UPDATE SET parse_status = excluded.parse_status
        """,
        (work_id,),
    )
    document_id = int(conn.execute("SELECT document_id FROM documents WHERE work_id = ?", (work_id,)).fetchone()[0])
    conn.execute(
        """
        INSERT INTO chunks (
          work_id, document_id, section_id, block_start_id, block_end_id, chunk_role,
          page_hint, section_hint, text, raw_text, clean_text
        ) VALUES (?, ?, NULL, NULL, NULL, 'section_child', '1', 'B meson analysis', ?, ?, ?)
        """,
        (
            work_id,
            document_id,
            "The B meson analysis measures branching fractions and constrains the CKM matrix.",
            "The B meson analysis measures branching fractions and constrains the CKM matrix.",
            "The B meson analysis measures branching fractions and constrains the CKM matrix.",
        ),
    )


if __name__ == "__main__":
    unittest.main()
