from __future__ import annotations

import contextlib
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
from hep_rag_v2.results import ensure_result_schema  # noqa: E402


class CommunitySummaryTests(unittest.TestCase):
    def test_rebuild_and_search_community_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            with _patch_workspace(tmp):
                db.ensure_db()
                with db.connect() as conn:
                    collection_id = upsert_collection(conn, {"name": "default"})
                    work_a, work_b, work_c = _seed_works(conn, collection_id)
                    _seed_community_inputs(
                        conn,
                        collection_id,
                        work_a=work_a,
                        work_b=work_b,
                        work_c=work_c,
                    )

                    scoped = rebuild_community_summaries(conn, collection="default")
                    global_summary = rebuild_community_summaries(conn, collection=None)
                    results = search_community_summaries(
                        conn,
                        query="CMS jet tagging graph neural network",
                        collection="default",
                        limit=4,
                    )

                self.assertEqual(scoped["algorithm"], "weighted_components_v1")
                self.assertGreaterEqual(scoped["total"], 1)
                self.assertGreaterEqual(scoped["edge_count"], 1)
                self.assertGreaterEqual(global_summary["total"], scoped["total"])
                self.assertGreaterEqual(len(results), 1)
                self.assertEqual(results[0]["search_type"], "community")
                self.assertTrue(results[0]["summary_id"].startswith("community_summary:"))
                self.assertGreaterEqual(results[0]["work_count"], 2)
                self.assertGreaterEqual(results[0]["edge_count"], 1)
                self.assertGreaterEqual(len(results[0]["representative_works"]), 1)
                self.assertIn("signal_mix", results[0]["metadata"])


def _seed_works(conn, collection_id: int) -> tuple[int, int, int]:
    hits = [
        {
            "metadata": {
                "control_number": 9101,
                "titles": [{"title": "CMS jet tagging with graph neural networks"}],
                "abstracts": [{"value": "CMS jet tagging measurement with graph neural network classifiers."}],
                "arxiv_eprints": [{"value": "2604.10001"}],
            }
        },
        {
            "metadata": {
                "control_number": 9102,
                "titles": [{"title": "Advances in CMS jet tagging calibration"}],
                "abstracts": [{"value": "CMS reports updated jet tagging measurements and calibration studies."}],
                "arxiv_eprints": [{"value": "2604.10002"}],
            }
        },
        {
            "metadata": {
                "control_number": 9103,
                "titles": [{"title": "ATLAS heavy-ion tracking upgrades"}],
                "abstracts": [{"value": "A separate detector-upgrade study outside the CMS jet tagging cluster."}],
                "arxiv_eprints": [{"value": "2604.10003"}],
            }
        },
    ]
    for hit in hits:
        upsert_work_from_hit(conn, collection_id=collection_id, hit=hit)
    work_a = int(conn.execute("SELECT work_id FROM works WHERE canonical_id = '9101'").fetchone()[0])
    work_b = int(conn.execute("SELECT work_id FROM works WHERE canonical_id = '9102'").fetchone()[0])
    work_c = int(conn.execute("SELECT work_id FROM works WHERE canonical_id = '9103'").fetchone()[0])
    conn.execute(
        "UPDATE works SET year = 2025, citation_count = CASE work_id WHEN ? THEN 8 WHEN ? THEN 6 ELSE 1 END WHERE work_id IN (?, ?, ?)",
        (work_a, work_b, work_a, work_b, work_c),
    )
    return (work_a, work_b, work_c)


def _seed_community_inputs(conn, collection_id: int, *, work_a: int, work_b: int, work_c: int) -> None:
    ensure_result_schema(conn)
    ensure_method_schema(conn)

    conn.execute("INSERT INTO collaborations (name) VALUES ('CMS')")
    conn.execute("INSERT INTO collaborations (name) VALUES ('ATLAS')")
    cms_id = int(conn.execute("SELECT collaboration_id FROM collaborations WHERE name = 'CMS'").fetchone()[0])
    atlas_id = int(conn.execute("SELECT collaboration_id FROM collaborations WHERE name = 'ATLAS'").fetchone()[0])
    conn.executemany(
        "INSERT INTO work_collaborations (work_id, collaboration_id) VALUES (?, ?)",
        [(work_a, cms_id), (work_b, cms_id), (work_c, atlas_id)],
    )

    conn.execute(
        "INSERT INTO topics (source, topic_key, label) VALUES (?, ?, ?)",
        ("manual", "jet_tagging", "jet tagging"),
    )
    conn.execute(
        "INSERT INTO topics (source, topic_key, label) VALUES (?, ?, ?)",
        ("manual", "tracking", "tracking"),
    )
    jet_topic_id = int(
        conn.execute(
            "SELECT topic_id FROM topics WHERE source = ? AND topic_key = ?",
            ("manual", "jet_tagging"),
        ).fetchone()[0]
    )
    tracking_topic_id = int(
        conn.execute(
            "SELECT topic_id FROM topics WHERE source = ? AND topic_key = ?",
            ("manual", "tracking"),
        ).fetchone()[0]
    )
    conn.executemany(
        "INSERT INTO work_topics (work_id, topic_id, score) VALUES (?, ?, ?)",
        [(work_a, jet_topic_id, 0.95), (work_b, jet_topic_id, 0.91), (work_c, tracking_topic_id, 0.84)],
    )

    conn.executemany(
        """
        INSERT INTO result_objects (work_id, collection_id, object_key, label, result_kind, summary_text, status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (work_a, collection_id, "result:measurement:a", "jet tagging measurement", "measurement", "CMS jet tagging measurement", "ready"),
            (work_b, collection_id, "result:measurement:b", "calibration measurement", "measurement", "CMS jet tagging calibration measurement", "ready"),
        ],
    )
    conn.executemany(
        """
        INSERT INTO method_objects (work_id, collection_id, object_key, name, method_family, summary_text, status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (work_a, collection_id, "method:multivariate:a", "graph neural network tagger", "multivariate", "GNN-based jet tagging", "ready"),
            (work_b, collection_id, "method:multivariate:b", "deep jet calibration", "multivariate", "multivariate calibration workflow", "ready"),
        ],
    )

    conn.execute(
        """
        INSERT INTO similarity_edges (src_work_id, dst_work_id, metric, score)
        VALUES (?, ?, ?, ?)
        """,
        (min(work_a, work_b), max(work_a, work_b), "cosine::hash-idf-v1", 0.74),
    )
    conn.commit()


@contextlib.contextmanager
def _patch_workspace(tmp: Path):
    original = paths.workspace_root()
    try:
        paths.set_workspace_root(tmp)
        yield
    finally:
        paths.set_workspace_root(original)


if __name__ == "__main__":
    unittest.main()
