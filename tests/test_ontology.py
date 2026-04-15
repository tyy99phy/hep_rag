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
from hep_rag_v2.metadata import upsert_collection, upsert_work_from_hit  # noqa: E402
from hep_rag_v2.methods import ensure_method_schema  # noqa: E402
from hep_rag_v2.ontology import rebuild_ontology_summaries, search_ontology_summaries  # noqa: E402
from hep_rag_v2.results import ensure_result_schema  # noqa: E402


class OntologySummaryTests(unittest.TestCase):
    def test_rebuild_and_search_ontology_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            with _patch_workspace(tmp):
                db.ensure_db()
                with db.connect() as conn:
                    collection_id = upsert_collection(conn, {"name": "default"})
                    work_a, work_b = _seed_works(conn, collection_id)
                    _seed_ontology_inputs(conn, collection_id, work_a=work_a, work_b=work_b)

                    scoped = rebuild_ontology_summaries(conn, collection="default")
                    global_summary = rebuild_ontology_summaries(conn, collection=None)
                    results = search_ontology_summaries(
                        conn,
                        query="CMS jet tagging latest results",
                        collection="default",
                        limit=6,
                    )

                self.assertEqual(scoped["by_kind"]["collaboration"], 1)
                self.assertEqual(scoped["by_kind"]["topic"], 1)
                self.assertEqual(scoped["by_kind"]["result_kind"], 1)
                self.assertEqual(scoped["by_kind"]["method_family"], 1)
                self.assertGreaterEqual(global_summary["total"], scoped["total"])
                self.assertGreaterEqual(len(results), 2)
                self.assertTrue(any(item["facet_kind"] == "collaboration" for item in results))
                self.assertTrue(any(item["facet_kind"] == "topic" for item in results))
                self.assertEqual(results[0]["search_type"], "ontology")
                self.assertTrue(results[0]["summary_id"].startswith("ontology_summary:"))
                self.assertGreaterEqual(len(results[0]["representative_works"]), 1)


def _seed_works(conn, collection_id: int) -> tuple[int, int]:
    hit_a = {
        "metadata": {
            "control_number": 9001,
            "titles": [{"title": "CMS jet tagging with graph neural networks"}],
            "abstracts": [{"value": "CMS jet tagging measurement with graph neural network classifiers."}],
            "arxiv_eprints": [{"value": "2604.00001"}],
        }
    }
    hit_b = {
        "metadata": {
            "control_number": 9002,
            "titles": [{"title": "Advances in CMS jet tagging calibration"}],
            "abstracts": [{"value": "CMS reports updated jet tagging measurements and calibration studies."}],
            "arxiv_eprints": [{"value": "2604.00002"}],
        }
    }
    upsert_work_from_hit(conn, collection_id=collection_id, hit=hit_a)
    upsert_work_from_hit(conn, collection_id=collection_id, hit=hit_b)
    work_a = int(conn.execute("SELECT work_id FROM works WHERE canonical_id = '9001'").fetchone()[0])
    work_b = int(conn.execute("SELECT work_id FROM works WHERE canonical_id = '9002'").fetchone()[0])
    conn.execute("UPDATE works SET year = 2025 WHERE work_id IN (?, ?)", (work_a, work_b))
    return (work_a, work_b)


def _seed_ontology_inputs(conn, collection_id: int, *, work_a: int, work_b: int) -> None:
    ensure_result_schema(conn)
    ensure_method_schema(conn)

    conn.execute("INSERT INTO collaborations (name) VALUES ('CMS')")
    collaboration_id = int(conn.execute("SELECT collaboration_id FROM collaborations WHERE name = 'CMS'").fetchone()[0])
    conn.executemany(
        "INSERT INTO work_collaborations (work_id, collaboration_id) VALUES (?, ?)",
        [(work_a, collaboration_id), (work_b, collaboration_id)],
    )

    conn.execute(
        "INSERT INTO topics (source, topic_key, label) VALUES (?, ?, ?)",
        ("manual", "jet_tagging", "jet tagging"),
    )
    topic_id = int(
        conn.execute(
            "SELECT topic_id FROM topics WHERE source = ? AND topic_key = ?",
            ("manual", "jet_tagging"),
        ).fetchone()[0]
    )
    conn.executemany(
        "INSERT INTO work_topics (work_id, topic_id, score) VALUES (?, ?, ?)",
        [(work_a, topic_id, 0.95), (work_b, topic_id, 0.91)],
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
