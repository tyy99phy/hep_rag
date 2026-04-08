from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hep_rag_v2.db import connect, ensure_db
from hep_rag_v2 import paths
from hep_rag_v2.metadata import upsert_collection, upsert_work_from_hit


def _make_hit(
    control_number: int,
    title: str,
    *,
    doc_types: list[str] | None = None,
    year: int | None = None,
    collaborations: list[str] | None = None,
    arxiv_id: str | None = None,
    report_numbers: list[str] | None = None,
) -> dict:
    metadata: dict = {
        "control_number": control_number,
        "titles": [{"title": title}],
        "document_type": list(doc_types or ["article"]),
    }
    if year is not None:
        metadata["earliest_date"] = f"{year}-01-01"
        metadata["publication_info"] = [{"year": year}]
    if collaborations:
        metadata["collaborations"] = [{"value": v} for v in collaborations]
    if arxiv_id:
        metadata["arxiv_eprints"] = [{"value": arxiv_id}]
    if report_numbers:
        metadata["report_numbers"] = [{"value": v} for v in report_numbers]
    return {"metadata": metadata}


class WorkFamilyTests(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self._original_root = paths.workspace_root()
        paths.set_workspace_root(Path(self._td.name))
        ensure_db()

    def tearDown(self) -> None:
        paths.set_workspace_root(self._original_root)
        self._td.cleanup()

    def test_same_title_different_doctype_merged_into_one_family(self) -> None:
        with connect() as conn:
            cid = upsert_collection(conn, {"name": "test", "label": "test"})

            hit_note = _make_hit(
                100,
                "Measurements of same-sign WW in association with two jets at 13 TeV",
                doc_types=["note"],
                year=2020,
                collaborations=["CMS"],
                report_numbers=["CMS-SMP-20-006"],
            )
            hit_article = _make_hit(
                200,
                "Measurements of same-sign WW in association with two jets at 13 TeV",
                doc_types=["article"],
                year=2020,
                collaborations=["CMS"],
                report_numbers=["CMS-SMP-20-006"],
                arxiv_id="2005.01173",
            )

            upsert_work_from_hit(conn, collection_id=cid, hit=hit_note)
            upsert_work_from_hit(conn, collection_id=cid, hit=hit_article)

            families = conn.execute("SELECT family_id FROM work_family_members").fetchall()
            family_ids = {int(row["family_id"]) for row in families}
            self.assertEqual(len(family_ids), 1, "Both works should belong to the same family")

    def test_different_titles_stay_in_separate_families(self) -> None:
        with connect() as conn:
            cid = upsert_collection(conn, {"name": "test", "label": "test"})

            hit_a = _make_hit(
                300,
                "Search for supersymmetry in hadronic final states",
                doc_types=["article"],
                year=2022,
                collaborations=["CMS"],
            )
            hit_b = _make_hit(
                400,
                "Observation of electroweak production of same-sign W boson pairs",
                doc_types=["article"],
                year=2022,
                collaborations=["CMS"],
            )

            upsert_work_from_hit(conn, collection_id=cid, hit=hit_a)
            upsert_work_from_hit(conn, collection_id=cid, hit=hit_b)

            families = conn.execute("SELECT family_id FROM work_family_members").fetchall()
            family_ids = {int(row["family_id"]) for row in families}
            self.assertEqual(len(family_ids), 2, "Different titles should produce separate families")

    def test_title_normalized_column_populated_on_insert(self) -> None:
        with connect() as conn:
            cid = upsert_collection(conn, {"name": "test", "label": "test"})

            hit = _make_hit(500, "CMS benchmark study 0: same-sign W boson scattering")
            upsert_work_from_hit(conn, collection_id=cid, hit=hit)

            row = conn.execute("SELECT title_normalized FROM works WHERE canonical_id = '500'").fetchone()
            self.assertIsNotNone(row)
            self.assertTrue(len(row["title_normalized"]) > 0)
            self.assertIn("same sign", row["title_normalized"])


if __name__ == "__main__":
    unittest.main()
