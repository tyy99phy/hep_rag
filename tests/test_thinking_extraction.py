from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hep_rag_v2 import db, paths
from hep_rag_v2.config import default_config, runtime_collection_config
from hep_rag_v2.metadata import upsert_collection, upsert_work_from_hit
from hep_rag_v2.pipeline import _queue_derived_maintenance, _sync_thinking_engine_extractions


class ThinkingExtractionTests(unittest.TestCase):
    def _insert_hit(self, conn, *, collection_id: int, control_number: int, title: str, abstract: str) -> int:
        hit = {
            "metadata": {
                "control_number": control_number,
                "titles": [{"title": title}],
                "abstracts": [{"value": abstract}],
                "publication_info": [{"year": 2025}],
                "document_type": ["article"],
            }
        }
        upsert_work_from_hit(conn, collection_id=collection_id, hit=hit)
        row = conn.execute(
            "SELECT work_id FROM works WHERE canonical_source = 'inspire' AND canonical_id = ?",
            (str(control_number),),
        ).fetchone()
        self.assertIsNotNone(row)
        return int(row["work_id"])

    def test_queue_derived_maintenance_adds_thinking_engine_lanes(self) -> None:
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
                    work_id = self._insert_hit(
                        conn,
                        collection_id=collection_id,
                        control_number=301,
                        title="Likelihood measurement",
                        abstract="We measure the cross section with a profile likelihood fit.",
                    )
                    summary = _queue_derived_maintenance(
                        conn,
                        collection_id=collection_id,
                        collection_name="default",
                        work_ids=[work_id],
                        reason="unit_test",
                    )

                self.assertEqual(summary["queued"], 7)
                self.assertEqual(summary["dirty"]["results"], 1)
                self.assertEqual(summary["dirty"]["methods"], 1)
                self.assertEqual(summary["dirty"]["transfer"], 1)
                self.assertEqual(summary["dirty"]["structure"], 1)
        finally:
            paths.set_workspace_root(original_root)

    def test_sync_thinking_engine_extractions_supports_metadata_only_and_transfer_candidates(self) -> None:
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
                    first = self._insert_hit(
                        conn,
                        collection_id=collection_id,
                        control_number=401,
                        title="Rare decay measurement",
                        abstract="We measure the branching fraction with a profile likelihood fit and set a 95% CL upper limit.",
                    )
                    second = self._insert_hit(
                        conn,
                        collection_id=collection_id,
                        control_number=402,
                        title="Control-region study",
                        abstract="A profile likelihood fit and control region strategy improve the cross section determination.",
                    )
                    _queue_derived_maintenance(
                        conn,
                        collection_id=collection_id,
                        collection_name="default",
                        work_ids=[first, second],
                        reason="unit_test",
                    )
                    summary = _sync_thinking_engine_extractions(
                        conn,
                        collection_name="default",
                        work_ids=[first, second],
                    )
                    result_rows = conn.execute(
                        """
                        SELECT work_id, status, summary_text, signature_json, evidence_json
                        FROM result_objects
                        ORDER BY work_id
                        """
                    ).fetchall()
                    method_rows = conn.execute(
                        """
                        SELECT work_id, status, summary_text, signature_json
                        FROM method_objects
                        ORDER BY work_id
                        """
                    ).fetchall()
                    transfer_rows = conn.execute(
                        """
                        SELECT
                          mo.work_id AS source_work_id,
                          tc.target_work_id,
                          tc.target_context_json,
                          json_extract(tc.target_context_json, '$.shared_method_label') AS shared_method_label,
                          tc.status
                        FROM transfer_candidates tc
                        LEFT JOIN method_objects mo ON mo.method_object_id = tc.source_method_object_id
                        ORDER BY source_work_id, tc.target_work_id
                        """
                    ).fetchall()
                    transfer_edge_rows = conn.execute(
                        """
                        SELECT evidence_json
                        FROM transfer_edges
                        ORDER BY transfer_edge_id
                        """
                    ).fetchall()
                    dirty_rows = conn.execute(
                        "SELECT lane, COUNT(*) AS n FROM dirty_objects GROUP BY lane ORDER BY lane"
                    ).fetchall()

                dirty = {str(row["lane"]): int(row["n"]) for row in dirty_rows}
                self.assertEqual(summary["results"]["ready"], 2)
                self.assertEqual(summary["methods"]["ready"], 2)
                self.assertEqual(len(result_rows), 2)
                self.assertTrue(all(row["status"] == "ready" for row in result_rows))
                self.assertTrue(all("metadata_only" in str(row["summary_text"]) for row in result_rows))
                result_signatures = [json.loads(str(row["signature_json"])) for row in result_rows]
                result_evidence = [json.loads(str(row["evidence_json"])) for row in result_rows]
                self.assertTrue(all(items[0]["contract_version"] == "v1" for items in result_signatures))
                self.assertTrue(all(items[0]["object_type"] == "result_signature" for items in result_signatures))
                self.assertTrue(all(items[0]["source_kind"] == "extraction" for items in result_signatures))
                self.assertTrue(all(items[0]["status"] == "ready" for items in result_signatures))
                self.assertTrue(all(items[0]["source_refs"] for items in result_signatures))
                self.assertTrue(all(items[0]["derivation"] == "normalized" for items in result_signatures))
                self.assertTrue(all(items[0]["evidence_bundle"]["object_type"] == "evidence_bundle" for items in result_signatures))
                self.assertTrue(all(items[0]["object_type"] == "evidence_bundle" for items in result_evidence))
                self.assertEqual(len(method_rows), 2)
                self.assertTrue(all(row["status"] == "ready" for row in method_rows))
                method_signatures = [json.loads(str(row["signature_json"])) for row in method_rows]
                self.assertTrue(all(items[0]["contract_version"] == "v1" for items in method_signatures))
                self.assertTrue(all(items[0]["object_type"] == "method_signature" for items in method_signatures))
                self.assertTrue(all(items[0]["source_kind"] == "extraction" for items in method_signatures))
                self.assertTrue(all(items[0]["status"] == "ready" for items in method_signatures))
                self.assertTrue(all(items[0]["source_refs"] for items in method_signatures))
                self.assertTrue(all(items[0]["derivation"] == "normalized" for items in method_signatures))
                self.assertTrue(all(items[0]["evidence_bundle"]["object_type"] == "evidence_bundle" for items in method_signatures))
                self.assertGreaterEqual(len(transfer_rows), 2)
                self.assertTrue(all(row["shared_method_label"] == "profile likelihood" for row in transfer_rows))
                work_capsules = [json.loads(str(row["target_context_json"])) for row in transfer_rows]
                self.assertTrue(all(item["contract_version"] == "v1" for item in work_capsules))
                self.assertTrue(all(item["object_type"] == "work_capsule" for item in work_capsules))
                self.assertTrue(all(item["source_kind"] == "extraction" for item in work_capsules))
                self.assertTrue(all(item["status"] == "ready" for item in work_capsules))
                self.assertTrue(all(item["source_refs"] for item in work_capsules))
                self.assertTrue(all(item["derivation"] == "aggregated" for item in work_capsules))
                self.assertTrue(all(item["work_id"] == row["target_work_id"] for item, row in zip(work_capsules, transfer_rows)))
                self.assertTrue(all(item["title"] for item in work_capsules))
                self.assertTrue(all(item["trace_step"]["object_type"] == "trace_step" for item in work_capsules))
                transfer_evidence = [json.loads(str(row["evidence_json"])) for row in transfer_edge_rows]
                self.assertTrue(all(items[0]["object_type"] == "evidence_bundle" for items in transfer_evidence))
                self.assertNotIn("results", dirty)
                self.assertNotIn("methods", dirty)
                self.assertNotIn("transfer", dirty)
                self.assertEqual(dirty["search"], 2)
                self.assertEqual(dirty["vectors"], 2)
                self.assertEqual(dirty["graph"], 2)
                self.assertEqual(dirty["structure"], 2)
        finally:
            paths.set_workspace_root(original_root)


if __name__ == "__main__":
    unittest.main()
