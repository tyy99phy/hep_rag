from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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
                    with mock.patch(
                        "hep_rag_v2.pipeline.build_work_structures",
                        return_value={"processed": 2, "ready": 2, "flagged": 0, "review_relaxed": 0, "items": []},
                    ):
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
                self.assertEqual(summary["structure"]["processed"], 2)
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
                self.assertNotIn("structure", dirty)
                self.assertEqual(dirty["search"], 2)
                self.assertEqual(dirty["vectors"], 2)
                self.assertEqual(dirty["graph"], 2)
        finally:
            paths.set_workspace_root(original_root)

    def test_sync_thinking_engine_extractions_runs_structure_before_downstream_lanes(self) -> None:
        call_order: list[str] = []

        def _record(name: str, payload: dict[str, int]) -> mock.Mock:
            return mock.Mock(side_effect=lambda *args, **kwargs: call_order.append(name) or payload)

        with db.connect() as conn, \
            mock.patch("hep_rag_v2.pipeline.build_work_structures", _record("structure", {"processed": 1, "ready": 1})), \
            mock.patch("hep_rag_v2.pipeline.build_result_objects", _record("results", {"ready": 1})), \
            mock.patch("hep_rag_v2.pipeline.build_method_objects", _record("methods", {"ready": 1})), \
            mock.patch("hep_rag_v2.pipeline.build_transfer_candidates", _record("transfer", {"ready": 1})), \
            mock.patch("hep_rag_v2.pipeline.clear_dirty_work_ids") as clear_dirty:
            summary = _sync_thinking_engine_extractions(conn, collection_name="default", work_ids=[101])

        self.assertEqual(call_order, ["structure", "results", "methods", "transfer"])
        self.assertEqual(summary["structure"]["ready"], 1)
        clear_dirty.assert_any_call(conn, lane="structure", collection="default", work_ids=[101])



    def test_sync_thinking_engine_extractions_uses_structure_capsules_as_upstream_truth(self) -> None:
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
                        control_number=451,
                        title="Capsule seeded work A",
                        abstract="Narrative abstract without extraction keywords.",
                    )
                    second = self._insert_hit(
                        conn,
                        collection_id=collection_id,
                        control_number=452,
                        title="Capsule seeded work B",
                        abstract="Another narrative abstract without extraction keywords.",
                    )
                    _queue_derived_maintenance(
                        conn,
                        collection_id=collection_id,
                        collection_name="default",
                        work_ids=[first, second],
                        reason="unit_test",
                    )
                    conn.execute(
                        """
                        INSERT INTO work_capsules (
                          work_id, collection_id, profile, builder, is_review, status, capsule_text,
                          result_signature_json, method_signature_json, anomaly_code, anomaly_detail
                        ) VALUES (?, ?, 'default', 'test', 0, 'ready', ?, ?, ?, NULL, NULL)
                        ON CONFLICT(work_id) DO UPDATE SET
                          status = excluded.status,
                          capsule_text = excluded.capsule_text,
                          result_signature_json = excluded.result_signature_json,
                          method_signature_json = excluded.method_signature_json,
                          updated_at = CURRENT_TIMESTAMP
                        """,
                        (
                            first,
                            collection_id,
                            'capsule A',
                            json.dumps([{"kind": "measurement", "label": "measurement", "evidence": "capsule result A"}]),
                            json.dumps([{"kind": "statistical_fit", "label": "profile likelihood", "evidence": "capsule method shared"}]),
                        ),
                    )
                    conn.execute(
                        """
                        INSERT INTO work_capsules (
                          work_id, collection_id, profile, builder, is_review, status, capsule_text,
                          result_signature_json, method_signature_json, anomaly_code, anomaly_detail
                        ) VALUES (?, ?, 'default', 'test', 0, 'ready', ?, ?, ?, NULL, NULL)
                        ON CONFLICT(work_id) DO UPDATE SET
                          status = excluded.status,
                          capsule_text = excluded.capsule_text,
                          result_signature_json = excluded.result_signature_json,
                          method_signature_json = excluded.method_signature_json,
                          updated_at = CURRENT_TIMESTAMP
                        """,
                        (
                            second,
                            collection_id,
                            'capsule B',
                            json.dumps([{"kind": "upper_limit", "label": "upper limit", "evidence": "capsule result B"}]),
                            json.dumps([{"kind": "statistical_fit", "label": "profile likelihood", "evidence": "capsule method shared"}]),
                        ),
                    )

                    with mock.patch(
                        "hep_rag_v2.pipeline.build_work_structures",
                        return_value={"processed": 2, "ready": 2, "flagged": 0, "review_relaxed": 0, "items": []},
                    ):
                        summary = _sync_thinking_engine_extractions(
                            conn,
                            collection_name="default",
                            work_ids=[first, second],
                        )
                    result_rows = conn.execute(
                        "SELECT work_id, status, summary_text, signature_json FROM result_objects ORDER BY work_id"
                    ).fetchall()
                    method_rows = conn.execute(
                        "SELECT work_id, status, summary_text, signature_json FROM method_objects ORDER BY work_id"
                    ).fetchall()
                    transfer_rows = conn.execute(
                        "SELECT mo.work_id AS source_work_id, tc.target_work_id, tc.status FROM transfer_candidates tc JOIN method_objects mo ON mo.method_object_id = tc.source_method_object_id ORDER BY source_work_id, tc.target_work_id"
                    ).fetchall()

                self.assertEqual(summary["results"]["ready"], 2)
                self.assertEqual(summary["methods"]["ready"], 2)
                self.assertTrue(all(row["status"] == "ready" for row in result_rows))
                self.assertTrue(all("source=structure_capsule" in str(row["summary_text"]) for row in result_rows))
                self.assertEqual([json.loads(str(row["signature_json"]))[0]["label"] for row in result_rows], ["measurement", "upper limit"])
                self.assertTrue(all(row["status"] == "ready" for row in method_rows))
                self.assertTrue(all("source=structure_capsule" in str(row["summary_text"]) for row in method_rows))
                self.assertTrue(all(json.loads(str(row["signature_json"]))[0]["label"] == "profile likelihood" for row in method_rows))
                self.assertGreaterEqual(len(transfer_rows), 2)
                self.assertTrue(all(row["status"] == "ready" for row in transfer_rows))
        finally:
            paths.set_workspace_root(original_root)

    def test_sync_thinking_engine_extractions_prefers_structure_payload_even_for_heuristic_builder(self) -> None:
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
                        control_number=461,
                        title="Heuristic capsule work A",
                        abstract="Narrative abstract without extraction keywords.",
                    )
                    second = self._insert_hit(
                        conn,
                        collection_id=collection_id,
                        control_number=462,
                        title="Heuristic capsule work B",
                        abstract="Another narrative abstract without extraction keywords.",
                    )
                    _queue_derived_maintenance(
                        conn,
                        collection_id=collection_id,
                        collection_name="default",
                        work_ids=[first, second],
                        reason="unit_test",
                    )
                    for work_id, capsule_text, result_label in (
                        (first, "heuristic capsule A", "measurement"),
                        (second, "heuristic capsule B", "upper limit"),
                    ):
                        conn.execute(
                            """
                            INSERT INTO work_capsules (
                              work_id, collection_id, profile, builder, is_review, status, capsule_text,
                              result_signature_json, method_signature_json, anomaly_code, anomaly_detail
                            ) VALUES (?, ?, 'default', 'heuristic-v1', 0, 'ready', ?, ?, ?, NULL, NULL)
                            ON CONFLICT(work_id) DO UPDATE SET
                              builder = excluded.builder,
                              status = excluded.status,
                              capsule_text = excluded.capsule_text,
                              result_signature_json = excluded.result_signature_json,
                              method_signature_json = excluded.method_signature_json,
                              updated_at = CURRENT_TIMESTAMP
                            """,
                            (
                                work_id,
                                collection_id,
                                capsule_text,
                                json.dumps([{"kind": result_label.replace(" ", "_"), "label": result_label, "evidence": f"{capsule_text} result"}]),
                                json.dumps([{"kind": "statistical_fit", "label": "profile likelihood", "evidence": f"{capsule_text} method"}]),
                            ),
                        )

                    with mock.patch(
                        "hep_rag_v2.pipeline.build_work_structures",
                        return_value={"processed": 2, "ready": 2, "flagged": 0, "review_relaxed": 0, "items": []},
                    ):
                        summary = _sync_thinking_engine_extractions(
                            conn,
                            collection_name="default",
                            work_ids=[first, second],
                        )
                    result_rows = conn.execute(
                        "SELECT work_id, summary_text, signature_json FROM result_objects ORDER BY work_id"
                    ).fetchall()
                    method_rows = conn.execute(
                        "SELECT work_id, summary_text, signature_json FROM method_objects ORDER BY work_id"
                    ).fetchall()
                    transfer_rows = conn.execute(
                        "SELECT mo.work_id AS source_work_id, tc.target_work_id, tc.status FROM transfer_candidates tc JOIN method_objects mo ON mo.method_object_id = tc.source_method_object_id ORDER BY source_work_id, tc.target_work_id"
                    ).fetchall()

                self.assertEqual(summary["results"]["ready"], 2)
                self.assertEqual(summary["methods"]["ready"], 2)
                self.assertEqual(summary["transfer"]["ready"], 2)
                self.assertEqual([json.loads(str(row["signature_json"]))[0]["label"] for row in result_rows], ["measurement", "upper limit"])
                self.assertTrue(all("source=structure_capsule" in str(row["summary_text"]) for row in result_rows))
                self.assertTrue(all(json.loads(str(row["signature_json"]))[0]["label"] == "profile likelihood" for row in method_rows))
                self.assertTrue(all("source=structure_capsule" in str(row["summary_text"]) for row in method_rows))
                self.assertGreaterEqual(len(transfer_rows), 2)
                self.assertTrue(all(row["status"] == "ready" for row in transfer_rows))
        finally:
            paths.set_workspace_root(original_root)

    def test_sync_thinking_engine_extractions_propagates_structure_needs_review_status(self) -> None:
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
                        control_number=453,
                        title="No extraction signatures",
                        abstract="Narrative abstract without measurement or method markers.",
                    )
                    _queue_derived_maintenance(
                        conn,
                        collection_id=collection_id,
                        collection_name="default",
                        work_ids=[work_id],
                        reason="unit_test",
                    )
                    conn.execute(
                        """
                        INSERT INTO work_capsules (
                          work_id, collection_id, profile, builder, is_review, status, capsule_text,
                          result_signature_json, method_signature_json, anomaly_code, anomaly_detail
                        ) VALUES (?, ?, 'default', 'test', 0, 'needs_attention', ?, '[]', '[]', 'missing_required_signatures', 'missing: result, method')
                        ON CONFLICT(work_id) DO UPDATE SET
                          status = excluded.status,
                          capsule_text = excluded.capsule_text,
                          result_signature_json = excluded.result_signature_json,
                          method_signature_json = excluded.method_signature_json,
                          anomaly_code = excluded.anomaly_code,
                          anomaly_detail = excluded.anomaly_detail,
                          updated_at = CURRENT_TIMESTAMP
                        """,
                        (work_id, collection_id, 'capsule review required'),
                    )

                    summary = _sync_thinking_engine_extractions(
                        conn,
                        collection_name="default",
                        work_ids=[work_id],
                    )
                    result_row = conn.execute(
                        "SELECT status FROM result_objects WHERE work_id = ?",
                        (work_id,),
                    ).fetchone()
                    method_row = conn.execute(
                        "SELECT status FROM method_objects WHERE work_id = ?",
                        (work_id,),
                    ).fetchone()

                self.assertEqual(summary["results"]["needs_review"], 1)
                self.assertEqual(summary["methods"]["needs_review"], 1)
                self.assertEqual(summary["transfer"]["needs_review"], 1)
                self.assertEqual(result_row["status"], "needs_review")
                self.assertEqual(method_row["status"], "needs_review")
        finally:
            paths.set_workspace_root(original_root)


if __name__ == "__main__":
    unittest.main()
