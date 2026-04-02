from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hep_rag_v2 import cli, db, paths
from hep_rag_v2.config import default_config, resolve_embedding_profile
from hep_rag_v2.metadata import upsert_collection, upsert_work_from_hit
from hep_rag_v2.pdg import import_pdg_source
from hep_rag_v2.structure import build_work_structures


@contextlib.contextmanager
def _patch_workspace(tmp: Path):
    original = paths.workspace_root()
    try:
        paths.set_workspace_root(tmp)
        yield
    finally:
        paths.set_workspace_root(original)


class StructurePipelineTests(unittest.TestCase):
    def _insert_hit(self, conn, *, collection_id: int, control_number: int, title: str, abstract: str, document_type: list[str] | None = None) -> int:
        hit = {
            "metadata": {
                "control_number": control_number,
                "titles": [{"title": title}],
                "abstracts": [{"value": abstract}],
                "publication_info": [{"year": 2025}],
                "document_type": document_type or ["article"],
            }
        }
        upsert_work_from_hit(conn, collection_id=collection_id, hit=hit)
        row = conn.execute(
            "SELECT work_id FROM works WHERE canonical_source = 'inspire' AND canonical_id = ?",
            (str(control_number),),
        ).fetchone()
        self.assertIsNotNone(row)
        return int(row["work_id"])

    def test_build_work_structures_extracts_required_signatures_for_non_review_work(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            with _patch_workspace(tmp):
                db.ensure_db()
                with db.connect() as conn:
                    collection_id = upsert_collection(conn, {"name": "default", "label": "Default"})
                    work_id = self._insert_hit(
                        conn,
                        collection_id=collection_id,
                        control_number=101,
                        title="Rare decay measurement",
                        abstract="We measure the branching fraction with a profile likelihood fit and set a 95% CL upper limit.",
                    )

                    summary = build_work_structures(conn, work_ids=[work_id], collection="default")
                    row = conn.execute(
                        "SELECT status, result_signature_json, method_signature_json, anomaly_code FROM work_capsules WHERE work_id = ?",
                        (work_id,),
                    ).fetchone()

                self.assertEqual(summary["processed"], 1)
                self.assertEqual(summary["ready"], 1)
                self.assertIsNotNone(row)
                self.assertEqual(row["status"], "ready")
                self.assertIsNone(row["anomaly_code"])
                result_payload = json.loads(row["result_signature_json"])
                method_payload = json.loads(row["method_signature_json"])
                self.assertTrue(any(item["kind"] in {"measurement", "upper_limit"} for item in result_payload))
                self.assertTrue(any(item["kind"] == "statistical_fit" for item in method_payload))

    def test_build_work_structures_marks_missing_required_signatures(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            with _patch_workspace(tmp):
                db.ensure_db()
                with db.connect() as conn:
                    collection_id = upsert_collection(conn, {"name": "default", "label": "Default"})
                    work_id = self._insert_hit(
                        conn,
                        collection_id=collection_id,
                        control_number=102,
                        title="Phenomenology note",
                        abstract="We discuss phenomenological implications for dark matter couplings.",
                    )

                    summary = build_work_structures(conn, work_ids=[work_id], collection="default")
                    row = conn.execute(
                        "SELECT status, anomaly_code, anomaly_detail FROM work_capsules WHERE work_id = ?",
                        (work_id,),
                    ).fetchone()

                self.assertEqual(summary["processed"], 1)
                self.assertEqual(summary["flagged"], 1)
                self.assertIsNotNone(row)
                self.assertEqual(row["status"], "needs_attention")
                self.assertEqual(row["anomaly_code"], "missing_required_signatures")
                self.assertIn("result", str(row["anomaly_detail"]))
                self.assertIn("method", str(row["anomaly_detail"]))

    def test_build_work_structures_allows_review_without_required_signatures(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            with _patch_workspace(tmp):
                db.ensure_db()
                with db.connect() as conn:
                    collection_id = upsert_collection(conn, {"name": "default", "label": "Default"})
                    work_id = self._insert_hit(
                        conn,
                        collection_id=collection_id,
                        control_number=103,
                        title="Review of heavy flavor anomalies",
                        abstract="We review recent anomalies across flavor observables.",
                        document_type=["review"],
                    )

                    summary = build_work_structures(conn, work_ids=[work_id], collection="default")
                    row = conn.execute(
                        "SELECT status, is_review, anomaly_code FROM work_capsules WHERE work_id = ?",
                        (work_id,),
                    ).fetchone()

                self.assertEqual(summary["processed"], 1)
                self.assertEqual(summary["review_relaxed"], 1)
                self.assertEqual(int(row["is_review"]), 1)
                self.assertEqual(row["status"], "review_relaxed")
                self.assertIsNone(row["anomaly_code"])


class PdgImportTests(unittest.TestCase):
    def test_import_pdg_source_materializes_capsules_from_mineru_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            with _patch_workspace(tmp):
                bundle_dir = tmp / "pdg_bundle"
                bundle_dir.mkdir(parents=True)
                (bundle_dir / "pdg_full.md").write_text("# PDG\n", encoding="utf-8")
                (bundle_dir / "pdg_content_list.json").write_text(
                    json.dumps(
                        [
                            {"type": "text", "text_level": 1, "text": "PDG Review of B Mesons", "page_idx": 1},
                            {"type": "text", "text_level": 1, "text": "1 Properties", "page_idx": 1},
                            {"type": "text", "text": "The B0 meson mass is measured precisely.", "page_idx": 1},
                            {"type": "text", "text_level": 1, "text": "2 Decays", "page_idx": 2},
                            {"type": "text", "text": "Semileptonic decays constrain CKM elements.", "page_idx": 2},
                        ],
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )

                db.ensure_db()
                with db.connect() as conn:
                    summary = import_pdg_source(
                        conn,
                        source_path=bundle_dir,
                        source_id="pdg-2026",
                        title="PDG 2026",
                    )
                    source_row = conn.execute("SELECT source_id, capsule_count FROM pdg_sources WHERE source_id = 'pdg-2026'").fetchone()
                    capsule_rows = conn.execute("SELECT title, capsule_text FROM pdg_sections WHERE source_id = 'pdg-2026' ORDER BY order_index").fetchall()

                self.assertEqual(summary["source_id"], "pdg-2026")
                self.assertEqual(summary["capsule_count"], 2)
                self.assertIsNotNone(source_row)
                self.assertEqual(int(source_row["capsule_count"]), 2)
                self.assertEqual(len(capsule_rows), 2)
                self.assertIn("Properties", capsule_rows[0]["title"])
                self.assertIn("measured precisely", capsule_rows[0]["capsule_text"])

    def test_cli_import_pdg_works_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            with _patch_workspace(tmp):
                bundle_dir = tmp / "pdg_bundle"
                bundle_dir.mkdir(parents=True)
                (bundle_dir / "pdg_full.md").write_text("# PDG\n", encoding="utf-8")
                (bundle_dir / "pdg_content_list.json").write_text(
                    json.dumps(
                        [
                            {"type": "text", "text_level": 1, "text": "PDG Topic", "page_idx": 1},
                            {"type": "text", "text_level": 1, "text": "1 Summary", "page_idx": 1},
                            {"type": "text", "text": "This section summarizes the topic.", "page_idx": 1},
                        ],
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )

                parser = cli.build_parser()
                args = parser.parse_args(["import-pdg", "--source", str(bundle_dir), "--source-id", "pdg-cli", "--title", "PDG CLI"])
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    args.func(args)
                payload = json.loads(out.getvalue())

                self.assertEqual(payload["source_id"], "pdg-cli")
                self.assertEqual(payload["capsule_count"], 1)


class ConfigProfileTests(unittest.TestCase):
    def test_resolve_embedding_profile_exposes_explicit_local_profile(self) -> None:
        config = default_config()
        config["profiles"]["embedding"] = "semantic_small_local"
        resolved = resolve_embedding_profile(config)
        self.assertEqual(resolved["name"], "semantic_small_local")
        self.assertEqual(resolved["model"], "sentence-transformers:BAAI/bge-small-en-v1.5")
        self.assertEqual(resolved["runtime"]["device"], "cuda")
        self.assertGreaterEqual(int(resolved["runtime"]["batch_size"]), 16)


if __name__ == "__main__":
    unittest.main()
