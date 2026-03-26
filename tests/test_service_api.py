from __future__ import annotations

import contextlib
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hep_rag_v2 import db, paths
from hep_rag_v2.api import create_app
from hep_rag_v2.config import default_config
from hep_rag_v2.fulltext import import_mineru_source, materialize_mineru_document
from hep_rag_v2.metadata import upsert_collection, upsert_work_from_hit
from hep_rag_v2.service import audit_document_payload, show_document_payload, show_graph_payload, workspace_status_payload


class ServiceLayerTests(unittest.TestCase):
    def test_workspace_and_inspection_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            with _patch_workspace(tmp):
                work_id, peer_work_id = _seed_workspace(tmp)

                status = workspace_status_payload()
                self.assertEqual(status["snapshot"]["works"], 2)
                self.assertEqual(status["collections"][0]["collection"], "default")

                graph = show_graph_payload(
                    work_id=work_id,
                    edge_kind="similarity",
                    collection="default",
                    similarity_model="hash-idf-v1",
                )
                self.assertEqual(graph["work"]["work_id"], work_id)
                self.assertEqual(graph["neighbors"][0]["neighbor_work_id"], peer_work_id)

                document = show_document_payload(work_id=work_id, limit=5)
                self.assertEqual(document["document"]["work_id"], work_id)
                self.assertGreaterEqual(len(document["sections_sample"]), 1)
                self.assertIn("body", document["block_roles"])

                audit = audit_document_payload(work_id=work_id, limit=5)
                self.assertEqual(audit["work"]["work_id"], work_id)
                self.assertIn("recommendation", audit)
                self.assertIn("noise", audit)


class ApiLayerTests(unittest.TestCase):
    def test_root_ui_is_served(self) -> None:
        original_root = paths.workspace_root()
        try:
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                app = create_app(config_loader=_api_config_loader(tmp))
                with TestClient(app) as client:
                    response = client.get("/")
                    self.assertEqual(response.status_code, 200)
                    self.assertIn("hep-rag Console", response.text)
                    self.assertIn("/workspace/status", response.text)
                    self.assertIn("/auth/status", response.text)
                    health = client.get("/health").json()
                    self.assertEqual(
                        Path(health["api_db_path"]),
                        tmp / "workspace" / "db" / "hep_rag_api.db",
                    )
        finally:
            paths.set_workspace_root(original_root)

    def test_retrieve_endpoint_uses_pipeline(self) -> None:
        original_root = paths.workspace_root()
        try:
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                app = create_app(config_loader=_api_config_loader(tmp))
                with TestClient(app) as client, mock.patch("hep_rag_v2.api.app.retrieve", return_value={"query": "q", "works": []}) as retrieve_mock:
                    response = client.post("/retrieve", json={"query": "q", "limit": 3, "target": "works"})
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json()["query"], "q")
                    retrieve_mock.assert_called_once_with(
                        mock.ANY,
                        query="q",
                        limit=3,
                        target="works",
                        collection_name=None,
                        model=None,
                    )
                    config = retrieve_mock.call_args.args[0]
                    self.assertFalse(config["llm"]["enabled"])
                    self.assertEqual(Path(config["workspace"]["root"]), (tmp / "workspace").resolve())
        finally:
            paths.set_workspace_root(original_root)

    def test_ingest_job_endpoint_records_progress_and_result(self) -> None:
        def _fake_ingest(config, *, query, limit, collection_name, download_limit, parse_limit, replace_existing, skip_parse, skip_index, skip_graph, progress):
            progress("searching INSPIRE...")
            progress("downloading PDFs...")
            return {
                "query": query,
                "limit": limit,
                "collection": collection_name or "default",
                "ok": True,
            }

        original_root = paths.workspace_root()
        try:
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                app = create_app(config_loader=_api_config_loader(tmp))
                with TestClient(app) as client, mock.patch("hep_rag_v2.api.app.ingest_online", side_effect=_fake_ingest):
                    response = client.post(
                        "/jobs/ingest-online",
                        json={"query": "CMS VBS SSWW", "limit": 4, "collection_name": "default"},
                    )
                    self.assertEqual(response.status_code, 200)
                    job = response.json()
                    job_id = job["job_id"]
                    self.assertEqual(job["request"]["query"], "CMS VBS SSWW")

                    payload = _wait_for_job(client, job_id)
                    self.assertEqual(payload["status"], "succeeded")
                    self.assertEqual(payload["result"]["query"], "CMS VBS SSWW")
                    self.assertTrue((tmp / "workspace" / "db" / "hep_rag_api.db").exists())

                    events = client.get(f"/jobs/{job_id}/events").json()["events"]
                    self.assertTrue(any("searching INSPIRE" in item["message"] for item in events))
                    self.assertTrue(any("job finished successfully" in item["message"] for item in events))
        finally:
            paths.set_workspace_root(original_root)

    def test_private_endpoints_require_token_when_configured(self) -> None:
        original_root = paths.workspace_root()
        try:
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                app = create_app(
                    config_loader=_api_config_loader(
                        tmp,
                        {"api": {"auth_token": "secret-token"}},
                    )
                )
                with TestClient(app) as client, mock.patch("hep_rag_v2.api.app.retrieve", return_value={"query": "q", "works": []}) as retrieve_mock:
                    response = client.get("/")
                    self.assertEqual(response.status_code, 200)

                    auth_status = client.get("/auth/status")
                    self.assertEqual(auth_status.status_code, 200)
                    self.assertTrue(auth_status.json()["auth_enabled"])
                    self.assertEqual(
                        Path(auth_status.json()["api_db_path"]),
                        tmp / "workspace" / "db" / "hep_rag_api.db",
                    )

                    unauthorized = client.post("/retrieve", json={"query": "q"})
                    self.assertEqual(unauthorized.status_code, 401)

                    authorized = client.post(
                        "/retrieve",
                        json={"query": "q"},
                        headers={"Authorization": "Bearer secret-token"},
                    )
                    self.assertEqual(authorized.status_code, 200)
                    self.assertEqual(authorized.json()["query"], "q")
                    retrieve_mock.assert_called_once()
        finally:
            paths.set_workspace_root(original_root)

    def test_job_history_persists_across_app_recreation(self) -> None:
        def _fake_ingest(config, *, query, limit, collection_name, download_limit, parse_limit, replace_existing, skip_parse, skip_index, skip_graph, progress):
            progress("searching INSPIRE...")
            return {"query": query, "collection": collection_name or "default", "ok": True}

        original_root = paths.workspace_root()
        try:
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                loader = _api_config_loader(tmp)

                app = create_app(config_loader=loader)
                with TestClient(app) as client, mock.patch("hep_rag_v2.api.app.ingest_online", side_effect=_fake_ingest):
                    response = client.post("/jobs/ingest-online", json={"query": "CMS VBS SSWW", "limit": 2})
                    self.assertEqual(response.status_code, 200)
                    job_id = response.json()["job_id"]
                    payload = _wait_for_job(client, job_id)
                    self.assertEqual(payload["status"], "succeeded")
                    self.assertTrue((tmp / "workspace" / "db" / "hep_rag_api.db").exists())

                reopened = create_app(config_loader=loader)
                with TestClient(reopened) as client:
                    jobs = client.get("/jobs").json()
                    self.assertIn(job_id, {item["job_id"] for item in jobs})
                    events = client.get(f"/jobs/{job_id}/events").json()["events"]
                    self.assertTrue(any(item["type"] == "progress" for item in events))
                    self.assertTrue(any("job finished successfully" in item["message"] for item in events))
        finally:
            paths.set_workspace_root(original_root)

    def test_progress_events_do_not_fail_job_when_main_db_is_write_locked(self) -> None:
        def _fake_ingest(config, *, query, limit, collection_name, download_limit, parse_limit, replace_existing, skip_parse, skip_index, skip_graph, progress):
            with db.connect() as conn:
                conn.execute("CREATE TABLE IF NOT EXISTS lock_test (id INTEGER PRIMARY KEY, value TEXT)")
                conn.execute("INSERT INTO lock_test (value) VALUES (?)", ("locked",))
                progress("progress emitted while metadata transaction holds the write lock")
            return {"query": query, "ok": True}

        original_root = paths.workspace_root()
        try:
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                app = create_app(config_loader=_api_config_loader(tmp))
                with TestClient(app) as client, mock.patch("hep_rag_v2.api.app.ingest_online", side_effect=_fake_ingest):
                    response = client.post(
                        "/jobs/ingest-online",
                        json={"query": "CMS VBS SSWW", "limit": 2},
                    )
                    self.assertEqual(response.status_code, 200)
                    job_id = response.json()["job_id"]

                    payload = _wait_for_job(client, job_id)
                    self.assertEqual(payload["status"], "succeeded")

                    events = client.get(f"/jobs/{job_id}/events").json()["events"]
                    self.assertTrue(
                        any("write lock" in item["message"] for item in events),
                        msg=events,
                    )
        finally:
            paths.set_workspace_root(original_root)


def _seed_workspace(tmp: Path) -> tuple[int, int]:
    db.ensure_db()
    with db.connect() as conn:
        collection_id = upsert_collection(conn, {"name": "default"})
        primary_hit = {
            "metadata": {
                "control_number": 333,
                "titles": [{"title": "Observation of electroweak production of same-sign W boson pairs"}],
                "abstracts": [{"value": "CMS observes same-sign WW production via vector boson scattering."}],
                "arxiv_eprints": [{"value": "2401.00001"}],
            }
        }
        peer_hit = {
            "metadata": {
                "control_number": 334,
                "titles": [{"title": "Measurements of same-sign WW production in association with two jets"}],
                "abstracts": [{"value": "A related study of same-sign WW production."}],
                "arxiv_eprints": [{"value": "2401.00002"}],
            }
        }
        upsert_work_from_hit(conn, collection_id=collection_id, hit=primary_hit)
        upsert_work_from_hit(conn, collection_id=collection_id, hit=peer_hit)
        work_id = int(conn.execute("SELECT work_id FROM works WHERE canonical_id = '333'").fetchone()[0])
        peer_work_id = int(conn.execute("SELECT work_id FROM works WHERE canonical_id = '334'").fetchone()[0])
        conn.execute(
            """
            INSERT INTO similarity_edges (src_work_id, dst_work_id, metric, score)
            VALUES (?, ?, ?, ?)
            """,
            (work_id, peer_work_id, "cosine::hash-idf-v1", 0.82),
        )

        bundle_dir = tmp / "bundle_service"
        bundle_dir.mkdir()
        (bundle_dir / "paper_full.md").write_text("# CMS note\n", encoding="utf-8")
        (bundle_dir / "paper_content_list.json").write_text(
            json.dumps(
                [
                    {"type": "text", "text_level": 1, "text": "CMS note", "page_idx": 1},
                    {"type": "text", "text_level": 1, "text": "Abstract", "page_idx": 1},
                    {"type": "text", "text": "CMS observes same-sign WW production via vector boson scattering.", "page_idx": 1},
                    {"type": "text", "text_level": 1, "text": "1 Results", "page_idx": 2},
                    {"type": "text", "text": "The observed significance exceeds the standard model expectation.", "page_idx": 2},
                    {"type": "text", "text_level": 1, "text": "References", "page_idx": 3},
                    {"type": "text", "text": "[1] Reference entry.", "page_idx": 3},
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        dest_dir = paths.PARSED_DIR / "default" / "2401.00001"
        import_mineru_source(source_path=bundle_dir, dest_dir=dest_dir)
        materialize_mineru_document(conn, work_id=work_id, manifest_path=dest_dir / "manifest.json")
        conn.commit()
    return (work_id, peer_work_id)


def _wait_for_job(client: TestClient, job_id: str, *, timeout_sec: float = 5.0) -> dict[str, object]:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        response = client.get(f"/jobs/{job_id}")
        payload = response.json()
        if payload["status"] in {"succeeded", "failed"}:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"Timed out waiting for job {job_id}")


@contextlib.contextmanager
def _patch_workspace(tmp: Path):
    original = paths.workspace_root()
    try:
        paths.set_workspace_root(tmp)
        yield
    finally:
        paths.set_workspace_root(original)


def _api_config_loader(tmp: Path, overrides: dict[str, object] | None = None):
    workspace_root = tmp / "workspace"
    config = default_config(workspace_root=workspace_root)
    config["llm"]["enabled"] = False
    if overrides:
        _deep_merge(config, overrides)

    def _loader() -> tuple[Path, dict[str, object]]:
        return (tmp / "hep-rag.yaml", config)

    return _loader


def _deep_merge(target: dict[str, object], updates: dict[str, object]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value


if __name__ == "__main__":
    unittest.main()
