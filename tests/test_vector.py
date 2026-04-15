from __future__ import annotations

import contextlib
import io
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hep_rag_v2 import cli, db, paths
from hep_rag_v2.fulltext import import_mineru_source, materialize_mineru_document
from hep_rag_v2.graph import graph_neighbors, rebuild_graph_edges
from hep_rag_v2.metadata import upsert_collection, upsert_work_from_hit
from hep_rag_v2.query import build_match_queries, rewrite_query_for_embedding
from hep_rag_v2.search import rebuild_search_indices
from hep_rag_v2.vector import (
    HASH_IDF_VECTOR_MODEL,
    rebuild_vector_indices,
    route_query,
    search_works_hybrid,
    search_chunks_vector,
    search_works_vector,
)
from hep_rag_v2.vector.embedding import configure_embedding_runtime, _embed_corpus


class TestVectorSearch(unittest.TestCase):
    def test_parser_source_declares_explicit_sync_commands(self) -> None:
        source = (SRC / "hep_rag_v2" / "cli" / "_parser.py").read_text(encoding="utf-8")

        self.assertIn('sub.add_parser("sync-search"', source)
        self.assertIn('sub.add_parser("sync-vectors"', source)
        self.assertIn('sub.add_parser("sync-graph"', source)
        self.assertIn("s.set_defaults(func=cmd_sync_search)", source)
        self.assertIn("s.set_defaults(func=cmd_sync_vectors)", source)
        self.assertIn("s.set_defaults(func=cmd_sync_graph)", source)

    def test_parser_source_declares_sync_scope_and_filter_arguments(self) -> None:
        source = (SRC / "hep_rag_v2" / "cli" / "_parser.py").read_text(encoding="utf-8")

        self.assertGreaterEqual(source.count('s.add_argument("--scope", choices=["all", "dirty"], default="dirty")'), 3)
        self.assertGreaterEqual(source.count('s.add_argument("--collection", default=None)'), 3)
        self.assertGreaterEqual(source.count('s.add_argument("--updated-since", default=None)'), 3)

    def test_sync_search_handler_exists_in_source(self) -> None:
        source = (SRC / "hep_rag_v2" / "cli" / "search.py").read_text(encoding="utf-8")
        self.assertIn("def cmd_sync_search(args: argparse.Namespace) -> None:", source)

    def test_sync_vectors_handler_exists_in_source(self) -> None:
        source = (SRC / "hep_rag_v2" / "cli" / "search.py").read_text(encoding="utf-8")
        self.assertIn("def cmd_sync_vectors(args: argparse.Namespace) -> None:", source)

    def test_sync_graph_handler_exists_in_source(self) -> None:
        source = (SRC / "hep_rag_v2" / "cli" / "search.py").read_text(encoding="utf-8")
        self.assertIn("def cmd_sync_graph(args: argparse.Namespace) -> None:", source)

    def test_sync_source_uses_maintenance_interfaces(self) -> None:
        source = (SRC / "hep_rag_v2" / "cli" / "search.py").read_text(encoding="utf-8")

        self.assertIn("from hep_rag_v2.maintenance import (", source)
        self.assertIn("select_dirty_work_ids", source)
        self.assertIn("clear_dirty_work_ids", source)
        self.assertIn("dirty_counts", source)
        self.assertIn("start_maintenance_job", source)
        self.assertIn("finish_maintenance_job", source)

    def test_query_rewrite_and_match_queries_strip_intent_words(self) -> None:
        rewritten = rewrite_query_for_embedding("综述一下 H -> aa 相关工作")
        self.assertIn("higgs", rewritten)
        self.assertIn("pseudoscalar", rewritten)
        self.assertNotIn("综述", rewritten)
        self.assertNotIn("相关工作", rewritten)

        queries = build_match_queries("综述一下 H -> aa 相关工作")
        self.assertGreaterEqual(len(queries), 1)
        self.assertIn('"higgs"', queries[0])
        self.assertIn('"pseudoscalar"', queries[0])
        self.assertNotIn("综述", " ".join(queries))

    def test_query_rewrite_handles_hep_abbreviations_and_result_style_queries(self) -> None:
        rewritten = rewrite_query_for_embedding("总结CMS VBS SSWW的最新结果")
        self.assertIn("cms collaboration", rewritten)
        self.assertIn("vector boson scattering", rewritten)
        self.assertIn("same-sign", rewritten)
        self.assertNotIn("总结", rewritten)
        self.assertNotIn("最新", rewritten)

        queries = build_match_queries("CMS current SSWW best results")
        self.assertGreaterEqual(len(queries), 1)
        self.assertIn('"cms"', queries[0])
        self.assertIn('"same sign ww"', queries[0])
        self.assertNotIn("current", " ".join(queries))
        self.assertNotIn("best", " ".join(queries))
        self.assertNotIn("results", " ".join(queries))

        english_routing = route_query("CMS current SSWW best results")
        self.assertEqual(english_routing["target"], "works")
        self.assertEqual(english_routing["graph_expand"], 3)

        chinese_routing = route_query("总结CMS VBS SSWW的最新结果")
        self.assertEqual(chinese_routing["target"], "works")
        self.assertGreaterEqual(chinese_routing["graph_expand"], 3)

    def test_build_vector_index_and_search_works_and_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            with _patch_workspace(tmp):
                db.ensure_db()
                with db.connect() as conn:
                    collection_id = upsert_collection(conn, {"name": "cms_rare_decay"})
                    rare_hit = {
                        "metadata": {
                            "control_number": 801,
                            "titles": [{"title": "Observation of the rare decay of the eta meson to four muons"}],
                            "abstracts": [{"value": "A rare decay search with four muons and 101 inverse femtobarns."}],
                            "arxiv_eprints": [{"value": "2601.00001"}],
                            "keywords": [{"value": "rare decay"}],
                        }
                    }
                    higgs_hit = {
                        "metadata": {
                            "control_number": 802,
                            "titles": [{"title": "Search for an exotic Higgs boson decay"}],
                            "abstracts": [{"value": "We study an exotic Higgs final state with photons."}],
                            "arxiv_eprints": [{"value": "2601.00002"}],
                            "keywords": [{"value": "Higgs"}],
                        }
                    }
                    upsert_work_from_hit(conn, collection_id=collection_id, hit=rare_hit)
                    upsert_work_from_hit(conn, collection_id=collection_id, hit=higgs_hit)
                    work_id = int(
                        conn.execute(
                            "SELECT work_id FROM works WHERE canonical_id = '801'"
                        ).fetchone()[0]
                    )

                    bundle_dir = tmp / "bundle_vector"
                    bundle_dir.mkdir()
                    (bundle_dir / "paper_full.md").write_text("# Rare decay note\n", encoding="utf-8")
                    (bundle_dir / "paper_content_list.json").write_text(
                        json.dumps(
                            [
                                {"type": "text", "text_level": 1, "text": "Rare decay note", "page_idx": 1},
                                {"type": "text", "text_level": 1, "text": "Abstract", "page_idx": 1},
                                {"type": "text", "text": "A rare decay search with a profile likelihood analysis.", "page_idx": 1},
                                {"type": "text", "text_level": 1, "text": "1 Method", "page_idx": 2},
                                {"type": "text", "text": "We fit the signal model with profile likelihood templates and detector efficiency terms.", "page_idx": 2},
                                {"type": "text", "text_level": 1, "text": "References", "page_idx": 3},
                                {"type": "text", "text": "[1] Bibliography entry.", "page_idx": 3},
                            ],
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )
                    dest_dir = paths.PARSED_DIR / "cms_rare_decay" / "2601.00001"
                    import_mineru_source(source_path=bundle_dir, dest_dir=dest_dir)
                    materialize_mineru_document(conn, work_id=work_id, manifest_path=dest_dir / "manifest.json")

                    progress_messages: list[str] = []
                    summary = rebuild_vector_indices(conn, target="all", progress=progress_messages.append)
                    conn.commit()

                    self.assertEqual(summary["works"], 2)
                    self.assertGreaterEqual(summary["chunks"], 2)
                    self.assertTrue(any("building work vector index" in item for item in progress_messages))
                    self.assertTrue(any("work vector index ready" in item for item in progress_messages))
                    self.assertTrue(any("building chunk vector index" in item for item in progress_messages))
                    self.assertTrue(any("chunk vector index ready" in item for item in progress_messages))

                    work_results = search_works_vector(conn, query="rare decay four muons", limit=5)
                    self.assertGreaterEqual(len(work_results), 1)
                    self.assertEqual(work_results[0]["canonical_id"], "801")

                    chunk_results = search_chunks_vector(conn, query="profile likelihood detector efficiency", limit=5)
                    self.assertGreaterEqual(len(chunk_results), 1)
                    self.assertIn("profile likelihood", chunk_results[0]["clean_text"].lower())
                    self.assertNotIn("bibliography entry", chunk_results[0]["clean_text"].lower())

    def test_sentence_transformer_runtime_applies_mirror_cache_and_batch_size(self) -> None:
        model = "sentence-transformers:unit/test-model"
        captured_env: dict[str, str | None] = {}
        snapshot_calls: dict[str, object] = {}
        original_endpoint = "https://huggingface.co"
        original_template = original_endpoint + "/{repo_id}/resolve/{revision}/{filename}"

        class _FakeEncoder:
            def encode(self, texts, **kwargs):
                self.texts = list(texts)
                self.kwargs = dict(kwargs)
                return np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)

        fake_encoder = _FakeEncoder()
        fake_constants = ModuleType("huggingface_hub.constants")
        fake_constants.ENDPOINT = original_endpoint
        fake_constants.HUGGINGFACE_CO_URL_TEMPLATE = original_template
        fake_huggingface_hub = ModuleType("huggingface_hub")
        fake_huggingface_hub.constants = fake_constants

        def _snapshot_download(*args, **kwargs):
            snapshot_calls["args"] = args
            snapshot_calls["kwargs"] = dict(kwargs)
            snapshot_calls["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT")
            snapshot_calls["HUGGINGFACE_HUB_ENDPOINT"] = os.environ.get("HUGGINGFACE_HUB_ENDPOINT")
            snapshot_calls["HF_CONSTANTS_ENDPOINT"] = fake_constants.ENDPOINT
            snapshot_calls["HF_CONSTANTS_TEMPLATE"] = fake_constants.HUGGINGFACE_CO_URL_TEMPLATE
            return "/tmp/fake-snapshot"

        fake_huggingface_hub.snapshot_download = _snapshot_download

        def _constructor(*args, **kwargs):
            captured_env["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT")
            captured_env["HUGGINGFACE_HUB_ENDPOINT"] = os.environ.get("HUGGINGFACE_HUB_ENDPOINT")
            captured_env["HF_CONSTANTS_ENDPOINT"] = fake_constants.ENDPOINT
            captured_env["HF_CONSTANTS_TEMPLATE"] = fake_constants.HUGGINGFACE_CO_URL_TEMPLATE
            captured_env["args"] = args
            captured_env["kwargs"] = kwargs
            return fake_encoder

        fake_module = SimpleNamespace(SentenceTransformer=_constructor)
        with mock.patch.dict(
            sys.modules,
            {
                "sentence_transformers": fake_module,
                "huggingface_hub": fake_huggingface_hub,
                "huggingface_hub.constants": fake_constants,
            },
        ):
            configure_embedding_runtime(
                model=model,
                settings={
                    "allow_silent_fallback": False,
                    "runtime": {
                        "device": "cpu",
                        "batch_size": 7,
                        "huggingface": {
                            "endpoint": "https://hf-mirror.com",
                            "cache_dir": "/tmp/hf-cache",
                            "local_files_only": False,
                            "token": "secret-token",
                        },
                    },
                },
            )
            vectors, _extras, actual_dim = _embed_corpus(["hello world"], model=model, dim=384)
            self.assertEqual(fake_constants.ENDPOINT, original_endpoint)
            self.assertEqual(fake_constants.HUGGINGFACE_CO_URL_TEMPLATE, original_template)

        self.assertEqual(actual_dim, 3)
        self.assertEqual(vectors.shape, (1, 3))
        self.assertEqual(snapshot_calls["args"], ())
        self.assertEqual(
            snapshot_calls["kwargs"],
            {
                "repo_id": "unit/test-model",
                "cache_dir": "/tmp/hf-cache",
                "local_files_only": False,
                "token": "secret-token",
            },
        )
        self.assertEqual(snapshot_calls["HF_ENDPOINT"], "https://hf-mirror.com")
        self.assertEqual(snapshot_calls["HUGGINGFACE_HUB_ENDPOINT"], "https://hf-mirror.com")
        self.assertEqual(snapshot_calls["HF_CONSTANTS_ENDPOINT"], "https://hf-mirror.com")
        self.assertEqual(
            snapshot_calls["HF_CONSTANTS_TEMPLATE"],
            "https://hf-mirror.com/{repo_id}/resolve/{revision}/{filename}",
        )
        self.assertEqual(captured_env["HF_ENDPOINT"], "https://hf-mirror.com")
        self.assertEqual(captured_env["HUGGINGFACE_HUB_ENDPOINT"], "https://hf-mirror.com")
        self.assertEqual(captured_env["HF_CONSTANTS_ENDPOINT"], "https://hf-mirror.com")
        self.assertEqual(
            captured_env["HF_CONSTANTS_TEMPLATE"],
            "https://hf-mirror.com/{repo_id}/resolve/{revision}/{filename}",
        )
        self.assertEqual(captured_env["args"], ("/tmp/fake-snapshot",))
        self.assertEqual(
            captured_env["kwargs"],
            {
                "device": "cpu",
                "cache_folder": "/tmp/hf-cache",
                "local_files_only": True,
                "token": "secret-token",
            },
        )
        self.assertEqual(fake_encoder.texts, ["hello world"])
        self.assertEqual(fake_encoder.kwargs["batch_size"], 7)
        self.assertTrue(fake_encoder.kwargs["convert_to_numpy"])
        self.assertTrue(fake_encoder.kwargs["normalize_embeddings"])

    def test_hash_idf_vector_model_handles_natural_language_query_noise(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            with _patch_workspace(tmp):
                db.ensure_db()
                with db.connect() as conn:
                    collection_id = upsert_collection(conn, {"name": "cms_rare_decay"})
                    for control_number, title, abstract in [
                        (801, "Observation of the rare decay of the eta meson to four muons", "A rare decay search with four muons."),
                        (802, "Search for an exotic Higgs boson decay", "We study an exotic Higgs final state with photons."),
                    ]:
                        hit = {
                            "metadata": {
                                "control_number": control_number,
                                "titles": [{"title": title}],
                                "abstracts": [{"value": abstract}],
                            }
                        }
                        upsert_work_from_hit(conn, collection_id=collection_id, hit=hit)

                    rebuild_vector_indices(conn, target="works", model=HASH_IDF_VECTOR_MODEL)
                    conn.commit()

                    results = search_works_vector(
                        conn,
                        query="综述一下 rare decay four muons 相关工作",
                        limit=5,
                        model=HASH_IDF_VECTOR_MODEL,
                    )
                    self.assertGreaterEqual(len(results), 1)
                    self.assertEqual(results[0]["canonical_id"], "801")

    def test_hybrid_work_search_prefers_cms_ssww_results_for_result_queries(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            with _patch_workspace(tmp):
                db.ensure_db()
                with db.connect() as conn:
                    collection_id = upsert_collection(conn, {"name": "cms_ssww"})
                    for control_number, title, abstract, collabs in [
                        (
                            901,
                            "Measurements of production cross sections of polarized same-sign W boson pairs in association with two jets in proton-proton collisions at 13 TeV",
                            "The first measurements of polarized same-sign W boson pairs are reported with the CMS detector in vector boson scattering.",
                            [{"value": "CMS"}],
                        ),
                        (
                            902,
                            "Observation of electroweak production of same-sign W boson pairs in the two jet and two same-sign lepton final state in proton-proton collisions at 13 TeV",
                            "CMS observes electroweak same-sign W boson pair production in association with two jets.",
                            [{"value": "CMS"}],
                        ),
                        (
                            903,
                            "Measurement and interpretation of same-sign W boson pair production in association with two jets in pp collisions at 13 TeV with the ATLAS detector",
                            "ATLAS measures same-sign W boson pair production in association with two jets.",
                            [{"value": "ATLAS"}],
                        ),
                        (
                            904,
                            "Stairway to discovery: A report on the CMS programme of cross section measurements from millibarns to femtobarns",
                            "A broad CMS cross section programme overview covering many measurements and future prospects.",
                            [{"value": "CMS"}],
                        ),
                    ]:
                        hit = {
                            "metadata": {
                                "control_number": control_number,
                                "titles": [{"title": title}],
                                "abstracts": [{"value": abstract}],
                                "collaborations": collabs,
                            }
                        }
                        upsert_work_from_hit(conn, collection_id=collection_id, hit=hit)

                    rebuild_search_indices(conn, target="works")
                    rebuild_vector_indices(conn, target="works", model=HASH_IDF_VECTOR_MODEL)
                    conn.commit()

                    rows = search_works_hybrid(
                        conn,
                        query="CMS current SSWW best results",
                        collection="cms_ssww",
                        limit=4,
                        model=HASH_IDF_VECTOR_MODEL,
                    )
                    self.assertGreaterEqual(len(rows), 2)
                    self.assertEqual(rows[0]["canonical_id"], "901")
                    self.assertIn(rows[1]["canonical_id"], {"902", "904"})

    def test_cli_search_hybrid_auto_routes_broad_queries_to_work_level(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            with _patch_workspace(tmp):
                db.ensure_db()
                with db.connect() as conn:
                    collection_id = upsert_collection(conn, {"name": "cms_rare_decay"})
                    shared_refs = [{"control_number": 901}, {"control_number": 902}]
                    for control_number, title, abstract in [
                        (701, "Search for exotic Higgs boson decays to four photons", "A broad Higgs exotic decay analysis."),
                        (702, "Search for exotic Higgs boson decays to two muons and two b quarks", "A related exotic Higgs decay analysis."),
                        (901, "Reference one", "Foundational method paper."),
                        (902, "Reference two", "Foundational detector paper."),
                    ]:
                        hit = {
                            "metadata": {
                                "control_number": control_number,
                                "titles": [{"title": title}],
                                "abstracts": [{"value": abstract}],
                                "references": shared_refs if control_number in {701, 702} else [],
                            }
                        }
                        upsert_work_from_hit(conn, collection_id=collection_id, hit=hit)

                    rebuild_search_indices(conn, target="works")
                    rebuild_vector_indices(conn, target="works")
                    rebuild_graph_edges(conn, target="bibliographic-coupling", collection="cms_rare_decay", min_shared=2)
                    conn.commit()

                self.assertEqual(route_query("综述一下 exotic Higgs decays 的相关工作")["target"], "works")

                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    cli.cmd_search_hybrid(
                        SimpleNamespace(
                            query="综述一下 exotic Higgs decays 的相关工作",
                            target="auto",
                            collection="cms_rare_decay",
                            limit=5,
                            model="hash-v1",
                            graph_expand=None,
                            seed_limit=5,
                        )
                    )
                payload = json.loads(out.getvalue())
                self.assertEqual(payload["routing"]["target"], "works")
                self.assertEqual(payload["routing"]["graph_expand"], 5)
                self.assertGreaterEqual(len(payload["results"]), 1)
                self.assertIn(payload["results"][0]["canonical_id"], {"701", "702"})

    def test_similarity_graph_builds_neighbors_from_work_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            with _patch_workspace(tmp):
                db.ensure_db()
                with db.connect() as conn:
                    collection_id = upsert_collection(conn, {"name": "cms_rare_decay"})
                    for control_number, title, abstract in [
                        (811, "Observation of the rare decay to four muons", "A rare decay search with four muons."),
                        (812, "Measurement of a rare decay with four muons", "Rare decay study with muons and signal extraction."),
                        (813, "Search for top quark flavor changing interactions", "A top quark Higgs analysis with diphotons."),
                    ]:
                        hit = {
                            "metadata": {
                                "control_number": control_number,
                                "titles": [{"title": title}],
                                "abstracts": [{"value": abstract}],
                            }
                        }
                        upsert_work_from_hit(conn, collection_id=collection_id, hit=hit)

                    progress_messages: list[str] = []
                    rebuild_vector_indices(
                        conn,
                        target="works",
                        model=HASH_IDF_VECTOR_MODEL,
                        progress=progress_messages.append,
                    )
                    rebuild_graph_edges(
                        conn,
                        target="similarity",
                        collection="cms_rare_decay",
                        similarity_model=HASH_IDF_VECTOR_MODEL,
                        similarity_top_k=2,
                        similarity_min_score=0.2,
                        progress=progress_messages.append,
                    )
                    conn.commit()

                    work_id = int(
                        conn.execute(
                            "SELECT work_id FROM works WHERE canonical_id = '811'"
                        ).fetchone()[0]
                    )
                    neighbors = graph_neighbors(
                        conn,
                        work_id=work_id,
                        edge_kind="similarity",
                        collection="cms_rare_decay",
                        limit=5,
                    )
                    self.assertGreaterEqual(len(neighbors), 1)
                    self.assertEqual(neighbors[0]["canonical_id"], "812")
                    self.assertEqual(neighbors[0]["edge_kind"], "similarity")
                    self.assertTrue(any("building work vector index" in item for item in progress_messages))
                    self.assertTrue(any("work vector index ready" in item for item in progress_messages))
                    self.assertTrue(any("building similarity edges" in item for item in progress_messages))
                    self.assertTrue(any("similarity edges ready" in item for item in progress_messages))


@contextlib.contextmanager
def _patch_workspace(tmp: Path):
    original = paths.workspace_root()
    try:
        paths.set_workspace_root(tmp)
        yield
    finally:
        paths.set_workspace_root(original)
