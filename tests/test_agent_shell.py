from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hep_rag_v2.evidence import EvidenceRegistry
from hep_rag_v2.retrieval_adapter import TypedRetrievalResult, normalize_retrieval_payload
from hep_rag_v2.service.facade import HepRagServiceFacade
from hep_rag_v2.service.factory import create_tool_registry


class RetrievalAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = {
            "query": "CMS VBS SSWW",
            "collection": "default",
            "requested_target": "auto",
            "routing": {
                "target": "works",
                "graph_expand": 2,
                "reasons": ["hep_expansion", "default:evidence_lookup"],
            },
            "model": "hash-idf-v1",
            "works": [
                {
                    "work_id": 11,
                    "raw_title": "Observation of electroweak production of same-sign W boson pairs",
                    "abstract": "CMS observes same-sign WW production via vector boson scattering.",
                    "rank": 1,
                    "hybrid_score": 0.91,
                    "canonical_source": "inspire",
                    "canonical_id": "1624170",
                    "year": 2020,
                }
            ],
            "evidence_chunks": [
                {
                    "chunk_id": 101,
                    "work_id": 11,
                    "raw_title": "Observation of electroweak production of same-sign W boson pairs",
                    "clean_text": "The observed significance exceeds the background expectation.",
                    "section_hint": "Results",
                    "chunk_role": "body_chunk",
                    "rank": 1,
                    "hybrid_score": 0.87,
                    "canonical_source": "inspire",
                    "canonical_id": "1624170",
                }
            ],
        }

    def test_normalize_retrieval_payload_returns_uniform_typed_result(self) -> None:
        result = normalize_retrieval_payload(self.payload)

        self.assertIsInstance(result, TypedRetrievalResult)
        self.assertEqual(result.metadata.query, "CMS VBS SSWW")
        self.assertEqual(result.metadata.requested_target, "auto")
        self.assertEqual(result.metadata.target, "works")
        self.assertEqual(result.metadata.graph_expand, 2)
        self.assertEqual(result.metadata.reasons, ("hep_expansion", "default:evidence_lookup"))
        self.assertEqual(len(result.works), 1)
        self.assertEqual(len(result.chunks), 1)
        self.assertEqual(result.primary_items()[0].source_type, "chunk")
        self.assertEqual(result.primary_items(prefer_chunks=False)[0].source_type, "work")


class EvidenceRegistryTests(unittest.TestCase):
    def test_registry_dedupes_records_and_assigns_stable_refs(self) -> None:
        registry = EvidenceRegistry()
        registry.register_work(
            {
                "work_id": 11,
                "raw_title": "Observation of electroweak production of same-sign W boson pairs",
                "abstract": "CMS observes same-sign WW production via vector boson scattering.",
                "hybrid_score": 0.91,
            }
        )
        registry.register_chunk(
            {
                "chunk_id": 101,
                "work_id": 11,
                "raw_title": "Observation of electroweak production of same-sign W boson pairs",
                "clean_text": "The observed significance exceeds the background expectation.",
                "hybrid_score": 0.87,
            }
        )
        registry.register_chunk(
            {
                "chunk_id": 101,
                "work_id": 11,
                "raw_title": "Observation of electroweak production of same-sign W boson pairs",
                "clean_text": "Duplicate chunk should be ignored.",
                "hybrid_score": 0.42,
            }
        )

        records = registry.records()
        self.assertEqual([item.ref for item in records], ["W1", "C1"])
        self.assertEqual(registry.resolve("W1").work_id, 11)
        self.assertEqual(registry.resolve("C1").chunk_id, 101)
        self.assertEqual(len(records), 2)


class ServiceFacadeTests(unittest.TestCase):
    @mock.patch("hep_rag_v2.service.facade.pipeline_retrieve")
    def test_facade_retrieve_enriches_payload_with_typed_and_registry_views(self, retrieve_mock: mock.Mock) -> None:
        retrieve_mock.return_value = {
            "query": "CMS VBS SSWW",
            "collection": "default",
            "requested_target": "works",
            "routing": {"target": "works", "graph_expand": 0, "reasons": ["manual_target"]},
            "model": "hash-idf-v1",
            "works": [
                {
                    "work_id": 11,
                    "raw_title": "Observation of electroweak production of same-sign W boson pairs",
                    "abstract": "CMS observes same-sign WW production via vector boson scattering.",
                    "rank": 1,
                }
            ],
            "evidence_chunks": [],
        }

        facade = HepRagServiceFacade(config={"retrieval": {}})
        payload = facade.retrieve(query="CMS VBS SSWW", target="works", limit=3)

        self.assertEqual(payload["query"], "CMS VBS SSWW")
        self.assertEqual(payload["typed_retrieval"]["metadata"]["target"], "works")
        self.assertEqual(payload["evidence_registry"]["items"][0]["ref"], "E1")
        retrieve_mock.assert_called_once_with(
            {"retrieval": {}},
            query="CMS VBS SSWW",
            limit=3,
            target="works",
            collection_name=None,
            max_parallelism=None,
            model=None,
            progress=None,
        )

    @mock.patch("hep_rag_v2.service.facade.pipeline_retrieve")
    def test_facade_generate_ideas_returns_ranked_candidates_with_trace(self, retrieve_mock: mock.Mock) -> None:
        retrieve_mock.return_value = {
            "query": "same-sign WW transfer ideas",
            "collection": "default",
            "requested_target": "works",
            "routing": {"target": "works", "graph_expand": 1, "reasons": ["manual_target"]},
            "model": "hash-idf-v1",
            "works": [
                {
                    "work_id": 11,
                    "raw_title": "Observation of electroweak production of same-sign W boson pairs",
                    "abstract": "CMS observes same-sign WW production via vector boson scattering.",
                    "rank": 1,
                    "hybrid_score": 0.91,
                    "canonical_source": "inspire",
                    "canonical_id": "1624170",
                }
            ],
            "evidence_chunks": [
                {
                    "chunk_id": 101,
                    "work_id": 11,
                    "raw_title": "Observation of electroweak production of same-sign W boson pairs",
                    "clean_text": "The observed significance exceeds the background expectation.",
                    "rank": 1,
                    "hybrid_score": 0.87,
                    "canonical_source": "inspire",
                    "canonical_id": "1624170",
                }
            ],
        }

        progress = mock.Mock()
        facade = HepRagServiceFacade(config={"retrieval": {}, "ideas": {"top_k": 2}}, progress=progress)

        payload = facade.generate_ideas(query="same-sign WW transfer ideas", limit=3)

        self.assertEqual(payload["query"], "same-sign WW transfer ideas")
        self.assertGreaterEqual(len(payload["ideas"]), 1)
        self.assertIn("trace", payload)
        self.assertEqual(payload["trace"]["steps"][0]["step_type"], "retrieve")
        self.assertEqual(payload["ideas"][0]["object_type"], "idea_candidate")
        self.assertEqual(payload["evidence_registry"]["items"][0]["object_type"], "idea_candidate")
        progress.assert_called()


class ToolRegistryTests(unittest.TestCase):
    @mock.patch("hep_rag_v2.service.factory.create_facade")
    def test_tool_registry_registers_and_invokes_service_tools(self, create_facade_mock: mock.Mock) -> None:
        facade = mock.Mock()
        facade.retrieve.return_value = {"query": "CMS VBS SSWW", "works": []}
        create_facade_mock.return_value = facade

        registry = create_tool_registry({"retrieval": {}}, collection_name="default", limit=5)
        payload = registry.invoke("retrieve", query="CMS VBS SSWW")

        self.assertEqual(payload["query"], "CMS VBS SSWW")
        facade.retrieve.assert_called_once_with(
            query="CMS VBS SSWW",
            collection_name="default",
            target=None,
            limit=5,
            max_parallelism=None,
            model=None,
        )
        self.assertIn("retrieve", [tool.name for tool in registry.list_tools()])


if __name__ == "__main__":
    unittest.main()
