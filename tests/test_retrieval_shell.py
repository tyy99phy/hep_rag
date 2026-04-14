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
from hep_rag_v2.retrieval_adapter import build_retrieval_shell
from hep_rag_v2.service.facade import HepRagServiceFacade
from hep_rag_v2.tools.registry import build_default_tool_registry


class RetrievalShellTests(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = {
            "query": "CMS VBS SSWW",
            "collection": "default",
            "works": [
                {
                    "work_id": 11,
                    "raw_title": "Observation of electroweak production of same-sign W boson pairs",
                    "abstract": "CMS observes same-sign WW production via vector boson scattering.",
                    "canonical_source": "inspire",
                    "canonical_id": "1624170",
                    "rank": 1,
                    "hybrid_score": 0.91,
                    "year": 2020,
                }
            ],
            "evidence_chunks": [
                {
                    "chunk_id": 101,
                    "work_id": 11,
                    "raw_title": "Observation of electroweak production of same-sign W boson pairs",
                    "clean_text": "The observed significance exceeds the standard model background expectation.",
                    "section_hint": "Results",
                    "chunk_role": "body_chunk",
                    "canonical_source": "inspire",
                    "canonical_id": "1624170",
                    "rank": 1,
                    "hybrid_score": 0.87,
                }
            ],
        }

    def test_build_retrieval_shell_adds_uniform_results_and_citations(self) -> None:
        shell = build_retrieval_shell(self.payload)

        self.assertEqual(shell["query"], "CMS VBS SSWW")
        self.assertEqual([item["source_type"] for item in shell["results"]], ["chunk", "work"])
        self.assertEqual(shell["results"][0]["text"], self.payload["evidence_chunks"][0]["clean_text"])
        self.assertEqual(shell["results"][1]["text"], self.payload["works"][0]["abstract"])
        self.assertEqual(
            [item["citation"] for item in shell["evidence_registry"]["items"]],
            ["[chunk:101]", "[work:11]"],
        )

    def test_evidence_registry_dedupes_repeated_results(self) -> None:
        shell = build_retrieval_shell(self.payload)
        registry = EvidenceRegistry()

        first = registry.register(shell["results"][0])
        second = registry.register(shell["results"][0])

        self.assertEqual(first["citation"], "[chunk:101]")
        self.assertEqual(second["citation"], "[chunk:101]")
        self.assertEqual(len(registry.export()["items"]), 1)

    def test_service_facade_wraps_retrieve_with_uniform_shell(self) -> None:
        with mock.patch("hep_rag_v2.service.facade.pipeline_retrieve", return_value=self.payload) as retrieve_mock:
            facade = HepRagServiceFacade(config={"retrieval": {}})
            payload = facade.retrieve(query="CMS VBS SSWW", limit=4, target="works", collection_name="default")

        self.assertEqual([item["source_type"] for item in payload["results"]], ["chunk", "work"])
        retrieve_mock.assert_called_once_with(
            {"retrieval": {}},
            query="CMS VBS SSWW",
            limit=4,
            target="works",
            collection_name="default",
            max_parallelism=None,
            model=None,
            progress=None,
        )

    def test_default_tool_registry_exposes_retrieve_and_answer(self) -> None:
        facade = mock.Mock()
        facade.retrieve.return_value = {"query": "q", "results": []}
        facade.ask.return_value = {"query": "q", "answer": "ok"}
        facade.generate_ideas.return_value = {"query": "q", "ideas": []}
        facade.workspace_status.return_value = {"snapshot": {"works": 0}}

        registry = build_default_tool_registry(facade)

        self.assertEqual(sorted(registry.names()), ["ask", "generate_ideas", "retrieve", "workspace_status"])
        self.assertEqual(registry.invoke("retrieve", query="q"), {"query": "q", "results": []})
        self.assertEqual(registry.invoke("ask", query="q"), {"query": "q", "answer": "ok"})
        self.assertEqual(registry.invoke("generate_ideas", query="q"), {"query": "q", "ideas": []})
        self.assertEqual(registry.invoke("workspace_status"), {"snapshot": {"works": 0}})


if __name__ == "__main__":
    unittest.main()
