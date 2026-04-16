from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hep_rag_v2.config import default_config  # noqa: E402
from hep_rag_v2.rag import ask  # noqa: E402


class _QueuedClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def chat(self, **kwargs: object) -> dict[str, str]:
        self.calls.append(dict(kwargs))
        content = self.responses.pop(0)
        return {
            "model": "gpt-5.4",
            "content": content,
        }


class RagAnswerStrategyTests(unittest.TestCase):
    def test_ask_uses_community_map_reduce_when_overview_community_is_available(self) -> None:
        config = default_config()
        config["llm"]["enabled"] = True
        config["llm"]["backend"] = "openai_compatible"
        config["llm"]["api_base"] = "http://127.0.0.1:9999/v1"
        config["llm"]["api_key"] = "sk-test"
        config["llm"]["model"] = "gpt-5.4"

        retrieval_payload = {
            "query": "总结CMS jet tagging最新进展",
            "collection": "default",
            "search_scope": {"key": "default", "label": "Default", "collection_name": "default"},
            "requested_target": "works",
            "routing": {"target": "works", "graph_expand": 3, "reasons": ["default:result_lookup"]},
            "model": "sentence-transformers:BAAI/bge-small-en-v1.5",
            "community_summaries": [
                {
                    "summary_id": "community_summary:default:overview_jet",
                    "label": "CMS jet tagging community",
                    "summary_text": "Overview of CMS jet tagging literature and measurements.",
                    "community_level": "overview",
                    "work_count": 4,
                    "edge_count": 5,
                    "metadata": {
                        "member_work_ids": [11, 12],
                        "topics": ["jet tagging=2"],
                        "collaborations": ["CMS=2"],
                        "child_labels": ["CMS / jet tagging community", "CMS / heavy flavor tagging community"],
                    },
                    "representative_works": [{"title": "CMS jet tagging with graph neural networks", "year": 2025}],
                    "rank": 1,
                    "hybrid_score": 2.4,
                },
                {
                    "summary_id": "community_summary:default:fine_jet",
                    "label": "CMS / jet tagging community",
                    "summary_text": "Fine community focused on graph neural network taggers.",
                    "community_level": "fine",
                    "parent_summary_id": "community_summary:default:overview_jet",
                    "work_count": 2,
                    "edge_count": 2,
                    "metadata": {
                        "member_work_ids": [11],
                        "topics": ["jet tagging=1"],
                        "collaborations": ["CMS=1"],
                    },
                    "representative_works": [{"title": "CMS jet tagging with graph neural networks", "year": 2025}],
                    "rank": 2,
                    "hybrid_score": 2.1,
                },
            ],
            "ontology_summaries": [
                {
                    "summary_id": "ontology_summary:default:topic:jet_tagging",
                    "facet_kind": "topic",
                    "label": "jet tagging",
                    "summary_text": "Topic summary for jet tagging in CMS.",
                    "metadata": {"topics": ["jet tagging"]},
                    "work_count": 3,
                    "signal_count": 3,
                    "rank": 1,
                    "hybrid_score": 1.4,
                }
            ],
            "works": [
                {
                    "work_id": 11,
                    "raw_title": "CMS jet tagging with graph neural networks",
                    "canonical_source": "inspire",
                    "canonical_id": "9101",
                    "year": 2025,
                    "rank": 1,
                    "hybrid_score": 0.91,
                }
            ],
            "evidence_chunks": [
                {
                    "chunk_id": 201,
                    "work_id": 11,
                    "raw_title": "CMS jet tagging with graph neural networks",
                    "clean_text": "CMS reports graph neural network taggers with improved discrimination.",
                    "section_hint": "Results",
                    "rank": 1,
                    "hybrid_score": 0.88,
                }
            ],
        }
        client = _QueuedClient(
            [
                "Community Focus: CMS jet tagging cluster [G1][O1]\nEvidence-backed Points: GNN taggers dominate [W1][C1]\nOpen Gaps: calibration across taggers remains open [G2]",
                "CMS jet tagging has recently concentrated on graph-neural-network taggers and related calibration studies [G1][G2][O1][W1][C1].",
            ]
        )

        with (
            mock.patch("hep_rag_v2.rag.retrieve", return_value=retrieval_payload),
            mock.patch("hep_rag_v2.rag._build_llm_client", return_value=client),
        ):
            payload = ask(config, query="总结CMS jet tagging最新进展", mode="survey")

        self.assertEqual(payload["answer_strategy"], "community_map_reduce")
        self.assertEqual(len(payload["community_map_notes"]), 1)
        self.assertEqual(payload["community_map_notes"][0]["summary_id"], "community_summary:default:overview_jet")
        self.assertEqual(len(client.calls), 2)
        first_prompt = client.calls[0]["messages"][1]["content"]  # type: ignore[index]
        second_prompt = client.calls[1]["messages"][1]["content"]  # type: ignore[index]
        self.assertIn("当前要分析的社区簇", str(first_prompt))
        self.assertIn("[G1] CMS jet tagging community", str(first_prompt))
        self.assertIn("community map notes", str(second_prompt))

    def test_ask_falls_back_to_single_pass_without_overview_community(self) -> None:
        config = default_config()
        config["llm"]["enabled"] = True
        config["llm"]["backend"] = "openai_compatible"
        config["llm"]["api_base"] = "http://127.0.0.1:9999/v1"
        config["llm"]["api_key"] = "sk-test"
        config["llm"]["model"] = "gpt-5.4"

        retrieval_payload = {
            "query": "CMS SSWW",
            "collection": "default",
            "search_scope": {"key": "default", "label": "Default", "collection_name": "default"},
            "requested_target": "works",
            "routing": {"target": "works", "graph_expand": 0, "reasons": ["manual_target"]},
            "model": "hash-idf-v1",
            "community_summaries": [
                {
                    "summary_id": "community_summary:default:fine_only",
                    "label": "CMS / SSWW community",
                    "summary_text": "Fine community without overview parent.",
                    "community_level": "fine",
                    "work_count": 2,
                    "edge_count": 2,
                    "metadata": {"member_work_ids": [21]},
                    "rank": 1,
                    "hybrid_score": 1.2,
                }
            ],
            "ontology_summaries": [],
            "works": [
                {
                    "work_id": 21,
                    "raw_title": "Observation of electroweak production of same-sign W boson pairs",
                    "canonical_source": "inspire",
                    "canonical_id": "1624170",
                    "year": 2020,
                    "rank": 1,
                    "hybrid_score": 0.9,
                }
            ],
            "evidence_chunks": [],
        }
        client = _QueuedClient(["Single pass answer [W1]."])

        with (
            mock.patch("hep_rag_v2.rag.retrieve", return_value=retrieval_payload),
            mock.patch("hep_rag_v2.rag._build_llm_client", return_value=client),
        ):
            payload = ask(config, query="CMS SSWW", mode="answer")

        self.assertEqual(payload["answer_strategy"], "single_pass")
        self.assertEqual(payload["community_map_notes"], [])
        self.assertEqual(len(client.calls), 1)


if __name__ == "__main__":
    unittest.main()
