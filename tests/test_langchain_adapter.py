from __future__ import annotations

import unittest
from unittest import mock

from langchain_core.messages import HumanMessage, SystemMessage

from hep_rag_v2.integrations.langchain_adapter import (
    HepRagChatModel,
    HepRagRetriever,
    build_langchain_document_audit_tool,
    build_langchain_document_tool,
    build_langchain_graph_tool,
    build_langchain_answer_chain,
    build_langchain_answer_runnable,
    build_langchain_retrieval_tool,
    build_langchain_toolkit,
    retrieval_to_documents,
)


class LangChainAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "llm": {
                "enabled": True,
                "backend": "openai_compatible",
                "api_base": "https://example.com/v1",
                "api_key": "EMPTY",
                "model": "gpt-5.4",
                "chat_path": "/chat/completions",
                "temperature": 0.1,
                "max_tokens": 256,
            }
        }
        self.retrieval_payload = {
            "query": "CMS VBS SSWW",
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

    def test_retrieval_to_documents_prefers_chunks(self) -> None:
        docs = retrieval_to_documents(self.retrieval_payload)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].metadata["source_type"], "chunk")
        self.assertEqual(docs[0].metadata["chunk_id"], 101)
        self.assertIn("significance", docs[0].page_content)

        work_docs = retrieval_to_documents(self.retrieval_payload, prefer_chunks=False)
        self.assertEqual(len(work_docs), 1)
        self.assertEqual(work_docs[0].metadata["source_type"], "work")
        self.assertEqual(work_docs[0].metadata["work_id"], 11)
        self.assertIn("vector boson scattering", work_docs[0].page_content)

    @mock.patch("hep_rag_v2.integrations.langchain_adapter.retrieve")
    def test_retriever_wraps_existing_pipeline(self, retrieve_mock: mock.Mock) -> None:
        retrieve_mock.return_value = self.retrieval_payload

        retriever = HepRagRetriever(
            config={"retrieval": {}},
            collection_name="default",
            target="works",
            limit=4,
            model="hash-idf-v1",
        )
        docs = retriever.invoke("CMS VBS SSWW")

        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].metadata["source_type"], "chunk")
        retrieve_mock.assert_called_once_with(
            {"retrieval": {}},
            query="CMS VBS SSWW",
            limit=4,
            target="works",
            collection_name="default",
            model="hash-idf-v1",
            progress=None,
        )

    @mock.patch("hep_rag_v2.integrations.langchain_adapter._build_llm_client")
    def test_chat_model_uses_existing_client_contract(self, build_client_mock: mock.Mock) -> None:
        client = mock.Mock()
        client.chat.return_value = {
            "model": "gpt-5.4",
            "content": "Answer body<STOP>ignored",
            "raw": {"id": "resp-1"},
        }
        build_client_mock.return_value = client

        model = HepRagChatModel(config=self.config)
        response = model.invoke(
            [SystemMessage(content="system"), HumanMessage(content="question")],
            stop=["<STOP>"],
        )

        self.assertEqual(response.content, "Answer body")
        client.chat.assert_called_once()
        call = client.chat.call_args.kwargs
        self.assertEqual(call["temperature"], 0.1)
        self.assertEqual(call["max_tokens"], 256)
        self.assertEqual(
            call["messages"],
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "question"},
            ],
        )

    @mock.patch("hep_rag_v2.integrations.langchain_adapter._build_llm_client")
    @mock.patch("hep_rag_v2.integrations.langchain_adapter.retrieve")
    def test_answer_runnable_returns_structured_payload(
        self,
        retrieve_mock: mock.Mock,
        build_client_mock: mock.Mock,
    ) -> None:
        retrieve_mock.return_value = self.retrieval_payload
        client = mock.Mock()
        client.chat.return_value = {
            "model": "gpt-5.4",
            "content": "Structured answer",
            "raw": {"id": "resp-2"},
        }
        build_client_mock.return_value = client

        runnable = build_langchain_answer_runnable(
            self.config,
            collection_name="default",
            limit=3,
        )
        payload = runnable.invoke({"query": "总结 CMS VBS SSWW 的最新结果", "mode": "survey"})

        self.assertEqual(payload["answer"], "Structured answer")
        self.assertEqual(payload["mode"], "survey")
        self.assertEqual(payload["retrieval"]["query"], "CMS VBS SSWW")
        self.assertEqual(len(payload["documents"]), 1)
        self.assertEqual(payload["documents"][0].metadata["source_type"], "chunk")
        retrieve_mock.assert_called_once_with(
            self.config,
            query="总结 CMS VBS SSWW 的最新结果",
            limit=3,
            target=None,
            collection_name="default",
            model=None,
            progress=None,
        )

    @mock.patch("hep_rag_v2.integrations.langchain_adapter._build_llm_client")
    @mock.patch("hep_rag_v2.integrations.langchain_adapter.retrieve")
    def test_answer_chain_returns_plain_text(
        self,
        retrieve_mock: mock.Mock,
        build_client_mock: mock.Mock,
    ) -> None:
        retrieve_mock.return_value = self.retrieval_payload
        client = mock.Mock()
        client.chat.return_value = {
            "model": "gpt-5.4",
            "content": "Plain answer",
            "raw": {"id": "resp-3"},
        }
        build_client_mock.return_value = client

        chain = build_langchain_answer_chain(self.config, collection_name="default")
        answer = chain.invoke("CMS VBS SSWW")

        self.assertEqual(answer, "Plain answer")

    @mock.patch("hep_rag_v2.integrations.langchain_adapter.retrieve")
    def test_retrieval_tool_uses_existing_pipeline(self, retrieve_mock: mock.Mock) -> None:
        retrieve_mock.return_value = self.retrieval_payload

        tool = build_langchain_retrieval_tool(
            {"retrieval": {}},
            collection_name="default",
            target="chunks",
            limit=2,
        )
        payload = tool.invoke("CMS VBS SSWW")

        self.assertEqual(payload["query"], "CMS VBS SSWW")
        retrieve_mock.assert_called_once_with(
            {"retrieval": {}},
            query="CMS VBS SSWW",
            limit=2,
            target="chunks",
            collection_name="default",
            model=None,
            progress=None,
        )

    @mock.patch("hep_rag_v2.integrations.langchain_adapter.show_graph_payload")
    def test_graph_tool_uses_graph_payload_helper(self, show_graph_payload_mock: mock.Mock) -> None:
        show_graph_payload_mock.return_value = {"neighbors": [{"neighbor_work_id": 12}]}

        tool = build_langchain_graph_tool(collection_name="default", similarity_model="hash-idf-v1", limit=5)
        payload = tool.invoke({"work_id": 11, "edge_kind": "similarity"})

        self.assertEqual(payload["neighbors"][0]["neighbor_work_id"], 12)
        show_graph_payload_mock.assert_called_once_with(
            work_id=11,
            id_type=None,
            id_value=None,
            edge_kind="similarity",
            collection="default",
            limit=5,
            similarity_model="hash-idf-v1",
        )

    @mock.patch("hep_rag_v2.integrations.langchain_adapter.show_document_payload")
    def test_document_tool_uses_document_payload_helper(self, show_document_payload_mock: mock.Mock) -> None:
        show_document_payload_mock.return_value = {"document": {"document_id": 7}}

        tool = build_langchain_document_tool(limit=9)
        payload = tool.invoke({"id_type": "inspire", "id_value": "1624170"})

        self.assertEqual(payload["document"]["document_id"], 7)
        show_document_payload_mock.assert_called_once_with(
            work_id=None,
            id_type="inspire",
            id_value="1624170",
            limit=9,
        )

    @mock.patch("hep_rag_v2.integrations.langchain_adapter.audit_document_payload")
    def test_document_audit_tool_uses_audit_helper(self, audit_document_payload_mock: mock.Mock) -> None:
        audit_document_payload_mock.return_value = {"ready": True}

        tool = build_langchain_document_audit_tool(limit=8)
        payload = tool.invoke({"work_id": 11})

        self.assertTrue(payload["ready"])
        audit_document_payload_mock.assert_called_once_with(
            work_id=11,
            id_type=None,
            id_value=None,
            limit=8,
        )

    def test_toolkit_exposes_expected_tools(self) -> None:
        tools = build_langchain_toolkit(
            self.config,
            collection_name="default",
            target="works",
            limit=4,
            model="hash-idf-v1",
        )
        names = [tool.name for tool in tools]

        self.assertEqual(
            names,
            [
                "hep_rag_retrieve",
                "hep_rag_answer",
                "hep_rag_graph_neighbors",
                "hep_rag_show_document",
                "hep_rag_audit_document",
            ],
        )

        tools_no_debug = build_langchain_toolkit(
            self.config,
            include_debug_tools=False,
        )
        self.assertEqual(
            [tool.name for tool in tools_no_debug],
            [
                "hep_rag_retrieve",
                "hep_rag_answer",
                "hep_rag_graph_neighbors",
                "hep_rag_show_document",
            ],
        )


if __name__ == "__main__":
    unittest.main()
