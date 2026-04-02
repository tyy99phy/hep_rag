from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.modules.pop("hep_rag_v2", None)
sys.path.insert(0, str(SRC))
importlib.invalidate_caches()

from hep_rag_v2.evidence import EvidenceRegistry
from hep_rag_v2.retrieval_adapter import adapt_chunk_hit, adapt_work_hit


class TestEvidenceRegistry(unittest.TestCase):
    def test_register_dedupes_identical_evidence_keys(self) -> None:
        registry = EvidenceRegistry()
        chunk = adapt_chunk_hit(
            {
                "chunk_id": 301,
                "work_id": 101,
                "chunk_role": "paragraph",
                "section_hint": "4 Results",
                "page_hint": "p.7",
                "clean_text": "Observed significance is 3.2 sigma in the signal region.",
                "raw_title": "Observation of a rare decay",
                "canonical_source": "inspire",
                "canonical_id": "101",
                "rank": 1,
            }
        )

        first = registry.register(chunk)
        second = registry.register(chunk)

        self.assertEqual(first.citation_id, "E1")
        self.assertEqual(second.citation_id, "E1")
        self.assertEqual(first.evidence_key, "chunk:301")
        self.assertEqual(len(registry), 1)
        self.assertEqual(registry.items()[0].occurrences, 2)

    def test_register_many_preserves_order_across_work_and_chunk_lanes(self) -> None:
        registry = EvidenceRegistry()
        work = adapt_work_hit(
            {
                "work_id": 101,
                "raw_title": "Observation of a rare decay",
                "abstract": "A measurement with strong evidence.",
                "canonical_source": "inspire",
                "canonical_id": "101",
                "rank": 1,
            }
        )
        chunk = adapt_chunk_hit(
            {
                "chunk_id": 301,
                "work_id": 101,
                "chunk_role": "paragraph",
                "section_hint": "4 Results",
                "page_hint": "p.7",
                "clean_text": "Observed significance is 3.2 sigma in the signal region.",
                "raw_title": "Observation of a rare decay",
                "canonical_source": "inspire",
                "canonical_id": "101",
                "rank": 1,
            }
        )

        citations = registry.register_many([work, chunk])

        self.assertEqual([item.citation_id for item in citations], ["E1", "E2"])
        self.assertEqual([item.result.object_type for item in registry.items()], ["work", "chunk"])
        rendered = registry.to_payload()
        self.assertEqual(rendered[0]["citation_id"], "E1")
        self.assertEqual(rendered[0]["object_type"], "work")
        self.assertEqual(rendered[1]["citation_id"], "E2")
        self.assertEqual(rendered[1]["section_hint"], "4 Results")


if __name__ == "__main__":
    unittest.main()
