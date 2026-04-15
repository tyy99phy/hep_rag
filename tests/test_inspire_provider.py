from __future__ import annotations

import unittest
from unittest import mock

from hep_rag_v2.providers.inspire import list_pdf_candidates, search_literature


class InspireProviderTests(unittest.TestCase):
    def test_list_pdf_candidates_accepts_inspire_file_urls_for_notes(self) -> None:
        hit = {
            "metadata": {
                "documents": [
                    {
                        "filename": "SMP-20-006-pas.pdf",
                        "url": "https://inspirehep.net/files/2ebb251491d44fdc9253ffbc8b83ebd2",
                    }
                ]
            }
        }

        self.assertEqual(
            list_pdf_candidates(hit),
            [
                {
                    "url": "https://inspirehep.net/files/2ebb251491d44fdc9253ffbc8b83ebd2",
                    "source": "documents",
                }
            ],
        )

    def test_list_pdf_candidates_accepts_fulltext_documents_without_pdf_suffix(self) -> None:
        hit = {
            "metadata": {
                "documents": [
                    {
                        "filename": "document",
                        "fulltext": True,
                        "url": "https://inspirehep.net/files/7bac2dc822d774d497bfb677477dfe0a",
                    }
                ],
                "dois": [{"value": "10.22323/1.449.0322"}],
            }
        }

        candidates = list_pdf_candidates(hit)
        self.assertEqual(candidates[0]["source"], "documents")
        self.assertEqual(
            candidates[0]["url"],
            "https://inspirehep.net/files/7bac2dc822d774d497bfb677477dfe0a",
        )
        self.assertEqual(candidates[1]["source"], "doi")

    def test_search_literature_emits_page_progress(self) -> None:
        payloads = [
            {
                "hits": {
                    "total": 3,
                    "hits": [
                        {"metadata": {"control_number": 1}},
                        {"metadata": {"control_number": 2}},
                    ],
                },
                "links": {"next": "https://inspirehep.net/api/literature?page=2"},
            },
            {
                "hits": {
                    "total": 3,
                    "hits": [
                        {"metadata": {"control_number": 3}},
                    ],
                },
                "links": {},
            },
        ]
        messages: list[str] = []

        with mock.patch("hep_rag_v2.providers.inspire._http_get_json", side_effect=payloads):
            hits = search_literature(
                "collaboration:CMS",
                limit=3,
                page_size=2,
                progress=messages.append,
                progress_label="INSPIRE query 1/1",
                sleep_sec=0.0,
            )

        self.assertEqual([hit["metadata"]["control_number"] for hit in hits], [1, 2, 3])
        self.assertEqual(
            messages,
            [
                "INSPIRE query 1/1 page 1/2 fetched 2 hits (2/3 accumulated).",
                "INSPIRE query 1/1 page 2/2 fetched 1 hits (3/3 accumulated).",
            ],
        )


if __name__ == "__main__":
    unittest.main()
