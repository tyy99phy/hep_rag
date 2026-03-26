from __future__ import annotations

import unittest

from hep_rag_v2.providers.inspire import list_pdf_candidates


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


if __name__ == "__main__":
    unittest.main()
