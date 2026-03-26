from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from hep_rag_v2.providers.mineru_api import MinerUClient


class MinerUApiTests(unittest.TestCase):
    def test_create_batch_upload_accepts_code_zero(self) -> None:
        client = MinerUClient(
            api_base="https://mineru.net/api/v4",
            api_token="token",
        )
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "code": 0,
            "msg": "ok",
            "data": {
                "batch_id": "batch-1",
                "file_urls": ["https://example.com/upload.pdf"],
            },
        }

        with mock.patch("hep_rag_v2.providers.mineru_api.requests.post", return_value=response):
            payload = client._create_batch_upload(pdf_path=Path("paper.pdf"), data_id="paper")

        self.assertEqual(payload["batch_id"], "batch-1")
        self.assertEqual(payload["file_urls"], ["https://example.com/upload.pdf"])

    def test_poll_batch_accepts_code_zero(self) -> None:
        client = MinerUClient(
            api_base="https://mineru.net/api/v4",
            api_token="token",
        )
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "code": 0,
            "msg": "ok",
            "data": {
                "batch_id": "batch-2",
                "state": "done",
                "data_id": "paper",
                "full_zip_url": "https://example.com/result.zip",
            },
        }

        with mock.patch("hep_rag_v2.providers.mineru_api.requests.get", return_value=response):
            result = client._poll_batch(batch_id="batch-2")

        self.assertEqual(result.batch_id, "batch-2")
        self.assertEqual(result.state, "done")
        self.assertEqual(result.full_zip_url, "https://example.com/result.zip")


if __name__ == "__main__":
    unittest.main()
