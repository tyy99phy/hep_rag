from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


@dataclass(frozen=True)
class MinerUTaskResult:
    batch_id: str
    data_id: str | None
    state: str
    full_zip_url: str | None
    raw_payload: dict[str, Any]


class MinerUClient:
    def __init__(
        self,
        *,
        api_base: str,
        api_token: str,
        model_version: str = "pipeline",
        is_ocr: bool = False,
        enable_formula: bool = True,
        enable_table: bool = True,
        language: str = "en",
        poll_interval_sec: int = 10,
        max_wait_sec: int = 1800,
        timeout_sec: int = 120,
    ) -> None:
        self.api_base = str(api_base).rstrip("/")
        self.api_token = str(api_token).strip()
        self.model_version = str(model_version).strip() or "pipeline"
        self.is_ocr = bool(is_ocr)
        self.enable_formula = bool(enable_formula)
        self.enable_table = bool(enable_table)
        self.language = str(language).strip() or "en"
        self.poll_interval_sec = max(1, int(poll_interval_sec))
        self.max_wait_sec = max(self.poll_interval_sec, int(max_wait_sec))
        self.timeout_sec = max(5, int(timeout_sec))
        if not self.api_token:
            raise ValueError("MinerU api_token is required.")


    def submit_local_pdf(self, pdf_path: Path, *, data_id: str | None = None) -> MinerUTaskResult:
        upload_meta = self._create_batch_upload(pdf_path=pdf_path, data_id=data_id)
        urls = list(upload_meta["file_urls"])
        if not urls:
            raise RuntimeError("MinerU returned no upload URLs.")
        self._upload_binary(urls[0], pdf_path)
        result = self._poll_batch(batch_id=str(upload_meta["batch_id"]))
        if result.state != "done":
            raise RuntimeError(f"MinerU parse did not finish successfully: state={result.state}")
        return result


    def download_result_zip(self, task: MinerUTaskResult, *, output_path: Path) -> Path:
        if not task.full_zip_url:
            raise ValueError("MinerU task has no full_zip_url.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(task.full_zip_url, timeout=self.timeout_sec, stream=True) as response:
            response.raise_for_status()
            temp_path = output_path.with_suffix(output_path.suffix + ".part")
            with temp_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        handle.write(chunk)
            temp_path.replace(output_path)
        return output_path


    def _create_batch_upload(self, *, pdf_path: Path, data_id: str | None) -> dict[str, Any]:
        url = f"{self.api_base}/file-urls/batch"
        payload = {
            "files": [
                {
                    "name": pdf_path.name,
                    "data_id": data_id or _safe_data_id(pdf_path.stem),
                }
            ],
            "model_version": self.model_version,
        }
        file_info = payload["files"][0]
        file_info["is_ocr"] = self.is_ocr
        file_info["enable_formula"] = self.enable_formula
        file_info["enable_table"] = self.enable_table
        file_info["language"] = self.language

        response = requests.post(
            url,
            headers=self._headers(),
            json=payload,
            timeout=self.timeout_sec,
        )
        response.raise_for_status()
        body = response.json()
        if _payload_code(body) != 0:
            raise RuntimeError(f"MinerU create batch failed: {body}")
        data = body.get("data") or {}
        batch_id = str(data.get("batch_id") or "").strip()
        file_urls = data.get("file_urls") or []
        if not batch_id:
            raise RuntimeError(f"MinerU create batch missing batch_id: {body}")
        return {
            "batch_id": batch_id,
            "file_urls": file_urls,
            "raw_payload": body,
        }


    def _upload_binary(self, url: str, pdf_path: Path) -> None:
        with pdf_path.open("rb") as handle:
            response = requests.put(url, data=handle, timeout=self.timeout_sec)
        response.raise_for_status()


    def _poll_batch(self, *, batch_id: str) -> MinerUTaskResult:
        url = f"{self.api_base}/extract-results/batch/{batch_id}"
        deadline = time.monotonic() + float(self.max_wait_sec)
        last_payload: dict[str, Any] | None = None

        while time.monotonic() <= deadline:
            try:
                response = requests.get(url, headers=self._headers(), timeout=self.timeout_sec)
                response.raise_for_status()
                payload = response.json()
                last_payload = payload
                if _payload_code(payload) != 0:
                    raise RuntimeError(f"MinerU batch status failed: {payload}")
                result = _extract_result(payload)
                if result.state in {"done", "failed"}:
                    return result
            except requests.RequestException as exc:
                last_payload = {"request_error": str(exc)}
            time.sleep(self.poll_interval_sec)

        raise TimeoutError(f"MinerU batch polling timed out: batch_id={batch_id}, last_payload={last_payload}")


    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_token}",
        }


def _extract_result(payload: dict[str, Any]) -> MinerUTaskResult:
    data = payload.get("data") or {}
    result = data.get("extract_result")
    if isinstance(result, dict):
        state = str(result.get("state") or "").strip()
        return MinerUTaskResult(
            batch_id=str(data.get("batch_id") or ""),
            data_id=str(result.get("data_id") or "").strip() or None,
            state=state or "unknown",
            full_zip_url=str(result.get("full_zip_url") or "").strip() or None,
            raw_payload=payload,
        )
    if isinstance(result, list) and result:
        first = result[0] if isinstance(result[0], dict) else {}
        state = str(first.get("state") or "").strip()
        return MinerUTaskResult(
            batch_id=str(data.get("batch_id") or ""),
            data_id=str(first.get("data_id") or "").strip() or None,
            state=state or "unknown",
            full_zip_url=str(first.get("full_zip_url") or "").strip() or None,
            raw_payload=payload,
        )

    if isinstance(data, dict) and data.get("state"):
        state = str(data.get("state") or "").strip()
        return MinerUTaskResult(
            batch_id=str(data.get("batch_id") or ""),
            data_id=str(data.get("data_id") or "").strip() or None,
            state=state or "unknown",
            full_zip_url=str(data.get("full_zip_url") or "").strip() or None,
            raw_payload=payload,
        )

    results = data.get("extract_results") or data.get("results") or []
    if isinstance(results, list) and results:
        first = results[0] if isinstance(results[0], dict) else {}
        state = str(first.get("state") or "").strip()
        return MinerUTaskResult(
            batch_id=str(data.get("batch_id") or ""),
            data_id=str(first.get("data_id") or "").strip() or None,
            state=state or "unknown",
            full_zip_url=str(first.get("full_zip_url") or "").strip() or None,
            raw_payload=payload,
        )

    raise RuntimeError(f"Unrecognized MinerU batch payload: {payload}")


def _safe_data_id(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))
    return cleaned[:128].strip("_") or "paper"


def _payload_code(payload: dict[str, Any]) -> int:
    try:
        return int(payload.get("code", -1))
    except (TypeError, ValueError):
        return -1
