from __future__ import annotations

import hashlib
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import requests

from hep_rag_v2.metadata import first_arxiv_id, first_doi, first_title, year_from_metadata


INSPIRE_API = "https://inspirehep.net/api/literature"


def build_search_query(
    query: str,
    *,
    published_only: bool = False,
    query_suffix: str = "",
) -> str:
    parts = [str(query or "").strip()]
    if published_only and "collection:" not in parts[0].casefold():
        parts.append("collection:Published")
    if query_suffix:
        parts.append(str(query_suffix).strip())
    return " and ".join(part for part in parts if part)


def search_literature(
    query: str,
    *,
    limit: int = 20,
    page_size: int = 25,
    fields: list[str] | None = None,
    published_only: bool = False,
    query_suffix: str = "",
    timeout: int = 60,
    retries: int = 3,
    sleep_sec: float = 0.2,
) -> list[dict[str, Any]]:
    page_size = max(1, min(int(page_size), 50))
    effective_query = build_search_query(
        query,
        published_only=published_only,
        query_suffix=query_suffix,
    )
    hits: list[dict[str, Any]] = []
    params: dict[str, Any] | None = {
        "q": effective_query,
        "size": page_size,
        "page": 1,
    }
    if fields:
        params["fields"] = ",".join(fields)

    url = INSPIRE_API
    while len(hits) < max(1, limit):
        payload = _http_get_json(url, params=params, timeout=timeout, retries=retries)
        batch = (((payload.get("hits") or {}).get("hits")) or [])
        if not batch:
            break
        hits.extend(batch)
        if len(hits) >= limit:
            break
        next_url = str((payload.get("links") or {}).get("next") or "").strip()
        if not next_url:
            break
        url = next_url
        params = None
        if sleep_sec > 0:
            time.sleep(sleep_sec)
    return hits[:limit]


def summarize_hit(hit: dict[str, Any]) -> dict[str, Any]:
    metadata = hit.get("metadata") or {}
    pdf_candidates = list_pdf_candidates(hit)
    identifiers = _identity_map(metadata)
    return {
        "title": first_title(metadata),
        "year": year_from_metadata(metadata),
        "identifiers": identifiers,
        "pdf_candidates": pdf_candidates,
        "pdf_candidate_count": len(pdf_candidates),
        "source_url": str((hit.get("links") or {}).get("self") or "") or None,
    }


def list_pdf_candidates(
    hit: dict[str, Any],
    *,
    resolve_arxiv_from_doi: bool = False,
    timeout: int = 10,
    retries: int = 3,
) -> list[dict[str, str]]:
    metadata = hit.get("metadata") or hit
    candidates: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(url: str | None, source: str) -> None:
        value = str(url or "").strip()
        if not value or value in seen:
            return
        seen.add(value)
        candidates.append({"url": value, "source": source})

    for item in metadata.get("documents") or []:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        if _is_pdf_document_candidate(item, url):
            add(url, "documents")

    for item in metadata.get("files") or []:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or item.get("file") or item.get("path") or "").strip()
        if _is_pdf_file_candidate(item, url):
            add(url, "files")

    arxiv_id = first_arxiv_id(metadata)
    doi = first_doi(metadata)
    if not arxiv_id and doi and resolve_arxiv_from_doi:
        arxiv_id = doi_to_arxiv(doi, timeout=timeout, retries=retries)
    if arxiv_id:
        add(f"https://arxiv.org/pdf/{arxiv_id}.pdf", "arxiv")

    if doi:
        add(f"https://doi.org/{doi}", "doi")

    return candidates


def doi_to_arxiv(
    doi: str,
    *,
    timeout: int = 10,
    retries: int = 3,
) -> str | None:
    value = str(doi or "").strip()
    if not value:
        return None

    for attempt in range(1, max(1, retries) + 1):
        try:
            response = requests.get(
                "http://export.arxiv.org/api/query",
                params={"search_query": f"doi:{value}"},
                headers={"User-Agent": "hep-rag-v2/0.2 (+doi-to-arxiv)"},
                timeout=timeout,
            )
            response.raise_for_status()
            root = ET.fromstring(response.text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            entry = root.find("atom:entry", ns)
            if entry is None:
                return None
            node = entry.find("atom:id", ns)
            if node is None or not node.text:
                return None
            return node.text.rsplit("/", 1)[-1].strip() or None
        except Exception:
            time.sleep(min(0.5 * attempt, 2.0))
    return None


def download_pdf_candidates(
    candidates: list[dict[str, str]],
    *,
    output_path: Path,
    timeout: int = 120,
    retries: int = 3,
    verify_ssl: bool = True,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    for candidate in candidates:
        url = str(candidate.get("url") or "").strip()
        source = str(candidate.get("source") or "").strip() or "unknown"
        if not url:
            continue
        try:
            _download_pdf(url, output_path=output_path, timeout=timeout, retries=retries, verify_ssl=verify_ssl)
            return {
                "ok": True,
                "url": url,
                "source": source,
                "path": str(output_path),
            }
        except Exception as exc:
            errors.append(f"{source}:{url} -> {exc}")

    return {
        "ok": False,
        "path": str(output_path),
        "errors": errors,
    }


def content_addressed_name(hit: dict[str, Any]) -> str:
    metadata = hit.get("metadata") or {}
    identities = _identity_map(metadata)
    for key in ("arxiv", "inspire", "doi"):
        value = str(identities.get(key) or "").strip()
        if value:
            return _safe_filename(value)
    fingerprint = hashlib.sha1(
        f"{first_title(metadata) or ''}|{year_from_metadata(metadata) or ''}".encode("utf-8")
    ).hexdigest()[:16]
    return fingerprint


def _download_pdf(
    url: str,
    *,
    output_path: Path,
    timeout: int,
    retries: int,
    verify_ssl: bool,
) -> None:
    headers = {
        "User-Agent": "hep-rag-v2/0.2 (+pdf downloader)",
        "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.1",
    }
    last_error: Exception | None = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            with requests.get(
                url,
                headers=headers,
                timeout=timeout,
                stream=True,
                allow_redirects=True,
                verify=verify_ssl,
            ) as response:
                response.raise_for_status()
                temp_path = output_path.with_suffix(output_path.suffix + ".part")
                first_chunk = b""
                with temp_path.open("wb") as handle:
                    for idx, chunk in enumerate(response.iter_content(chunk_size=1024 * 256)):
                        if not chunk:
                            continue
                        if idx == 0:
                            first_chunk = bytes(chunk)
                        handle.write(chunk)
                if not _looks_like_pdf_response(response.headers.get("content-type"), first_chunk):
                    temp_path.unlink(missing_ok=True)
                    raise ValueError("downloaded content is not a PDF")
                temp_path.replace(output_path)
                return
        except Exception as exc:
            last_error = exc
            time.sleep(min(1.5 * attempt, 5.0))
    raise RuntimeError(f"failed to download PDF: {last_error}")


def _http_get_json(
    url: str,
    *,
    params: dict[str, Any] | None,
    timeout: int,
    retries: int,
) -> dict[str, Any]:
    headers = {
        "User-Agent": "hep-rag-v2/0.2 (+online search)",
        "Accept": "application/json",
    }
    last_error: Exception | None = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            time.sleep(min(1.5 * attempt, 5.0))
    raise RuntimeError(f"failed to fetch INSPIRE payload: {last_error}")


def _looks_like_pdf_url(url: str) -> bool:
    value = str(url or "").casefold()
    return value.endswith(".pdf") or ".pdf?" in value or "/pdf/" in value or "arxiv.org/pdf/" in value


def _looks_like_inspire_file_url(url: str) -> bool:
    value = str(url or "").casefold()
    return "inspirehep.net/files/" in value


def _is_pdf_document_candidate(item: dict[str, Any], url: str) -> bool:
    if _looks_like_pdf_url(url) or _looks_like_inspire_file_url(url):
        return True
    filename = str(item.get("filename") or "").casefold()
    if filename.endswith(".pdf"):
        return True
    description = str(item.get("description") or "").casefold()
    if "pdf" in description or "fulltext" in description:
        return True
    return bool(item.get("fulltext"))


def _is_pdf_file_candidate(item: dict[str, Any], url: str) -> bool:
    if _looks_like_pdf_url(url) or _looks_like_inspire_file_url(url):
        return True
    for key in ("filename", "file", "path"):
        value = str(item.get(key) or "").casefold()
        if value.endswith(".pdf"):
            return True
    return False


def _looks_like_pdf_response(content_type: str | None, first_chunk: bytes) -> bool:
    ctype = str(content_type or "").casefold()
    if "pdf" in ctype:
        return True
    return first_chunk.startswith(b"%PDF-")


def _identity_map(metadata: dict[str, Any]) -> dict[str, str | None]:
    control_number = metadata.get("control_number")
    inspire = str(control_number).strip() if control_number is not None else None
    return {
        "inspire": inspire,
        "arxiv": first_arxiv_id(metadata),
        "doi": first_doi(metadata),
    }


def _safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value)).strip("_") or "paper"
