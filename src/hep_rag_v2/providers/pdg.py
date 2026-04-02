from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import requests


DEFAULT_PDG_SLUG = "review-of-particle-physics"
DEFAULT_PDG_TIMEOUT = 120
DEFAULT_PDG_RETRIES = 3


def resolve_pdg_reference(
    *,
    edition: str | int,
    slug: str = DEFAULT_PDG_SLUG,
) -> dict[str, Any]:
    normalized_edition = _normalize_edition(edition)
    normalized_slug = _normalize_slug(slug)
    if normalized_slug != DEFAULT_PDG_SLUG:
        raise ValueError(f"Unsupported PDG slug: {slug}")

    canonical_id = f"pdg-{normalized_edition}-{normalized_slug}"
    pdf_name = f"rpp{normalized_edition}-rev-intro.pdf"
    return {
        "canonical_source": "pdg",
        "canonical_id": canonical_id,
        "edition": normalized_edition,
        "slug": normalized_slug,
        "title": f"Review of Particle Physics ({normalized_edition})",
        "year": int(normalized_edition),
        "landing_url": f"https://pdg.lbl.gov/{normalized_edition}/reviews/contents_sports.html",
        "pdf_url": f"https://pdg.lbl.gov/{normalized_edition}/reviews/{pdf_name}",
        "pdf_name": pdf_name,
    }


def stage_pdg_pdf(
    reference: dict[str, Any],
    *,
    output_path: Path,
    pdf_path: str | Path | None = None,
    download: bool = False,
    timeout: int = DEFAULT_PDG_TIMEOUT,
    retries: int = DEFAULT_PDG_RETRIES,
    verify_ssl: bool = True,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    local_path = Path(pdf_path).expanduser().resolve() if pdf_path is not None else None

    if local_path is not None:
        if not local_path.exists():
            raise FileNotFoundError(f"PDG PDF not found: {local_path}")
        if output_path.exists() and output_path.resolve() == local_path:
            return {
                "ok": True,
                "state": "cached",
                "path": str(output_path),
                "source": "local",
                "source_path": str(local_path),
            }
        shutil.copy2(local_path, output_path)
        return {
            "ok": True,
            "state": "copied",
            "path": str(output_path),
            "source": "local",
            "source_path": str(local_path),
        }

    if output_path.exists():
        return {
            "ok": True,
            "state": "cached",
            "path": str(output_path),
            "source": "cached",
        }

    pdf_url = str(reference.get("pdf_url") or "").strip()
    if not download:
        return {
            "ok": False,
            "state": "not_downloaded",
            "path": str(output_path),
            "source": "remote",
            "url": pdf_url,
        }

    _download_pdf(pdf_url, output_path=output_path, timeout=timeout, retries=retries, verify_ssl=verify_ssl)
    return {
        "ok": True,
        "state": "downloaded",
        "path": str(output_path),
        "source": "remote",
        "url": pdf_url,
    }


def _download_pdf(
    url: str,
    *,
    output_path: Path,
    timeout: int,
    retries: int,
    verify_ssl: bool,
) -> None:
    if not str(url or "").strip():
        raise ValueError("PDG PDF URL is required for download.")
    headers = {
        "User-Agent": "hep-rag-v2/0.2 (+pdg importer)",
        "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.1",
    }
    last_error: Exception | None = None
    for _attempt in range(1, max(1, retries) + 1):
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
                if not _looks_like_pdf(response.headers.get("content-type"), first_chunk):
                    temp_path.unlink(missing_ok=True)
                    raise ValueError("downloaded content is not a PDF")
                temp_path.replace(output_path)
                return
        except Exception as exc:
            last_error = exc
            output_path.with_suffix(output_path.suffix + ".part").unlink(missing_ok=True)
    if last_error is not None:
        raise last_error
    raise RuntimeError("PDG PDF download failed")


def _looks_like_pdf(content_type: str | None, first_chunk: bytes) -> bool:
    normalized = str(content_type or "").strip().casefold()
    if "pdf" in normalized:
        return True
    return bytes(first_chunk or b"").startswith(b"%PDF")


def _normalize_edition(value: str | int) -> str:
    text = str(value or "").strip()
    if len(text) != 4 or not text.isdigit():
        raise ValueError(f"Unsupported PDG edition: {value}")
    return text


def _normalize_slug(value: str) -> str:
    return "-".join(token for token in str(value or "").strip().casefold().replace("_", "-").split("-") if token)
