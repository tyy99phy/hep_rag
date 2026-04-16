from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable

import requests


PDG_ARTIFACT_CHOICES = ("full", "website", "book_pdf", "booklet_pdf", "sqlite")
PDG_SQLITE_VARIANTS = ("minimal", "all")
DEFAULT_PDG_TIMEOUT = 120
DEFAULT_PDG_RETRIES = 3
_SQLITE_VERSION_BY_EDITION = {
    "2024": "0.1.4",
}

_ARTIFACT_SPECS: dict[str, dict[str, str]] = {
    "website": {
        "slug": "website",
        "title": "PDG website corpus",
        "landing_url": "https://pdg.lbl.gov/{edition}/index.html",
        "download_url": "https://pdg.lbl.gov/{edition}/download/rpp-{edition}.zip",
        "file_name": "rpp-{edition}.zip",
    },
    "book_pdf": {
        "slug": "book-pdf",
        "title": "Review of Particle Physics ({edition}) Book PDF",
        "landing_url": "https://pdg.lbl.gov/{edition}/reviews/contents_sports.html",
        "download_url": "https://pdg.lbl.gov/{edition}/download/PhysRevD.110.030001.pdf",
        "file_name": "PhysRevD.110.030001.pdf",
    },
    "booklet_pdf": {
        "slug": "booklet-pdf",
        "title": "Review of Particle Physics ({edition}) Booklet PDF",
        "landing_url": "https://pdg.lbl.gov/{edition}/html/booklet.html",
        "download_url": "https://pdg.lbl.gov/{edition}/download/db{edition}.pdf",
        "file_name": "db{edition}.pdf",
    },
    "sqlite": {
        "slug": "sqlite",
        "title": "PDG SQLite database ({edition})",
        "landing_url": "https://pdg.lbl.gov/{edition}/api/index.html",
    },
}

ProgressCallback = Callable[[str], None] | None


def _emit_progress(progress: ProgressCallback, message: str) -> None:
    if progress is None:
        return
    text = str(message or "").strip()
    if text:
        progress(text)


def normalize_pdg_artifact(value: str | None) -> str:
    text = str(value or "full").strip().casefold().replace("-", "_")
    if text not in PDG_ARTIFACT_CHOICES:
        raise ValueError(f"Unsupported PDG artifact: {value}")
    return text


def normalize_pdg_sqlite_variant(value: str | None) -> str:
    text = str(value or "all").strip().casefold().replace("-", "_")
    if text not in PDG_SQLITE_VARIANTS:
        raise ValueError(f"Unsupported PDG sqlite variant: {value}")
    return text


def resolve_pdg_reference(
    *,
    edition: str | int,
    artifact: str = "website",
    sqlite_variant: str = "all",
) -> dict[str, Any]:
    normalized_edition = _normalize_edition(edition)
    normalized_artifact = normalize_pdg_artifact(artifact)
    if normalized_artifact == "full":
        raise ValueError("Artifact 'full' expands to multiple references; use resolve_pdg_references().")
    if normalized_artifact == "sqlite":
        return _resolve_sqlite_reference(normalized_edition, sqlite_variant=sqlite_variant)

    spec = _ARTIFACT_SPECS[normalized_artifact]
    file_name = spec["file_name"].format(edition=normalized_edition)
    canonical_id = f"pdg-{normalized_edition}-{spec['slug']}"
    return {
        "canonical_source": "pdg",
        "canonical_id": canonical_id,
        "edition": normalized_edition,
        "artifact_kind": normalized_artifact,
        "slug": spec["slug"],
        "title": spec["title"].format(edition=normalized_edition),
        "year": int(normalized_edition),
        "landing_url": spec["landing_url"].format(edition=normalized_edition),
        "download_url": spec["download_url"].format(edition=normalized_edition),
        "file_name": file_name,
    }


def resolve_pdg_references(
    *,
    edition: str | int,
    artifact: str = "full",
    sqlite_variant: str = "all",
) -> list[dict[str, Any]]:
    normalized_edition = _normalize_edition(edition)
    normalized_artifact = normalize_pdg_artifact(artifact)
    normalized_sqlite_variant = normalize_pdg_sqlite_variant(sqlite_variant)
    if normalized_artifact != "full":
        return [
            resolve_pdg_reference(
                edition=normalized_edition,
                artifact=normalized_artifact,
                sqlite_variant=normalized_sqlite_variant,
            )
        ]
    return [
        resolve_pdg_reference(edition=normalized_edition, artifact="website"),
        resolve_pdg_reference(edition=normalized_edition, artifact="sqlite", sqlite_variant=normalized_sqlite_variant),
        resolve_pdg_reference(edition=normalized_edition, artifact="book_pdf"),
        resolve_pdg_reference(edition=normalized_edition, artifact="booklet_pdf"),
    ]


def stage_pdg_artifact(
    reference: dict[str, Any],
    *,
    output_path: Path,
    source_path: str | Path | None = None,
    download: bool = False,
    timeout: int = DEFAULT_PDG_TIMEOUT,
    retries: int = DEFAULT_PDG_RETRIES,
    verify_ssl: bool = True,
    progress: ProgressCallback = None,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    local_path = Path(source_path).expanduser().resolve() if source_path is not None else None

    if local_path is not None:
        if not local_path.exists():
            raise FileNotFoundError(f"PDG artifact not found: {local_path}")
        if output_path.exists() and output_path.resolve() == local_path:
            _emit_progress(progress, f"using local PDG artifact in place: {output_path.name}")
            return {
                "ok": True,
                "state": "cached",
                "path": str(output_path),
                "source": "local",
                "source_path": str(local_path),
                "artifact_kind": str(reference.get("artifact_kind") or ""),
                "file_name": str(reference.get("file_name") or output_path.name),
            }
        _emit_progress(progress, f"copying local PDG artifact to workspace: {output_path.name}")
        shutil.copy2(local_path, output_path)
        return {
            "ok": True,
            "state": "copied",
            "path": str(output_path),
            "source": "local",
            "source_path": str(local_path),
            "artifact_kind": str(reference.get("artifact_kind") or ""),
            "file_name": str(reference.get("file_name") or output_path.name),
        }

    if output_path.exists():
        _emit_progress(progress, f"using cached PDG artifact: {output_path.name}")
        return {
            "ok": True,
            "state": "cached",
            "path": str(output_path),
            "source": "cached",
            "artifact_kind": str(reference.get("artifact_kind") or ""),
            "file_name": str(reference.get("file_name") or output_path.name),
        }

    download_url = str(reference.get("download_url") or "").strip()
    if not download:
        return {
            "ok": False,
            "state": "not_downloaded",
            "path": str(output_path),
            "source": "remote",
            "artifact_kind": str(reference.get("artifact_kind") or ""),
            "file_name": str(reference.get("file_name") or output_path.name),
            "url": download_url,
        }

    _download_artifact(
        reference,
        output_path=output_path,
        timeout=timeout,
        retries=retries,
        verify_ssl=verify_ssl,
        progress=progress,
    )
    return {
        "ok": True,
        "state": "downloaded",
        "path": str(output_path),
        "source": "remote",
        "artifact_kind": str(reference.get("artifact_kind") or ""),
        "file_name": str(reference.get("file_name") or output_path.name),
        "url": download_url,
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
    progress: ProgressCallback = None,
) -> dict[str, Any]:
    artifact_kind = str(reference.get("artifact_kind") or "").strip()
    if artifact_kind not in {"book_pdf", "booklet_pdf"}:
        raise ValueError(f"stage_pdg_pdf only supports PDF artifacts, got: {artifact_kind or 'unknown'}")
    payload = stage_pdg_artifact(
        reference,
        output_path=output_path,
        source_path=pdf_path,
        download=download,
        timeout=timeout,
        retries=retries,
        verify_ssl=verify_ssl,
        progress=progress,
    )
    payload["pdf_url"] = str(reference.get("download_url") or "")
    payload["pdf_name"] = str(reference.get("file_name") or output_path.name)
    return payload


def _resolve_sqlite_reference(edition: str, *, sqlite_variant: str) -> dict[str, Any]:
    normalized_variant = normalize_pdg_sqlite_variant(sqlite_variant)
    version = _SQLITE_VERSION_BY_EDITION.get(edition)
    if not version:
        raise ValueError(
            f"PDG SQLite version is not configured for edition {edition}. "
            "Use the official PDG API page to determine the current SQLite file name."
        )
    base_name = "pdgall" if normalized_variant == "all" else "pdg"
    file_name = f"{base_name}-{edition}-v{version}.sqlite"
    canonical_id = f"pdg-{edition}-sqlite-{normalized_variant}"
    description = "with historical Summary Table data" if normalized_variant == "all" else "minimal package data"
    return {
        "canonical_source": "pdg",
        "canonical_id": canonical_id,
        "edition": edition,
        "artifact_kind": "sqlite",
        "sqlite_variant": normalized_variant,
        "slug": f"sqlite-{normalized_variant}",
        "title": f"PDG SQLite database ({edition}, {description})",
        "year": int(edition),
        "landing_url": f"https://pdg.lbl.gov/{edition}/api/index.html",
        "download_url": f"https://pdg.lbl.gov/{edition}/api/{file_name}",
        "file_name": file_name,
    }


def _download_artifact(
    reference: dict[str, Any],
    *,
    output_path: Path,
    timeout: int,
    retries: int,
    verify_ssl: bool,
    progress: ProgressCallback = None,
) -> None:
    artifact_kind = str(reference.get("artifact_kind") or "").strip()
    url = str(reference.get("download_url") or "").strip()
    if not url:
        raise ValueError("PDG artifact URL is required for download.")
    headers = {
        "User-Agent": "hep-rag-v2/0.2 (+pdg importer)",
        "Accept": "*/*",
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
                first_chunk_captured = False
                total_bytes = int(response.headers.get("content-length") or 0) or None
                if total_bytes is not None:
                    _emit_progress(progress, f"downloading {output_path.name} ({_human_size(total_bytes)})...")
                else:
                    _emit_progress(progress, f"downloading {output_path.name}...")
                written = 0
                next_report = 0 if total_bytes else 64 * 1024 * 1024
                with temp_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 256):
                        if not chunk:
                            continue
                        if not first_chunk_captured:
                            first_chunk = bytes(chunk)
                            first_chunk_captured = True
                        handle.write(chunk)
                        written += len(chunk)
                        if total_bytes:
                            threshold = max(total_bytes // 20, 8 * 1024 * 1024)
                            if written >= next_report:
                                percent = min(100.0, (written / total_bytes) * 100.0)
                                _emit_progress(progress, f"{output_path.name}: {percent:.0f}% ({_human_size(written)}/{_human_size(total_bytes)})")
                                next_report = written + threshold
                        elif written >= next_report:
                            _emit_progress(progress, f"{output_path.name}: downloaded {_human_size(written)}")
                            next_report = written + 64 * 1024 * 1024
                if not _looks_like_expected_artifact(artifact_kind, response.headers.get("content-type"), first_chunk):
                    temp_path.unlink(missing_ok=True)
                    raise ValueError(f"downloaded content does not look like expected PDG artifact: {artifact_kind}")
                temp_path.replace(output_path)
                _emit_progress(progress, f"download complete: {output_path.name} ({_human_size(written)})")
                return
        except Exception as exc:
            last_error = exc
            output_path.with_suffix(output_path.suffix + ".part").unlink(missing_ok=True)
    if last_error is not None:
        raise last_error
    raise RuntimeError("PDG artifact download failed")


def _looks_like_expected_artifact(artifact_kind: str, content_type: str | None, first_chunk: bytes) -> bool:
    normalized_type = str(content_type or "").strip().casefold()
    head = bytes(first_chunk or b"")
    if artifact_kind in {"book_pdf", "booklet_pdf"}:
        return "pdf" in normalized_type or head.startswith(b"%PDF")
    if artifact_kind == "website":
        return "zip" in normalized_type or head.startswith(b"PK\x03\x04")
    if artifact_kind == "sqlite":
        return "sqlite" in normalized_type or head.startswith(b"SQLite format 3")
    return bool(head)


def _normalize_edition(value: str | int) -> str:
    text = str(value or "").strip()
    if len(text) != 4 or not text.isdigit():
        raise ValueError(f"Unsupported PDG edition: {value}")
    return text


def _human_size(num_bytes: int) -> str:
    size = float(max(0, int(num_bytes)))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"
