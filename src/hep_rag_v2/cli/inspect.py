from __future__ import annotations

import argparse
import json
import sqlite3
from typing import Any

from hep_rag_v2.service.inspect import (
    AUDIT_PATTERNS,
    READINESS_THRESHOLDS,
    audit_document_payload,
    show_document_payload,
)


def cmd_show_document(args: argparse.Namespace) -> None:
    payload = show_document_payload(
        work_id=args.work_id,
        id_type=args.id_type,
        id_value=args.id_value,
        limit=args.limit,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def cmd_audit_document(args: argparse.Namespace) -> None:
    payload = audit_document_payload(
        work_id=args.work_id,
        id_type=args.id_type,
        id_value=args.id_value,
        limit=args.limit,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _audit_document_payload(conn: sqlite3.Connection, *, document_id: int, limit: int) -> dict[str, Any]:
    from hep_rag_v2.service.inspect import _audit_document_payload as service_audit_document_payload

    return service_audit_document_payload(conn, document_id=document_id, limit=limit)


__all__ = [
    "AUDIT_PATTERNS",
    "READINESS_THRESHOLDS",
    "_audit_document_payload",
    "cmd_audit_document",
    "cmd_show_document",
]
