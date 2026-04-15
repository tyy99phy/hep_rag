from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any


ALL_SCOPE_KEY = "all"
ALL_SCOPE_LABEL = "All Collections"
_ALL_SCOPE_ALIASES = {"all", "*", "__all__", "all_collections"}
_COLLECTION_PREFIX = "collection:"


@dataclass(frozen=True)
class SearchScope:
    kind: str
    key: str
    label: str
    collection_name: str | None = None
    is_writable: bool = False

    def to_payload(self, **extra: Any) -> dict[str, Any]:
        payload = {
            "key": self.key,
            "kind": self.kind,
            "label": self.label,
            "collection_name": self.collection_name,
            "is_writable": self.is_writable,
        }
        payload.update(extra)
        return payload


def normalize_search_scope(raw: str | None) -> SearchScope:
    text = str(raw or "").strip()
    if not text or text.lower() in _ALL_SCOPE_ALIASES:
        return SearchScope(
            kind="all",
            key=ALL_SCOPE_KEY,
            label=ALL_SCOPE_LABEL,
            collection_name=None,
            is_writable=False,
        )

    if text.lower().startswith(_COLLECTION_PREFIX):
        text = text.split(":", 1)[1].strip()
        if not text:
            return SearchScope(
                kind="all",
                key=ALL_SCOPE_KEY,
                label=ALL_SCOPE_LABEL,
                collection_name=None,
                is_writable=False,
            )

    return SearchScope(
        kind="collection",
        key=f"{_COLLECTION_PREFIX}{text}",
        label=text,
        collection_name=text,
        is_writable=True,
    )


def available_search_scopes(
    conn: sqlite3.Connection,
    *,
    snapshot: dict[str, int] | None = None,
    collections: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    collection_rows = collections if collections is not None else _collection_rows(conn)
    total_works = int((snapshot or {}).get("works") or 0)
    scopes = [
        SearchScope(
            kind="all",
            key=ALL_SCOPE_KEY,
            label=ALL_SCOPE_LABEL,
            collection_name=None,
            is_writable=False,
        ).to_payload(works=total_works)
    ]
    for item in collection_rows:
        name = str(item.get("collection") or "").strip()
        if not name:
            continue
        scopes.append(
            SearchScope(
                kind="collection",
                key=f"{_COLLECTION_PREFIX}{name}",
                label=name,
                collection_name=name,
                is_writable=True,
            ).to_payload(works=int(item.get("works") or 0))
        )
    return scopes


def _collection_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in conn.execute(
            """
            SELECT c.name AS collection, COUNT(cw.work_id) AS works
            FROM collections c
            LEFT JOIN collection_works cw ON cw.collection_id = c.collection_id
            GROUP BY c.collection_id, c.name
            ORDER BY c.name
            """
        )
    ]
