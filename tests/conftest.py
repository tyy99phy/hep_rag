from __future__ import annotations

import contextlib
from pathlib import Path

import pytest

from hep_rag_v2 import paths


@pytest.fixture()
def workspace(tmp_path: Path):
    original = paths.workspace_root()
    try:
        paths.set_workspace_root(tmp_path)
        yield tmp_path
    finally:
        paths.set_workspace_root(original)
