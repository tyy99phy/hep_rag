from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hep_rag_v2 import paths


@pytest.fixture()
def workspace(tmp_path: Path):
    original = paths.workspace_root()
    try:
        paths.set_workspace_root(tmp_path)
        yield tmp_path
    finally:
        paths.set_workspace_root(original)
