from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hep_rag_v2.providers.openai_compatible import _extract_text


def test_extract_text_accepts_responses_output_items() -> None:
    body = {
        "id": "resp_test",
        "object": "response",
        "model": "gpt-5.4",
        "output": [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "Hello from responses API."},
                ],
            }
        ],
    }

    assert _extract_text(body) == "Hello from responses API."


def test_extract_text_reports_actionable_error_for_empty_text() -> None:
    body = {
        "id": "resp_test",
        "object": "chat.completion",
        "model": "gpt-5.4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": None,
                    "tool_calls": None,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"completion_tokens": 10, "total_tokens": 20, "prompt_tokens": 10},
    }

    with pytest.raises(RuntimeError) as exc_info:
        _extract_text(body)

    message = str(exc_info.value)
    assert "LLM endpoint returned no assistant text." in message
    assert "llm.api_base" in message
    assert "gpt-5.4" in message
