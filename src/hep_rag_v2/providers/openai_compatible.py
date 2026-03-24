from __future__ import annotations

from typing import Any
from urllib.parse import urljoin

import requests


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        api_base: str,
        api_key: str,
        model: str,
        chat_path: str = "/chat/completions",
        timeout_sec: int = 120,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.api_base = str(api_base).rstrip("/") + "/"
        self.api_key = str(api_key).strip()
        self.model = str(model).strip()
        self.chat_path = str(chat_path).strip() or "/chat/completions"
        self.timeout_sec = max(5, int(timeout_sec))
        self.extra_headers = dict(extra_headers or {})
        if not self.model:
            raise ValueError("LLM model is required.")


    def chat(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> dict[str, Any]:
        url = urljoin(self.api_base, self.chat_path.lstrip("/"))
        headers = {
            "Content-Type": "application/json",
            **self.extra_headers,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout_sec)
        response.raise_for_status()
        body = response.json()
        content = _extract_text(body)
        return {
            "model": self.model,
            "content": content,
            "raw": body,
        }


def _extract_text(body: dict[str, Any]) -> str:
    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError(f"OpenAI-compatible response missing choices: {body}")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = str(item.get("text") or "").strip()
                if text:
                    parts.append(text)
        if parts:
            return "\n".join(parts).strip()
    raise RuntimeError(f"OpenAI-compatible response missing text content: {body}")
