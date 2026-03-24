from __future__ import annotations

from functools import lru_cache
from typing import Any


class LocalTransformersClient:
    def __init__(
        self,
        *,
        model_name_or_path: str,
        device: str = "cpu",
        torch_dtype: str = "auto",
        trust_remote_code: bool = False,
    ) -> None:
        self.model_name_or_path = str(model_name_or_path).strip()
        self.device = str(device).strip() or "cpu"
        self.torch_dtype = str(torch_dtype).strip() or "auto"
        self.trust_remote_code = bool(trust_remote_code)
        if not self.model_name_or_path:
            raise ValueError("local_model_path is required for local_transformers backend.")

    def chat(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> dict[str, Any]:
        model, tokenizer = _load_model_bundle(
            self.model_name_or_path,
            self.device,
            self.torch_dtype,
            self.trust_remote_code,
        )
        import torch

        prompt = _render_chat_prompt(tokenizer, messages)
        inputs = tokenizer(prompt, return_tensors="pt")
        if self.device != "auto":
            inputs = {key: value.to(model.device) for key, value in inputs.items()}

        generation_kwargs = {
            "max_new_tokens": int(max_tokens),
            "do_sample": float(temperature) > 0.0,
            "temperature": max(0.0, float(temperature)),
            "pad_token_id": tokenizer.eos_token_id,
        }
        with torch.no_grad():
            output_ids = model.generate(**inputs, **generation_kwargs)

        input_length = int(inputs["input_ids"].shape[-1])
        new_ids = output_ids[0][input_length:]
        content = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        return {
            "model": self.model_name_or_path,
            "content": content,
            "raw": {
                "backend": "local_transformers",
                "device": self.device,
            },
        }


@lru_cache(maxsize=2)
def _load_model_bundle(
    model_name_or_path: str,
    device: str,
    torch_dtype: str,
    trust_remote_code: bool,
):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "local_transformers backend requires torch and transformers. "
            "Install torch first, then install transformers or use `pip install -e .[local-llm]`."
        ) from exc

    dtype = _resolve_dtype(torch, torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
    }
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    if device == "auto":
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_kwargs,
    )
    if device != "auto":
        model = model.to(device)
    model.eval()
    return (model, tokenizer)


def _resolve_dtype(torch_module, value: str):
    lowered = str(value or "").strip().casefold()
    if not lowered or lowered == "auto":
        return None
    mapping = {
        "float16": torch_module.float16,
        "fp16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "bf16": torch_module.bfloat16,
        "float32": torch_module.float32,
        "fp32": torch_module.float32,
    }
    if lowered not in mapping:
        raise ValueError(f"Unsupported torch_dtype: {value}")
    return mapping[lowered]


def _render_chat_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    lines: list[str] = []
    for item in messages:
        role = str(item.get("role") or "user").strip()
        content = str(item.get("content") or "").strip()
        lines.append(f"{role.upper()}:\n{content}")
    lines.append("ASSISTANT:\n")
    return "\n\n".join(lines)
