from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import sys
import threading
import warnings
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from hep_rag_v2.textnorm import normalize_search_text


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

HASH_VECTOR_MODEL = "hash-v1"
HASH_IDF_VECTOR_MODEL = "hash-idf-v1"
DEFAULT_VECTOR_MODEL = HASH_IDF_VECTOR_MODEL
DEFAULT_VECTOR_DIM = 768

EMBEDDING_STOPWORDS = {
    "search",
    "searches",
    "observation",
    "observations",
    "measurement",
    "measurements",
    "result",
    "results",
    "study",
    "studies",
    "paper",
    "presents",
    "presented",
    "reported",
    "report",
    "collision",
    "collisions",
    "proton",
    "pp",
    "tev",
    "sqrt",
    "final",
    "state",
    "states",
}
DEFAULT_SENTENCE_TRANSFORMER_BATCH_SIZE = 32
_SENTENCE_TRANSFORMER_RUNTIME_LOCK = threading.Lock()
_SENTENCE_TRANSFORMER_RUNTIME: dict[str, dict[str, Any]] = {}
_MISSING = object()


# ---------------------------------------------------------------------------
# LocalIndex dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LocalIndex:
    ids: np.ndarray
    vectors: np.ndarray
    extras: dict[str, np.ndarray]


# ---------------------------------------------------------------------------
# Model validation helpers
# ---------------------------------------------------------------------------

def _validate_model(model: str) -> None:
    if model in {HASH_VECTOR_MODEL, HASH_IDF_VECTOR_MODEL}:
        return
    if _is_sentence_transformer_model(model):
        return
    raise ValueError(f"Unsupported vector model: {model}")


def _is_sentence_transformer_model(model: str) -> bool:
    return model.startswith("st:") or model.startswith("sentence-transformers:")


def _sentence_transformer_name(model: str) -> str:
    prefix, _, value = model.partition(":")
    if prefix not in {"st", "sentence-transformers"} or not value.strip():
        raise ValueError(f"Unsupported sentence-transformers model spec: {model}")
    return value.strip()


def configure_embedding_runtime(*, model: str, settings: dict[str, Any] | None = None) -> None:
    if not _is_sentence_transformer_model(model):
        return
    runtime = _normalize_sentence_transformer_runtime(settings)
    with _SENTENCE_TRANSFORMER_RUNTIME_LOCK:
        _SENTENCE_TRANSFORMER_RUNTIME[str(model)] = runtime
    _get_sentence_transformer.cache_clear()


def _runtime_for_model(model: str) -> dict[str, Any]:
    with _SENTENCE_TRANSFORMER_RUNTIME_LOCK:
        settings = dict(_SENTENCE_TRANSFORMER_RUNTIME.get(str(model)) or {})
    return _normalize_sentence_transformer_runtime(settings or None)


def _normalize_sentence_transformer_runtime(settings: dict[str, Any] | None) -> dict[str, Any]:
    settings = dict(settings or {})
    runtime = dict(settings.get("runtime") or settings)
    huggingface = dict(runtime.get("huggingface") or {})
    return {
        "device": str(runtime.get("device") or "auto").strip() or "auto",
        "batch_size": max(1, int(runtime.get("batch_size") or DEFAULT_SENTENCE_TRANSFORMER_BATCH_SIZE)),
        "allow_silent_fallback": bool(settings.get("allow_silent_fallback", runtime.get("allow_silent_fallback", False))),
        "huggingface": {
            "endpoint": str(huggingface.get("endpoint") or "").strip() or None,
            "cache_dir": str(huggingface.get("cache_dir") or "").strip() or None,
            "local_files_only": bool(huggingface.get("local_files_only", False)),
            "token": str(huggingface.get("token") or "").strip() or None,
        },
    }


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def _safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "vector"


def _aggregate_text_map(conn: sqlite3.Connection, query: str) -> dict[int, str]:
    mapping: dict[int, list[str]] = defaultdict(list)
    for row in conn.execute(query):
        owner_id = int(row["owner_id"])
        value = str(row["value"] or "").strip()
        if not value:
            continue
        if value in mapping[owner_id]:
            continue
        mapping[owner_id].append(value)
    return {
        owner_id: " ".join(values)
        for owner_id, values in mapping.items()
    }


# ---------------------------------------------------------------------------
# Hash / Hash-IDF embedding
# ---------------------------------------------------------------------------

def _embedding_tokens(text: str) -> list[str]:
    normalized = normalize_search_text(text)
    out: list[str] = []
    for token in re.findall(r"\w+", normalized, flags=re.UNICODE):
        lowered = token.casefold()
        if lowered in EMBEDDING_STOPWORDS:
            continue
        out.append(lowered)
    return out


def _hash_features(tokens: list[str]) -> list[str]:
    features = list(tokens)
    features.extend(f"{left}__{right}" for left, right in zip(tokens, tokens[1:]))
    return features


def _hashed_feature(feature: str, *, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=16).digest()
    index = int.from_bytes(digest[:8], "little", signed=False) % dim
    sign = 1.0 if (digest[8] & 1) else -1.0
    return (index, sign)


def _hash_embed_text(text: str, *, dim: int, bucket_idf: np.ndarray | None = None) -> np.ndarray:
    tokens = _embedding_tokens(text)
    return _hash_embed_tokens(tokens, dim=dim, bucket_idf=bucket_idf)


def _hash_embed_tokens(tokens: list[str], *, dim: int, bucket_idf: np.ndarray | None = None) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    if not tokens:
        return vector

    counts: dict[str, float] = defaultdict(float)
    for feature in _hash_features(tokens):
        counts[feature] += 1.0 if "__" not in feature else 0.5

    for feature, weight in counts.items():
        index, sign = _hashed_feature(feature, dim=dim)
        if bucket_idf is not None and bucket_idf.shape[0] == dim:
            weight *= float(bucket_idf[index])
        vector[index] += float(weight) * sign

    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        vector /= norm
    return vector


def _embed_corpus_hash(texts: list[str], *, dim: int) -> np.ndarray:
    if not texts:
        return np.zeros((0, dim), dtype=np.float32)
    matrix = np.zeros((len(texts), dim), dtype=np.float32)
    for idx, text in enumerate(texts):
        matrix[idx] = _hash_embed_text(text, dim=dim)
    return matrix


def _embed_corpus_hash_with_idf(texts: list[str], *, dim: int) -> tuple[np.ndarray, np.ndarray]:
    if not texts:
        return (
            np.zeros((0, dim), dtype=np.float32),
            np.ones((dim,), dtype=np.float32),
        )

    doc_count = len(texts)
    doc_freq = np.zeros((dim,), dtype=np.float32)
    tokenized: list[list[str]] = []
    for text in texts:
        tokens = _embedding_tokens(text)
        tokenized.append(tokens)
        seen_buckets: set[int] = set()
        for feature in _hash_features(tokens):
            index, _sign = _hashed_feature(feature, dim=dim)
            seen_buckets.add(index)
        for index in seen_buckets:
            doc_freq[index] += 1.0

    bucket_idf = np.log((1.0 + float(doc_count)) / (1.0 + doc_freq)) + 1.0
    matrix = np.zeros((doc_count, dim), dtype=np.float32)
    for idx, tokens in enumerate(tokenized):
        matrix[idx] = _hash_embed_tokens(tokens, dim=dim, bucket_idf=bucket_idf)
    return (matrix, bucket_idf.astype(np.float32))


# ---------------------------------------------------------------------------
# Sentence-transformers wrappers
# ---------------------------------------------------------------------------

@contextmanager
def _temporary_huggingface_endpoint(endpoint: str | None):
    endpoint = str(endpoint or "").strip().rstrip("/")
    if not endpoint:
        yield
        return
    keys = ("HF_ENDPOINT", "HUGGINGFACE_HUB_ENDPOINT")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ[key] = endpoint
        with _temporary_huggingface_hub_constants(endpoint):
            yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def _temporary_huggingface_hub_constants(endpoint: str):
    constants_module = sys.modules.get("huggingface_hub.constants")
    if constants_module is None:
        yield
        return

    template = endpoint + "/{repo_id}/resolve/{revision}/{filename}"
    previous = {
        "ENDPOINT": getattr(constants_module, "ENDPOINT", _MISSING),
        "HUGGINGFACE_CO_URL_TEMPLATE": getattr(constants_module, "HUGGINGFACE_CO_URL_TEMPLATE", _MISSING),
    }
    try:
        setattr(constants_module, "ENDPOINT", endpoint)
        setattr(constants_module, "HUGGINGFACE_CO_URL_TEMPLATE", template)
        yield
    finally:
        for name, value in previous.items():
            if value is _MISSING:
                try:
                    delattr(constants_module, name)
                except AttributeError:
                    pass
            else:
                setattr(constants_module, name, value)


def _huggingface_cache_repo_dir(model_name: str, cache_dir: str | None) -> Path | None:
    root = str(cache_dir or os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE") or "").strip()
    if not root:
        return None
    return Path(root).expanduser() / f"models--{model_name.replace('/', '--')}"


def _cached_sentence_transformer_snapshot(model_name: str, cache_dir: str | None) -> Path | None:
    repo_dir = _huggingface_cache_repo_dir(model_name, cache_dir)
    if repo_dir is None:
        return None
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return None

    revisions: list[str] = []
    for ref_name in ("main", "master"):
        ref_path = repo_dir / "refs" / ref_name
        if ref_path.is_file():
            revision = ref_path.read_text(encoding="utf-8").strip()
            if revision:
                revisions.append(revision)
    revisions.extend(
        child.name
        for child in sorted(
            snapshots_dir.iterdir(),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        if child.is_dir()
    )

    seen: set[str] = set()
    for revision in revisions:
        if revision in seen:
            continue
        seen.add(revision)
        snapshot_dir = snapshots_dir / revision
        if not snapshot_dir.is_dir():
            continue
        has_config = (snapshot_dir / "config.json").exists()
        has_weights = any((snapshot_dir / name).exists() for name in ("model.safetensors", "pytorch_model.bin"))
        has_tokenizer = any((snapshot_dir / name).exists() for name in ("tokenizer.json", "vocab.txt", "tokenizer_config.json"))
        if has_config and has_weights and has_tokenizer:
            return snapshot_dir
    return None


def _prepare_sentence_transformer_source(
    *,
    model: str,
    endpoint: str | None,
    cache_dir: str | None,
    local_files_only: bool,
    token: str | None,
) -> tuple[str, bool]:
    model_name = _sentence_transformer_name(model)
    cached_snapshot = _cached_sentence_transformer_snapshot(model_name, cache_dir)
    if cached_snapshot is not None:
        return (str(cached_snapshot), True)
    if local_files_only:
        return (model_name, True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return (model_name, False)

    with _temporary_huggingface_endpoint(endpoint):
        snapshot_dir = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            local_files_only=False,
            token=token,
        )
    return (str(Path(snapshot_dir).expanduser()), True)


def _resolve_sentence_transformer_device(*, model: str, runtime: dict[str, Any]) -> str | None:
    requested = str(runtime.get("device") or "auto").strip() or "auto"
    if requested in {"auto", ""}:
        return None
    if not requested.startswith("cuda"):
        return requested

    allow_fallback = bool(runtime.get("allow_silent_fallback", False))
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "torch is required for sentence-transformers embeddings. Install a CUDA-compatible torch build for this environment."
        ) from exc

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        cuda_available = bool(torch.cuda.is_available())
    warning_text = " ".join(str(item.message).strip() for item in captured if str(item.message).strip())

    if not cuda_available:
        if allow_fallback:
            return "cpu"
        detail = (
            f"Requested embedding runtime.device={requested!r} for model={model}, but torch.cuda.is_available() is false. "
            f"Current torch={torch.__version__} with CUDA build {torch.version.cuda or 'cpu-only'}."
        )
        if warning_text:
            detail += f" PyTorch reported: {warning_text}"
        detail += " Install a torch build compatible with the local NVIDIA driver, or upgrade the driver. CPU fallback is disabled."
        raise RuntimeError(detail)

    if requested == "cuda":
        return "cuda"
    try:
        index = int(requested.split(":", 1)[1])
    except (IndexError, ValueError):
        raise RuntimeError(f"Unsupported CUDA device spec: {requested}. Use 'cuda' or 'cuda:<index>'.")
    device_count = int(torch.cuda.device_count())
    if index < 0 or index >= device_count:
        raise RuntimeError(
            f"Requested embedding runtime.device={requested!r}, but only {device_count} CUDA device(s) are visible."
        )
    return requested


@lru_cache(maxsize=8)
def _get_sentence_transformer(
    model: str,
    device: str | None,
    endpoint: str | None,
    cache_dir: str | None,
    local_files_only: bool,
    token: str | None,
    allow_silent_fallback: bool,
) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is not installed. Run `pip install -e .[embeddings]` to use this model."
        ) from exc
    runtime = {
        "device": device,
        "allow_silent_fallback": allow_silent_fallback,
    }
    resolved_device = _resolve_sentence_transformer_device(model=model, runtime=runtime)
    source_path, source_local_only = _prepare_sentence_transformer_source(
        model=model,
        endpoint=endpoint,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        token=token,
    )
    with _temporary_huggingface_endpoint(endpoint):
        try:
            return SentenceTransformer(
                source_path,
                device=resolved_device,
                cache_folder=cache_dir,
                local_files_only=source_local_only,
                token=token,
            )
        except Exception as exc:
            mirror_hint = (
                f" Configure embedding.runtime.huggingface.endpoint for a reachable mirror; current endpoint={endpoint!r}."
                if endpoint
                else " Configure embedding.runtime.huggingface.endpoint to use a reachable HuggingFace mirror."
            )
            raise RuntimeError(
                f"Failed to load sentence-transformers model {model}. {mirror_hint} Original error: {exc}"
            ) from exc


def _embed_corpus_sentence_transformers(texts: list[str], *, model: str) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    runtime = _runtime_for_model(model)
    huggingface = runtime.get("huggingface") or {}
    encoder = _get_sentence_transformer(
        model,
        runtime.get("device"),
        huggingface.get("endpoint"),
        huggingface.get("cache_dir"),
        bool(huggingface.get("local_files_only", False)),
        huggingface.get("token"),
        bool(runtime.get("allow_silent_fallback", False)),
    )
    vectors = encoder.encode(
        texts,
        batch_size=int(runtime.get("batch_size") or DEFAULT_SENTENCE_TRANSFORMER_BATCH_SIZE),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


def _embed_query_sentence_transformers(query: str, *, model: str) -> np.ndarray:
    vectors = _embed_corpus_sentence_transformers([query], model=model)
    if vectors.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return np.asarray(vectors[0], dtype=np.float32)


# ---------------------------------------------------------------------------
# Corpus / query embedding dispatch
# ---------------------------------------------------------------------------

def _embed_corpus(texts: list[str], *, model: str, dim: int) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
    if model == HASH_VECTOR_MODEL:
        vectors = _embed_corpus_hash(texts, dim=dim)
        return (vectors, {}, dim)
    if model == HASH_IDF_VECTOR_MODEL:
        vectors, bucket_idf = _embed_corpus_hash_with_idf(texts, dim=dim)
        return (vectors, {"bucket_idf": bucket_idf}, dim)
    if _is_sentence_transformer_model(model):
        vectors = _embed_corpus_sentence_transformers(texts, model=model)
        actual_dim = int(vectors.shape[1]) if vectors.ndim == 2 and vectors.size else 0
        return (vectors, {}, actual_dim)
    raise ValueError(f"Unsupported vector model: {model}")


def _query_vector(query: str, *, index: LocalIndex, model: str) -> np.ndarray:
    from hep_rag_v2.query import rewrite_query_for_embedding

    rewritten_query = rewrite_query_for_embedding(query)
    if model == HASH_VECTOR_MODEL:
        return _hash_embed_text(rewritten_query, dim=int(index.vectors.shape[1]))
    if model == HASH_IDF_VECTOR_MODEL:
        return _hash_embed_text(
            rewritten_query,
            dim=int(index.vectors.shape[1]),
            bucket_idf=index.extras.get("bucket_idf"),
        )
    if _is_sentence_transformer_model(model):
        return _embed_query_sentence_transformers(rewritten_query, model=model)
    raise ValueError(f"Unsupported vector model: {model}")


def _score_query(query: str, *, index: LocalIndex, model: str) -> np.ndarray:
    if index.vectors.size == 0:
        return np.zeros((0,), dtype=np.float32)
    query_vec = _query_vector(query, index=index, model=model)
    if not float(np.linalg.norm(query_vec)):
        return np.zeros((index.vectors.shape[0],), dtype=np.float32)
    return index.vectors @ query_vec
