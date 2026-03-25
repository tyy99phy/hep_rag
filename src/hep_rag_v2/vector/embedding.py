from __future__ import annotations

import hashlib
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
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

@lru_cache(maxsize=2)
def _get_sentence_transformer(model_name: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is not installed. Run `pip install -e .[embeddings]` to use this model."
        ) from exc
    return SentenceTransformer(model_name)


def _embed_corpus_sentence_transformers(texts: list[str], *, model: str) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    encoder = _get_sentence_transformer(_sentence_transformer_name(model))
    vectors = encoder.encode(
        texts,
        batch_size=32,
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
