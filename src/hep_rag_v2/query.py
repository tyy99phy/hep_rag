from __future__ import annotations

import re
from dataclasses import dataclass

from hep_rag_v2.textnorm import normalize_search_text


RELATION_QUERY_PATTERNS = (
    "related work",
    "related works",
    "review",
    "survey",
    "summary",
    "summarize",
    "overview",
    "compare",
    "comparison",
    "similar",
    "connection",
    "connections",
    "relationship",
    "relationships",
    "background",
    "history",
    "timeline",
    "evolution",
    "landscape",
    "总结",
    "整理",
    "概述",
    "总览",
    "综述",
    "相关工作",
    "联系",
    "关系",
    "背景",
    "脉络",
    "演化",
    "发展",
    "对比",
    "比较",
)

INTENT_TOKEN_PATTERNS = (
    "综述",
    "相关工作",
    "介绍",
    "总结",
    "整理",
    "比较",
    "对比",
    "相关",
    "review",
    "survey",
    "compare",
    "comparison",
    "background",
    "history",
    "timeline",
    "summary",
    "summarize",
    "overview",
    "latest",
    "current",
    "recent",
    "best",
    "result",
    "results",
    "status",
    "update",
    "work",
    "works",
    "paper",
    "papers",
)

COMMON_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "what",
    "which",
    "with",
    "best",
    "current",
    "latest",
    "recent",
    "result",
    "results",
    "status",
    "summary",
    "overview",
    "一下",
    "一下子",
    "介绍一下",
    "告诉我",
    "帮我",
    "请",
    "请问",
    "看看",
    "来吧",
}
RESULT_QUERY_PATTERNS = (
    "summary",
    "summarize",
    "overview",
    "current result",
    "current results",
    "latest result",
    "latest results",
    "recent result",
    "recent results",
    "best result",
    "best results",
    "status",
    "update",
    "总结",
    "整理",
    "概述",
    "总览",
    "最新",
    "结果",
    "当前",
    "目前",
    "现状",
    "进展",
)
INTENT_PHRASE_PATTERNS = (
    r"综述(?:一下)?",
    r"总结(?:一下)?",
    r"整理(?:一下)?",
    r"介绍(?:一下)?",
    r"相关工作",
    r"最新(?:结果|进展)?",
    r"当前(?:结果|状态|现状)?",
    r"目前(?:结果|状态|现状)?",
    r"\b(?:summary|summarize|overview)\b",
    r"\b(?:current|latest|recent|newest|best)\s+(?:result|results|status|overview|summary)\b",
    r"\b(?:current|latest|recent|newest|best)\b",
    r"\bresults?\b",
)
HEP_QUERY_EXPANSIONS = (
    (r"\bCMS\b", " cms cms collaboration "),
    (r"\bATLAS\b", " atlas atlas collaboration "),
    (r"\bVBS\b", " vbs vector boson scattering electroweak production "),
    (r"\bSS\s*WW\b|\bSSWW\b", " ssww same-sign WW same-sign W boson pairs "),
    (r"same[- ]sign\s+ww", " ssww same-sign WW same-sign W boson pairs "),
    (r"W\s*[±+]\s*W\s*[±+]", " ssww same-sign WW same-sign W boson pairs "),
)
SINGLE_CJK_TOKEN_RE = re.compile(r"^[\u3400-\u9fff]$")

CONCEPT_DEFINITIONS = (
    {
        "name": "cms",
        "triggers": ("cms",),
        "aliases": ("cms", "cms collaboration"),
        "semantic_terms": ("cms collaboration",),
    },
    {
        "name": "atlas",
        "triggers": ("atlas",),
        "aliases": ("atlas", "atlas collaboration"),
        "semantic_terms": ("atlas collaboration",),
    },
    {
        "name": "vbs",
        "triggers": ("vbs",),
        "aliases": ("vbs", "vector boson scattering", "electroweak production"),
        "semantic_terms": ("vector boson scattering", "electroweak production"),
    },
    {
        "name": "same_sign_ww",
        "triggers": ("ssww",),
        "aliases": (
            "ssww",
            "same sign ww",
            "same sign w boson pairs",
            "same electric charge",
        ),
        "semantic_terms": ("same-sign W boson pairs", "same-sign WW"),
    },
    {
        "name": "higgs",
        "triggers": ("h", "higgs"),
        "aliases": ("h", "higgs"),
        "semantic_terms": ("higgs boson",),
    },
    {
        "name": "pseudoscalar",
        "triggers": ("aa", "pseudoscalar", "pseudoscalars", "alp", "alps"),
        "aliases": ("aa", "pseudoscalar", "pseudoscalars", "alp", "alps"),
        "semantic_terms": ("pseudoscalar boson", "light pseudoscalar"),
    },
    {
        "name": "photon",
        "triggers": ("photon", "photons", "diphoton", "diphotons", "gamma", "gammas"),
        "aliases": ("photon", "photons", "diphoton", "diphotons", "gamma", "gammas"),
        "semantic_terms": ("photon", "diphoton"),
    },
    {
        "name": "muon",
        "triggers": ("mu", "muon", "muons", "dimuon", "dimuons", "trimuon", "trimuons"),
        "aliases": ("mu", "muon", "muons", "dimuon", "dimuons", "trimuon", "trimuons"),
        "semantic_terms": ("muon", "dimuon"),
    },
    {
        "name": "tau",
        "triggers": ("tau", "taus", "ditau", "ditaus"),
        "aliases": ("tau", "taus", "ditau", "ditaus"),
        "semantic_terms": ("tau lepton", "ditau"),
    },
    {
        "name": "electron",
        "triggers": ("electron", "electrons", "dielectron", "dielectrons"),
        "aliases": ("electron", "electrons", "dielectron", "dielectrons"),
        "semantic_terms": ("electron", "dielectron"),
    },
    {
        "name": "top",
        "triggers": ("top", "topquark", "tops"),
        "aliases": ("top", "topquark", "tops"),
        "semantic_terms": ("top quark",),
    },
    {
        "name": "exotic_decay",
        "triggers": ("exotic", "decay", "decays", "branching"),
        "aliases": ("exotic", "decay", "decays", "branching"),
        "semantic_terms": ("exotic decay",),
    },
)


@dataclass(frozen=True)
class QueryProfile:
    raw: str
    normalized: str
    tokens: tuple[str, ...]
    content_tokens: tuple[str, ...]
    match_groups: tuple[tuple[str, ...], ...]
    relation_patterns: tuple[str, ...]
    concept_names: tuple[str, ...]


def analyze_query(text: str) -> QueryProfile:
    raw = str(text or "")
    normalized = _normalize_query_text(raw)
    tokens = tuple(_tokenize(normalized))
    relation_patterns = tuple(pattern for pattern in RELATION_QUERY_PATTERNS if pattern in raw.casefold())

    content_tokens: list[str] = []
    for token in tokens:
        if _is_intent_token(token):
            continue
        if token not in content_tokens:
            content_tokens.append(token)

    groups: list[tuple[str, ...]] = []
    concepts: list[str] = []
    consumed: set[str] = set()
    content_token_set = set(content_tokens)
    for definition in CONCEPT_DEFINITIONS:
        triggers = {item.casefold() for item in definition["triggers"]}
        if not (triggers & content_token_set):
            continue
        aliases = tuple(dict.fromkeys(item.casefold() for item in definition["aliases"]))
        groups.append(aliases)
        concepts.append(str(definition["name"]))
        consumed.update(_definition_consumed_tokens(definition))

    for token in content_tokens:
        if token in consumed:
            continue
        groups.append((token,))

    return QueryProfile(
        raw=raw,
        normalized=normalized,
        tokens=tokens,
        content_tokens=tuple(content_tokens),
        match_groups=tuple(groups),
        relation_patterns=relation_patterns,
        concept_names=tuple(concepts),
    )


def build_match_queries(text: str) -> list[str]:
    profile = analyze_query(text)
    queries: list[str] = []

    if profile.match_groups:
        strict_groups = profile.match_groups[:4]
        queries.append(" AND ".join(_group_query(group) for group in strict_groups))
        if len(profile.match_groups) > 1:
            relaxed_groups = profile.match_groups[:3]
            queries.append(" OR ".join(_group_query(group) for group in relaxed_groups))

    if profile.content_tokens:
        queries.append(" AND ".join(_quote_token(token) for token in profile.content_tokens[:6]))
        queries.append(" OR ".join(_quote_token(token) for token in profile.content_tokens[:8]))
    elif profile.tokens:
        queries.append(" OR ".join(_quote_token(token) for token in profile.tokens[:8]))

    seen: set[str] = set()
    out: list[str] = []
    for query in queries:
        value = str(query or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def query_match_stats(text: str, profile: QueryProfile) -> tuple[int, float]:
    groups = profile.match_groups or tuple((token,) for token in profile.content_tokens)
    if not groups:
        return (0, 0.0)

    tokens = set(_tokenize(normalize_search_text(text)))
    if not tokens:
        return (0, 0.0)

    hits = 0
    for group in groups:
        if any(_alias_matches(alias, tokens) for alias in group):
            hits += 1
    return (hits, hits / float(len(groups)))


def rewrite_query_for_embedding(text: str) -> str:
    profile = analyze_query(text)
    terms: list[str] = []
    concept_term_map = {
        str(item["name"]): tuple(str(term) for term in item.get("semantic_terms") or ())
        for item in CONCEPT_DEFINITIONS
    }
    for concept_name in profile.concept_names:
        for term in concept_term_map.get(concept_name, ()):
            if term not in terms:
                terms.append(term)
    if "higgs" in profile.concept_names and "pseudoscalar" in profile.concept_names:
        combo = "higgs boson decay to pseudoscalar pair"
        if combo not in terms:
            terms.append(combo)
    for token in profile.content_tokens:
        if token not in terms:
            terms.append(token)
    if terms:
        return " ".join(terms)
    if profile.tokens:
        return " ".join(profile.tokens)
    return str(text or "")


def is_relation_query(text: str) -> bool:
    return bool(analyze_query(text).relation_patterns)


def is_result_query(text: str) -> bool:
    value = str(text or "").casefold()
    return any(pattern in value for pattern in RESULT_QUERY_PATTERNS)


def _normalize_query_text(text: str) -> str:
    value = str(text or "")
    for pattern in INTENT_PHRASE_PATTERNS:
        value = re.sub(pattern, " ", value, flags=re.IGNORECASE)
    for pattern, replacement in HEP_QUERY_EXPANSIONS:
        value = re.sub(pattern, replacement, value, flags=re.IGNORECASE)
    return normalize_search_text(value)


def _tokenize(text: str) -> list[str]:
    return [token.casefold() for token in re.findall(r"\w+", text or "", flags=re.UNICODE)]


def _is_intent_token(token: str) -> bool:
    value = str(token or "").casefold()
    if not value:
        return True
    if SINGLE_CJK_TOKEN_RE.fullmatch(value):
        return True
    if value in COMMON_STOPWORDS:
        return True
    if any(pattern in value for pattern in INTENT_TOKEN_PATTERNS):
        return True
    return value.isdigit() and len(value) > 4


def _definition_consumed_tokens(definition: dict[str, object]) -> set[str]:
    tokens: set[str] = set()
    for field in ("triggers", "aliases", "semantic_terms"):
        for item in definition.get(field, ()) or ():
            tokens.update(_tokenize(str(item)))
    return tokens


def _group_query(group: tuple[str, ...]) -> str:
    if len(group) == 1:
        return _quote_token(group[0])
    return "(" + " OR ".join(_quote_token(token) for token in group) + ")"


def _quote_token(token: str) -> str:
    return '"' + str(token).replace('"', "") + '"'


def _alias_matches(alias: str, tokens: set[str]) -> bool:
    parts = [part for part in str(alias or "").split() if part]
    if not parts:
        return False
    return all(part in tokens for part in parts)
