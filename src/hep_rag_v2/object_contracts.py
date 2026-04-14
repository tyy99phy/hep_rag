from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

CURRENT_OBJECT_CONTRACT_VERSION = "v1"
ALLOWED_STATUSES = {"ready", "partial", "needs_review", "failed"}
ALLOWED_DERIVATIONS = {"extracted", "normalized", "aggregated", "ranked", "summarized", "inferred"}


class ContractModel:
    contract_name = "ContractModel"
    object_type = "contract_model"

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ContractModel:
        raise NotImplementedError

    def to_payload(self) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def schema(cls) -> dict[str, Any]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class WorkCapsule(ContractModel):
    object_id: str
    work_id: int | str
    title: str
    status: str
    source_kind: str
    source_refs: tuple[str, ...]
    derivation: str
    abstract: str | None = None
    collection_id: int | str | None = None
    collection_name: str | None = None
    chunk_refs: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    evidence_summary: str | None = None
    has_fulltext_support: bool | None = None
    created_at: str | None = None
    updated_at: str | None = None
    notes: str | None = None
    contract_version: str = CURRENT_OBJECT_CONTRACT_VERSION

    contract_name = "WorkCapsule"
    object_type = "work_capsule"

    def __post_init__(self) -> None:
        _validate_common(self, object_type=self.object_type)
        _require_text(self.title, field_name="title")
        object.__setattr__(self, "chunk_refs", _coerce_refs(self.chunk_refs, field_name="chunk_refs"))
        object.__setattr__(self, "metadata", _coerce_mapping(self.metadata))

    def to_payload(self) -> dict[str, Any]:
        payload = _common_payload(self, object_type=self.object_type)
        payload.update(
            _drop_none(
                {
                    "work_id": self.work_id,
                    "title": self.title,
                    "abstract": self.abstract,
                    "collection_id": self.collection_id,
                    "collection_name": self.collection_name,
                    "chunk_refs": list(self.chunk_refs),
                    "metadata": dict(self.metadata),
                    "evidence_summary": self.evidence_summary,
                    "has_fulltext_support": self.has_fulltext_support,
                }
            )
        )
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> WorkCapsule:
        _expect_object_type(payload, cls.object_type)
        return cls(
            object_id=_get_required_text(payload, "object_id"),
            work_id=_get_required_scalar(payload, "work_id"),
            title=_get_required_text(payload, "title"),
            status=_get_required_text(payload, "status"),
            source_kind=_get_required_text(payload, "source_kind"),
            source_refs=_coerce_refs(payload.get("source_refs"), field_name="source_refs"),
            derivation=_get_required_text(payload, "derivation"),
            abstract=_text_or_none(payload.get("abstract")),
            collection_id=payload.get("collection_id"),
            collection_name=_text_or_none(payload.get("collection_name")),
            chunk_refs=_coerce_refs(payload.get("chunk_refs") or (), field_name="chunk_refs"),
            metadata=_coerce_mapping(payload.get("metadata") or {}),
            evidence_summary=_text_or_none(payload.get("evidence_summary")),
            has_fulltext_support=_bool_or_none(payload.get("has_fulltext_support")),
            created_at=_text_or_none(payload.get("created_at")),
            updated_at=_text_or_none(payload.get("updated_at")),
            notes=_text_or_none(payload.get("notes")),
            contract_version=_get_required_text(payload, "contract_version"),
        )

    @classmethod
    def schema(cls) -> dict[str, Any]:
        schema = _common_schema(object_type=cls.object_type)
        schema["required"] += ["work_id", "title"]
        schema["properties"].update(
            {
                "work_id": {"type": ["integer", "string"]},
                "title": {"type": "string"},
                "abstract": {"type": "string"},
                "collection_id": {"type": ["integer", "string"]},
                "collection_name": {"type": "string"},
                "chunk_refs": {"type": "array", "items": {"type": "string"}},
                "metadata": {"type": "object"},
                "evidence_summary": {"type": "string"},
                "has_fulltext_support": {"type": "boolean"},
            }
        )
        return schema


@dataclass(frozen=True, slots=True)
class ResultSignature(ContractModel):
    object_id: str
    work_id: int | str
    label: str
    result_kind: str
    summary_text: str
    status: str
    source_kind: str
    source_refs: tuple[str, ...]
    derivation: str
    confidence: float | None = None
    value_items: tuple[dict[str, Any], ...] = ()
    evidence_items: tuple[dict[str, Any], ...] = ()
    context_items: tuple[dict[str, Any], ...] = ()
    work_capsule_id: str | None = None
    ambiguities: tuple[str, ...] = ()
    created_at: str | None = None
    updated_at: str | None = None
    notes: str | None = None
    contract_version: str = CURRENT_OBJECT_CONTRACT_VERSION

    contract_name = "ResultSignature"
    object_type = "result_signature"

    def __post_init__(self) -> None:
        _validate_common(self, object_type=self.object_type)
        _require_text(self.label, field_name="label")
        _require_text(self.result_kind, field_name="result_kind")
        _require_text(self.summary_text, field_name="summary_text")
        if self.confidence is not None:
            object.__setattr__(self, "confidence", _coerce_confidence(self.confidence, field_name="confidence"))
        object.__setattr__(self, "value_items", _coerce_mapping_items(self.value_items, field_name="value_items"))
        object.__setattr__(self, "evidence_items", _coerce_mapping_items(self.evidence_items, field_name="evidence_items"))
        object.__setattr__(self, "context_items", _coerce_mapping_items(self.context_items, field_name="context_items"))
        object.__setattr__(self, "ambiguities", _coerce_refs(self.ambiguities, field_name="ambiguities"))

    def to_payload(self) -> dict[str, Any]:
        payload = _common_payload(self, object_type=self.object_type)
        payload.update(
            _drop_none(
                {
                    "work_id": self.work_id,
                    "label": self.label,
                    "result_kind": self.result_kind,
                    "summary_text": self.summary_text,
                    "confidence": self.confidence,
                    "value_items": [dict(item) for item in self.value_items],
                    "evidence_items": [dict(item) for item in self.evidence_items],
                    "context_items": [dict(item) for item in self.context_items],
                    "work_capsule_id": self.work_capsule_id,
                    "ambiguities": list(self.ambiguities),
                }
            )
        )
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ResultSignature:
        _expect_object_type(payload, cls.object_type)
        return cls(
            object_id=_get_required_text(payload, "object_id"),
            work_id=_get_required_scalar(payload, "work_id"),
            label=_get_required_text(payload, "label"),
            result_kind=_get_required_text(payload, "result_kind"),
            summary_text=_get_required_text(payload, "summary_text"),
            status=_get_required_text(payload, "status"),
            source_kind=_get_required_text(payload, "source_kind"),
            source_refs=_coerce_refs(payload.get("source_refs"), field_name="source_refs"),
            derivation=_get_required_text(payload, "derivation"),
            confidence=_float_or_none(payload.get("confidence")),
            value_items=_coerce_mapping_items(payload.get("value_items") or (), field_name="value_items"),
            evidence_items=_coerce_mapping_items(payload.get("evidence_items") or (), field_name="evidence_items"),
            context_items=_coerce_mapping_items(payload.get("context_items") or (), field_name="context_items"),
            work_capsule_id=_text_or_none(payload.get("work_capsule_id")),
            ambiguities=_coerce_refs(payload.get("ambiguities") or (), field_name="ambiguities"),
            created_at=_text_or_none(payload.get("created_at")),
            updated_at=_text_or_none(payload.get("updated_at")),
            notes=_text_or_none(payload.get("notes")),
            contract_version=_get_required_text(payload, "contract_version"),
        )

    @classmethod
    def schema(cls) -> dict[str, Any]:
        schema = _common_schema(object_type=cls.object_type)
        schema["required"] += ["work_id", "label", "result_kind", "summary_text"]
        schema["properties"].update(
            {
                "work_id": {"type": ["integer", "string"]},
                "label": {"type": "string"},
                "result_kind": {"type": "string"},
                "summary_text": {"type": "string"},
                "confidence": {"type": "number"},
                "value_items": {"type": "array", "items": {"type": "object"}},
                "evidence_items": {"type": "array", "items": {"type": "object"}},
                "context_items": {"type": "array", "items": {"type": "object"}},
                "work_capsule_id": {"type": "string"},
                "ambiguities": {"type": "array", "items": {"type": "string"}},
            }
        )
        return schema


@dataclass(frozen=True, slots=True)
class MethodSignature(ContractModel):
    object_id: str
    work_id: int | str
    label: str
    method_kind: str
    summary_text: str
    status: str
    source_kind: str
    source_refs: tuple[str, ...]
    derivation: str
    normalized_text: str | None = None
    confidence: float | None = None
    evidence_items: tuple[dict[str, Any], ...] = ()
    application_links: tuple[dict[str, Any], ...] = ()
    work_capsule_id: str | None = None
    ambiguities: tuple[str, ...] = ()
    created_at: str | None = None
    updated_at: str | None = None
    notes: str | None = None
    contract_version: str = CURRENT_OBJECT_CONTRACT_VERSION

    contract_name = "MethodSignature"
    object_type = "method_signature"

    def __post_init__(self) -> None:
        _validate_common(self, object_type=self.object_type)
        _require_text(self.label, field_name="label")
        _require_text(self.method_kind, field_name="method_kind")
        _require_text(self.summary_text, field_name="summary_text")
        if self.confidence is not None:
            object.__setattr__(self, "confidence", _coerce_confidence(self.confidence, field_name="confidence"))
        object.__setattr__(self, "evidence_items", _coerce_mapping_items(self.evidence_items, field_name="evidence_items"))
        object.__setattr__(self, "application_links", _coerce_mapping_items(self.application_links, field_name="application_links"))
        object.__setattr__(self, "ambiguities", _coerce_refs(self.ambiguities, field_name="ambiguities"))

    def to_payload(self) -> dict[str, Any]:
        payload = _common_payload(self, object_type=self.object_type)
        payload.update(
            _drop_none(
                {
                    "work_id": self.work_id,
                    "label": self.label,
                    "method_kind": self.method_kind,
                    "summary_text": self.summary_text,
                    "normalized_text": self.normalized_text,
                    "confidence": self.confidence,
                    "evidence_items": [dict(item) for item in self.evidence_items],
                    "application_links": [dict(item) for item in self.application_links],
                    "work_capsule_id": self.work_capsule_id,
                    "ambiguities": list(self.ambiguities),
                }
            )
        )
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> MethodSignature:
        _expect_object_type(payload, cls.object_type)
        return cls(
            object_id=_get_required_text(payload, "object_id"),
            work_id=_get_required_scalar(payload, "work_id"),
            label=_get_required_text(payload, "label"),
            method_kind=_get_required_text(payload, "method_kind"),
            summary_text=_get_required_text(payload, "summary_text"),
            status=_get_required_text(payload, "status"),
            source_kind=_get_required_text(payload, "source_kind"),
            source_refs=_coerce_refs(payload.get("source_refs"), field_name="source_refs"),
            derivation=_get_required_text(payload, "derivation"),
            normalized_text=_text_or_none(payload.get("normalized_text")),
            confidence=_float_or_none(payload.get("confidence")),
            evidence_items=_coerce_mapping_items(payload.get("evidence_items") or (), field_name="evidence_items"),
            application_links=_coerce_mapping_items(payload.get("application_links") or (), field_name="application_links"),
            work_capsule_id=_text_or_none(payload.get("work_capsule_id")),
            ambiguities=_coerce_refs(payload.get("ambiguities") or (), field_name="ambiguities"),
            created_at=_text_or_none(payload.get("created_at")),
            updated_at=_text_or_none(payload.get("updated_at")),
            notes=_text_or_none(payload.get("notes")),
            contract_version=_get_required_text(payload, "contract_version"),
        )

    @classmethod
    def schema(cls) -> dict[str, Any]:
        schema = _common_schema(object_type=cls.object_type)
        schema["required"] += ["work_id", "label", "method_kind", "summary_text"]
        schema["properties"].update(
            {
                "work_id": {"type": ["integer", "string"]},
                "label": {"type": "string"},
                "method_kind": {"type": "string"},
                "summary_text": {"type": "string"},
                "normalized_text": {"type": "string"},
                "confidence": {"type": "number"},
                "evidence_items": {"type": "array", "items": {"type": "object"}},
                "application_links": {"type": "array", "items": {"type": "object"}},
                "work_capsule_id": {"type": "string"},
                "ambiguities": {"type": "array", "items": {"type": "string"}},
            }
        )
        return schema


@dataclass(frozen=True, slots=True)
class EvidenceBundle(ContractModel):
    object_id: str
    status: str
    source_kind: str
    source_refs: tuple[str, ...]
    derivation: str
    subject_refs: tuple[str, ...]
    items: tuple[dict[str, Any], ...]
    work_capsule_refs: tuple[str, ...] = ()
    coverage_summary: str | None = None
    ambiguities: tuple[str, ...] = ()
    created_at: str | None = None
    updated_at: str | None = None
    notes: str | None = None
    contract_version: str = CURRENT_OBJECT_CONTRACT_VERSION

    contract_name = "EvidenceBundle"
    object_type = "evidence_bundle"

    def __post_init__(self) -> None:
        _validate_common(self, object_type=self.object_type)
        object.__setattr__(self, "subject_refs", _coerce_refs(self.subject_refs, field_name="subject_refs"))
        object.__setattr__(self, "work_capsule_refs", _coerce_refs(self.work_capsule_refs, field_name="work_capsule_refs"))
        object.__setattr__(self, "ambiguities", _coerce_refs(self.ambiguities, field_name="ambiguities"))
        object.__setattr__(self, "items", _coerce_evidence_items(self.items))

    def to_payload(self) -> dict[str, Any]:
        payload = _common_payload(self, object_type=self.object_type)
        payload.update(
            _drop_none(
                {
                    "subject_refs": list(self.subject_refs),
                    "items": [dict(item) for item in self.items],
                    "work_capsule_refs": list(self.work_capsule_refs),
                    "coverage_summary": self.coverage_summary,
                    "ambiguities": list(self.ambiguities),
                }
            )
        )
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> EvidenceBundle:
        _expect_object_type(payload, cls.object_type)
        return cls(
            object_id=_get_required_text(payload, "object_id"),
            status=_get_required_text(payload, "status"),
            source_kind=_get_required_text(payload, "source_kind"),
            source_refs=_coerce_refs(payload.get("source_refs"), field_name="source_refs"),
            derivation=_get_required_text(payload, "derivation"),
            subject_refs=_coerce_refs(payload.get("subject_refs"), field_name="subject_refs"),
            items=_coerce_evidence_items(payload.get("items")),
            work_capsule_refs=_coerce_refs(payload.get("work_capsule_refs") or (), field_name="work_capsule_refs"),
            coverage_summary=_text_or_none(payload.get("coverage_summary")),
            ambiguities=_coerce_refs(payload.get("ambiguities") or (), field_name="ambiguities"),
            created_at=_text_or_none(payload.get("created_at")),
            updated_at=_text_or_none(payload.get("updated_at")),
            notes=_text_or_none(payload.get("notes")),
            contract_version=_get_required_text(payload, "contract_version"),
        )

    @classmethod
    def schema(cls) -> dict[str, Any]:
        schema = _common_schema(object_type=cls.object_type)
        schema["required"] += ["subject_refs", "items"]
        schema["properties"].update(
            {
                "subject_refs": {"type": "array", "items": {"type": "string"}},
                "items": {"type": "array", "items": {"type": "object"}},
                "work_capsule_refs": {"type": "array", "items": {"type": "string"}},
                "coverage_summary": {"type": "string"},
                "ambiguities": {"type": "array", "items": {"type": "string"}},
            }
        )
        return schema


@dataclass(frozen=True, slots=True)
class TraceStep(ContractModel):
    object_id: str
    step_type: str
    summary: str
    status: str
    source_kind: str
    source_refs: tuple[str, ...]
    derivation: str
    step_index: int | None = None
    input_refs: tuple[str, ...] = ()
    output_refs: tuple[str, ...] = ()
    target: str | None = None
    idea_ids: tuple[str, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)
    ambiguities: tuple[str, ...] = ()
    created_at: str | None = None
    updated_at: str | None = None
    notes: str | None = None
    contract_version: str = CURRENT_OBJECT_CONTRACT_VERSION

    contract_name = "TraceStep"
    object_type = "trace_step"

    def __post_init__(self) -> None:
        _validate_common(self, object_type=self.object_type)
        _require_text(self.step_type, field_name="step_type")
        _require_text(self.summary, field_name="summary")
        object.__setattr__(self, "input_refs", _coerce_refs(self.input_refs, field_name="input_refs"))
        object.__setattr__(self, "output_refs", _coerce_refs(self.output_refs, field_name="output_refs"))
        object.__setattr__(self, "idea_ids", _coerce_refs(self.idea_ids, field_name="idea_ids"))
        object.__setattr__(self, "ambiguities", _coerce_refs(self.ambiguities, field_name="ambiguities"))
        object.__setattr__(self, "metrics", _coerce_mapping(self.metrics))
        if self.step_index is not None:
            object.__setattr__(self, "step_index", int(self.step_index))

    def to_payload(self) -> dict[str, Any]:
        payload = _common_payload(self, object_type=self.object_type)
        payload.update(
            _drop_none(
                {
                    "step_type": self.step_type,
                    "summary": self.summary,
                    "step_index": self.step_index,
                    "input_refs": list(self.input_refs),
                    "output_refs": list(self.output_refs),
                    "target": self.target,
                    "idea_ids": list(self.idea_ids),
                    "metrics": dict(self.metrics),
                    "ambiguities": list(self.ambiguities),
                }
            )
        )
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> TraceStep:
        _expect_object_type(payload, cls.object_type)
        return cls(
            object_id=_get_required_text(payload, "object_id"),
            step_type=_get_required_text(payload, "step_type"),
            summary=_get_required_text(payload, "summary"),
            status=_get_required_text(payload, "status"),
            source_kind=_get_required_text(payload, "source_kind"),
            source_refs=_coerce_refs(payload.get("source_refs"), field_name="source_refs"),
            derivation=_get_required_text(payload, "derivation"),
            step_index=_int_or_none(payload.get("step_index")),
            input_refs=_coerce_refs(payload.get("input_refs") or (), field_name="input_refs"),
            output_refs=_coerce_refs(payload.get("output_refs") or (), field_name="output_refs"),
            target=_text_or_none(payload.get("target")),
            idea_ids=_coerce_refs(payload.get("idea_ids") or (), field_name="idea_ids"),
            metrics=_coerce_mapping(payload.get("metrics") or {}),
            ambiguities=_coerce_refs(payload.get("ambiguities") or (), field_name="ambiguities"),
            created_at=_text_or_none(payload.get("created_at")),
            updated_at=_text_or_none(payload.get("updated_at")),
            notes=_text_or_none(payload.get("notes")),
            contract_version=_get_required_text(payload, "contract_version"),
        )

    @classmethod
    def schema(cls) -> dict[str, Any]:
        schema = _common_schema(object_type=cls.object_type)
        schema["required"] += ["step_type", "summary"]
        schema["properties"].update(
            {
                "step_type": {"type": "string"},
                "summary": {"type": "string"},
                "step_index": {"type": "integer"},
                "input_refs": {"type": "array", "items": {"type": "string"}},
                "output_refs": {"type": "array", "items": {"type": "string"}},
                "target": {"type": "string"},
                "idea_ids": {"type": "array", "items": {"type": "string"}},
                "metrics": {"type": "object"},
                "ambiguities": {"type": "array", "items": {"type": "string"}},
            }
        )
        return schema


def export_object_contract_schemas() -> dict[str, Any]:
    objects = {
        WorkCapsule.contract_name: WorkCapsule.schema(),
        ResultSignature.contract_name: ResultSignature.schema(),
        MethodSignature.contract_name: MethodSignature.schema(),
        EvidenceBundle.contract_name: EvidenceBundle.schema(),
        TraceStep.contract_name: TraceStep.schema(),
    }
    return {"version": CURRENT_OBJECT_CONTRACT_VERSION, "objects": objects}


def _common_payload(item: Any, *, object_type: str) -> dict[str, Any]:
    return _drop_none(
        {
            "contract_version": item.contract_version,
            "object_type": object_type,
            "object_id": item.object_id,
            "source_kind": item.source_kind,
            "status": item.status,
            "source_refs": list(item.source_refs),
            "derivation": item.derivation,
            "created_at": item.created_at,
            "updated_at": item.updated_at,
            "notes": item.notes,
        }
    )


def _common_schema(*, object_type: str) -> dict[str, Any]:
    return {
        "type": "object",
        "required": [
            "contract_version",
            "object_type",
            "object_id",
            "source_kind",
            "status",
            "source_refs",
            "derivation",
        ],
        "properties": {
            "contract_version": {"type": "string", "const": CURRENT_OBJECT_CONTRACT_VERSION},
            "object_type": {"type": "string", "const": object_type},
            "object_id": {"type": "string"},
            "source_kind": {"type": "string"},
            "status": {"type": "string", "enum": sorted(ALLOWED_STATUSES)},
            "source_refs": {"type": "array", "items": {"type": "string"}},
            "derivation": {"type": "string", "enum": sorted(ALLOWED_DERIVATIONS)},
            "created_at": {"type": "string"},
            "updated_at": {"type": "string"},
            "notes": {"type": "string"},
        },
    }


def _validate_common(item: Any, *, object_type: str) -> None:
    _validate_version(item.contract_version)
    _require_text(item.object_id, field_name="object_id")
    _require_text(item.source_kind, field_name="source_kind")
    _validate_status(item.status)
    object.__setattr__(item, "source_refs", _coerce_refs(item.source_refs, field_name="source_refs"))
    _validate_derivation(item.derivation)
    if getattr(item, "object_type", object_type) != object_type:
        raise ValueError(f"object_type must be {object_type}.")


def _validate_version(value: Any) -> str:
    version = _require_text(value, field_name="contract_version")
    if version != CURRENT_OBJECT_CONTRACT_VERSION:
        raise ValueError(f"Unsupported contract_version: {version}")
    return version


def _validate_status(value: Any) -> str:
    status = _require_text(value, field_name="status")
    if status not in ALLOWED_STATUSES:
        raise ValueError(f"status must be one of {sorted(ALLOWED_STATUSES)}")
    return status


def _validate_derivation(value: Any) -> str:
    derivation = _require_text(value, field_name="derivation")
    if derivation not in ALLOWED_DERIVATIONS:
        raise ValueError(f"derivation must be one of {sorted(ALLOWED_DERIVATIONS)}")
    return derivation


def _expect_object_type(payload: Mapping[str, Any], expected: str) -> None:
    actual = _get_required_text(payload, "object_type")
    if actual != expected:
        raise ValueError(f"object_type must be {expected}, got {actual}")


def _get_required_text(payload: Mapping[str, Any], key: str) -> str:
    return _require_text(payload.get(key), field_name=key)


def _get_required_scalar(payload: Mapping[str, Any], key: str) -> Any:
    value = payload.get(key)
    if value is None or value == "":
        raise ValueError(f"{key} is required.")
    return value


def _require_text(value: Any, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required.")
    return text


def _coerce_refs(value: Any, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        raise ValueError(f"{field_name} is required.")
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list or tuple.")
    return tuple(_require_text(item, field_name=f"{field_name}[]") for item in value)


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    raise ValueError(f"Expected mapping payload, got {type(value)!r}.")


def _coerce_mapping_items(value: Any, *, field_name: str) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list or tuple.")
    items: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError(f"{field_name} entries must be mappings.")
        items.append(dict(item))
    return tuple(items)


def _coerce_evidence_items(value: Any) -> tuple[dict[str, Any], ...]:
    items = _coerce_mapping_items(value, field_name="items")
    validated: list[dict[str, Any]] = []
    for item in items:
        if not str(item.get("evidence_key") or "").strip() and not str(item.get("citation_id") or "").strip():
            raise ValueError("EvidenceBundle.items entries require evidence_key or citation_id.")
        validated.append(item)
    return tuple(validated)


def _coerce_confidence(value: Any, *, field_name: str) -> float:
    out = float(value)
    if out < 0.0 or out > 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1.")
    return out


def _text_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _drop_none(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}
