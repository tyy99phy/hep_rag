# HEP Core Object Contracts

Status: normative contract for the repo-owned substrate layer
Version: `v1`
Last updated: 2026-04-14

## 1. Purpose

This document defines the stable object-contract layer that `hep_rag` owns inside the repository. These contracts are
the common language for:

1. extraction outputs,
2. typed retrieval / evidence payloads,
3. service and API responses,
4. benchmark and gold fixtures.

This document is intentionally about **substrate**, not harness behavior. It freezes what the objects mean and which
fields they must carry, so future prompt / agent / orchestration layers can consume the same durable shapes without
re-defining them.

## 2. Architectural boundary

### In scope: substrate

The substrate layer includes:

- typed result extraction output,
- typed method extraction output,
- typed transfer candidates,
- evidence packaging,
- replayable reasoning-trace summaries,
- serialization / validation / versioned payload rules.

### Out of scope: harness

The harness layer is intentionally not specified here. That includes:

- prompt wording,
- model-routing policy,
- multi-step agent control flow,
- ranking heuristics beyond emitted object fields,
- UI composition,
- benchmark runner strategy beyond the object payloads it consumes.

Rule: if a consumer can change without changing the meaning of the emitted object, it belongs to the harness, not this
contract.

## 3. Global rules

## 3.1 Contract versioning

Every top-level contract object MUST carry `contract_version`.

- Initial version: `"v1"`
- A patch that only adds optional fields MAY keep `v1`.
- Removing a field, changing field semantics, or changing requiredness MUST create a new major contract version.

Consumers MUST reject or explicitly downgrade unknown major versions rather than silently guessing.

## 3.2 Shared field semantics

Unless stated otherwise, these rules apply to all five objects.

| Field | Required | Meaning |
| --- | --- | --- |
| `contract_version` | yes | Contract major version, initially `"v1"`. |
| `object_type` | yes | Contract discriminator; allowed values are listed below. |
| `object_id` | yes | Stable object identifier inside the payload domain. Must be deterministic when the source object is durable. |
| `source_kind` | yes | Producer provenance class, e.g. `extraction`, `retrieval`, `service`, `api`, or `benchmark_fixture`. |
| `status` | yes | Producer confidence state: `ready`, `partial`, `needs_review`, or `failed`. |
| `created_at` | optional | Creation timestamp of the payload object if available. |
| `updated_at` | optional | Last refresh timestamp if available. |
| `notes` | optional | Human-readable implementation notes; never normative. |

Allowed `object_type` values in this spec: `work_capsule`, `result_signature`, `method_signature`, `evidence_bundle`, `trace_step`.

### Status semantics

- `ready`: producer believes the object is usable without manual intervention.
- `partial`: object is emitted but some required semantic content could not be filled with high confidence.
- `needs_review`: object is structurally complete enough to inspect,
  but the producer is explicitly asking for manual review before downstream reliance.
- `failed`: producer attempted generation but the object should not be treated as valid substrate.

`failed` objects MAY exist in operational payloads for observability, but MUST NOT be treated as positive evidence.

## 3.3 Null / omission rules

- Required fields MUST be present.
- Optional fields SHOULD be omitted when unknown rather than filled with fabricated placeholders.
- Empty string values SHOULD be avoided unless an empty string is semantically distinct from omission.
- Empty arrays are allowed when the field itself is required and the producer is asserting “known empty”.

## 3.4 Provenance rules

Each object MUST carry enough provenance for a downstream consumer to answer:

1. where this object came from,
2. what upstream work / chunk / typed object it refers to,
3. whether the content is directly extracted, inferred, or summarized.

Minimum provenance vocabulary:

- `source_kind`: emitting surface,
- `source_refs`: stable references to upstream records when available,
- `derivation`: one of `extracted`, `normalized`, `aggregated`, `ranked`, `summarized`, or `inferred`.

If a field value is inferred rather than directly extracted, that fact MUST be representable in provenance or
field-level support metadata.

## 3.5 Ambiguity and manual review

Ambiguity is first-class substrate state.

Producers MUST NOT erase ambiguity by pretending uncertain content is final. When evidence is weak, conflicting, or
underspecified, the object SHOULD still be emitted with:

- `status: "partial"` or `"needs_review"`,
- explicit ambiguity notes or support metadata,
- provenance that distinguishes direct evidence from inferred fill-ins.

## 4. Object contracts

## 4.1 `WorkCapsule`

### Purpose

`WorkCapsule` is the repo-owned compact representation of a work-level unit that downstream layers can pass around
without re-querying ad-hoc tables or raw retrieval hits. It is the anchor object that other contracts can point back to.

### Required fields

| Field | Type | Meaning |
| --- | --- | --- |
| `contract_version` | string | Must be `"v1"`. |
| `object_type` | string | Must be `"work_capsule"`. |
| `object_id` | string | Stable capsule id, normally derived from `work_id` or another durable repo id. |
| `work_id` | integer or string | Stable underlying work identifier. |
| `title` | string | Canonical work title as known to the repo. |
| `status` | string | Shared status semantics. |
| `source_kind` | string | Producer surface. |
| `derivation` | string | Usually `normalized` or `aggregated`. |
| `source_refs` | array | Stable upstream references used to construct the capsule. |

### Optional fields

- `abstract`
- `collection_id`
- `collection_name`
- `chunk_refs` — retrievable chunk ids if the capsule is backed by chunked text
- `metadata` — non-normative extra fields that do not redefine required semantics
- `evidence_summary` — short summary of what text support is available
- `has_fulltext_support` — boolean indicator that chunk-level support exists

### Semantics

- A `WorkCapsule` is not itself a result, method, or transfer claim.
- It is the stable context object that names the work and its retrievable identity.
- A capsule MAY be emitted from metadata-only ingest; in that case it SHOULD preserve that fact in provenance and MAY omit `chunk_refs`.

## 4.2 `ResultSignature`

### Purpose

`ResultSignature` captures the normalized substrate representation of a result-like finding for a work. It is
intentionally lighter than a full scientific claim graph. It exists so retrieval, evidence, service, and benchmark
layers can agree on what “a result object” means.

### Required fields

| Field | Type | Meaning |
| --- | --- | --- |
| `contract_version` | string | Must be `"v1"`. |
| `object_type` | string | Must be `"result_signature"`. |
| `object_id` | string | Stable result-signature id. |
| `work_id` | integer or string | Owning work id. |
| `label` | string | Human-readable result label. |
| `result_kind` | string | Normalized result class such as `measurement`, `upper_limit`, `significance`, or `exclusion`. |
| `summary_text` | string | Short normalized summary of the result object. |
| `status` | string | Shared status semantics. |
| `source_kind` | string | Producer surface. |
| `derivation` | string | Usually `extracted` or `normalized`. |
| `source_refs` | array | Work / chunk / row references used to build the signature. |

### Optional fields

- `confidence`
- `value_items` — structured value payloads such as comparator, numeric value, unit, uncertainty, and local context
- `evidence_items` — supporting snippets or references
- `context_items` — dataset / section / selection hints
- `work_capsule_id`
- `ambiguities` — unresolved semantic questions

### Semantics

- `result_kind` normalizes the coarse result family; it does not need to encode every physics-specific nuance.
- `summary_text` MUST be readable without dereferencing raw storage.
- If numeric structure is unknown but a result-like statement is still supported,
  the signature MAY omit `value_items` and use `status: "partial"`.
- Multiple `ResultSignature` objects MAY exist per work if the producer can distinguish them stably.

## 4.3 `MethodSignature`

### Purpose

`MethodSignature` captures a reusable, normalized method cue that can support retrieval, comparison, and transfer
reasoning. It is the stable substrate counterpart to method extraction tables and typed retrieval items.

### Required fields

| Field | Type | Meaning |
| --- | --- | --- |
| `contract_version` | string | Must be `"v1"`. |
| `object_type` | string | Must be `"method_signature"`. |
| `object_id` | string | Stable method-signature id. |
| `work_id` | integer or string | Owning work id. |
| `label` | string | Human-readable method label. |
| `method_kind` | string | Normalized family such as `statistical_fit`, `multivariate`, `background_estimation`, or `reconstruction`. |
| `summary_text` | string | Short normalized statement of the method cue. |
| `status` | string | Shared status semantics. |
| `source_kind` | string | Producer surface. |
| `derivation` | string | Usually `extracted` or `normalized`. |
| `source_refs` | array | Work / chunk / row references used to build the signature. |

### Optional fields

- `normalized_text`
- `confidence`
- `evidence_items`
- `application_links` — references showing where the method was applied
- `work_capsule_id`
- `ambiguities`

### Semantics

- `MethodSignature` is about the reusable method cue, not the full procedural narrative.
- `normalized_text` SHOULD exist when multiple surface phrasings collapse to one reusable signature.
- If a signature is only weakly implied, emit it as `partial` or `needs_review`
  rather than upgrading it to a fully trusted transfer primitive.

## 4.4 `EvidenceBundle`

### Purpose

`EvidenceBundle` packages the evidence items a downstream consumer needs in order to inspect, cite, or replay a
result/method/idea/transfer output. It is the contract-level wrapper around evidence registries and related supporting
objects.

### Required fields

| Field | Type | Meaning |
| --- | --- | --- |
| `contract_version` | string | Must be `"v1"`. |
| `object_type` | string | Must be `"evidence_bundle"`. |
| `object_id` | string | Stable bundle id inside the response or artifact. |
| `status` | string | Shared status semantics. |
| `source_kind` | string | Producer surface. |
| `derivation` | string | Usually `aggregated` or `normalized`. |
| `subject_refs` | array | Object ids whose support is being packaged. |
| `items` | array | Evidence entries. Must be present even if empty. |

### Evidence entry minimum shape

Each `items[]` entry SHOULD expose, when available:

- `evidence_key`
- `citation_id`
- `object_type`
- `object_id`
- `work_id`
- `chunk_id`
- `title`
- `text` or `content`
- `section_hint`
- `page_hint`
- `occurrences`

### Optional bundle fields

- `source_refs`
- `work_capsule_refs`
- `coverage_summary`
- `ambiguities`

### Semantics

- `EvidenceBundle` is the canonical transport object for support payloads.
- The bundle MAY contain work-level, chunk-level, or typed-object support together.
- Bundles SHOULD preserve stable citation ids when a response format exposes citations.
- Duplicate evidence MAY be coalesced if `occurrences` records the collapse.

## 4.5 `TraceStep`

### Purpose

`TraceStep` is the replayable substrate summary of a reasoning step. It exists so services and benchmarks can demand
structured trace output without binding the repo to any particular prompt strategy or chain-of-thought format.

### Required fields

| Field | Type | Meaning |
| --- | --- | --- |
| `contract_version` | string | Must be `"v1"`. |
| `object_type` | string | Must be `"trace_step"`. |
| `object_id` | string | Stable step id within the trace. |
| `step_type` | string | Stable reasoning-step discriminator such as `retrieve` or `generate_idea`. |
| `summary` | string | Replayable concise summary of what happened at this step. |
| `status` | string | Shared status semantics. |
| `source_kind` | string | Producer surface. |
| `derivation` | string | Usually `summarized` or `aggregated`. |

### Optional fields

- `step_index`
- `source_refs`
- `input_refs`
- `output_refs`
- `target`
- `idea_ids`
- `metrics`
- `ambiguities`

### Semantics

- `TraceStep` is a summary artifact, not raw hidden reasoning.
- `summary` MUST be safe to persist, benchmark, and expose in API responses.
- Step objects SHOULD link to the objects they consumed or produced via refs instead of embedding full downstream payloads.
- A trace MAY be partial; missing steps are not errors if the producer marks the trace status honestly.

## 5. Cross-object invariants

The following invariants hold across the five contracts:

1. Every `ResultSignature` and `MethodSignature` MUST be traceable to a `work_id`.
2. Every `EvidenceBundle` MUST identify which object ids it supports through `subject_refs`.
3. Every `TraceStep` SHOULD point to the relevant result / method / idea / evidence objects through refs when those objects exist.
4. A consumer MUST be able to understand object role from `object_type` alone.
5. Two objects with the same `object_type`, `object_id`, and `contract_version`
   inside the same artifact MUST refer to the same semantic object.

## 6. Serialization rules

- Canonical transport is JSON-compatible dictionaries / arrays.
- Field names use `snake_case`.
- Timestamps SHOULD use ISO-8601 strings.
- Producers MAY expose dataclass / Pydantic / typed-model wrappers, but JSON field semantics are authoritative.
- Unknown optional fields MUST be preserved by pass-through consumers when practical.

## 7. Validation expectations

A validating implementation for `v1` MUST check at least:

1. required fields exist,
2. `object_type` matches the expected contract,
3. `contract_version` is supported,
4. `status` is from the allowed vocabulary,
5. key relationship fields (`work_id`, `subject_refs`, `items`, `step_type`, etc.) have the expected top-level shape.

Validation in `v1` MAY remain structural rather than physics-semantic. For example, a validator may check that
`result_kind` is present without proving the scientific claim is correct.

## 8. Mapping guidance for current repo surfaces

This section is descriptive, not normative, but it anchors the contract to current repo surfaces.

- `results.py` current `result_objects`, `result_values`, and `result_context` map naturally into `ResultSignature`.
- `methods.py` current `method_objects`, `method_signatures`, and `method_application_links` map naturally into `MethodSignature`.
- `transfer.py` current `transfer_candidates` and `transfer_edges` provide the substrate inputs
  that downstream layers can reference from `TraceStep` and `EvidenceBundle`, even though
  `TransferCandidate` is not one of the five frozen contracts in this document.
- `evidence.py` current registry payloads map naturally into `EvidenceBundle.items`.
- `service/facade.py` current `trace.steps` payloads map naturally into `TraceStep`.
- Work-level retrieval and service payloads SHOULD expose `WorkCapsule`
  instead of ad-hoc work dicts when the dedicated contract module lands.

## 9. Non-goals for `v1`

`v1` deliberately does not standardize:

- a full scientific ontology,
- prompt-time chain-of-thought,
- ranking policy for transfer or idea generation,
- benchmark scoring logic,
- UI-facing rendering conventions,
- every intermediate typed object in the repo.

The aim is a stable common language, not an exhaustive research-knowledge schema.

## 10. Migration notes

When existing surfaces still emit ad-hoc dictionaries, they SHOULD either:

1. align directly to these field names, or
2. provide explicit mappers into the contract objects.

During migration, ad-hoc legacy keys MAY temporarily coexist, but contract-aligned keys must remain the source of truth
once both are present.

## 11. Minimal example shapes

### `WorkCapsule`

```json
{
  "contract_version": "v1",
  "object_type": "work_capsule",
  "object_id": "work:123",
  "work_id": 123,
  "title": "Observation of ...",
  "status": "ready",
  "source_kind": "retrieval",
  "derivation": "normalized",
  "source_refs": ["works:123"],
  "has_fulltext_support": true
}
```

### `ResultSignature`

```json
{
  "contract_version": "v1",
  "object_type": "result_signature",
  "object_id": "result:123:default",
  "work_id": 123,
  "label": "branching fraction",
  "result_kind": "measurement",
  "summary_text": "Observation of ... | result signatures: measurement | source=chunks",
  "status": "ready",
  "source_kind": "extraction",
  "derivation": "normalized",
  "source_refs": ["works:123", "result_objects:77"]
}
```

### `MethodSignature`

```json
{
  "contract_version": "v1",
  "object_type": "method_signature",
  "object_id": "method:123:default",
  "work_id": 123,
  "label": "profile likelihood",
  "method_kind": "statistical_fit",
  "summary_text": "Observation of ... | method signatures: profile likelihood | source=chunks",
  "status": "ready",
  "source_kind": "extraction",
  "derivation": "normalized",
  "source_refs": ["works:123", "method_objects:45"]
}
```

### `EvidenceBundle`

```json
{
  "contract_version": "v1",
  "object_type": "evidence_bundle",
  "object_id": "evidence:ideas:run-1",
  "status": "ready",
  "source_kind": "service",
  "derivation": "aggregated",
  "subject_refs": ["idea_candidate:idea-1"],
  "items": [
    {
      "citation_id": "E1",
      "evidence_key": "idea_candidate:idea-1",
      "object_type": "idea_candidate",
      "object_id": "idea_candidate:idea-1",
      "title": "Observation of ...",
      "occurrences": 1
    }
  ]
}
```

### `TraceStep`

```json
{
  "contract_version": "v1",
  "object_type": "trace_step",
  "object_id": "trace:run-1:step-1",
  "step_type": "retrieve",
  "summary": "retrieved 3 evidence items",
  "status": "ready",
  "source_kind": "service",
  "derivation": "summarized",
  "step_index": 1
}
```
