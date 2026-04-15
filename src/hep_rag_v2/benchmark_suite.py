from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    case_id: str
    category: str
    prompt: str
    target_lane: str
    contract_focus: tuple[str, ...]
    database_value: str
    expected_failure_without_db: str
    must_mention: tuple[str, ...]
    must_not_confuse: tuple[str, ...]
    evidence_expectation: str
    gold_evidence_ids: tuple[str, ...] = ()
    gold_result_signature: str | None = None
    gold_method_signature: str | None = None
    gold_trace_outline: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BenchmarkScenario:
    name: str
    description: str
    database_enabled: bool
    retrieval_mode: str
    answer_mode: str
    contract_targets: tuple[str, ...] = ()


def default_case_fixture_path(*, repo_root: str | Path | None = None) -> Path:
    root = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()
    return root / "tests" / "fixtures" / "rag_effect_benchmark_cases.json"


def load_benchmark_cases(path: str | Path | None = None) -> list[BenchmarkCase]:
    fixture_path = Path(path or default_case_fixture_path()).expanduser().resolve()
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Benchmark case fixture must be a list: {fixture_path}")
    return [_coerce_case(item) for item in payload]


def default_benchmark_scenarios() -> list[BenchmarkScenario]:
    return [
        BenchmarkScenario(
            name="llm_only",
            description="不接数据库，只把用户问题直接交给弱模型。",
            database_enabled=False,
            retrieval_mode="none",
            answer_mode="free_generation",
            contract_targets=(),
        ),
        BenchmarkScenario(
            name="llm_plus_retrieve",
            description="接数据库，只提供检索证据给弱模型，不做额外结构化重写。",
            database_enabled=True,
            retrieval_mode="works_chunks_hybrid",
            answer_mode="evidence_augmented_generation",
            contract_targets=("evidence_bundle",),
        ),
        BenchmarkScenario(
            name="llm_plus_retrieve_and_structure",
            description="接数据库，并要求模型基于结构化 retrieval shell 与证据注册表回答。",
            database_enabled=True,
            retrieval_mode="typed_retrieval_shell",
            answer_mode="structured_grounded_generation",
            contract_targets=("work_capsule", "evidence_bundle"),
        ),
        BenchmarkScenario(
            name="thinking_engine_trace",
            description="接数据库，并要求系统输出带 reasoning trace 的 idea-generation substrate 结果。",
            database_enabled=True,
            retrieval_mode="typed_retrieval_with_trace",
            answer_mode="trace_backed_idea_generation",
            contract_targets=("work_capsule", "evidence_bundle", "trace_step"),
        ),
    ]


def build_benchmark_manifest(
    *,
    cases: list[BenchmarkCase] | None = None,
    scenarios: list[BenchmarkScenario] | None = None,
    model_label: str = "weak-model",
) -> dict[str, Any]:
    loaded_cases = cases or load_benchmark_cases()
    loaded_scenarios = scenarios or default_benchmark_scenarios()
    categories = sorted({item.category for item in loaded_cases})
    target_lanes = sorted({item.target_lane for item in loaded_cases})
    contract_objects = sorted(
        {
            contract
            for item in loaded_cases
            for contract in item.contract_focus
        }
        | {
            contract
            for scenario in loaded_scenarios
            for contract in scenario.contract_targets
        }
    )
    object_gold_fields = [
        "gold_evidence_ids",
        "gold_result_signature",
        "gold_method_signature",
        "gold_trace_outline",
    ]
    return {
        "model_label": model_label,
        "case_count": len(loaded_cases),
        "scenario_count": len(loaded_scenarios),
        "categories": categories,
        "target_lanes": target_lanes,
        "contract_objects": contract_objects,
        "contract_wire_format": {
            "contract_version": "v1",
            "required_fields": [
                "object_id",
                "source_kind",
                "status",
                "source_refs",
                "derivation",
            ],
        },
        "object_gold_fields": object_gold_fields,
        "object_gold_case_count": sum(
            1
            for item in loaded_cases
            if item.gold_evidence_ids
            or item.gold_result_signature is not None
            or item.gold_method_signature is not None
            or item.gold_trace_outline
        ),
        "cases": [_json_ready(asdict(item)) for item in loaded_cases],
        "scenarios": [_json_ready(asdict(item)) for item in loaded_scenarios],
    }


def write_benchmark_manifest(
    output_path: str | Path,
    *,
    cases: list[BenchmarkCase] | None = None,
    scenarios: list[BenchmarkScenario] | None = None,
    model_label: str = "weak-model",
) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_benchmark_manifest(cases=cases, scenarios=scenarios, model_label=model_label)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _coerce_case(item: Any) -> BenchmarkCase:
    if not isinstance(item, dict):
        raise ValueError(f"Benchmark case must be a mapping, got: {type(item)!r}")
    return BenchmarkCase(
        case_id=str(item["case_id"]),
        category=str(item["category"]),
        prompt=str(item["prompt"]),
        target_lane=str(item["target_lane"]),
        contract_focus=tuple(str(token) for token in item.get("contract_focus") or []),
        database_value=str(item["database_value"]),
        expected_failure_without_db=str(item["expected_failure_without_db"]),
        must_mention=tuple(str(token) for token in item.get("must_mention") or []),
        must_not_confuse=tuple(str(token) for token in item.get("must_not_confuse") or []),
        evidence_expectation=str(item["evidence_expectation"]),
        gold_evidence_ids=tuple(str(token) for token in item.get("gold_evidence_ids") or []),
        gold_result_signature=_coerce_optional_string(item.get("gold_result_signature")),
        gold_method_signature=_coerce_optional_string(item.get("gold_method_signature")),
        gold_trace_outline=tuple(str(token) for token in item.get("gold_trace_outline") or []),
    )


def _coerce_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _json_ready(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload, ensure_ascii=False))
