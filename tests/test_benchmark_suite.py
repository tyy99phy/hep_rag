from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hep_rag_v2.benchmark_suite import (  # noqa: E402
    build_benchmark_manifest,
    default_benchmark_scenarios,
    default_case_fixture_path,
    load_benchmark_cases,
    write_benchmark_manifest,
)


class BenchmarkSuiteTests(unittest.TestCase):
    def test_default_case_fixture_loads_expected_categories(self) -> None:
        cases = load_benchmark_cases(default_case_fixture_path(repo_root=ROOT))

        self.assertGreaterEqual(len(cases), 6)
        self.assertIn("latest_result_summary", {item.category for item in cases})
        self.assertIn("method_transfer", {item.category for item in cases})

    def test_default_case_fixture_includes_object_gold_fields(self) -> None:
        cases = load_benchmark_cases(default_case_fixture_path(repo_root=ROOT))
        result_case = next(item for item in cases if item.case_id == "result-expected-vs-observed")
        trace_case = next(item for item in cases if item.case_id == "thinking-engine-traceable-idea")

        self.assertGreaterEqual(len(result_case.gold_evidence_ids), 1)
        self.assertEqual(result_case.gold_result_signature, "expected_vs_observed_limit")
        self.assertIsNone(result_case.gold_method_signature)
        self.assertEqual(trace_case.gold_method_signature, "method_transfer_traceable_idea")
        self.assertEqual(trace_case.gold_trace_outline, (
            "retrieve_method_evidence",
            "compare_transfer_candidates",
            "state_risk_and_next_check",
        ))

    def test_default_scenarios_cover_ablation_matrix(self) -> None:
        scenarios = default_benchmark_scenarios()

        self.assertEqual([item.name for item in scenarios], [
            "llm_only",
            "llm_plus_retrieve",
            "llm_plus_retrieve_and_structure",
            "thinking_engine_trace",
        ])
        self.assertFalse(scenarios[0].database_enabled)
        self.assertTrue(scenarios[1].database_enabled)
        self.assertEqual(scenarios[-1].answer_mode, "trace_backed_idea_generation")

    def test_write_benchmark_manifest_exports_json(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "rag-effect.json"
            write_benchmark_manifest(path, model_label="tiny-model")

            self.assertTrue(path.exists())
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["model_label"], "tiny-model")
            self.assertGreaterEqual(payload["case_count"], 6)
            self.assertEqual(payload["scenario_count"], 4)

    def test_manifest_exposes_target_lanes(self) -> None:
        manifest = build_benchmark_manifest(model_label="weak-model")
        self.assertIn("works_chunks", manifest["target_lanes"])
        self.assertIn("methods_future", manifest["target_lanes"])
        self.assertIn("thinking_engine", manifest["categories"])
        self.assertIn("work_capsule", manifest["contract_objects"])
        self.assertIn("evidence_bundle", manifest["contract_objects"])
        self.assertEqual(manifest["contract_wire_format"]["contract_version"], "v1")
        self.assertEqual(
            manifest["contract_wire_format"]["required_fields"],
            ["object_id", "source_kind", "status", "source_refs", "derivation"],
        )
        self.assertEqual(
            manifest["object_gold_fields"],
            [
                "gold_evidence_ids",
                "gold_result_signature",
                "gold_method_signature",
                "gold_trace_outline",
            ],
        )
        self.assertGreaterEqual(manifest["object_gold_case_count"], 4)
        trace_case = next(item for item in manifest["cases"] if item["case_id"] == "thinking-engine-traceable-idea")
        self.assertEqual(trace_case["gold_method_signature"], "method_transfer_traceable_idea")
        self.assertEqual(
            trace_case["gold_trace_outline"],
            [
                "retrieve_method_evidence",
                "compare_transfer_candidates",
                "state_risk_and_next_check",
            ],
        )


if __name__ == "__main__":
    unittest.main()
