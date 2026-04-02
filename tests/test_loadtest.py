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

from hep_rag_v2.loadtest import BenchmarkOptions, BenchmarkTier, benchmark_scale, default_scale_tiers


class LoadtestBenchmarkTests(unittest.TestCase):
    def test_default_scale_tiers_cover_step3_targets(self) -> None:
        tiers = default_scale_tiers()

        self.assertEqual(sorted(tiers), ["10k", "100k", "50k"])
        self.assertEqual(tiers["10k"].work_count, 10_000)
        self.assertEqual(tiers["50k"].work_count, 50_000)
        self.assertEqual(tiers["100k"].work_count, 100_000)

    def test_dry_run_returns_benchmark_plan_without_creating_workspace_state(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            result = benchmark_scale(
                default_scale_tiers()["100k"],
                options=BenchmarkOptions(workspace_root=Path(td), dry_run=True),
            )

            self.assertTrue(result["dry_run"])
            self.assertEqual(result["tier"], "100k")
            self.assertEqual(result["work_count"], 100_000)
            self.assertIn("export_path", result)
            self.assertFalse((Path(td) / "db" / "hep_rag_v2.db").exists())

    def test_small_benchmark_run_exports_metrics_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            result = benchmark_scale(
                BenchmarkTier(name="tiny", work_count=24),
                options=BenchmarkOptions(
                    workspace_root=Path(td),
                    query_repeats=1,
                    export_filename="tiny-benchmark.json",
                ),
            )

            self.assertFalse(result["dry_run"])
            self.assertEqual(result["ingest"]["work_count"], 24)
            self.assertEqual(result["search"]["index_rows"], 24)
            self.assertGreaterEqual(result["search"]["latency_ms"]["p50"], 0.0)
            self.assertGreaterEqual(result["vectors"]["latency_ms"]["p95"], 0.0)
            self.assertTrue(Path(result["export_path"]).exists())

            payload = json.loads(Path(result["export_path"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["tier"], "tiny")
            self.assertEqual(payload["ingest"]["work_count"], 24)
            self.assertEqual(payload["vectors"]["index_rows"], 24)


if __name__ == "__main__":
    unittest.main()
