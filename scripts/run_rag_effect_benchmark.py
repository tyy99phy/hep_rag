from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hep_rag_v2.benchmark_suite import (  # noqa: E402
    build_benchmark_manifest,
    default_case_fixture_path,
    load_benchmark_cases,
    write_benchmark_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="输出弱模型 + 外接数据库增益 benchmark manifest。")
    parser.add_argument("--cases", type=Path, default=None, help="benchmark case fixture JSON 路径")
    parser.add_argument("--model-label", default="weak-model", help="模型标签，用于结果归档")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="manifest 输出路径；默认写到 ./.omx/benchmarks/<model-label>-rag-effect-manifest.json",
    )
    parser.add_argument("--print-cases", action="store_true", help="额外打印 case 摘要")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fixture = args.cases or default_case_fixture_path(repo_root=ROOT)
    cases = load_benchmark_cases(fixture)
    output = args.output or (ROOT / ".omx" / "benchmarks" / f"{args.model_label}-rag-effect-manifest.json")
    manifest_path = write_benchmark_manifest(output, cases=cases, model_label=args.model_label)
    manifest = build_benchmark_manifest(cases=cases, model_label=args.model_label)
    print(json.dumps({"fixture": str(fixture), "output": str(manifest_path), **manifest}, ensure_ascii=False, indent=2))
    if args.print_cases:
        print("\n# case-summary")
        for item in cases:
            print(f"- {item.case_id}: {item.category} -> {item.target_lane}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
