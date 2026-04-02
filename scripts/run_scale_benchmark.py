from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hep_rag_v2.loadtest import BenchmarkOptions, benchmark_scale, default_scale_tiers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行 metadata-only scale benchmark harness。")
    parser.add_argument("--tier", choices=sorted(default_scale_tiers()), required=True, help="基准 tier：10k/50k/100k")
    parser.add_argument("--workspace-root", type=Path, default=None, help="独立 benchmark workspace 根目录")
    parser.add_argument("--query-repeats", type=int, default=3, help="每组查询重复次数")
    parser.add_argument("--query-limit", type=int, default=10, help="每次检索返回条目上限")
    parser.add_argument("--dry-run", action="store_true", help="只输出计划，不创建数据库")
    parser.add_argument("--skip-search", action="store_true", help="跳过 BM25 work-search 基准")
    parser.add_argument("--skip-vectors", action="store_true", help="跳过 work-vector 基准")
    parser.add_argument("--export-filename", default=None, help="导出 JSON 文件名")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = benchmark_scale(
        default_scale_tiers()[args.tier],
        options=BenchmarkOptions(
            workspace_root=args.workspace_root,
            query_repeats=max(1, args.query_repeats),
            query_limit=max(1, args.query_limit),
            build_search=not args.skip_search,
            build_vectors=not args.skip_vectors,
            dry_run=args.dry_run,
            export_filename=args.export_filename,
        ),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
