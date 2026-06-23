from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from vidinspect_agent.agent import VidInspectAgent, load_config


def _print_summary(console: Console, summary) -> None:
    table = Table(title="VidInspect 质检结果")
    table.add_column("文件")
    table.add_column("状态")
    table.add_column("问题")

    for report in summary.reports:
        issues = [
            r.message
            for r in report.results
            if r.severity.value in {"fail", "warn"}
        ]
        table.add_row(
            str(report.path),
            "[green]PASS[/green]" if report.passed else "[red]FAIL[/red]",
            "; ".join(issues) if issues else "-",
        )
    console.print(table)
    console.print(
        f"\n合计: {summary.total} | 通过: {summary.passed} | 失败: {summary.failed}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vidinspect",
        description="自动化视频数据质检 Agent",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    inspect_parser = sub.add_parser("inspect", help="检查视频文件或目录")
    inspect_parser.add_argument("paths", nargs="+", type=Path, help="视频路径或目录")
    inspect_parser.add_argument(
        "-r", "--recursive", action="store_true", help="递归扫描子目录"
    )
    inspect_parser.add_argument(
        "-c", "--config", type=Path, default=None, help="配置文件路径"
    )
    inspect_parser.add_argument(
        "-o", "--output", type=Path, default=None, help="JSON 报告输出路径"
    )

    args = parser.parse_args(argv)
    console = Console()

    if args.command == "inspect":
        config = load_config(args.config)
        agent = VidInspectAgent(config)
        summary = agent.inspect_paths(args.paths, recursive=args.recursive)

        _print_summary(console, summary)

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(summary.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            console.print(f"\n报告已写入: {args.output}")

        fail_on_error = config.get("output", {}).get("fail_on_error", True)
        if fail_on_error and summary.failed > 0:
            return 1
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
