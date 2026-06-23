from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from vidinspect_agent.checkers.base import BaseChecker
from vidinspect_agent.models import CheckResult, Severity


class VisualChecker(BaseChecker):
    """Lightweight visual heuristics via ffmpeg filters."""

    name = "visual"

    def check(self, path: Path, metadata: dict[str, Any]) -> list[CheckResult]:
        thresholds = self.config.get("thresholds", {})
        max_black = thresholds.get("max_black_frame_ratio", 0.3)

        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            "blackdetect=d=0.1:pix_th=0.10",
            "-an",
            "-f",
            "null",
            "-",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        black_lines = [line for line in proc.stderr.splitlines() if "black_start" in line]
        duration = metadata.get("duration_sec") or 0.0

        if duration <= 0:
            return [
                CheckResult(
                    name="black_frames",
                    severity=Severity.WARN,
                    message="无法评估黑屏比例（缺少时长信息）",
                )
            ]

        black_duration = 0.0
        for line in black_lines:
            parts = dict(
                item.split(":", 1) for item in line.split() if ":" in item
            )
            try:
                start = float(parts.get("black_start", 0))
                end = float(parts.get("black_end", start))
                black_duration += max(0.0, end - start)
            except ValueError:
                continue

        ratio = black_duration / duration if duration else 0.0
        if ratio > max_black:
            return [
                CheckResult(
                    name="black_frames",
                    severity=Severity.WARN,
                    message=f"黑屏比例偏高: {ratio:.1%} (上限 {max_black:.0%})",
                    details={"black_ratio": ratio},
                )
            ]

        return [
            CheckResult(
                name="black_frames",
                severity=Severity.PASS,
                message=f"黑屏比例 {ratio:.1%}",
                details={"black_ratio": ratio},
            )
        ]
