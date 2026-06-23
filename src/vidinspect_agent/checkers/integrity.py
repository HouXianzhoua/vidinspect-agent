from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from vidinspect_agent.checkers.base import BaseChecker
from vidinspect_agent.models import CheckResult, Severity


class IntegrityChecker(BaseChecker):
    name = "integrity"

    def check(self, path: Path, metadata: dict[str, Any]) -> list[CheckResult]:
        results: list[CheckResult] = []

        if not path.is_file():
            return [
                CheckResult(
                    name="file_exists",
                    severity=Severity.FAIL,
                    message="文件不存在",
                )
            ]

        if path.stat().st_size == 0:
            return [
                CheckResult(
                    name="file_size",
                    severity=Severity.FAIL,
                    message="文件为空",
                )
            ]

        results.append(
            CheckResult(
                name="file_size",
                severity=Severity.PASS,
                message=f"文件大小 {path.stat().st_size} bytes",
                details={"size_bytes": path.stat().st_size},
            )
        )

        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "null",
            "-",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0 or proc.stderr.strip():
            results.append(
                CheckResult(
                    name="decode",
                    severity=Severity.FAIL,
                    message="视频解码失败或存在损坏帧",
                    details={"stderr": proc.stderr.strip()[:500]},
                )
            )
        else:
            results.append(
                CheckResult(
                    name="decode",
                    severity=Severity.PASS,
                    message="视频可完整解码",
                )
            )

        return results
