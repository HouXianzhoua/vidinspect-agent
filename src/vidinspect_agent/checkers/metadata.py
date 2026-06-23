from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from vidinspect_agent.checkers.base import BaseChecker
from vidinspect_agent.models import CheckResult, Severity


def probe_video(path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "ffprobe failed")
    return json.loads(proc.stdout)


class MetadataChecker(BaseChecker):
    name = "metadata"

    def check(self, path: Path, metadata: dict[str, Any]) -> list[CheckResult]:
        thresholds = self.config.get("thresholds", {})
        results: list[CheckResult] = []

        width = metadata.get("width")
        height = metadata.get("height")
        fps = metadata.get("fps")
        duration = metadata.get("duration_sec")

        min_w = thresholds.get("min_width", 640)
        min_h = thresholds.get("min_height", 480)
        min_fps = thresholds.get("min_fps", 15)
        max_fps = thresholds.get("max_fps", 120)
        min_duration = thresholds.get("min_duration_sec", 0.5)

        if width is None or height is None:
            results.append(
                CheckResult(
                    name="resolution",
                    severity=Severity.FAIL,
                    message="无法读取视频分辨率",
                )
            )
        elif width < min_w or height < min_h:
            results.append(
                CheckResult(
                    name="resolution",
                    severity=Severity.FAIL,
                    message=f"分辨率过低: {width}x{height} (最低 {min_w}x{min_h})",
                    details={"width": width, "height": height},
                )
            )
        else:
            results.append(
                CheckResult(
                    name="resolution",
                    severity=Severity.PASS,
                    message=f"分辨率 {width}x{height}",
                    details={"width": width, "height": height},
                )
            )

        if fps is None:
            results.append(
                CheckResult(
                    name="fps",
                    severity=Severity.WARN,
                    message="无法读取帧率",
                )
            )
        elif fps < min_fps or fps > max_fps:
            results.append(
                CheckResult(
                    name="fps",
                    severity=Severity.FAIL,
                    message=f"帧率异常: {fps:.2f} (允许 {min_fps}-{max_fps})",
                    details={"fps": fps},
                )
            )
        else:
            results.append(
                CheckResult(
                    name="fps",
                    severity=Severity.PASS,
                    message=f"帧率 {fps:.2f}",
                    details={"fps": fps},
                )
            )

        if duration is None:
            results.append(
                CheckResult(
                    name="duration",
                    severity=Severity.WARN,
                    message="无法读取时长",
                )
            )
        elif duration < min_duration:
            results.append(
                CheckResult(
                    name="duration",
                    severity=Severity.FAIL,
                    message=f"时长过短: {duration:.2f}s (最低 {min_duration}s)",
                    details={"duration_sec": duration},
                )
            )
        else:
            results.append(
                CheckResult(
                    name="duration",
                    severity=Severity.PASS,
                    message=f"时长 {duration:.2f}s",
                    details={"duration_sec": duration},
                )
            )

        return results
