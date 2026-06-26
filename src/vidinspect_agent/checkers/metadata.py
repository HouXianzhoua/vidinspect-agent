from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from vidinspect_agent.checkers.base import BaseChecker
from vidinspect_agent.models import CheckResult, Severity


def _severity(value: str) -> Severity:
    try:
        return Severity(str(value).lower())
    except ValueError:
        return Severity.WARN


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

        spec = self._check_declared_vs_measured(metadata)
        if spec is not None:
            results.append(spec)

        return results

    def _check_declared_vs_measured(self, metadata: dict[str, Any]) -> CheckResult | None:
        """「声明 vs 实测」交叉核对（§3）：info.json 声明视频规格 vs ffprobe 实测。

        仅当 LeRobot 摄入层注入了 ``declared_video`` 时生效（纯视频无声明 → 返回 ``None``，
        不产生该项）。比较 codec / 分辨率 / fps / pix_fmt / has_audio，任一不一致默认 WARN
        （声明与实际不符通常意味着导出 / 转码环节有偏差，值得人工复核）。
        """
        cfg = self.config.get("metadata", {})
        if not cfg.get("spec_match", True):
            return None
        lr = metadata.get("lerobot")
        declared = lr.get("declared_video") if isinstance(lr, dict) else None
        if not isinstance(declared, dict) or not any(
            declared.get(k) is not None for k in ("codec", "width", "height", "fps", "pix_fmt")
        ):
            return None

        fps_tol = float(cfg.get("fps_tol", 1.0))
        fail_severity = _severity(cfg.get("spec_match_severity", "warn"))

        mismatches: list[str] = []
        checked: dict[str, Any] = {}

        def _cmp(field: str, declared_val: Any, measured_val: Any, *, num_tol: float | None = None):
            if declared_val is None:
                return
            checked[field] = {"declared": declared_val, "measured": measured_val}
            if measured_val is None:
                mismatches.append(f"{field}: 声明 {declared_val} / 实测 缺失")
                return
            if num_tol is not None:
                try:
                    if abs(float(declared_val) - float(measured_val)) > num_tol:
                        mismatches.append(
                            f"{field}: 声明 {declared_val} / 实测 {round(float(measured_val), 3)}"
                        )
                    return
                except (TypeError, ValueError):
                    pass
            if str(declared_val).strip().lower() != str(measured_val).strip().lower():
                mismatches.append(f"{field}: 声明 {declared_val} / 实测 {measured_val}")

        _cmp("codec", declared.get("codec"), metadata.get("codec"))
        _cmp("width", declared.get("width"), metadata.get("width"), num_tol=0)
        _cmp("height", declared.get("height"), metadata.get("height"), num_tol=0)
        _cmp("fps", declared.get("fps"), metadata.get("fps"), num_tol=fps_tol)
        _cmp("pix_fmt", declared.get("pix_fmt"), metadata.get("pix_fmt"))
        if declared.get("has_audio") is not None:
            _cmp("has_audio", bool(declared.get("has_audio")), bool(metadata.get("has_audio")))

        details = {"declared_vs_measured": checked, "n_mismatch": len(mismatches)}
        if mismatches:
            return CheckResult(
                name="spec_match",
                severity=fail_severity,
                message="声明规格与实测不一致: " + "；".join(mismatches),
                details=details,
            )
        return CheckResult(
            name="spec_match",
            severity=Severity.PASS,
            message="声明规格与实测一致",
            details=details,
        )
