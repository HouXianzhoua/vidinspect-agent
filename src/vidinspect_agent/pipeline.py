from __future__ import annotations

from pathlib import Path
from typing import Any

from vidinspect_agent.checkers import IntegrityChecker, MetadataChecker, VisualChecker
from vidinspect_agent.checkers.base import BaseChecker
from vidinspect_agent.checkers.metadata import probe_video
from vidinspect_agent.models import CheckResult, Severity, VideoReport


def _build_checkers(config: dict[str, Any]) -> list[BaseChecker]:
    checks = config.get("checks", {})
    checkers: list[BaseChecker] = []
    if checks.get("integrity", True):
        checkers.append(IntegrityChecker(config))
    if checks.get("metadata", True):
        checkers.append(MetadataChecker(config))
    if checks.get("visual", True):
        checkers.append(VisualChecker(config))
    return checkers


def _extract_metadata(probe: dict[str, Any]) -> dict[str, Any]:
    video_stream = next(
        (s for s in probe.get("streams", []) if s.get("codec_type") == "video"),
        {},
    )
    fmt = probe.get("format", {})

    width = video_stream.get("width")
    height = video_stream.get("height")
    fps = None
    rate = video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate")
    if rate and rate != "0/0" and "/" in rate:
        num, den = rate.split("/", 1)
        if float(den) != 0:
            fps = float(num) / float(den)

    duration = fmt.get("duration")
    duration_sec = float(duration) if duration is not None else None

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "duration_sec": duration_sec,
        "codec": video_stream.get("codec_name"),
        "format": fmt.get("format_name"),
        "size_bytes": int(fmt["size"]) if fmt.get("size") else None,
    }


def _report_failed(results: list[CheckResult]) -> bool:
    return any(r.severity == Severity.FAIL for r in results)


def inspect_video(path: Path, config: dict[str, Any]) -> VideoReport:
    path = path.resolve()
    results: list[CheckResult] = []
    metadata: dict[str, Any] = {}

    try:
        probe = probe_video(path)
        metadata = _extract_metadata(probe)
    except Exception as exc:  # noqa: BLE001 - surface probe errors as check results
        results.append(
            CheckResult(
                name="probe",
                severity=Severity.FAIL,
                message=str(exc),
            )
        )
        return VideoReport(path=path, passed=False, results=results, metadata=metadata)

    for checker in _build_checkers(config):
        results.extend(checker.check(path, metadata))

    return VideoReport(
        path=path,
        passed=not _report_failed(results),
        results=results,
        metadata=metadata,
    )
