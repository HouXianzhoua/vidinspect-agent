from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yaml

from vidinspect_agent.lerobot import GroupResolver
from vidinspect_agent.models import InspectionSummary
from vidinspect_agent.pipeline import inspect_video

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".m4v"}


def load_config(config_path: Path | None = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).resolve().parents[2] / "config" / "default.yaml"
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def discover_videos(root: Path, recursive: bool = False) -> list[Path]:
    root = root.resolve()
    if root.is_file():
        return [root]
    pattern = "**/*" if recursive else "*"
    return sorted(
        p
        for p in root.glob(pattern)
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


class VidInspectAgent:
    """Orchestrates batch video quality inspection."""

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or load_config()

    def inspect_paths(
        self,
        paths: Iterable[Path],
        *,
        recursive: bool = False,
    ) -> InspectionSummary:
        videos: list[Path] = []
        for path in paths:
            videos.extend(discover_videos(Path(path), recursive=recursive))

        # LeRobot 摄入层：把组级信息（robot/task/目标物体/声明规格/标定/parquet 指针）按组
        # 缓存后注入每个视频的 metadata；非 LeRobot 组或关闭时退化为纯视频检测。
        lerobot_enabled = self.config.get("lerobot", {}).get("enabled", True)
        resolver = GroupResolver(enabled=lerobot_enabled)

        summary = InspectionSummary(total=len(videos))
        for video in videos:
            extra_metadata = resolver.metadata_for(video)
            report = inspect_video(video, self.config, extra_metadata=extra_metadata)
            summary.reports.append(report)
            if report.passed:
                summary.passed += 1
            else:
                summary.failed += 1
        return summary
