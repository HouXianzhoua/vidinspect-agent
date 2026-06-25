"""画面卡死 / 长时间卡帧检测器（质检规范序号 5）。

规范：视频中出现长时间卡帧（画面卡死 —— 解码器卡住、采集线程阻塞，
连续多帧完全相同 / 几乎相同）。

与 ``dup_frame`` 的区别：``dup_frame`` 关注「复制帧伪装高帧率导致的整体卡顿 /
时间变慢」，以比例（keep_ratio / dup_ratio_strict）与周期性为判据；而本检测器
关注 **单段最长冻结时长**——视频任意位置出现一段足够长的连续冻结即算卡死，
即使该段只占全片很小比例（此时 dup_frame 的 static_like 不会触发）。

原理：顺序解码整段视频为下采样灰度，逐帧算 ``mean|ΔY|``，用严格阈值
``freeze_thr``（默认 0.1，近似「同一帧」）得到冻结掩码，统计最长连续冻结段，
按 fps 换算成秒：

    max_freeze_sec = (最长连续 diff<freeze_thr 的帧数) / fps

``max_freeze_sec > max_freeze_sec_thr``（默认 2.0s）即命中（默认 severity=warn）。

score 语义：越高越好，取 ``1 - max_freeze_sec/thr`` 截断到 [0, 1]。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from vidinspect_agent.checkers._frames import probe_fps, stream_gray_diffs
from vidinspect_agent.checkers.base import BaseChecker
from vidinspect_agent.models import CheckResult, Severity


class FreezeChecker(BaseChecker):
    """长时间卡帧 / 画面卡死检测。"""

    name = "freeze"

    def check(self, path: Path, metadata: dict[str, Any]) -> list[CheckResult]:
        cfg = self.config.get("freeze", {})
        freeze_thr = cfg.get("freeze_thr", 0.1)
        max_freeze_sec_thr = cfg.get("max_freeze_sec", 2.0)
        downscale = tuple(cfg.get("downscale", (64, 48)))
        timeout = cfg.get("timeout", 60.0)
        fail_severity = _severity(cfg.get("severity", "warn"))

        fps = metadata.get("fps") or probe_fps(str(path))
        if not fps or fps <= 0:
            return [self._warn("无法获取帧率，跳过卡帧检测", {"error": "no_fps"})]

        diffs, n, err = stream_gray_diffs(str(path), downscale, timeout)
        if diffs is None:
            return [self._warn(f"卡帧检测未完成: {err}", {"error": err})]

        frozen = diffs < freeze_thr
        max_run, run_start = _longest_run(frozen)
        # 连续 k 个冻结 diff 对应 k+1 帧停在同一画面，时长约 k / fps。
        max_freeze_sec = max_run / fps
        freeze_ratio = float(frozen.mean())

        score = float(np.clip(1.0 - max_freeze_sec / max_freeze_sec_thr, 0.0, 1.0))
        details = {
            "score": round(score, 4),
            "max_freeze_sec": round(max_freeze_sec, 3),
            "max_freeze_frames": int(max_run),
            "freeze_start_sec": round(run_start / fps, 3) if max_run else None,
            "freeze_ratio": round(freeze_ratio, 4),
            "max_freeze_sec_thr": max_freeze_sec_thr,
            "freeze_thr": freeze_thr,
            "fps": round(float(fps), 3),
            "total_frames": n,
        }

        if max_freeze_sec > max_freeze_sec_thr:
            return [
                CheckResult(
                    name="freeze",
                    severity=fail_severity,
                    message=(
                        f"疑似画面卡死: 最长卡帧 {max_freeze_sec:.1f}s "
                        f"(上限 {max_freeze_sec_thr:.1f}s)"
                    ),
                    details=details,
                )
            ]
        return [
            CheckResult(
                name="freeze",
                severity=Severity.PASS,
                message=f"无明显卡帧: 最长卡帧 {max_freeze_sec:.1f}s",
                details=details,
            )
        ]

    @staticmethod
    def _warn(msg: str, details: dict[str, Any]) -> CheckResult:
        return CheckResult(
            name="freeze",
            severity=Severity.WARN,
            message=msg,
            details=details,
        )


def _longest_run(mask: np.ndarray) -> tuple[int, int]:
    """返回最长连续 True 段的长度与起始索引 (length, start)。"""
    best_len = best_start = 0
    cur_len = 0
    cur_start = 0
    for i, v in enumerate(mask):
        if v:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
        else:
            cur_len = 0
    return best_len, best_start


def _severity(value: str) -> Severity:
    try:
        return Severity(str(value).lower())
    except ValueError:
        return Severity.WARN
