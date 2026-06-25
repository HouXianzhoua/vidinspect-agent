"""开始 / 结束归位停留时间检测器（质检规范序号 3）。

规范：视频开头或结尾存在超过 2s 的静止（机械臂归位后长时间停留、空等）。

与 ``static`` 检测器的区别：``static`` 关注「整段是否几乎无运动」并对全片做均匀
采样，无法定位首尾的静止时长；本检测器顺序解码整段视频，逐帧算 ``mean|ΔY|``，
统计 **开头连续静止帧数** 与 **结尾连续静止帧数**，按 fps 换算成秒：

    leading_static_sec  = (开头连续 diff<motion_thr 的帧数) / fps
    trailing_static_sec = (结尾连续 diff<motion_thr 的帧数) / fps

任一端超过 ``max_static_sec``（默认 2.0s）即命中（默认 severity=warn）。

**自适应阈值**：64×48 下采样会压缩运动幅度的动态范围（全画面运动 ~1.3 vs 静止
噪声 ~0.5），固定绝对阈值很脆弱。因此「静止」阈值取自该视频自身的运动区间：

    lo = p10(diffs)   hi = p90(diffs)
    motion_thr_eff = max(abs_floor, lo + rel_frac * (hi - lo))

这样能自适应不同噪声/运动尺度：有明显运动的视频阈值自然抬高（不把弱运动误判静止），
而真正的归位停留（接近噪声下限）仍落在阈值之下。整段几乎无运动的视频（无运动参照）
交由 ``static`` 检测器兜底，本检测器不会过度误报。

score 语义：越高越好。这里取 ``1 - max(leading, trailing)/max_static_sec`` 截断到
[0, 1]，越接近 0 表示首尾静止越久。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from vidinspect_agent.checkers._frames import probe_fps, stream_gray_diffs
from vidinspect_agent.checkers.base import BaseChecker
from vidinspect_agent.models import CheckResult, Severity


class EndpointStaticChecker(BaseChecker):
    """开头 / 结尾归位停留时间过长检测。"""

    name = "endpoint_static"

    def check(self, path: Path, metadata: dict[str, Any]) -> list[CheckResult]:
        cfg = self.config.get("endpoint_static", {})
        abs_floor = cfg.get("abs_floor", 0.3)
        rel_frac = cfg.get("rel_frac", 0.35)
        max_static_sec = cfg.get("max_static_sec", 2.0)
        downscale = tuple(cfg.get("downscale", (64, 48)))
        timeout = cfg.get("timeout", 60.0)
        fail_severity = _severity(cfg.get("severity", "warn"))

        fps = metadata.get("fps") or probe_fps(str(path))
        if not fps or fps <= 0:
            return [self._warn("无法获取帧率，跳过首尾静止检测", {"error": "no_fps"})]

        diffs, n, err = stream_gray_diffs(str(path), downscale, timeout)
        if diffs is None:
            return [self._warn(f"首尾静止检测未完成: {err}", {"error": err})]

        lo = float(np.percentile(diffs, 10))
        hi = float(np.percentile(diffs, 90))
        motion_thr = max(abs_floor, lo + rel_frac * (hi - lo))
        is_static = diffs < motion_thr
        leading_frames = _leading_run(is_static)
        trailing_frames = _trailing_run(is_static)
        # diffs[i] 描述第 i→i+1 帧的变化；连续 k 个静止 diff 对应 k+1 帧静止区间，
        # 时长约为 k / fps（区间端点之间的播放时长）。
        leading_sec = leading_frames / fps
        trailing_sec = trailing_frames / fps

        worst = max(leading_sec, trailing_sec)
        score = float(np.clip(1.0 - worst / max_static_sec, 0.0, 1.0))
        details = {
            "score": round(score, 4),
            "leading_static_sec": round(leading_sec, 3),
            "trailing_static_sec": round(trailing_sec, 3),
            "max_static_sec": max_static_sec,
            "motion_thr_eff": round(motion_thr, 4),
            "fps": round(float(fps), 3),
            "total_frames": n,
        }

        hits = []
        if leading_sec > max_static_sec:
            hits.append(f"开头静止 {leading_sec:.1f}s")
        if trailing_sec > max_static_sec:
            hits.append(f"结尾静止 {trailing_sec:.1f}s")

        if hits:
            return [
                CheckResult(
                    name="endpoint_static",
                    severity=fail_severity,
                    message=f"首尾归位停留过长: {'、'.join(hits)} (上限 {max_static_sec:.1f}s)",
                    details=details,
                )
            ]
        return [
            CheckResult(
                name="endpoint_static",
                severity=Severity.PASS,
                message=(
                    f"首尾停留正常: 开头 {leading_sec:.1f}s / 结尾 {trailing_sec:.1f}s "
                    f"(上限 {max_static_sec:.1f}s)"
                ),
                details=details,
            )
        ]

    @staticmethod
    def _warn(msg: str, details: dict[str, Any]) -> CheckResult:
        return CheckResult(
            name="endpoint_static",
            severity=Severity.WARN,
            message=msg,
            details=details,
        )


def _leading_run(mask: np.ndarray) -> int:
    """开头连续 True 的个数。"""
    idx = np.argmin(mask)  # 第一个 False 的位置
    if mask.all():
        return int(mask.size)
    return int(idx)


def _trailing_run(mask: np.ndarray) -> int:
    """结尾连续 True 的个数。"""
    if mask.all():
        return int(mask.size)
    rev = mask[::-1]
    return int(np.argmin(rev))


def _severity(value: str) -> Severity:
    try:
        return Severity(str(value).lower())
    except ValueError:
        return Severity.WARN
