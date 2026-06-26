"""画面过暗 / 欠曝检测器（质检规范序号 20 子项：相机画面问题）。

规范序号 20「相机画面问题」（偏色 / 白平衡异常 / 黑屏 / 花屏等）定义过宽，按子项拆分后
本检测器只负责其中**最确定、误报率最低**的一项——**画面整体过暗 / 欠曝**：

- 黑屏（接近全黑、无信号）由 `visual`（blackdetect）覆盖；
- 花屏 / 解码损坏由 `integrity`（ffmpeg 全解码）覆盖；
- 偏色 / 白平衡异常属色彩判断，真实场景误报率高（暖光、彩色桌布、单色主体均会触发），
  暂不实现。

原理：均匀采样若干帧转灰度（≈ 亮度 Y），逐帧取全画面平均亮度，再对各帧取**中位数**
（对个别曝光突变帧鲁棒）。`luma_median < min_luma`（默认 40.0，0–255 量纲）即判过暗。

判定与黑屏的边界：纯黑（mean ≈ 0–16）归 `visual`；本项抓的是**整体偏暗但非全黑**的
欠曝画面（能看到内容但明显过暗）。两者信号互补。

降级：缺 OpenCV / 读帧失败 → WARN，不抛异常。

score 语义：越高越好，取 ``clip(luma_median / min_luma, 0, 1)``。

> 注意：合法的昏暗场景（如刻意压暗的环境）也可能落在阈值下，故默认仅报 WARN 作人工
> 复核提示；阈值建议按数据源在 ``config['brightness']`` 标定。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from vidinspect_agent.checkers._frames import read_frames_gray
from vidinspect_agent.checkers.base import BaseChecker
from vidinspect_agent.models import CheckResult, Severity


class BrightnessChecker(BaseChecker):
    """画面过暗 / 欠曝检测（全画面平均亮度统计）。"""

    name = "brightness"

    def check(self, path: Path, metadata: dict[str, Any]) -> list[CheckResult]:
        cfg = self.config.get("brightness", {})
        min_luma = float(cfg.get("min_luma", 40.0))
        n_frames = int(cfg.get("n_frames", 16))
        max_h = int(cfg.get("max_h", 240))
        use_baseline = cfg.get("use_stats_baseline", True)
        baseline_rel_frac = float(cfg.get("baseline_rel_frac", 0.4))
        fail_severity = _severity(cfg.get("severity", "warn"))

        # 每数据源亮度基线：用 stats.json 像素均值（§1 注入）按比例定阈值，替代写死的 40。
        # 不同数据源 / 机位的正常亮度分布差异大，固定阈值要么误报要么漏报。缺基线时退回固定值。
        baseline = _pixel_luma_baseline(metadata) if use_baseline else None
        min_luma_eff = min_luma
        threshold_source = "fixed"
        if baseline is not None and baseline > 0:
            min_luma_eff = baseline * baseline_rel_frac
            threshold_source = "stats_baseline"

        try:
            import cv2  # noqa: F401  (确认依赖存在)
        except Exception:  # noqa: BLE001
            return [self._warn("缺少 OpenCV，跳过过暗检测", {"error": "no_opencv"})]

        gray = read_frames_gray(str(path), n_frames, max_h)
        if gray is None or gray.shape[0] < 1:
            return [self._warn("无法采样帧用于过暗检测", {"error": "read_failed"})]

        luma_per_frame = [float(fr.mean()) for fr in gray]
        stats = evaluate_brightness(luma_per_frame, min_luma_eff)

        details = {
            "score": round(stats["score"], 4),
            "luma_median": round(stats["luma_median"], 3),
            "luma_mean": round(stats["luma_mean"], 3),
            "luma_min": round(stats["luma_min"], 3),
            "min_luma": round(min_luma_eff, 3),
            "min_luma_base": min_luma,
            "threshold_source": threshold_source,
            "pixel_luma_baseline": round(baseline, 3) if baseline is not None else None,
            "frames_used": len(luma_per_frame),
        }

        if stats["detected"]:
            return [
                CheckResult(
                    name="brightness",
                    severity=fail_severity,
                    message=(
                        f"疑似画面过暗/欠曝: 平均亮度={stats['luma_median']:.1f} "
                        f"(下限 {min_luma_eff:.1f})"
                    ),
                    details=details,
                )
            ]
        return [
            CheckResult(
                name="brightness",
                severity=Severity.PASS,
                message=f"画面亮度正常: 平均亮度={stats['luma_median']:.1f}",
                details=details,
            )
        ]

    @staticmethod
    def _warn(msg: str, details: dict[str, Any]) -> CheckResult:
        return CheckResult(
            name="brightness",
            severity=Severity.WARN,
            message=msg,
            details=details,
        )


def _pixel_luma_baseline(metadata: dict[str, Any]) -> float | None:
    """从 §1 摄入层注入的 ``metadata["lerobot"]["pixel_luma_baseline"]`` 取像素亮度基线。"""
    lr = metadata.get("lerobot")
    if not isinstance(lr, dict):
        return None
    value = lr.get("pixel_luma_baseline")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def evaluate_brightness(luma_per_frame, min_luma: float = 40.0) -> dict[str, Any]:
    """对逐帧平均亮度序列做过暗判定（纯函数，便于单测）。

    取各帧平均亮度的**中位数**与 ``min_luma`` 比较；``luma_median < min_luma`` 即判过暗。
    返回 ``detected`` / ``luma_median`` / ``luma_mean`` / ``luma_min`` / ``score``。
    空序列返回 ``detected=False`` 且各统计量为 0（交由上层降级）。
    """
    arr = np.asarray(list(luma_per_frame), dtype=np.float64)
    if arr.size == 0:
        return {
            "detected": False,
            "luma_median": 0.0,
            "luma_mean": 0.0,
            "luma_min": 0.0,
            "score": 0.0,
        }
    luma_median = float(np.median(arr))
    score = float(np.clip(luma_median / min_luma, 0.0, 1.0)) if min_luma > 0 else 0.0
    return {
        "detected": luma_median < min_luma,
        "luma_median": luma_median,
        "luma_mean": float(arr.mean()),
        "luma_min": float(arr.min()),
        "score": score,
    }


def _severity(value: str) -> Severity:
    try:
        return Severity(str(value).lower())
    except ValueError:
        return Severity.WARN
