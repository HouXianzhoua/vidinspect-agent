"""噪点检测器（质检规范序号 11）。

规范：视频存在严重噪点（高 ISO / 弱光增益 / 传感器噪声，画面布满颗粒）。

原理：Immerkær 快速噪声方差估计（"Fast Noise Variance Estimation", CVIU 1996）。
对灰度帧卷积一个二阶 Laplacian 掩码 M，该掩码对常量与一阶/二阶线性亮度变化的
响应为零（可消去大部分边缘与平滑渐变），残差主要由高频噪声贡献。噪声标准差估计：

        | 1 -2  1 |
    M = |-2  4 -2 |
        | 1 -2  1 |

    sigma = sqrt(pi/2) * Σ|M * I| / (6 * (W-2) * (H-2))

均匀采样若干帧分别估计 sigma，取 **中位数**（对个别强纹理 / 文字帧鲁棒）。
``sigma_median > max_noise_sigma``（默认 8.0，0–255 量纲）即命中（默认 severity=warn）。

降级：缺 OpenCV / 读帧失败 / 分辨率过小 → WARN，不抛异常。

score 语义：越高越好，取 ``1 - sigma_median/max_noise_sigma`` 截断到 [0, 1]。

> 注意：本方法是启发式估计，强纹理 / 高频细节场景可能抬高 sigma，故默认仅报 WARN，
> 作为人工复核的提示项；阈值可按数据源在 ``config['noise']`` 调整。
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from vidinspect_agent.checkers._frames import read_frames_gray
from vidinspect_agent.checkers.base import BaseChecker
from vidinspect_agent.models import CheckResult, Severity

_MASK = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
_SQRT_PI_2 = math.sqrt(math.pi / 2.0)


class NoiseChecker(BaseChecker):
    """严重噪点检测（Immerkær 噪声方差估计）。"""

    name = "noise"

    def check(self, path: Path, metadata: dict[str, Any]) -> list[CheckResult]:
        cfg = self.config.get("noise", {})
        max_noise_sigma = cfg.get("max_noise_sigma", 8.0)
        n_frames = cfg.get("n_frames", 16)
        max_h = cfg.get("max_h", 720)
        fail_severity = _severity(cfg.get("severity", "warn"))

        try:
            import cv2  # noqa: F401  (确认依赖存在)
        except Exception:  # noqa: BLE001
            return [self._warn("缺少 OpenCV，跳过噪点检测", {"error": "no_opencv"})]

        gray = read_frames_gray(str(path), n_frames, max_h)
        if gray is None or gray.shape[0] < 1:
            return [self._warn("无法采样帧用于噪点检测", {"error": "read_failed"})]

        sigmas = [s for s in (_estimate_sigma(fr) for fr in gray) if s is not None]
        if not sigmas:
            return [self._warn("帧分辨率过小，无法估计噪点", {"error": "too_small"})]

        sigmas_arr = np.asarray(sigmas, dtype=np.float64)
        sigma_median = float(np.median(sigmas_arr))
        sigma_mean = float(sigmas_arr.mean())
        sigma_max = float(sigmas_arr.max())

        score = float(np.clip(1.0 - sigma_median / max_noise_sigma, 0.0, 1.0))
        details = {
            "score": round(score, 4),
            "sigma_median": round(sigma_median, 3),
            "sigma_mean": round(sigma_mean, 3),
            "sigma_max": round(sigma_max, 3),
            "max_noise_sigma": max_noise_sigma,
            "frames_used": int(sigmas_arr.size),
        }

        if sigma_median > max_noise_sigma:
            return [
                CheckResult(
                    name="noise",
                    severity=fail_severity,
                    message=(
                        f"疑似严重噪点: 噪声 sigma={sigma_median:.2f} "
                        f"(上限 {max_noise_sigma:.1f})"
                    ),
                    details=details,
                )
            ]
        return [
            CheckResult(
                name="noise",
                severity=Severity.PASS,
                message=f"噪点正常: 噪声 sigma={sigma_median:.2f}",
                details=details,
            )
        ]

    @staticmethod
    def _warn(msg: str, details: dict[str, Any]) -> CheckResult:
        return CheckResult(
            name="noise",
            severity=Severity.WARN,
            message=msg,
            details=details,
        )


def _estimate_sigma(frame: np.ndarray) -> float | None:
    """单帧 Immerkær 噪声标准差估计；分辨率过小返回 None。"""
    import cv2

    h, w = frame.shape[:2]
    if h < 3 or w < 3:
        return None
    src = frame if frame.dtype == np.float32 else frame.astype(np.float32)
    conv = cv2.filter2D(src, -1, _MASK, borderType=cv2.BORDER_REPLICATE)
    interior = conv[1:-1, 1:-1]
    total = float(np.sum(np.abs(interior), dtype=np.float64))
    return _SQRT_PI_2 * total / (6.0 * (w - 2) * (h - 2))


def _severity(value: str) -> Severity:
    try:
        return Severity(str(value).lower())
    except ValueError:
        return Severity.WARN
