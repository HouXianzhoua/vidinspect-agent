"""静态 / 无效操作检测器（移植自 video_quality_pipeline 的 StaticDetector，
源工具 robomind2_filter/static_arm）。

两个后端，config['static']['backend'] 选择：
  - 'lite'（纯 CPU）：相邻采样帧灰度 L1 差，sigmoid 软化为 motion_score。
        score = 1 / (1 + exp(-5*(peak_diff/1.5 - 1)))
        problem（静止）= score < thr            默认 thr=0.30
  - 'raft'（需 GPU + RAFT 源码/权重）：RAFT 稠密光流的 active_ratio。
        rel_i  = (top5% 光流幅值均值) / min(H,W)
        score  = active_ratio = mean(rel_i >= 0.012)
        problem（静止）= score < raft_thr        默认 raft_thr=0.10（已标定）

score 语义：越高越好（1.0=有正常运动，0.0=全程静止）。RAFT 后端在 torch/decord/
权重缺失或无 GPU 时不会让流水线崩溃，而是降级为 warn 结果。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from vidinspect_agent.checkers._frames import read_frames_gray
from vidinspect_agent.checkers.base import BaseChecker
from vidinspect_agent.models import CheckResult, Severity

THRES_GRAY = 1.5          # lite：灰度差阈值
THRES_REL = 0.012         # raft：相对运动阈值
DEFAULT_THR_LITE = 0.30
DEFAULT_THR_RAFT = 0.10


class StaticChecker(BaseChecker):
    """机械臂静止 / 无效操作检测（lite CPU 后端 + 可选 raft GPU 后端）。"""

    name = "static"

    def check(self, path: Path, metadata: dict[str, Any]) -> list[CheckResult]:
        cfg = self.config.get("static", {})
        backend = cfg.get("backend", "lite")
        if backend == "raft":
            return self._detect_raft(path, cfg)
        return self._detect_lite(path, cfg)

    # ---- lite (CPU) ----
    def _detect_lite(self, path: Path, cfg: dict) -> list[CheckResult]:
        thr = cfg.get("thr", DEFAULT_THR_LITE)
        n_frames = cfg.get("n_frames", 32)
        max_h = cfg.get("max_h", 240)
        fail_severity = _severity(cfg.get("severity", "warn"))

        gray = read_frames_gray(str(path), n_frames, max_h)
        if gray is None or gray.shape[0] < 2:
            return [
                CheckResult(
                    name="static",
                    severity=Severity.WARN,
                    message="无法采样帧用于静态检测",
                    details={"backend": "lite", "error": "read_failed"},
                )
            ]

        diffs = np.abs(gray[1:] - gray[:-1]).mean(axis=(1, 2))
        peak_diff = float(diffs.max())
        mean_diff = float(diffs.mean())
        x = peak_diff / THRES_GRAY
        score = float(1.0 / (1.0 + np.exp(-5.0 * (x - 1.0))))
        problem = score < thr

        details = {
            "score": round(score, 4),
            "peak_diff": round(peak_diff, 4),
            "mean_diff": round(mean_diff, 4),
            "thr": thr,
            "backend": "lite",
        }
        if problem:
            return [
                CheckResult(
                    name="static",
                    severity=fail_severity,
                    message=f"疑似静止/无效操作: motion_score={score:.3f} (<{thr})",
                    details=details,
                )
            ]
        return [
            CheckResult(
                name="static",
                severity=Severity.PASS,
                message=f"运动正常: motion_score={score:.3f}",
                details=details,
            )
        ]

    # ---- raft (GPU) ----
    def _detect_raft(self, path: Path, cfg: dict) -> list[CheckResult]:
        fail_severity = _severity(cfg.get("severity", "warn"))
        thr = cfg.get("raft_thr", DEFAULT_THR_RAFT)
        n_frames = cfg.get("raft_n_frames", cfg.get("n_frames", 40))
        max_h = cfg.get("max_h", 240)
        try:
            return self._run_raft(path, cfg, thr, n_frames, max_h, fail_severity)
        except Exception as exc:  # noqa: BLE001 - 降级为 warn，避免拖垮流水线
            return [
                CheckResult(
                    name="static",
                    severity=Severity.WARN,
                    message=f"RAFT 后端不可用，已跳过静态检测: {type(exc).__name__}",
                    details={"backend": "raft", "error": f"{type(exc).__name__}: {exc}"},
                )
            ]

    def _run_raft(self, path, cfg, thr, n_frames, max_h, fail_severity):
        import torch

        from vidinspect_agent.checkers import _raft
        from vidinspect_agent.checkers._frames import read_frames_rgb_tensor

        device_str = cfg.get("device") or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        device = torch.device(device_str)
        raft_info = _raft.load_raft(cfg, device)

        frames = read_frames_rgb_tensor(str(path), n_frames, max_h)
        if frames is None:
            return [
                CheckResult(
                    name="static",
                    severity=Severity.WARN,
                    message="无法采样帧用于 RAFT 静态检测",
                    details={"backend": "raft", "error": "read_failed"},
                )
            ]

        iters = cfg.get("raft_iters", 12)
        batch_size = cfg.get("raft_batch_size", 40)
        with torch.inference_mode():
            flows = _raft.compute_raft_flows(
                frames, raft_info["model"], device, iters, batch_size
            )
        if not flows:
            details = {"backend": "raft", "active_ratio": 0.0, "thr": thr}
            return [
                CheckResult(
                    name="static",
                    severity=fail_severity,
                    message="疑似静止/无效操作: active_ratio=0.000",
                    details=details,
                )
            ]

        scale = float(min(frames.shape[2], frames.shape[3]))
        rel = []
        for flow in flows:
            f = flow.numpy()
            rad = np.sqrt(f[0] ** 2 + f[1] ** 2)
            k = max(int(rad.size * 0.05), 1)
            rel.append(
                float(np.mean(np.partition(rad.flatten(), -k)[-k:])) / max(scale, 1.0)
            )
        rel = np.asarray(rel, dtype=np.float32)
        active_ratio = float((rel >= THRES_REL).mean())
        problem = active_ratio < thr
        details = {
            "score": round(active_ratio, 4),
            "active_ratio": round(active_ratio, 4),
            "peak_motion": round(float(rel.max()), 5),
            "thr": thr,
            "backend": "raft",
            "device": str(device),
        }
        if problem:
            return [
                CheckResult(
                    name="static",
                    severity=fail_severity,
                    message=f"疑似静止/无效操作: active_ratio={active_ratio:.3f} (<{thr})",
                    details=details,
                )
            ]
        return [
            CheckResult(
                name="static",
                severity=Severity.PASS,
                message=f"运动正常: active_ratio={active_ratio:.3f}",
                details=details,
            )
        ]


def _severity(value: str) -> Severity:
    try:
        return Severity(str(value).lower())
    except ValueError:
        return Severity.WARN
