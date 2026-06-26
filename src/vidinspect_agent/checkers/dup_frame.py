"""重复帧 / 卡顿检测器（移植自 video_quality_pipeline 的 DupFrameDetector，
源工具 tool/stutter_detect）。

ffmpeg 把视频缩成 64x48 灰度 raw 流，逐帧算 mean|ΔY|；同一组帧差上算两套量：
  - 宽阈值 diff_thr (=0.5) → dup_mask → keep_ratio / mean_gap
  - 严格阈值 strict_thr (=0.05) → dup_ratio_strict（只算"真复制"）

阈值在 20fps 上标定。帧率越高相邻帧天然越相似，故用 norm = fps / fps_ref
归一化阈值，避免把高帧率静态/慢动作正常视频误判成"时间变慢"。

判定（OR 联合两条规则）：
  规则 A 严重 stutter：periodic（周期复制）OR static_like（长段静止复制）
  规则 B 时间变慢：strict 复制比超 fps 归一化阈值，且相似帧段不过长（连续性约束）

score = keep_ratio（越高越好）。stutter_reason ∈ {"", "A", "B", "AB"}。
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from vidinspect_agent.checkers._frames import probe_fps
from vidinspect_agent.checkers.base import BaseChecker
from vidinspect_agent.models import CheckResult, Severity


class DupFrameChecker(BaseChecker):
    """复制帧伪装高帧率导致卡顿检测。"""

    name = "dup_frame"

    def check(self, path: Path, metadata: dict[str, Any]) -> list[CheckResult]:
        cfg = self.config.get("dup_frame", {})
        diff_thr = cfg.get("diff_thr", 0.5)
        dup_strict_thr = cfg.get("strict_thr", 0.05)
        keep_ratio_thr = cfg.get("keep_ratio_thr", 0.20)
        mean_gap_thr = cfg.get("mean_gap_thr", 6.0)
        dup_ratio_strict_thr = cfg.get("ratio_strict_thr", 0.095)
        keep_ratio_lo = cfg.get("keep_ratio_lo", 0.15)
        mid_gap_ratio_thr = cfg.get("mid_gap_ratio_thr", 0.45)
        static_keep_ratio_thr = cfg.get("static_keep_ratio_thr", 0.09)
        fps_ref = cfg.get("fps_ref", 20.0)
        strict_fps_pow = cfg.get("strict_fps_pow", 1.5)
        mean_gap_cap = cfg.get("mean_gap_cap", 16.0)
        downscale = tuple(cfg.get("downscale", (64, 48)))
        timeout = cfg.get("timeout", 60.0)
        prefer_declared_fps = cfg.get("prefer_declared_fps", True)
        fail_severity = _severity(cfg.get("severity", "warn"))

        # fps 用于把 20fps 上标定的阈值做归一化。优先用 info.json 声明帧率（更权威；
        # 全量 29/30/28 因组而异），缺声明时退回 ffprobe 实测 / 探测。
        declared_fps = _declared_fps(metadata) if prefer_declared_fps else None
        fps = declared_fps or metadata.get("fps") or probe_fps(str(path))
        fps_source = (
            "declared" if declared_fps
            else ("measured" if metadata.get("fps") else "probed")
        )
        norm = (fps / fps_ref) if (fps and fps > 0 and fps_ref > 0) else 1.0

        W, H = downscale
        chunk = W * H
        t0 = time.time()
        proc = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-i", str(path),
             "-vf", f"scale={W}:{H},format=gray", "-f", "rawvideo", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10 ** 7,
        )
        prev = None
        n = 0
        dup_mask: list[int] = []
        dup_strict_count = 0
        try:
            while True:
                if time.time() - t0 > timeout:
                    proc.kill()
                    proc.wait()
                    return [self._error(f"timeout>{timeout}s")]
                buf = proc.stdout.read(chunk)
                if len(buf) < chunk:
                    break
                g = np.frombuffer(buf, dtype=np.uint8).reshape(H, W)
                if prev is not None:
                    d = float(np.mean(np.abs(g.astype(np.int16) - prev.astype(np.int16))))
                    dup_mask.append(1 if d < diff_thr else 0)
                    if d < dup_strict_thr:
                        dup_strict_count += 1
                prev = g
                n += 1
            proc.wait(timeout=5)
        except Exception as e:  # noqa: BLE001
            try:
                proc.kill()
            except Exception:
                pass
            return [self._error(f"{type(e).__name__}: {e}")]

        if n < 2:
            return [self._error(f"too_few_frames={n}")]

        dup = np.asarray(dup_mask, dtype=np.int8)
        total_pairs = len(dup)
        dup_count = int(dup.sum())
        keep_ratio = 1.0 - dup_count / total_pairs
        dup_ratio_strict = dup_strict_count / total_pairs

        runs, cur = [], 0
        for v in dup:
            if v:
                cur += 1
            elif cur:
                runs.append(cur)
                cur = 0
        if cur:
            runs.append(cur)
        if runs:
            runs_np = np.asarray(runs)
            max_gap = int(runs_np.max())
            mean_gap = float(runs_np.mean())
            gap_p95 = int(np.percentile(runs_np, 95))
            mid_gap_ratio = float(((runs_np >= 4) & (runs_np <= 14)).mean())
            short_gap_ratio = float((runs_np <= 3).mean())
            long_gap_ratio = float((runs_np > 30).mean())
        else:
            max_gap, mean_gap, gap_p95 = 0, 0.0, 0
            mid_gap_ratio = short_gap_ratio = long_gap_ratio = 0.0

        mean_gap_thr_eff = mean_gap_thr * norm
        static_mean_gap_eff = 10.0 * norm
        static_max_gap_eff = 20.0 * norm
        strict_thr_eff = dup_ratio_strict_thr * (norm ** strict_fps_pow)
        mean_gap_cap_eff = mean_gap_cap * norm

        base_a = (keep_ratio < keep_ratio_thr) and (mean_gap >= mean_gap_thr_eff)
        periodic_stutter = base_a and (mid_gap_ratio >= mid_gap_ratio_thr)
        static_like = (
            keep_ratio < static_keep_ratio_thr
            and mean_gap >= static_mean_gap_eff
            and max_gap >= static_max_gap_eff
        )
        rule_a = periodic_stutter or static_like
        rule_b = (
            (dup_ratio_strict > strict_thr_eff)
            and (keep_ratio >= keep_ratio_lo)
            and (mean_gap < mean_gap_cap_eff)
        )
        reason = ("A" if rule_a else "") + ("B" if rule_b else "")
        problem = bool(rule_a or rule_b)

        details = {
            "stutter_reason": reason,
            "score": round(keep_ratio, 4),
            "keep_ratio": round(keep_ratio, 4),
            "dup_ratio_strict": round(dup_ratio_strict, 4),
            "mean_gap": round(mean_gap, 3),
            "max_gap": max_gap,
            "gap_p95": gap_p95,
            "mid_gap_ratio": round(mid_gap_ratio, 4),
            "short_gap_ratio": round(short_gap_ratio, 4),
            "long_gap_ratio": round(long_gap_ratio, 4),
            "total_frames": n,
            "fps": round(fps, 3) if fps else None,
            "fps_source": fps_source,
            "fps_norm": round(norm, 3),
        }
        if problem:
            return [
                CheckResult(
                    name="dup_frame",
                    severity=fail_severity,
                    message=(
                        f"疑似复制帧/卡顿 (reason={reason}): "
                        f"keep_ratio={keep_ratio:.3f} strict={dup_ratio_strict:.3f}"
                    ),
                    details=details,
                )
            ]
        return [
            CheckResult(
                name="dup_frame",
                severity=Severity.PASS,
                message=f"无明显复制帧: keep_ratio={keep_ratio:.3f}",
                details=details,
            )
        ]

    @staticmethod
    def _error(msg: str) -> CheckResult:
        return CheckResult(
            name="dup_frame",
            severity=Severity.WARN,
            message=f"复制帧检测未完成: {msg}",
            details={"error": msg},
        )


def _declared_fps(metadata: dict[str, Any]) -> float | None:
    """从 §1 摄入层注入的 LeRobot 上下文取 info.json 声明帧率；缺省返回 ``None``。"""
    lr = metadata.get("lerobot")
    if not isinstance(lr, dict):
        return None
    declared = lr.get("declared_video")
    fps = declared.get("fps") if isinstance(declared, dict) else None
    if fps is None:
        fps = lr.get("declared_fps")
    try:
        fps = float(fps) if fps is not None else None
    except (TypeError, ValueError):
        return None
    return fps if (fps and fps > 0) else None


def _severity(value: str) -> Severity:
    try:
        return Severity(str(value).lower())
    except ValueError:
        return Severity.WARN
