"""跳帧 / 瞬移检测器（移植自 video_quality_pipeline 的 JumpDetector，
源工具 jump_frame 的 local_max_ratio）。

逐帧连续读取（缩到 max_h=120 灰度），算相邻帧 L1 差序列 diff_i，用 ±W 帧局部
均值（排除自身）做分母得 local_ratio_i，取最大值：

    local_ratio_max = max_i  diff_i / mean(diff_{i-W..i+W} \\ {i})

判定（全部满足才算 jump）：
  ratio_hit       local_ratio_max >= threshold[robot]
  magnitude_ok    峰值绝对帧差 peak_abs >= jump_peak_abs_min 或整段几乎冻结
  universal_ok    非持续运动(high_run<sustained_run) 且 非边界初始化伪影
  isolation_ok    静态相机条件下要求"单帧孤立尖峰"

机器人阈值来自 jump_frame filter_meta.json，可用 config['jump']['thresholds'] 覆盖；
无机器人标签时用 __default__（4.0）。score = 1/(1+log(max(local_ratio,1)))，越高越好。
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from vidinspect_agent.checkers.base import BaseChecker
from vidinspect_agent.models import CheckResult, Severity

# 各机器人跳帧阈值（来自 jump_frame filter_meta.json 的 threshold_table）
DEFAULT_JUMP_THRESHOLDS = {
    "tienkung_station_dualArm-gripper": 10.0,
    "tienkung_pro2_dualArm-gripper": 7.5,
    "tiangong_dualArm": 6.0,
    "tienkung_pro2_dualArm": 8.0,
    "tienkung_sim_dualArm": 7.5,
    "tienyi_dualArm": 8.0,
    "tiangong_dexHand": 10.0,
    "tienyi_mobile_dualArm": 5.0,
    "__default__": 4.0,
}

MAX_H = 120
WINDOW = 30


class JumpChecker(BaseChecker):
    """跳帧 / 瞬移检测。"""

    name = "jump"

    def check(self, path: Path, metadata: dict[str, Any]) -> list[CheckResult]:
        import cv2

        cfg = self.config.get("jump", {})
        thr_table = cfg.get("thresholds", DEFAULT_JUMP_THRESHOLDS)
        robot = metadata.get("robot")
        thr = thr_table.get(robot, thr_table.get("__default__", 4.0))
        peak_abs_min = cfg.get("peak_abs_min", 3.0)
        frozen_mean = cfg.get("frozen_mean", 0.05)
        sustained_run = cfg.get("sustained_run", 4)
        edge_frames = cfg.get("edge_frames", 1)
        edge_bg_max = cfg.get("edge_bg_max", 0.6)
        isolation_mode = cfg.get("isolation_mode", "auto")
        static_camera = cfg.get("static_camera", None)
        isolation_min = cfg.get("isolation_min", 2.5)
        high_run_max = cfg.get("high_run_max", 1)
        static_bg_max = cfg.get("static_bg_max", 0.6)
        fail_severity = _severity(cfg.get("severity", "warn"))

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return [self._error("open_failed")]
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        prev_gray = None
        diffs: list[float] = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            h, w = f.shape[:2]
            if h > MAX_H:
                f = cv2.resize(f, (max(1, int(w * MAX_H / h)), MAX_H),
                               interpolation=cv2.INTER_LINEAR)
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
            if prev_gray is not None:
                diffs.append(float(np.abs(gray - prev_gray).mean()))
            prev_gray = gray
        cap.release()

        n = len(diffs)
        if n < 2:
            return [
                CheckResult(
                    name="jump",
                    severity=Severity.PASS,
                    message="帧数过少，跳过跳帧检测",
                    details={"score": 1.0, "local_ratio_max": 1.0,
                             "total_frames": total_frames, "thr": thr, "robot": robot},
                )
            ]

        diffs_arr = np.asarray(diffs, dtype=np.float64)
        global_mean = float(diffs_arr.mean()) + 1e-8
        global_max_ratio = float(diffs_arr.max() / global_mean)

        W = WINDOW
        if n < 2 * W + 3:
            local_max = global_max_ratio
            local_idx = int(np.argmax(diffs_arr))
        else:
            cs = np.cumsum(np.concatenate([[0.0], diffs_arr]))
            local_max, local_idx = 1.0, 0
            for i in range(n):
                lo, hi = max(0, i - W), min(n, i + W + 1)
                local_sum = cs[hi] - cs[lo] - diffs_arr[i]
                local_cnt = (hi - lo) - 1
                if local_cnt < 1 or local_sum < 1e-6:
                    continue
                r = diffs_arr[i] / (local_sum / local_cnt)
                if r > local_max:
                    local_max, local_idx = r, i

        peak_abs = float(diffs_arr[local_idx])
        lo = max(0, local_idx - 5)
        hi = min(n, local_idx + 6)
        neigh = np.concatenate([diffs_arr[lo:local_idx], diffs_arr[local_idx + 1:hi]])
        peak_neighbor_max = float(neigh.max()) if neigh.size else 0.0
        peak_isolation = peak_abs / max(peak_neighbor_max, 1e-6)
        high_thr = 0.5 * peak_abs
        run_lo = run_hi = local_idx
        while run_lo - 1 >= 0 and diffs_arr[run_lo - 1] >= high_thr:
            run_lo -= 1
        while run_hi + 1 < n and diffs_arr[run_hi + 1] >= high_thr:
            run_hi += 1
        peak_high_run = int(run_hi - run_lo + 1)
        wlo, whi = max(0, local_idx - W), min(n, local_idx + W + 1)
        bg = np.concatenate([diffs_arr[wlo:run_lo], diffs_arr[run_hi + 1:whi]])
        bg_med = float(np.median(bg)) if bg.size else 0.0

        score = 1.0 / (1.0 + math.log(max(local_max, 1.0)))
        ratio_hit = local_max >= thr
        magnitude_ok = (peak_abs >= peak_abs_min) or (global_mean <= frozen_mean)
        sustained_motion = peak_high_run >= sustained_run
        at_edge = (local_idx <= edge_frames) or (local_idx >= n - 1 - edge_frames)
        edge_artifact = at_edge and (peak_high_run >= 2) and (bg_med <= edge_bg_max)
        universal_ok = (not sustained_motion) and (not edge_artifact)
        isolated_spike = (peak_isolation >= isolation_min) and (peak_high_run <= high_run_max)
        is_agibot = isinstance(robot, str) and robot.startswith("agibot")
        if isolation_mode == "off":
            isolation_ok = True
        elif isolation_mode == "all":
            isolation_ok = isolated_spike
        elif isolation_mode == "agibot":
            isolation_ok = (not is_agibot) or isolated_spike
        else:  # auto
            apply_iso = static_camera if static_camera is not None else is_agibot
            isolation_ok = (not apply_iso) or isolated_spike

        problem = bool(ratio_hit and magnitude_ok and universal_ok and isolation_ok)
        details = {
            "score": round(score, 4),
            "local_ratio_max": round(float(local_max), 2),
            "local_ratio_idx": int(local_idx),
            "peak_abs": round(peak_abs, 3),
            "peak_isolation": round(peak_isolation, 3),
            "peak_high_run": peak_high_run,
            "bg_med": round(bg_med, 4),
            "global_mean": round(global_mean, 4),
            "ratio_hit": bool(ratio_hit),
            "magnitude_ok": bool(magnitude_ok),
            "universal_ok": bool(universal_ok),
            "isolation_ok": bool(isolation_ok),
            "total_frames": total_frames,
            "thr": thr,
            "robot": robot,
        }
        if problem:
            return [
                CheckResult(
                    name="jump",
                    severity=fail_severity,
                    message=(
                        f"疑似跳帧/瞬移: local_ratio_max={local_max:.2f} "
                        f"(>={thr}) peak_abs={peak_abs:.2f}"
                    ),
                    details=details,
                )
            ]
        return [
            CheckResult(
                name="jump",
                severity=Severity.PASS,
                message=f"无明显跳帧: local_ratio_max={local_max:.2f}",
                details=details,
            )
        ]

    @staticmethod
    def _error(msg: str) -> CheckResult:
        return CheckResult(
            name="jump",
            severity=Severity.WARN,
            message=f"跳帧检测未完成: {msg}",
            details={"error": msg},
        )


def _severity(value: str) -> Severity:
    try:
        return Severity(str(value).lower())
    except ValueError:
        return Severity.WARN
