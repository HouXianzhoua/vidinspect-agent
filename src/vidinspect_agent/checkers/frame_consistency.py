"""画面与关节一致性检测器（质检规范序号 18：画面保持一致）。

规范：不要出现「主画面机械臂在动、而对应夹爪（腕部）相机画面不动」的情况。

腕部相机（``camera_left`` / ``camera_right``）刚性装在左 / 右夹爪上，随对应机械臂一起运动
（见 ``docs/dataset_inputs.md §2``）：某只臂一旦运动，其对应腕部相机画面就应大幅变化。
若该臂关节在动（parquet 地面真值）而腕部相机画面却几乎不动（采集冻结 / 串流延迟 / 掉帧 /
贴错相机），即为「画面与关节不一致」。

**为什么不用 AI**：判定拆成两个都不需要语义识别的量——

- 「该侧臂动没动」：直接取 parquet 的 ``puppet.arm_{left,right}_position_align`` 逐帧关节速度，
  是运动的地面真值（不必从俯视画面里分割左右臂，那才需要 AI）。
- 「腕部相机画面动没动」：相邻帧 ``mean|ΔY|`` 帧差。

判据（相对底噪，抗下采样尺度漂移）：以「该臂静止帧」估计该相机自身的运动底噪
``cam_floor``；当臂在动、而相机帧差 ≤ ``max(cam_floor*floor_k, abs_static_floor)``
（远低于"随臂运动本应有的量级"）的连续段超过 ``min_inconsistent_sec`` 即命中。

设计特性：

- 「某只臂确实没动」天然不误报——该侧没有"臂在动"的帧则不评估（PASS）。
- 完全冻结是本判据的子集（``cam_motion≈0`` 必 ≤ 阈值），是 ``freeze`` 的超集；
  ``freeze`` 仍独立负责「与关节无关的单段长冻结（规范5）」。
- 非腕部机位（``camera_top`` 俯视固定）/ 非 LeRobot 视频 → 不适用，PASS 跳过（不产生噪声）。
- 缺 parquet 关节 / 缺 pyarrow / 解码失败 / 无 fps → WARN 优雅降级，不阻塞流水线。

score 语义：越高越好，取 ``1 - max_inconsistent_sec/min_inconsistent_sec`` 截断到 [0, 1]。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from vidinspect_agent.checkers._frames import probe_fps, stream_gray_diffs
from vidinspect_agent.checkers._joints import (
    arm_joints_for,
    camera_side,
    is_wrist_camera,
    per_frame_speed,
)
from vidinspect_agent.checkers.base import BaseChecker
from vidinspect_agent.models import CheckResult, Severity

DEFAULT_JOINT_MOVE_SPEED = 0.01    # rad/帧：判定该侧臂单帧"在动"的关节速度阈值
DEFAULT_FLOOR_K = 3.0              # 画面帧差需超过自身底噪的倍数，否则算"没动"
DEFAULT_ABS_STATIC_FLOOR = 0.1    # 帧差(0-255)绝对下限（近似"同一帧"，兜底压缩噪声）
DEFAULT_MIN_INCONSISTENT_SEC = 1.0
DEFAULT_MIN_STILL_FRAMES = 5      # 估底噪所需的最少"臂静止"帧数，不足则退回 q10


class FrameConsistencyChecker(BaseChecker):
    """主画面机械臂在动、对应腕部相机画面不动 → 画面/关节不一致（规范18）。"""

    name = "frame_consistency"

    def check(self, path: Path, metadata: dict[str, Any]) -> list[CheckResult]:
        cfg = self.config.get("frame_consistency", {})
        move_speed = cfg.get("joint_move_speed", DEFAULT_JOINT_MOVE_SPEED)
        floor_k = cfg.get("floor_k", DEFAULT_FLOOR_K)
        abs_static_floor = cfg.get("abs_static_floor", DEFAULT_ABS_STATIC_FLOOR)
        min_inconsistent_sec = cfg.get("min_inconsistent_sec", DEFAULT_MIN_INCONSISTENT_SEC)
        min_still_frames = cfg.get("min_still_frames", DEFAULT_MIN_STILL_FRAMES)
        downscale = tuple(cfg.get("downscale", (64, 48)))
        timeout = cfg.get("timeout", 60.0)
        fail_severity = _severity(cfg.get("severity", "warn"))

        lerobot_meta = metadata.get("lerobot")
        camera_key = lerobot_meta.get("camera_key") if isinstance(lerobot_meta, dict) else None
        side = camera_side(camera_key)

        # 非腕部机位（俯视固定 / 无机位信息 / 非 LeRobot 视频）→ 规范18 不适用，PASS 跳过。
        if not is_wrist_camera(camera_key):
            reason = "fixed_or_top_camera" if side == "top" else "not_wrist_camera"
            return [self._skip(
                "非腕部相机（规范18只校验随臂运动的腕部相机），跳过",
                {"camera_side": side, "camera_key": camera_key, "skipped": reason},
            )]

        fps = metadata.get("fps") or probe_fps(str(path))
        if not fps or fps <= 0:
            return [self._warn("无法获取帧率，跳过画面一致性检测", {"error": "no_fps"})]

        arms = arm_joints_for(path, metadata)
        speed = per_frame_speed(arms, side=side) if arms else None
        if speed is None:
            return [self._warn(
                "无该侧臂关节数据（缺 parquet / pyarrow），无法校验画面一致性(规范18)",
                {"error": "no_joints", "camera_side": side},
            )]

        diffs, n, err = stream_gray_diffs(str(path), downscale, timeout)
        if diffs is None:
            return [self._warn(f"画面一致性检测未完成: {err}", {"error": err})]

        ev = evaluate_frame_consistency(
            speed,
            diffs,
            float(fps),
            joint_move_speed=move_speed,
            floor_k=floor_k,
            abs_static_floor=abs_static_floor,
            min_inconsistent_sec=min_inconsistent_sec,
            min_still_frames=min_still_frames,
        )

        details = {
            "score": ev["score"],
            "camera_side": side,
            "max_inconsistent_sec": ev["max_inconsistent_sec"],
            "max_inconsistent_frames": ev["max_inconsistent_frames"],
            "inconsistent_start_sec": (
                round(ev["inconsistent_start"] / fps, 3) if ev["max_inconsistent_frames"] else None
            ),
            "cam_floor": ev["cam_floor"],
            "floor_estimated": ev["floor_estimated"],
            "static_thr": ev["static_thr"],
            "arm_moving_ratio": ev["arm_moving_ratio"],
            "min_inconsistent_sec": min_inconsistent_sec,
            "joint_move_speed": move_speed,
            "floor_k": floor_k,
            "reason": ev["reason"],
            "fps": round(float(fps), 3),
            "cam_frames": n,
            "joint_frames": int(speed.size + 1),
        }

        if ev["reason"] == "arm_idle":
            return [CheckResult(
                name=self.name,
                severity=Severity.PASS,
                message="该侧臂全程几乎未动作，无需校验画面一致性(规范18)",
                details=details,
            )]
        if not ev["evaluated"]:
            return [self._warn("画面一致性无法评估（有效帧不足）", details)]

        if ev["detected"]:
            return [CheckResult(
                name=self.name,
                severity=fail_severity,
                message=(
                    f"疑似画面与关节不一致(规范18): {side} 臂在动但腕部相机画面静止 "
                    f"{ev['max_inconsistent_sec']:.1f}s (上限 {min_inconsistent_sec:.1f}s)"
                ),
                details=details,
            )]
        return [CheckResult(
            name=self.name,
            severity=Severity.PASS,
            message=f"画面与关节一致: 最长不一致 {ev['max_inconsistent_sec']:.1f}s",
            details=details,
        )]

    def _warn(self, msg: str, details: dict[str, Any]) -> CheckResult:
        return CheckResult(name=self.name, severity=Severity.WARN, message=msg, details=details)

    def _skip(self, msg: str, details: dict[str, Any]) -> CheckResult:
        return CheckResult(name=self.name, severity=Severity.PASS, message=msg, details=details)


def evaluate_frame_consistency(
    joint_speed: np.ndarray,
    cam_motion: np.ndarray,
    fps: float,
    *,
    joint_move_speed: float = DEFAULT_JOINT_MOVE_SPEED,
    floor_k: float = DEFAULT_FLOOR_K,
    abs_static_floor: float = DEFAULT_ABS_STATIC_FLOOR,
    min_inconsistent_sec: float = DEFAULT_MIN_INCONSISTENT_SEC,
    min_still_frames: int = DEFAULT_MIN_STILL_FRAMES,
) -> dict[str, Any]:
    """纯函数：由该侧臂逐帧关节速度 + 腕部相机逐帧帧差判画面是否与关节一致（便于单测）。

    入参：
      - ``joint_speed``：该侧臂相邻帧关节速度（rad/帧），长度 ``Tj``（见 ``_joints.per_frame_speed``）。
      - ``cam_motion``：腕部相机相邻帧 ``mean|ΔY|``（0–255），长度 ``Tc``（见 ``stream_gray_diffs``）。
      - ``fps``：视频帧率，用于把"不一致帧段"换算成秒。

    以相机时间轴为主轴，把关节速度按比例最近邻重采样对齐到 ``Tc``（容忍 parquet 行数与解码
    帧数的轻微长度差）。判定：

      arm_moving[i]  = joint_speed_aligned[i] >= joint_move_speed
      cam_floor      = median(cam_motion[ 臂静止帧 ])          # 该相机自身底噪
      static_thr     = max(cam_floor * floor_k, abs_static_floor)
      bad[i]         = arm_moving[i] AND cam_motion[i] <= static_thr
      detected       = 最长连续 bad 段 / fps > min_inconsistent_sec

    返回 dict（``reason`` ∈ {``ok``, ``arm_idle``, ``insufficient``}）。
    """
    base = {
        "detected": False,
        "evaluated": False,
        "reason": "insufficient",
        "max_inconsistent_sec": 0.0,
        "max_inconsistent_frames": 0,
        "inconsistent_start": 0,
        "cam_floor": 0.0,
        "floor_estimated": False,
        "static_thr": 0.0,
        "arm_moving_ratio": 0.0,
        "score": 1.0,
        "n_frames": 0,
    }
    if (
        joint_speed is None
        or cam_motion is None
        or joint_speed.size == 0
        or cam_motion.size == 0
        or fps <= 0
    ):
        return base

    n = int(cam_motion.size)
    s = _resample_nearest(np.asarray(joint_speed, dtype=np.float64), n)
    m = np.asarray(cam_motion, dtype=np.float64)

    arm_moving = s >= joint_move_speed
    moving_ratio = float(arm_moving.mean())
    if not arm_moving.any():
        return {**base, "evaluated": True, "reason": "arm_idle",
                "arm_moving_ratio": 0.0, "n_frames": n}

    # 底噪需由"该臂静止帧"估计；静止帧不足时无法可靠估相对底噪，退回纯绝对阈值
    # （只抓硬冻结，不因"整段都在动"而把正常大运动误判为静止）。
    still = ~arm_moving
    if int(still.sum()) >= min_still_frames:
        cam_floor = float(np.median(m[still]))
        floor_estimated = True
        static_thr = max(cam_floor * floor_k, abs_static_floor)
    else:
        cam_floor = 0.0
        floor_estimated = False
        static_thr = abs_static_floor

    bad = arm_moving & (m <= static_thr)
    max_run, run_start = _longest_run(bad)
    max_inconsistent_sec = max_run / fps
    detected = max_inconsistent_sec > min_inconsistent_sec
    score = float(np.clip(1.0 - max_inconsistent_sec / min_inconsistent_sec, 0.0, 1.0)) \
        if min_inconsistent_sec > 0 else (0.0 if detected else 1.0)

    return {
        "detected": bool(detected),
        "evaluated": True,
        "reason": "ok",
        "max_inconsistent_sec": round(max_inconsistent_sec, 3),
        "max_inconsistent_frames": int(max_run),
        "inconsistent_start": int(run_start),
        "cam_floor": round(cam_floor, 4),
        "floor_estimated": floor_estimated,
        "static_thr": round(static_thr, 4),
        "arm_moving_ratio": round(moving_ratio, 4),
        "score": round(score, 4),
        "n_frames": n,
    }


def _resample_nearest(series: np.ndarray, target_len: int) -> np.ndarray:
    """把 ``series`` 按比例最近邻重采样到 ``target_len``（索引 i ↔ 比例 i/(target_len-1)）。"""
    n = int(series.size)
    if target_len <= 0 or n == 0:
        return np.empty(0, dtype=np.float64)
    if n == target_len:
        return series
    idx = np.round(np.linspace(0, n - 1, target_len)).astype(int)
    return series[idx]


def _longest_run(mask: np.ndarray) -> tuple[int, int]:
    """返回最长连续 True 段的 (长度, 起始索引)。"""
    best_len = best_start = 0
    cur_len = cur_start = 0
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
