"""视频帧数量问题 —— 末尾多余动作检测器（质检规范序号 24）。

规范：**允许**视频总帧数大于「标注的最后一个动作的结束帧」（末尾留有静止冗余帧无妨），
但**不允许**在该结束帧之后还存在**实际动作帧**（机械臂仍在操作却未被标注覆盖）。

因此本检测器的判据不是「帧数对不对」，而是「标注末动作结束帧之后是否还有真实动作」：

1. 从 LeRobot ``labels/labels.json``（经摄入层注入 ``metadata["lerobot"]["subtasks"]``）
   取所有子任务 ``end_frame`` 的**最大值**作为「标注末动作结束帧」``last_end``——
   子任务区间偶有乱序 / 重叠（真实样本里存在），取 max 比取末项更稳。
2. 从同 episode 的 puppet parquet 关节读逐帧关节速度（``_joints`` 共享 helper，地面真值），
   只看 ``last_end`` 之后那段（``speed[last_end:]``，即所有产生于 ``last_end`` 之后的帧）。
3. 该段里出现一段足够长的连续「关节在动」（``>= move_speed`` 且连续 ``>= min_action_sec``）
   即判定**末尾有多余动作**（命中，默认 severity=warn）；只是静止停留则不算（规范允许）。

为何用关节而非像素：本项要判的是「**机器人动作**是否越过标注末尾」，关节位移是动作的
地面真值；像素差会被腕部相机自身运动 / 光照 / 噪声干扰而误报。无关节信号时优雅降级为 WARN。

score 语义：越高越好，取 ``1 - tail_action_sec / min_action_sec`` 截断到 [0, 1]，
越接近 0 表示末尾多余动作越长。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from vidinspect_agent.checkers._frames import probe_fps
from vidinspect_agent.checkers._joints import arm_joints_for, per_frame_speed
from vidinspect_agent.checkers.base import BaseChecker
from vidinspect_agent.models import CheckResult, Severity


class TailActionChecker(BaseChecker):
    """末尾多余动作（标注末动作结束帧之后仍有实际动作帧）检测。"""

    name = "tail_action"

    def check(self, path: Path, metadata: dict[str, Any]) -> list[CheckResult]:
        cfg = self.config.get("tail_action", {})
        move_speed = cfg.get("move_speed", 0.01)
        min_action_sec = cfg.get("min_action_sec", 0.3)
        fail_severity = _severity(cfg.get("severity", "warn"))

        subtasks = _subtasks(metadata)
        if subtasks is None:
            return [self._warn("无 LeRobot 子任务标注，跳过末尾多余动作检测", {"error": "no_labels"})]

        last_end = last_labeled_end_frame(subtasks)
        if last_end is None:
            return [self._warn("子任务标注缺少有效 end_frame，跳过检测", {"error": "no_end_frame"})]

        fps = metadata.get("fps") or probe_fps(str(path))
        if not fps or fps <= 0:
            return [self._warn("无法获取帧率，跳过末尾多余动作检测", {"error": "no_fps"})]

        arms = arm_joints_for(path, metadata)
        speed = per_frame_speed(arms) if arms else None
        if speed is None:
            return [
                self._warn(
                    "无关节信号（缺 parquet / pyarrow / 列），无法判定末尾是否有动作",
                    {"error": "no_joint_signal", "last_end_frame": last_end},
                )
            ]

        result = evaluate_tail_action(
            speed, last_end, float(fps), move_speed=move_speed, min_action_sec=min_action_sec
        )
        details = {
            "score": round(result["score"], 4),
            "last_end_frame": result["last_end_frame"],
            "total_frames": result["total_frames"],
            "tail_extra_frames": result["tail_extra_frames"],
            "tail_moving_frames": result["tail_moving_frames"],
            "tail_longest_run": result["tail_longest_run"],
            "tail_action_sec": round(result["tail_action_sec"], 3),
            "min_action_sec": min_action_sec,
            "move_speed": move_speed,
            "fps": round(float(fps), 3),
        }

        if result["detected"]:
            return [
                CheckResult(
                    name="tail_action",
                    severity=fail_severity,
                    message=(
                        f"末尾存在多余动作: 标注末动作结束帧 {last_end} 之后仍有 "
                        f"{result['tail_action_sec']:.1f}s 关节运动（规范24）"
                    ),
                    details=details,
                )
            ]
        return [
            CheckResult(
                name="tail_action",
                severity=Severity.PASS,
                message=(
                    f"末尾无多余动作: 标注末动作结束帧 {last_end} 之后关节静止"
                    f"（额外 {result['tail_extra_frames']} 帧）"
                ),
                details=details,
            )
        ]

    @staticmethod
    def _warn(msg: str, details: dict[str, Any]) -> CheckResult:
        return CheckResult(
            name="tail_action",
            severity=Severity.WARN,
            message=msg,
            details=details,
        )


# --------------------------------------------------------------------------- #
# 纯逻辑（可单测，不触碰 IO）
# --------------------------------------------------------------------------- #
def last_labeled_end_frame(subtasks: Any) -> Optional[int]:
    """所有子任务 ``end_frame`` 的最大值（标注覆盖到的最后一个动作帧）；无有效值 → ``None``。

    取 max 而非末项：真实样本里子任务区间偶有乱序 / 重叠（如某条 start 反而更小），
    用最大结束帧才能稳健表达「标注覆盖的最后一帧」。
    """
    if not isinstance(subtasks, list):
        return None
    ends: list[int] = []
    for s in subtasks:
        if not isinstance(s, dict):
            continue
        ef = s.get("end_frame")
        if isinstance(ef, bool):
            continue
        if isinstance(ef, (int, float)) and float(ef) == float(ef):  # 排除 NaN
            ends.append(int(ef))
    return max(ends) if ends else None


def evaluate_tail_action(
    speed: np.ndarray,
    last_end_frame: int,
    fps: float,
    *,
    move_speed: float = 0.01,
    min_action_sec: float = 0.3,
) -> dict[str, Any]:
    """判定「标注末动作结束帧之后是否还有实际动作」（纯函数）。

    ``speed``：逐帧关节速度（相邻帧关节向量 L2 距离），长度 ``T-1``，``speed[i]`` 描述
    第 ``i → i+1`` 帧的运动。总帧数 ``T = len(speed) + 1``。

    末尾段 ``tail = speed[last_end_frame:]``：所有产生于 ``last_end_frame`` 之后的帧的运动。
    该段中**最长连续**「关节在动（``>= move_speed``）」达到 ``min_action_sec`` 秒
    （即 ``round(min_action_sec * fps)`` 帧，下限 1 帧）即判定命中。
    """
    speed = np.asarray(speed, dtype=np.float64).reshape(-1)
    total_frames = int(speed.size + 1)
    last_end = max(0, int(last_end_frame))
    min_run = max(1, int(round(min_action_sec * fps))) if fps > 0 else 1

    tail = speed[last_end:] if last_end < speed.size else speed[:0]
    tail_extra_frames = int(tail.size)

    moving = tail >= move_speed
    longest_run = _longest_true_run(moving)
    tail_moving_frames = int(moving.sum())
    tail_action_sec = (longest_run / fps) if fps > 0 else 0.0

    detected = bool(longest_run >= min_run and tail_extra_frames > 0)
    ref = max(min_action_sec, 1e-9)
    score = float(np.clip(1.0 - tail_action_sec / ref, 0.0, 1.0))

    return {
        "evaluated": True,
        "detected": detected,
        "score": score,
        "last_end_frame": last_end,
        "total_frames": total_frames,
        "tail_extra_frames": tail_extra_frames,
        "tail_moving_frames": tail_moving_frames,
        "tail_longest_run": int(longest_run),
        "tail_action_sec": tail_action_sec,
        "min_run": min_run,
    }


def _longest_true_run(mask: np.ndarray) -> int:
    """布尔数组里最长连续 True 段的长度。"""
    best = cur = 0
    for v in np.asarray(mask).reshape(-1):
        if v:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return int(best)


def _subtasks(metadata: dict[str, Any]) -> Optional[list[Any]]:
    """从 metadata 取 LeRobot 子任务列表；非 LeRobot 组 → ``None``。"""
    lr = metadata.get("lerobot")
    if not isinstance(lr, dict):
        return None
    subtasks = lr.get("subtasks")
    if not isinstance(subtasks, list):
        return None
    return subtasks


def _severity(value: str) -> Severity:
    try:
        return Severity(str(value).lower())
    except ValueError:
        return Severity.WARN
