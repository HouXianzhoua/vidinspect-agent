"""§3 关节交叉验证共享 helper（docs/detector_dataset_impact.md §3）。

``endpoint_static`` / ``freeze`` / ``jump`` 的像素核心逻辑保留，叠加 LeRobot parquet 的
**逐帧关节运动量**作交叉验证：关节是运动的地面真值，可纠正 64×48 像素自适应阈值的脆弱性，
并覆盖规范 18（主画面机械臂在动、腕部相机画面冻结 → 画面/关节不一致）。

读取复用 §1/§2 已有的 parquet 解析（``static._read_arm_joints`` + ``lerobot`` 定位），
不重复造轮子。任何一步不可用（无 parquet / 缺 pyarrow / 列缺失 / 读失败）一律返回
``None`` / 空，调用方据此退回纯像素行为。

纯函数（``per_frame_speed`` 之后的判定）便于单测，不触碰 IO。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

# 腕部随动相机（随对应臂运动）↔ 固定俯视机位。camera_left/right 装在左/右夹爪上。
_WRIST_SIDES = ("left", "right")


def camera_side(camera_key: str | None) -> Optional[str]:
    """从 camera key 解析机位 side：``left`` / ``right``（腕部随动）/ ``top``（俯视固定）。"""
    if not camera_key:
        return None
    key = str(camera_key).lower()
    for token in ("left", "right", "top"):
        if key.endswith(token):
            return token
    return None


def is_wrist_camera(camera_key: str | None) -> bool:
    """该机位是否为腕部随动相机（其画面应随对应臂运动）。"""
    return camera_side(camera_key) in _WRIST_SIDES


def arm_joints_for(path: str | Path, metadata: dict[str, Any]) -> dict[str, np.ndarray]:
    """读取 puppet 各侧逐帧臂关节 ``{side: [T, D]}``；不可用 → ``{}``。

    复用 ``static`` 关节后端的稳健读取（parquet 指针优先，否则由视频路径自定位）。
    """
    try:
        from vidinspect_agent.checkers.static import (
            _locate_group_parquet,
            _parquet_pointer,
            _read_arm_joints,
        )
    except Exception:  # noqa: BLE001 - §2 后端不可用则降级
        return {}
    try:
        parquet = _parquet_pointer(metadata) or _locate_group_parquet(Path(path))
        if parquet is None:
            return {}
        return _read_arm_joints(parquet) or {}
    except Exception:  # noqa: BLE001 - 任意读取异常降级为无关节
        return {}


def per_frame_speed(
    arms: dict[str, np.ndarray], side: Optional[str] = None
) -> Optional[np.ndarray]:
    """某侧（或整体）逐帧关节速度（相邻帧关节向量 L2 距离），长度 ``T-1``；不可用 → ``None``。

    - ``side`` ∈ {``left``, ``right``}：仅该侧臂（腕部相机随其运动）。
    - ``side`` 为 ``None`` / ``top`` / 缺该侧：所有可用臂在帧轴对齐到最短后拼接（整体参照）。
    """
    if not arms:
        return None
    mats: list[np.ndarray]
    if side in _WRIST_SIDES and isinstance(arms.get(side), np.ndarray):
        mats = [arms[side]]
    else:
        mats = list(arms.values())
    mats = [m for m in mats if isinstance(m, np.ndarray) and m.ndim == 2 and m.shape[0] >= 2]
    if not mats:
        return None
    n = min(m.shape[0] for m in mats)
    joints = np.concatenate([m[:n] for m in mats], axis=1).astype(np.float64)
    return np.linalg.norm(np.diff(joints, axis=0), axis=1)


# --------------------------------------------------------------------------- #
# 纯判定逻辑（可单测）
# --------------------------------------------------------------------------- #
def leading_static_frames(speed: np.ndarray, move_speed: float) -> int:
    """开头连续「关节几乎不动」（speed < move_speed）的帧数。"""
    moving = speed >= move_speed
    if not moving.any():
        return int(speed.size)
    return int(np.argmax(moving))


def trailing_static_frames(speed: np.ndarray, move_speed: float) -> int:
    """结尾连续「关节几乎不动」的帧数。"""
    moving = speed >= move_speed
    if not moving.any():
        return int(speed.size)
    return int(np.argmax(moving[::-1]))


def joint_endpoint_static_seconds(
    speed: np.ndarray, fps: float, move_speed: float
) -> tuple[float, float]:
    """由逐帧关节速度算首 / 尾「关节静止」时长（秒）。``fps<=0`` → (0, 0)。"""
    if speed is None or speed.size == 0 or fps <= 0:
        return 0.0, 0.0
    lead = leading_static_frames(speed, move_speed)
    trail = trailing_static_frames(speed, move_speed)
    return lead / fps, trail / fps


def joint_moving_in_fraction(
    speed: np.ndarray, frac_start: float, frac_end: float, move_speed: float
) -> Optional[bool]:
    """在 [frac_start, frac_end] 这段（占整条速度时间轴的比例）内关节是否在动。

    用比例窗口而非绝对帧号对齐，容忍像素帧序与 parquet 行数的轻微长度差。
    窗口为空 / 速度不可用 → ``None``（不可判）。
    """
    if speed is None or speed.size == 0:
        return None
    n = speed.size
    lo = int(np.clip(np.floor(frac_start * n), 0, n - 1))
    hi = int(np.clip(np.ceil(frac_end * n), lo + 1, n))
    window = speed[lo:hi]
    if window.size == 0:
        return None
    return bool(window.max() >= move_speed)
