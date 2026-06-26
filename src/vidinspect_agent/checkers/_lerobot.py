"""逐帧夹爪开合真实信号 → 逐帧「是否闭合」（object_slip §2.3）。

``object_slip``（规范21）的唯一判据是「持有结束时夹爪**是否仍闭合**」。原实现让多模态
模型逐帧推断 ``gripper_closed``，既贵又不稳；而该信号本就以真实值存在于 LeRobot 组的
parquet 里——``puppet.end_effector_*_position_align``（见 ``docs/dataset_inputs.md §3``）。

本模块在视频 ↔ parquet 自定位之上做三件纯粹的事：

1. 取出 puppet 夹爪开合列（各侧标量序列，逐视频帧一行）。
2. 用每 episode 自身的开合区间做**相对归一化**阈值化成逐帧「是否闭合」（对单位 / 标定不敏感）。
3. 把逐视频帧的闭合序列映射到检测器实际抽到的采样帧时间轴。

任何一步不可用（无 parquet / 缺 ``pyarrow`` / 列缺失 / 读失败）一律返回 ``None`` / 空，
由 ``object_slip`` 优雅降级回模型推断的 ``gripper_closed``。
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Optional

# 各侧夹爪开合列候选名（按优先级）。puppet.end_effector_*_position_align.data，float32[1]。
_GRIPPER_COLUMNS: dict[str, tuple[str, ...]] = {
    "left": ("puppet.end_effector_left_position_align.data",),
    "right": ("puppet.end_effector_right_position_align.data",),
    "single": ("puppet.end_effector_position_align.data",),
}


# --------------------------------------------------------------------------- #
# parquet 自定位
# --------------------------------------------------------------------------- #
def find_episode_parquet(video_path: str | Path) -> Optional[Path]:
    """从视频路径推断同 episode 的 parquet 路径；非 LeRobot 布局 / 找不到返回 ``None``。

    标准布局 ``videos/<chunk>/<cam>/<stem>.mp4 → data/<chunk>/<stem>.parquet`` 直接命中；
    chunk 命名不一致时退化为在该组 ``data/`` 下按同名 stem 递归查找（限定组根内，不全盘扫描）。
    """
    p = Path(video_path).resolve()
    stem = p.stem  # e.g. episode_000129

    cam_dir = p.parent
    chunk_dir = cam_dir.parent
    videos_dir = chunk_dir.parent
    if videos_dir.name != "videos":
        return None

    root = videos_dir.parent
    cand = root / "data" / chunk_dir.name / f"{stem}.parquet"
    if cand.is_file():
        return cand

    data_dir = root / "data"
    if data_dir.is_dir():
        for hit in sorted(data_dir.rglob(f"{stem}.parquet")):
            return hit
    return None


# --------------------------------------------------------------------------- #
# parquet 夹爪开合列读取
# --------------------------------------------------------------------------- #
def read_gripper_opening(parquet_path: str | Path) -> dict[str, list[float]]:
    """读取各侧 puppet 夹爪开合标量序列，返回 ``{side: [opening_per_frame, ...]}``。

    仅含实际存在的列；缺 pyarrow / 列缺失 / 读失败 → ``{}``。值统一为 float（缺失 → NaN）。
    """
    try:
        import pyarrow.parquet as pq
    except Exception:  # noqa: BLE001 - 依赖缺失则降级（上层回退模型信号）
        return {}
    try:
        names = pq.read_schema(parquet_path).names
    except Exception:  # noqa: BLE001
        return {}

    out: dict[str, list[float]] = {}
    for side, cands in _GRIPPER_COLUMNS.items():
        col = _pick_column(names, side, cands)
        if col is None:
            continue
        try:
            cells = pq.read_table(parquet_path, columns=[col]).column(0).to_pylist()
        except Exception:  # noqa: BLE001
            continue
        vals = _to_scalars(cells)
        if vals:
            out[side] = vals
    return out


def extract_gripper_opening(columns: dict[str, list[Any]]) -> dict[str, list[float]]:
    """从「列名→逐帧值」的 parquet 列字典里抽出各侧夹爪开合标量序列。"""
    names = list(columns.keys())
    out: dict[str, list[float]] = {}
    for side, cands in _GRIPPER_COLUMNS.items():
        col = _pick_column(names, side, cands)
        if col is None:
            continue
        cells = columns.get(col)
        if not cells:
            continue
        vals = _to_scalars(cells)
        if vals:
            out[side] = vals
    return out


def gripper_opening_from_metadata(metadata: dict[str, Any]) -> Optional[dict[str, list[float]]]:
    """从 metadata 的 LeRobot parquet 指针读出各侧夹爪开合标量序列（§1 摄入层路径）。

    返回 ``{side: [...]}``（仅含实际存在的列）；无 parquet 指针 / 缺 pyarrow / 读失败 /
    无夹爪列 → ``None``。
    """
    lerobot_meta = metadata.get("lerobot")
    if not isinstance(lerobot_meta, dict):
        return None
    parquet_path = lerobot_meta.get("parquet_path")
    if not parquet_path:
        return None
    try:
        from vidinspect_agent.lerobot import load_episode_frames

        columns = load_episode_frames(parquet_path)
    except Exception:  # noqa: BLE001
        return None
    return extract_gripper_opening(columns) or None


def _pick_column(names: list[str], side: str, cands: tuple[str, ...]) -> Optional[str]:
    """挑某侧列：优先精确候选名，再退化为「含 side 且以 .data 结尾」的列。"""
    for cand in cands:
        if cand in names:
            return cand
    loose = re.compile(rf".*{side}.*position.*\.data$")
    for n in names:
        if loose.search(n):
            return n
    return None


# --------------------------------------------------------------------------- #
# 夹爪开合 → 逐帧「是否闭合」→ 采样帧对齐（纯函数，可单测）
# --------------------------------------------------------------------------- #
def opening_to_closed(
    values: list[Any],
    *,
    closed_is_low: bool = True,
    closed_frac: float = 0.5,
    min_span: float = 1e-6,
) -> list[Optional[bool]]:
    """把夹爪开合标量序列阈值化为逐帧「是否闭合」（``True``/``False``/``None``=不可判）。

    用每 episode 自身的开合区间做相对归一化（取稳健分位 q05/q95 估两端），对绝对单位 /
    标定不敏感：

    - 区间 ``span = hi - lo`` ≤ ``min_span``（夹爪整段几乎不动）→ 无从区分开合，全 ``None``。
    - ``closed_is_low=True``：开合值越**小**越闭合（多数 LeRobot/RoboMIND 约定，0≈合）；
      归一化值 ≤ ``closed_frac`` 判闭合。``closed_is_low=False`` 则相反（值越大越闭合）。
    - 非有限值（NaN/None）→ 该帧 ``None``。
    """
    n = len(values)
    finite = [fv for fv in (_as_float(v) for v in values) if math.isfinite(fv)]
    if len(finite) < 2:
        return [None] * n

    lo = _quantile(finite, 0.05)
    hi = _quantile(finite, 0.95)
    span = hi - lo
    if not math.isfinite(span) or span <= min_span:
        return [None] * n

    out: list[Optional[bool]] = []
    for v in values:
        fv = _as_float(v)
        if not math.isfinite(fv):
            out.append(None)
            continue
        norm = (fv - lo) / span
        closed = norm <= closed_frac if closed_is_low else norm >= (1.0 - closed_frac)
        out.append(bool(closed))
    return out


def map_to_sampled(
    closed_per_frame: list[Optional[bool]],
    frame_times: list[float],
    video_fps: float,
) -> list[Optional[bool]]:
    """把逐视频帧的闭合序列对齐到检测器抽到的采样帧时间轴。

    parquet 一行对应一视频帧，``frame_index`` 与 mp4 帧同步；采样帧第 i 帧的时刻 ``t_i``
    由抽帧器给出。用 ``round(t * fps)`` 取最近的视频帧行；越界 / 无效 fps → 该采样帧
    ``None``（不可判，保守跳过）。
    """
    n_rows = len(closed_per_frame)
    if video_fps <= 0 or n_rows == 0:
        return [None] * len(frame_times)
    out: list[Optional[bool]] = []
    for t in frame_times:
        idx = int(round(t * video_fps))
        out.append(closed_per_frame[idx] if 0 <= idx < n_rows else None)
    return out


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def _as_float(v: Any) -> float:
    try:
        return float(v) if v is not None else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _to_scalars(cells: list[Any]) -> list[float]:
    """把 parquet 单元（list<float>[1] 或标量）统一成 float 标量列（None/空 → NaN）。"""
    out: list[float] = []
    for c in cells:
        if isinstance(c, (list, tuple)):
            c = c[0] if c else None
        out.append(_as_float(c))
    return out


def _quantile(xs: list[float], q: float) -> float:
    """有限值列表的线性插值分位数（无需预排序）。"""
    s = sorted(float(x) for x in xs)
    if not s:
        return float("nan")
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] + (s[hi] - s[lo]) * frac
