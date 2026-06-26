"""静态 / 无效操作检测器（移植自 video_quality_pipeline 的 StaticDetector，
源工具 robomind2_filter/static_arm）。

三个后端，config['static']['backend'] 选择：
  - 'lite'（纯 CPU）：相邻采样帧灰度 L1 差，sigmoid 软化为 motion_score。
        score = 1 / (1 + exp(-5*(peak_diff/1.5 - 1)))
        problem（静止）= score < thr            默认 thr=0.30
  - 'raft'（需 GPU + RAFT 源码/权重）：RAFT 稠密光流的 active_ratio。
        rel_i  = (top5% 光流幅值均值) / min(H,W)
        score  = active_ratio = mean(rel_i >= 0.012)
        problem（静止）= score < raft_thr        默认 raft_thr=0.10（已标定）
  - 'joint'（LeRobot parquet 关节后端，见 docs/detector_dataset_impact.md §2.1）：
        直接用 puppet 左/右臂逐帧关节位置判静止，绕开「机械臂在大片静止背景中只占
        小块、帧差信号弱」的像素级固有局限。关节是运动的地面真值。
        max_range = max_j (q99_j - q01_j)   # 整段每个关节的稳健峰峰值
        problem（静止）= max_range < joint_range_thr   默认 0.05 rad
        parquet 缺失 / 不可读时回退到 joint_fallback 后端（默认 lite），优雅降级。

score 语义：越高越好（1.0=有正常运动，0.0=全程静止）。raft / joint 后端在依赖缺失、
无 GPU、找不到 parquet 时不会让流水线崩溃，而是降级 / 回退。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

import numpy as np

from vidinspect_agent.checkers._frames import read_frames_gray
from vidinspect_agent.checkers.base import BaseChecker
from vidinspect_agent.models import CheckResult, Severity

THRES_GRAY = 1.5          # lite：灰度差阈值
THRES_REL = 0.012         # raft：相对运动阈值
DEFAULT_THR_LITE = 0.30
DEFAULT_THR_RAFT = 0.10
DEFAULT_JOINT_RANGE_THR = 0.05    # joint：整段关节稳健峰峰值(rad)下限
DEFAULT_JOINT_MOVE_SPEED = 0.005  # joint：单帧"在动"的关节速度阈值(rad/帧)

# puppet 臂关节列候选名（按优先级）。puppet.arm_*_position_align.data，float32[7]。
_ARM_COLUMN_CANDIDATES: dict[str, tuple[str, ...]] = {
    "left": ("puppet.arm_left_position_align.data",),
    "right": ("puppet.arm_right_position_align.data",),
    "single": ("puppet.arm_position_align.data",),
}


class StaticChecker(BaseChecker):
    """机械臂静止 / 无效操作检测（lite CPU + 可选 raft GPU + 可选 joint 关节后端）。"""

    name = "static"

    def check(self, path: Path, metadata: dict[str, Any]) -> list[CheckResult]:
        cfg = self.config.get("static", {})
        backend = cfg.get("backend", "lite")
        if backend == "joint":
            return self._detect_joint(path, cfg, metadata)
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

    # ---- joint (LeRobot parquet 关节后端) ----
    def _detect_joint(
        self, path: Path, cfg: dict, metadata: dict[str, Any]
    ) -> list[CheckResult]:
        fail_severity = _severity(cfg.get("severity", "warn"))
        range_thr = cfg.get("joint_range_thr", DEFAULT_JOINT_RANGE_THR)
        move_speed = cfg.get("joint_move_speed", DEFAULT_JOINT_MOVE_SPEED)
        fallback = cfg.get("joint_fallback", "lite")

        parquet = _parquet_pointer(metadata) or _locate_group_parquet(path)
        if parquet is None:
            return self._joint_fallback(path, cfg, metadata, fallback, "parquet_not_found")
        arms = _read_arm_joints(parquet)
        if not arms:
            return self._joint_fallback(path, cfg, metadata, fallback, "joint_read_failed")

        ev = evaluate_joint_static(arms, range_thr=range_thr, move_speed=move_speed)
        details = {
            "score": ev["score"],
            "max_range": ev["max_range"],
            "peak_speed": ev["peak_speed"],
            "mean_speed": ev["mean_speed"],
            "moving_ratio": ev["moving_ratio"],
            "thr": range_thr,
            "n_frames": ev["n_frames"],
            "arms": ev["arms"],
            "backend": "joint",
            "parquet": str(parquet),
        }
        if ev["detected"]:
            return [
                CheckResult(
                    name="static",
                    severity=fail_severity,
                    message=(
                        f"疑似静止/无效操作: 关节峰峰值 max_range={ev['max_range']:.4f}rad "
                        f"(<{range_thr})"
                    ),
                    details=details,
                )
            ]
        return [
            CheckResult(
                name="static",
                severity=Severity.PASS,
                message=f"运动正常: 关节峰峰值 max_range={ev['max_range']:.4f}rad",
                details=details,
            )
        ]

    def _joint_fallback(
        self, path: Path, cfg: dict, metadata: dict[str, Any], fallback: str, reason: str
    ) -> list[CheckResult]:
        """parquet 缺失/不可读时回退到像素后端（或直接 WARN）。"""
        if fallback == "raft":
            results = self._detect_raft(path, cfg)
        elif fallback == "lite":
            results = self._detect_lite(path, cfg)
        else:  # none：不回退，直接报无法用关节后端评估
            return [
                CheckResult(
                    name="static",
                    severity=Severity.WARN,
                    message=f"关节后端不可用（{reason}），未回退",
                    details={"backend": "joint", "error": reason},
                )
            ]
        for r in results:
            r.details = {
                **r.details,
                "joint_fallback_from": "joint",
                "joint_fallback_reason": reason,
            }
        return results

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


def evaluate_joint_static(
    arm_arrays: dict[str, np.ndarray],
    range_thr: float = DEFAULT_JOINT_RANGE_THR,
    move_speed: float = DEFAULT_JOINT_MOVE_SPEED,
) -> dict[str, Any]:
    """纯函数：由 puppet 臂关节逐帧数组判整段是否静止（便于单测）。

    入参 ``arm_arrays``：``{"left": [T,D], "right": [T,D], ...}``，每个为某只臂的逐帧
    关节位置（弧度）。多只臂在帧轴对齐到最短长度后拼接到关节轴，统一评估。

    指标：
      - ``max_range`` = max_j (q99_j - q01_j)：每个关节整段的**稳健峰峰值**取最大；
        用 1%/99% 分位代替 max-min，抗单帧传感跳变。
      - ``peak_speed`` / ``mean_speed``：相邻帧关节向量 L2 速度（rad/帧）。
      - ``moving_ratio``：速度 ≥ ``move_speed`` 的帧占比。

    判定（静止）：``max_range < range_thr``。``score`` 越高越好，归一到 [0,1]，
    在阈值处约为 1.0（``min(1, max_range/range_thr)``）。
    """
    mats = [
        a
        for a in arm_arrays.values()
        if isinstance(a, np.ndarray) and a.ndim == 2 and a.shape[0] >= 2
    ]
    if not mats:
        return {
            "detected": False,
            "evaluated": False,
            "max_range": 0.0,
            "peak_speed": 0.0,
            "mean_speed": 0.0,
            "moving_ratio": 0.0,
            "score": 1.0,
            "n_frames": 0,
            "arms": [],
        }

    n = min(m.shape[0] for m in mats)
    joints = np.concatenate([m[:n] for m in mats], axis=1).astype(np.float64)

    hi = np.percentile(joints, 99, axis=0)
    lo = np.percentile(joints, 1, axis=0)
    max_range = float(np.max(hi - lo))

    vel = np.linalg.norm(np.diff(joints, axis=0), axis=1)
    peak_speed = float(vel.max()) if vel.size else 0.0
    mean_speed = float(vel.mean()) if vel.size else 0.0
    moving_ratio = float((vel >= move_speed).mean()) if vel.size else 0.0

    detected = max_range < range_thr
    score = float(min(1.0, max_range / range_thr)) if range_thr > 0 else 1.0
    return {
        "detected": bool(detected),
        "evaluated": True,
        "max_range": round(max_range, 6),
        "peak_speed": round(peak_speed, 6),
        "mean_speed": round(mean_speed, 6),
        "moving_ratio": round(moving_ratio, 4),
        "score": round(score, 4),
        "n_frames": int(n),
        "arms": sorted(k for k, v in arm_arrays.items()
                       if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[0] >= 2),
    }


def _parquet_pointer(metadata: dict[str, Any]) -> Optional[Path]:
    """优先用 §1 摄入层注入的 ``metadata["lerobot"]["parquet_path"]`` 指针。"""
    lerobot_meta = metadata.get("lerobot")
    if isinstance(lerobot_meta, dict):
        p = lerobot_meta.get("parquet_path")
        if p:
            return Path(p)
    return None


def _locate_group_parquet(path: Path) -> Optional[Path]:
    """未经 §1 摄入时，由视频路径定位同 episode 的 parquet（复用 §1 lerobot 模块）。

    依赖组根的 ``meta/info.json`` 识别 LeRobot 组；非 LeRobot 布局 / 解析失败 → ``None``。
    """
    try:
        from vidinspect_agent import lerobot as lr

        root = lr.find_group_root(path)
        if root is None:
            return None
        episode = lr.parse_episode_index(path)
        if episode is None:
            return None
        group = lr.load_group(root)
        return group.parquet_by_episode.get(episode)
    except Exception:  # noqa: BLE001 - 定位失败一律降级回退
        return None


def _read_arm_joints(parquet_path: Path) -> dict[str, np.ndarray]:
    """读取 puppet 各侧逐帧臂关节位置 ``{side: [T, D]}``；缺 pyarrow / 列缺失 / 读失败 → ``{}``。"""
    try:
        from vidinspect_agent.lerobot import load_episode_frames

        columns = load_episode_frames(parquet_path)
    except Exception:  # noqa: BLE001 - 缺 pyarrow / 读失败 → 上层回退像素后端
        return {}

    names = list(columns.keys())
    arms: dict[str, np.ndarray] = {}
    for side, cands in _ARM_COLUMN_CANDIDATES.items():
        col = _find_arm_column(names, side, cands)
        if col is None:
            continue
        arr = _rows_to_2d(columns.get(col))
        if arr is not None and arr.ndim == 2 and arr.shape[0] >= 2:
            arms[side] = arr
    return arms


def _find_arm_column(names: list[str], side: str, cands: tuple[str, ...]) -> Optional[str]:
    for cand in cands:
        if cand in names:
            return cand
    loose = re.compile(rf".*arm.*{side}.*position.*\.data$")
    for n in names:
        if loose.search(n):
            return n
    return None


def _rows_to_2d(seq: Any) -> Optional[np.ndarray]:
    """把逐帧关节单元（list[D] 或 {data: [...]} 结构）整理成 ``[T, D]`` 数组。"""
    if not seq:
        return None
    rows: list[list[float]] = []
    for cell in seq:
        if cell is None:
            continue
        if isinstance(cell, dict):
            cell = cell.get("data")
        if cell is None:
            continue
        try:
            vec = np.asarray(cell, dtype=np.float64).ravel()
        except (TypeError, ValueError):
            return None
        if vec.size == 0:
            continue
        rows.append(vec.tolist())
    if len(rows) < 2:
        return None
    width = len(rows[0])
    if any(len(r) != width for r in rows):
        return None
    return np.asarray(rows, dtype=np.float64)


def _severity(value: str) -> Severity:
    try:
        return Severity(str(value).lower())
    except ValueError:
        return Severity.WARN
