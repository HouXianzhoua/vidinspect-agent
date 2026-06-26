"""二次抓取检测器（质检规范序号 1，多模态）。

规范：视频中不应出现多次抓取同一物体的情况，每次抓取都应是单次且有效的。

整段视频直接问模型「是否发生二次抓取」做不出来——它要同时干两件事：逐帧感知
（哪只机械臂在抓、抓的是什么）+ 跨时间推理（判断「再次」）。本检测器把它拆成
**逐帧感知交给模型 + 时序判定留给代码**（与 gripper_offscreen 同一套路）：

1. 本地抽帧 → 让多模态模型逐帧、**逐夹爪**回 ``{side, holding, object_label, confidence}``；
2. 代码侧按夹爪 ``side``（left / right / single）分别归约成「该臂当前是否持有」序列；
3. 去抖：丢弃过短的「持有」段（误检），桥接过短的「释放」缝隙（遮挡闪断）；
4. **逐臂**统计独立「持有段」数：某只机械臂 ≥2 段（被真释放隔开）即判二次抓取。

**关键：按机械臂（单臂或双臂）分别判定。** 每只机械臂都是单目标，故同一只臂
再次抓取即命中；双臂各自抓取一次（甚至 A→B 交接）则正常，不会误报。判定始终
留在代码侧（逐臂去抖后的持有段计数），确定可复现。缺 API key / SDK 缺失 /
抽帧失败 / 接口异常 → 一律降级为 WARN，不阻塞流水线。

**夹爪信号来源（规范1 §2.2 改造）**：抓取 / 释放的「时序」本就以真实信号存在于
LeRobot 组的 parquet 里（``puppet.end_effector_*_position_align``，见
``docs/dataset_inputs.md §3``）。逐臂判定是单目标、忽略物体身份，故「夹爪闭合段数」
即「抓取次数」——当视频位于 LeRobot v3.0 组内、能自定位到同 episode parquet 且装有
``pyarrow`` 时，**优先用 parquet 夹爪开合真实信号逐臂判定**：完全不调多模态模型
（无需 API key、零付费调用、用全分辨率时间轴），去抖 / 计数落在真实信号上更稳。
任一前置不满足（配置关闭 / 非 LeRobot 布局 / 缺 parquet / 缺 pyarrow / 缺视频 fps /
开合区间过小不可判）→ 自动回退到原多模态逐帧推断路径，行为同改造前。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from vidinspect_agent.checkers import _lerobot
from vidinspect_agent.checkers._frames import sample_frames_jpeg
from vidinspect_agent.checkers._vision import build_backend
from vidinspect_agent.checkers.base import BaseChecker
from vidinspect_agent.models import CheckResult, Severity

_PROVIDER_DEFAULTS = {
    "gemini": {"model": "gemini-2.5-flash", "api_key_env": "GEMINI_API_KEY"},
    "openai": {"model": "gpt-4o", "api_key_env": "OPENAI_API_KEY"},
}

_SENTINEL = "__hold__"  # 逐臂判定时只数持有段，用占位标签（忽略物体身份）

# side 归一化映射 → 内部规范值；空 / 未知归到 single（单臂场景）。
_SIDE_ALIASES = {
    "left": "left", "l": "left", "左": "left", "left_arm": "left", "leftarm": "left",
    "right": "right", "r": "right", "右": "right", "right_arm": "right", "rightarm": "right",
}
_SIDE_DISPLAY = {"left": "左臂", "right": "右臂", "single": "机械臂"}

_PROMPT_GRASP = (
    "你是机器人采集数据的视频质检员。下面按时间顺序给出从同一段视频均匀抽取的若干帧，"
    "每帧前用 'Frame i:' 标注帧号（从 0 开始）。\n"
    "任务：逐帧判断画面中**每一只**机械臂【夹爪】当前是否正抓持着某个物体。\n"
    "【夹爪】指机械臂末端的执行器 / 抓手 / gripper（夹持手指、吸盘或灵巧手等）。\n"
    "输出规则：\n"
    "- 本数据可能是单臂或双臂。双臂时请按画面**左右**区分，side 填 'left' 或 'right'；"
    "单臂时 side 填 'single'。同一只机械臂在所有帧务必使用同一 side 值，不要左右互换。\n"
    "- 每帧的 grippers 列出该帧可见的各只夹爪各一条；某帧看不到某只夹爪可不列出该条。\n"
    "- 夹爪已夹住 / 吸住物体并使其随夹爪移动（抓取中 / 搬运中 / 提起）→ holding=true；\n"
    "- 仅靠近、轻触但未夹起，或物体已静置于桌面 / 已放下 → holding=false；\n"
    "- holding=true 时用简短稳定的英文名称填 object_label（如 'red_cube'）；否则留空。\n"
    "- 因遮挡难以判断时，结合前后帧连续性给出最可能判断，并相应降低 confidence(0~1)。\n"
    "对每一帧都必须返回一条结果，index 与 'Frame i' 的 i 一致。\n"
    "{robot_hint}"
)


def _robot_hint(robot: Any) -> str:
    if isinstance(robot, str) and robot:
        return f"参考：本视频机器人型号为 {robot}，据此理解夹爪外形与单/双臂构型。\n"
    return ""


def _normalize_side(side: Any) -> str:
    s = str(side or "").strip().lower()
    return _SIDE_ALIASES.get(s, "single" if not s else s)


class RegraspChecker(BaseChecker):
    """二次抓取检测（多模态，逐帧抓取状态 + 代码侧去抖计数）。"""

    name = "regrasp"

    def check(self, path: Path, metadata: dict[str, Any]) -> list[CheckResult]:
        cfg = self.config.get("regrasp", {})
        fail_severity = _severity(cfg.get("severity", "warn"))

        # §2.2：优先用 parquet 夹爪开合真实信号逐臂判定（零付费调用 / 无需 API key / 全分辨率）。
        parquet_results = self._check_via_parquet(path, metadata, cfg, fail_severity)
        if parquet_results is not None:
            return parquet_results

        # 回退：多模态逐帧推断（原行为，需 API key + 抽帧）。
        return self._check_via_model(path, metadata, cfg, fail_severity)

    def _check_via_model(
        self,
        path: Path,
        metadata: dict[str, Any],
        cfg: dict[str, Any],
        fail_severity: Severity,
    ) -> list[CheckResult]:
        provider = str(cfg.get("provider", "gemini")).lower()
        min_conf = float(cfg.get("min_confidence", 0.0))

        defaults = _PROVIDER_DEFAULTS.get(provider, {})
        pcfg = cfg.get(provider, {})
        model = pcfg.get("model", defaults.get("model"))
        api_key_env = pcfg.get("api_key_env", defaults.get("api_key_env", "GEMINI_API_KEY"))
        if model is None:
            return [self._warn(f"未知 provider: {provider}", {"error": "bad_provider"})]
        api_key = os.environ.get(api_key_env)
        if not api_key:
            return [self._warn(f"未设置 {api_key_env}，跳过二次抓取检测",
                               {"error": "missing_api_key"})]

        try:
            backend = build_backend(provider, model, api_key, cfg)
        except Exception as exc:  # noqa: BLE001
            return [self._warn(f"后端初始化失败: {exc}", {"error": "backend_init"})]

        sample_fps = float(cfg.get("sample_fps", 2.0))
        max_h = int(cfg.get("frame_max_h", 360))
        max_frames = int(cfg.get("max_frames", 120))
        timeout = float(cfg.get("timeout", 120.0))

        # 长视频自适应降采样：保证抽帧总数不超过 max_frames，覆盖全片。
        duration = metadata.get("duration_sec")
        eff_fps = sample_fps
        if duration and duration > 0:
            eff_fps = min(sample_fps, max_frames / float(duration))
        eff_fps = max(eff_fps, 1e-3)

        frames, err = sample_frames_jpeg(
            str(path), sample_fps=eff_fps, max_h=max_h,
            max_frames=max_frames, timeout=timeout,
        )
        if frames is None:
            return [self._warn(f"抽帧失败，跳过二次抓取检测: {err}", {"error": err})]

        prompt = _PROMPT_GRASP.format(robot_hint=_robot_hint(metadata.get("robot")))
        try:
            verdicts = backend.classify_grasp_frames(frames, prompt)
        except Exception as exc:  # noqa: BLE001 - 调用异常一律降级
            return [self._warn(f"{provider} 调用失败: {type(exc).__name__}: {exc}",
                               {"error": "backend_error"})]

        n = len(frames)
        # 逐帧、逐夹爪归约成「该臂是否持有」序列：seq[side][i] = 标签 或 None。
        seqs: dict[str, list[str | None]] = {}
        for idx in range(n):
            for g in verdicts.get(idx) or []:
                if not g.get("holding"):
                    continue
                if min_conf > 0 and float(g.get("confidence", 1.0)) < min_conf:
                    continue
                side = _normalize_side(g.get("side"))
                seq = seqs.setdefault(side, [None] * n)
                seq[idx] = _normalize(g.get("object_label")) or _SENTINEL

        # 去抖阈值（秒 → 帧）；两者均下限 2 帧，过滤单帧误检 / 单帧遮挡闪断。
        min_hold_frames = max(2, round(float(cfg.get("min_hold_sec", 0.5)) * eff_fps))
        min_release_frames = max(2, round(float(cfg.get("min_release_sec", 1.0)) * eff_fps))

        # 逐臂判定：每只机械臂单目标，故忽略物体标签只数持有段（single_object=True）。
        arm_counts: dict[str, int] = {}
        arm_starts_sec: dict[str, list[float]] = {}
        offenders: list[tuple[str, int]] = []
        for side, seq in seqs.items():
            analysis = detect_regrasp(
                seq, single_object=True,
                min_hold_frames=min_hold_frames, min_release_frames=min_release_frames,
            )
            count = analysis["counts"].get(_SENTINEL, 0)
            arm_counts[side] = count
            arm_starts_sec[side] = [round(a / eff_fps, 2)
                                    for a in analysis["starts"].get(_SENTINEL, [])]
            if count >= 2:
                offenders.append((side, count))

        details = {
            "provider": provider,
            "model": model,
            "arm_grasp_counts": arm_counts,
            "offending_arms": {s: c for s, c in offenders},
            "grasp_start_sec": {s: arm_starts_sec[s] for s, _c in offenders},
            "sample_fps": round(eff_fps, 3),
            "min_hold_sec": round(min_hold_frames / eff_fps, 2),
            "min_release_sec": round(min_release_frames / eff_fps, 2),
            "n_frames": n,
        }

        if offenders:
            parts = [f"{_SIDE_DISPLAY.get(s, s)}抓取 {c} 次" for s, c in offenders]
            msg = "疑似二次抓取: " + "；".join(parts) + "（每臂应为 1 次）"
            return [CheckResult(name=self.name, severity=fail_severity,
                                message=msg, details=details)]
        return [CheckResult(name=self.name, severity=Severity.PASS,
                            message="未检测到二次抓取", details=details)]

    def _check_via_parquet(
        self,
        path: Path,
        metadata: dict[str, Any],
        cfg: dict[str, Any],
        fail_severity: Severity,
    ) -> list[CheckResult] | None:
        """§2.2：用 parquet 夹爪开合真实信号逐臂判二次抓取（不调模型 / 不抽帧）。

        逐臂单目标、忽略物体身份，故「夹爪闭合段数」即「抓取次数」，时序完全由真实信号
        驱动。任一前置不满足（配置关闭 / 非 LeRobot 组 / 缺 parquet / 缺 pyarrow / 缺视频
        fps / 各侧开合整段不可判）→ 返回 ``None``，由调用方回退到多模态路径。
        """
        if not cfg.get("use_parquet_gripper", True):
            return None
        video_fps = metadata.get("fps")
        if not video_fps or video_fps <= 0:
            return None
        parquet_path = _lerobot.find_episode_parquet(path)
        if parquet_path is None:
            return None
        openings = _lerobot.read_gripper_opening(parquet_path)
        if not openings:
            return None

        closed_is_low = bool(cfg.get("gripper_closed_is_low", True))
        closed_frac = float(cfg.get("gripper_closed_frac", 0.5))
        min_span = float(cfg.get("gripper_min_span", 1e-6))
        # parquet 一行 = 一视频帧，故按视频 fps 把去抖阈值（秒）换算成帧；两者下限 2 帧。
        fps = float(video_fps)
        min_hold_frames = max(2, round(float(cfg.get("min_hold_sec", 0.5)) * fps))
        min_release_frames = max(2, round(float(cfg.get("min_release_sec", 1.0)) * fps))

        arm_counts: dict[str, int] = {}
        arm_starts_sec: dict[str, list[float]] = {}
        offenders: list[tuple[str, int]] = []
        evaluated = 0
        for side, values in openings.items():
            closed = _lerobot.opening_to_closed(
                values, closed_is_low=closed_is_low,
                closed_frac=closed_frac, min_span=min_span,
            )
            if all(c is None for c in closed):
                continue  # 该侧开合整段不可判（夹爪几乎不动）→ 跳过该侧
            evaluated += 1
            # 夹爪闭合 = 持有（逐臂单目标，无需认物体）；未知(None) 按未持有处理。
            seq = [_SENTINEL if c else None for c in closed]
            analysis = detect_regrasp(
                seq, single_object=True,
                min_hold_frames=min_hold_frames, min_release_frames=min_release_frames,
            )
            count = analysis["counts"].get(_SENTINEL, 0)
            arm_counts[side] = count
            arm_starts_sec[side] = [round(a / fps, 2)
                                    for a in analysis["starts"].get(_SENTINEL, [])]
            if count >= 2:
                offenders.append((side, count))

        if evaluated == 0:
            return None  # parquet 在但各侧开合全不可判 → 回退模型路径

        details = {
            "source": "parquet",
            "parquet": str(parquet_path),
            "arm_grasp_counts": arm_counts,
            "offending_arms": {s: c for s, c in offenders},
            "grasp_start_sec": {s: arm_starts_sec[s] for s, _c in offenders},
            "fps": round(fps, 3),
            "min_hold_sec": round(min_hold_frames / fps, 2),
            "min_release_sec": round(min_release_frames / fps, 2),
            "gripper_closed_is_low": closed_is_low,
            "n_frames": len(next(iter(openings.values()))),
        }
        if offenders:
            parts = [f"{_SIDE_DISPLAY.get(s, s)}抓取 {c} 次" for s, c in offenders]
            msg = "疑似二次抓取: " + "；".join(parts) + "（每臂应为 1 次）"
            return [CheckResult(name=self.name, severity=fail_severity,
                                message=msg, details=details)]
        return [CheckResult(name=self.name, severity=Severity.PASS,
                            message="未检测到二次抓取（parquet 夹爪信号）", details=details)]

    def _warn(self, msg: str, details: dict[str, Any]) -> CheckResult:
        return CheckResult(name=self.name, severity=Severity.WARN, message=msg, details=details)


# ------------------------------ 纯逻辑（可单测） ------------------------------

def _normalize(label: Any) -> str:
    return str(label or "").strip().lower()


def _segments(seq: list[str | None]) -> list[list]:
    """按相邻相等值切段，返回 [[label, start, end_excl], ...]。"""
    segs: list[list] = []
    i, n = 0, len(seq)
    while i < n:
        j = i
        while j < n and seq[j] == seq[i]:
            j += 1
        segs.append([seq[i], i, j])
        i = j
    return segs


def _rebuild(segs: list[list], n: int) -> list[str | None]:
    out: list[str | None] = [None] * n
    for label, a, b in segs:
        for k in range(a, b):
            out[k] = label
    return out


def detect_regrasp(
    seq: list[str | None],
    *,
    single_object: bool,
    min_hold_frames: int,
    min_release_frames: int,
) -> dict[str, Any]:
    """在逐帧「持有标签」序列上判定二次抓取。

    - ``seq``：第 i 帧持有的物体标签（``None`` 表示未持有）。
    - 去抖：丢弃长度 < ``min_hold_frames`` 的持有段（误检）；桥接长度
      < ``min_release_frames`` 且两侧为同一物体（或 ``single_object``）的释放缝隙（遮挡）。
    - 统计每个物体的独立持有段数，≥2 即二次抓取。

    返回 ``{detected, counts, repeated, starts, hold_segments}``。
    """
    n = len(seq)
    if n == 0:
        return {"detected": False, "counts": {}, "repeated": {},
                "starts": {}, "hold_segments": []}

    # 1) 丢弃过短的「持有」段（疑似单帧误检）。
    segs = _segments(seq)
    for s in segs:
        if s[0] is not None and (s[2] - s[1]) < min_hold_frames:
            s[0] = None
    seq = _rebuild(segs, n)

    # 2) 桥接过短的「释放」缝隙（遮挡闪断）：仅当两侧持有为同一物体（或 single_object）。
    segs = _segments(seq)
    for idx in range(1, len(segs) - 1):
        label, a, b = segs[idx]
        if label is not None or (b - a) >= min_release_frames:
            continue
        left, right = segs[idx - 1][0], segs[idx + 1][0]
        if left is not None and right is not None and (single_object or left == right):
            segs[idx][0] = left
    seq = _rebuild(segs, n)

    # 3) 统计去抖后的持有段。
    holds = [(label, a, b) for label, a, b in _segments(seq) if label is not None]
    counts: dict[str, int] = {}
    starts: dict[str, list[int]] = {}
    for label, a, _b in holds:
        key = _SENTINEL if single_object else label
        counts[key] = counts.get(key, 0) + 1
        starts.setdefault(key, []).append(a)

    repeated = {label: c for label, c in counts.items() if c >= 2}
    return {
        "detected": bool(repeated),
        "counts": counts,
        "repeated": repeated,
        "starts": starts,
        "hold_segments": holds,
    }


def _severity(value: str) -> Severity:
    try:
        return Severity(str(value).lower())
    except ValueError:
        return Severity.WARN
