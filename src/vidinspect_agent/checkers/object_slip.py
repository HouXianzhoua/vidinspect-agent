"""物体滑落检测器（质检规范序号 21，多模态）。

规范：夹爪夹住物体以后，物体滑落（脱手 / 掉落）。

与「二次抓取」（regrasp，序号 1）共用同一套架构（本地抽帧 → 多模态逐帧感知 →
代码侧时序状态机），但**判据不同**，不是它的子集：

- regrasp 看「同一只臂的持有段数 ≥2」——副作用上能抓到「滑落后又捡起」这一子集，
  但**会错标成二次抓取**，且抓不到「滑落后不补抓」「夹空空走」。
- object_slip 看「持有结束时夹爪**是否仍闭合**」：
  - 主动张开放下 → 夹爪张开（``gripper_closed=False``）→ 正常释放，不报；
  - 物体滑落 → 夹爪仍闭合但物体没了（``holding`` 由 True→False 而 ``gripper_closed``
    仍为 True）→ 命中。

因此本检测器在 regrasp 的逐帧 schema 上**额外要求模型回 ``gripper_closed``**，判定仍
留在代码侧（逐臂去抖 + 持有段末尾的夹爪状态判别），确定可复现。缺 API key / SDK /
抽帧失败 / 接口异常 → 一律降级 WARN，不阻塞流水线。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from vidinspect_agent.checkers._frames import sample_frames_jpeg
from vidinspect_agent.checkers._vision import build_backend
from vidinspect_agent.checkers.base import BaseChecker
from vidinspect_agent.checkers.regrasp import (
    _normalize,
    _normalize_side,
    _rebuild,
    _segments,
    _severity,
    _SIDE_DISPLAY,
)
from vidinspect_agent.models import CheckResult, Severity

_PROVIDER_DEFAULTS = {
    "gemini": {"model": "gemini-2.5-flash", "api_key_env": "GEMINI_API_KEY"},
    "openai": {"model": "gpt-4o", "api_key_env": "OPENAI_API_KEY"},
}

_PROMPT_SLIP = (
    "你是机器人采集数据的视频质检员。下面按时间顺序给出从同一段视频均匀抽取的若干帧，"
    "每帧前用 'Frame i:' 标注帧号（从 0 开始）。\n"
    "任务：逐帧判断画面中**每一只**机械臂【夹爪】的两件事：(1) 当前是否正抓持着某个物体；"
    "(2) 夹爪本身当前是【闭合】还是【张开】。\n"
    "【夹爪】指机械臂末端的执行器 / 抓手 / gripper（夹持手指、吸盘或灵巧手等）。\n"
    "输出规则：\n"
    "- 本数据可能是单臂或双臂。双臂时请按画面**左右**区分，side 填 'left' 或 'right'；"
    "单臂时 side 填 'single'。同一只机械臂在所有帧务必使用同一 side 值，不要左右互换。\n"
    "- 每帧的 grippers 列出该帧可见的各只夹爪各一条；某帧看不到某只夹爪可不列出该条。\n"
    "- holding：夹爪已夹住 / 吸住物体并使其随夹爪移动（抓取中 / 搬运中 / 提起）→ true；"
    "仅靠近、轻触但未夹起，或物体已静置桌面 / 已放下 → false。\n"
    "- gripper_closed：夹爪手指处于**闭合/夹紧**状态填 true；明显**张开/松开**填 false。"
    "这与 holding 是两件事——夹爪可以闭合却没夹住任何东西（物体已滑落）。请尽量每条都给出。\n"
    "- holding=true 时用简短稳定的英文名称填 object_label（如 'red_cube'）；否则留空。\n"
    "- 因遮挡难以判断时，结合前后帧连续性给出最可能判断，并相应降低 confidence(0~1)。\n"
    "对每一帧都必须返回一条结果，index 与 'Frame i' 的 i 一致。\n"
    "{robot_hint}"
)


def _robot_hint(robot: Any) -> str:
    if isinstance(robot, str) and robot:
        return f"参考：本视频机器人型号为 {robot}，据此理解夹爪外形与单/双臂构型。\n"
    return ""


class ObjectSlipChecker(BaseChecker):
    """物体滑落检测（多模态，逐帧抓取/夹爪状态 + 代码侧状态机）。"""

    name = "object_slip"

    def check(self, path: Path, metadata: dict[str, Any]) -> list[CheckResult]:
        cfg = self.config.get("object_slip", {})
        provider = str(cfg.get("provider", "gemini")).lower()
        min_conf = float(cfg.get("min_confidence", 0.0))
        fail_severity = _severity(cfg.get("severity", "warn"))

        defaults = _PROVIDER_DEFAULTS.get(provider, {})
        pcfg = cfg.get(provider, {})
        model = pcfg.get("model", defaults.get("model"))
        api_key_env = pcfg.get("api_key_env", defaults.get("api_key_env", "GEMINI_API_KEY"))
        if model is None:
            return [self._warn(f"未知 provider: {provider}", {"error": "bad_provider"})]
        api_key = os.environ.get(api_key_env)
        if not api_key:
            return [self._warn(f"未设置 {api_key_env}，跳过物体滑落检测",
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
            return [self._warn(f"抽帧失败，跳过物体滑落检测: {err}", {"error": err})]

        prompt = _PROMPT_SLIP.format(robot_hint=_robot_hint(metadata.get("robot")))
        try:
            verdicts = backend.classify_grasp_frames(frames, prompt)
        except Exception as exc:  # noqa: BLE001 - 调用异常一律降级
            return [self._warn(f"{provider} 调用失败: {type(exc).__name__}: {exc}",
                               {"error": "backend_error"})]

        n = len(frames)
        # 逐帧、逐夹爪归约成每只臂的两条序列：hold[side][i]=持有标签或 None；
        # closed[side][i]=夹爪是否闭合(True/False/None=未知)。
        hold_seqs: dict[str, list[str | None]] = {}
        closed_seqs: dict[str, list[bool | None]] = {}
        for idx in range(n):
            for g in verdicts.get(idx) or []:
                side = _normalize_side(g.get("side"))
                closed_seqs.setdefault(side, [None] * n)[idx] = g.get("gripper_closed")
                if not g.get("holding"):
                    continue
                if min_conf > 0 and float(g.get("confidence", 1.0)) < min_conf:
                    continue
                hold_seqs.setdefault(side, [None] * n)[idx] = (
                    _normalize(g.get("object_label")) or _SENTINEL
                )

        # 去抖阈值（秒 → 帧）；下限 2 帧，过滤单帧误检 / 单帧遮挡闪断。
        min_hold_frames = max(2, round(float(cfg.get("min_hold_sec", 0.5)) * eff_fps))
        min_release_frames = max(2, round(float(cfg.get("min_release_sec", 1.0)) * eff_fps))
        release_window_frames = max(
            1, round(float(cfg.get("release_window_sec", 0.5)) * eff_fps)
        )

        arm_slips: dict[str, int] = {}
        slip_sec: dict[str, list[float]] = {}
        offenders: list[tuple[str, int]] = []
        for side, hold_seq in hold_seqs.items():
            closed_seq = closed_seqs.get(side, [None] * n)
            analysis = detect_slip(
                hold_seq, closed_seq,
                min_hold_frames=min_hold_frames,
                min_release_frames=min_release_frames,
                release_window_frames=release_window_frames,
            )
            count = len(analysis["events"])
            arm_slips[side] = count
            slip_sec[side] = [round(e["release_frame"] / eff_fps, 2)
                              for e in analysis["events"]]
            if count >= 1:
                offenders.append((side, count))

        details = {
            "provider": provider,
            "model": model,
            "arm_slip_counts": arm_slips,
            "offending_arms": {s: c for s, c in offenders},
            "slip_at_sec": {s: slip_sec[s] for s, _c in offenders},
            "sample_fps": round(eff_fps, 3),
            "min_hold_sec": round(min_hold_frames / eff_fps, 2),
            "min_release_sec": round(min_release_frames / eff_fps, 2),
            "release_window_sec": round(release_window_frames / eff_fps, 2),
            "n_frames": n,
        }

        if offenders:
            parts = [f"{_SIDE_DISPLAY.get(s, s)}疑似滑落 {c} 次" for s, c in offenders]
            msg = "疑似物体滑落: " + "；".join(parts) + "（夹爪仍闭合但物体脱手）"
            return [CheckResult(name=self.name, severity=fail_severity,
                                message=msg, details=details)]
        return [CheckResult(name=self.name, severity=Severity.PASS,
                            message="未检测到物体滑落", details=details)]

    def _warn(self, msg: str, details: dict[str, Any]) -> CheckResult:
        return CheckResult(name=self.name, severity=Severity.WARN, message=msg, details=details)


# ------------------------------ 纯逻辑（可单测） ------------------------------

_SENTINEL = "__hold__"  # 与 regrasp 一致：逐臂单目标，只关心「是否持有」忽略物体身份


def detect_slip(
    hold_seq: list[str | None],
    closed_seq: list[bool | None],
    *,
    min_hold_frames: int,
    min_release_frames: int,
    release_window_frames: int,
) -> dict[str, Any]:
    """在「逐帧是否持有 + 夹爪是否闭合」序列上判定物体滑落。

    - ``hold_seq[i]``：第 i 帧该臂持有的物体标签（``None`` 表示未持有）。
    - ``closed_seq[i]``：第 i 帧夹爪是否闭合（``True`` / ``False`` / ``None``=未知）。
    - 去抖：丢弃 < ``min_hold_frames`` 的持有段；桥接 < ``min_release_frames`` 的释放缝隙
      （与 regrasp 同一套，压遮挡/误检噪声）。
    - 判定：对每个**在视频结束前结束**的持有段，检查其后 ``release_window_frames`` 帧
      （不越过下一持有段）内的夹爪状态：
        - 窗口内出现夹爪张开（``gripper_closed=False``）→ 正常放下，不计；
        - 窗口内已知状态全为闭合（``True``）→ 夹爪仍闭合却已脱手 → **滑落**；
        - 窗口内全未知 → 无法判别，跳过（保守不误报）。

    返回 ``{detected, events, n_release}``，``events`` 为
    ``[{label, hold_start, release_frame}, ...]``。
    """
    n = len(hold_seq)
    if n == 0:
        return {"detected": False, "events": [], "n_release": 0}

    # 1) 丢弃过短的「持有」段（疑似单帧误检）。
    segs = _segments(hold_seq)
    for s in segs:
        if s[0] is not None and (s[2] - s[1]) < min_hold_frames:
            s[0] = None
    seq = _rebuild(segs, n)

    # 2) 桥接过短的「释放」缝隙（遮挡闪断）：逐臂单目标，两侧持有即桥接（忽略标签）。
    segs = _segments(seq)
    for idx in range(1, len(segs) - 1):
        label, a, b = segs[idx]
        if label is not None or (b - a) >= min_release_frames:
            continue
        left, right = segs[idx - 1][0], segs[idx + 1][0]
        if left is not None and right is not None:
            segs[idx][0] = left
    seq = _rebuild(segs, n)

    holds = [(label, a, b) for label, a, b in _segments(seq) if label is not None]

    events: list[dict[str, Any]] = []
    n_release = 0
    for i, (label, a, b) in enumerate(holds):
        if b >= n:
            continue  # 持有持续到片尾，未观察到释放 → 不判
        n_release += 1
        # 检查窗口：释放点之后若干帧，但不越过下一持有段。
        gap_end = holds[i + 1][1] if i + 1 < len(holds) else n
        end = min(b + release_window_frames, gap_end)
        states = [closed_seq[k] for k in range(b, end)]
        known = [s for s in states if s is not None]
        if not known:
            continue  # 夹爪状态全未知 → 保守跳过
        if any(s is False for s in known):
            continue  # 夹爪张开 → 正常放下
        events.append({"label": label, "hold_start": a, "release_frame": b})

    return {"detected": bool(events), "events": events, "n_release": n_release}
