"""夹取位置过于极限检测器（质检规范序号 16，多模态）。

规范：抓取的是物体边缘而不是主体——夹爪应夹在物体主体 / 重心附近，夹在边缘 / 棱角 /
极限位置容易夹不稳、夹偏或脱手，属低质量采集。

「夹爪夹的是物体边缘还是主体」本质是**语义识别 + 空间关系**判断——需要先定位被操作物体、
再判断夹爪接触点相对物体的位置，纯像素算法做不稳健，故走多模态。与 colormatch 一样，
本项可按**逐帧独立判定 + 代码侧占比聚合**处理（不需要时序状态机）：

1. 本地抽帧（少量、可覆盖全片）→ 让多模态模型**逐帧**判定「夹爪夹住物体时是否夹在边缘」；
   只对「能看到夹爪正夹住某物体」的帧返回，其余帧（未夹住 / 看不清）省略不返回；
2. 代码侧聚合：在能判定的帧里，「夹在边缘」帧占比 ≥ 阈值即命中。

判定留在代码侧（命中帧占比 / 阈值），确定可复现。因「边缘 vs 主体」带主观性，命中默认
WARN（人工复核提示），可在 config 改为 fail。缺 API key / SDK / 抽帧失败 / 接口异常 →
一律降级为 WARN，不阻塞流水线。
"""
from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any

from vidinspect_agent.checkers._frames import sample_frames_jpeg
from vidinspect_agent.checkers._vision import build_backend
from vidinspect_agent.checkers.base import BaseChecker
from vidinspect_agent.checkers.colormatch import _robot_hint, _task_hint
from vidinspect_agent.checkers.regrasp import _severity
from vidinspect_agent.models import CheckResult, Severity

_PROVIDER_DEFAULTS = {
    "gemini": {"model": "gemini-2.5-flash", "api_key_env": "GEMINI_API_KEY"},
    "openai": {"model": "gpt-4o", "api_key_env": "OPENAI_API_KEY"},
}

_PROMPT_EDGE_GRASP = (
    "你是机器人采集数据的视频质检员。下面按时间顺序给出从同一段视频均匀抽取的若干帧，"
    "每帧前用 'Frame i:' 标注帧号（从 0 开始）。\n"
    "任务：逐帧判断画面中**夹爪夹住物体时的夹取位置是否过于极限**——即夹爪是否夹在物体的"
    "**边缘 / 棱角 / 尖端 / 一角**，而不是物体的**主体 / 中部 / 重心附近**。\n"
    "【被操作物体】指机器人夹爪正在抓取 / 夹持的目标物体，不是夹爪本身、不是机械臂连杆、"
    "也不是无关的背景道具。\n"
    "判定规则：\n"
    "- 夹爪接触 / 夹持点明显落在物体边缘、棱角、薄边、一个小角上（夹取面积小、易夹偏 / 脱手）"
    " → edge_grasp=true；\n"
    "- 夹爪夹在物体主体、中部、重心附近，夹持稳固 → edge_grasp=false；\n"
    "- 用简短稳定的英文名称填 object_label（如 'white_bowl'）；判断不确定时降低 confidence(0~1)。\n"
    "- 某帧看不到夹爪正夹住物体（夹爪张开 / 未接触物体 / 物体被完全遮挡 / 已出画 / 看不清接触点）"
    "→ 该帧可省略不返回。\n"
    "只对「能看清夹爪正夹住某物体且能判断夹取位置」的帧返回结果，index 与 'Frame i' 的 i 一致。\n"
    "{robot_hint}{task_hint}"
)


class EdgeGraspChecker(BaseChecker):
    """夹取位置过于极限检测（多模态，逐帧边缘 / 主体判定 + 代码侧占比聚合）。"""

    name = "edge_grasp"

    def check(self, path: Path, metadata: dict[str, Any]) -> list[CheckResult]:
        cfg = self.config.get("edge_grasp", {})
        provider = str(cfg.get("provider", "gemini")).lower()
        min_conf = float(cfg.get("min_confidence", 0.0))
        hit_ratio_thr = float(cfg.get("hit_ratio", 0.5))
        min_judged = int(cfg.get("min_judged_frames", 2))
        fail_severity = _severity(cfg.get("severity", "warn"))

        defaults = _PROVIDER_DEFAULTS.get(provider, {})
        pcfg = cfg.get(provider, {})
        model = pcfg.get("model", defaults.get("model"))
        api_key_env = pcfg.get("api_key_env", defaults.get("api_key_env", "GEMINI_API_KEY"))
        if model is None:
            return [self._warn(f"未知 provider: {provider}", {"error": "bad_provider"})]
        api_key = os.environ.get(api_key_env)
        if not api_key:
            return [self._warn(f"未设置 {api_key_env}，跳过夹取位置极限检测",
                               {"error": "missing_api_key"})]

        try:
            backend = build_backend(provider, model, api_key, cfg)
        except Exception as exc:  # noqa: BLE001
            return [self._warn(f"后端初始化失败: {exc}", {"error": "backend_init"})]

        sample_fps = float(cfg.get("sample_fps", 2.0))
        max_h = int(cfg.get("frame_max_h", 360))
        max_frames = int(cfg.get("max_frames", 60))
        timeout = float(cfg.get("timeout", 120.0))

        # 夹取位置需要看清接触点细节，但仍是「逐帧独立判」，长视频自适应降采样覆盖全片。
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
            return [self._warn(f"抽帧失败，跳过夹取位置极限检测: {err}", {"error": err})]

        prompt = _PROMPT_EDGE_GRASP.format(
            robot_hint=_robot_hint(metadata.get("robot")),
            task_hint=_task_hint(metadata),
        )
        try:
            verdicts = backend.classify_edge_grasp_frames(frames, prompt)
        except Exception as exc:  # noqa: BLE001 - 调用异常一律降级
            return [self._warn(f"{provider} 调用失败: {type(exc).__name__}: {exc}",
                               {"error": "backend_error"})]

        n = len(frames)
        frame_verdicts = [verdicts.get(i) for i in range(n)]
        analysis = evaluate_edge_grasp(
            frame_verdicts, hit_ratio_thr=hit_ratio_thr, min_confidence=min_conf
        )

        # 取「夹在边缘」帧里出现最多的物体标签作复核展示。
        labels = Counter(
            v["object_label"]
            for v in frame_verdicts
            if v and v.get("edge") and v.get("object_label")
        )
        top_label = labels.most_common(1)[0][0] if labels else ""

        details = {
            "provider": provider,
            "model": model,
            "hit_ratio": round(analysis["hit_ratio"], 3),
            "n_edge": analysis["n_edge"],
            "n_judged": analysis["n_judged"],
            "hit_ratio_thr": hit_ratio_thr,
            "object_label": top_label,
            "sample_fps": round(eff_fps, 3),
            "n_frames": n,
        }

        if analysis["n_judged"] < min_judged:
            return [self._warn(
                f"未能稳定判定夹取位置（有效判定 {analysis['n_judged']} 帧 < {min_judged}），无法评估",
                {**details, "error": "too_few_judged"},
            )]

        if analysis["detected"]:
            obj = f"（物体: {top_label}）" if top_label else ""
            msg = (
                f"疑似夹取位置过于极限（夹在边缘而非主体）{obj}: "
                f"{analysis['n_edge']}/{analysis['n_judged']} 帧判定夹在边缘 "
                f"(占比 {analysis['hit_ratio']:.0%} ≥ {hit_ratio_thr:.0%})"
            )
            return [CheckResult(name=self.name, severity=fail_severity,
                                message=msg, details=details)]
        return [CheckResult(name=self.name, severity=Severity.PASS,
                            message="夹取位置正常（夹在物体主体），未见夹取过于极限", details=details)]

    def _warn(self, msg: str, details: dict[str, Any]) -> CheckResult:
        return CheckResult(name=self.name, severity=Severity.WARN, message=msg, details=details)


# ------------------------------ 纯逻辑（可单测） ------------------------------

def evaluate_edge_grasp(
    frame_verdicts: list[dict[str, Any] | None],
    *,
    hit_ratio_thr: float,
    min_confidence: float = 0.0,
) -> dict[str, Any]:
    """在逐帧「夹爪是否夹在物体边缘而非主体」判定上聚合出整段结论。

    - ``frame_verdicts[i]``：第 i 帧的判定 ``{edge: bool, confidence: float, ...}``；
      ``None`` 表示该帧模型未返回（未见夹爪夹住物体 / 看不清接触点）→ 不计入。
    - ``min_confidence > 0`` 时，置信度低于它的帧判定也跳过（不计入）。
    - 在所有**计入的帧**中，``edge=True`` 帧占比 ≥ ``hit_ratio_thr`` 即命中。

    返回 ``{detected, hit_ratio, n_edge, n_judged, score}``，``score`` 越高越好
    （= 1 - hit_ratio，越多帧夹在主体越好）。
    """
    n_judged = 0
    n_edge = 0
    for v in frame_verdicts:
        if not v:
            continue
        edge = v.get("edge")
        if edge is None:
            continue
        if min_confidence > 0 and float(v.get("confidence", 1.0)) < min_confidence:
            continue
        n_judged += 1
        if edge:
            n_edge += 1

    hit_ratio = (n_edge / n_judged) if n_judged else 0.0
    detected = n_judged >= 1 and hit_ratio >= hit_ratio_thr
    return {
        "detected": detected,
        "hit_ratio": hit_ratio,
        "n_edge": n_edge,
        "n_judged": n_judged,
        "score": 1.0 - hit_ratio,
    }
