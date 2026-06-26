"""首帧夹爪遮挡操作物品检测器（质检规范序号 15，多模态）。

规范：视频**首帧**存在夹爪（机械臂末端执行器）遮挡被操作物品，导致开局看不清目标物体
的位置 / 大小。理想的采集应让操作物体在开始时清晰可见、不被夹爪挡住。

「夹爪是否挡住被操作物体」本质是**语义识别**（先定位被操作物体，再判断它是否被夹爪 /
机械臂遮挡），纯像素算法绕不开「先定位目标物」这一步，故走多模态。与 colormatch 一样
本项是**静态属性**（看视频开局即可），不需要时序状态机：

1. 只抽取视频**开头一小段**（``head_sec``，默认 1s）的若干帧 → 让模型**逐帧**判定
   「被操作物体是否被夹爪 / 机械臂遮挡」；
2. 代码侧聚合：在能识别出被操作物体的首段帧里，「被遮挡」帧占比 ≥ 阈值即命中。

只抽首段（``sample_frames_jpeg(duration_sec=head_sec)``）既忠实于规范「首帧」语义，又比
整段抽帧省成本、避免抽到中段正常抓取时的遮挡造成误报。判定留在代码侧（命中帧占比 /
阈值），确定可复现。命中默认 WARN（人工复核提示），可在 config 改为 fail。缺 API key /
SDK / 抽帧失败 / 接口异常 → 一律降级为 WARN，不阻塞流水线。
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

_PROMPT_OCCLUSION = (
    "你是机器人采集数据的视频质检员。下面按时间顺序给出从同一段视频**开头**均匀抽取的若干帧，"
    "每帧前用 'Frame i:' 标注帧号（从 0 开始）。\n"
    "任务：逐帧判断画面中**被操作的目标物体**是否被**夹爪 / 机械臂末端执行器 / 机械臂连杆**"
    "遮挡，导致看不清该物体的位置或大小。\n"
    "【被操作物体】指机器人夹爪即将抓取 / 正在操作的目标物体，不是夹爪本身、不是机械臂，"
    "也不是无关的背景道具。\n"
    "判定规则：\n"
    "- 目标物体的主体被夹爪 / 机械臂明显挡住，轮廓 / 位置 / 大小难以看清 → "
    "gripper_occludes_object=true；\n"
    "- 目标物体完整可见、未被夹爪 / 机械臂遮挡（夹爪在旁边、上方但没挡住物体也算 false）→ "
    "gripper_occludes_object=false；\n"
    "- 用简短稳定的英文名称填 object_label（如 'red_cube'）；判断不确定时降低 confidence(0~1)。\n"
    "- 某帧根本看不到被操作物体（已出画 / 无法识别哪个是目标物，且并非被夹爪挡住）→ 该帧可省略不返回；"
    "若是因为被夹爪完全挡住而看不到，应返回 gripper_occludes_object=true。\n"
    "只对能判断的帧返回结果，index 与 'Frame i' 的 i 一致。\n"
    "{robot_hint}{task_hint}"
)


class OcclusionChecker(BaseChecker):
    """首帧夹爪遮挡操作物品检测（多模态，首段逐帧遮挡判定 + 代码侧占比聚合）。"""

    name = "occlusion"

    def check(self, path: Path, metadata: dict[str, Any]) -> list[CheckResult]:
        cfg = self.config.get("occlusion", {})
        provider = str(cfg.get("provider", "gemini")).lower()
        min_conf = float(cfg.get("min_confidence", 0.0))
        hit_ratio_thr = float(cfg.get("hit_ratio", 0.5))
        min_judged = int(cfg.get("min_judged_frames", 1))
        fail_severity = _severity(cfg.get("severity", "warn"))

        defaults = _PROVIDER_DEFAULTS.get(provider, {})
        pcfg = cfg.get(provider, {})
        model = pcfg.get("model", defaults.get("model"))
        api_key_env = pcfg.get("api_key_env", defaults.get("api_key_env", "GEMINI_API_KEY"))
        if model is None:
            return [self._warn(f"未知 provider: {provider}", {"error": "bad_provider"})]
        api_key = os.environ.get(api_key_env)
        if not api_key:
            return [self._warn(f"未设置 {api_key_env}，跳过首帧遮挡检测",
                               {"error": "missing_api_key"})]

        try:
            backend = build_backend(provider, model, api_key, cfg)
        except Exception as exc:  # noqa: BLE001
            return [self._warn(f"后端初始化失败: {exc}", {"error": "backend_init"})]

        sample_fps = float(cfg.get("sample_fps", 2.0))
        max_h = int(cfg.get("frame_max_h", 360))
        max_frames = int(cfg.get("max_frames", 8))
        head_sec = float(cfg.get("head_sec", 1.0))
        timeout = float(cfg.get("timeout", 120.0))

        # 只关心视频开头：抽取首段 head_sec 秒（不超过视频时长）。
        duration = metadata.get("duration_sec")
        eff_head = head_sec
        if duration and duration > 0:
            eff_head = min(head_sec, float(duration))
        eff_head = max(eff_head, 1e-3)

        frames, err = sample_frames_jpeg(
            str(path), sample_fps=sample_fps, max_h=max_h,
            max_frames=max_frames, timeout=timeout, duration_sec=eff_head,
        )
        if frames is None:
            return [self._warn(f"抽帧失败，跳过首帧遮挡检测: {err}", {"error": err})]

        prompt = _PROMPT_OCCLUSION.format(
            robot_hint=_robot_hint(metadata.get("robot")),
            task_hint=_task_hint(metadata),
        )
        try:
            verdicts = backend.classify_occlusion_frames(frames, prompt)
        except Exception as exc:  # noqa: BLE001 - 调用异常一律降级
            return [self._warn(f"{provider} 调用失败: {type(exc).__name__}: {exc}",
                               {"error": "backend_error"})]

        n = len(frames)
        frame_verdicts = [verdicts.get(i) for i in range(n)]
        analysis = evaluate_occlusion(
            frame_verdicts, hit_ratio_thr=hit_ratio_thr, min_confidence=min_conf
        )

        # 取「被遮挡」帧里出现最多的物体标签作复核展示。
        labels = Counter(
            v["object_label"]
            for v in frame_verdicts
            if v and v.get("occluded") and v.get("object_label")
        )
        top_label = labels.most_common(1)[0][0] if labels else ""

        details = {
            "provider": provider,
            "model": model,
            "hit_ratio": round(analysis["hit_ratio"], 3),
            "n_occluded": analysis["n_occluded"],
            "n_judged": analysis["n_judged"],
            "hit_ratio_thr": hit_ratio_thr,
            "object_label": top_label,
            "head_sec": round(eff_head, 3),
            "sample_fps": round(sample_fps, 3),
            "n_frames": n,
        }

        if analysis["n_judged"] < min_judged:
            return [self._warn(
                f"首段未能识别被操作物体（有效判定 {analysis['n_judged']} 帧 < {min_judged}），无法评估",
                {**details, "error": "too_few_judged"},
            )]

        if analysis["detected"]:
            obj = f"（物体: {top_label}）" if top_label else ""
            msg = (
                f"疑似首帧夹爪遮挡操作物品{obj}: "
                f"{analysis['n_occluded']}/{analysis['n_judged']} 首段帧判定被遮挡 "
                f"(占比 {analysis['hit_ratio']:.0%} ≥ {hit_ratio_thr:.0%})"
            )
            return [CheckResult(name=self.name, severity=fail_severity,
                                message=msg, details=details)]
        return [CheckResult(name=self.name, severity=Severity.PASS,
                            message="视频首段操作物体清晰可见，未见夹爪遮挡", details=details)]

    def _warn(self, msg: str, details: dict[str, Any]) -> CheckResult:
        return CheckResult(name=self.name, severity=Severity.WARN, message=msg, details=details)


# ------------------------------ 纯逻辑（可单测） ------------------------------

def evaluate_occlusion(
    frame_verdicts: list[dict[str, Any] | None],
    *,
    hit_ratio_thr: float,
    min_confidence: float = 0.0,
) -> dict[str, Any]:
    """在逐帧「被操作物体是否被夹爪遮挡」判定上聚合出首段结论。

    - ``frame_verdicts[i]``：第 i 帧（视频开头一段内）的判定 ``{occluded: bool, confidence: float, ...}``；
      ``None`` 表示该帧模型未返回（无可辨识的被操作物体）→ 不计入。
    - ``min_confidence > 0`` 时，置信度低于它的帧判定也跳过（不计入）。
    - 在所有**计入的帧**中，``occluded=True`` 帧占比 ≥ ``hit_ratio_thr`` 即命中
      （只抽了首段，故"多数首段帧被遮挡"即对应规范「首帧存在遮挡」）。

    返回 ``{detected, hit_ratio, n_occluded, n_judged, score}``，``score`` 越高越好
    （= 1 - hit_ratio，越不被遮挡越好）。
    """
    n_judged = 0
    n_occluded = 0
    for v in frame_verdicts:
        if not v:
            continue
        occluded = v.get("occluded")
        if occluded is None:
            continue
        if min_confidence > 0 and float(v.get("confidence", 1.0)) < min_confidence:
            continue
        n_judged += 1
        if occluded:
            n_occluded += 1

    hit_ratio = (n_occluded / n_judged) if n_judged else 0.0
    detected = n_judged >= 1 and hit_ratio >= hit_ratio_thr
    return {
        "detected": detected,
        "hit_ratio": hit_ratio,
        "n_occluded": n_occluded,
        "n_judged": n_judged,
        "score": 1.0 - hit_ratio,
    }
