"""误夹桌布检测器（质检规范序号 17，多模态）。

规范：含桌布 / 台布场景下，夹爪夹取物品时不应连带桌布一起被夹起，导致桌布变形。

「夹爪是否连带桌布被夹起 / 牵拉变形」本质是**语义识别 + 空间关系**判断——纯像素算法难以
稳健区分「正常操作桌面上的物体」与「物体连同其下桌布被一起拎起」，故走多模态。与 colormatch
一样属**静态/逐帧可判属性**，不需要时序状态机，但多一道**前置门控**：

1. 本地抽帧（少量、覆盖全片）→ 让模型**逐帧**判定两件事：(a) 该帧是否为含桌布场景；
   (b) 含桌布时夹爪是否正连带桌布被夹起 / 明显牵拉变形；
2. 代码侧聚合：先看「含桌布」帧是否足够（不足 → 本项不适用，PASS）；在含桌布帧里
   「误夹」帧占比 ≥ 阈值且达到最少命中帧数即命中。

判定留在代码侧（占比 / 阈值），确定可复现。命中默认 WARN（人工复核提示），可在 config
改为 fail。缺 API key / SDK / 抽帧失败 / 接口异常 → 一律降级为 WARN，不阻塞流水线。
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

_PROMPT_TABLECLOTH = (
    "你是机器人采集数据的视频质检员。下面按时间顺序给出从同一段视频均匀抽取的若干帧，"
    "每帧前用 'Frame i:' 标注帧号（从 0 开始）。\n"
    "背景：桌面上可能铺有【桌布 / 台布 / 软垫布料】。质检目标是检出"
    "「夹爪夹取物品时，连带下方桌布一起被夹起 / 拎起，导致桌布被牵拉、隆起或明显变形」的情况。\n"
    "任务：逐帧判断两件事：\n"
    "1. has_tablecloth：该帧画面里桌面是否铺有桌布 / 台布等布料（硬质桌面、托盘、无布料 → false）；\n"
    "2. tablecloth_caught：【仅在含桌布时判断】夹爪此刻是否正连带桌布一起被夹起 / 拎起，"
    "或桌布被明显牵拉、隆起、堆褶变形。\n"
    "判定规则：\n"
    "- 桌布被夹爪 / 被夹起的物体带起离开台面、出现明显隆起或牵拉褶皱 → tablecloth_caught=true；\n"
    "- 夹爪正常抓取桌面上的物体、桌布平整未被带动（即便夹爪贴近桌布）→ tablecloth_caught=false；\n"
    "- 无桌布场景（has_tablecloth=false）时 tablecloth_caught 一律填 false；\n"
    "- 用简短稳定的英文名称填 object_label（被操作物体，如 'red_block'）；不确定时降低 confidence(0~1)。\n"
    "- 某帧完全看不清桌面 / 夹爪（全遮挡、画面异常）→ 该帧可省略不返回。\n"
    "对每个能判断的帧返回结果，index 与 'Frame i' 的 i 一致。\n"
    "{robot_hint}{task_hint}"
)


class TableclothChecker(BaseChecker):
    """误夹桌布检测（多模态，逐帧含桌布门控 + 代码侧误夹占比聚合）。"""

    name = "tablecloth"

    def check(self, path: Path, metadata: dict[str, Any]) -> list[CheckResult]:
        cfg = self.config.get("tablecloth", {})
        provider = str(cfg.get("provider", "gemini")).lower()
        min_conf = float(cfg.get("min_confidence", 0.0))
        hit_ratio_thr = float(cfg.get("hit_ratio", 0.3))
        min_present = int(cfg.get("min_cloth_frames", 2))
        min_hit = int(cfg.get("min_hit_frames", 2))
        fail_severity = _severity(cfg.get("severity", "warn"))

        defaults = _PROVIDER_DEFAULTS.get(provider, {})
        pcfg = cfg.get(provider, {})
        model = pcfg.get("model", defaults.get("model"))
        api_key_env = pcfg.get("api_key_env", defaults.get("api_key_env", "GEMINI_API_KEY"))
        if model is None:
            return [self._warn(f"未知 provider: {provider}", {"error": "bad_provider"})]
        api_key = os.environ.get(api_key_env)
        if not api_key:
            return [self._warn(f"未设置 {api_key_env}，跳过误夹桌布检测",
                               {"error": "missing_api_key"})]

        try:
            backend = build_backend(provider, model, api_key, cfg)
        except Exception as exc:  # noqa: BLE001
            return [self._warn(f"后端初始化失败: {exc}", {"error": "backend_init"})]

        sample_fps = float(cfg.get("sample_fps", 2.0))
        max_h = int(cfg.get("frame_max_h", 360))
        max_frames = int(cfg.get("max_frames", 60))
        timeout = float(cfg.get("timeout", 120.0))

        # 误夹多发生在夹取/抬起的短窗口内，采样率略高于 colormatch 以免漏掉短暂误夹段；
        # 长视频自适应降采样保证覆盖全片且不超过 max_frames。
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
            return [self._warn(f"抽帧失败，跳过误夹桌布检测: {err}", {"error": err})]

        prompt = _PROMPT_TABLECLOTH.format(
            robot_hint=_robot_hint(metadata.get("robot")),
            task_hint=_task_hint(metadata),
        )
        try:
            verdicts = backend.classify_tablecloth_frames(frames, prompt)
        except Exception as exc:  # noqa: BLE001 - 调用异常一律降级
            return [self._warn(f"{provider} 调用失败: {type(exc).__name__}: {exc}",
                               {"error": "backend_error"})]

        n = len(frames)
        frame_verdicts = [verdicts.get(i) for i in range(n)]
        analysis = evaluate_tablecloth(
            frame_verdicts,
            hit_ratio_thr=hit_ratio_thr,
            min_hit_frames=min_hit,
            min_confidence=min_conf,
        )

        labels = Counter(
            v["object_label"]
            for v in frame_verdicts
            if v and v.get("has_cloth") and v.get("caught") and v.get("object_label")
        )
        top_label = labels.most_common(1)[0][0] if labels else ""

        details = {
            "provider": provider,
            "model": model,
            "hit_ratio": round(analysis["hit_ratio"], 3),
            "n_caught": analysis["n_caught"],
            "n_cloth": analysis["n_cloth"],
            "hit_ratio_thr": hit_ratio_thr,
            "min_hit_frames": min_hit,
            "object_label": top_label,
            "sample_fps": round(eff_fps, 3),
            "n_frames": n,
        }

        # 前置门控：含桌布帧不足 → 本项不适用，判 PASS（非降级）。
        if analysis["n_cloth"] < min_present:
            return [CheckResult(
                name=self.name, severity=Severity.PASS,
                message=f"未检出含桌布场景（含桌布 {analysis['n_cloth']} 帧 < {min_present}），本项不适用",
                details=details,
            )]

        if analysis["detected"]:
            obj = f"（物体: {top_label}）" if top_label else ""
            msg = (
                f"疑似操作时误夹桌布致其变形{obj}: "
                f"{analysis['n_caught']}/{analysis['n_cloth']} 含桌布帧判定误夹 "
                f"(占比 {analysis['hit_ratio']:.0%} ≥ {hit_ratio_thr:.0%})"
            )
            return [CheckResult(name=self.name, severity=fail_severity,
                                message=msg, details=details)]
        return [CheckResult(name=self.name, severity=Severity.PASS,
                            message="含桌布场景下未见夹爪误夹桌布", details=details)]

    def _warn(self, msg: str, details: dict[str, Any]) -> CheckResult:
        return CheckResult(name=self.name, severity=Severity.WARN, message=msg, details=details)


# ------------------------------ 纯逻辑（可单测） ------------------------------

def evaluate_tablecloth(
    frame_verdicts: list[dict[str, Any] | None],
    *,
    hit_ratio_thr: float,
    min_hit_frames: int = 2,
    min_confidence: float = 0.0,
) -> dict[str, Any]:
    """在逐帧「含桌布 + 是否误夹桌布」判定上聚合出整段结论。

    - ``frame_verdicts[i]``：第 i 帧判定 ``{has_cloth: bool, caught: bool, confidence: float}``；
      ``None`` 表示该帧模型未返回（无可判定信息）→ 不计入。
    - ``min_confidence > 0`` 时，置信度低于它的帧也跳过（不计入）。
    - 仅在 ``has_cloth=True`` 的帧（``n_cloth``）里统计误夹占比；``caught=True`` 计入 ``n_caught``。
    - 命中条件：``n_caught >= min_hit_frames`` 且 ``hit_ratio = n_caught / n_cloth >= hit_ratio_thr``。
      （含桌布帧是否足以判定「场景是否适用」由上层门控，这里只在有含桌布帧时给出占比结论。）

    返回 ``{detected, hit_ratio, n_caught, n_cloth, score}``，``score`` 越高越好（= 1 - hit_ratio）。
    """
    n_cloth = 0
    n_caught = 0
    for v in frame_verdicts:
        if not v:
            continue
        has_cloth = v.get("has_cloth")
        if not has_cloth:
            continue
        if min_confidence > 0 and float(v.get("confidence", 1.0)) < min_confidence:
            continue
        n_cloth += 1
        if v.get("caught"):
            n_caught += 1

    hit_ratio = (n_caught / n_cloth) if n_cloth else 0.0
    detected = (
        n_cloth >= 1
        and n_caught >= min_hit_frames
        and hit_ratio >= hit_ratio_thr
    )
    return {
        "detected": detected,
        "hit_ratio": hit_ratio,
        "n_caught": n_caught,
        "n_cloth": n_cloth,
        "score": 1.0 - hit_ratio,
    }
