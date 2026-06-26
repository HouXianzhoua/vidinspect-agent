"""操作物与桌面颜色相同检测器（质检规范序号 19，多模态）。

规范：操作物品与桌面颜色（及纹理）过于接近，很难分辨出操作物品的位置和大小。

「物体与桌面是否同色难分辨」本质是**语义识别 + 主观可辨识度**判断——纯像素算法绕不开
「先定位被操作物体」这一语义步骤，不稳健，故走多模态。与 regrasp / object_slip 不同，
本项是**静态属性**（整段视频一个结论），不需要时序状态机：

1. 本地抽帧（少量、可覆盖全片）→ 让多模态模型**逐帧**判定「被操作物体是否与桌面同色难分辨」；
2. 代码侧聚合：在能识别出被操作物体的帧里，「难分辨」帧占比 ≥ 阈值即命中。

判定留在代码侧（命中帧占比 / 阈值），确定可复现。因「难分辨」带主观性，命中默认 WARN
（人工复核提示），可在 config 改为 fail。缺 API key / SDK / 抽帧失败 / 接口异常 →
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
from vidinspect_agent.checkers.regrasp import _severity
from vidinspect_agent.models import CheckResult, Severity

_PROVIDER_DEFAULTS = {
    "gemini": {"model": "gemini-2.5-flash", "api_key_env": "GEMINI_API_KEY"},
    "openai": {"model": "gpt-4o", "api_key_env": "OPENAI_API_KEY"},
}

_PROMPT_COLORMATCH = (
    "你是机器人采集数据的视频质检员。下面按时间顺序给出从同一段视频均匀抽取的若干帧，"
    "每帧前用 'Frame i:' 标注帧号（从 0 开始）。\n"
    "任务：逐帧判断画面中**被操作的物体**与其所在的**桌面 / 台面背景**是否因颜色（及纹理）"
    "过于接近，导致难以分辨该物体的位置和大小。\n"
    "【被操作物体】指机器人夹爪要抓取 / 正在操作的目标物体，不是夹爪本身、不是机械臂连杆、"
    "也不是无关的背景道具。\n"
    "判定规则：\n"
    "- 物体与桌面颜色 / 明暗 / 纹理高度接近，轮廓难以分辨、位置或大小看不清 → "
    "hard_to_distinguish=true；\n"
    "- 物体与桌面有明显颜色 / 明暗 / 纹理对比，轮廓清晰可辨 → hard_to_distinguish=false；\n"
    "- 用简短稳定的英文名称填 object_label（如 'white_bowl'）；判断不确定时降低 confidence(0~1)。\n"
    "- 某帧看不到明显的被操作物体（已被夹爪完全遮挡 / 已出画 / 无法识别）→ 该帧可省略不返回。\n"
    "只对能识别出被操作物体的帧返回结果，index 与 'Frame i' 的 i 一致。\n"
    "{robot_hint}{task_hint}"
)


def _robot_hint(robot: Any) -> str:
    if isinstance(robot, str) and robot:
        return f"参考：本视频机器人型号为 {robot}，据此理解夹爪外形与被操作物体。\n"
    return ""


def _task_hint(metadata: dict[str, Any]) -> str:
    """从 metadata 里抽取「被操作物体 / 任务描述」提示，帮模型定位被操作物。

    前向兼容钩子：当 LeRobot 摄入层把 ``target_objects``（来自 labels.json 子任务名）
    或 ``task``（来自 tasks.jsonl / info.json.metadata.language_instruction）填入 metadata
    时自动生效；缺省（当前 pipeline）则返回空串，不影响纯视频检测。
    """
    parts: list[str] = []

    objects = metadata.get("target_objects")
    names = ""
    if isinstance(objects, (list, tuple, set)):
        names = "、".join(str(o).strip() for o in objects if str(o).strip())
    elif isinstance(objects, str):
        names = objects.strip()
    if names:
        parts.append(f"本视频被操作的目标物体包括：{names}。请据此定位被操作物体。")

    task = metadata.get("task")
    if isinstance(task, str) and task.strip():
        parts.append(f"任务描述：{task.strip()}。可据此推断被操作物体。")

    if not parts:
        return ""
    return "参考信息：" + " ".join(parts) + "\n"


class ColorMatchChecker(BaseChecker):
    """操作物与桌面颜色相同检测（多模态，逐帧可辨识度 + 代码侧占比聚合）。"""

    name = "colormatch"

    def check(self, path: Path, metadata: dict[str, Any]) -> list[CheckResult]:
        cfg = self.config.get("colormatch", {})
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
            return [self._warn(f"未设置 {api_key_env}，跳过操作物同色检测",
                               {"error": "missing_api_key"})]

        try:
            backend = build_backend(provider, model, api_key, cfg)
        except Exception as exc:  # noqa: BLE001
            return [self._warn(f"后端初始化失败: {exc}", {"error": "backend_init"})]

        sample_fps = float(cfg.get("sample_fps", 1.0))
        max_h = int(cfg.get("frame_max_h", 360))
        max_frames = int(cfg.get("max_frames", 16))
        timeout = float(cfg.get("timeout", 120.0))

        # 静态属性，少量均匀抽帧即可；长视频自适应降采样保证覆盖全片且不超过 max_frames。
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
            return [self._warn(f"抽帧失败，跳过操作物同色检测: {err}", {"error": err})]

        prompt = _PROMPT_COLORMATCH.format(
            robot_hint=_robot_hint(metadata.get("robot")),
            task_hint=_task_hint(metadata),
        )
        try:
            verdicts = backend.classify_colormatch_frames(frames, prompt)
        except Exception as exc:  # noqa: BLE001 - 调用异常一律降级
            return [self._warn(f"{provider} 调用失败: {type(exc).__name__}: {exc}",
                               {"error": "backend_error"})]

        n = len(frames)
        frame_verdicts = [verdicts.get(i) for i in range(n)]
        analysis = evaluate_colormatch(
            frame_verdicts, hit_ratio_thr=hit_ratio_thr, min_confidence=min_conf
        )

        # 取「难分辨」帧里出现最多的物体标签作复核展示。
        labels = Counter(
            v["object_label"]
            for v in frame_verdicts
            if v and v.get("hard") and v.get("object_label")
        )
        top_label = labels.most_common(1)[0][0] if labels else ""

        details = {
            "provider": provider,
            "model": model,
            "hit_ratio": round(analysis["hit_ratio"], 3),
            "n_hard": analysis["n_hard"],
            "n_judged": analysis["n_judged"],
            "hit_ratio_thr": hit_ratio_thr,
            "object_label": top_label,
            "sample_fps": round(eff_fps, 3),
            "n_frames": n,
        }

        if analysis["n_judged"] < min_judged:
            return [self._warn(
                f"未能稳定识别被操作物体（有效判定 {analysis['n_judged']} 帧 < {min_judged}），无法评估",
                {**details, "error": "too_few_judged"},
            )]

        if analysis["detected"]:
            obj = f"（物体: {top_label}）" if top_label else ""
            msg = (
                f"疑似操作物与桌面同色难分辨{obj}: "
                f"{analysis['n_hard']}/{analysis['n_judged']} 帧判定难分辨 "
                f"(占比 {analysis['hit_ratio']:.0%} ≥ {hit_ratio_thr:.0%})"
            )
            return [CheckResult(name=self.name, severity=fail_severity,
                                message=msg, details=details)]
        return [CheckResult(name=self.name, severity=Severity.PASS,
                            message="操作物与桌面色彩对比清晰，未见同色难分辨", details=details)]

    def _warn(self, msg: str, details: dict[str, Any]) -> CheckResult:
        return CheckResult(name=self.name, severity=Severity.WARN, message=msg, details=details)


# ------------------------------ 纯逻辑（可单测） ------------------------------

def evaluate_colormatch(
    frame_verdicts: list[dict[str, Any] | None],
    *,
    hit_ratio_thr: float,
    min_confidence: float = 0.0,
) -> dict[str, Any]:
    """在逐帧「被操作物体是否与桌面同色难分辨」判定上聚合出整段结论。

    - ``frame_verdicts[i]``：第 i 帧的判定 ``{hard: bool, confidence: float, ...}``；
      ``None`` 表示该帧模型未返回（无可辨识的被操作物体）→ 不计入。
    - ``min_confidence > 0`` 时，置信度低于它的帧判定也跳过（不计入）。
    - 在所有**计入的帧**中，``hard=True`` 帧占比 ≥ ``hit_ratio_thr`` 即命中。

    返回 ``{detected, hit_ratio, n_hard, n_judged, score}``，``score`` 越高越好
    （= 1 - hit_ratio，越容易分辨越好）。
    """
    n_judged = 0
    n_hard = 0
    for v in frame_verdicts:
        if not v:
            continue
        hard = v.get("hard")
        if hard is None:
            continue
        if min_confidence > 0 and float(v.get("confidence", 1.0)) < min_confidence:
            continue
        n_judged += 1
        if hard:
            n_hard += 1

    hit_ratio = (n_hard / n_judged) if n_judged else 0.0
    detected = n_judged >= 1 and hit_ratio >= hit_ratio_thr
    return {
        "detected": detected,
        "hit_ratio": hit_ratio,
        "n_hard": n_hard,
        "n_judged": n_judged,
        "score": 1.0 - hit_ratio,
    }
