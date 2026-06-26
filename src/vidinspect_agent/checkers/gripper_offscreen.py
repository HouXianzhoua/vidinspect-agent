"""夹爪出境检测器（质检规范序号 12，多模态）。

规范：夹爪（机械臂末端执行器 / gripper）持续离开相机画面 1s 以上即不合格。

纯像素算法难以稳健判定「夹爪是否在画面内」（需识别并跟踪该语义实体），故走多模态。
支持两个维度的可插拔配置（见 docs/detectors.md 的取舍说明）：

- ``mode``：``image``（默认，本地抽帧→逐帧判 visible→代码算最长连续出镜时长）或
  ``video``（整段视频交模型直接返回出镜区间，依赖原生视频理解，目前仅 gemini）。
- ``provider``：``gemini`` / ``openai``，对应 ``_vision`` 里的后端实现。

判定始终留在代码侧（连续出镜帧数 / 采样率，或区间时长），确定可复现。缺 API key /
SDK 缺失 / 抽帧失败 / 接口异常 / 后端不支持该模式 → 一律降级为 WARN，不阻塞流水线。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from vidinspect_agent.checkers._frames import sample_frames_jpeg
from vidinspect_agent.checkers._vision import VideoModeUnsupported, build_backend
from vidinspect_agent.checkers.base import BaseChecker
from vidinspect_agent.models import CheckResult, Severity

_PROVIDER_DEFAULTS = {
    "gemini": {"model": "gemini-2.5-flash", "api_key_env": "GEMINI_API_KEY"},
    "openai": {"model": "gpt-4o", "api_key_env": "OPENAI_API_KEY"},
}

_GRIPPER_DEF = (
    "【夹爪】指机械臂末端的执行器 / 抓手 / gripper / end-effector"
    "（夹持手指、吸盘或灵巧手等），不是被操作的物体，也不是机械臂的连杆。\n"
)

_PROMPT_IMAGE = (
    "你是机器人采集数据的视频质检员。下面按时间顺序给出从同一段视频均匀抽取的若干帧，"
    "每帧前用 'Frame i:' 标注帧号（从 0 开始）。\n"
    "任务：逐帧判断【夹爪】是否出现在画面内。\n" + _GRIPPER_DEF +
    "判定规则：\n"
    "- 夹爪只要有任意可见部分在画面内 → gripper_visible=true；\n"
    "- 夹爪完全移出画面边界、或被完全遮挡看不到 → gripper_visible=false；\n"
    "- 仅当确实无法判断时才标 false，并尽量在 notes 说明。\n"
    "对每一帧都必须返回一条结果，index 与 'Frame i' 的 i 一致。\n"
    "{robot_hint}"
)

_PROMPT_VIDEO = (
    "你是机器人采集数据的视频质检员。请观看这段视频，找出【夹爪】完全离开相机画面"
    "（出镜 / 完全移出画面边界）的所有时间区间。\n" + _GRIPPER_DEF +
    "对每个出镜区间给出 start_sec、end_sec（秒）与 confidence（0~1）。"
    "夹爪始终在画面内则返回空列表。\n"
    "{robot_hint}"
)


def _robot_hint(robot: Any) -> str:
    if isinstance(robot, str) and robot:
        return f"参考：本视频机器人型号为 {robot}，据此理解夹爪外形。\n"
    return ""


class GripperOffscreenChecker(BaseChecker):
    """夹爪出境检测（多模态，mode × provider 可插拔）。"""

    name = "gripper_offscreen"

    def check(self, path: Path, metadata: dict[str, Any]) -> list[CheckResult]:
        cfg = self.config.get("gripper_offscreen", {})
        mode = str(cfg.get("mode", "image")).lower()
        provider = str(cfg.get("provider", "gemini")).lower()
        min_offscreen_sec = float(cfg.get("min_offscreen_sec", 1.0))
        fail_severity = _severity(cfg.get("severity", "warn"))

        defaults = _PROVIDER_DEFAULTS.get(provider, {})
        pcfg = cfg.get(provider, {})
        model = pcfg.get("model", defaults.get("model"))
        api_key_env = pcfg.get("api_key_env", defaults.get("api_key_env", "GEMINI_API_KEY"))
        if model is None:
            return [self._warn(f"未知 provider: {provider}", {"error": "bad_provider"})]
        api_key = os.environ.get(api_key_env)
        if not api_key:
            return [self._warn(f"未设置 {api_key_env}，跳过夹爪出境检测",
                               {"error": "missing_api_key"})]

        try:
            backend = build_backend(provider, model, api_key, cfg)
        except Exception as exc:  # noqa: BLE001
            return [self._warn(f"后端初始化失败: {exc}", {"error": "backend_init"})]

        if mode == "image":
            return self._check_image(path, metadata, backend, cfg, min_offscreen_sec,
                                     fail_severity, provider, model)
        if mode == "video":
            return self._check_video(path, metadata, backend, cfg, min_offscreen_sec,
                                     fail_severity, provider, model)
        return [self._warn(f"未知 mode: {mode}", {"error": "bad_mode"})]

    # ---------------------------- image 模式 ----------------------------

    def _check_image(self, path, metadata, backend, cfg, min_offscreen_sec,
                     fail_severity, provider, model) -> list[CheckResult]:
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
            return [self._warn(f"抽帧失败，跳过夹爪出境检测: {err}", {"error": err})]

        prompt = _PROMPT_IMAGE.format(robot_hint=_robot_hint(metadata.get("robot")))
        try:
            verdicts = backend.classify_frames(frames, prompt)
        except Exception as exc:  # noqa: BLE001 - 调用异常一律降级
            return [self._warn(f"{provider} 调用失败: {type(exc).__name__}: {exc}",
                               {"error": "backend_error"})]

        n = len(frames)
        visible = [True] * n
        for idx, vis in verdicts.items():
            if 0 <= idx < n:
                visible[idx] = bool(vis)
        offscreen = [not v for v in visible]

        max_run, run_start = _longest_run(offscreen)
        max_offscreen_sec = max_run / eff_fps  # 连续 k 帧覆盖约 k/eff_fps 秒
        offscreen_ratio = (sum(offscreen) / n) if n else 0.0

        details = {
            "mode": "image",
            "provider": provider,
            "model": model,
            "max_offscreen_sec": round(max_offscreen_sec, 2),
            "max_offscreen_frames": int(max_run),
            "offscreen_start_sec": round(run_start / eff_fps, 2) if max_run else None,
            "offscreen_ratio": round(offscreen_ratio, 3),
            "min_offscreen_sec": min_offscreen_sec,
            "sample_fps": round(eff_fps, 3),
            "n_frames": n,
        }
        return self._verdict(max_offscreen_sec, min_offscreen_sec, fail_severity, details)

    # ---------------------------- video 模式 ----------------------------

    def _check_video(self, path, metadata, backend, cfg, min_offscreen_sec,
                     fail_severity, provider, model) -> list[CheckResult]:
        min_conf = float(cfg.get("min_confidence", 0.5))
        prompt = _PROMPT_VIDEO.format(robot_hint=_robot_hint(metadata.get("robot")))
        try:
            intervals = backend.detect_video_intervals(str(path), prompt)
        except VideoModeUnsupported as exc:
            return [self._warn(f"{exc}（请改用 mode=image）", {"error": "video_unsupported"})]
        except Exception as exc:  # noqa: BLE001
            return [self._warn(f"{provider} 调用失败: {type(exc).__name__}: {exc}",
                               {"error": "backend_error"})]

        spans = [(s, e, c) for (s, e, c) in intervals if c >= min_conf and e > s]
        max_offscreen_sec = max((e - s for s, e, _c in spans), default=0.0)
        start = next((s for s, e, _c in spans if (e - s) == max_offscreen_sec), None)

        details = {
            "mode": "video",
            "provider": provider,
            "model": model,
            "max_offscreen_sec": round(max_offscreen_sec, 2),
            "offscreen_start_sec": round(start, 2) if start is not None else None,
            "n_intervals": len(spans),
            "min_offscreen_sec": min_offscreen_sec,
            "min_confidence": min_conf,
        }
        return self._verdict(max_offscreen_sec, min_offscreen_sec, fail_severity, details)

    # ---------------------------- 公共判定 ----------------------------

    def _verdict(self, max_offscreen_sec, min_offscreen_sec, fail_severity,
                 details) -> list[CheckResult]:
        if max_offscreen_sec > min_offscreen_sec:
            return [
                CheckResult(
                    name=self.name,
                    severity=fail_severity,
                    message=(
                        f"疑似夹爪出境: 最长连续出镜 {max_offscreen_sec:.1f}s "
                        f"(上限 {min_offscreen_sec:.1f}s)"
                    ),
                    details=details,
                )
            ]
        return [
            CheckResult(
                name=self.name,
                severity=Severity.PASS,
                message=f"夹爪未明显出境: 最长连续出镜 {max_offscreen_sec:.1f}s",
                details=details,
            )
        ]

    def _warn(self, msg: str, details: dict[str, Any]) -> CheckResult:
        return CheckResult(
            name=self.name,
            severity=Severity.WARN,
            message=msg,
            details=details,
        )


def _longest_run(mask: list[bool]) -> tuple[int, int]:
    """返回最长连续 True 段的 (长度, 起始索引)。"""
    best_len = best_start = 0
    cur_len = cur_start = 0
    for i, v in enumerate(mask):
        if v:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
        else:
            cur_len = 0
    return best_len, best_start


def _severity(value: str) -> Severity:
    try:
        return Severity(str(value).lower())
    except ValueError:
        return Severity.WARN
