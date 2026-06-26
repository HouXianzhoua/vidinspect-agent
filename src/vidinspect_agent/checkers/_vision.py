"""可插拔多模态视觉后端，供 gripper_offscreen 等多模态检测器复用。

抽象出统一接口，屏蔽不同厂商 SDK 差异，并支持两种输入模式：

- ``classify_frames``（image 模式）：收一串抽样 JPEG 帧，逐帧返回「夹爪是否在画面内」
  （``{frame_index: visible_bool}``）。任何多模态模型都能跑（gemini / openai）。
- ``detect_video_intervals``（video 模式）：收整段视频，直接返回「夹爪出镜」时间区间
  ``[(start_sec, end_sec, confidence), ...]``。依赖原生视频理解，目前仅 gemini 支持，
  openai 后端会抛 ``VideoModeUnsupported``，由上层降级为 WARN。

所有后端方法在失败时**抛异常**（不静默返回），交由检测器统一 try/except 降级为 WARN，
保持与项目其它检测器一致的优雅降级语义。
"""
from __future__ import annotations

import base64
import json
import time
from abc import ABC, abstractmethod
from typing import Any

Frame = tuple[float, bytes]  # (timestamp_sec, jpeg_bytes)
Interval = tuple[float, float, float]  # (start_sec, end_sec, confidence)


class VideoModeUnsupported(Exception):
    """后端不支持 video 模式时抛出。"""


class VisionBackend(ABC):
    """多模态视觉后端统一接口。"""

    name: str

    def __init__(self, model: str, api_key: str, cfg: dict[str, Any]) -> None:
        self.model = model
        self.api_key = api_key
        self.cfg = cfg

    @abstractmethod
    def classify_frames(self, frames: list[Frame], prompt: str) -> dict[int, bool]:
        """image 模式：逐帧判定，返回 {frame_index: gripper_visible}。"""

    def classify_grasp_frames(
        self, frames: list[Frame], prompt: str
    ) -> dict[int, list[dict[str, Any]]]:
        """逐帧、逐夹爪抓取判定：返回 {frame_index: [{side, holding, object_label, confidence}]}。

        供 regrasp（二次抓取）检测器复用。默认不支持，由子类实现。
        """
        raise NotImplementedError(f"{self.name} 不支持 classify_grasp_frames")

    def classify_colormatch_frames(
        self, frames: list[Frame], prompt: str
    ) -> dict[int, dict[str, Any]]:
        """逐帧判定「被操作物体是否与桌面同色难分辨」：

        返回 ``{frame_index: {hard, object_label, confidence}}``。供 colormatch
        （操作物与桌面颜色相同，规范19）检测器复用。默认不支持，由子类实现。
        """
        raise NotImplementedError(f"{self.name} 不支持 classify_colormatch_frames")

    def classify_edge_grasp_frames(
        self, frames: list[Frame], prompt: str
    ) -> dict[int, dict[str, Any]]:
        """逐帧判定「夹爪是否夹在物体边缘而非主体」：

        返回 ``{frame_index: {edge, object_label, confidence}}``。供 edge_grasp
        （夹取位置过于极限，规范16）检测器复用。默认不支持，由子类实现。
        """
        raise NotImplementedError(f"{self.name} 不支持 classify_edge_grasp_frames")

    def detect_video_intervals(self, video_path: str, prompt: str) -> list[Interval]:
        """video 模式：返回夹爪出镜区间。默认不支持。"""
        raise VideoModeUnsupported(f"{self.name} 不支持 video 模式")


# ---- 结构化输出 schema（image 模式逐帧 / video 模式区间）----

def _frame_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "frames": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "gripper_visible": {"type": "boolean"},
                    },
                    "required": ["index", "gripper_visible"],
                },
            },
            "notes": {"type": "string"},
        },
        "required": ["frames"],
    }


def _grasp_frame_schema() -> dict[str, Any]:
    """逐帧、逐夹爪的抓取状态（支持单臂 / 双臂，按 side 区分各机械臂）。

    ``gripper_closed`` 为可选字段：regrasp（二次抓取）不要求它，object_slip（物体滑落）
    用它区分「主动张开放下」与「夹爪仍闭合但物体没了（滑落）」。
    """
    return {
        "type": "object",
        "properties": {
            "frames": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "grippers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "side": {"type": "string"},
                                    "holding": {"type": "boolean"},
                                    "gripper_closed": {"type": "boolean"},
                                    "object_label": {"type": "string"},
                                    "confidence": {"type": "number"},
                                },
                                "required": ["side", "holding"],
                            },
                        },
                    },
                    "required": ["index", "grippers"],
                },
            },
            "notes": {"type": "string"},
        },
        "required": ["frames"],
    }


def _parse_grasp(data: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    """返回 {frame_index: [{side, holding, gripper_closed, object_label, confidence}, ...]}。

    ``gripper_closed`` 为 ``True`` / ``False`` / ``None``（模型未给出时为 None，slip 判定按未知处理）。
    """
    out: dict[int, list[dict[str, Any]]] = {}
    for fv in data.get("frames", []):
        try:
            idx = int(fv["index"])
        except (KeyError, TypeError, ValueError):
            continue
        grippers: list[dict[str, Any]] = []
        for g in fv.get("grippers", []) or []:
            closed = g.get("gripper_closed")
            grippers.append({
                "side": str(g.get("side", "") or "").strip().lower(),
                "holding": bool(g.get("holding", False)),
                "gripper_closed": None if closed is None else bool(closed),
                "object_label": str(g.get("object_label", "") or "").strip(),
                "confidence": float(g.get("confidence", 1.0) or 1.0),
            })
        out[idx] = grippers
    return out


def _colormatch_frame_schema() -> dict[str, Any]:
    """逐帧「被操作物体与桌面是否同色难分辨」判定 schema（规范19）。"""
    return {
        "type": "object",
        "properties": {
            "frames": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "hard_to_distinguish": {"type": "boolean"},
                        "object_label": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["index", "hard_to_distinguish"],
                },
            },
            "notes": {"type": "string"},
        },
        "required": ["frames"],
    }


def _parse_colormatch(data: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """返回 {frame_index: {hard, object_label, confidence}}。

    仅收录模型显式返回的帧；未返回的帧由上层按「该帧无可辨识的被操作物体」跳过。
    """
    out: dict[int, dict[str, Any]] = {}
    for fv in data.get("frames", []):
        try:
            idx = int(fv["index"])
        except (KeyError, TypeError, ValueError):
            continue
        out[idx] = {
            "hard": bool(fv.get("hard_to_distinguish", False)),
            "object_label": str(fv.get("object_label", "") or "").strip(),
            "confidence": float(fv.get("confidence", 1.0) or 1.0),
        }
    return out


def _edge_grasp_frame_schema() -> dict[str, Any]:
    """逐帧「夹爪是否夹在物体边缘而非主体」判定 schema（规范16）。"""
    return {
        "type": "object",
        "properties": {
            "frames": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "edge_grasp": {"type": "boolean"},
                        "object_label": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["index", "edge_grasp"],
                },
            },
            "notes": {"type": "string"},
        },
        "required": ["frames"],
    }


def _parse_edge_grasp(data: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """返回 {frame_index: {edge, object_label, confidence}}。

    仅收录模型显式返回的帧；未返回的帧由上层按「该帧未见夹爪夹住物体」跳过。
    """
    out: dict[int, dict[str, Any]] = {}
    for fv in data.get("frames", []):
        try:
            idx = int(fv["index"])
        except (KeyError, TypeError, ValueError):
            continue
        out[idx] = {
            "edge": bool(fv.get("edge_grasp", False)),
            "object_label": str(fv.get("object_label", "") or "").strip(),
            "confidence": float(fv.get("confidence", 1.0) or 1.0),
        }
    return out


def _interval_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "offscreen_intervals": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "start_sec": {"type": "number"},
                        "end_sec": {"type": "number"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["start_sec", "end_sec", "confidence"],
                },
            },
            "notes": {"type": "string"},
        },
        "required": ["offscreen_intervals"],
    }


# ---------------------------- Gemini ----------------------------

class GeminiBackend(VisionBackend):
    name = "gemini"

    def _client(self):
        from google import genai

        return genai.Client(api_key=self.api_key)

    def _frame_contents(self, frames: list[Frame], prompt: str) -> list[Any]:
        from google.genai import types

        contents: list[Any] = [prompt]
        for i, (_t, jpeg) in enumerate(frames):
            contents.append(f"Frame {i}:")
            contents.append(types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"))
        return contents

    def _gen_json(self, contents: list[Any], schema: dict[str, Any]) -> dict[str, Any]:
        resp = self._client().models.generate_content(
            model=self.model,
            contents=contents,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": schema,
                "temperature": float(self.cfg.get("temperature", 0.0)),
            },
        )
        return json.loads(resp.text)

    def classify_frames(self, frames: list[Frame], prompt: str) -> dict[int, bool]:
        data = self._gen_json(self._frame_contents(frames, prompt), _frame_schema())
        return {
            int(fv["index"]): bool(fv["gripper_visible"])
            for fv in data.get("frames", [])
        }

    def classify_grasp_frames(
        self, frames: list[Frame], prompt: str
    ) -> dict[int, list[dict[str, Any]]]:
        data = self._gen_json(self._frame_contents(frames, prompt), _grasp_frame_schema())
        return _parse_grasp(data)

    def classify_colormatch_frames(
        self, frames: list[Frame], prompt: str
    ) -> dict[int, dict[str, Any]]:
        data = self._gen_json(self._frame_contents(frames, prompt), _colormatch_frame_schema())
        return _parse_colormatch(data)

    def classify_edge_grasp_frames(
        self, frames: list[Frame], prompt: str
    ) -> dict[int, dict[str, Any]]:
        data = self._gen_json(self._frame_contents(frames, prompt), _edge_grasp_frame_schema())
        return _parse_edge_grasp(data)

    def detect_video_intervals(self, video_path: str, prompt: str) -> list[Interval]:
        timeout = float(self.cfg.get("timeout", 120.0))
        client = self._client()
        f = client.files.upload(file=video_path)
        t0 = time.time()
        while getattr(f.state, "name", str(f.state)) != "ACTIVE":
            if getattr(f.state, "name", str(f.state)) == "FAILED":
                raise RuntimeError("Gemini Files API 处理失败")
            if time.time() - t0 > timeout:
                raise TimeoutError(f"Files API 处理超时 > {timeout}s")
            time.sleep(3)
            f = client.files.get(name=f.name)

        data = self._gen_json([f, prompt], _interval_schema())
        return [
            (float(iv["start_sec"]), float(iv["end_sec"]), float(iv.get("confidence", 1.0)))
            for iv in data.get("offscreen_intervals", [])
        ]


# ---------------------------- OpenAI ----------------------------

class OpenAIBackend(VisionBackend):
    name = "openai"

    def _frame_content(self, frames: list[Frame], prompt: str) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for i, (_t, jpeg) in enumerate(frames):
            b64 = base64.b64encode(jpeg).decode("ascii")
            content.append({"type": "text", "text": f"Frame {i}:"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
            )
        return content

    def _gen_json(
        self, content: list[dict[str, Any]], schema: dict[str, Any], schema_name: str
    ) -> dict[str, Any]:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=float(self.cfg.get("temperature", 0.0)),
            response_format={
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema, "strict": False},
            },
        )
        return json.loads(resp.choices[0].message.content)

    def classify_frames(self, frames: list[Frame], prompt: str) -> dict[int, bool]:
        data = self._gen_json(
            self._frame_content(frames, prompt), _frame_schema(), "gripper_frames"
        )
        return {
            int(fv["index"]): bool(fv["gripper_visible"])
            for fv in data.get("frames", [])
        }

    def classify_grasp_frames(
        self, frames: list[Frame], prompt: str
    ) -> dict[int, list[dict[str, Any]]]:
        data = self._gen_json(
            self._frame_content(frames, prompt), _grasp_frame_schema(), "grasp_frames"
        )
        return _parse_grasp(data)

    def classify_colormatch_frames(
        self, frames: list[Frame], prompt: str
    ) -> dict[int, dict[str, Any]]:
        data = self._gen_json(
            self._frame_content(frames, prompt), _colormatch_frame_schema(), "colormatch_frames"
        )
        return _parse_colormatch(data)

    def classify_edge_grasp_frames(
        self, frames: list[Frame], prompt: str
    ) -> dict[int, dict[str, Any]]:
        data = self._gen_json(
            self._frame_content(frames, prompt), _edge_grasp_frame_schema(), "edge_grasp_frames"
        )
        return _parse_edge_grasp(data)

    # OpenAI 走 image 模式；video 模式沿用基类抛 VideoModeUnsupported。


_BACKENDS: dict[str, type[VisionBackend]] = {
    "gemini": GeminiBackend,
    "openai": OpenAIBackend,
}


def build_backend(provider: str, model: str, api_key: str, cfg: dict[str, Any]) -> VisionBackend:
    cls = _BACKENDS.get(provider)
    if cls is None:
        raise ValueError(f"未知 provider: {provider}（可选: {sorted(_BACKENDS)}）")
    return cls(model=model, api_key=api_key, cfg=cfg)
