"""Shared video sampling / probing helpers for the temporal checkers.

移植自 video_quality_pipeline/detectors.py 的通用读帧与 fps 探测逻辑，
供 static / dup_frame / jump 三个时序质检器复用。
"""
from __future__ import annotations

import subprocess
import time
from typing import Optional

import numpy as np


def stream_gray_diffs(
    video_path: str,
    downscale: tuple[int, int] = (64, 48),
    timeout: float = 60.0,
):
    """顺序解码整段视频为下采样灰度 raw 流，返回相邻帧 mean|ΔY| 序列。

    与 dup_frame 一致用 ffmpeg 把每帧缩成 ``downscale`` 灰度（默认 64×48），
    按原始播放顺序逐帧计算 ``mean(|Y_i - Y_{i-1}|)``（0–255 量纲）。适用于需要
    *连续帧序*（而非均匀采样）的检测器：开头/结尾静止时长、长时间卡帧。

    返回 ``(diffs, n_frames, err)``：

    - 成功：``diffs`` 为长度 ``n_frames-1`` 的 ``float64`` ndarray，``err=None``。
    - 失败 / 超时 / 帧数不足：``diffs=None``，``err`` 为简短错误标识。

    不抛异常，便于上层优雅降级为 WARN。
    """
    w, h = downscale
    chunk = w * h
    t0 = time.time()
    try:
        proc = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-i", video_path,
             "-vf", f"scale={w}:{h},format=gray", "-f", "rawvideo", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10 ** 7,
        )
    except Exception as exc:  # noqa: BLE001
        return None, 0, f"{type(exc).__name__}: {exc}"

    prev = None
    n = 0
    diffs: list[float] = []
    try:
        while True:
            if time.time() - t0 > timeout:
                proc.kill()
                proc.wait()
                return None, n, f"timeout>{timeout}s"
            buf = proc.stdout.read(chunk)
            if len(buf) < chunk:
                break
            g = np.frombuffer(buf, dtype=np.uint8).reshape(h, w).astype(np.int16)
            if prev is not None:
                diffs.append(float(np.mean(np.abs(g - prev))))
            prev = g
            n += 1
        proc.wait(timeout=5)
    except Exception as exc:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:
            pass
        return None, n, f"{type(exc).__name__}: {exc}"

    if n < 2:
        return None, n, f"too_few_frames={n}"
    return np.asarray(diffs, dtype=np.float64), n, None


def probe_fps(video_path: str, timeout: float = 15.0) -> Optional[float]:
    """用 ffprobe 读取平均帧率（r_frame_rate）。失败返回 None。

    dup_frame 的阈值是在 20fps 数据上标定的；帧率越高，相邻帧天然越相似，
    需据此归一化阈值，避免把高帧率静态场景误判成"时间变慢"。
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate", "-of",
             "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, timeout=timeout,
        ).stdout.strip()
        if not out:
            return None
        if "/" in out:
            num, den = out.split("/")
            den = float(den)
            if den == 0:
                return None
            return float(num) / den
        return float(out)
    except Exception:
        return None


def _sample_indices(total: int, n_frames: int):
    """均匀采样帧索引（与 video_quality_pipeline 一致）。"""
    if total <= 0:
        return []
    if total <= 50:
        n = min(n_frames, total)
        return list(np.round(np.linspace(0, total - 1, n)).astype(int))
    n = min(n_frames, total)
    intervals = np.linspace(0, total, n + 1, dtype=int)
    return [(intervals[i] + intervals[i + 1] - 1) // 2 for i in range(n)]


def read_frames_gray(video_path: str, n_frames: int = 32, max_h: int = 240):
    """均匀采样 n_frames 帧，返回灰度数组 [T, H, W] float32（static lite 用）。"""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    indices = _sample_indices(total, n_frames)
    grays = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        h, w = frame.shape[:2]
        if h > max_h:
            frame = cv2.resize(frame, (max(1, int(w * max_h / h)), max_h),
                               interpolation=cv2.INTER_LINEAR)
        grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32))
    cap.release()
    if len(grays) < 2:
        return None
    return np.stack(grays, axis=0)


def _read_frames_rgb_tensor_decord(video_path: str, n_frames: int, max_h: int):
    """用 decord 批量取帧并在解码阶段 resize；失败时抛异常给上层 fallback。"""
    import cv2
    import decord
    import torch

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total <= 1 or w <= 0 or h <= 0:
        return None

    if h > max_h:
        target_h = max_h
        target_w = max(1, int(round(w * max_h / h)))
    else:
        target_h, target_w = h, w

    decord.bridge.set_bridge("native")
    if target_h != h:
        vr = decord.VideoReader(video_path, num_threads=2,
                                width=target_w, height=target_h)
    else:
        vr = decord.VideoReader(video_path, num_threads=2)
    n = len(vr)
    if n <= 1:
        return None
    indices = _sample_indices(n, n_frames)
    if len(indices) < 2:
        return None
    arr = vr.get_batch(indices).asnumpy().astype(np.float32)  # RGB [T,H,W,3]
    if arr.shape[0] < 2:
        return None
    return torch.from_numpy(arr).permute(0, 3, 1, 2)


def _read_frames_rgb_tensor_cv2(video_path: str, n_frames: int, max_h: int):
    import cv2
    import torch

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    indices = _sample_indices(total, n_frames)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if h > max_h:
            rgb = cv2.resize(rgb, (max(1, int(w * max_h / h)), max_h),
                             interpolation=cv2.INTER_LINEAR)
        frames.append(rgb)
    cap.release()
    if len(frames) < 2:
        return None
    arr = np.stack(frames, axis=0).astype(np.float32)
    return torch.from_numpy(arr).permute(0, 3, 1, 2)


def read_frames_rgb_tensor(video_path: str, n_frames: int = 40, max_h: int = 240):
    """采样帧返回 torch [T, C, H, W] float32 [0,255]（RAFT 静态用）。

    优先 decord（解码阶段 resize，更快）；缺失或解码失败时回退纯 cv2，
    避免对少量坏编码 mp4 误报 read_failed。
    """
    try:
        return _read_frames_rgb_tensor_decord(video_path, n_frames, max_h)
    except ImportError:
        return _read_frames_rgb_tensor_cv2(video_path, n_frames, max_h)
    except Exception:
        return _read_frames_rgb_tensor_cv2(video_path, n_frames, max_h)
