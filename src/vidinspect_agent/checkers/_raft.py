"""RAFT 稠密光流模型加载与推理（移植自 video_quality_pipeline 的
SharedModelManager._load_raft 与 StaticDetector._raft_flows）。

模型按 (repo, weights, device) 缓存，进程内只加载一次。所有重依赖
（torch / easydict / RAFT 源码）均在函数内惰性 import，未安装时由上层
捕获为 warn，不影响纯 CPU 的 lite 后端。
"""
from __future__ import annotations

import os
import sys
import threading
from typing import Any

_RAFT_CACHE: dict[tuple, Any] = {}
_RAFT_LOCK = threading.Lock()


def resolve_repo(config: dict) -> str:
    """RAFT 源码路径优先级，与 devbox 实现一致：
    config['raft_repo'] > $RAFT_DIR > $WORLDARENA_DIR/WorldArena/third_party/RAFT
    > /media/damoxing/sunyifan/extra_models/RAFT
    """
    repo = (
        config.get("raft_repo")
        or os.environ.get("RAFT_DIR")
        or os.path.join(
            os.environ.get(
                "WORLDARENA_DIR",
                "/media/datasets/OminiEWM_Data/benchmark/WorldArena_models",
            ),
            "WorldArena", "third_party", "RAFT",
        )
    )
    if not os.path.isdir(repo):
        alt = "/media/damoxing/sunyifan/extra_models/RAFT"
        if os.path.isdir(alt):
            repo = alt
    return repo


def resolve_model_path(config: dict, repo: str) -> str:
    return config.get("raft_model_path") or os.path.join(
        repo, "models", "raft-sintel.pth"
    )


def load_raft(config: dict, device) -> dict:
    """加载（并缓存）RAFT 模型。权重缺失抛 FileNotFoundError，torch/源码缺失抛
    ImportError，均由调用方转成 warn 结果。"""
    repo = resolve_repo(config)
    model_path = resolve_model_path(config, repo)
    key = (repo, model_path, str(device))
    cached = _RAFT_CACHE.get(key)
    if cached is not None:
        return cached
    with _RAFT_LOCK:
        cached = _RAFT_CACHE.get(key)
        if cached is not None:
            return cached

        import torch

        for p in [repo, os.path.join(repo, "core")]:
            if os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"RAFT weights not found: {model_path}")

        from core.raft import RAFT as RAFTModel
        from easydict import EasyDict as edict

        args = edict({"model": model_path, "small": False,
                      "mixed_precision": True, "alternate_corr": False})
        model = RAFTModel(args)
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        new_ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
        model.load_state_dict(new_ckpt)
        model = model.to(device).eval()
        _RAFT_CACHE[key] = {"model": model}
        return _RAFT_CACHE[key]


def compute_raft_flows(frames, model, device, iters: int = 12, batch_size: int = 40):
    """逐相邻帧对计算光流，返回 list[Tensor[2,H,W]]（CPU）。OOM 自动减半 batch。"""
    import torch
    import torch.nn.functional as F

    iters = int(iters)
    batch_size = int(batch_size)
    T = frames.shape[0]
    if T < 2:
        return []

    def pad8(x):
        _, _, h, w = x.shape
        ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        return x

    starts = list(range(0, T - 1))
    flows = []
    cur_bs = batch_size
    with torch.no_grad():
        i = 0
        while i < len(starts):
            chunk = starts[i:i + cur_bs]
            idx1 = torch.tensor(chunk, dtype=torch.long)
            imgs1 = pad8(frames.index_select(0, idx1).contiguous().to(device))
            imgs2 = pad8(frames.index_select(0, idx1 + 1).contiguous().to(device))
            try:
                _, flow_up = model(imgs1, imgs2, iters=iters, test_mode=True)
                fc = flow_up.float().cpu()
                for j in range(fc.shape[0]):
                    flows.append(fc[j])
                i += cur_bs
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and cur_bs > 1:
                    cur_bs = max(1, cur_bs // 2)
                    torch.cuda.empty_cache()
                else:
                    raise
    return flows
