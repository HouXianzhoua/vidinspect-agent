"""LeRobot v3.0 数据集摄入 / 编排层（docs/detector_dataset_impact.md §1）。

把检测粒度从「单视频文件」抬到「LeRobot 组」：定位到含 ``meta/info.json`` 的组根目录，
逐组读取真实参数（**绝不硬编码** fps / 编码 / episode 数 / 是否有标定），把以下信息注入
每个视频的 ``metadata``，供检测器吃上多模态信号：

- ``robot``：来自 ``info.json.robot_type``（``jump`` 阈值表、各多模态检测器 robot_hint 的钩子）。
- ``task`` / ``target_objects``：来自 ``tasks.jsonl`` / ``episodes.jsonl`` / ``labels.json``
  子任务名 / ``info.json.metadata.language_instruction``（``colormatch`` task_hint 的钩子）。
- ``lerobot``：组级上下文（声明视频规格、episode 长度、标定是否齐全、对应 parquet 指针等），
  全部 JSON 可序列化，便于写入报告。

逐帧关节 / 夹爪 / 时间戳数组**不**塞进 metadata（体量大且非标量），而是通过 ``parquet_path``
暴露指针 + :func:`load_episode_frames` 按需读取，留给 §2 的关节后端检测器使用。

设计要点（docs/dataset_inputs.md §17）：

- 实际可检测 episode 数 = ``data/chunk-000/`` 下 parquet 文件数，**不是** ``total_episodes``。
- 文件名按真实 ``episode_{编号}.*`` 解析（``info.json`` 模板写的 ``file-{index}`` 与实际不符）。
- 优雅降级：标定缺失（25/500）、``key_frame`` 为空、某些机型无 ``head_position``、
  缺 pyarrow 等都要兼容，任何一步失败都不应让流水线崩溃。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# 组根目录里一定有 meta/info.json（不硬编码 lerobot_RoboMIND 名字，靠该标志识别）。
_INFO_REL = Path("meta") / "info.json"
# 从视频文件向上找组根的最大层数（cam/chunk/videos/root 通常 3 层，留足余量）。
_MAX_WALK_UP = 8
# 解析真实文件名里的 episode 编号：episode_000129.mp4 -> 129。
_EPISODE_RE = re.compile(r"episode_0*(\d+)", re.IGNORECASE)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def parse_episode_index(path: Path) -> int | None:
    """从真实文件名 ``episode_{编号}`` 解析 episode 编号；解析不出返回 ``None``。"""
    m = _EPISODE_RE.search(path.stem)
    return int(m.group(1)) if m else None


def camera_key_from_path(video_path: Path) -> str | None:
    """视频所在的相机目录名即 camera key（如 ``camera_observations.color_images.camera_left``）。"""
    parent = video_path.parent.name
    return parent or None


def find_group_root(path: Path) -> Path | None:
    """从给定路径向上回溯，定位含 ``meta/info.json`` 的 LeRobot 组根目录。

    支持传入组内任意文件 / 子目录，或直接传入组根本身。找不到返回 ``None``。
    """
    path = path.resolve()
    candidate = path if path.is_dir() else path.parent
    for _ in range(_MAX_WALK_UP + 1):
        if (candidate / _INFO_REL).is_file():
            return candidate
        if candidate.parent == candidate:
            break
        candidate = candidate.parent
    return None


def _clean(value: Any) -> Any:
    """部分 info.json 字段带前导空格（如 `` bgr``），字符串统一 strip。"""
    return value.strip() if isinstance(value, str) else value


def _declared_video_spec(info: dict[str, Any], camera_key: str | None) -> dict[str, Any]:
    """从 ``info.json.features`` 取声明视频规格（优先匹配 camera_key 对应特征）。"""
    features = info.get("features")
    if not isinstance(features, dict):
        return {}

    def _video_info(feat: Any) -> dict[str, Any] | None:
        if isinstance(feat, dict) and isinstance(feat.get("info"), dict):
            vi = feat["info"]
            if any(str(k).startswith("video.") for k in vi) or "has_audio" in vi:
                return vi
        return None

    chosen: dict[str, Any] | None = None
    if camera_key and camera_key in features:
        chosen = _video_info(features[camera_key])
    if chosen is None:
        for feat in features.values():
            chosen = _video_info(feat)
            if chosen is not None:
                break
    if chosen is None:
        return {}
    return {
        "codec": _clean(chosen.get("video.codec")),
        "height": chosen.get("video.height"),
        "width": chosen.get("video.width"),
        "fps": chosen.get("video.fps"),
        "pix_fmt": _clean(chosen.get("video.pix_fmt")),
        "channels": chosen.get("video.channels"),
        "has_audio": chosen.get("has_audio"),
        "is_depth_map": chosen.get("video.is_depth_map"),
    }


def _calibration_present(info: dict[str, Any]) -> bool:
    """标定是否齐全（475/500 组有）：内参 / 外参任一非空即视为含标定。"""
    for key in ("camera_intrinsics", "camera_extrinsics"):
        block = info.get(key)
        if isinstance(block, dict) and any(block.values()):
            return True
    return False


@dataclass
class LeRobotGroup:
    """一组 LeRobot v3.0 数据集的组级元信息（逐组真实读取）。"""

    root: Path
    info: dict[str, Any] = field(default_factory=dict)
    robot_type: str | None = None
    fps: float | None = None
    total_episodes: int | None = None
    total_tasks: int | None = None
    language_instruction: str | None = None
    calibration_present: bool = False
    # task_index -> 任务名（来自 tasks.jsonl）
    tasks: dict[int, str] = field(default_factory=dict)
    # episode_index -> {length, tasks, platform_episode_id}
    episodes: dict[int, dict[str, Any]] = field(default_factory=dict)
    # episode_index -> {subtasks, key_frame, target_objects}
    labels: dict[int, dict[str, Any]] = field(default_factory=dict)
    # episode_index -> parquet 文件路径（实际交付的才有）
    parquet_by_episode: dict[int, Path] = field(default_factory=dict)

    @property
    def delivered_episodes(self) -> int:
        """实际可检测 episode 数 = data/ 下 parquet 文件数（不是 total_episodes）。"""
        return len(self.parquet_by_episode)


def _parse_labels(raw: Any) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    items = raw.get("labels") if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        return out
    for item in items:
        if not isinstance(item, dict):
            continue
        ep = item.get("episode_index")
        if not isinstance(ep, int):
            continue
        subtasks = item.get("subtasks") or []
        target_objects = [
            str(s.get("label")).strip()
            for s in subtasks
            if isinstance(s, dict) and str(s.get("label", "")).strip()
        ]
        out[ep] = {
            "subtasks": subtasks,
            "key_frame": item.get("key_frame") or [],
            "target_objects": target_objects,
        }
    return out


def load_group(root: Path) -> LeRobotGroup:
    """读取一组数据的全部组级元信息，逐文件优雅降级（任一缺失不致命）。"""
    root = root.resolve()
    group = LeRobotGroup(root=root)

    try:
        info = _load_json(root / _INFO_REL)
    except Exception:  # noqa: BLE001 - 缺 info.json 时返回空组
        info = {}
    if isinstance(info, dict):
        group.info = info
        group.robot_type = _clean(info.get("robot_type"))
        group.fps = info.get("fps")
        group.total_episodes = info.get("total_episodes")
        group.total_tasks = info.get("total_tasks")
        meta_block = info.get("metadata")
        if isinstance(meta_block, dict):
            group.language_instruction = _clean(meta_block.get("language_instruction"))
        group.calibration_present = _calibration_present(info)

    tasks_path = root / "meta" / "tasks.jsonl"
    if tasks_path.is_file():
        for row in _load_jsonl(tasks_path):
            idx = row.get("task_index")
            name = row.get("task")
            if isinstance(idx, int) and name is not None:
                group.tasks[idx] = str(name)

    episodes_path = root / "meta" / "episodes.jsonl"
    if episodes_path.is_file():
        for row in _load_jsonl(episodes_path):
            idx = row.get("episode_index")
            if isinstance(idx, int):
                group.episodes[idx] = {
                    "length": row.get("length"),
                    "tasks": row.get("tasks") or [],
                    "platform_episode_id": row.get("platform_episode_id"),
                }

    labels_path = root / "labels" / "labels.json"
    if labels_path.is_file():
        try:
            group.labels = _parse_labels(_load_json(labels_path))
        except Exception:  # noqa: BLE001 - labels 缺失/损坏时降级为无标注
            group.labels = {}

    data_dir = root / "data"
    if data_dir.is_dir():
        for parquet in data_dir.rglob("episode_*.parquet"):
            ep = parse_episode_index(parquet)
            if ep is not None:
                group.parquet_by_episode.setdefault(ep, parquet)

    return group


def _episode_task(group: LeRobotGroup, episode_index: int | None) -> str | None:
    """组 episode 的任务描述：优先 episodes.jsonl 任务名，回退 language_instruction。"""
    ep = group.episodes.get(episode_index) if episode_index is not None else None
    if ep:
        names = [str(t).strip() for t in ep.get("tasks", []) if str(t).strip()]
        if names:
            return "、".join(names)
    if group.tasks:
        # 单任务数据集（多数情况）直接取唯一任务名。
        uniq = sorted(set(group.tasks.values()))
        if len(uniq) == 1:
            return uniq[0]
    return group.language_instruction


def build_video_metadata(group: LeRobotGroup, video_path: Path) -> dict[str, Any]:
    """把组级信息映射到单个视频，产出可注入 ``metadata`` 的（JSON 可序列化）字典。"""
    episode_index = parse_episode_index(video_path)
    camera_key = camera_key_from_path(video_path)
    parquet = group.parquet_by_episode.get(episode_index) if episode_index is not None else None
    label = group.labels.get(episode_index, {}) if episode_index is not None else {}
    ep_meta = group.episodes.get(episode_index, {}) if episode_index is not None else {}

    meta: dict[str, Any] = {}
    if group.robot_type:
        meta["robot"] = group.robot_type

    task = _episode_task(group, episode_index)
    if task:
        meta["task"] = task

    target_objects = label.get("target_objects") or []
    if target_objects:
        meta["target_objects"] = target_objects

    meta["lerobot"] = {
        "group_root": str(group.root),
        "episode_index": episode_index,
        "camera_key": camera_key,
        "episode_length": ep_meta.get("length"),
        "platform_episode_id": ep_meta.get("platform_episode_id"),
        "parquet_path": str(parquet) if parquet else None,
        "declared_video": _declared_video_spec(group.info, camera_key),
        "calibration_present": group.calibration_present,
        "language_instruction": group.language_instruction,
        "declared_fps": group.fps,
        "total_episodes": group.total_episodes,
        "delivered_episodes": group.delivered_episodes,
        "subtask_labels": target_objects,
    }
    return meta


class GroupResolver:
    """按组根缓存 :class:`LeRobotGroup`，让同组多视频（三路 × 多 episode）只加载一次。"""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._cache: dict[Path, LeRobotGroup | None] = {}

    def group_for(self, video_path: Path) -> LeRobotGroup | None:
        if not self.enabled:
            return None
        root = find_group_root(video_path)
        if root is None:
            return None
        if root not in self._cache:
            try:
                self._cache[root] = load_group(root)
            except Exception:  # noqa: BLE001 - 加载失败时缓存 None，不阻塞流水线
                self._cache[root] = None
        return self._cache[root]

    def metadata_for(self, video_path: Path) -> dict[str, Any]:
        """返回该视频的组级 metadata 注入项；非 LeRobot 组 / 关闭时返回空字典。"""
        group = self.group_for(Path(video_path))
        if group is None:
            return {}
        try:
            return build_video_metadata(group, Path(video_path))
        except Exception:  # noqa: BLE001 - 单视频映射失败不影响其纯视频检测
            return {}


def load_episode_frames(parquet_path: str | Path) -> dict[str, list[Any]]:
    """按需读取某 episode 的逐帧 parquet 数组（关节 / 夹爪 / 时间戳 / is_intervene / 帧索引）。

    供 §2 的关节后端检测器调用（``static`` 关节方差、``regrasp`` / ``object_slip`` 真实夹爪）。
    返回 ``{列名: 逐帧值列表}``。需要 ``pyarrow``（``pip install -e ".[lerobot]"``）；
    缺依赖或读取失败抛 :class:`RuntimeError`，由调用方决定降级。
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - 取决于可选依赖是否安装
        raise RuntimeError(
            "读取 parquet 需要 pyarrow，请安装：pip install -e \".[lerobot]\""
        ) from exc

    path = Path(parquet_path)
    if not path.is_file():
        raise RuntimeError(f"parquet 不存在: {path}")
    try:
        table = pq.read_table(path)
    except Exception as exc:  # noqa: BLE001 - 损坏 parquet 统一上抛
        raise RuntimeError(f"读取 parquet 失败: {path}: {exc}") from exc
    return {name: table.column(name).to_pylist() for name in table.column_names}
