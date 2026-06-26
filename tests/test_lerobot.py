import json

from vidinspect_agent.lerobot import (
    GroupResolver,
    build_video_metadata,
    camera_key_from_path,
    find_group_root,
    load_group,
    parse_episode_index,
)


def _make_group(root, *, robot="tienkung_station_dualArm-gripper", with_calib=True):
    """构造一个最小但结构真实的 LeRobot v3.0 组目录。"""
    meta = root / "meta"
    meta.mkdir(parents=True)
    cam_key = "camera_observations.color_images.camera_left"
    info = {
        "codebase_version": "v3.0",
        "robot_type": robot,
        "fps": 30,
        "total_episodes": 134,
        "total_tasks": 1,
        "metadata": {"language_instruction": "tidy up the books"},
        "features": {
            cam_key: {
                "dtype": "video",
                "info": {
                    "video.codec": "h264",
                    "video.height": 480,
                    "video.width": 640,
                    "video.fps": 30,
                    "video.pix_fmt": "yuv420p",
                    "video.channels": 3,
                    "has_audio": False,
                    "video.is_depth_map": False,
                },
            }
        },
    }
    if with_calib:
        info["camera_intrinsics"] = {"camera_left": {"matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}}
    (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (meta / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "test01"}) + "\n", encoding="utf-8"
    )
    (meta / "episodes.jsonl").write_text(
        json.dumps(
            {"episode_index": 129, "tasks": ["test01"], "length": 2940,
             "platform_episode_id": "pid-xyz"}
        )
        + "\n",
        encoding="utf-8",
    )

    labels_dir = root / "labels"
    labels_dir.mkdir()
    labels = {
        "labels": [
            {
                "episode_index": 129,
                "key_frame": [],
                "subtasks": [
                    {"start_frame": 0, "end_frame": 284, "label": "移动右臂抓取书籍"},
                    {"start_frame": 285, "end_frame": 600, "label": "机械臂归位"},
                ],
            }
        ]
    }
    (labels_dir / "labels.json").write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")

    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    (data_dir / "episode_000129.parquet").write_bytes(b"PAR1")

    video_dir = root / "videos" / "chunk-000" / cam_key
    video_dir.mkdir(parents=True)
    video = video_dir / "episode_000129.mp4"
    video.write_bytes(b"fake")
    return video


def test_parse_episode_index():
    from pathlib import Path

    assert parse_episode_index(Path("episode_000129.mp4")) == 129
    assert parse_episode_index(Path("episode_7.parquet")) == 7
    assert parse_episode_index(Path("clip.mp4")) is None


def test_camera_key_from_path():
    from pathlib import Path

    p = Path("/d/videos/chunk-000/camera_observations.color_images.camera_left/episode_000129.mp4")
    assert camera_key_from_path(p) == "camera_observations.color_images.camera_left"


def test_find_group_root_walks_up(tmp_path):
    video = _make_group(tmp_path / "grp")
    root = find_group_root(video)
    assert root == (tmp_path / "grp").resolve()


def test_find_group_root_none_for_plain_video(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    assert find_group_root(video) is None


def test_load_group_reads_real_params(tmp_path):
    _make_group(tmp_path / "grp")
    group = load_group(tmp_path / "grp")
    assert group.robot_type == "tienkung_station_dualArm-gripper"
    assert group.fps == 30
    assert group.total_episodes == 134
    assert group.tasks == {0: "test01"}
    assert group.episodes[129]["length"] == 2940
    assert group.labels[129]["target_objects"] == ["移动右臂抓取书籍", "机械臂归位"]
    assert group.calibration_present is True
    # 实际可检测 episode 数 = parquet 文件数，不是 total_episodes。
    assert group.delivered_episodes == 1
    assert group.total_episodes == 134


def test_build_video_metadata_fills_hooks(tmp_path):
    video = _make_group(tmp_path / "grp")
    group = load_group(tmp_path / "grp")
    meta = build_video_metadata(group, video)
    # jump 的 robot 钩子
    assert meta["robot"] == "tienkung_station_dualArm-gripper"
    # colormatch 的 task/target_objects 钩子
    assert meta["task"] == "test01"
    assert "移动右臂抓取书籍" in meta["target_objects"]
    lr = meta["lerobot"]
    assert lr["episode_index"] == 129
    assert lr["camera_key"].endswith("camera_left")
    assert lr["episode_length"] == 2940
    assert lr["declared_video"]["codec"] == "h264"
    assert lr["declared_video"]["fps"] == 30
    assert lr["calibration_present"] is True
    assert lr["parquet_path"].endswith("episode_000129.parquet")


def test_metadata_is_json_serializable(tmp_path):
    video = _make_group(tmp_path / "grp")
    group = load_group(tmp_path / "grp")
    meta = build_video_metadata(group, video)
    json.dumps(meta)  # 不应抛异常（报告会序列化 metadata）


def test_calibration_missing_degrades(tmp_path):
    _make_group(tmp_path / "grp", with_calib=False)
    group = load_group(tmp_path / "grp")
    assert group.calibration_present is False


def test_resolver_caches_group(tmp_path):
    video = _make_group(tmp_path / "grp")
    resolver = GroupResolver(enabled=True)
    g1 = resolver.group_for(video)
    g2 = resolver.group_for(video)
    assert g1 is g2  # 同组多视频只加载一次


def test_resolver_disabled_returns_empty(tmp_path):
    video = _make_group(tmp_path / "grp")
    resolver = GroupResolver(enabled=False)
    assert resolver.metadata_for(video) == {}


def test_resolver_plain_video_returns_empty(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    resolver = GroupResolver(enabled=True)
    assert resolver.metadata_for(video) == {}
