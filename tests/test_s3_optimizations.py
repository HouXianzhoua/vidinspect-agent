"""§3「可小优化的检测器（6）」单测（docs/detector_dataset_impact.md §3）。

覆盖：摄入层新增注入（stats 像素基线 / 子任务帧段）、metadata 声明-vs-实测、
dup_frame 声明 fps、endpoint_static / freeze 的关节交叉验证纯逻辑、brightness 基线、
jump 关节钩子。重 IO（解码视频 / 读真实 parquet）的部分以纯函数 + 合成数据覆盖。
"""
import json

import numpy as np

from vidinspect_agent.checkers._joints import (
    camera_side,
    is_wrist_camera,
    joint_endpoint_static_seconds,
    joint_moving_in_fraction,
    leading_static_frames,
    per_frame_speed,
    trailing_static_frames,
)
from vidinspect_agent.checkers.brightness import _pixel_luma_baseline
from vidinspect_agent.checkers.dup_frame import _declared_fps
from vidinspect_agent.checkers.endpoint_static import _trailing_homing_label
from vidinspect_agent.checkers.metadata import MetadataChecker
from vidinspect_agent.lerobot import build_video_metadata, load_group, pixel_luma_baseline


# --------------------------------------------------------------------------- #
# 摄入层：stats 像素基线 + 子任务帧段注入
# --------------------------------------------------------------------------- #
def _make_group_with_stats(root, *, mean=((0.5,), (0.4,), (0.3,))):
    meta = root / "meta"
    meta.mkdir(parents=True)
    cam_key = "camera_observations.color_images.camera_left"
    info = {
        "codebase_version": "v3.0",
        "robot_type": "tienkung_station_dualArm-gripper",
        "fps": 30,
        "features": {
            cam_key: {
                "dtype": "video",
                "info": {"video.codec": "h264", "video.height": 480, "video.width": 640,
                         "video.fps": 30, "video.pix_fmt": "yuv420p", "has_audio": False},
            }
        },
    }
    (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (meta / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 129, "tasks": ["test01"], "length": 600}) + "\n",
        encoding="utf-8",
    )
    stats = {cam_key: {"mean": [list(c) for c in mean]}}
    (meta / "stats.json").write_text(json.dumps(stats), encoding="utf-8")

    labels_dir = root / "labels"
    labels_dir.mkdir()
    labels = {"labels": [{"episode_index": 129, "key_frame": [], "subtasks": [
        {"start_frame": 0, "end_frame": 284, "label": "移动右臂抓取书籍"},
        {"start_frame": 285, "end_frame": 600, "label": "机械臂归位"},
    ]}]}
    (labels_dir / "labels.json").write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")

    video_dir = root / "videos" / "chunk-000" / cam_key
    video_dir.mkdir(parents=True)
    video = video_dir / "episode_000129.mp4"
    video.write_bytes(b"fake")
    return video


def test_pixel_luma_baseline_normalized_to_255():
    # 通道均值 [0.5,0.4,0.3] → 平均 0.4 → ×255 ≈ 102。
    stats = {"cam": {"mean": [[0.5], [0.4], [0.3]]}}
    base = pixel_luma_baseline(stats, "cam")
    assert base is not None and abs(base - 0.4 * 255.0) < 1e-6


def test_pixel_luma_baseline_already_0_255():
    stats = {"cam": {"mean": [120.0, 110.0, 100.0]}}
    base = pixel_luma_baseline(stats, "cam")
    assert abs(base - 110.0) < 1e-6


def test_pixel_luma_baseline_missing_returns_none():
    assert pixel_luma_baseline({}, "cam") is None
    assert pixel_luma_baseline({"cam": {}}, "cam") is None
    assert pixel_luma_baseline({"cam": {"mean": []}}, "cam") is None


def test_build_video_metadata_injects_baseline_and_subtasks(tmp_path):
    video = _make_group_with_stats(tmp_path / "grp")
    group = load_group(tmp_path / "grp")
    meta = build_video_metadata(group, video)
    lr = meta["lerobot"]
    assert abs(lr["pixel_luma_baseline"] - 0.4 * 255.0) < 1e-6
    assert len(lr["subtasks"]) == 2
    assert lr["subtasks"][-1]["label"] == "机械臂归位"
    json.dumps(meta)  # 仍可 JSON 序列化（报告会序列化 metadata）


# --------------------------------------------------------------------------- #
# metadata：声明 vs 实测
# --------------------------------------------------------------------------- #
def _spec_result(declared, measured):
    checker = MetadataChecker({})
    md = {"lerobot": {"declared_video": declared}, **measured}
    return checker._check_declared_vs_measured(md)


def test_spec_match_all_consistent():
    res = _spec_result(
        {"codec": "h264", "width": 640, "height": 480, "fps": 30, "pix_fmt": "yuv420p"},
        {"codec": "h264", "width": 640, "height": 480, "fps": 29.97, "pix_fmt": "yuv420p"},
    )
    assert res is not None and res.severity.value == "pass"
    assert res.details["n_mismatch"] == 0


def test_spec_match_codec_mismatch_flagged():
    res = _spec_result(
        {"codec": "h264", "fps": 30},
        {"codec": "av1", "fps": 30},
    )
    assert res.severity.value == "warn"
    assert "codec" in res.message


def test_spec_match_fps_within_tolerance_ok():
    res = _spec_result({"fps": 30}, {"fps": 29.5})
    assert res.severity.value == "pass"


def test_spec_match_absent_without_lerobot():
    # 纯视频（无 declared_video）→ 不产生 spec_match 项。
    assert MetadataChecker({})._check_declared_vs_measured({"codec": "h264"}) is None


def test_spec_match_disabled_via_config():
    checker = MetadataChecker({"metadata": {"spec_match": False}})
    md = {"lerobot": {"declared_video": {"codec": "h264"}}, "codec": "av1"}
    assert checker._check_declared_vs_measured(md) is None


# --------------------------------------------------------------------------- #
# dup_frame：声明 fps 钩子
# --------------------------------------------------------------------------- #
def test_declared_fps_from_declared_video():
    assert _declared_fps({"lerobot": {"declared_video": {"fps": 29}}}) == 29.0


def test_declared_fps_fallback_to_declared_fps_field():
    assert _declared_fps({"lerobot": {"declared_video": {}, "declared_fps": 28}}) == 28.0


def test_declared_fps_none_outside_lerobot():
    assert _declared_fps({"fps": 30}) is None
    assert _declared_fps({"lerobot": {"declared_video": {"fps": 0}}}) is None


# --------------------------------------------------------------------------- #
# _joints：关节运动量 + 交叉验证纯逻辑
# --------------------------------------------------------------------------- #
def test_camera_side_and_wrist():
    key = "camera_observations.color_images.camera_left"
    assert camera_side(key) == "left"
    assert is_wrist_camera(key) is True
    assert camera_side("x.camera_top") == "top"
    assert is_wrist_camera("x.camera_top") is False
    assert camera_side(None) is None


def test_per_frame_speed_side_and_overall():
    left = np.zeros((5, 7))
    left[2:] = 1.0  # 第 1→2 帧之间产生一次运动
    arms = {"left": left, "right": np.zeros((5, 7))}
    sp_left = per_frame_speed(arms, side="left")
    assert sp_left is not None and sp_left.shape == (4,)
    assert sp_left[1] > 0 and sp_left[0] == 0
    sp_all = per_frame_speed(arms, side=None)
    assert sp_all is not None and sp_all.shape == (4,)
    assert per_frame_speed({}, side="left") is None


def test_leading_trailing_static_frames():
    # 前 3 个静止、随后运动、末 2 个静止。
    speed = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    assert leading_static_frames(speed, 0.005) == 3
    assert trailing_static_frames(speed, 0.005) == 2
    # 全程静止。
    assert leading_static_frames(np.zeros(5), 0.005) == 5
    assert trailing_static_frames(np.zeros(5), 0.005) == 5


def test_joint_endpoint_static_seconds():
    speed = np.concatenate([np.zeros(60), np.ones(30), np.zeros(60)])  # 60 静 / 30 动 / 60 静
    lead, trail = joint_endpoint_static_seconds(speed, fps=30.0, move_speed=0.005)
    assert abs(lead - 2.0) < 1e-6 and abs(trail - 2.0) < 1e-6
    assert joint_endpoint_static_seconds(speed, fps=0, move_speed=0.005) == (0.0, 0.0)


def test_joint_moving_in_fraction():
    speed = np.zeros(100)
    speed[40:60] = 1.0  # 中段在动
    assert joint_moving_in_fraction(speed, 0.4, 0.6, 0.005) is True
    assert joint_moving_in_fraction(speed, 0.0, 0.2, 0.005) is False
    assert joint_moving_in_fraction(np.array([]), 0.0, 1.0, 0.005) is None


# --------------------------------------------------------------------------- #
# endpoint_static：末尾「归位」子任务
# --------------------------------------------------------------------------- #
def test_trailing_homing_label_detected():
    md = {"lerobot": {"subtasks": [
        {"label": "移动右臂抓取书籍"}, {"label": "机械臂归位"}]}}
    assert _trailing_homing_label(md) == "机械臂归位"


def test_trailing_homing_label_none_when_last_not_homing():
    md = {"lerobot": {"subtasks": [{"label": "移动右臂抓取书籍"}]}}
    assert _trailing_homing_label(md) is None
    assert _trailing_homing_label({}) is None


# --------------------------------------------------------------------------- #
# brightness：stats 基线钩子
# --------------------------------------------------------------------------- #
def test_brightness_pixel_baseline_hook():
    assert _pixel_luma_baseline({"lerobot": {"pixel_luma_baseline": 102.0}}) == 102.0
    assert _pixel_luma_baseline({"lerobot": {}}) is None
    assert _pixel_luma_baseline({}) is None
