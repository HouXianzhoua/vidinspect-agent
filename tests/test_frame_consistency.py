"""FrameConsistencyChecker 的纯判定逻辑测试（质检规范序号 18：画面保持一致）。

只覆盖纯函数 ``evaluate_frame_consistency``（不触碰 parquet I/O / 视频解码），与
``tests/test_static_joint.py`` 的纯函数测试风格一致。

约定：
- ``joint_speed``：该侧臂相邻帧关节速度（rad/帧）。
- ``cam_motion``：腕部相机相邻帧 mean|ΔY|（0–255）。
"""
import numpy as np

from vidinspect_agent.checkers.frame_consistency import evaluate_frame_consistency


def test_flags_arm_moving_but_camera_frozen():
    # 臂全程在动，但腕部相机画面全程冻结（帧差≈0）→ 画面/关节不一致命中。
    n = 60
    joint_speed = np.full(n, 0.05)          # 持续运动
    cam_motion = np.zeros(n)                # 相机彻底冻结
    out = evaluate_frame_consistency(joint_speed, cam_motion, fps=30.0, min_inconsistent_sec=1.0)
    assert out["detected"] is True
    assert out["reason"] == "ok"
    assert out["max_inconsistent_sec"] > 1.0


def test_passes_when_camera_tracks_arm_motion():
    # 臂在动、相机画面也大幅变动（远高于底噪）→ 一致，不命中。
    n = 60
    joint_speed = np.full(n, 0.05)
    cam_motion = np.full(n, 8.0)            # 腕部相机随臂运动，帧差大
    out = evaluate_frame_consistency(joint_speed, cam_motion, fps=30.0)
    assert out["detected"] is False
    assert out["reason"] == "ok"
    assert out["score"] == 1.0


def test_idle_arm_is_not_evaluated():
    # 该侧臂全程几乎不动（关节速度低于阈值）→ 无运动参照，不评估（PASS），不误报。
    n = 60
    joint_speed = np.full(n, 0.0005)        # < joint_move_speed 默认 0.01
    cam_motion = np.zeros(n)                # 相机也静（合理：臂没动）
    out = evaluate_frame_consistency(joint_speed, cam_motion, fps=30.0)
    assert out["reason"] == "arm_idle"
    assert out["detected"] is False
    assert out["arm_moving_ratio"] == 0.0


def test_short_inconsistency_below_threshold_passes():
    # 仅极短一段（< min_inconsistent_sec）臂动而画面静 → 不足以命中。
    n = 60
    joint_speed = np.full(n, 0.05)
    cam_motion = np.full(n, 8.0)
    cam_motion[10:20] = 0.0                 # 10 帧 ≈ 0.33s @30fps < 1.0s
    out = evaluate_frame_consistency(joint_speed, cam_motion, fps=30.0, min_inconsistent_sec=1.0)
    assert out["detected"] is False
    assert 0.0 < out["max_inconsistent_sec"] <= 0.4


def test_partial_freeze_relative_to_floor_is_flagged():
    # 相机"活着但严重卡顿"：底噪附近徘徊（远低于随臂运动应有的量级）且臂在动 → 命中。
    # 底噪由臂静止帧估计；这里前 20 帧臂静止用于估 floor，后 40 帧臂动但画面仍贴底噪。
    n = 60
    joint_speed = np.concatenate([np.full(20, 0.0), np.full(40, 0.05)])
    rng = np.random.default_rng(0)
    floor_noise = np.abs(rng.normal(0.0, 0.05, n)) + 0.2   # 相机底噪 ~0.2 量级
    cam_motion = floor_noise.copy()        # 臂动期间画面仍只有底噪（没随臂动）
    out = evaluate_frame_consistency(
        joint_speed, cam_motion, fps=30.0, min_inconsistent_sec=1.0, floor_k=3.0
    )
    assert out["detected"] is True
    assert out["cam_floor"] > 0.0
    assert out["static_thr"] >= out["cam_floor"] * 3.0 - 1e-9


def test_length_mismatch_is_aligned():
    # 关节序列与相机序列长度不一致（parquet 行数 vs 解码帧数）→ 比例对齐，不报错。
    joint_speed = np.full(45, 0.05)        # 较短
    cam_motion = np.zeros(60)              # 较长（主轴）
    out = evaluate_frame_consistency(joint_speed, cam_motion, fps=30.0)
    assert out["n_frames"] == 60
    assert out["detected"] is True


def test_empty_inputs_not_evaluated():
    out = evaluate_frame_consistency(np.empty(0), np.empty(0), fps=30.0)
    assert out["evaluated"] is False
    assert out["reason"] == "insufficient"
    assert out["detected"] is False


def test_zero_fps_not_evaluated():
    out = evaluate_frame_consistency(np.full(30, 0.05), np.zeros(30), fps=0.0)
    assert out["evaluated"] is False
    assert out["detected"] is False
