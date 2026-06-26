"""TailActionChecker 的纯逻辑测试（质检规范序号 24，末尾多余动作）。

只覆盖纯函数 ``evaluate_tail_action`` / ``last_labeled_end_frame``（不触碰 parquet I/O /
视频解码），与 ``tests/test_static_joint.py`` 的纯函数测试风格一致。
"""
import numpy as np

from vidinspect_agent.checkers.tail_action import (
    evaluate_tail_action,
    last_labeled_end_frame,
)

FPS = 30.0  # min_action_sec=0.3 → min_run = round(9) = 9 帧


def _speed(total_frames: int) -> np.ndarray:
    """长度 T-1 的全静止逐帧速度数组。"""
    return np.zeros(total_frames - 1, dtype=np.float64)


def test_static_tail_not_flagged():
    # 标注末动作结束帧之后关节静止（仅留冗余静止帧）→ 规范允许，不命中。
    speed = _speed(60)
    speed[:30] = 0.05  # 标注覆盖段内有动作
    out = evaluate_tail_action(speed, last_end_frame=30, fps=FPS)
    assert out["detected"] is False
    assert out["tail_extra_frames"] == 29
    assert out["tail_longest_run"] == 0
    assert out["score"] == 1.0


def test_sustained_tail_action_flagged():
    # 末动作结束帧之后仍有持续关节运动 → 末尾多余动作，命中。
    speed = _speed(60)
    speed[30:] = 0.05
    out = evaluate_tail_action(speed, last_end_frame=30, fps=FPS)
    assert out["detected"] is True
    assert out["tail_longest_run"] == 29
    assert out["tail_action_sec"] > 0.3
    assert out["score"] == 0.0


def test_isolated_tail_glitch_debounced():
    # 末尾仅 3 帧零星跳变（< min_run=9）→ 去抖后不误报。
    speed = _speed(60)
    speed[40:43] = 0.05
    out = evaluate_tail_action(speed, last_end_frame=30, fps=FPS)
    assert out["detected"] is False
    assert out["tail_longest_run"] == 3
    assert out["tail_moving_frames"] == 3


def test_debounce_boundary_exact_min_run_flagged():
    # 恰好达到 min_run=9 帧连续运动即命中（边界）。
    speed = _speed(60)
    speed[30:39] = 0.05  # 连续 9 帧
    out = evaluate_tail_action(speed, last_end_frame=30, fps=FPS)
    assert out["min_run"] == 9
    assert out["tail_longest_run"] == 9
    assert out["detected"] is True


def test_motion_before_last_end_ignored():
    # 运动全部发生在标注覆盖段内，末尾静止 → 不命中（只看末动作结束帧之后）。
    speed = _speed(60)
    speed[:30] = 0.2
    out = evaluate_tail_action(speed, last_end_frame=30, fps=FPS)
    assert out["detected"] is False


def test_empty_tail_when_label_covers_all_frames():
    # 标注末动作结束帧 >= 末帧（无冗余帧）→ tail 为空，不命中。
    speed = _speed(60)
    speed[:] = 0.05
    out = evaluate_tail_action(speed, last_end_frame=100, fps=FPS)
    assert out["evaluated"] is True
    assert out["detected"] is False
    assert out["tail_extra_frames"] == 0


def test_total_frames_reported_from_speed():
    speed = _speed(45)
    out = evaluate_tail_action(speed, last_end_frame=10, fps=FPS)
    assert out["total_frames"] == 45


def test_last_labeled_end_frame_takes_max():
    # 子任务区间乱序 / 重叠时，取最大 end_frame 作「标注末动作结束帧」。
    subtasks = [
        {"start_frame": 0, "end_frame": 284, "label": "a"},
        {"start_frame": 2414, "end_frame": 2535, "label": "b"},  # start 乱序
        {"start_frame": 2536, "end_frame": 2900, "label": "c"},
        {"start_frame": 2901, "end_frame": 2940, "label": "归位"},
    ]
    assert last_labeled_end_frame(subtasks) == 2940


def test_last_labeled_end_frame_invalid_returns_none():
    assert last_labeled_end_frame([]) is None
    assert last_labeled_end_frame(None) is None
    assert last_labeled_end_frame([{"start_frame": 0}]) is None
    assert last_labeled_end_frame([{"end_frame": True}]) is None  # bool 不算有效帧号
