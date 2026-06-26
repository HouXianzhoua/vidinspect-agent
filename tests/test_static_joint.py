"""StaticChecker 的 joint（LeRobot parquet 关节）后端测试（detector_dataset_impact §2.1）。

只覆盖纯逻辑 ``evaluate_joint_static``（不触碰 parquet I/O / 视频解码），与现有
``tests/test_agent.py`` 的纯函数测试风格一致。
"""
import numpy as np

from vidinspect_agent.checkers.static import evaluate_joint_static


def test_joint_static_flags_motionless_arms():
    # 双臂逐帧关节几乎不动（仅 1e-4 量级抖动）→ 峰峰值远小于阈值 → 判静止。
    rng = np.random.default_rng(0)
    left = np.full((50, 7), 0.3) + rng.normal(0, 1e-4, (50, 7))
    right = np.full((50, 7), -0.2) + rng.normal(0, 1e-4, (50, 7))
    out = evaluate_joint_static({"left": left, "right": right}, range_thr=0.05)
    assert out["detected"] is True
    assert out["max_range"] < 0.05
    assert out["n_frames"] == 50
    assert out["arms"] == ["left", "right"]


def test_joint_static_passes_when_arm_moves():
    # 一只臂某关节做大幅扫动（0→1 rad）→ 峰峰值超阈值 → 正常运动，不命中。
    t = np.linspace(0, 1.0, 60)
    left = np.zeros((60, 7))
    left[:, 0] = t  # 单关节从 0 扫到 1 rad
    right = np.full((60, 7), 0.1)
    out = evaluate_joint_static({"left": left, "right": right}, range_thr=0.05)
    assert out["detected"] is False
    assert out["max_range"] > 0.05
    assert out["score"] == 1.0


def test_joint_static_robust_to_sparse_glitches():
    # 整段静止但有零星传感跳变（少数帧 +5 rad）：q01/q99 稳健分位把尾部离群点排除 →
    # 仍判静止（朴素 max-min 会被单点拉爆而漏判）。
    arr = np.full((2000, 7), 0.5)
    arr[[137, 942, 1685], 2] += 5.0  # 3/2000 ≈ 0.15% 的离群帧，落在 q99 之外
    out = evaluate_joint_static({"left": arr}, range_thr=0.05)
    assert out["detected"] is True
    assert np.ptp(arr[:, 2]) > 0.05  # 朴素峰峰值会误判，凸显稳健分位的必要


def test_joint_static_single_arm_supported():
    # 单臂数据（仅一侧）也能评估。
    t = np.linspace(0, 0.8, 40)
    single = np.zeros((40, 7))
    single[:, 3] = t
    out = evaluate_joint_static({"single": single}, range_thr=0.05)
    assert out["evaluated"] is True
    assert out["detected"] is False


def test_joint_static_aligns_arms_to_shortest():
    # 双臂帧数不一致时对齐到最短长度，不报错。
    left = np.full((30, 7), 0.2)
    right = np.full((25, 7), 0.2)
    out = evaluate_joint_static({"left": left, "right": right}, range_thr=0.05)
    assert out["n_frames"] == 25
    assert out["detected"] is True


def test_joint_static_empty_is_not_flagged():
    out = evaluate_joint_static({}, range_thr=0.05)
    assert out["detected"] is False
    assert out["evaluated"] is False
    assert out["score"] == 1.0
