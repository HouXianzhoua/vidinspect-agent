from vidinspect_agent.agent import discover_videos
from vidinspect_agent.checkers.brightness import evaluate_brightness
from vidinspect_agent.checkers.colormatch import evaluate_colormatch
from vidinspect_agent.checkers.object_slip import detect_slip
from vidinspect_agent.checkers.regrasp import detect_regrasp
from vidinspect_agent.models import CheckResult, Severity, VideoReport


def _detect(seq, single_object=False):
    return detect_regrasp(
        seq, single_object=single_object, min_hold_frames=2, min_release_frames=2
    )


def test_regrasp_single_grasp_passes():
    seq = [None, "a", "a", "a", None]
    assert _detect(seq)["detected"] is False


def test_regrasp_same_object_twice_detected():
    seq = ["a", "a", "a", None, None, None, "a", "a", "a"]
    out = _detect(seq)
    assert out["detected"] is True
    assert out["counts"]["a"] == 2


def test_regrasp_different_objects_not_flagged():
    # 抓 A→放 A→抓 B：正常的多物体顺序操作，标签模式不应命中。
    seq = ["a", "a", "a", None, None, None, "b", "b", "b"]
    assert _detect(seq)["detected"] is False


def test_regrasp_occlusion_flicker_bridged():
    # 持有中单帧遮挡闪断，应被桥接为同一持有段，不误报。
    seq = ["a", "a", "a", None, "a", "a", "a"]
    out = _detect(seq)
    assert out["detected"] is False
    assert out["counts"]["a"] == 1


def test_regrasp_short_spurious_hold_dropped():
    # 单帧误检的"持有"应被丢弃。
    seq = [None, None, "a", None, None, "b", "b", None]
    out = _detect(seq)
    assert "a" not in out["counts"]
    assert out["detected"] is False


def test_regrasp_single_object_mode_ignores_labels():
    # single_object：忽略标签，再次抓取任意物体即命中。
    seq = ["a", "a", "a", None, None, None, "b", "b", "b"]
    out = _detect(seq, single_object=True)
    assert out["detected"] is True


def test_regrasp_per_arm_same_arm_twice_flagged():
    # 同一只机械臂抓取→释放→再抓取（臂内 single_object 语义）→ 命中。
    arm = ["a", "a", "a", None, None, None, "a", "a", "a"]
    assert _detect(arm, single_object=True)["detected"] is True


def test_regrasp_per_arm_two_arms_each_once_not_flagged():
    # 双臂各抓一次：按臂分别建序列，每条均单段 → 均不命中（含 A→B 交接场景）。
    left = ["a", "a", "a", None, None, None]
    right = [None, None, None, "b", "b", "b"]
    assert _detect(left, single_object=True)["detected"] is False
    assert _detect(right, single_object=True)["detected"] is False


def _slip(hold, closed, window=2):
    return detect_slip(
        hold, closed,
        min_hold_frames=2, min_release_frames=2, release_window_frames=window,
    )


def test_slip_closed_after_release_detected():
    # 持有结束后夹爪仍闭合(True)却没了物体 → 滑落。
    hold = ["a", "a", "a", None, None, None]
    closed = [True, True, True, True, True, True]
    out = _slip(hold, closed)
    assert out["detected"] is True
    assert out["events"][0]["release_frame"] == 3


def test_slip_open_after_release_is_normal_place():
    # 持有结束后夹爪张开(False) → 正常放下，不报。
    hold = ["a", "a", "a", None, None, None]
    closed = [True, True, True, False, False, False]
    assert _slip(hold, closed)["detected"] is False


def test_slip_unknown_gripper_state_skipped():
    # 释放后夹爪状态全未知(None) → 保守不误报。
    hold = ["a", "a", "a", None, None, None]
    closed = [True, True, True, None, None, None]
    assert _slip(hold, closed)["detected"] is False


def test_slip_hold_until_end_not_flagged():
    # 持有持续到片尾，未观察到释放 → 不判（可能仍正常搬运中）。
    hold = ["a", "a", "a", "a"]
    closed = [True, True, True, True]
    out = _slip(hold, closed)
    assert out["detected"] is False
    assert out["n_release"] == 0


def test_slip_short_spurious_hold_dropped():
    # 单帧误检"持有"应被去抖丢弃，不产生滑落事件。
    hold = [None, "a", None, None, None]
    closed = [True, True, True, True, True]
    assert _slip(hold, closed)["detected"] is False


def test_slip_window_does_not_cross_next_hold():
    # 释放缝隙内夹爪状态未知，下一持有段夹爪闭合；窗口不应越过下一持有段把"再抓"误判为滑落。
    hold = ["a", "a", None, None, "a", "a"]
    closed = [True, True, None, None, True, True]
    out = _slip(hold, closed, window=4)
    assert out["detected"] is False


def test_brightness_underexposed_detected():
    # 各帧平均亮度都很低（欠曝），中位数 < 阈值 → 命中。
    out = evaluate_brightness([18.0, 22.0, 20.0, 19.0], min_luma=40.0)
    assert out["detected"] is True
    assert out["luma_median"] < 40.0


def test_brightness_normal_not_flagged():
    # 正常亮度画面，不命中。
    out = evaluate_brightness([120.0, 130.0, 125.0], min_luma=40.0)
    assert out["detected"] is False
    assert out["score"] == 1.0


def test_brightness_median_robust_to_single_bright_frame():
    # 整体偏暗，仅个别帧曝光突变变亮；中位数仍落在阈值下 → 命中（对突变帧鲁棒）。
    out = evaluate_brightness([18.0, 20.0, 200.0, 19.0], min_luma=40.0)
    assert out["detected"] is True


def test_brightness_empty_sequence_safe():
    out = evaluate_brightness([], min_luma=40.0)
    assert out["detected"] is False
    assert out["luma_median"] == 0.0


def _cm(verdicts, thr=0.5, min_conf=0.0):
    return evaluate_colormatch(verdicts, hit_ratio_thr=thr, min_confidence=min_conf)


def test_colormatch_majority_hard_detected():
    # 多数可判定帧都"难分辨"，占比 ≥ 阈值 → 命中。
    verdicts = [
        {"hard": True, "confidence": 0.9},
        {"hard": True, "confidence": 0.8},
        {"hard": False, "confidence": 0.9},
        {"hard": True, "confidence": 0.7},
    ]
    out = _cm(verdicts)
    assert out["detected"] is True
    assert out["n_hard"] == 3 and out["n_judged"] == 4


def test_colormatch_clear_contrast_not_flagged():
    # 多数帧轮廓清晰 → 占比低于阈值 → 不命中。
    verdicts = [
        {"hard": False, "confidence": 0.9},
        {"hard": False, "confidence": 0.9},
        {"hard": True, "confidence": 0.6},
    ]
    out = _cm(verdicts)
    assert out["detected"] is False
    assert out["score"] > 0.5


def test_colormatch_unreturned_frames_not_counted():
    # 模型未返回的帧(None)不计入分母，只在可辨识帧里算占比。
    verdicts = [None, None, {"hard": True, "confidence": 0.9}, {"hard": True, "confidence": 0.9}]
    out = _cm(verdicts)
    assert out["n_judged"] == 2 and out["hit_ratio"] == 1.0
    assert out["detected"] is True


def test_colormatch_low_confidence_frames_skipped():
    # 启用 min_confidence 后，低置信度帧被跳过，不计入。
    verdicts = [
        {"hard": True, "confidence": 0.2},
        {"hard": False, "confidence": 0.9},
        {"hard": False, "confidence": 0.9},
    ]
    out = _cm(verdicts, min_conf=0.5)
    assert out["n_judged"] == 2 and out["n_hard"] == 0
    assert out["detected"] is False


def test_colormatch_no_judged_frames_safe():
    # 全部帧无可辨识物体 → 无有效判定，hit_ratio=0，不命中（上层据 n_judged 报 WARN）。
    out = _cm([None, None, None])
    assert out["n_judged"] == 0
    assert out["detected"] is False


def test_video_report_to_dict():
    report = VideoReport(
        path="/tmp/sample.mp4",
        passed=False,
        results=[
            CheckResult(
                name="resolution",
                severity=Severity.FAIL,
                message="分辨率过低",
            )
        ],
        metadata={"width": 320, "height": 240},
    )
    data = report.to_dict()
    assert data["passed"] is False
    assert data["results"][0]["severity"] == "fail"
    assert data["metadata"]["width"] == 320


def test_discover_videos_file(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    found = discover_videos(video)
    assert found == [video]
