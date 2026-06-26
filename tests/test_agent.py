import pytest

from vidinspect_agent.agent import discover_videos
from vidinspect_agent.checkers import _lerobot
from vidinspect_agent.checkers.brightness import evaluate_brightness
from vidinspect_agent.checkers.colormatch import _task_hint, evaluate_colormatch
from vidinspect_agent.checkers.edge_grasp import evaluate_edge_grasp
from vidinspect_agent.checkers.object_slip import _pick_closed, detect_slip
from vidinspect_agent.checkers.occlusion import evaluate_occlusion
from vidinspect_agent.checkers.regrasp import detect_regrasp
from vidinspect_agent.checkers.tablecloth import evaluate_tablecloth
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


def test_opening_to_closed_low_is_closed():
    # 默认约定：开合值越小越闭合。低值帧→闭合(True)，高值帧→张开(False)。
    closed = _lerobot.opening_to_closed([0.0, 0.0, 1.0, 1.0], closed_is_low=True)
    assert closed == [True, True, False, False]


def test_opening_to_closed_high_is_closed():
    # 反向约定：值越大越闭合。
    closed = _lerobot.opening_to_closed([0.0, 0.0, 1.0, 1.0], closed_is_low=False)
    assert closed == [False, False, True, True]


def test_opening_to_closed_constant_is_unknown():
    # 夹爪整段几乎不动（区间过小）→ 无从区分开合 → 全 None（不可判）。
    closed = _lerobot.opening_to_closed([0.5, 0.5, 0.5, 0.5])
    assert closed == [None, None, None, None]


def test_opening_to_closed_nan_frame_is_unknown():
    # 单帧 NaN/None → 该帧 None，其余按区间判定。
    closed = _lerobot.opening_to_closed([0.0, float("nan"), 1.0], closed_is_low=True)
    assert closed[0] is True and closed[1] is None and closed[2] is False


def test_map_to_sampled_aligns_by_time():
    # 逐视频帧闭合序列按 round(t*fps) 映射到采样帧时间轴。
    per_frame = [True, True, False, False, False, False]  # 30fps 下第 0..5 帧
    times = [0.0, 0.1, 0.2]  # fps=10 采样：对应视频帧 round(t*30)=0,3,6
    out = _lerobot.map_to_sampled(per_frame, times, video_fps=30.0)
    assert out == [True, False, None]  # 第三个越界(6>=6)→None


def test_map_to_sampled_no_fps_all_unknown():
    assert _lerobot.map_to_sampled([True, False], [0.0, 1.0], video_fps=0.0) == [None, None]


def test_find_episode_parquet_none_outside_lerobot(tmp_path):
    # 非 LeRobot 布局的普通视频路径 → None，object_slip 据此回退模型信号。
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    assert _lerobot.find_episode_parquet(video) is None


def test_find_episode_parquet_locates_sibling(tmp_path):
    # 标准布局 videos/<chunk>/<cam>/<stem>.mp4 → data/<chunk>/<stem>.parquet。
    root = tmp_path / "lerobot_RoboMIND"
    cam = root / "videos" / "chunk-000" / "camera_top"
    cam.mkdir(parents=True)
    video = cam / "episode_000129.mp4"
    video.write_bytes(b"fake")
    data = root / "data" / "chunk-000"
    data.mkdir(parents=True)
    parquet = data / "episode_000129.parquet"
    parquet.write_bytes(b"fake")
    assert _lerobot.find_episode_parquet(video) == parquet


def test_gripper_opening_from_metadata_none_without_pointer():
    # 对齐摄入层：无 metadata['lerobot']['parquet_path'] 指针 → None（上层回退自定位/模型）。
    assert _lerobot.gripper_opening_from_metadata({}) is None
    assert _lerobot.gripper_opening_from_metadata({"lerobot": {}}) is None
    assert _lerobot.gripper_opening_from_metadata({"lerobot": {"parquet_path": None}}) is None


def test_pick_closed_prefers_parquet_exact_side():
    parquet = {"left": [True, True], "right": [False, False]}
    seq, src = _pick_closed("left", parquet, {"left": [None, None]}, 2)
    assert src == "parquet" and seq == [True, True]


def test_pick_closed_single_maps_to_sole_parquet_side():
    parquet = {"left": [True, False]}
    seq, src = _pick_closed("single", parquet, {}, 2)
    assert src == "parquet" and seq == [True, False]


def test_pick_closed_falls_back_to_model_when_unmatched():
    # 模型判为 single，但 parquet 有左右两侧 → 无法稳妥配对 → 回退模型信号。
    parquet = {"left": [True, True], "right": [False, False]}
    seq, src = _pick_closed("single", parquet, {"single": [True, None]}, 2)
    assert src == "model" and seq == [True, None]


def test_pick_closed_falls_back_when_no_parquet():
    seq, src = _pick_closed("left", {}, {"left": [None, True]}, 2)
    assert src == "model" and seq == [None, True]


def test_slip_uses_parquet_closed_signal_end_to_end():
    # parquet 开合轨迹 → 闭合序列 → detect_slip：持有[0,3)，释放窗口[3,5)仍闭合 → 滑落。
    hold = ["a", "a", "a", None, None, None]
    opening = [0.1, 0.1, 0.1, 0.1, 0.1, 0.9]  # 仅末帧张开，释放窗口内仍闭合
    closed = _lerobot.opening_to_closed(opening, closed_is_low=True)
    out = detect_slip(hold, closed, min_hold_frames=2, min_release_frames=2,
                      release_window_frames=2)
    assert out["detected"] is True


def test_slip_parquet_open_after_release_is_normal_place():
    # 释放后夹爪明确张开（高开合值）→ 正常放下，不报。
    hold = ["a", "a", "a", None, None, None]
    opening = [0.1, 0.1, 0.1, 0.9, 0.9, 0.9]
    closed = _lerobot.opening_to_closed(opening, closed_is_low=True)
    out = detect_slip(hold, closed, min_hold_frames=2, min_release_frames=2,
                      release_window_frames=2)
    assert out["detected"] is False


def _write_lerobot_episode(root, opening_by_side, *, episode="episode_000129"):
    """构造最小 LeRobot 组：videos/.../<ep>.mp4 + data/.../<ep>.parquet（含夹爪开合列）。"""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    cam = root / "videos" / "chunk-000" / "camera_top"
    cam.mkdir(parents=True)
    video = cam / f"{episode}.mp4"
    video.write_bytes(b"fake")
    data = root / "data" / "chunk-000"
    data.mkdir(parents=True)
    cols = {
        f"puppet.end_effector_{side}_position_align.data": [[float(v)] for v in vals]
        for side, vals in opening_by_side.items()
    }
    pq.write_table(pa.table(cols), str(data / f"{episode}.parquet"))
    return video


def test_regrasp_parquet_double_grasp_end_to_end():
    # §2.2：闭(低)→开(高)→闭(低) 的开合轨迹 → 逐臂 single_object 计 2 段 → 命中。
    opening = [0.0] * 8 + [1.0] * 12 + [0.0] * 8
    closed = _lerobot.opening_to_closed(opening, closed_is_low=True)
    seq = ["__hold__" if c else None for c in closed]
    out = detect_regrasp(seq, single_object=True, min_hold_frames=5, min_release_frames=10)
    assert out["detected"] is True
    assert out["counts"]["__hold__"] == 2


def test_regrasp_checker_parquet_flags_double_grasp(tmp_path):
    # §2.2 端到端：RegraspChecker 走 parquet 夹爪真实信号，无需 API key 即判定。
    from vidinspect_agent.checkers.regrasp import RegraspChecker

    opening = [0.0] * 8 + [1.0] * 12 + [0.0] * 8  # 同一只左臂两次闭合
    video = _write_lerobot_episode(tmp_path / "lerobot_RoboMIND", {"left": opening})
    cfg = {"regrasp": {"severity": "warn", "min_hold_sec": 0.5, "min_release_sec": 1.0}}
    # fps=10 → min_hold=5 帧、min_release=10 帧。
    results = RegraspChecker(cfg).check(video, {"fps": 10.0})
    assert len(results) == 1
    res = results[0]
    assert res.details["source"] == "parquet"
    assert res.details["arm_grasp_counts"]["left"] == 2
    assert res.severity == Severity.WARN


def test_regrasp_checker_parquet_single_grasp_passes(tmp_path):
    from vidinspect_agent.checkers.regrasp import RegraspChecker

    opening = [1.0] * 6 + [0.0] * 12 + [1.0] * 6  # 仅一次抓取 → 不命中
    video = _write_lerobot_episode(tmp_path / "lerobot_RoboMIND", {"left": opening})
    cfg = {"regrasp": {"severity": "warn", "min_hold_sec": 0.5, "min_release_sec": 1.0}}
    res = RegraspChecker(cfg).check(video, {"fps": 10.0})[0]
    assert res.details["source"] == "parquet"
    assert res.severity == Severity.PASS
    assert res.details["arm_grasp_counts"]["left"] == 1


def test_regrasp_non_lerobot_falls_back_to_model(tmp_path, monkeypatch):
    # 非 LeRobot 布局 → parquet 路径返回 None → 回退模型；无 API key → WARN 跳过。
    from vidinspect_agent.checkers.regrasp import RegraspChecker

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    res = RegraspChecker({"regrasp": {}}).check(video, {"fps": 30.0})[0]
    assert res.severity == Severity.WARN
    assert res.details.get("error") == "missing_api_key"


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


def _occ(verdicts, thr=0.5, min_conf=0.0):
    return evaluate_occlusion(verdicts, hit_ratio_thr=thr, min_confidence=min_conf)


def test_occlusion_majority_occluded_detected():
    # 首段多数可判定帧都被夹爪遮挡，占比 ≥ 阈值 → 命中。
    verdicts = [
        {"occluded": True, "confidence": 0.9},
        {"occluded": True, "confidence": 0.8},
        {"occluded": False, "confidence": 0.9},
    ]
    out = _occ(verdicts)
    assert out["detected"] is True
    assert out["n_occluded"] == 2 and out["n_judged"] == 3


def test_occlusion_clear_object_not_flagged():
    # 首段物体清晰可见 → 占比低于阈值 → 不命中。
    verdicts = [
        {"occluded": False, "confidence": 0.9},
        {"occluded": False, "confidence": 0.9},
        {"occluded": True, "confidence": 0.6},
    ]
    out = _occ(verdicts)
    assert out["detected"] is False
    assert out["score"] > 0.5


def _tc(verdicts, thr=0.3, min_hit=2, min_conf=0.0):
    return evaluate_tablecloth(
        verdicts, hit_ratio_thr=thr, min_hit_frames=min_hit, min_confidence=min_conf
    )


def test_tablecloth_caught_detected():
    # 含桌布帧里多帧判定误夹，占比 ≥ 阈值且命中帧数达标 → 命中。
    verdicts = [
        {"has_cloth": True, "caught": True, "confidence": 0.9},
        {"has_cloth": True, "caught": True, "confidence": 0.8},
        {"has_cloth": True, "caught": False, "confidence": 0.9},
        {"has_cloth": True, "caught": False, "confidence": 0.7},
    ]
    out = _tc(verdicts)
    assert out["detected"] is True
    assert out["n_caught"] == 2 and out["n_cloth"] == 4


def test_tablecloth_clean_not_flagged():
    # 含桌布但夹爪未误夹（桌布平整）→ 不命中。
    verdicts = [
        {"has_cloth": True, "caught": False, "confidence": 0.9},
        {"has_cloth": True, "caught": False, "confidence": 0.9},
        {"has_cloth": True, "caught": True, "confidence": 0.6},
    ]
    out = _tc(verdicts)
    assert out["detected"] is False
    assert out["score"] > 0.5


def _eg(verdicts, thr=0.5, min_conf=0.0):
    return evaluate_edge_grasp(verdicts, hit_ratio_thr=thr, min_confidence=min_conf)


def test_edge_grasp_majority_edge_detected():
    # 多数可判定帧都"夹在边缘"，占比 ≥ 阈值 → 命中。
    verdicts = [
        {"edge": True, "confidence": 0.9},
        {"edge": True, "confidence": 0.8},
        {"edge": False, "confidence": 0.9},
        {"edge": True, "confidence": 0.7},
    ]
    out = _eg(verdicts)
    assert out["detected"] is True
    assert out["n_edge"] == 3 and out["n_judged"] == 4


def test_edge_grasp_body_grasp_not_flagged():
    # 多数帧夹在主体 → 占比低于阈值 → 不命中。
    verdicts = [
        {"edge": False, "confidence": 0.9},
        {"edge": False, "confidence": 0.9},
        {"edge": True, "confidence": 0.6},
    ]
    out = _eg(verdicts)
    assert out["detected"] is False
    assert out["score"] > 0.5


def test_occlusion_unreturned_frames_not_counted():
    # 模型未返回的帧(None)不计入分母，只在可判定帧里算占比。
    verdicts = [None, {"occluded": True, "confidence": 0.9}]
    out = _occ(verdicts)
    assert out["n_judged"] == 1 and out["hit_ratio"] == 1.0
    assert out["detected"] is True


def test_occlusion_low_confidence_frames_skipped():
    # 启用 min_confidence 后，低置信度帧被跳过，不计入。
    verdicts = [
        {"occluded": True, "confidence": 0.2},
        {"occluded": False, "confidence": 0.9},
    ]
    out = _occ(verdicts, min_conf=0.5)
    assert out["n_judged"] == 1 and out["n_occluded"] == 0
    assert out["detected"] is False


def test_occlusion_no_judged_frames_safe():
    # 首段全部帧无可辨识物体 → 无有效判定，不命中（上层据 n_judged 报 WARN）。
    out = _occ([None, None])
    assert out["n_judged"] == 0
    assert out["detected"] is False


def test_edge_grasp_unreturned_frames_not_counted():
    # 模型未返回的帧(None，未夹住物体/看不清接触点)不计入分母，只在能判定帧里算占比。
    verdicts = [None, None, {"edge": True, "confidence": 0.9}, {"edge": True, "confidence": 0.9}]
    out = _eg(verdicts)
    assert out["n_judged"] == 2 and out["hit_ratio"] == 1.0
    assert out["detected"] is True


def test_edge_grasp_low_confidence_frames_skipped():
    # 启用 min_confidence 后，低置信度帧被跳过，不计入。
    verdicts = [
        {"edge": True, "confidence": 0.2},
        {"edge": False, "confidence": 0.9},
        {"edge": False, "confidence": 0.9},
    ]
    out = _eg(verdicts, min_conf=0.5)
    assert out["n_judged"] == 2 and out["n_edge"] == 0
    assert out["detected"] is False


def test_edge_grasp_no_judged_frames_safe():
    # 全部帧未见夹爪夹住物体 → 无有效判定，不命中（上层据 n_judged 报 WARN）。
    out = _eg([None, None, None])
    assert out["n_judged"] == 0
    assert out["detected"] is False


def test_edge_grasp_checker_missing_api_key(tmp_path, monkeypatch):
    # 无 API key → WARN 跳过，不阻塞流水线。
    from vidinspect_agent.checkers.edge_grasp import EdgeGraspChecker

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    res = EdgeGraspChecker({"edge_grasp": {}}).check(video, {"fps": 30.0})[0]
    assert res.severity == Severity.WARN
    assert res.details.get("error") == "missing_api_key"


def test_tablecloth_non_cloth_frames_excluded():
    # 无桌布的帧不计入分母，只在含桌布帧里算占比。
    verdicts = [
        {"has_cloth": False, "caught": False, "confidence": 0.9},
        None,
        {"has_cloth": True, "caught": True, "confidence": 0.9},
        {"has_cloth": True, "caught": True, "confidence": 0.9},
    ]
    out = _tc(verdicts)
    assert out["n_cloth"] == 2 and out["hit_ratio"] == 1.0
    assert out["detected"] is True


def test_tablecloth_single_caught_frame_suppressed():
    # 仅 1 帧误夹（< min_hit_frames）→ 不命中，压制单帧误判。
    verdicts = [
        {"has_cloth": True, "caught": True, "confidence": 0.9},
        {"has_cloth": True, "caught": False, "confidence": 0.9},
    ]
    out = _tc(verdicts)
    assert out["n_caught"] == 1
    assert out["detected"] is False


def test_tablecloth_no_cloth_scene_not_detected():
    # 全程无桌布 → n_cloth=0，不命中（上层据 min_cloth_frames 判 PASS 不适用）。
    verdicts = [
        {"has_cloth": False, "caught": False, "confidence": 0.9},
        {"has_cloth": False, "caught": False, "confidence": 0.9},
    ]
    out = _tc(verdicts)
    assert out["n_cloth"] == 0
    assert out["detected"] is False


def test_tablecloth_low_confidence_frames_skipped():
    # 启用 min_confidence 后，低置信度的含桌布帧被跳过，不计入。
    verdicts = [
        {"has_cloth": True, "caught": True, "confidence": 0.2},
        {"has_cloth": True, "caught": False, "confidence": 0.9},
        {"has_cloth": True, "caught": False, "confidence": 0.9},
    ]
    out = _tc(verdicts, min_conf=0.5)
    assert out["n_cloth"] == 2 and out["n_caught"] == 0
    assert out["detected"] is False


def test_task_hint_empty_when_no_metadata():
    # 当前 pipeline 不填这些字段 → 空串，不影响纯视频检测。
    assert _task_hint({}) == ""
    assert _task_hint({"robot": "tienkung"}) == ""


def test_task_hint_from_target_objects_list():
    hint = _task_hint({"target_objects": ["书籍", "文件夹"]})
    assert "书籍" in hint and "文件夹" in hint
    assert hint.endswith("\n")


def test_task_hint_from_task_description():
    hint = _task_hint({"task": "整理书籍上架"})
    assert "整理书籍上架" in hint


def test_task_hint_combines_objects_and_task():
    hint = _task_hint({"target_objects": "书籍", "task": "整理书籍上架"})
    assert "书籍" in hint and "整理书籍上架" in hint


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
