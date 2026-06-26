from vidinspect_agent.agent import discover_videos
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
