from vidinspect_agent.agent import discover_videos
from vidinspect_agent.models import CheckResult, Severity, VideoReport


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
