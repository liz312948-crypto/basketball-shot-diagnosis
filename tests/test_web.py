import os

from fastapi.testclient import TestClient

from app.jobs import JobArtifacts, JobRecord, JobStatus, LayeredResult
from app.models import ConfidenceLevel, PrecheckResult
from app.web import app, job_store, resolve_server_host, resolve_server_port


def test_home_page_loads() -> None:
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "投篮动作诊断" in response.text
    assert "上传投篮视频" in response.text
    assert "中文结果 / 精度优先" in response.text
    assert 'id="uploadStatus"' in response.text
    assert "xhr.upload.addEventListener(\"progress\"" in response.text
    assert "正在上传视频" in response.text


def test_create_job_returns_job_id() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/jobs",
        files={"video": ("clip.mp4", b"fake-bytes", "video/mp4")},
    )

    assert response.status_code == 202
    payload = response.json()
    assert "job_id" in payload
    assert payload["status"] == "uploaded"
    assert payload["progress_percent"] == 10
    assert "准备进入分析" in payload["status_detail"]


def test_unknown_job_returns_404() -> None:
    client = TestClient(app)

    response = client.get("/api/jobs/missing")

    assert response.status_code == 404


def test_result_page_shows_confidence_and_summary(tmp_path) -> None:
    precheck = PrecheckResult(
        score=0.82,
        confidence=ConfidenceLevel.HIGH,
        run_enhanced_analysis=True,
        view_type="side",
        summary="视频质量较好，适合做增强诊断。",
        reasons=["机位稳定，主体完整"],
        recommendations=[],
    )
    job_store.records["demo-job"] = JobRecord(
        id="demo-job",
        filename="demo.mp4",
        input_path=tmp_path / "demo.mp4",
        status=JobStatus.DONE,
        result=LayeredResult(
            precheck=precheck,
            overall_summary="Overall motion is stable but the release arc is flat.",
            findings=["Release arc is too flat."],
            drills=["Add close-range high-arc form shooting."],
            enhanced_summary="已进入增强诊断。",
            artifacts=JobArtifacts(highlight_video="demo_highlight.mp4"),
        ),
    )

    client = TestClient(app)
    response = client.get("/jobs/demo-job")

    assert response.status_code == 200
    assert "可信度 高" in response.text
    assert "机位：侧面" in response.text
    assert "主要问题" in response.text
    assert "训练建议" in response.text
    assert "诊断视频" in response.text
    assert "识别到关键问题时会插入定格画面" in response.text
    assert 'src="/media/demo_highlight.mp4"' in response.text
    assert "下载分析报告 JSON" not in response.text
    assert "下载事件时间轴 JSON" not in response.text
    assert "单独打开诊断视频" in response.text


def test_result_page_shows_stage_breakdown_when_enhanced_data_exists(tmp_path) -> None:
    precheck = PrecheckResult(
        score=0.9,
        confidence=ConfidenceLevel.HIGH,
        run_enhanced_analysis=True,
        view_type="side",
        summary="视频质量较好，适合做增强诊断。",
        reasons=["机位稳定，主体完整"],
        recommendations=[],
    )
    result = LayeredResult(
        precheck=precheck,
        overall_summary="Overall summary",
        findings=["Issue"],
        drills=["Drill"],
        enhanced_summary="Enhanced enabled.",
    )
    result.stage_breakdown = [
        {
            "name": "release",
            "status": "needs-work",
            "summary": "Release arc is too flat.",
            "focus_points": ["Lift the release arc."],
        }
    ]
    result.template_comparison = [
        "Compared with the ideal side-view template, the release arc is flatter."
    ]
    job_store.records["stage-job"] = JobRecord(
        id="stage-job",
        filename="demo.mp4",
        input_path=tmp_path / "demo.mp4",
        status=JobStatus.DONE,
        result=result,
    )

    client = TestClient(app)
    response = client.get("/jobs/stage-job")

    assert response.status_code == 200
    assert "出手阶段" in response.text
    assert "出手弧线偏平，容易让球路前冲。" in response.text
    assert "把出手弧线抬高一点。" in response.text
    assert "模板对比" in response.text
    assert "和理想侧面模板相比，这次出手弧线偏平。" in response.text


def test_result_page_hides_raw_json_artifact_links(tmp_path) -> None:
    precheck = PrecheckResult(
        score=0.76,
        confidence=ConfidenceLevel.HIGH,
        run_enhanced_analysis=True,
        view_type="45deg",
        summary="视频质量较好，适合做增强诊断。",
        reasons=["机位稳定"],
        recommendations=[],
    )
    job_store.records["artifact-job"] = JobRecord(
        id="artifact-job",
        filename="demo.mp4",
        input_path=tmp_path / "demo.mp4",
        status=JobStatus.DONE,
        result=LayeredResult(
            precheck=precheck,
            overall_summary="summary",
            findings=["Issue"],
            drills=["Drill"],
            artifacts=JobArtifacts(
                analysis_json="artifact_analysis.json",
                events_json="artifact_events.json",
                highlight_video="artifact_highlight.mp4",
            ),
        ),
    )

    client = TestClient(app)
    response = client.get("/jobs/artifact-job")

    assert response.status_code == 200
    assert "下载分析报告 JSON" not in response.text
    assert "下载事件时间轴 JSON" not in response.text
    assert "单独打开诊断视频" in response.text


def test_status_page_uses_api_polling_instead_of_full_page_reload(tmp_path) -> None:
    job_store.records["poll-job"] = JobRecord(
        id="poll-job",
        filename="demo.mp4",
        input_path=tmp_path / "demo.mp4",
        status=JobStatus.RENDERING,
        progress_percent=94,
        status_detail="正在生成诊断视频片段 2/3。",
    )

    client = TestClient(app)
    response = client.get("/jobs/poll-job")

    assert response.status_code == 200
    assert "/api/jobs/poll-job" in response.text
    assert "window.location.href = \"/jobs/poll-job\"" not in response.text
    assert 'id="progressFill"' in response.text
    assert 'id="etaText"' in response.text
    assert 'class="stage-dot stage-dot-live"' in response.text
    assert "window.location.replace(\"/jobs/poll-job\")" in response.text
    assert "视频已上传，正在分析" in response.text
    assert "正在生成诊断视频片段 2/3。" in response.text
    assert "94%" in response.text


def test_status_page_uses_larger_eta_for_large_videos(tmp_path) -> None:
    large_video = tmp_path / "huge.mp4"
    with large_video.open("wb") as handle:
        handle.truncate(220 * 1024 * 1024)

    job_store.records["big-job"] = JobRecord(
        id="big-job",
        filename="huge.mp4",
        input_path=large_video,
        status=JobStatus.RENDERING,
    )

    client = TestClient(app)
    response = client.get("/jobs/big-job")

    assert response.status_code == 200
    assert "预计还需 40 到 90 秒" in response.text
    assert "预计还需 3 到 8 秒" not in response.text


def test_resolve_server_defaults_to_localhost_when_not_on_render(monkeypatch) -> None:
    monkeypatch.delenv("HOST", raising=False)
    monkeypatch.delenv("PORT", raising=False)
    monkeypatch.delenv("RENDER", raising=False)

    assert resolve_server_host() == "127.0.0.1"
    assert resolve_server_port() == 7860


def test_resolve_server_uses_public_binding_for_render(monkeypatch) -> None:
    monkeypatch.delenv("HOST", raising=False)
    monkeypatch.setenv("PORT", "10000")
    monkeypatch.setenv("RENDER", "true")

    assert resolve_server_host() == "0.0.0.0"
    assert resolve_server_port() == 10000


def test_resolve_server_prefers_explicit_host(monkeypatch) -> None:
    monkeypatch.setenv("HOST", "0.0.0.0")
    monkeypatch.setenv("PORT", "9000")

    assert resolve_server_host() == "0.0.0.0"
    assert resolve_server_port() == 9000
