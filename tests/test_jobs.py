from pathlib import Path

from app.detector import ShotEvent
from app.jobs import InMemoryJobStore, JobStatus, _select_render_events


def test_job_store_runs_base_only_when_precheck_blocks_enhanced(tmp_path: Path) -> None:
    store = InMemoryJobStore(tmp_path, tmp_path / "outputs")
    job = store.create_job("demo.mp4")

    store.complete_pipeline(
        job.id,
        precheck_score=0.31,
        detector_events=[],
        report_overall="base result",
    )

    saved = store.get(job.id)
    assert saved is not None
    assert saved.status == JobStatus.DONE
    assert saved.result is not None
    assert saved.result.precheck.run_enhanced_analysis is False


def test_job_runner_marks_failed_when_video_cannot_be_decoded(tmp_path: Path) -> None:
    store = InMemoryJobStore(tmp_path, tmp_path / "outputs")
    job = store.create_job("broken.mp4")
    job.input_path.write_bytes(b"not-a-real-video")

    store.run_pipeline(job.id)

    saved = store.get(job.id)
    assert saved is not None
    assert saved.status == JobStatus.FAILED
    assert saved.error is not None


def test_job_store_reloads_persisted_job_records(tmp_path: Path) -> None:
    store = InMemoryJobStore(tmp_path, tmp_path / "outputs")
    job = store.create_job("demo.mp4")

    store.complete_pipeline(
        job.id,
        precheck_score=0.82,
        detector_events=[],
        report_overall="enhanced ready",
    )

    reloaded = InMemoryJobStore(tmp_path, tmp_path / "outputs")
    saved = reloaded.get(job.id)
    assert saved is not None
    assert saved.status == JobStatus.DONE
    assert saved.result is not None
    assert saved.result.precheck.confidence.value == "high"


def test_job_store_get_recovers_missing_in_memory_record_from_disk(tmp_path: Path) -> None:
    store = InMemoryJobStore(tmp_path, tmp_path / "outputs")
    job = store.create_job("demo.mp4")
    store.records.pop(job.id)

    recovered = store.get(job.id)

    assert recovered is not None
    assert recovered.id == job.id
    assert recovered.status == JobStatus.UPLOADED


def test_select_render_events_limits_hosted_demo_large_video(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RENDER", "true")
    video = tmp_path / "big.mp4"
    with video.open("wb") as handle:
        handle.truncate(250 * 1024 * 1024)

    events = [
        ShotEvent(time=float(index), score=0.9 - index * 0.01, reason="test")
        for index in range(10)
    ]

    selected = _select_render_events(events, video)

    assert len(selected) == 2
    assert [event.time for event in selected] == [0.0, 1.0]


def test_select_render_events_limits_hosted_demo_small_video(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RENDER", "true")
    video = tmp_path / "small.mp4"
    video.write_bytes(b"small")

    events = [
        ShotEvent(time=float(index), score=0.95 - index * 0.01, reason="test")
        for index in range(5)
    ]

    selected = _select_render_events(events, video)

    assert len(selected) == 3
    assert [event.time for event in selected] == [0.0, 1.0, 2.0]
