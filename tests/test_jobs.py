from pathlib import Path

from app.jobs import InMemoryJobStore, JobStatus


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
