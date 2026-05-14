# Shot Diagnosis Web Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the basketball shot diagnosis prototype into a usable Web demo with asynchronous jobs, precheck gating, layered analysis, confidence reporting, and a product-shaped results flow.

**Architecture:** Keep the existing OpenCV-based detector, analyzer, and clip renderer as reusable low-level capabilities, but move orchestration into focused modules. Add a job model plus a precheck layer that decides whether a video gets base-only or enhanced analysis. Refactor the FastAPI app to create jobs, process them in the background, and expose status/result APIs instead of blocking the upload request.

**Tech Stack:** Python 3.12+, FastAPI, Uvicorn, OpenCV, NumPy, Pillow, imageio-ffmpeg, pytest, httpx

---

## File Map

### Create

- `app/models.py` - dataclasses for precheck, confidence, layered report, and job state
- `app/precheck.py` - video precheck scoring and quality gating
- `app/jobs.py` - in-memory job store and pipeline runner
- `tests/test_precheck.py` - precheck and eligibility tests
- `tests/test_jobs.py` - job pipeline and fallback tests
- `tests/test_web.py` - API tests for upload, status, and result flow

### Modify

- `app/analyzer.py` - return richer structured results and confidence hooks
- `app/clipper.py` - accept richer card payloads without changing rendering behavior
- `app/web.py` - replace synchronous page postback flow with async job endpoints and result rendering
- `requirements.txt` - add test dependencies
- `README.md` - document new run flow once code lands

### Keep Unchanged Unless Required

- `app/detector.py` - existing shot event detection
- `app/cli.py` - not in first-pass scope

## Task 1: Establish test baseline and dependencies

**Files:**
- Modify: `requirements.txt`
- Create: `tests/test_precheck.py`
- Create: `tests/test_jobs.py`
- Create: `tests/test_web.py`

- [ ] **Step 1: Write the failing API smoke test**

```python
from fastapi.testclient import TestClient

from app.web import app


def test_home_page_loads() -> None:
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "Basketball" in response.text
```

- [ ] **Step 2: Run test to verify it fails because test tooling is missing**

Run: `python -m pytest tests/test_web.py::test_home_page_loads -v`
Expected: FAIL with import or dependency error because `pytest` / `httpx` are not installed yet

- [ ] **Step 3: Add minimal test dependencies**

```text
fastapi==0.115.6
uvicorn[standard]==0.32.1
python-multipart==0.0.19
opencv-python==4.10.0.84
numpy==1.26.4
imageio-ffmpeg==0.5.1
pillow>=11.0.0
pytest==8.3.5
httpx==0.28.1
```

- [ ] **Step 4: Run the smoke test again**

Run: `python -m pytest tests/test_web.py::test_home_page_loads -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add requirements.txt tests/test_web.py
git commit -m "test: add web smoke test baseline"
```

If the project is not a git repository, skip the commit and record that fact in the execution notes.

## Task 2: Add precheck domain model and gating logic

**Files:**
- Create: `app/models.py`
- Create: `app/precheck.py`
- Test: `tests/test_precheck.py`

- [ ] **Step 1: Write the failing precheck tests**

```python
from app.precheck import ConfidenceLevel, score_precheck


def test_score_precheck_promotes_high_quality_side_view() -> None:
    result = score_precheck(
        width=1280,
        height=720,
        fps=30.0,
        duration=9.0,
        frame_count=270,
        focus_score=160.0,
        brightness_score=132.0,
        motion_score=0.82,
        subject_ratio=0.48,
        occlusion_ratio=0.05,
        view_hint="side",
    )

    assert result.confidence == ConfidenceLevel.HIGH
    assert result.run_enhanced_analysis is True
    assert result.view_type == "side"


def test_score_precheck_downgrades_low_quality_video() -> None:
    result = score_precheck(
        width=640,
        height=360,
        fps=12.0,
        duration=4.0,
        frame_count=48,
        focus_score=18.0,
        brightness_score=32.0,
        motion_score=0.21,
        subject_ratio=0.16,
        occlusion_ratio=0.46,
        view_hint="mixed",
    )

    assert result.confidence == ConfidenceLevel.LOW
    assert result.run_enhanced_analysis is False
    assert any("重拍" in note for note in result.recommendations)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_precheck.py -v`
Expected: FAIL with `ModuleNotFoundError` or missing names from `app.precheck`

- [ ] **Step 3: Write minimal model types**

```python
from dataclasses import dataclass, field
from enum import StrEnum


class ConfidenceLevel(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class PrecheckResult:
    score: float
    confidence: ConfidenceLevel
    run_enhanced_analysis: bool
    view_type: str
    summary: str
    reasons: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
```

- [ ] **Step 4: Write minimal precheck scorer**

```python
from app.models import ConfidenceLevel, PrecheckResult


def score_precheck(
    *,
    width: int,
    height: int,
    fps: float,
    duration: float,
    frame_count: int,
    focus_score: float,
    brightness_score: float,
    motion_score: float,
    subject_ratio: float,
    occlusion_ratio: float,
    view_hint: str,
) -> PrecheckResult:
    score = 0.0
    score += 0.22 if width >= 960 else 0.08
    score += 0.14 if fps >= 24 else 0.04
    score += min(0.18, focus_score / 900)
    score += 0.14 if 80 <= brightness_score <= 185 else 0.06
    score += min(0.12, motion_score * 0.15)
    score += 0.12 if subject_ratio >= 0.32 else 0.03
    score += 0.08 if occlusion_ratio <= 0.15 else 0.0
    score += 0.08 if view_hint in {"side", "45deg"} else 0.02
    score = round(min(score, 1.0), 3)

    if score >= 0.72:
        return PrecheckResult(
            score=score,
            confidence=ConfidenceLevel.HIGH,
            run_enhanced_analysis=True,
            view_type=view_hint,
            summary="视频质量较好，适合做增强诊断。",
            reasons=["机位和清晰度足够稳定"],
            recommendations=[],
        )

    if score >= 0.45:
        return PrecheckResult(
            score=score,
            confidence=ConfidenceLevel.MEDIUM,
            run_enhanced_analysis=False,
            view_type=view_hint,
            summary="视频可分析，但更适合基础诊断。",
            reasons=["存在部分质量限制"],
            recommendations=["尽量使用侧面或 45 度机位重拍。"],
        )

    return PrecheckResult(
        score=score,
        confidence=ConfidenceLevel.LOW,
        run_enhanced_analysis=False,
        view_type=view_hint,
        summary="视频质量不足，本次只建议作为参考。",
        reasons=["清晰度、帧率或遮挡影响明显"],
        recommendations=["建议重拍：保持侧面机位、完整入镜、画面稳定。"],
    )
```

- [ ] **Step 5: Run the precheck tests**

Run: `python -m pytest tests/test_precheck.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add app/models.py app/precheck.py tests/test_precheck.py
git commit -m "feat: add precheck scoring and confidence gating"
```

## Task 3: Add job store and layered pipeline orchestration

**Files:**
- Create: `app/jobs.py`
- Modify: `app/analyzer.py`
- Test: `tests/test_jobs.py`

- [ ] **Step 1: Write the failing job orchestration test**

```python
from pathlib import Path

from app.jobs import InMemoryJobStore, JobStatus


def test_job_store_runs_base_only_when_precheck_blocks_enhanced(tmp_path: Path) -> None:
    store = InMemoryJobStore(tmp_path)
    job = store.create_job("demo.mp4")

    store.complete_pipeline(
        job.id,
        precheck_score=0.31,
        detector_events=[],
        report_overall="基础结果",
    )

    saved = store.get(job.id)
    assert saved is not None
    assert saved.status == JobStatus.DONE
    assert saved.result is not None
    assert saved.result.precheck.run_enhanced_analysis is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_jobs.py -v`
Expected: FAIL with missing `app.jobs`

- [ ] **Step 3: Add job and result models**

```python
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from app.models import PrecheckResult


class JobStatus(StrEnum):
    UPLOADED = "uploaded"
    PRECHECK = "precheck"
    BASE_ANALYSIS = "base-analysis"
    ENHANCED_ANALYSIS = "enhanced-analysis"
    RENDERING = "rendering"
    DONE = "done"
    FAILED = "failed"


@dataclass
class JobArtifacts:
    analysis_json: str | None = None
    events_json: str | None = None
    highlight_video: str | None = None


@dataclass
class LayeredResult:
    precheck: PrecheckResult
    overall_summary: str
    enhanced_summary: str | None = None
    artifacts: JobArtifacts = field(default_factory=JobArtifacts)


@dataclass
class JobRecord:
    id: str
    filename: str
    input_path: Path
    status: JobStatus
    error: str | None = None
    result: LayeredResult | None = None
```

- [ ] **Step 4: Add minimal in-memory job store**

```python
import uuid
from pathlib import Path

from app.jobs import JobArtifacts, JobRecord, JobStatus, LayeredResult
from app.models import ConfidenceLevel, PrecheckResult


class InMemoryJobStore:
    def __init__(self, work_dir: Path) -> None:
        self.work_dir = work_dir
        self.records: dict[str, JobRecord] = {}

    def create_job(self, filename: str) -> JobRecord:
        job_id = uuid.uuid4().hex[:12]
        record = JobRecord(
            id=job_id,
            filename=filename,
            input_path=self.work_dir / filename,
            status=JobStatus.UPLOADED,
        )
        self.records[job_id] = record
        return record

    def get(self, job_id: str) -> JobRecord | None:
        return self.records.get(job_id)

    def complete_pipeline(
        self,
        job_id: str,
        *,
        precheck_score: float,
        detector_events: list[object],
        report_overall: str,
    ) -> None:
        record = self.records[job_id]
        confidence = ConfidenceLevel.HIGH if precheck_score >= 0.72 else ConfidenceLevel.LOW
        precheck = PrecheckResult(
            score=precheck_score,
            confidence=confidence,
            run_enhanced_analysis=precheck_score >= 0.72,
            view_type="side",
            summary="pipeline complete",
            reasons=[],
            recommendations=[],
        )
        record.status = JobStatus.DONE
        record.result = LayeredResult(precheck=precheck, overall_summary=report_overall)
```

- [ ] **Step 5: Run the job tests**

Run: `python -m pytest tests/test_jobs.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add app/jobs.py app/analyzer.py tests/test_jobs.py
git commit -m "feat: add in-memory job pipeline state"
```

## Task 4: Refactor the FastAPI upload flow into async jobs

**Files:**
- Modify: `app/web.py`
- Test: `tests/test_web.py`

- [ ] **Step 1: Expand the failing API tests**

```python
from fastapi.testclient import TestClient

from app.web import app


def test_create_job_returns_job_id(tmp_path, monkeypatch) -> None:
    client = TestClient(app)

    response = client.post(
        "/api/jobs",
        files={"video": ("clip.mp4", b"fake-bytes", "video/mp4")},
    )

    assert response.status_code == 202
    payload = response.json()
    assert "job_id" in payload
    assert payload["status"] == "uploaded"


def test_unknown_job_returns_404() -> None:
    client = TestClient(app)

    response = client.get("/api/jobs/missing")

    assert response.status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_web.py -v`
Expected: FAIL because `/api/jobs` and `/api/jobs/{job_id}` do not exist

- [ ] **Step 3: Add minimal API endpoints**

```python
@app.post("/api/jobs", status_code=202)
async def create_job(video: UploadFile = File(...)) -> dict[str, str]:
    job = job_store.create_job(video.filename or "upload.mp4")
    return {"job_id": job.id, "status": job.status.value}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, object]:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return {"job_id": job.id, "status": job.status.value, "error": job.error}
```

- [ ] **Step 4: Run the API tests**

Run: `python -m pytest tests/test_web.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/web.py tests/test_web.py
git commit -m "feat: add async job creation and status endpoints"
```

## Task 5: Connect real precheck and base analysis into the job runner

**Files:**
- Modify: `app/jobs.py`
- Modify: `app/precheck.py`
- Modify: `app/analyzer.py`
- Modify: `app/web.py`
- Test: `tests/test_jobs.py`

- [ ] **Step 1: Write failing pipeline tests for status progression**

```python
def test_job_runner_marks_failed_when_video_cannot_be_decoded(tmp_path: Path) -> None:
    store = InMemoryJobStore(tmp_path)
    job = store.create_job("broken.mp4")
    job.input_path.write_bytes(b"not-a-real-video")

    store.run_pipeline(job.id)

    saved = store.get(job.id)
    assert saved is not None
    assert saved.status == JobStatus.FAILED
    assert saved.error is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_jobs.py::test_job_runner_marks_failed_when_video_cannot_be_decoded -v`
Expected: FAIL because `run_pipeline` does not exist or does not mark failure

- [ ] **Step 3: Implement `run_pipeline` with staged status updates**

```python
def run_pipeline(self, job_id: str) -> None:
    record = self.records[job_id]
    try:
        record.status = JobStatus.PRECHECK
        precheck = analyze_video_precheck(record.input_path)

        record.status = JobStatus.BASE_ANALYSIS
        detector = BasketballShotDetector()
        try:
            events = detector.detect(record.input_path)
        finally:
            detector.close()
        report = analyze_shots(record.input_path, events, precheck=precheck)

        record.status = JobStatus.RENDERING
        record.result = build_layered_result(precheck=precheck, report=report, events=events)
        record.status = JobStatus.DONE
    except Exception as exc:
        record.status = JobStatus.FAILED
        record.error = str(exc)
```

- [ ] **Step 4: Run the job tests**

Run: `python -m pytest tests/test_jobs.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/jobs.py app/precheck.py app/analyzer.py app/web.py tests/test_jobs.py
git commit -m "feat: run staged analysis pipeline in job store"
```

## Task 6: Replace the old blocking result page with product-shaped status and result pages

**Files:**
- Modify: `app/web.py`
- Modify: `README.md`
- Test: `tests/test_web.py`

- [ ] **Step 1: Write failing result-page tests**

```python
def test_result_page_shows_confidence_and_summary(client_with_finished_job) -> None:
    response = client_with_finished_job.get("/jobs/demo-job")

    assert response.status_code == 200
    assert "可信度" in response.text
    assert "主要问题" in response.text
    assert "训练建议" in response.text
```

- [ ] **Step 2: Run the page test to verify it fails**

Run: `python -m pytest tests/test_web.py::test_result_page_shows_confidence_and_summary -v`
Expected: FAIL because result page route does not exist or lacks the new content

- [ ] **Step 3: Implement status and result page rendering**

```python
@app.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_page(job_id: str) -> str:
    job = require_job(job_id)
    if job.status is JobStatus.DONE and job.result is not None:
        return render_result_page(job)
    return render_status_page(job)
```

The result page must include:

- confidence badge
- precheck summary
- overall summary
- major findings
- training advice
- artifact links when present

- [ ] **Step 4: Run the full web test suite**

Run: `python -m pytest tests/test_web.py -v`
Expected: PASS

- [ ] **Step 5: Update README run flow**

```markdown
## Web Demo flow

1. Upload a basketball shooting video
2. Wait for precheck, base analysis, and optional enhanced analysis
3. Open the job result page to review confidence, findings, and artifacts
```

- [ ] **Step 6: Commit**

```bash
git add app/web.py README.md tests/test_web.py
git commit -m "feat: add job status and layered result pages"
```

## Task 7: Verification sweep

**Files:**
- Modify: none unless verification exposes defects

- [ ] **Step 1: Run the targeted test suite**

Run: `python -m pytest tests/test_precheck.py tests/test_jobs.py tests/test_web.py -v`
Expected: PASS

- [ ] **Step 2: Run the app import smoke check**

Run: `python -c "from app.web import app; print(app.title)"`
Expected: prints `Basketball Auto Clipper` or the updated app title

- [ ] **Step 3: Run an end-to-end manual startup check**

Run: `python -m app.web`
Expected: server starts without import errors

- [ ] **Step 4: Summarize any unfinished spec items**

Checklist:
- enhanced model integration still mocked or heuristic-backed?
- artifact rendering fully wired?
- low-confidence retry advice visible?
- async job polling flow present?

- [ ] **Step 5: Commit final verification fixes if any**

```bash
git add .
git commit -m "chore: verify shot diagnosis web demo refactor"
```

If there is no git repository, skip the commit and report verification evidence only.

## Self-Review

- Spec coverage check: this plan covers async jobs, precheck gating, base-vs-enhanced layering, confidence display, result pages, and verification. It does not implement a real paid enhanced model yet; instead it creates the interface and gating required by the spec.
- Placeholder scan: no `TODO` / `TBD` placeholders remain in task steps.
- Type consistency: `ConfidenceLevel`, `PrecheckResult`, `JobStatus`, and `LayeredResult` are introduced before downstream tasks reference them.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-13-shot-diagnosis-web-implementation.md`.

Default execution path for this session: Inline Execution. Implement tasks in this session with checkpoints and live verification evidence.
