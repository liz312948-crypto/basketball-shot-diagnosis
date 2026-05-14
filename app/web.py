from __future__ import annotations

import json
import os
import shutil
from html import escape
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .jobs import InMemoryJobStore, JobRecord, JobStatus


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
STATIC_DIR = BASE_DIR / "static"

for directory in (UPLOAD_DIR, OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="投篮诊断")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

job_store = InMemoryJobStore(UPLOAD_DIR, OUTPUT_DIR)

STAGE_PROGRESS: dict[str, int] = {
    JobStatus.UPLOADED.value: 10,
    JobStatus.PRECHECK.value: 28,
    JobStatus.BASE_ANALYSIS.value: 52,
    JobStatus.ENHANCED_ANALYSIS.value: 74,
    JobStatus.RENDERING.value: 90,
    JobStatus.DONE.value: 100,
    JobStatus.FAILED.value: 100,
}

STAGE_LABELS: dict[str, str] = {
    JobStatus.UPLOADED.value: "文件已接收，准备进入分析",
    JobStatus.PRECHECK.value: "正在检查机位、清晰度和可分析性",
    JobStatus.BASE_ANALYSIS.value: "正在执行基础动作诊断",
    JobStatus.ENHANCED_ANALYSIS.value: "正在执行增强阶段分析",
    JobStatus.RENDERING.value: "正在生成报告和诊断视频",
    JobStatus.DONE.value: "分析完成",
    JobStatus.FAILED.value: "分析失败",
}

STAGE_ETA: dict[str, str] = {
    JobStatus.UPLOADED.value: "预计还需 25 到 40 秒",
    JobStatus.PRECHECK.value: "预计还需 20 到 35 秒",
    JobStatus.BASE_ANALYSIS.value: "预计还需 12 到 24 秒",
    JobStatus.ENHANCED_ANALYSIS.value: "预计还需 8 到 16 秒",
    JobStatus.RENDERING.value: "预计还需 3 到 8 秒",
    JobStatus.DONE.value: "正在跳转结果页",
    JobStatus.FAILED.value: "任务已停止",
}

CONFIDENCE_LABELS: dict[str, str] = {
    "high": "高",
    "medium": "中",
    "low": "低",
}

VIEW_LABELS: dict[str, str] = {
    "side": "侧面",
    "45deg": "45 度",
    "mixed": "混合机位",
}

ENHANCED_STAGE_NAMES: dict[str, str] = {
    "setup": "准备阶段",
    "dip": "下蹲蓄力",
    "lift": "向上发力",
    "release": "出手阶段",
    "follow-through": "随球动作",
}

ENHANCED_STAGE_STATUS: dict[str, str] = {
    "stable": "稳定",
    "needs-work": "需改进",
    "limited-signal": "信号有限",
}

LEGACY_COPY_MAP: dict[str, str] = {
    "Enhanced analysis confirms that the clip is good enough for stage-by-stage feedback. The weakest stages should drive the next training block.": "增强分析认为这段视频足够清晰，可以继续看分阶段反馈。后续训练优先围绕得分最低的阶段进行修正。",
    "Enhanced enabled.": "视频质量足够，已进入增强诊断。",
    "The body stays controlled before the shot.": "投篮前的身体控制比较稳定。",
    "The gather-to-dip timing looks segmented and loses flow.": "举球到下蹲之间有停顿，动作连贯性不足。",
    "The upward lift creates enough height into the release.": "上升发力带出的高度基本够用。",
    "The release path drifts or flattens too early, reducing repeatability.": "出手路线过早变平或左右漂移，影响重复性。",
    "The clip captures enough post-release follow-through to support feedback.": "随球动作画面足够完整，可以继续给反馈。",
    "Release arc is too flat.": "出手弧线偏平，容易让球路前冲。",
    "Keep the torso quiet before the gather.": "接球前保持躯干安静。",
    "Land balanced after the shot.": "出手落地后维持平衡。",
    "Connect the dip to the upward drive.": "把下蹲和上升发力连起来。",
    "Avoid pausing before the lift.": "避免上举前出现停顿。",
    "Drive up before extending the arm.": "先向上发力，再伸臂出手。",
    "Create more vertical lift into the shot.": "增加更明确的向上抬升。",
    "Finish with a higher release arc.": "把出手弧线抬高一点。",
    "Lift the release arc.": "把出手弧线抬高一点。",
    "Keep the release line centered.": "尽量让出手线保持居中。",
    "Hold the finish for a beat.": "出手后多停一拍。",
    "Keep the shooting line visible after release.": "让随球动作尽量留在画面里。",
    "Compared with the ideal side-view template, the release arc is within a usable range.": "和理想侧面模板相比，这次出手弧线仍在可接受范围内。",
    "The ball path shows more horizontal drift than the template baseline.": "这次球路的水平漂移比模板基线更大。",
    "Compared with the ideal side-view template, the release arc is flatter.": "和理想侧面模板相比，这次出手弧线偏平。",
    "The ball path stays close to the template centerline.": "这次球路基本贴近模板的中线。",
}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/process")
def process_get() -> RedirectResponse:
    return RedirectResponse(url="/", status_code=303)


@app.post("/process")
async def process_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    pre_seconds: float = Form(5.0),
    post_seconds: float = Form(5.0),
    threshold: float = Form(0.38),
    sample_every: float = Form(0.35),
) -> RedirectResponse:
    del pre_seconds, post_seconds, threshold, sample_every
    job = await _create_job_from_upload(background_tasks, video)
    return RedirectResponse(url=f"/jobs/{job.id}", status_code=303)


@app.post("/api/jobs", status_code=202)
async def create_job(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
) -> dict[str, object]:
    job = await _create_job_from_upload(background_tasks, video)
    return {
        "job_id": job.id,
        "status": job.status.value,
        "progress_percent": job.progress_percent,
        "status_detail": job.status_detail,
    }


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, object]:
    job = _require_job(job_id)
    payload: dict[str, object] = {
        "job_id": job.id,
        "status": job.status.value,
        "error": job.error,
        "progress_percent": job.progress_percent,
        "status_detail": job.status_detail,
    }
    if job.result is not None:
        payload["result"] = {
            "confidence": job.result.precheck.confidence.value,
            "summary": job.result.overall_summary,
            "findings": job.result.findings,
            "drills": job.result.drills,
            "enhanced_summary": job.result.enhanced_summary,
            "stage_breakdown": job.result.stage_breakdown,
            "template_comparison": job.result.template_comparison,
        }
    return payload


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_page(job_id: str) -> str:
    job = _require_job(job_id)
    if job.status is JobStatus.DONE and job.result is not None:
        return _render_result_page(job)
    return _render_status_page(job)


@app.api_route("/download/{filename}", methods=["GET", "HEAD"])
def download(filename: str) -> FileResponse:
    path = _download_path(filename)
    return FileResponse(path, filename=filename)


@app.api_route("/media/{filename}", methods=["GET", "HEAD"])
def media(filename: str) -> FileResponse:
    path = _download_path(filename)
    return FileResponse(path, media_type="video/mp4", content_disposition_type="inline")


async def _create_job_from_upload(background_tasks: BackgroundTasks, video: UploadFile) -> JobRecord:
    job = job_store.create_job(video.filename or "upload.mp4")
    with job.input_path.open("wb") as handle:
        shutil.copyfileobj(video.file, handle)
    background_tasks.add_task(job_store.run_pipeline, job.id)
    return job


def _require_job(job_id: str) -> JobRecord:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


def _render_status_page(job: JobRecord) -> str:
    stage_value = job.status.value
    progress = job.progress_percent if job.progress_percent is not None else STAGE_PROGRESS.get(stage_value, 0)
    stage_label = _translate_stage_label(stage_value)
    stage_eta = job.status_detail or _translate_stage_eta(stage_value, job)
    stage_meta_json = json.dumps(
        {
            key: {
                "progress": STAGE_PROGRESS[key],
                "label": STAGE_LABELS[key],
                "eta": _translate_stage_eta(key, job),
            }
            for key in STAGE_PROGRESS
        },
        ensure_ascii=False,
    )

    poll_script = ""
    if job.status not in {JobStatus.DONE, JobStatus.FAILED}:
        poll_script = f"""
        <script>
          const stageEl = document.getElementById("jobStage");
          const stageLabelEl = document.getElementById("jobStageLabel");
          const etaEl = document.getElementById("etaText");
          const errorEl = document.getElementById("jobError");
          const progressFillEl = document.getElementById("progressFill");
          const progressValueEl = document.getElementById("progressValue");
          const stageMeta = {stage_meta_json};

          function updateStageUi(status) {{
            const meta = stageMeta[status] || {{ progress: 0, label: status, eta: "正在估算剩余时间" }};
            if (stageEl) stageEl.textContent = meta.label;
            if (stageLabelEl) stageLabelEl.textContent = meta.label;
            if (etaEl) etaEl.textContent = meta.eta;
            if (progressFillEl) progressFillEl.style.width = meta.progress + "%";
            if (progressValueEl) progressValueEl.textContent = meta.progress + "%";
          }}

          async function pollJob() {{
            try {{
              const response = await fetch("/api/jobs/{job.id}", {{ cache: "no-store" }});
              if (!response.ok) {{
                throw new Error("状态轮询失败");
              }}
              const payload = await response.json();
              updateStageUi(payload.status);
              if (typeof payload.progress_percent === "number") {{
                if (progressFillEl) progressFillEl.style.width = payload.progress_percent + "%";
                if (progressValueEl) progressValueEl.textContent = payload.progress_percent + "%";
              }}
              if (payload.status_detail && etaEl) {{
                etaEl.textContent = payload.status_detail;
              }}
              if (payload.error && errorEl) {{
                errorEl.textContent = "错误：" + payload.error;
              }}
              if (payload.status === "done" || payload.status === "failed") {{
                window.location.replace("/jobs/{job.id}");
                return;
              }}
            }} catch (error) {{
              if (errorEl) {{
                errorEl.textContent = error.message || "状态轮询失败";
              }}
            }}
            setTimeout(pollJob, 2200);
          }}

          updateStageUi("{escape(stage_value)}");
          setTimeout(pollJob, 2200);
        </script>
        """

    error_html = f"错误：{escape(job.error)}" if job.status is JobStatus.FAILED and job.error else ""
    return f"""
    <!doctype html>
    <html lang="zh-CN">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>分析中</title>
      <link rel="stylesheet" href="/static/styles.css">
    </head>
    <body>
      <main class="shell">
        <section class="panel status-panel">
          <p class="eyebrow">投篮诊断任务</p>
          <div class="status-hero">
            <div class="status-copy">
              <h1>视频已上传，正在分析</h1>
              <p class="overall">当前阶段：<strong id="jobStage">{escape(stage_label)}</strong></p>
              <p id="jobStageLabel" class="status-label">{escape(stage_label)}</p>
              <p id="etaText" class="status-eta">{escape(stage_eta)}</p>
            </div>
            <div class="status-orb">
              <span class="stage-dot stage-dot-live"></span>
              <span id="progressValue" class="status-percent">{progress}%</span>
            </div>
          </div>
          <div class="progress-shell" aria-label="分析进度">
            <div id="progressFill" class="progress-fill" style="width: {progress}%"></div>
          </div>
          <div class="status-grid">
            <article class="status-card">
              <h3>1. 已接收视频</h3>
              <p>文件已经进入分析队列，准备读取元数据。</p>
            </article>
            <article class="status-card">
              <h3>2. 视频预检</h3>
              <p>检查机位、清晰度、主体完整度和可分析性。</p>
            </article>
            <article class="status-card">
              <h3>3. 基础诊断</h3>
              <p>提取主要动作问题和训练建议。</p>
            </article>
            <article class="status-card">
              <h3>4. 增强诊断</h3>
              <p>质量足够时追加阶段拆解和模板对比。</p>
            </article>
            <article class="status-card">
              <h3>5. 结果生成</h3>
              <p>输出报告、JSON 文件和可直接播放的诊断视频。</p>
            </article>
          </div>
          <p id="jobError" class="error">{error_html}</p>
          <div class="actions">
            <a class="primary" href="/">继续上传</a>
          </div>
        </section>
      </main>
      {poll_script}
    </body>
    </html>
    """


def _render_result_page(job: JobRecord) -> str:
    assert job.result is not None
    result = job.result

    confidence_label = _translate_confidence(result.precheck.confidence.value)
    view_label = _translate_view(result.precheck.view_type)
    findings = _render_list_items(result.findings)
    drills = _render_list_items(result.drills)
    reasons = _render_list_items(result.precheck.reasons)
    recommendations = _render_list_items(result.precheck.recommendations)
    stage_cards = _render_stage_cards(result.stage_breakdown)
    template_notes = _render_list_items(result.template_comparison)
    embedded_video = _render_embedded_video(result)
    artifacts_html = _render_artifact_links(result)
    enhanced_summary = (
        f"<p class='overall'>{escape(_translate_legacy_copy(result.enhanced_summary))}</p>"
        if result.enhanced_summary
        else "<p class='overall'>这段视频没有进入增强诊断，因此本页只展示基础诊断结果。</p>"
    )

    return f"""
    <!doctype html>
    <html lang="zh-CN">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>投篮诊断结果</title>
      <link rel="stylesheet" href="/static/styles.css">
    </head>
    <body>
      <main class="shell">
        <section class="panel">
          <div class="title-row">
            <div>
              <p class="eyebrow">投篮诊断结果</p>
              <h1>本次投篮诊断已完成</h1>
            </div>
            <span class="badge">可信度 {escape(confidence_label)}</span>
          </div>
          <p class="overall">{escape(result.overall_summary)}</p>
          <p class="metrics">机位：{escape(view_label)} | 预检得分：{result.precheck.score:.2f}</p>
          <p class="overall">可信度说明：{escape(result.precheck.summary)}</p>
          {enhanced_summary}
          <div class="actions">
            <a class="primary" href="/">继续上传</a>
          </div>
        </section>

        <section class="panel analysis-panel">
          <h2>诊断视频</h2>
          {embedded_video}
        </section>

        <section class="panel analysis-panel">
          <h2>主要问题</h2>
          <ul>{findings}</ul>
        </section>

        <section class="panel analysis-panel">
          <h2>训练建议</h2>
          <ul>{drills}</ul>
        </section>

        <section class="panel analysis-panel">
          <h2>可信度说明</h2>
          <ul>{reasons or "<li>当前没有额外的可信度补充说明。</li>"}</ul>
          <h3>重拍建议</h3>
          <ul>{recommendations or "<li>当前视频质量已达到本次分析要求，无需额外重拍建议。</li>"}</ul>
        </section>

        <section class="panel analysis-panel">
          <h2>阶段拆解</h2>
          <div class="analysis-grid">{stage_cards or "<p class='muted'>本次没有可展示的阶段拆解数据。</p>"}</div>
        </section>

        <section class="panel analysis-panel">
          <h2>模板对比</h2>
          <ul>{template_notes or "<li>本次没有额外的模板对比说明。</li>"}</ul>
        </section>

        <section class="panel analysis-panel">
          <h2>视频操作</h2>
          <div class="actions">{artifacts_html}</div>
        </section>
      </main>
    </body>
    </html>
    """


def _render_embedded_video(result: object) -> str:
    video_name = getattr(getattr(result, "artifacts", None), "highlight_video", None)
    if not video_name:
        return "<p class='muted'>本次没有生成可播放的诊断视频。</p>"
    safe_name = escape(str(video_name))
    return f"""
    <video class="shot-video result-video" src="/media/{safe_name}" controls preload="metadata"></video>
    <p class="metrics">诊断视频已直接嵌入网页。识别到关键问题时会插入定格画面，方便你直接停在错误帧上看动作细节。</p>
    """


def _render_stage_cards(stage_breakdown: list[dict[str, object]]) -> str:
    if not stage_breakdown:
        return ""

    cards: list[str] = []
    for stage in stage_breakdown:
        name = _translate_stage_name(str(stage.get("name", "")))
        status = _translate_stage_status(str(stage.get("status", "")))
        summary = escape(_translate_legacy_copy(stage.get("summary", "")))
        focus_points = "".join(
            f"<li>{escape(_translate_legacy_copy(item))}</li>" for item in stage.get("focus_points", [])
        )
        cards.append(
            f"""
            <article class="analysis-card">
              <div class="card-head">
                <h3>{escape(name)}</h3>
                <span class="level">{escape(status)}</span>
              </div>
              <p>{summary}</p>
              <ul>{focus_points}</ul>
            </article>
            """
        )
    return "".join(cards)


def _render_artifact_links(result: object) -> str:
    artifacts = getattr(result, "artifacts", None)
    links: list[str] = []
    if getattr(artifacts, "highlight_video", None):
        highlight_video = escape(str(artifacts.highlight_video))
        links.append(
            f'<a href="/media/{highlight_video}" target="_blank" rel="noopener">单独打开诊断视频</a>'
        )
    return "".join(links) or "<span class='muted'>本次没有额外的视频按钮。</span>"


def _render_list_items(items: list[str]) -> str:
    return "".join(f"<li>{escape(_translate_legacy_copy(item))}</li>" for item in items)


def _translate_confidence(value: str) -> str:
    return CONFIDENCE_LABELS.get(value, value)


def _translate_view(value: str) -> str:
    return VIEW_LABELS.get(value, value)


def _translate_stage_name(value: str) -> str:
    return ENHANCED_STAGE_NAMES.get(value, value)


def _translate_stage_status(value: str) -> str:
    return ENHANCED_STAGE_STATUS.get(value, value)


def _translate_legacy_copy(value: object) -> str:
    text = str(value)
    return LEGACY_COPY_MAP.get(text, text)


def _translate_stage_label(stage_value: str) -> str:
    return STAGE_LABELS.get(stage_value, stage_value)


def _translate_stage_eta(stage_value: str, job: JobRecord | None = None) -> str:
    if stage_value in {JobStatus.DONE.value, JobStatus.FAILED.value}:
        return STAGE_ETA.get(stage_value, "正在估算剩余时间")
    if job is None:
        return STAGE_ETA.get(stage_value, "正在估算剩余时间")

    size_bytes = _safe_file_size(job.input_path)
    if size_bytes >= 200 * 1024 * 1024:
        eta = {
            JobStatus.UPLOADED.value: "大视频预计还需 2 到 4 分钟",
            JobStatus.PRECHECK.value: "预计还需 90 到 180 秒",
            JobStatus.BASE_ANALYSIS.value: "预计还需 60 到 140 秒",
            JobStatus.ENHANCED_ANALYSIS.value: "预计还需 50 到 120 秒",
            JobStatus.RENDERING.value: "预计还需 40 到 90 秒",
        }
        return eta.get(stage_value, "正在估算剩余时间")

    if size_bytes >= 80 * 1024 * 1024:
        eta = {
            JobStatus.UPLOADED.value: "预计还需 1 到 2 分钟",
            JobStatus.PRECHECK.value: "预计还需 50 到 100 秒",
            JobStatus.BASE_ANALYSIS.value: "预计还需 35 到 80 秒",
            JobStatus.ENHANCED_ANALYSIS.value: "预计还需 25 到 60 秒",
            JobStatus.RENDERING.value: "预计还需 20 到 45 秒",
        }
        return eta.get(stage_value, "正在估算剩余时间")

    return STAGE_ETA.get(stage_value, "正在估算剩余时间")


def _safe_file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _download_path(filename: str) -> Path:
    safe_name = Path(filename).name
    if safe_name != filename:
        raise HTTPException(status_code=400, detail="invalid filename")

    for directory in (OUTPUT_DIR, UPLOAD_DIR):
        path = directory / safe_name
        if path.exists() and path.is_file():
            return path
    raise HTTPException(status_code=404, detail="file not found")


def main() -> None:
    import uvicorn

    uvicorn.run(
        "app.web:app",
        host=resolve_server_host(),
        port=resolve_server_port(),
        reload=False,
    )


def resolve_server_host() -> str:
    explicit = os.getenv("HOST")
    if explicit:
        return explicit
    if os.getenv("RENDER"):
        return "0.0.0.0"
    return "127.0.0.1"


def resolve_server_port() -> int:
    value = os.getenv("PORT")
    if not value:
        return 7860
    return int(value)


if __name__ == "__main__":
    main()
