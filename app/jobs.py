from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path

from .analyzer import AnalysisReport, analyze_shots, write_analysis_json
from .clipper import render_vertical_highlights, write_events_json
from .detector import BasketballShotDetector, ShotEvent
from .enhanced_analysis import EnhancedAnalysis, build_enhanced_analysis
from .models import ConfidenceLevel, PrecheckResult
from .precheck import analyze_video_precheck


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
    findings: list[str] = field(default_factory=list)
    drills: list[str] = field(default_factory=list)
    enhanced_summary: str | None = None
    stage_breakdown: list[dict[str, object]] = field(default_factory=list)
    template_comparison: list[str] = field(default_factory=list)
    artifacts: JobArtifacts = field(default_factory=JobArtifacts)


@dataclass
class JobRecord:
    id: str
    filename: str
    input_path: Path
    status: JobStatus
    error: str | None = None
    result: LayeredResult | None = None
    progress_percent: int | None = None
    status_detail: str | None = None


class InMemoryJobStore:
    def __init__(self, work_dir: Path, output_dir: Path | None = None) -> None:
        self.work_dir = work_dir
        self.output_dir = output_dir or work_dir.parent / "outputs"
        self.state_dir = self.work_dir.parent / "jobs"
        self.records: dict[str, JobRecord] = {}
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._load_records()

    def create_job(self, filename: str) -> JobRecord:
        job_id = uuid.uuid4().hex[:12]
        safe_name = Path(filename or "upload.mp4").name
        record = JobRecord(
            id=job_id,
            filename=safe_name,
            input_path=self.work_dir / f"{job_id}_{safe_name}",
            status=JobStatus.UPLOADED,
            progress_percent=10,
            status_detail="文件已接收，准备进入分析。",
        )
        self.records[job_id] = record
        self._persist_record(record)
        return record

    def save_upload(self, job_id: str, source_path: Path) -> JobRecord:
        record = self.records[job_id]
        shutil.copyfile(source_path, record.input_path)
        self._persist_record(record)
        return record

    def get(self, job_id: str) -> JobRecord | None:
        record = self.records.get(job_id)
        if record is not None:
            return record
        loaded = self._load_record_from_path(self._state_path(job_id))
        if loaded is not None:
            self.records[job_id] = loaded
        return loaded

    def complete_pipeline(
        self,
        job_id: str,
        *,
        precheck_score: float,
        detector_events: list[ShotEvent],
        report_overall: str,
    ) -> None:
        del detector_events
        record = self.records[job_id]
        confidence = (
            ConfidenceLevel.HIGH
            if precheck_score >= 0.72
            else ConfidenceLevel.MEDIUM
            if precheck_score >= 0.45
            else ConfidenceLevel.LOW
        )
        precheck = PrecheckResult(
            score=precheck_score,
            confidence=confidence,
            run_enhanced_analysis=precheck_score >= 0.72,
            view_type="side" if precheck_score >= 0.45 else "mixed",
            summary="测试环境下的模拟流水线结果。",
            reasons=[],
            recommendations=["建议使用稳定的侧面机位重拍。"] if confidence is ConfidenceLevel.LOW else [],
        )
        record.status = JobStatus.DONE
        record.progress_percent = 100
        record.status_detail = "分析完成。"
        record.result = LayeredResult(
            precheck=precheck,
            overall_summary=report_overall,
            findings=[],
            drills=[],
            enhanced_summary="视频质量足够，已进入增强诊断。" if precheck.run_enhanced_analysis else None,
        )
        self._persist_record(record)

    def run_pipeline(self, job_id: str) -> None:
        record = self.records[job_id]
        try:
            record.status = JobStatus.PRECHECK
            record.progress_percent = 28
            record.status_detail = "正在检查机位、清晰度和可分析性。"
            self._persist_record(record)
            precheck = analyze_video_precheck(record.input_path)

            record.status = JobStatus.BASE_ANALYSIS
            record.progress_percent = 52
            record.status_detail = "正在提取投篮事件并执行基础动作诊断。"
            self._persist_record(record)
            detector = BasketballShotDetector()
            try:
                events = detector.detect(record.input_path)
            finally:
                detector.close()

            report = analyze_shots(record.input_path, events)
            enhanced: EnhancedAnalysis | None = None
            if precheck.run_enhanced_analysis and report.analyses:
                record.status = JobStatus.ENHANCED_ANALYSIS
                record.progress_percent = 74
                record.status_detail = "视频质量足够，正在生成增强阶段分析。"
                self._persist_record(record)
                enhanced = build_enhanced_analysis(precheck=precheck, report=report)

            record.status = JobStatus.RENDERING
            record.progress_percent = 90
            record.status_detail = "正在准备诊断视频和结果页资源。"
            self._persist_record(record)
            artifacts = self._write_artifacts(record, report, events)
            record.result = build_layered_result(
                precheck=precheck,
                report=report,
                events=events,
                artifacts=artifacts,
                enhanced=enhanced,
            )
            record.status = JobStatus.DONE
            record.progress_percent = 100
            record.status_detail = "分析完成。"
            self._persist_record(record)
        except Exception as exc:
            record.status = JobStatus.FAILED
            record.error = str(exc)
            record.progress_percent = 100
            record.status_detail = "任务失败。"
            self._persist_record(record)

    def _write_artifacts(
        self,
        record: JobRecord,
        report: AnalysisReport,
        events: list[ShotEvent],
    ) -> JobArtifacts:
        artifacts = JobArtifacts()
        events_path = self.output_dir / f"{record.id}_events.json"
        analysis_path = self.output_dir / f"{record.id}_analysis.json"
        write_events_json(events, events_path)
        write_analysis_json(report, analysis_path)
        artifacts.events_json = events_path.name
        artifacts.analysis_json = analysis_path.name

        if events:
            output_video = self.output_dir / f"{record.id}_highlight.mp4"
            try:
                render_events = _select_render_events(events, record.input_path)
                record.progress_percent = 91
                record.status_detail = f"正在生成诊断视频片段，预计处理 {len(render_events)} 个关键投篮。"
                self._persist_record(record)
                render_vertical_highlights(
                    record.input_path,
                    render_events,
                    output_video,
                    diagnosis_cards=_diagnosis_cards_data(report, render_events),
                    pre_seconds=_render_pre_seconds(record.input_path),
                    post_seconds=_render_post_seconds(record.input_path),
                    slate_seconds=_render_slate_seconds(record.input_path),
                    freeze_seconds=_render_freeze_seconds(record.input_path),
                    target_width=_render_target_width(record.input_path),
                    target_height=_render_target_height(record.input_path),
                    progress_callback=lambda current, total: self._update_render_progress(record, current, total),
                )
                record.progress_percent = 99
                record.status_detail = "诊断视频已生成，正在整理最终结果。"
                self._persist_record(record)
                artifacts.highlight_video = output_video.name
            except Exception:
                artifacts.highlight_video = None
        return artifacts

    def _update_render_progress(self, record: JobRecord, current: int, total: int) -> None:
        total = max(1, total)
        # Keep rendering inside 91-98 so the final jump to done still feels coherent.
        progress = 91 + min(7, int((current / total) * 7))
        record.progress_percent = progress
        record.status_detail = f"正在生成诊断视频片段 {current}/{total}。"
        self._persist_record(record)

    def _state_path(self, job_id: str) -> Path:
        return self.state_dir / f"{job_id}.json"

    def _persist_record(self, record: JobRecord) -> None:
        payload: dict[str, object] = {
            "id": record.id,
            "filename": record.filename,
            "input_path": str(record.input_path),
            "status": record.status.value,
            "error": record.error,
            "progress_percent": record.progress_percent,
            "status_detail": record.status_detail,
            "result": None,
        }
        if record.result is not None:
            payload["result"] = {
                "precheck": {
                    "score": record.result.precheck.score,
                    "confidence": record.result.precheck.confidence.value,
                    "run_enhanced_analysis": record.result.precheck.run_enhanced_analysis,
                    "view_type": record.result.precheck.view_type,
                    "summary": record.result.precheck.summary,
                    "reasons": record.result.precheck.reasons,
                    "recommendations": record.result.precheck.recommendations,
                },
                "overall_summary": record.result.overall_summary,
                "findings": record.result.findings,
                "drills": record.result.drills,
                "enhanced_summary": record.result.enhanced_summary,
                "stage_breakdown": record.result.stage_breakdown,
                "template_comparison": record.result.template_comparison,
                "artifacts": asdict(record.result.artifacts),
            }
        self._state_path(record.id).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_records(self) -> None:
        for path in self.state_dir.glob("*.json"):
            loaded = self._load_record_from_path(path)
            if loaded is not None:
                self.records[loaded.id] = loaded

    def _load_record_from_path(self, path: Path) -> JobRecord | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            result_payload = payload.get("result")
            result = None
            if isinstance(result_payload, dict):
                pre = result_payload["precheck"]
                result = LayeredResult(
                    precheck=PrecheckResult(
                        score=float(pre["score"]),
                        confidence=ConfidenceLevel(str(pre["confidence"])),
                        run_enhanced_analysis=bool(pre["run_enhanced_analysis"]),
                        view_type=str(pre["view_type"]),
                        summary=str(pre["summary"]),
                        reasons=list(pre.get("reasons", [])),
                        recommendations=list(pre.get("recommendations", [])),
                    ),
                    overall_summary=str(result_payload.get("overall_summary", "")),
                    findings=list(result_payload.get("findings", [])),
                    drills=list(result_payload.get("drills", [])),
                    enhanced_summary=(
                        str(result_payload["enhanced_summary"])
                        if result_payload.get("enhanced_summary") is not None
                        else None
                    ),
                    stage_breakdown=list(result_payload.get("stage_breakdown", [])),
                    template_comparison=list(result_payload.get("template_comparison", [])),
                    artifacts=JobArtifacts(**result_payload.get("artifacts", {})),
                )
            return JobRecord(
                id=str(payload["id"]),
                filename=str(payload["filename"]),
                input_path=Path(str(payload["input_path"])),
                status=JobStatus(str(payload["status"])),
                error=(str(payload["error"]) if payload.get("error") is not None else None),
                result=result,
                progress_percent=(
                    int(payload["progress_percent"])
                    if payload.get("progress_percent") is not None
                    else None
                ),
                status_detail=(
                    str(payload["status_detail"])
                    if payload.get("status_detail") is not None
                    else None
                ),
            )
        except Exception:
            return None


def build_layered_result(
    *,
    precheck: PrecheckResult,
    report: AnalysisReport,
    events: list[ShotEvent],
    artifacts: JobArtifacts | None = None,
    enhanced: EnhancedAnalysis | None = None,
) -> LayeredResult:
    findings: list[str] = []
    drills: list[str] = []
    for item in report.analyses[:3]:
        findings.extend(item.findings[:2])
        drills.extend(item.drills[:1])
    if not findings:
        findings.append("暂时没有检测到足够清晰的投篮事件，无法进一步标记具体问题。")
    if not drills:
        drills.extend(precheck.recommendations or ["建议补拍更稳定的侧面机位视频。"])
    return LayeredResult(
        precheck=precheck,
        overall_summary=report.overall,
        findings=findings[:5],
        drills=drills[:4],
        enhanced_summary=(enhanced.summary if enhanced is not None else None),
        stage_breakdown=([asdict(stage) for stage in enhanced.stages] if enhanced is not None else []),
        template_comparison=(enhanced.template_comparison if enhanced is not None else []),
        artifacts=artifacts or JobArtifacts(),
    )


def _diagnosis_cards_data(report: AnalysisReport, events: list[ShotEvent] | None = None) -> list[dict[str, object]]:
    selected_times = {round(event.time, 2) for event in events} if events is not None else None
    cards: list[dict[str, object]] = []
    for item in report.analyses:
        if selected_times is not None and round(item.time, 2) not in selected_times:
            continue
        cards.append(
            {
                "title": f"第 {item.index} 次投篮 · {item.time:.2f} 秒",
                "level": item.level,
                "summary": item.summary,
                "findings": item.findings,
                "drills": item.drills,
            }
        )
    return cards


def _select_render_events(events: list[ShotEvent], input_path: Path) -> list[ShotEvent]:
    if len(events) <= 3:
        return events
    try:
        size_bytes = input_path.stat().st_size
    except OSError:
        size_bytes = 0

    if _is_hosted_demo():
        limit = 3
        if size_bytes >= 200 * 1024 * 1024:
            limit = 2
        elif size_bytes >= 80 * 1024 * 1024:
            limit = 3
    else:
        limit = 8
        if size_bytes >= 200 * 1024 * 1024:
            limit = 4
        elif size_bytes >= 80 * 1024 * 1024:
            limit = 6

    ranked = sorted(events, key=lambda event: event.score, reverse=True)[:limit]
    return sorted(ranked, key=lambda event: event.time)


def _render_target_width(input_path: Path) -> int:
    try:
        size_bytes = input_path.stat().st_size
    except OSError:
        size_bytes = 0
    if _is_hosted_demo():
        return 540 if size_bytes >= 80 * 1024 * 1024 else 720
    return 720 if size_bytes >= 80 * 1024 * 1024 else 1080


def _render_target_height(input_path: Path) -> int:
    try:
        size_bytes = input_path.stat().st_size
    except OSError:
        size_bytes = 0
    if _is_hosted_demo():
        return 960 if size_bytes >= 80 * 1024 * 1024 else 1280
    return 1280 if size_bytes >= 80 * 1024 * 1024 else 1920


def _render_pre_seconds(input_path: Path) -> float:
    return 2.2 if _is_hosted_demo() else 5.0


def _render_post_seconds(input_path: Path) -> float:
    return 2.8 if _is_hosted_demo() else 5.0


def _render_slate_seconds(input_path: Path) -> float:
    return 1.6 if _is_hosted_demo() else 2.4


def _render_freeze_seconds(input_path: Path) -> float:
    return 0.8 if _is_hosted_demo() else 1.8


def _is_hosted_demo() -> bool:
    return bool(os.getenv("RENDER"))
