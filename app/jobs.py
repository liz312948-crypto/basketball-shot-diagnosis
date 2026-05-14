from __future__ import annotations

import json
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
        return self.records.get(job_id)

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
            self._persist_record(record)
            precheck = analyze_video_precheck(record.input_path)

            record.status = JobStatus.BASE_ANALYSIS
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
                self._persist_record(record)
                enhanced = build_enhanced_analysis(precheck=precheck, report=report)

            record.status = JobStatus.RENDERING
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
            self._persist_record(record)
        except Exception as exc:
            record.status = JobStatus.FAILED
            record.error = str(exc)
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
                render_vertical_highlights(
                    record.input_path,
                    events,
                    output_video,
                    diagnosis_cards=_diagnosis_cards_data(report),
                    slate_seconds=2.4,
                )
                artifacts.highlight_video = output_video.name
            except Exception:
                artifacts.highlight_video = None
        return artifacts

    def _state_path(self, job_id: str) -> Path:
        return self.state_dir / f"{job_id}.json"

    def _persist_record(self, record: JobRecord) -> None:
        payload: dict[str, object] = {
            "id": record.id,
            "filename": record.filename,
            "input_path": str(record.input_path),
            "status": record.status.value,
            "error": record.error,
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
                record = JobRecord(
                    id=str(payload["id"]),
                    filename=str(payload["filename"]),
                    input_path=Path(str(payload["input_path"])),
                    status=JobStatus(str(payload["status"])),
                    error=(str(payload["error"]) if payload.get("error") is not None else None),
                    result=result,
                )
                self.records[record.id] = record
            except Exception:
                continue


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


def _diagnosis_cards_data(report: AnalysisReport) -> list[dict[str, object]]:
    cards: list[dict[str, object]] = []
    for item in report.analyses:
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
