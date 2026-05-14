from __future__ import annotations

from dataclasses import dataclass

from .analyzer import AnalysisReport
from .models import PrecheckResult


@dataclass(frozen=True)
class StageAssessment:
    name: str
    status: str
    summary: str
    focus_points: list[str]


@dataclass(frozen=True)
class EnhancedAnalysis:
    summary: str
    stages: list[StageAssessment]
    template_comparison: list[str]


def build_enhanced_analysis(
    *,
    precheck: PrecheckResult,
    report: AnalysisReport,
) -> EnhancedAnalysis | None:
    if not precheck.run_enhanced_analysis or not report.analyses:
        return None

    metrics = _aggregate_metrics(report)
    stages = [
        _setup_stage(metrics),
        _dip_stage(metrics),
        _lift_stage(metrics),
        _release_stage(metrics),
        _follow_through_stage(metrics),
    ]
    return EnhancedAnalysis(
        summary=(
            "增强分析认为这段视频足够清晰，可以继续看分阶段反馈。"
            "后续训练优先围绕得分最低的阶段进行修正。"
        ),
        stages=stages,
        template_comparison=_template_comparison(metrics),
    )


def _aggregate_metrics(report: AnalysisReport) -> dict[str, float]:
    analyses = report.analyses
    count = max(1, len(analyses))
    return {
        "arc_lift": sum(float(item.metrics.get("arc_lift", 0.0)) for item in analyses) / count,
        "lateral_drift": sum(float(item.metrics.get("lateral_drift", 0.0)) for item in analyses) / count,
        "release_smoothness": sum(float(item.metrics.get("release_smoothness", 0.0)) for item in analyses) / count,
        "motion_stability": sum(float(item.metrics.get("motion_stability", 0.0)) for item in analyses) / count,
        "visible_frames": sum(float(item.metrics.get("ball_visible_frames", 0.0)) for item in analyses) / count,
    }


def _setup_stage(metrics: dict[str, float]) -> StageAssessment:
    stable = metrics["motion_stability"] >= 0.65
    return StageAssessment(
        name="setup",
        status="stable" if stable else "needs-work",
        summary=(
            "投篮前的身体控制比较稳定。"
            if stable
            else "投篮准备阶段身体波动偏大，基础姿态不够稳定。"
        ),
        focus_points=["接球前保持躯干安静。", "出手落地后维持平衡。"],
    )


def _dip_stage(metrics: dict[str, float]) -> StageAssessment:
    smooth = metrics["release_smoothness"] >= 0.42
    return StageAssessment(
        name="dip",
        status="stable" if smooth else "needs-work",
        summary=(
            "举球到下蹲的节奏连接比较顺。"
            if smooth
            else "举球到下蹲之间有停顿，动作连贯性不足。"
        ),
        focus_points=["把下蹲和上升发力连起来。", "避免上举前出现停顿。"],
    )


def _lift_stage(metrics: dict[str, float]) -> StageAssessment:
    lift = metrics["arc_lift"] >= 0.085
    return StageAssessment(
        name="lift",
        status="stable" if lift else "needs-work",
        summary=(
            "上升发力带出的高度基本够用。"
            if lift
            else "上升高度偏浅，通常会让出手弧线变平。"
        ),
        focus_points=["先向上发力，再伸臂出手。", "增加更明确的向上抬升。"],
    )


def _release_stage(metrics: dict[str, float]) -> StageAssessment:
    aligned = metrics["lateral_drift"] <= 0.14 and metrics["release_smoothness"] >= 0.4
    return StageAssessment(
        name="release",
        status="stable" if aligned else "needs-work",
        summary=(
            "出手路线比较直接，重复性还可以。"
            if aligned
            else "出手路线过早变平或左右漂移，影响重复性。"
        ),
        focus_points=["把出手弧线抬高一点。", "尽量让出手线保持居中。"],
    )


def _follow_through_stage(metrics: dict[str, float]) -> StageAssessment:
    visible = metrics["visible_frames"] >= 4
    return StageAssessment(
        name="follow-through",
        status="stable" if visible else "limited-signal",
        summary=(
            "随球动作画面足够完整，可以继续给反馈。"
            if visible
            else "随球动作拍得不够完整，这一阶段的判断可信度较低。"
        ),
        focus_points=["出手后多停一拍。", "让随球动作尽量留在画面里。"],
    )


def _template_comparison(metrics: dict[str, float]) -> list[str]:
    notes: list[str] = []
    if metrics["arc_lift"] < 0.085:
        notes.append("和理想侧面模板相比，这次出手弧线偏平。")
    else:
        notes.append("和理想侧面模板相比，这次出手弧线仍在可接受范围内。")
    if metrics["lateral_drift"] > 0.14:
        notes.append("这次球路的水平漂移比模板基线更大。")
    else:
        notes.append("这次球路基本贴近模板的中线。")
    return notes
