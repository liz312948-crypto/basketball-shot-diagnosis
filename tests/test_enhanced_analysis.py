from app.analyzer import AnalysisReport, ShotAnalysis
from app.enhanced_analysis import build_enhanced_analysis
from app.models import ConfidenceLevel, PrecheckResult


def test_build_enhanced_analysis_returns_stage_breakdown_for_high_confidence() -> None:
    precheck = PrecheckResult(
        score=0.86,
        confidence=ConfidenceLevel.HIGH,
        run_enhanced_analysis=True,
        view_type="side",
        summary="high confidence",
        reasons=["stable side view"],
        recommendations=[],
    )
    report = AnalysisReport(
        overall="base summary",
        analyses=[
            ShotAnalysis(
                index=1,
                time=1.2,
                level="optimize",
                summary="flat release",
                metrics={
                    "ball_visible_frames": 6,
                    "track_quality": "good",
                    "arc_lift": 0.06,
                    "lateral_drift": 0.11,
                    "release_smoothness": 0.31,
                    "motion_stability": 0.71,
                },
                findings=["arc is low"],
                drills=["high arc form shooting"],
            )
        ],
    )

    enhanced = build_enhanced_analysis(precheck=precheck, report=report)

    assert enhanced is not None
    assert enhanced.summary
    assert len(enhanced.stages) == 5
    assert any(stage.name == "release" for stage in enhanced.stages)
    assert enhanced.template_comparison


def test_build_enhanced_analysis_skips_when_precheck_blocks_it() -> None:
    precheck = PrecheckResult(
        score=0.44,
        confidence=ConfidenceLevel.MEDIUM,
        run_enhanced_analysis=False,
        view_type="mixed",
        summary="base only",
        reasons=["view unstable"],
        recommendations=["retry"],
    )
    report = AnalysisReport(overall="base summary", analyses=[])

    enhanced = build_enhanced_analysis(precheck=precheck, report=report)

    assert enhanced is None
