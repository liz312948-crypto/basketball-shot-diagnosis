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
