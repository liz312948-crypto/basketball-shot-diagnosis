from app.analyzer import _build_metrics, _diagnose


def test_build_metrics_exposes_release_height_and_path_consistency() -> None:
    ball_points = [
        (0.00, 410.0, 620.0, 40.0),
        (0.12, 430.0, 540.0, 42.0),
        (0.24, 455.0, 470.0, 39.0),
        (0.36, 480.0, 430.0, 38.0),
        (0.48, 505.0, 410.0, 37.0),
    ]

    metrics = _build_metrics(
        ball_points=ball_points,
        motion_values=[0.18, 0.22, 0.24, 0.19],
        width=720,
        height=1280,
    )

    assert "release_height_ratio" in metrics
    assert "path_direction_consistency" in metrics
    assert float(metrics["release_height_ratio"]) > 0.45
    assert float(metrics["path_direction_consistency"]) > 0.7


def test_diagnose_flags_low_release_height() -> None:
    findings, drills = _diagnose(
        {
            "ball_visible_frames": 5,
            "track_quality": "good",
            "arc_lift": 0.11,
            "lateral_drift": 0.07,
            "release_smoothness": 0.62,
            "motion_stability": 0.78,
            "release_height_ratio": 0.18,
            "path_direction_consistency": 0.74,
        }
    )

    assert any("出手点" in finding for finding in findings)
    assert any("出手高度" in drill or "抬高" in drill for drill in drills)
