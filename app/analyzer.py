from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .detector import ShotEvent


@dataclass(frozen=True)
class ShotAnalysis:
    index: int
    time: float
    level: str
    summary: str
    metrics: dict[str, float | int | str]
    findings: list[str]
    drills: list[str]


@dataclass(frozen=True)
class AnalysisReport:
    overall: str
    analyses: list[ShotAnalysis]


def analyze_shots(video_path: str | Path, events: list[ShotEvent]) -> AnalysisReport:
    if not events:
        return AnalysisReport(
            overall="还没有检测到可分析的投篮片段，本次无法生成动作诊断。",
            analyses=[],
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1)
    duration = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps

    analyses = [
        _analyze_single_shot(cap, event, index, fps, width, height, duration)
        for index, event in enumerate(events, start=1)
    ]
    cap.release()

    weak_count = sum(1 for item in analyses if item.level == "需要重点改进")
    optimize_count = sum(1 for item in analyses if item.level == "可以优化")

    if weak_count:
        overall = (
            f"本次共分析 {len(analyses)} 次投篮，其中 {weak_count} 次暴露出明显动作风险。"
            "优先关注出手高度、球路弧线和出手方向的一致性。"
        )
    elif optimize_count:
        overall = (
            f"本次共分析 {len(analyses)} 次投篮，整体可用，但仍有 {optimize_count} 次存在可优化细节。"
            "建议继续稳定出手节奏，并减少左右漂移。"
        )
    else:
        overall = (
            f"本次共分析 {len(analyses)} 次投篮，整体节奏比较稳定。"
            "建议继续保持完整随球动作，并关注每次出手的一致性。"
        )
    return AnalysisReport(overall=overall, analyses=analyses)


def write_analysis_json(report: AnalysisReport, output_path: str | Path) -> None:
    Path(output_path).write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _analyze_single_shot(
    cap: cv2.VideoCapture,
    event: ShotEvent,
    index: int,
    fps: float,
    width: int,
    height: int,
    duration: float,
) -> ShotAnalysis:
    start = max(0.0, event.time - 2.2)
    end = min(duration, event.time + 1.8) if duration else event.time + 1.8
    step = max(1, int(fps * 0.12))
    start_frame = int(start * fps)
    end_frame = int(end * fps)

    ball_points: list[tuple[float, float, float, float]] = []
    motion_values: list[float] = []
    prev_gray: np.ndarray | None = None

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_index = start_frame
    while frame_index <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break

        if (frame_index - start_frame) % step == 0:
            time_s = frame_index / fps
            center = _find_ball(frame)
            if center is not None:
                x, y, area = center
                ball_points.append((time_s, x, y, area))
            motion, prev_gray = _motion_value(frame, prev_gray)
            if prev_gray is not None:
                motion_values.append(motion)

        frame_index += 1

    metrics = _build_metrics(ball_points, motion_values, width, height)
    findings, drills = _diagnose(metrics)
    level = _level_from_findings(findings)
    summary = _summary_from_level(level)

    return ShotAnalysis(
        index=index,
        time=round(event.time, 2),
        level=level,
        summary=summary,
        metrics=metrics,
        findings=findings,
        drills=drills,
    )


def _find_ball(frame: np.ndarray) -> tuple[float, float, float] | None:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([5, 55, 45]), np.array([30, 255, 255]))
    mask = cv2.medianBlur(mask, 5)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, float, float, float]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 18:
            continue
        perimeter = cv2.arcLength(contour, True)
        circularity = 4.0 * np.pi * area / (perimeter * perimeter + 1e-6)
        if circularity < 0.18:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        x = moments["m10"] / moments["m00"]
        y = moments["m01"] / moments["m00"]
        score = area * max(0.2, circularity)
        candidates.append((score, x, y, area))
    if not candidates:
        return None
    _, x, y, area = max(candidates, key=lambda item: item[0])
    return x, y, area


def _motion_value(frame: np.ndarray, prev_gray: np.ndarray | None) -> tuple[float, np.ndarray]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (240, 426))
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    if prev_gray is None:
        return 0.0, gray
    return float(np.mean(cv2.absdiff(gray, prev_gray)) / 32.0), gray


def _build_metrics(
    ball_points: list[tuple[float, float, float, float]],
    motion_values: list[float],
    width: int,
    height: int,
) -> dict[str, float | int | str]:
    metrics: dict[str, float | int | str] = {
        "ball_visible_frames": len(ball_points),
        "track_quality": "low",
        "arc_lift": 0.0,
        "lateral_drift": 0.0,
        "release_smoothness": 0.0,
        "motion_stability": 0.0,
        "release_height_ratio": 0.0,
        "path_direction_consistency": 0.0,
    }
    if len(ball_points) >= 2:
        xs = np.array([point[1] for point in ball_points], dtype=float)
        ys = np.array([point[2] for point in ball_points], dtype=float)
        first_y = float(ys[0])
        highest_y = float(np.min(ys))
        arc_lift = max(0.0, (first_y - highest_y) / max(1, height))
        lateral_drift = float(np.std(xs) / max(1, width))
        release_height_ratio = max(0.0, min(1.0, 1.0 - first_y / max(1, height)))

        y_steps = np.diff(ys)
        x_steps = np.diff(xs)
        upward_ratio = float(np.mean(y_steps < 0)) if len(y_steps) else 0.0
        drift_variation = float(np.std(x_steps) / max(1, width)) if len(x_steps) else 0.0
        path_direction_consistency = max(0.0, min(1.0, upward_ratio * 0.85 + (1.0 - drift_variation * 8.0) * 0.15))
        release_smoothness = max(
            0.0,
            min(1.0, upward_ratio * 0.75 + path_direction_consistency * 0.2 - lateral_drift * 0.75),
        )

        metrics.update(
            {
                "track_quality": "good" if len(ball_points) >= 5 else "medium",
                "arc_lift": round(arc_lift, 3),
                "lateral_drift": round(lateral_drift, 3),
                "release_smoothness": round(release_smoothness, 3),
                "release_height_ratio": round(release_height_ratio, 3),
                "path_direction_consistency": round(path_direction_consistency, 3),
            }
        )

    if motion_values:
        motion_std = float(np.std(motion_values))
        stability = max(0.0, min(1.0, 1.0 - motion_std * 3.0))
        metrics["motion_stability"] = round(stability, 3)
    return metrics


def _diagnose(metrics: dict[str, float | int | str]) -> tuple[list[str], list[str]]:
    findings: list[str] = []
    drills: list[str] = []
    visible = int(metrics["ball_visible_frames"])
    arc_lift = float(metrics["arc_lift"])
    lateral_drift = float(metrics["lateral_drift"])
    smoothness = float(metrics["release_smoothness"])
    stability = float(metrics["motion_stability"])
    release_height = float(metrics.get("release_height_ratio", 0.0))
    path_consistency = float(metrics.get("path_direction_consistency", 0.0))

    if visible < 3:
        findings.append("篮球在画面中可追踪的帧数太少，出手与球路信息不完整。")
        drills.append("重拍时保持球、手肘和篮筐尽量同框，再做 10 次定点投篮。")
    if release_height < 0.24 and visible >= 3:
        findings.append("出手点偏低，球离手时的高度不足，容易把弧线压平。")
        drills.append("做单手近筐托举投篮，刻意把出手高度抬高到眉线以上。")
    if arc_lift < 0.08:
        findings.append("球路上升幅度偏低，出手弧线不够，容易形成平推。")
        drills.append("做近距离高弧线投篮练习，重点体会先向上再向前的出手路径。")
    if lateral_drift > 0.18:
        findings.append("球路左右漂移明显，出手方向控制不够稳定。")
        drills.append("做单手定点投篮，辅助手只负责扶球，出手后食指保持指向篮心。")
    if smoothness < 0.35 and visible >= 3:
        findings.append("出手轨迹不够连贯，动作节奏存在停顿或发力断点。")
        drills.append("做连续举球到出手练习，保持下肢发力、手肘上举、拨球一气呵成。")
    if path_consistency < 0.58 and visible >= 3:
        findings.append("球路前几帧方向不够一致，说明离手瞬间的发力线还不稳定。")
        drills.append("拍慢动作时重点检查手腕跟随方向，避免离手瞬间横向甩臂。")
    if stability < 0.58:
        findings.append("投篮前后身体和画面波动较大，重心控制不稳。")
        drills.append("练习原地 1-2 步接球投篮，落地后保持身体朝向篮筐 1 秒。")

    if not findings:
        findings.append("这次投篮的球路和动作节奏相对稳定，没有明显异常。")
        drills.append("保持同一出手节奏连续投 20 球，记录长短和左右偏差。")
    return findings, drills


def _level_from_findings(findings: list[str]) -> str:
    weak_terms = ("偏低", "漂移", "不够", "波动", "太少", "断点", "不稳", "不足")
    issue_count = sum(any(term in finding for term in weak_terms) for finding in findings)
    if issue_count >= 3:
        return "需要重点改进"
    if issue_count >= 1:
        return "可以优化"
    return "表现稳定"


def _summary_from_level(level: str) -> str:
    if level == "表现稳定":
        return "这次出手节奏比较顺，球路没有明显异常。"
    if level == "可以优化":
        return "这次出手能识别到有效球路，但仍存在一到两个可调整点。"
    return "这次投篮暴露出多个动作风险点，建议优先做基础稳定性训练。"
