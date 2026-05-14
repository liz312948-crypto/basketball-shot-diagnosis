from __future__ import annotations

from statistics import mean
from pathlib import Path

import cv2
import numpy as np

from .models import ConfidenceLevel, PrecheckResult


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
    score += 0.04 if duration >= 5 and frame_count >= max(90, int(fps * 3)) else 0.0
    score = round(min(score, 1.0), 3)

    if score >= 0.72:
        return PrecheckResult(
            score=score,
            confidence=ConfidenceLevel.HIGH,
            run_enhanced_analysis=True,
            view_type=view_hint,
            summary="视频质量较好，适合做增强诊断。",
            reasons=["机位、清晰度和主体完整度较稳定。"],
            recommendations=[],
        )
    if score >= 0.45:
        return PrecheckResult(
            score=score,
            confidence=ConfidenceLevel.MEDIUM,
            run_enhanced_analysis=False,
            view_type=view_hint,
            summary="视频可分析，但更适合基础诊断。",
            reasons=["存在部分清晰度、遮挡或机位限制。"],
            recommendations=["尽量使用侧面或 45 度机位重拍。"],
        )
    return PrecheckResult(
        score=score,
        confidence=ConfidenceLevel.LOW,
        run_enhanced_analysis=False,
        view_type=view_hint,
        summary="视频质量不足，本次结果只能作为低可信参考。",
        reasons=["清晰度、帧率或遮挡影响明显。"],
        recommendations=["建议重拍：保持侧面机位、完整入镜、画面稳定。"],
    )


def analyze_video_precheck(video_path: str | Path) -> PrecheckResult:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frame_count / fps if fps > 0 else 0.0

    focus_scores: list[float] = []
    brightness_scores: list[float] = []
    motion_scores: list[float] = []
    subject_ratios: list[float] = []
    prev_gray: np.ndarray | None = None

    sample_limit = 18
    sample_stride = max(1, frame_count // sample_limit) if frame_count else 1
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % sample_stride == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            focus_scores.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
            brightness_scores.append(float(np.mean(gray)))
            subject_ratios.append(_estimate_subject_ratio(gray))
            motion, prev_gray = _motion_score(gray, prev_gray)
            if prev_gray is not None:
                motion_scores.append(motion)
        frame_index += 1

    cap.release()

    focus_score = mean(focus_scores) if focus_scores else 0.0
    brightness_score = mean(brightness_scores) if brightness_scores else 0.0
    motion_score = mean(motion_scores) if motion_scores else 0.0
    subject_ratio = mean(subject_ratios) if subject_ratios else 0.0
    occlusion_ratio = max(0.0, min(1.0, 1.0 - subject_ratio))
    view_hint = _infer_view_hint(width=width, height=height, motion_score=motion_score)

    return score_precheck(
        width=width,
        height=height,
        fps=fps,
        duration=duration,
        frame_count=frame_count,
        focus_score=focus_score,
        brightness_score=brightness_score,
        motion_score=motion_score,
        subject_ratio=subject_ratio,
        occlusion_ratio=occlusion_ratio,
        view_hint=view_hint,
    )


def _estimate_subject_ratio(gray: np.ndarray) -> float:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    active = float(np.count_nonzero(mask))
    return active / max(1.0, float(mask.size))


def _motion_score(gray: np.ndarray, prev_gray: np.ndarray | None) -> tuple[float, np.ndarray]:
    reduced = cv2.resize(gray, (320, 180))
    reduced = cv2.GaussianBlur(reduced, (7, 7), 0)
    if prev_gray is None:
        return 0.0, reduced
    diff = cv2.absdiff(reduced, prev_gray)
    return min(1.0, float(np.mean(diff) / 32.0)), reduced


def _infer_view_hint(*, width: int, height: int, motion_score: float) -> str:
    if width >= height and motion_score >= 0.25:
        return "side"
    if motion_score >= 0.15:
        return "45deg"
    return "mixed"

