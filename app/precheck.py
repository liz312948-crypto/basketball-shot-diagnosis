from __future__ import annotations

from pathlib import Path
from statistics import mean

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
    pose_visibility: float,
    ball_presence: float,
) -> PrecheckResult:
    score = 0.0
    score += 0.18 if width >= 960 else 0.07
    score += 0.12 if fps >= 24 else 0.04
    score += min(0.14, focus_score / 1100)
    score += 0.10 if 80 <= brightness_score <= 185 else 0.04
    score += min(0.08, motion_score * 0.10)
    score += 0.10 if subject_ratio >= 0.32 else 0.03
    score += 0.06 if occlusion_ratio <= 0.15 else 0.0
    score += 0.08 if view_hint in {"side", "45deg"} else 0.02
    score += 0.04 if duration >= 5 and frame_count >= max(90, int(fps * 3)) else 0.0
    score += min(0.14, pose_visibility * 0.16)
    score += min(0.10, ball_presence * 0.14)
    score = round(min(score, 1.0), 3)

    strong_pose = pose_visibility >= 0.38
    strong_ball = ball_presence >= 0.16
    enhanced_ready = score >= 0.72 and strong_pose and strong_ball and view_hint in {"side", "45deg"}
    medium_ready = score >= 0.45 and (pose_visibility >= 0.2 or ball_presence >= 0.1)

    reasons: list[str] = []
    recommendations: list[str] = []

    if strong_pose:
        reasons.append("主体关键点可见性较好，能较稳定地观察上肢出手动作。")
    elif pose_visibility >= 0.2:
        reasons.append("主体大致可见，但手腕、肘部或肩线信息不够稳定。")
        recommendations.append("建议重拍时尽量让头部、肩部、手肘和手腕都完整入镜。")
    else:
        reasons.append("主体关键点信号偏弱，动作诊断容易被机位和遮挡干扰。")
        recommendations.append("建议重拍并优先使用侧面或 45 度机位，让投篮人占画面更大比例。")

    if strong_ball:
        reasons.append("球体信号存在度较好，能支持基础球路分析。")
    elif ball_presence >= 0.1:
        reasons.append("能间歇性看到球，但球路连续性一般。")
        recommendations.append("建议重拍时让球和篮筐尽量同时出现在画面里，减少出手后丢球。")
    else:
        reasons.append("球信号过弱，球路和出手点判断可信度较低。")
        recommendations.append("建议重拍并提高光照、缩短拍摄距离，避免球体过小或与背景混色。")

    if enhanced_ready:
        return PrecheckResult(
            score=score,
            confidence=ConfidenceLevel.HIGH,
            run_enhanced_analysis=True,
            view_type=view_hint,
            summary="视频质量较好，适合做增强诊断。",
            reasons=reasons,
            recommendations=recommendations,
        )
    if medium_ready:
        if not recommendations:
            recommendations.append("尽量使用侧面或 45 度机位重拍，可明显提升动作诊断可信度。")
        return PrecheckResult(
            score=score,
            confidence=ConfidenceLevel.MEDIUM,
            run_enhanced_analysis=False,
            view_type=view_hint,
            summary="视频可分析，但更适合基础诊断。",
            reasons=reasons,
            recommendations=recommendations,
        )
    if not recommendations:
        recommendations.append("建议重拍：保持侧面机位、完整入镜，并让球体和篮筐更清楚。")
    return PrecheckResult(
        score=score,
        confidence=ConfidenceLevel.LOW,
        run_enhanced_analysis=False,
        view_type=view_hint,
        summary="视频质量不足，本次结果只能作为低可信参考。",
        reasons=reasons,
        recommendations=recommendations,
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

    pose_estimator = _PoseVisibilityEstimator()
    focus_scores: list[float] = []
    brightness_scores: list[float] = []
    motion_scores: list[float] = []
    subject_ratios: list[float] = []
    pose_scores: list[float] = []
    ball_scores: list[float] = []
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
            pose_scores.append(pose_estimator.score(frame))
            ball_scores.append(_orange_ball_presence(frame))
            motion, prev_gray = _motion_score(gray, prev_gray)
            if prev_gray is not None:
                motion_scores.append(motion)
        frame_index += 1

    cap.release()
    pose_estimator.close()

    focus_score = mean(focus_scores) if focus_scores else 0.0
    brightness_score = mean(brightness_scores) if brightness_scores else 0.0
    motion_score = mean(motion_scores) if motion_scores else 0.0
    subject_ratio = mean(subject_ratios) if subject_ratios else 0.0
    pose_visibility = mean(pose_scores) if pose_scores else 0.0
    ball_presence = mean(ball_scores) if ball_scores else 0.0
    occlusion_ratio = max(0.0, min(1.0, 1.0 - subject_ratio))
    view_hint = _infer_view_hint(width=width, height=height, motion_score=motion_score, pose_visibility=pose_visibility)

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
        pose_visibility=pose_visibility,
        ball_presence=ball_presence,
    )


class _PoseVisibilityEstimator:
    def __init__(self) -> None:
        self._pose = None
        self._mp_pose = None
        try:
            import mediapipe as mp  # type: ignore

            self._mp_pose = mp.solutions.pose
            self._pose = self._mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.45,
                min_tracking_confidence=0.45,
            )
        except Exception:
            self._pose = None
            self._mp_pose = None

    def score(self, frame: np.ndarray) -> float:
        if self._pose is None or self._mp_pose is None:
            return 0.0

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)
        if not result.pose_landmarks:
            return 0.0

        lm = result.pose_landmarks.landmark
        p = self._mp_pose.PoseLandmark
        indices = [
            p.NOSE,
            p.LEFT_SHOULDER,
            p.RIGHT_SHOULDER,
            p.LEFT_ELBOW,
            p.RIGHT_ELBOW,
            p.LEFT_WRIST,
            p.RIGHT_WRIST,
            p.LEFT_HIP,
            p.RIGHT_HIP,
        ]
        visibilities = [float(lm[index].visibility) for index in indices]
        return max(0.0, min(1.0, float(mean(visibilities))))

    def close(self) -> None:
        if self._pose is not None:
            self._pose.close()


def _estimate_subject_ratio(gray: np.ndarray) -> float:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    active = float(np.count_nonzero(mask))
    return active / max(1.0, float(mask.size))


def _orange_ball_presence(frame: np.ndarray) -> float:
    small = cv2.resize(frame, (0, 0), fx=0.35, fy=0.35)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([5, 65, 65]), np.array([28, 255, 255]))
    mask = cv2.medianBlur(mask, 5)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    image_area = small.shape[0] * small.shape[1]
    best = 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 8:
            continue
        perimeter = cv2.arcLength(contour, True)
        circularity = 4.0 * np.pi * area / (perimeter * perimeter + 1e-6)
        size_score = min(1.0, area / (image_area * 0.0025))
        best = max(best, float(circularity * 0.6 + size_score * 0.4))
    return min(1.0, best)


def _motion_score(gray: np.ndarray, prev_gray: np.ndarray | None) -> tuple[float, np.ndarray]:
    reduced = cv2.resize(gray, (320, 180))
    reduced = cv2.GaussianBlur(reduced, (7, 7), 0)
    if prev_gray is None:
        return 0.0, reduced
    diff = cv2.absdiff(reduced, prev_gray)
    return min(1.0, float(np.mean(diff) / 32.0)), reduced


def _infer_view_hint(*, width: int, height: int, motion_score: float, pose_visibility: float) -> str:
    if width >= height and pose_visibility >= 0.35 and motion_score >= 0.18:
        return "side"
    if pose_visibility >= 0.22 and motion_score >= 0.12:
        return "45deg"
    return "mixed"
