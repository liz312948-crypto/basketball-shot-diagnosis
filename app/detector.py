from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class ShotEvent:
    time: float
    score: float
    reason: str


@dataclass(frozen=True)
class DetectionSettings:
    sample_every: float = 0.35
    threshold: float = 0.38
    merge_gap: float = 4.0
    max_events: int = 60


class BasketballShotDetector:
    def __init__(self, settings: DetectionSettings | None = None) -> None:
        self.settings = settings or DetectionSettings()
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

    def detect(self, video_path: str | Path) -> list[ShotEvent]:
        path = str(video_path)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频：{path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        duration = frame_count / fps if frame_count else 0.0
        step_frames = max(1, int(fps * self.settings.sample_every))

        raw_events: list[ShotEvent] = []
        prev_gray: np.ndarray | None = None
        prev_pose_score = 0.0
        frame_index = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index % step_frames != 0:
                frame_index += 1
                continue

            time_s = frame_index / fps
            pose_score, pose_reason = self._pose_shot_score(frame)
            color_score = self._orange_ball_score(frame)
            motion_score, prev_gray = self._motion_score(frame, prev_gray)
            lift_bonus = max(0.0, pose_score - prev_pose_score) * 0.35

            if self._pose is not None:
                score = min(1.0, pose_score * 0.76 + motion_score * 0.14 + color_score * 0.10 + lift_bonus)
                reason = pose_reason
            else:
                score = min(1.0, motion_score * 0.48 + color_score * 0.52)
                reason = "motion+orange-ball heuristic"

            prev_pose_score = pose_score
            effective_threshold = self.settings.threshold
            if self._pose is None:
                effective_threshold = min(effective_threshold, 0.38)

            if score >= effective_threshold:
                raw_events.append(ShotEvent(time=round(time_s, 2), score=round(score, 3), reason=reason))

            frame_index += 1

        cap.release()
        return self._merge_events(raw_events, duration)

    def close(self) -> None:
        if self._pose is not None:
            self._pose.close()

    def _pose_shot_score(self, frame: np.ndarray) -> tuple[float, str]:
        if self._pose is None or self._mp_pose is None:
            return 0.0, "pose unavailable"

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)
        if not result.pose_landmarks:
            return 0.0, "no person detected"

        lm = result.pose_landmarks.landmark
        p = self._mp_pose.PoseLandmark
        nose = lm[p.NOSE]
        left_wrist = lm[p.LEFT_WRIST]
        right_wrist = lm[p.RIGHT_WRIST]
        left_elbow = lm[p.LEFT_ELBOW]
        right_elbow = lm[p.RIGHT_ELBOW]
        left_shoulder = lm[p.LEFT_SHOULDER]
        right_shoulder = lm[p.RIGHT_SHOULDER]

        visible = np.mean(
            [
                nose.visibility,
                left_wrist.visibility,
                right_wrist.visibility,
                left_elbow.visibility,
                right_elbow.visibility,
                left_shoulder.visibility,
                right_shoulder.visibility,
            ]
        )
        if visible < 0.42:
            return 0.0, "low pose confidence"

        highest_wrist_y = min(left_wrist.y, right_wrist.y)
        head_clearance = max(0.0, nose.y - highest_wrist_y)
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2.0
        wrists_above_shoulders = max(0.0, shoulder_y - highest_wrist_y)
        elbow_extension = self._extension_score(left_shoulder, left_elbow, left_wrist)
        elbow_extension = max(elbow_extension, self._extension_score(right_shoulder, right_elbow, right_wrist))

        score = head_clearance * 2.4 + wrists_above_shoulders * 1.6 + elbow_extension * 0.35
        return min(1.0, score), "wrists above head with extended shooting arm"

    @staticmethod
    def _extension_score(shoulder, elbow, wrist) -> float:
        upper = np.array([shoulder.x - elbow.x, shoulder.y - elbow.y])
        lower = np.array([wrist.x - elbow.x, wrist.y - elbow.y])
        denom = np.linalg.norm(upper) * np.linalg.norm(lower)
        if denom <= 1e-6:
            return 0.0
        cos_angle = float(np.dot(upper, lower) / denom)
        return max(0.0, min(1.0, (cos_angle + 1.0) / 2.0))

    @staticmethod
    def _orange_ball_score(frame: np.ndarray) -> float:
        small = cv2.resize(frame, (0, 0), fx=0.35, fy=0.35)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([5, 70, 70]), np.array([28, 255, 255]))
        mask = cv2.medianBlur(mask, 5)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0

        best = 0.0
        image_area = small.shape[0] * small.shape[1]
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 8:
                continue
            perimeter = cv2.arcLength(contour, True)
            circularity = 4.0 * np.pi * area / (perimeter * perimeter + 1e-6)
            size_score = min(1.0, area / (image_area * 0.0025))
            best = max(best, float(circularity * 0.65 + size_score * 0.35))
        return min(1.0, best)

    @staticmethod
    def _motion_score(frame: np.ndarray, prev_gray: np.ndarray | None) -> tuple[float, np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 180))
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        if prev_gray is None:
            return 0.0, gray
        diff = cv2.absdiff(gray, prev_gray)
        motion = float(np.mean(diff) / 32.0)
        return min(1.0, motion), gray

    def _merge_events(self, events: Iterable[ShotEvent], duration: float) -> list[ShotEvent]:
        merged: list[ShotEvent] = []
        for event in sorted(events, key=lambda item: item.time):
            if event.time < 1.0 or (duration and event.time > duration - 1.0):
                continue
            if not merged or event.time - merged[-1].time > self.settings.merge_gap:
                merged.append(event)
                continue
            if event.score > merged[-1].score:
                merged[-1] = event
        return merged[: self.settings.max_events]
