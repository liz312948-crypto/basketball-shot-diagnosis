from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .detector import ShotEvent


def video_duration(video_path: str | Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return frame_count / fps if frame_count else 0.0


def write_events_json(events: list[ShotEvent], output_path: str | Path) -> None:
    Path(output_path).write_text(
        json.dumps([asdict(event) for event in events], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def select_freeze_time(event_time: float, start: float, end: float) -> float:
    return max(start, min(event_time, end))


def render_vertical_highlights(
    input_video: str | Path,
    events: list[ShotEvent],
    output_video: str | Path,
    pre_seconds: float = 5.0,
    post_seconds: float = 5.0,
    target_width: int = 1080,
    target_height: int = 1920,
    diagnosis_cards: list[dict[str, object]] | None = None,
    slate_seconds: float = 3.0,
    freeze_seconds: float = 1.8,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Path:
    if not events:
        raise ValueError("没有检测到可剪辑的投篮片段。")

    input_path = Path(input_video)
    output_path = Path(output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = video_duration(input_path)

    with tempfile.TemporaryDirectory(prefix="basketball_clips_") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        clip_paths: list[Path] = []

        if diagnosis_cards:
            timeline = [
                (event, max(0.0, event.time - pre_seconds), min(duration, event.time + post_seconds))
                for event in sorted(events, key=lambda item: item.time)
            ]
        else:
            timeline = [
                (None, start, end)
                for start, end in _event_segments(events, duration, pre_seconds, post_seconds)
            ]

        for index, (event, start, end) in enumerate(timeline, start=1):
            if progress_callback is not None:
                progress_callback(index, len(timeline))
            clip_duration = max(0.2, end - start)
            action_path = tmp_dir / f"clip_{index:03d}.mp4"
            _render_clip(input_path, action_path, start, clip_duration, target_width, target_height)

            if diagnosis_cards and event is not None:
                card = diagnosis_cards[min(index - 1, len(diagnosis_cards) - 1)]
                freeze_path = tmp_dir / f"freeze_{index:03d}.mp4"
                freeze_time = select_freeze_time(event.time, start, end)
                _render_issue_freeze(
                    input_path,
                    freeze_path,
                    freeze_time,
                    card,
                    target_width,
                    target_height,
                    freeze_seconds,
                )
                slate_path = tmp_dir / f"diagnosis_{index:03d}.mp4"
                _render_diagnosis_slate(
                    slate_path,
                    card,
                    target_width,
                    target_height,
                    slate_seconds,
                )
                clip_paths.extend([action_path, freeze_path, slate_path])
            else:
                clip_paths.append(action_path)

        _concat_mp4(clip_paths, output_path)

    return output_path


def render_individual_shot_clips(
    input_video: str | Path,
    events: list[ShotEvent],
    output_dir: str | Path,
    output_stem: str,
    pre_seconds: float = 5.0,
    post_seconds: float = 5.0,
    target_width: int = 1080,
    target_height: int = 1920,
    diagnosis_cards: list[dict[str, object]] | None = None,
    slate_seconds: float = 3.0,
    freeze_seconds: float = 1.8,
) -> list[Path]:
    input_path = Path(input_video)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    duration = video_duration(input_path)

    clip_paths: list[Path] = []
    for index, event in enumerate(events, start=1):
        safe_time = f"{event.time:06.2f}".replace(".", "_")
        clip_path = output_path / f"{output_stem}_clip_{index:02d}_at_{safe_time}s.mp4"
        start = max(0.0, event.time - pre_seconds)
        end = min(duration, event.time + post_seconds) if duration else event.time + post_seconds
        if diagnosis_cards:
            with tempfile.TemporaryDirectory(prefix="basketball_single_") as tmp_dir_name:
                tmp_dir = Path(tmp_dir_name)
                action_path = tmp_dir / "action.mp4"
                freeze_path = tmp_dir / "freeze.mp4"
                slate_path = tmp_dir / "diagnosis.mp4"
                card = diagnosis_cards[min(index - 1, len(diagnosis_cards) - 1)]
                _render_clip(input_path, action_path, start, max(0.2, end - start), target_width, target_height)
                _render_issue_freeze(
                    input_path,
                    freeze_path,
                    select_freeze_time(event.time, start, end),
                    card,
                    target_width,
                    target_height,
                    freeze_seconds,
                )
                _render_diagnosis_slate(
                    slate_path,
                    card,
                    target_width,
                    target_height,
                    slate_seconds,
                )
                _concat_mp4([action_path, freeze_path, slate_path], clip_path)
        else:
            _render_clip(input_path, clip_path, start, max(0.2, end - start), target_width, target_height)
        clip_paths.append(clip_path)
    return clip_paths


def _event_segments(
    events: list[ShotEvent],
    duration: float,
    pre_seconds: float,
    post_seconds: float,
) -> list[tuple[float, float]]:
    segments: list[tuple[float, float]] = []
    for event in sorted(events, key=lambda item: item.time):
        start = max(0.0, event.time - pre_seconds)
        end = min(duration, event.time + post_seconds) if duration else event.time + post_seconds
        if not segments or start > segments[-1][1]:
            segments.append((start, end))
            continue
        previous_start, previous_end = segments[-1]
        segments[-1] = (previous_start, max(previous_end, end))
    return segments


def _render_clip(
    input_path: Path,
    clip_path: Path,
    start: float,
    clip_duration: float,
    target_width: int,
    target_height: int,
) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    vf = (
        f"scale={target_width}:{target_height}:force_original_aspect_ratio=increase,"
        f"crop={target_width}:{target_height},setsar=1"
    )
    cmd = [
        ffmpeg,
        "-y",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{clip_duration:.3f}",
        "-i",
        str(input_path),
        "-vf",
        vf,
        "-r",
        "30",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-an",
        "-movflags",
        "+faststart",
        str(clip_path),
    ]
    _run(cmd)


def _render_issue_freeze(
    input_path: Path,
    freeze_path: Path,
    freeze_time: float,
    card: dict[str, object],
    target_width: int,
    target_height: int,
    duration: float,
) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    image_path = freeze_path.with_suffix(".png")
    frame = _extract_frame(input_path, freeze_time)
    _write_issue_freeze_image(image_path, frame, card, target_width, target_height)
    cmd = [
        ffmpeg,
        "-y",
        "-loop",
        "1",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(image_path),
        "-vf",
        "format=yuv420p",
        "-r",
        "30",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-an",
        "-movflags",
        "+faststart",
        str(freeze_path),
    ]
    _run(cmd)


def _render_diagnosis_slate(
    slate_path: Path,
    card: dict[str, object],
    target_width: int,
    target_height: int,
    duration: float,
) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    image_path = slate_path.with_suffix(".png")
    _write_diagnosis_image(image_path, card, target_width, target_height)
    cmd = [
        ffmpeg,
        "-y",
        "-loop",
        "1",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(image_path),
        "-vf",
        "format=yuv420p",
        "-r",
        "30",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-an",
        "-movflags",
        "+faststart",
        str(slate_path),
    ]
    _run(cmd)


def _write_diagnosis_image(
    image_path: Path,
    card: dict[str, object],
    width: int,
    height: int,
) -> None:
    image = Image.new("RGB", (width, height), "#f5f7f8")
    draw = ImageDraw.Draw(image)
    font_regular = _font(42)
    font_bold = _font(58)
    font_small = _font(34)

    draw.rounded_rectangle((70, 95, width - 70, height - 95), radius=28, fill="#ffffff", outline="#d9e1e2", width=3)
    draw.text((110, 140), str(card.get("title", "投篮诊断")), fill="#0d6b73", font=font_bold)
    draw.text((110, 220), str(card.get("level", "")), fill="#c14f26", font=font_regular)

    y = 310
    sections = [
        ("总结", [str(card.get("summary", ""))]),
        ("发现", [str(item) for item in card.get("findings", [])][:3]),
        ("建议", [str(item) for item in card.get("drills", [])][:2]),
    ]
    for heading, lines in sections:
        draw.text((110, y), heading, fill="#162021", font=font_regular)
        y += 58
        for line in lines:
            wrapped = _wrap_text(draw, line, font_small, width - 220)
            for wrapped_line in wrapped[:3]:
                draw.text((130, y), wrapped_line, fill="#355153", font=font_small)
                y += 46
            y += 10
        y += 18

    draw.text((110, height - 170), "上一段视频已经在问题时刻定格，先看错在哪一帧，再看训练建议。", fill="#62787a", font=font_small)
    image.save(image_path)


def _write_issue_freeze_image(
    image_path: Path,
    frame: np.ndarray,
    card: dict[str, object],
    width: int,
    height: int,
) -> None:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    image = _resize_and_crop(image, width, height)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font_title = _font(44)
    font_body = _font(30)

    draw.rounded_rectangle((48, 54, width - 48, 210), radius=24, fill=(11, 31, 34, 208))
    draw.text((82, 88), "问题时刻定格", fill="#ffffff", font=font_title)

    findings = [str(item) for item in card.get("findings", [])]
    primary_issue = findings[0] if findings else "这一帧需要重点看出手路径、身体平衡和球路方向。"
    for idx, line in enumerate(_wrap_text(draw, primary_issue, font_body, width - 164)[:3]):
        draw.text((82, 146 + idx * 38), line, fill="#d9ecee", font=font_body)

    draw.rounded_rectangle((48, height - 240, width - 48, height - 54), radius=24, fill=(255, 255, 255, 216))
    draw.text((82, height - 208), "建议盯住这里看：", fill="#0d6b73", font=font_title)
    focus_points = [str(item) for item in card.get("drills", [])[:2]] or ["对照出手方向、弧线高度和身体稳定性。"]
    y = height - 152
    for point in focus_points:
        for line in _wrap_text(draw, f"• {point}", font_body, width - 164)[:2]:
            draw.text((82, y), line, fill="#23484b", font=font_body)
            y += 34
        y += 8

    composed = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    composed.save(image_path)


def _extract_frame(input_path: Path, timestamp: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    target_frame = max(0, int(timestamp * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise ValueError(f"无法提取视频帧: {input_path} @ {timestamp:.2f}s")
    return frame


def _resize_and_crop(image: Image.Image, width: int, height: int) -> Image.Image:
    src_w, src_h = image.size
    scale = max(width / src_w, height / src_h)
    resized = image.resize((int(src_w * scale), int(src_h * scale)))
    left = max(0, (resized.width - width) // 2)
    top = max(0, (resized.height - height) // 2)
    return resized.crop((left, top, left + width, top + height))


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for char in text:
        candidate = current + char
        if draw.textlength(candidate, font=font) <= max_width:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = char
    if current:
        lines.append(current)
    return lines or [text]


def _concat_mp4(clip_paths: list[Path], output_path: Path) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.TemporaryDirectory(prefix="basketball_concat_") as tmp_dir_name:
        concat_file = Path(tmp_dir_name) / "concat.txt"
        concat_file.write_text(
            "".join(f"file '{clip.as_posix()}'\n" for clip in clip_paths),
            encoding="utf-8",
        )
        concat_cmd = [
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",
            str(output_path),
        ]
        _run(concat_cmd)


def _run(cmd: list[str]) -> None:
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"FFmpeg 执行失败: {message}")
