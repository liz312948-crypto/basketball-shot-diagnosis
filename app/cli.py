from __future__ import annotations

import argparse
from pathlib import Path

from .clipper import render_vertical_highlights, write_events_json
from .detector import BasketballShotDetector, DetectionSettings


def main() -> None:
    parser = argparse.ArgumentParser(description="自动识别篮球投篮并生成竖屏短视频。")
    parser.add_argument("video", help="整场比赛视频路径")
    parser.add_argument("--out", default="outputs", help="输出目录")
    parser.add_argument("--pre", type=float, default=5.0, help="事件前保留秒数")
    parser.add_argument("--post", type=float, default=5.0, help="事件后保留秒数")
    parser.add_argument("--threshold", type=float, default=0.38, help="检测阈值，越高越保守")
    parser.add_argument("--sample-every", type=float, default=0.35, help="扫描间隔秒数")
    args = parser.parse_args()

    video_path = Path(args.video)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem

    detector = BasketballShotDetector(
        DetectionSettings(sample_every=args.sample_every, threshold=args.threshold)
    )
    try:
        events = detector.detect(video_path)
    finally:
        detector.close()

    events_path = output_dir / f"{stem}_events.json"
    output_video = output_dir / f"{stem}_vertical.mp4"
    write_events_json(events, events_path)
    render_vertical_highlights(video_path, events, output_video, args.pre, args.post)

    print(f"检测到 {len(events)} 个片段")
    print(f"事件文件：{events_path.resolve()}")
    print(f"竖屏视频：{output_video.resolve()}")


if __name__ == "__main__":
    main()
