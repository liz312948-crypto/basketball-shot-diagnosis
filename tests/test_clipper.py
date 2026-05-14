from app import clipper
from app.clipper import select_freeze_time


def test_select_freeze_time_stays_inside_clip_bounds() -> None:
    assert select_freeze_time(event_time=4.5, start=2.0, end=8.0) == 4.5
    assert select_freeze_time(event_time=1.0, start=2.0, end=8.0) == 2.0
    assert select_freeze_time(event_time=9.0, start=2.0, end=8.0) == 8.0


def test_overlay_copy_uses_readable_chinese() -> None:
    copy = clipper._overlay_copy()

    assert copy["freeze_title"] == "问题时刻定格"
    assert copy["focus_title"] == "建议盯住这里看："
    assert "上一段视频已经在问题时刻定格" in copy["slate_footer"]
    assert "出手路径" in copy["fallback_issue"]


def test_font_candidates_cover_linux_chinese_fonts() -> None:
    candidates = [str(path).replace("\\", "/") for path in clipper._font_candidates()]

    assert "C:/Windows/Fonts/msyh.ttc" in candidates
    assert "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc" in candidates
    assert "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc" in candidates
