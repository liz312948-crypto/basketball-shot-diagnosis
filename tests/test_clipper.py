from app.clipper import select_freeze_time


def test_select_freeze_time_stays_inside_clip_bounds() -> None:
    assert select_freeze_time(event_time=4.5, start=2.0, end=8.0) == 4.5
    assert select_freeze_time(event_time=1.0, start=2.0, end=8.0) == 2.0
    assert select_freeze_time(event_time=9.0, start=2.0, end=8.0) == 8.0
