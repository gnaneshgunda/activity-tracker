from types import SimpleNamespace

import app


def test_timeline_reference_falls_back_when_recording_start_is_none():
    row = SimpleNamespace(t_start=123.0, t_end=200.0)

    ref, ts_str = app._timeline_reference_and_label(row, None)

    assert ref == 123.0
    assert ts_str == "+0s from start"


def test_timeline_reference_uses_recording_start_when_available():
    row = SimpleNamespace(t_start=123.0, t_end=200.0)

    ref, ts_str = app._timeline_reference_and_label(row, 100.0)

    assert ref == 100.0
    assert ts_str == "+23s from start"
