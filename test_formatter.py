"""Tests for formatter.py (B10) -- coverage-aware output formatter.

Spec reference: making.md Stage 12 (B10).

Test groups
-----------
- ``format_answer`` pass-through for non-grounding routes (Task 1/2/B_ENERGY/
  PERSONALIZATION/UNKNOWN) -- no coverage check, no store required.
- ``format_answer`` Task 3/4/B_ANOMALY coverage gate:
  - Path (a): coverage > 0 + raw_ptr valid -> render as-is.
  - Path (b): coverage = 0, widening succeeds -> widened interval + note.
  - Path (c): coverage = 0, no near signal -> N/A fallback.
- ``check_coverage`` unit tests (isolated gate logic).
- ``render_text`` output shape checks.
- ``format_answer`` with ``store=None`` logs a warning and passes through.
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
import unittest
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np

from formatter import (
    CoverageCheckResult,
    FormattedAnswer,
    WIDEN_PROBE_STEP_S,
    WIDEN_SEARCH_RADIUS_S,
    EvidenceInterval,
    check_coverage,
    format_answer,
    render_text,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

#: Baseline Unix timestamp used in all tests (arbitrary but deterministic).
T0 = 1_700_000_000.0


def _make_evidence(
    *,
    task: str = "task3",
    segment_id: str = "seg-001",
    t_start: float = T0,
    t_end: float = T0 + 20.0,
    label: str = "walking",
    answer: str = "She started walking at 10:00 UTC.",
    confidence: float = 0.85,
    explanation: str = "Cadence 95 bpm, jerk energy 0.12.",
) -> dict:
    return {
        "task": task,
        "segment_id": segment_id,
        "t_start": t_start,
        "t_end": t_end,
        "label": label,
        "answer": answer,
        "confidence": confidence,
        "explanation": explanation,
    }


def _make_store(
    *,
    coverage_fraction: float = 0.9,
    raw_arr: Optional[np.ndarray] = None,
    timeline_row=None,
) -> MagicMock:
    """Build a mock ActivityStore with configurable get_coverage behaviour."""
    store = MagicMock()
    store.get_coverage.return_value = coverage_fraction
    store.get_segment_raw.return_value = (
        raw_arr if raw_arr is not None else np.zeros((25, 3))
    )
    store.get_timeline_row.return_value = timeline_row
    return store


# ---------------------------------------------------------------------------
# Minimal TimelineRow-like object for mock stores
# ---------------------------------------------------------------------------

@dataclass
class _FakeTimelineRow:
    t_start: float
    t_end: float
    raw_ptr: Optional[str] = None
    fs: Optional[float] = None
    coverage_s: float = 0.0


# ===========================================================================
# 1. Pass-through for non-grounding routes
# ===========================================================================


class TestPassThroughRoutes(unittest.TestCase):
    """Task 1/2/B_ENERGY/PERSONALIZATION/UNKNOWN must pass through unchanged."""

    def _check_passthrough(self, task: str) -> None:
        ev = {
            "answer": f"{task} answer",
            "confidence": 0.7,
            "explanation": f"{task} explanation",
            "label": "walking",
        }
        result = format_answer(task, ev)  # store=None is fine for passthrough
        self.assertEqual(result.task, task)
        self.assertEqual(result.answer, f"{task} answer")
        self.assertAlmostEqual(result.confidence, 0.7)
        self.assertIsNone(result.evidence_t_start)
        self.assertIsNone(result.evidence_t_end)
        self.assertIsNone(result.coverage_fraction)
        self.assertFalse(result.widened)
        self.assertTrue(result.raw_ptr_valid)

    def test_task1(self):
        self._check_passthrough("task1")

    def test_task2(self):
        self._check_passthrough("task2")

    def test_b_energy(self):
        self._check_passthrough("b_energy")

    def test_personalization(self):
        self._check_passthrough("personalization")

    def test_unknown(self):
        self._check_passthrough("unknown")


# ===========================================================================
# 2. Path (a) -- coverage > 0, raw_ptr valid -> render as-is
# ===========================================================================


class TestPathACoverageOk(unittest.TestCase):
    """When coverage > 0 and raw_ptr resolves, render the timestamp as-is."""

    def test_task3_renders_as_is(self):
        store = _make_store(coverage_fraction=0.75)
        ev = _make_evidence(task="task3")
        result = format_answer("task3", ev, store)

        self.assertFalse(result.widened)
        self.assertTrue(result.raw_ptr_valid)
        self.assertAlmostEqual(result.coverage_fraction, 0.75)
        self.assertAlmostEqual(result.evidence_t_start, T0)
        self.assertAlmostEqual(result.evidence_t_end, T0 + 20.0)
        self.assertEqual(result.answer, ev["answer"])
        self.assertNotIn("N/A", result.answer)
        self.assertNotIn("[Coverage note:", result.explanation)

    def test_task4_renders_as_is(self):
        store = _make_store(coverage_fraction=0.5)
        ev = _make_evidence(task="task4")
        result = format_answer("task4", ev, store)
        self.assertFalse(result.widened)
        self.assertFalse(result.fallback_na if hasattr(result, "fallback_na") else False)
        self.assertEqual(result.answer, ev["answer"])

    def test_b_anomaly_renders_as_is(self):
        store = _make_store(coverage_fraction=1.0)
        ev = _make_evidence(task="b_anomaly")
        result = format_answer("b_anomaly", ev, store)
        self.assertFalse(result.widened)
        self.assertEqual(result.evidence_t_start, T0)

    def test_coverage_fraction_propagated(self):
        store = _make_store(coverage_fraction=0.6)
        ev = _make_evidence(task="task3")
        result = format_answer("task3", ev, store)
        self.assertAlmostEqual(result.coverage_fraction, 0.6)


# ===========================================================================
# 3. Path (b) -- coverage = 0, widening succeeds
# ===========================================================================


class TestPathBWidening(unittest.TestCase):
    """When coverage = 0 but a covered neighbour exists, widen the interval."""

    def _make_widening_store(
        self,
        *,
        original_t_start: float,
        original_t_end: float,
        covered_offset: float = 5.0,
        seg_t_start: Optional[float] = None,
        seg_t_end: Optional[float] = None,
    ) -> MagicMock:
        """
        Build a store where:
        - get_coverage for the original interval returns 0.
        - get_coverage for a window near (mid + covered_offset) returns 0.8.
        - get_coverage for all other windows returns 0.
        """
        mid = (original_t_start + original_t_end) / 2.0
        probe_w = max(2.0, original_t_end - original_t_start)
        # The probe that should hit is centred at mid + covered_offset.
        hit_start = mid + covered_offset - probe_w / 2.0
        hit_end = hit_start + probe_w

        row = _FakeTimelineRow(
            t_start=seg_t_start if seg_t_start is not None else original_t_start - 10.0,
            t_end=seg_t_end if seg_t_end is not None else original_t_end + 100.0,
        )

        def _get_cov(sid, ts, te):
            # Return non-zero only for the window centred at the hit.
            if abs(ts - hit_start) < 0.05 and abs(te - hit_end) < 0.05:
                return 0.8
            return 0.0

        store = MagicMock()
        store.get_coverage.side_effect = _get_cov
        store.get_segment_raw.return_value = np.zeros((25, 3))
        store.get_timeline_row.return_value = row
        return store

    def test_widened_flag_set(self):
        store = self._make_widening_store(
            original_t_start=T0,
            original_t_end=T0 + 10.0,
            covered_offset=5.0,
        )
        ev = _make_evidence(task="task3", t_start=T0, t_end=T0 + 10.0)
        result = format_answer("task3", ev, store)

        self.assertTrue(result.widened)
        self.assertIsNotNone(result.evidence_t_start)
        self.assertIsNotNone(result.evidence_t_end)
        self.assertNotEqual(result.evidence_t_start, T0)

    def test_explanation_contains_coverage_note(self):
        store = self._make_widening_store(
            original_t_start=T0,
            original_t_end=T0 + 10.0,
            covered_offset=5.0,
        )
        ev = _make_evidence(task="task3", t_start=T0, t_end=T0 + 10.0)
        result = format_answer("task3", ev, store)

        self.assertIn("[Coverage note:", result.explanation)
        self.assertIn("No sensor burst was recorded", result.explanation)

    def test_answer_contains_widened_note(self):
        store = self._make_widening_store(
            original_t_start=T0,
            original_t_end=T0 + 10.0,
            covered_offset=5.0,
        )
        ev = _make_evidence(task="task3", t_start=T0, t_end=T0 + 10.0)
        result = format_answer("task3", ev, store)

        self.assertIn("widened", result.answer)

    def test_coverage_fraction_nonzero_after_widening(self):
        store = self._make_widening_store(
            original_t_start=T0,
            original_t_end=T0 + 10.0,
            covered_offset=5.0,
        )
        ev = _make_evidence(task="task3", t_start=T0, t_end=T0 + 10.0)
        result = format_answer("task3", ev, store)

        self.assertIsNotNone(result.coverage_fraction)
        self.assertGreater(result.coverage_fraction, 0.0)

    def test_raw_ptr_valid_after_widening(self):
        store = self._make_widening_store(
            original_t_start=T0,
            original_t_end=T0 + 10.0,
            covered_offset=5.0,
        )
        ev = _make_evidence(task="task3", t_start=T0, t_end=T0 + 10.0)
        result = format_answer("task3", ev, store)
        self.assertTrue(result.raw_ptr_valid)

    def test_render_text_shows_widened_tag(self):
        store = self._make_widening_store(
            original_t_start=T0,
            original_t_end=T0 + 10.0,
            covered_offset=5.0,
        )
        ev = _make_evidence(task="task3", t_start=T0, t_end=T0 + 10.0)
        result = format_answer("task3", ev, store)
        text = render_text(result)
        self.assertIn("[WIDENED]", text)


# ===========================================================================
# 4. Path (c) -- coverage = 0, no near signal -> N/A fallback
# ===========================================================================


class TestPathCFallback(unittest.TestCase):
    """When coverage = 0 and no near signal, answer must be N/A."""

    def _make_zero_coverage_store(
        self,
        *,
        seg_t_start: float = T0 - 5.0,
        seg_t_end: float = T0 + 25.0,
    ) -> MagicMock:
        """Store where every get_coverage call returns 0."""
        row = _FakeTimelineRow(t_start=seg_t_start, t_end=seg_t_end)
        store = MagicMock()
        store.get_coverage.return_value = 0.0
        store.get_segment_raw.return_value = None
        store.get_timeline_row.return_value = row
        return store

    def test_answer_is_na(self):
        store = self._make_zero_coverage_store()
        ev = _make_evidence(task="task3", t_start=T0, t_end=T0 + 10.0)
        result = format_answer("task3", ev, store, widen_radius_s=10.0)

        self.assertIn("N/A", result.answer)
        self.assertFalse(result.widened)
        self.assertFalse(result.raw_ptr_valid)
        self.assertIsNone(result.evidence_t_start)
        self.assertIsNone(result.evidence_t_end)

    def test_coverage_fraction_is_zero(self):
        store = self._make_zero_coverage_store()
        ev = _make_evidence(task="task3", t_start=T0, t_end=T0 + 10.0)
        result = format_answer("task3", ev, store, widen_radius_s=10.0)
        self.assertAlmostEqual(result.coverage_fraction, 0.0)

    def test_explanation_contains_coverage_note(self):
        store = self._make_zero_coverage_store()
        ev = _make_evidence(task="task3", t_start=T0, t_end=T0 + 10.0)
        result = format_answer("task3", ev, store, widen_radius_s=10.0)
        self.assertIn("[Coverage note:", result.explanation)
        self.assertIn("structural gap", result.explanation)

    def test_b_anomaly_also_falls_back(self):
        store = self._make_zero_coverage_store()
        ev = _make_evidence(task="b_anomaly", t_start=T0, t_end=T0 + 5.0)
        result = format_answer("b_anomaly", ev, store, widen_radius_s=10.0)
        self.assertIn("N/A", result.answer)

    def test_render_text_interval_is_na(self):
        store = self._make_zero_coverage_store()
        ev = _make_evidence(task="task3", t_start=T0, t_end=T0 + 10.0)
        result = format_answer("task3", ev, store, widen_radius_s=10.0)
        text = render_text(result)
        self.assertIn("Interval   : N/A", text)
        self.assertNotIn("[WIDENED]", text)


# ===========================================================================
# 5. check_coverage unit tests (isolated gate logic)
# ===========================================================================


class TestCheckCoverage(unittest.TestCase):
    """Isolated tests for the check_coverage gate."""

    def _store_with_profile(self, profile: dict) -> MagicMock:
        """
        profile maps (t_start_round, t_end_round) -> fraction.
        Rounds to 1 decimal place for key lookup.
        """
        def _get_cov(sid, ts, te):
            key = (round(ts, 1), round(te, 1))
            return profile.get(key, 0.0)

        row = _FakeTimelineRow(t_start=T0 - 10.0, t_end=T0 + 100.0)
        store = MagicMock()
        store.get_coverage.side_effect = _get_cov
        store.get_segment_raw.return_value = np.zeros((10, 3))
        store.get_timeline_row.return_value = row
        return store

    def test_covered_interval_returns_correct_fraction(self):
        store = _make_store(coverage_fraction=0.6)
        result = check_coverage(store, "s1", T0, T0 + 10.0)
        self.assertAlmostEqual(result.fraction, 0.6)
        self.assertFalse(result.widened)
        self.assertFalse(result.fallback_na)
        self.assertTrue(result.raw_ptr_valid)

    def test_zero_coverage_triggers_widening(self):
        # Original window returns 0; probe window 5 s to the right returns 0.8.
        store = _make_store(coverage_fraction=0.0)

        mid = T0 + 5.0
        probe_w = 10.0  # t_end - t_start

        def _cov(sid, ts, te):
            # Match the probe centred ~5 s to the right of mid.
            if abs(ts - (mid + 5.0 - probe_w / 2.0)) < 0.1:
                return 0.8
            return 0.0

        row = _FakeTimelineRow(t_start=T0 - 10.0, t_end=T0 + 200.0)
        store.get_coverage.side_effect = _cov
        store.get_timeline_row.return_value = row

        result = check_coverage(store, "s1", T0, T0 + 10.0, widen_probe_step_s=5.0)
        self.assertTrue(result.widened)
        self.assertGreater(result.fraction, 0.0)
        self.assertIsNotNone(result.widened_t_start)
        self.assertIsNotNone(result.widened_t_end)

    def test_zero_coverage_no_neighbour_gives_fallback_na(self):
        store = MagicMock()
        store.get_coverage.return_value = 0.0
        store.get_segment_raw.return_value = None
        row = _FakeTimelineRow(t_start=T0, t_end=T0 + 20.0)
        store.get_timeline_row.return_value = row

        result = check_coverage(
            store, "s1", T0 + 5.0, T0 + 15.0, widen_radius_s=5.0
        )
        self.assertTrue(result.fallback_na)
        self.assertFalse(result.widened)
        self.assertAlmostEqual(result.fraction, 0.0)

    def test_widening_clamped_to_segment_span(self):
        """Widening must not extend beyond the segment boundaries."""
        call_log: list[tuple[float, float]] = []

        def _cov(sid, ts, te):
            call_log.append((ts, te))
            return 0.0

        row = _FakeTimelineRow(t_start=T0, t_end=T0 + 30.0)
        store = MagicMock()
        store.get_coverage.side_effect = _cov
        store.get_segment_raw.return_value = None
        store.get_timeline_row.return_value = row

        check_coverage(
            store, "s1", T0 + 10.0, T0 + 20.0,
            widen_radius_s=50.0, widen_probe_step_s=1.0,
        )
        for ts, te in call_log:
            self.assertGreaterEqual(ts, T0 - 1e-6, msg=f"Probe {ts:.1f} < seg_t_start {T0:.1f}")
            self.assertLessEqual(te, T0 + 30.0 + 1e-6, msg=f"Probe {te:.1f} > seg_t_end {T0 + 30.0:.1f}")

    def test_raw_ptr_invalid_triggers_widening(self):
        """Coverage > 0 but raw_ptr missing should start widening."""
        call_count = [0]

        def _cov(sid, ts, te):
            call_count[0] += 1
            # First call (original window): return 0.5 to trigger raw_ptr check.
            if call_count[0] == 1:
                return 0.5
            # Subsequent probes: return 0.9 so widening succeeds.
            return 0.9

        row = _FakeTimelineRow(t_start=T0 - 10.0, t_end=T0 + 100.0)
        store = MagicMock()
        store.get_coverage.side_effect = _cov
        store.get_segment_raw.return_value = None  # raw_ptr broken
        store.get_timeline_row.return_value = row

        result = check_coverage(store, "s1", T0, T0 + 10.0)
        # raw_ptr was broken, so should have fallen through to widening.
        # Widening should have found coverage 0.9 on the first probe.
        self.assertTrue(result.widened or result.fallback_na,
                        "Expected widened or fallback_na when raw_ptr is broken")


# ===========================================================================
# 6. render_text shape tests
# ===========================================================================


class TestRenderText(unittest.TestCase):
    """render_text must produce the fixed template regardless of content."""

    def test_task3_good_coverage(self):
        fa = FormattedAnswer(
            task="task3",
            answer="She started walking at 10:00.",
            confidence=0.85,
            explanation="Cadence 95 bpm.",
            evidence_t_start=T0,
            evidence_t_end=T0 + 20.0,
            coverage_fraction=0.9,
            widened=False,
            raw_ptr_valid=True,
            label="walking",
        )
        text = render_text(fa)
        self.assertIn("=== Task 3", text)
        self.assertIn("Answer     :", text)
        self.assertIn("Confidence : 85%", text)
        self.assertIn("Interval   :", text)
        self.assertIn("coverage: 90%", text)
        self.assertNotIn("[WIDENED]", text)

    def test_task3_widened(self):
        fa = FormattedAnswer(
            task="task3",
            answer="She started walking (widened).",
            confidence=0.7,
            explanation="Original interval had no signal.",
            evidence_t_start=T0 + 5.0,
            evidence_t_end=T0 + 25.0,
            coverage_fraction=0.6,
            widened=True,
            raw_ptr_valid=True,
            label="walking",
        )
        text = render_text(fa)
        self.assertIn("[WIDENED]", text)
        self.assertIn("coverage: 60%", text)

    def test_fallback_na_interval_is_na(self):
        fa = FormattedAnswer(
            task="task3",
            answer="N/A (onset outside recorded coverage)",
            confidence=0.5,
            explanation="Gap note.",
            evidence_t_start=None,
            evidence_t_end=None,
            coverage_fraction=0.0,
            widened=False,
            raw_ptr_valid=False,
            label="walking",
        )
        text = render_text(fa)
        self.assertIn("Interval   : N/A", text)
        self.assertNotIn("[WIDENED]", text)

    def test_task1_passthrough_no_interval(self):
        fa = FormattedAnswer(
            task="task1",
            answer="She was walking.",
            confidence=0.9,
            explanation="Label probability: 0.9.",
            evidence_t_start=None,
            evidence_t_end=None,
            coverage_fraction=None,
            widened=False,
            raw_ptr_valid=True,
            label="walking",
        )
        text = render_text(fa)
        self.assertIn("=== Task 1", text)
        self.assertIn("Interval   : N/A", text)
        # coverage fraction is None so "(coverage:..." should NOT appear
        self.assertNotIn("coverage:", text)

    def test_confidence_none_renders_na(self):
        fa = FormattedAnswer(
            task="task2",
            answer="45 minutes.",
            confidence=None,
            explanation="Sum of labeled minutes.",
        )
        text = render_text(fa)
        self.assertIn("Confidence : N/A", text)


# ===========================================================================
# 7. format_answer with store=None (warning path)
# ===========================================================================


class TestNoStore(unittest.TestCase):
    def test_task3_no_store_passes_through_with_warning(self):
        ev = _make_evidence(task="task3")
        with self.assertLogs("formatter", level="WARNING") as cm:
            result = format_answer("task3", ev, store=None)
        self.assertTrue(
            any("no store provided" in msg.lower() for msg in cm.output),
            "Expected a 'no store' warning",
        )
        self.assertIsNone(result.coverage_fraction)
        self.assertEqual(result.answer, ev["answer"])
        self.assertAlmostEqual(result.evidence_t_start, T0)

    def test_task1_no_store_no_warning(self):
        ev = {"answer": "Walking.", "confidence": 0.9, "explanation": "OK."}
        # Should NOT emit a warning for non-grounding routes.
        import logging
        with self.assertRaises(AssertionError):
            with self.assertLogs("formatter", level="WARNING"):
                format_answer("task1", ev, store=None)


# ===========================================================================
# 8. Integration smoke test with a real in-memory ActivityStore
# ===========================================================================


class TestWithRealStore(unittest.TestCase):
    """Smoke tests using the actual ActivityStore from store.py."""

    def setUp(self):
        try:
            from store import ActivityStore, TimelineRow
            self._ActivityStore = ActivityStore
            self._TimelineRow = TimelineRow
        except ImportError:
            self.skipTest("store.py not available")

    def _build_store_with_signal(self, fs: float = 25.0) -> tuple:
        """Return (store, tmpdir, seg_id, seg_t_start, seg_t_end)."""
        tmpdir = tempfile.mkdtemp()
        self._tmpdir = tmpdir
        db_path = os.path.join(tmpdir, "test.db")
        store = self._ActivityStore(db_path, array_dir=tmpdir)

        seg_id = "smoke-seg-001"
        seg_t_start = T0
        seg_t_end = T0 + 20.0
        n_samples = int((seg_t_end - seg_t_start) * fs)
        body_acc = np.random.randn(n_samples, 3).astype(np.float32)

        row = self._TimelineRow(
            segment_id=seg_id,
            uuid="smoke-uuid",
            t_label_start_ref=int(T0),
            t_start=seg_t_start,
            t_end=seg_t_end,
            label="walking",
            confidence=0.8,
            coverage_s=20.0,
            label_minutes_covered=1,
            n_windows=10,
        )

        # Fake PreprocessedSignal-like object for upsert_segment.
        # Use a factory function to capture n_samples in a closure so
        # the inner class body can reference it without relying on the
        # enclosing method's local scope (which is not available at class
        # body definition time in all Python versions).
        def _make_fake_sig(n: int, f: float, acc: np.ndarray, t0: int):
            class _FakeSig:
                pass
            sig = _FakeSig()
            sig.t = np.arange(n, dtype=np.float64) / f
            sig.timestamp = t0
            sig.n_samples = n
            sig.body_acc = acc
            sig.fs = f
            return sig

        store.upsert_segment(row, raw_signal=_make_fake_sig(n_samples, fs, body_acc, int(T0)))
        return store, tmpdir, seg_id, seg_t_start, seg_t_end

    def tearDown(self):
        import shutil
        tmpdir = getattr(self, "_tmpdir", None)
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_path_a_real_store(self):
        store, tmpdir, seg_id, ts, te = self._build_store_with_signal()
        self._tmpdir = tmpdir
        ev = _make_evidence(
            task="task3",
            segment_id=seg_id,
            t_start=ts,
            t_end=te,
        )
        result = format_answer("task3", ev, store)
        self.assertFalse(result.widened)
        self.assertTrue(result.raw_ptr_valid)
        self.assertGreater(result.coverage_fraction, 0.0)
        store.close()

    def test_path_c_real_store_no_signal(self):
        """Segment with no raw_ptr -> fallback to N/A."""
        tmpdir = tempfile.mkdtemp()
        self._tmpdir = tmpdir
        db_path = os.path.join(tmpdir, "test2.db")
        store = self._ActivityStore(db_path, array_dir=tmpdir)

        seg_id = "no-signal-seg"
        row = self._TimelineRow(
            segment_id=seg_id,
            uuid="smoke-uuid",
            t_label_start_ref=int(T0),
            t_start=T0,
            t_end=T0 + 20.0,
            label="sitting",
            confidence=0.6,
            coverage_s=0.0,   # No signal
            label_minutes_covered=1,
            n_windows=0,
        )
        store.upsert_segment(row)  # No raw_signal -> raw_ptr=None

        ev = _make_evidence(task="task3", segment_id=seg_id, t_start=T0, t_end=T0 + 10.0)
        result = format_answer("task3", ev, store, widen_radius_s=5.0)

        self.assertIn("N/A", result.answer)
        self.assertFalse(result.widened)
        store.close()


if __name__ == "__main__":
    unittest.main()
