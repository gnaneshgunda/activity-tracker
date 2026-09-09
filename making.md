# Ask the Sensors — Updated Pipeline & Staged Build Prompts (v2)

Supersedes the original "Full Pipeline & Build Prompts" doc. Changes: two-tier time base (labeled-minute + captured-burst), gravity/body-acceleration split, cadence/SMA/axis-resolved gyro features, HMM/Viterbi segmentation, a physics-rule anomaly/event detector, an energy (MET) estimator, and a coverage field that travels through the whole pipeline so no answer claims precision the data doesn't have.

---

## Non-negotiable rules for every prompt below

Paste these with whichever stage prompt you run — as a standing project instruction in Claude Code, or repeated inline if you're pasting stage-by-stage into claude.ai.

**1. Disclose and cite, in the file itself.** Every generated file opens with:
```python
"""
Generated with the assistance of Claude (Anthropic).
Reviewed and modified by: <names>.
Primary source(s) for this design: <citation(s) — see "Cite:" line in this stage>.
"""
```
This satisfies the course's AI-disclosure rule and its "cite every external resource at the point you use it" rule in one place — a grader or teammate can tell which file came from where without reading the whole git history. Repeat the same citation in your report at the point you use that design choice.

**2. Don't force a citation where none exists.** Some stages below cite a specific paper because the choice is directly traceable to it. A few are standard technique with no single canonical source — those are marked "standard technique, no forced citation." A fabricated reference is a worse integrity problem than an honest "this is common practice" note, so don't let the assistant invent one to fill the slot.

**3. One stage = one commit.** Commit message suggestions are given per stage. Don't batch stages — the grading rubric explicitly reads commit history as evidence of real incremental work, not just the final diff.

---

## Source map

Some entries below only have title + venue + link confirmed — pull full author lists from the link before citing formally in your report; don't take my partial citation as the final form.

| Source | Backs |
|---|---|
| Vaizman, Ellis & Lanckriet, "Recognizing Detailed Human Context In-the-Wild from Smartphones and Smartwatches," IEEE Pervasive Computing 2017 (arXiv:1609.06354) | B0 (burst/sampling structure), B1 (the gravity/body split the official ExtraSensory feature set doesn't include) |
| DrHouse — Sui et al., Proc. ACM IMWUT 8(4), 2024, doi:10.1145/3699765 — **course reading [1]** | B9 (fusing sensed evidence + a knowledge base into grounded reasoning) |
| JARVIS for HVAC — Lee et al., Proc. ACM IMWUT 10(2) Art.51, 2026, doi:10.1145/3810210 — **course reading [2]** | B8 (multi-stage, sensor-grounded LLM QA routing) |
| SensorChat — Yu et al., Proc. ACM IMWUT 9(3), 2025, doi:10.1145/3749496 — **course reading [3]** | B8 (routing qualitative vs. quantitative questions over long-duration sensor data) |
| Sensor2Text, Proc. ACM IMWUT, 2024, doi:10.1145/3699747 — **course reading [4]** | B9 (NL interaction for daily activity tracking, framed around elderly/health monitoring — closest published match to Aparna's scenario) |
| Ainsworth et al., Compendium of Physical Activities (pacompendium.com) | B_E (MET values, kcal formula) |
| "Using Hidden Markov Models to Improve Quantifying Physical Activity in Accelerometer Data — A Simulation Study," PLOS ONE, 2014 (PMC4251969) | B3 (Viterbi-decoded HMM smoothing) |
| "Long-term Activities Segmentation using Viterbi Algorithm with a k-minimum-consecutive-states Constraint" (ResearchGate 268981604) | B3 (minimum-duration constraint) |
| "Development of a Wearable-Sensor-Based Fall Detection System" (PMC4346101) | B4.5 (impact-peak → stillness → orientation-change rule) |
| "Efficient Activity Recognition and Fall Detection Using Accelerometers" (Springer, doi:10.1007/978-3-642-41043-7_2) | B4.5 (same rule, independent source) |

Readings [1]–[4] are the four ACM links already given in your brief — you don't need to go find them, they're already relevant, I've just matched each to the block it actually informs.

---

## Updated architecture

```
 ExtraSensory per-user files (features_labels.csv.gz + matching raw sensor bursts)
        │
        ▼
 [B0] Ingestion & Alignment    → joined example: {uuid, t_label (60s slot),
        │                          raw_burst (~20s @ ~40Hz or None), label, coverage_s}
        ▼
 [B1] Preprocessing            → resampled 25Hz filtered + RAW acc/gyro (kept)
        │                          + gravity(t) + body_acc(t)  [NEW: gravity/body split]
        ▼
 [B2] Window Recognition       → per-window {label_probs, entropy, cadence_hz,
        │                          sma, gyro_rms_x/y/z} @ 1–2s windows
        ▼
 [B3] Segmentation (HMM/Viterbi, k-min-duration) → segments, CUSUM-snapped boundaries
        │
        ▼
 [B4] Signature Extraction     → per-segment physics signature (tilt, cadence_bpm,
        │                          jerk_energy, sma, gyro_rms x/y/z, orientation_stability)
        │
        ├───────────────────────────────┐
        ▼                                ▼
 [B4.5] Anomaly / Event Detector   [B5] Storage Layer
   (jerk-spike + orientation-        (segments + signatures + raw_ptr + coverage_s
    stability-collapse rule;          + uuid/timestamp provenance + anomaly-event table)
    IsolationForest fallback)               │
        │                                    │
        └────────────────┬───────────────────┘
                          ▼
               [B6] Daily Rollup ───────► [B_E] Energy / MET Estimator
                 (60s per labeled            (activity + duration + assumed weight
                  minute, not 20s)             → kcal estimate, flagged as population-
                          │                     level, callable standalone from B8 too)
                          ▼
               [B7] Rolling Trends (7d / monthly / yearly, n_days_with_data always reported)
                          │
                          ▼
               [B8] Query Router (+ energy queries, + anomaly/"unsteady" queries)
                          │
        ┌─────────────────┼──────────────────┬───────────────────┬────────────────┐
        ▼                 ▼                  ▼                   ▼                ▼
  Task 1 lookup    Task 2 aggregate    Task 3 grounding    Task 4 RAG+SLM    Energy / event
  (label/prob)     (sum/count/cmp,     (onset/interval,     (B9 exemplar +    lookup (B_E /
                    60s-per-minute)     coverage-checked)    physics-cited     B4.5 output)
                                                              explanation)
                          │
                          ▼
               [B10] Output Formatter
        (renders the fixed template; widens or refuses a timestamp citation
         if it falls outside real coverage, rather than fabricating precision)
                          │
                          ▼
               [B11] Streamlit UI (+ coverage indicator, + energy toggle)
```

---

## Updated data contracts

```python
# B0 output — one row per matched (uuid, labeled-minute) example
IngestedExample = {
    "uuid": str,
    "t_label_start": float,      # unix seconds, start of the labeled 60s minute
    "burst": {                   # the raw sensor burst inside this minute, if any
        "t": np.ndarray,         # seconds, native sample timestamps
        "acc": np.ndarray,       # (M,3) raw
        "gyro": np.ndarray,      # (M,3) raw
    } | None,
    "label": str | None,         # 7-class self-report, or None if missing/ambiguous
    "coverage_s": float,         # seconds of real burst inside this 60s label
}

# B1 output (per burst, resampled to 25Hz)
PreprocessedSignal = {
    "t": np.ndarray,             # seconds, 25Hz, burst-local
    "acc": np.ndarray,           # (N,3) Butterworth-filtered
    "acc_raw": np.ndarray,       # (N,3) unfiltered — for jerk/noise features
    "gyro": np.ndarray,          # (N,3) filtered
    "gyro_raw": np.ndarray,
    "gravity": np.ndarray,       # (N,3) low-cutoff estimate of gravity direction, |g|=1
    "body_acc": np.ndarray,      # (N,3) acc_raw - gravity ("user"/dynamic acceleration)
    "gaps": list[tuple[float,float]],   # interpolated/missing spans WITHIN the burst
}

# B2 output — one row per window, inside the captured burst only
Window = {
    "t_start": float, "t_end": float,
    "probs": dict[str, float],       # 7-class softmax, full distribution
    "entropy": float,                # confidence proxy for hedged Task 1/4 answers
    "cadence_hz": float | None,      # peak rate on the gravity-projected vertical axis
    "sma": float,                    # signal magnitude area of body_acc
    "gyro_rms_xyz": tuple[float, float, float],   # axis-resolved, for channel-level evidence
}

# B3 output — one row per segment (Viterbi-decoded, k-min-duration constrained)
Segment = {
    "segment_id": str, "t_start": float, "t_end": float,
    "label": str, "confidence": float,
    "label_minutes_covered": int,    # how many labeled ExtraSensory minutes this spans
}

# B4 output — attached to each Segment
Signature = {
    "mag_mean": float, "mag_std": float,
    "tilt_deg_mean": float, "tilt_deg_std": float,     # NEW — from gravity(t)
    "dom_freq_hz": float, "dom_freq_power": float,
    "cadence_bpm": float | None,                        # cross-checked vs. B2 peak-counting
    "jerk_energy": float,
    "sma": float,                                       # NEW
    "gyro_rms_x": float, "gyro_rms_y": float, "gyro_rms_z": float,   # NEW, axis-resolved
    "periodicity": float,
    "orientation_stability": float,
}

# B4.5 output — zero or more per segment
AnomalyEvent = {
    "event_id": str, "t_start": float, "t_end": float,
    "kind": str,                      # e.g. "possible_fall", "prolonged_immobility"
    "trigger_signature": dict,        # the jerk_energy / orientation_stability that fired it
    "method": str,                    # "physics_rule" | "isolation_forest"
    "coverage_s": float,
}

# B5 storage — TimelineRow, extended with provenance + coverage
TimelineRow = {
    **Segment, "signature": Signature, "raw_ptr": tuple[int, int],
    "uuid": str, "t_label_start_ref": float,     # traceable back to the ExtraSensory example
    "coverage_s": float,                          # seconds of this segment backed by real signal
}

# B_E output — attached wherever an activity+duration pair needs an energy estimate
EnergyEstimate = {
    "activity": str, "duration_s": float,
    "met": float, "assumed_weight_kg": float,     # explicit assumption, not measured
    "kcal_est": float,
    "basis": str,                                  # e.g. "Compendium of Physical Activities, walking/flat"
}
```

---

## Stage prompts

### Stage 1 — B0: Ingestion & Alignment (new)
**Commit:** `feat(ingestion): join features_labels examples to matching raw bursts, track coverage`

> Write `ingest.py`. For a given ExtraSensory UUID, load that user's `features_labels.csv.gz` row by row, and for each labeled minute locate and load the matching raw accelerometer+gyroscope burst from the raw-data release (join key: uuid + timestamp). Output a list of `IngestedExample` structs [paste struct]. If no matching raw burst exists for a labeled minute, set `burst=None` and `coverage_s=0` — never fabricate a burst. Restrict to the 7-class subset (lying down, sitting, standing in place, standing and moving, walking, running, bicycling); use the missing-label matrix to drop or flag ambiguous examples rather than guessing a label. Cite: Vaizman, Ellis & Lanckriet (2017) for the burst/sampling structure this join depends on — put it in the module docstring and reference it again at this point in your report. Include a unit test with a synthetic UUID that has one matched minute and one deliberately unmatched one.

### Stage 2 — B1: Preprocessing + gravity/body split (extended)
**Commit:** `feat(preprocess): resample to 25Hz, add gravity/body-acceleration split`

> Extend `preprocess.py` to take `IngestedExample.burst` and return `PreprocessedSignal` [paste struct]. Resample to 25Hz by linear interpolation, only across gaps under 2s within the burst — longer gaps go in `gaps` as NaN, not interpolated. Apply a 2nd-order Butterworth low-pass (cutoff ~10Hz) for the filtered arrays, but keep the untouched raw arrays. NEW: estimate the gravity vector via a much lower cutoff low-pass on raw acceleration (many HAR pipelines use ~0.3Hz as a starting point — tune to your data and say what you used), and compute `body_acc = acc_raw − gravity`. This is a genuine addition, not a restatement — the official ExtraSensory feature set was built on raw, gravity-included acceleration, not this split. Cite: Vaizman, Ellis & Lanckriet (2017) for that gap. Unit test with a synthetic constant-tilt signal and confirm the recovered gravity angle matches the injected tilt within a couple of degrees.

### Stage 3 — B2: Window Recognition + physics features (extended)
**Commit:** `feat(recognize): add cadence, SMA, axis-resolved gyro RMS, confidence to window features`

> Extend `recognize.py`'s window feature extraction (2s window, 50% overlap) to add: `cadence_hz` via peak-counting on the vertical component of `body_acc` (project onto the gravity direction from B1); `sma = mean(|body_acc|)` over the window; `gyro_rms_xyz` as three separate per-axis values — keep these axis-resolved, since Task 3/4 evidence has to name a specific channel, not just "gyroscope"; softmax entropy as a confidence proxy for hedged answers. Output `Window` structs [paste struct] with the full probability distribution, not argmax. Standard technique, no forced citation — cadence-by-peak-counting, SMA, and per-axis RMS are common accelerometry features, not tied to one paper; say so in the docstring instead of inventing a reference.

### Stage 4 — B3: HMM/Viterbi segmentation (upgraded from median filter)
**Commit:** `feat(segment): replace median-filter smoothing with Viterbi-decoded HMM + k-min-duration constraint`

> Rewrite `segment.py`'s smoothing step. Instead of a median filter, fit a first-order HMM over the 7 activity states — emission probabilities from B2's window softmax, transition matrix estimated from label-sequence counts in your training split (or a hand-specified physically-plausible prior if training counts are too sparse for a state). Run Viterbi to decode the most likely state sequence, adding a k-minimum-consecutive-states constraint so the decoder can't produce single-window blips shorter than a physically plausible minimum activity duration. After Viterbi fixes the coarse label sequence, keep the existing CUSUM boundary refinement (±3s around each decoded transition) to snap `t_start`/`t_end` precisely. Return `Segment` structs [paste struct], sorted, no overlaps, gaps only where `coverage_s` says there's no real signal. Cite: "Using Hidden Markov Models to Improve Quantifying Physical Activity in Accelerometer Data" (PLOS ONE, 2014) for the Viterbi-smoothing approach itself, and "Long-term Activities Segmentation using Viterbi Algorithm with a k-minimum-consecutive-states Constraint" for the duration constraint specifically — both in the source map above; pull full author details from the links before citing formally.

### Stage 5 — B4: Signature Extraction (extended)
**Commit:** `feat(signature): add tilt, SMA, per-axis gyro RMS to segment signatures`

> Extend `signature.py`'s `extract_signature(segment, preprocessed_signal) -> Signature` [paste struct] to add: `tilt_deg_mean`/`tilt_deg_std` from the `gravity(t)` array (device orientation relative to Earth — this is what should separate sitting from standing, not magnitude); `sma`; `gyro_rms_x/y/z` as three fields, not one. Keep `cadence_bpm` computed both ways — autocorrelation on the filtered signal (as before) and cross-checked against B2's peak-counted `cadence_hz` — and flag in the output if the two disagree by more than a small tolerance, since that disagreement is itself a useful low-confidence signal. This function still must not depend on the segment's `label` — Task 4 will call it on segments the classifier may have gotten wrong. Standard technique, no forced citation for the extraction itself; the tilt feature's rationale traces back to the Stage 2 citation (Vaizman et al. 2017), which you can reference again here rather than re-deriving.

### Stage 6 — B4.5: Anomaly / Event Detector (new)
**Commit:** `feat(anomaly): add physics-rule fall/event detector on top of Signature output`

> Write `anomaly.py`. Primary detector: a rule over `Signature` — flag an event when `jerk_energy` spikes above a threshold and, within a short window after, `orientation_stability` collapses (tilt settles near-horizontal) and stays there for longer than a brief stumble would. Output `AnomalyEvent` structs [paste struct] with `method="physics_rule"` and the actual triggering `jerk_energy`/`orientation_stability` values stored in `trigger_signature`, so the eventual Explanation field can cite real numbers instead of a bare verdict. Add a documented fallback path, `method="isolation_forest"`, trained on Signature vectors from segments with no flagged event, for use only if the rule's false-positive rate turns out too high once you test it against real data — don't build the fallback as the primary path without first checking whether the rule alone is good enough. Cite: "Development of a Wearable-Sensor-Based Fall Detection System" (PMC4346101) and "Efficient Activity Recognition and Fall Detection Using Accelerometers" (Springer) for the impact-peak → stillness → orientation-change pattern — both in the source map; pull full author details before citing formally.

### Stage 7 — B5: Storage Layer (extended)
**Commit:** `feat(store): add coverage, provenance keys, and an anomaly-event table`

> Extend `store.py`'s SQLite/parquet wrapper. `timeline` table rows are now `TimelineRow` [paste struct] — add `coverage_s` and the `uuid`/`t_label_start_ref` provenance pair so any answer can be traced back to the exact ExtraSensory example it came from. Add a second table for `AnomalyEvent` rows, keyed the same way. `get_segment_raw(segment_id)` still resolves `raw_ptr` to the actual signal array; add `get_coverage(segment_id, t_start, t_end)` that returns what fraction of a *queried* sub-interval (which may be narrower than the full segment) is actually backed by captured signal — B10 will need this to decide whether to cite a precise timestamp or widen the citation. Standard engineering, no forced citation — this block is infrastructure, not a research technique.

### Stage 8 — B6+B7: Daily Rollup & Rolling Trends (lightly updated)
**Commit:** `feat(rollup): compute daily/rolling totals using 60s-per-labeled-minute, not burst length`

> Write `rollup.py`. `compute_daily_rollup(user_id, date, timeline_rows) -> list[DailyRollup]` groups a day's segments by activity and computes total duration, bout count, avg bout length, first/last onset — **count each labeled minute as 60 seconds, not the ~20s burst length**; the burst is the evidence for the label, not the duration itself. `compute_rolling(daily_rollups_df, activity, window)` returns a rolling mean over `total_duration_s` for `window in {"7d","monthly","yearly"}` using calendar-aware resampling for monthly/yearly and a plain rolling window for 7d, and must always return `n_days_with_data` alongside the average. Standard descriptive statistics, no forced citation.

### Stage 9 — B_E: Energy / MET Estimator (new)
**Commit:** `feat(energy): add MET-lookup energy estimator, callable standalone or from rollup`

> Write `energy.py` with `estimate_energy(activity: str, duration_s: float, weight_kg: float = 62.0) -> EnergyEstimate` [paste struct]. Use a small lookup table of typical MET values for the 7 classes (roughly: lying ~1.0–1.3, sitting ~1.5, standing in place ~2.0, standing and moving ~2.3, walking ~3 [2–4.5 by pace], running ~7 [6–13+ by pace], bicycling ~7 [4–10+ by effort]), and convert with `kcal ≈ MET × weight_kg × 3.5 / 200 × (duration_s / 60)`. Make the function callable standalone from the query router (a one-off "how many calories during her walk" question) and from B6 (a daily kcal rollup) — don't couple it only to the rollup path. Set `basis` to name which compendium category you used. This must never claim more precision than it has: document in the docstring that these are population-level MET classifications, not a measurement of this specific user, and that `weight_kg` is an assumed default unless you have an actual value — the estimate is not new evidence and shouldn't be treated as strengthening any Task 3 grounding claim. Cite: Ainsworth et al., Compendium of Physical Activities (pacompendium.com).

### Stage 10 — B8: Query Router (extended)
**Commit:** `feat(router): add energy and anomaly/"unsteady" query routing`

> Extend `router.py`'s `route(question: str) -> dict` to also recognize energy/calorie questions (route to B_E) and anomaly/fall/"unsteady"-style questions (route to B4.5's stored events) alongside the existing Task1–4 + Personalization categories. Keep the rule-based keyword/regex classifier as the first pass, falling back to a small instruction-tuned SLM only when rules don't match confidently, logging which path was used. Cite: JARVIS for HVAC (Lee et al. 2026) for the general pattern of routing long-term sensor-grounded questions through staged reasoning, and SensorChat (Yu et al. 2025) specifically for splitting qualitative vs. quantitative question handling — both course readings, full citations in the source map above.

### Stage 11 — B9: Exemplar Bank & Open-World Reasoning (extended)
**Commit:** `feat(exemplars): feed physics signature to SLM narration, require cited numbers`

> Extend `exemplars.py`. Offline: for each of the 7 classes, select 3–5 representative training segments and store their (now physics-extended) `Signature` vectors with a short human-written description. At query time: `nearest_exemplars(query_signature, k=3)` via Euclidean/cosine distance. Then `explain(query_signature, nearest_exemplars, slm)` must prompt the SLM to compare the query's actual numeric signature values (tilt, cadence, jerk, SMA, per-axis gyro RMS) against the retrieved exemplar's, and write an explanation that names specific numbers and specific channels — not a generic activity description from the SLM's general knowledge. This is what B4.5's anomaly events should also route through for their narration. Cite: DrHouse (Sui et al. 2024) for combining sensed evidence with a knowledge base into grounded, checkable reasoning, and Sensor2Text (2024) as the closest published match to this exact elderly-monitoring, NL-interaction framing — both course readings.

### Stage 12 — B10: Output Formatter (extended, coverage-aware)
**Commit:** `feat(formatter): refuse or widen a timestamp citation that falls outside real coverage`

> Extend `formatter.py`'s `format_answer(...)` to call B5's `get_coverage(segment_id, t_start, t_end)` before rendering any Task 3/4 evidence. If the claimed interval's coverage is 0% (falls entirely in one of the structural gaps between captured bursts), either widen the cited interval to the nearest span that actually has real signal and say so in the Explanation, or fall back toward N/A rather than inventing a precise onset the data can't support — never render a fine-grained timestamp that isn't backed by real coverage. Keep the existing validation that every timestamp resolves to a real `raw_ptr` before rendering. Standard engineering/honesty rule, no forced citation — it's a direct consequence of the coverage design from Stages 1 and 7, reference those in your report instead of inventing an external source for this one.

### Stage 13 — B11: Streamlit UI (lightly extended)
**Commit:** `feat(ui): add coverage indicator and energy toggle to the app`

> Extend `app.py`'s three tabs: on Timeline, show a coverage indicator per segment (what fraction is backed by real signal) rather than presenting the whole timeline as uniformly certain; on Ask-a-Question, surface the confidence/entropy value alongside the answer when it's low; on Trends, add a toggle for estimated daily kcal (from B_E) next to the existing duration charts, labeled clearly as an estimate. No logic duplicated in the UI layer — wire to the block functions only. No forced citation — UI work, not a research technique.

---

## Suggested team split & checkpoints

- **Person A** (signal side): Stage 1 → 2 → 3 → 4
- **Person B** (features/storage/anomaly/exemplar side): Stage 5 → 6 → 7 → 11
- **Person C** (product/rollup/routing/UI side): Stage 8 → 9 → 10 → 12 → 13

**Checkpoint 1** (after Stage 7): B0–B5 produce a real Activity Timeline, with physics signatures and anomaly events, from actual ExtraSensory data.
**Checkpoint 2** (after Stage 13): B6–B11 answer a hand-written question of every task type — including an energy question and an anomaly/"unsteady" question — against that timeline.