# Ask the Sensors — Full Pipeline & Build Prompts

## 1. Architecture (block diagram)

```
 Raw IMU CSV (acc_x,y,z, gyro_x,y,z, timestamp)
        │
        ▼
 [B1] Preprocessing        → resampled 25Hz signal (raw + filtered, kept separately)
        │
        ▼
 [B2] Window Recognition   → per-window {label, confidence} @ 1–2s windows
        │
        ▼
 [B3] Segmentation         → variable-length segments (merge + boundary refine)
        │
        ▼
 [B4] Signature Extraction → per-segment feature vector (activity-agnostic)
        │
        ▼
 [B5] Storage Layer        → Activity Timeline (segments) + Raw Signal Store (pointers)
        │
        ├──────────────────────────────┐
        ▼                              ▼
 [B6] Daily Rollup             [B9] Exemplar Bank
   (per user, per day,           (few labeled segment
    per activity totals)          signatures per class,
        │                          built once offline)
        ▼                              │
 [B7] Rolling Trends                   │
   (7d / monthly / yearly)             │
        │                              │
        └──────────────┬───────────────┘
                        ▼
              [B8] Query Router
     (NL question → task type 1–4 → operation)
                        │
        ┌───────────────┼────────────────┬─────────────────┐
        ▼               ▼                ▼                 ▼
  Task 1 lookup   Task 2 aggregate  Task 3 grounding   Task 4 RAG+SLM
  (label/prob)    (sum/count/cmp)   (onset/interval)   (nearest exemplar
                                                          + SLM narration)
                        │
                        ▼
              [B10] Output Formatter
        (enforces the fixed Answer/Evidence template)
                        │
                        ▼
              [B11] Streamlit UI
```

Each block is a separate module with a fixed input/output contract, so you can build and unit-test them independently and hand them off between group members.

## 2. Data contracts between blocks

```python
# B1 output
PreprocessedSignal = {
    "t": np.ndarray,          # seconds, 25Hz
    "acc": np.ndarray,        # (N,3) filtered
    "acc_raw": np.ndarray,    # (N,3) unfiltered, kept for jerk/noise features
    "gyro": np.ndarray,       # (N,3) filtered
    "gyro_raw": np.ndarray,
    "gaps": list[tuple[float,float]],  # (start,end) of interpolated/missing spans
}

# B2 output — one row per window
Window = {"t_start": float, "t_end": float, "probs": dict[str,float]}  # 7 classes

# B3 output — one row per segment
Segment = {
    "segment_id": str, "t_start": float, "t_end": float,
    "label": str, "confidence": float,
}

# B4 output — attached to each Segment
Signature = {
    "mag_mean": float, "mag_std": float,
    "dom_freq_hz": float, "dom_freq_power": float,
    "cadence_bpm": float | None,
    "jerk_energy": float, "gyro_rms": float,
    "periodicity": float,       # autocorrelation peak strength
    "orientation_stability": float,
}

# B5 storage — Activity Timeline row
TimelineRow = {**Segment, "signature": Signature, "raw_ptr": tuple[int,int]}

# B6 output — Daily Rollup row
DailyRollup = {
    "user_id": str, "date": "YYYY-MM-DD", "activity": str,
    "total_duration_s": float, "num_bouts": int,
    "avg_bout_len_s": float, "first_onset": str, "last_onset": str,
}
```

Keep every block's function pure (input struct → output struct, no hidden global state). That's what lets Task 3/4 grounding trace an answer all the way back to `raw_ptr` without guessing.

## 3. Rolling trend design (B6 → B7)

Compute once per user per day, incrementally (append, don't recompute history):

- **7-day rolling avg**: `daily_rollup.groupby("activity")["total_duration_s"].rolling(7, min_periods=3).mean()`
- **Monthly rolling avg**: resample daily rollups to calendar month (`resample("MS").sum()` then divide by days-with-data, not calendar days, so missing-data days don't silently drag the average down) or a 30-day rolling window if you want a moving month instead of calendar-aligned.
- **Yearly**: same pattern at 365-day / annual resample. With only a few weeks of hackathon data this will mostly be illustrative — say so in the report rather than presenting it as validated.

Always report the number of days *with actual data* behind each average alongside the average itself — an "average" over 2 real days out of a 7-day window is not the same claim as one over 7.

## 4. Build-order prompts

Use these as standalone prompts (to an AI coding assistant, or as your own task tickets). Each is scoped to exactly one block, states its contract, and can be built/tested before the next block exists — feed it synthetic data matching the contract above if the upstream block isn't ready yet.

---

### Prompt — Block 1: Preprocessing
> Write a Python module `preprocess.py` that takes a raw accelerometer+gyroscope CSV (columns: timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, irregular sampling) and returns the `PreprocessedSignal` struct below: [paste struct]. Requirements: resample both streams to 25Hz via linear interpolation; only interpolate gaps under 2 seconds, longer gaps must be recorded in `gaps` and left as NaN, not silently filled; apply a light 2nd-order Butterworth low-pass filter (cutoff ~10Hz) to produce the filtered arrays but also keep the untouched raw arrays for later jerk/noise features; do not clip or normalize amplitude. Include a unit test with synthetic sine-wave input plus an injected 3-second dropout.

---

### Prompt — Block 2: Window Recognition
> Write `recognize.py` implementing a window classifier over a `PreprocessedSignal`. Slide a 2-second window with 50% overlap over the filtered acc+gyro arrays, extract a compact feature vector per window (mean/std/min/max per axis, magnitude mean/std, dominant FFT frequency and power, zero-crossing rate), and classify into the 7 activity classes using [gradient-boosted trees / light 1D-CNN — pick one and justify in a docstring]. Output a list of `Window` structs with per-class probabilities, not just argmax labels — downstream smoothing needs the full distribution. Include a stub `train(features, labels)` and `predict(features)` so the model is swappable.

---

### Prompt — Block 3: Segmentation
> Write `segment.py` that takes a list of `Window` structs and produces `Segment` structs. Step 1: apply a median filter (kernel=5) over the argmax window labels to suppress single-window flicker. Step 2: merge consecutive same-label windows into segments, carrying the mean confidence. Step 3: for each segment boundary, run a local change-point refinement (CUSUM on the accel-magnitude signal ±3 seconds around the naive boundary) to snap `t_start`/`t_end` to the actual transition rather than the window grid. Return segments sorted by time with no overlaps and no gaps except where `PreprocessedSignal.gaps` indicates missing data — those spans must not be assigned any label.

---

### Prompt — Block 4: Signature Extraction
> Write `signature.py` implementing `extract_signature(segment, preprocessed_signal) -> Signature` per the struct above. Compute magnitude mean/std from raw (unfiltered) acceleration for `jerk_energy`, but dominant frequency and periodicity from the filtered signal via FFT and autocorrelation respectively. `cadence_bpm` should be None for non-periodic segments (e.g. lying, sitting) and a peak-counted step/pedal rate for walking/running/bicycling. `orientation_stability` = inverse variance of the low-pass-filtered gravity-direction estimate over the segment. This function must not depend on the segment's `label` — it has to work identically on an unlabeled or novel segment, since Task 4 queries will call it on activity the classifier may have gotten wrong or that has no fixed label.

---

### Prompt — Block 5: Storage Layer
> Write `store.py` wrapping an SQLite (or simple parquet) file with two tables: `timeline` (one row per `TimelineRow`, indexed by user_id + t_start) and a raw-signal store that memory-maps the original arrays and exposes `get_raw(user_id, t_start, t_end) -> (acc, gyro)` by index lookup, never by copying. Provide `insert_segments(user_id, segments, signatures)`, `query_timeline(user_id, t_start, t_end, activity=None)`, and `get_segment_raw(segment_id)` that resolves `raw_ptr` back to actual signal for evidence citation.

---

### Prompt — Block 6+7: Daily Rollup & Rolling Trends
> Write `rollup.py`. `compute_daily_rollup(user_id, date, timeline_rows) -> list[DailyRollup]` groups a day's segments by activity and computes total duration, bout count, avg bout length, first/last onset. `compute_rolling(daily_rollups_df, activity, window)` returns a rolling mean over `total_duration_s` for `window in {"7d","monthly","yearly"}`, using calendar-aware pandas resampling for monthly/yearly and a plain rolling window for 7d, and must also return `n_days_with_data` alongside each average value — never return an average without it.

---

### Prompt — Block 8: Query Router
> Write `router.py` with `route(question: str) -> dict` that classifies a natural-language question into one of: Task1 (identification/verification), Task2 (duration/count/comparison), Task3 (grounding/onset), Task4 (open-world), or Personalization (trend/baseline comparison), and extracts the relevant slots (activity name, time range, comparison target) needed by the downstream handler. Use a small rule-based intent classifier first (keyword + regex patterns for "how long", "did she", "compared to", "increasing", etc.) with a fallback to a small instruction-tuned SLM only when rules don't match confidently — log which path was used, since your report needs to justify the design choice.

---

### Prompt — Block 9: Exemplar Bank & Open-World Reasoning
> Write `exemplars.py`. Offline: for each of the 7 classes, select 3–5 representative training segments and store their `Signature` vectors with a short human-written description of what that signature typically reflects (e.g. "smooth cyclic acceleration, steady low-frequency peak, no discrete impact spikes"). At query time: `nearest_exemplars(query_signature, k=3) -> list[(exemplar, distance, description)]` via simple Euclidean or cosine distance in signature space. Then `explain(query_signature, nearest_exemplars, slm)` prompts the SLM to compare the query's actual numeric signature values against the retrieved exemplar's, and write an explanation that cites specific numbers — the SLM must not be given free rein to describe the activity from general knowledge.

---

### Prompt — Block 10: Output Formatter
> Write `formatter.py` with `format_answer(answer, activity, timestamps, modality, channels, explanation) -> str` that renders the exact fixed template from the brief (Answer / Activity-Event / Evidence: Timestamp(s), Sensor Modality, Sensor Channel(s), Explanation), substituting "N/A" for any missing field, and validates that every timestamp passed in actually resolves to a real `raw_ptr` in the store before rendering — refuse to render (raise, don't silently pass through) if a timestamp doesn't trace back to real data.

---

### Prompt — Block 11: Streamlit UI
> Build `app.py` (see attached skeleton) with three tabs: Timeline upload+view, Ask-a-Question (free text → routed answer in the fixed template), and Trends (7-day/monthly/yearly rolling charts per activity, each labeled with days-of-data-behind-the-average). Wire it to the block functions above via clean imports, no logic duplicated in the UI layer itself.

## 5. Suggested build order for a 3-person group

1. **Person A**: B1 → B2 → B3 (signal side)
2. **Person B**: B4 → B5 → B9 (feature/storage/exemplar side)
3. **Person C**: B6/B7 → B8 → B10 → B11 (product/personalization/UI side)

Integrate at two checkpoints: once B1–B5 produce a real Activity Timeline from sample data (mid-challenge), and once B8–B11 can answer a hand-written question of each task type against it (end).
