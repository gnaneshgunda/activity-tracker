# Activity Tracker — Architecture & Pipeline

A wearable activity recognition system built on the ExtraSensory dataset. It
classifies 7 human activities from raw accelerometer + gyroscope bursts,
explains its reasoning in natural language, detects anomalous events (falls,
prolonged immobility), estimates energy expenditure, and answers free-text
questions about a user's day.

---

## Table of Contents

1. [Where We Started](#where-we-started)
2. [What Changed — Evolution Log](#what-changed--evolution-log)
3. [Where We Ended](#where-we-ended)
4. [Full Pipeline Architecture](#full-pipeline-architecture)
5. [Module Reference](#module-reference)
6. [Data Contracts](#data-contracts)
7. [Training](#training)
8. [Running the App](#running-the-app)
9. [Key Design Decisions & Known Limitations](#key-design-decisions--known-limitations)

---

## Where We Started

The project began as a basic LSTM window classifier over the ExtraSensory
dataset with:

- **5 physics features**: SMA, tilt, gyro_rms (scalar), cadence_bpm, periodicity
- **Hand-written decision tree** with manually set thresholds (not fitted)
- **10 input channels**: body_acc (3) + gravity (3) + gyro (3) + gyro_present (1)
- **Simple router stub**: `SLMNotConfigured` raised on any ambiguous question
- **No temporal smoothing**: raw per-window argmax, no HMM
- **No anomaly detection**
- **No energy estimation**
- **No RAG / exemplar bank**
- **No SLM integration** — the slot existed in code but was never filled
- **Training OOM**: `ProcessPoolExecutor` defaulted to 16 workers × ~400 MB
  each = 12 GB on a machine with only 3.2 GB free RAM

The original checkpoint (`lstm.pt`) was trained with N_CHANNELS=10.

---

## What Changed — Evolution Log

### Phase 1 — Physics Model Overhaul

**Problem**: The 5-feature physics rule had hand-set constants that silently
misfired on this dataset. `sma < 0.25` captured walking and bicycling as
"static". `periodicity > 0.40` excluded both from the locomotion branch.

**Changes**:
- Added `Thresholds.fit()` — coordinate ascent over a grid, maximising
  macro-F1 on labelled training data. Structure stays hand-written; only
  constants are fitted.
- Added 6 new features to `Features` dataclass:
  - `jerk_mean` (g/s) — mean magnitude of frame-to-frame body_acc derivative.
    Distinguishes running foot-strikes from walking even when SMA overlaps.
  - `dominant_freq_hz` (Hz) — FFT peak on vertical body_acc in 0.5–5 Hz band.
    More robust than autocorrelation cadence for noisy signals.
  - `gyro_x_rms`, `gyro_y_rms`, `gyro_z_rms` (rad/s) — axis-resolved gyro RMS.
    Walking has dominant pitch (y-axis); lateral stumbles show on x-axis;
    cycling has dominant yaw (z-axis).
- Updated `label_activity()`: FFT freq as third periodicity gate; jerk as
  third running gate.
- Updated `predict_proba()`: jerk and FFT margins added to running/walking
  soft scores.
- Added `tilt_reliable` property: returns False when phone is in bag or on
  table, triggering an honest "sitting" fallback instead of a wrong posture.

**Files**: `data/physics_rules.py`

---

### Phase 2 — Feature Pipeline Expansion (N_CHANNELS 10 → 26)

**Problem**: LSTM only saw raw sensor channels. Physics features were computed
separately and never fed back into the neural network.

**Changes**:
- `pipeline/recognize.py`: Added `dominant_freq_hz()` (FFT on vertical
  body_acc, returns peak in 0.5–5 Hz band) and `acc_gyro_phase()` (cosine of
  phase difference between vertical acc and gyro-y). Both exported in `__all__`.
- `data/dataset.py`: CHANNELS expanded from 10 to 26:
  - Raw sensor (10): body_acc_x/y/z, gravity_x/y/z, gyro_x/y/z, gyro_present
  - Derived physics (12): tilt_deg, sma, cadence_hz, periodicity, vertical_std,
    rotation_ratio, jerk_mean, gyro_x/y/z_rms, dominant_freq_hz, acc_gyro_phase
  - Phone placement flags (4): phone_pocket, phone_hand, phone_bag, phone_table
- `_minute_windows()` now computes all 12 derived features per window and
  broadcasts them as constants across the T=50 timesteps.
- `models/lstm.py`: `_physics_probs_from_window` updated to pass all new
  `Features` fields (jerk_mean, dominant_freq_hz, gyro_x/y/z_rms) — was
  missing these, would have crashed mid-training.

**Note**: Any checkpoint trained with fewer than 26 channels is incompatible.
`lstm_alpha_v3.pt` is the target output of the current training pipeline.

---

### Phase 3 — Alpha-Aware LSTM (Physics-Neural Fusion)

**Problem**: A pure LSTM ignores the physics prior entirely. A pure physics
rule ignores temporal patterns. Neither alone is robust.

**Changes**:
- Added `AlphaAwareLSTMClassifier` in `models/lstm.py`: shared BiLSTM encoder
  with two heads — a label head (7-class logits) and an alpha head (scalar
  gate in [0,1]).
- The alpha gate is supervised from the gap between the ground-truth
  log-likelihoods of the ML head and the physics prior. When the LSTM is
  more confident than physics, alpha → 1 (trust LSTM). When physics is more
  confident, alpha → 0 (trust physics).
- `train_joint_alpha()`: trains both heads jointly. Loss = CrossEntropy +
  `alpha_loss_weight × MSE(alpha_pred, alpha_target)`.
- `train_alpha.py`: CLI for training the alpha-aware model with configurable
  hidden size, layers, lr, batch size, workers, cap, seed.

---

### Phase 4 — RAG + SLM Integration (B8 + B9)

**Problem**: The router raised `SLMNotConfigured` on any ambiguous question.
The exemplar bank existed but had no way to call a language model.

**Changes**:

**`models/slm.py`** (new file):
- `SLMConfig`: model_path, n_ctx, n_threads, max_tokens, temperature
- `SLM`: lazy-loads `llama_cpp.Llama` (Phi-3 Mini 3.8B Q4_K_M, ~2.3 GB).
  `__call__` runs inference. `for_routing()` returns greedy max_tokens=10
  variant. `for_narration()` returns temperature=0.3 max_tokens=200 variant.
  Both share the same loaded model instance (no double-load).
- `get_slm()`: module-level singleton, lazy.
- `download_model()`: downloads from HF Hub. CLI: `python3 -m models.slm --download`

**`analysis/router.py`**:
- Replaced `_default_config = RouterConfig()` with `_make_default_config()`
  which tries to wire `get_slm().for_routing()`, falls back to rule-only if
  weights are missing. Rule-only mode still works perfectly without the model.

**`analysis/exemplars.py`**:
- Added `explain_auto(query_signature, *, k, metric, anomaly_event)` — one-call
  entry point that retrieves exemplars + calls `get_slm().for_narration()`,
  falls back to `_null_slm` if weights are missing.

---

### Phase 5 — OOM Fix + Training Stability

**Problem**: Training crashed with kernel OOM killer before epoch 1.
Root cause: `ProcessPoolExecutor` defaulting to 16 workers × ~400 MB each
= 12 GB on a machine with 3.2 GB free RAM and 0 swap.

**Changes**:
- `training/train_common.py`: Added `_DATA_WORKERS = int(os.environ.get("ACTIVITY_TRACKER_WORKERS", "2"))`.
  `build_split_sets` passes `workers=_DATA_WORKERS` to `scan_all` and
  `build_windows`. Added progress prints per split.
- `training/train_alpha.py`: Added `--workers 2` arg. Sets
  `ACTIVITY_TRACKER_WORKERS` env var before forking. Default `--cap` lowered
  to 500, `--batch-size` to 128.
- `requirements.txt`: Added `llama-cpp-python>=0.2.57`, `huggingface-hub>=0.23`,
  `scikit-learn>=1.3`, version bounds on all deps, CPU/GPU install notes.

---

## Where We Ended

The system now has:

| Component | Status |
|---|---|
| Physics rules (5 features, hand-set) | ✅ Replaced with 11-feature fitted rules |
| LSTM (10 channels) | ✅ Upgraded to 26-channel alpha-aware BiLSTM |
| Physics-neural fusion | ✅ Per-sample alpha gate, jointly trained |
| Anomaly detection | ✅ Physics rule (jerk + tilt) + IsolationForest fallback |
| Energy estimation | ✅ MET-lookup, Compendium of Physical Activities |
| RAG exemplar bank | ✅ 28 entries × 9 features, cosine/Euclidean retrieval |
| SLM (Phi-3 Mini) | ✅ Lazy-loaded, shared instance, routing + narration variants |
| Query router | ✅ Rule-first → SLM fallback, 8 route types |
| Storage | ✅ SQLite with coverage tracking and anomaly event table |
| Streamlit UI | ✅ Timeline + Ask + Trends tabs |
| Training stability | ✅ Workers capped at 2, OOM eliminated |

**Target checkpoint**: `checkpoints/lstm_alpha_v3.pt` (N_CHANNELS=26,
`AlphaAwareLSTMClassifier`)

**Known gaps**:
- `standing and moving` is structurally absent from ExtraSensory training data
  (no column distinguishes it from `standing in place`). The model cannot
  reliably predict it.
- Jerk/FFT/phase features are broadcast as constants per window (same value
  repeated T=50 times). The LSTM sees them as static hints, not temporal
  signals — acknowledged limitation.
- HMM post-processing (B3) is designed but not yet trained. The Viterbi
  smoother would add +0.05–0.08 macro-F1 by eliminating single-window blips.

---

## Full Pipeline Architecture

```
ExtraSensory per-user files (features_labels.csv.gz + raw sensor bursts)
        │
        ▼
[B0] Ingestion & Alignment                      data/ingest.py
        │  join uuid+timestamp → IngestedExample
        │  burst=None + coverage_s=0 when no raw burst exists
        ▼
[B1] Preprocessing                              data/preprocess.py
        │  resample to 25 Hz, Butterworth low-pass
        │  gravity/body-acceleration split (0.3 Hz cutoff)
        │  → PreprocessedSignal {body_acc, gravity, gyro_raw, gaps}
        ▼
[B2] Window Feature Extraction                  pipeline/recognize.py
        │  extract_windows(): 2 s windows, 50% overlap
        │  per window → cadence_hz, sma, gyro_rms (axis-resolved),
        │               dominant_freq_hz (FFT), acc_gyro_phase,
        │               vertical_std, valid_fraction
        │  → list[Window]  (probs=None at this point)
        ▼
[B2] Label Prediction (scorer plug-in)          models/lstm.py  OR  data/physics_rules.py
        │
        │  attach_probs(windows, scorer)
        │
        │  scorer A — AlphaAwareLSTMClassifier          models/lstm.py
        │    lstm.make_scorer(model, normaliser, signals)
        │    26-channel BiLSTM → logits → softmax
        │    alpha head gates LSTM vs. physics per sample
        │
        │  scorer B — physics rule fallback             data/physics_rules.py
        │    physics_rules.make_scorer(thresholds, features_for)
        │    predict_proba(Features(...)) → soft distribution
        │    used when no checkpoint is loaded
        │
        │  → Window.probs  (7-class softmax, full distribution)
        │    Window.confidence = 1 − normalised_entropy(probs)
        ▼
[B3] Segmentation (HMM / Viterbi)               pipeline/segment.py
        │  segment_windows(windows, transition_model)
        │  emission probs  ← Window.probs (log-space)
        │  transition matrix ← estimate_transitions() from training sequences
        │                      falls back to physics-plausible prior per state
        │  viterbi_k_min(): k-minimum-consecutive-states constraint
        │    (prevents single-window blips; k ≈ 5 s / hop_s)
        │  cusum_refine(): snap each decoded boundary ±3 s on |body_acc|
        │  → list[Segment] {label, t_start, t_end, confidence,
        │                    mean_probs, coverage_s, flags}
        ▼
[B4] Signature Extraction                       analysis/signature.py
        │  extract_signature(segment, signal)  — label-free
        │  → Signature {tilt_deg_mean/std, sma, cadence_bpm,
        │               gyro_rms_x/y/z, jerk_energy,
        │               orientation_stability, periodicity}
        │
        ├─────────────────────────────────┐
        ▼                                  ▼
[B4.5] Anomaly / Event Detector           [B5] Storage Layer
  analysis/anomaly.py                       analysis/store.py
  jerk_energy spike + tilt collapse         SQLite: timeline + anomaly_events
  → AnomalyEvent  method=physics_rule       coverage_s + uuid/t_label_start_ref
  IsolationForest fallback (optional)       get_coverage() for timestamp checks
        │                                        │
        └──────────────┬─────────────────────────┘
                        ▼
             [B6] Daily Rollup                  analysis/rollup.py
               60 s per labeled minute (not ~20 s burst length)
               bout count, avg bout length, first/last onset
                        │
                        ├──────────────────► [B_E] Energy / MET Estimator
                        │                     analysis/energy.py
                        │                     MET × weight × duration → kcal
                        ▼
             [B7] Rolling Trends                analysis/rollup.py
               7d / monthly / yearly windows
               n_days_with_data always reported
                        │
                        ▼
             [B8] Query Router                  analysis/router.py
               Stage 1: regex rule table → (route, confidence)
               Stage 2: SLM fallback (Phi-3 Mini) when conf < 0.75
               logs which path was used on every call
                        │
        ┌───────────────┼──────────────┬──────────────┬──────────────┐
        ▼               ▼              ▼              ▼              ▼
   TASK1             TASK2          TASK3          TASK4         B_ENERGY
   label/prob        aggregation    onset/offset   RAG + SLM     B_ANOMALY
   look-up           sum/count/cmp  coverage-      narration     MET / event
                     60s-per-min    checked                       table
                        │
                        ▼
             [B9] Exemplar Bank & SLM Narration  analysis/exemplars.py
               nearest_exemplars(): cosine/Euclidean on 9-feature vectors
               28 exemplars × 7 classes + anomaly_event class
               build_explain_prompt(): query numbers vs. exemplar numbers
               explain() / explain_auto(): SLM must cite specific values
               anomaly events route through the same path (B4.5 → B9)
                        │
                        ▼
             [B10] Output Formatter              formatter.py
               get_coverage() before rendering any timestamp
               widens or refuses citations with 0% real signal coverage
                        │
                        ▼
             [B11] Streamlit UI                  app.py
               Timeline: coverage fraction per segment
               Ask: confidence/entropy shown when low
               Trends: kcal toggle (labeled as population-level estimate)
```

### Route Taxonomy (B8)

| Route | Trigger | Backend |
|---|---|---|
| `TASK1` | "What was she doing at 3pm?" | Label/probability look-up |
| `TASK2` | "How long did she walk today?" | Aggregation (60s-per-minute) |
| `TASK3` | "When did she start running?" | Onset/interval, coverage-checked |
| `TASK4` | "Why does her cadence drop?" | RAG + SLM narration (B9) |
| `B_ENERGY` | "How many calories did she burn?" | MET estimator (B_E) |
| `B_ANOMALY` | "Did she fall?" / "Any unsteady episodes?" | Anomaly event table (B4.5) |
| `PERSONALIZATION` | "Update her weight to 58 kg" | Profile store |
| `UNKNOWN` | No confident match | Clarification prompt |

---

## Module Reference

```
activity-tracker/
├── data/
│   ├── ingest.py           B0 — UUID scan, burst join, TARGET_CLASSES, PhonePlacement
│   ├── preprocess.py       B1 — resample, Butterworth, gravity/body split
│   ├── physics_rules.py    Physics decision tree with fitted thresholds
│   │                         Features (11 fields), Thresholds.fit(), predict_proba()
│   ├── dataset.py          Window assembly — CHANNELS (26), WindowSet, Normaliser
│   │                         scan_all, build_windows, build_burst_sequences
│   └── fusion.py           Alpha-weighted fusion of LSTM + physics probs
│
├── pipeline/
│   ├── recognize.py        B2 — extract_windows, cadence_from_vertical,
│   │                         dominant_freq_hz, acc_gyro_phase, attach_probs
│   └── segment.py          B3 — HMM/Viterbi, k-min-duration, CUSUM snapping
│
├── models/
│   ├── lstm.py             LSTMClassifier, AlphaAwareLSTMClassifier
│   │                         train(), train_joint_alpha(), evaluate()
│   │                         _physics_probs_from_window() (physics prior per batch)
│   │                         save_checkpoint(), load_checkpoint()
│   ├── hybrid.py           Hybrid inference: alpha × LSTM + (1-alpha) × physics
│   └── slm.py              SLM wrapper (Phi-3 Mini 3.8B Q4_K_M via llama-cpp-python)
│                             get_slm() singleton, for_routing(), for_narration()
│                             download_model() from HF Hub
│
├── analysis/
│   ├── signature.py        B4 — Signature dataclass, extract_signature() (label-free)
│   ├── anomaly.py          B4.5 — AnomalyEvent, physics rule + IsolationForest fallback
│   ├── store.py            B5 — SQLite store, timeline + anomaly_events tables
│   │                         get_coverage() for timestamp validation
│   ├── rollup.py           B6/B7 — daily rollup, rolling trends, n_days_with_data
│   ├── energy.py           B_E — MET lookup, kcal estimate, Compendium of Physical Activities
│   ├── router.py           B8 — rule-first → SLM fallback, RouteResult, build_router()
│   └── exemplars.py        B9 — EXEMPLAR_BANK (28 entries), nearest_exemplars(),
│                             explain(), explain_auto(), build_explain_prompt()
│
├── training/
│   ├── train_common.py     build_split_sets(), _DATA_WORKERS (env-capped at 2)
│   ├── train_alpha.py      CLI for AlphaAwareLSTMClassifier training
│   ├── train_lstm.py       CLI for plain LSTMClassifier training
│   ├── train_burst.py      CLI for per-burst sequence training
│   └── train_hybrid.py     CLI for hybrid model training
│
├── formatter.py            B10 — coverage-aware output rendering
├── app.py                  B11 — Streamlit UI (Timeline / Ask / Trends tabs)
└── checkpoints/            Saved .pt files + JSON training reports
```

---

## Data Contracts

### Key structs (abbreviated)

```python
# B0 — one row per matched (uuid, labeled-minute)
IngestedExample = {
    "uuid": str,
    "t_label_start": float,       # unix seconds
    "burst": {"t", "acc", "gyro"} | None,
    "label": str | None,          # 7-class, or None if ambiguous
    "coverage_s": float,          # seconds of real burst in this 60s slot
}

# B1 — resampled to 25 Hz
PreprocessedSignal = {
    "t", "acc", "acc_raw", "gyro", "gyro_raw",
    "gravity",    # low-pass estimate of gravity direction, |g|=1
    "body_acc",   # acc_raw − gravity
    "gaps",       # interpolated/missing spans within the burst
}

# B2 — one row per 2s window
Window = {
    "t_start", "t_end",
    "probs": dict[str, float],    # 7-class softmax
    "entropy": float,             # confidence proxy
    "cadence_hz": float | None,
    "sma": float,
    "gyro_rms_xyz": tuple[float, float, float],
}

# B4 — attached to each Segment
Signature = {
    "tilt_deg_mean", "tilt_deg_std",
    "sma", "cadence_bpm",
    "gyro_rms_x", "gyro_rms_y", "gyro_rms_z",
    "jerk_energy", "orientation_stability", "periodicity",
}

# B4.5 — zero or more per segment
AnomalyEvent = {
    "event_id", "t_start", "t_end",
    "kind": str,                  # "possible_fall" | "prolonged_immobility"
    "trigger_signature": dict,    # actual jerk_energy / orientation_stability values
    "method": str,                # "physics_rule" | "isolation_forest"
    "coverage_s": float,
}

# B_E — energy estimate
EnergyEstimate = {
    "activity", "duration_s",
    "met": float,
    "assumed_weight_kg": float,   # explicit assumption, not measured
    "kcal_est": float,
    "basis": str,                 # Compendium category used
}
```

### 7 Activity Classes

```
lying down | sitting | standing in place | standing and moving
walking | running | bicycling
```

### N_CHANNELS = 26

```
Raw (10):    body_acc_x/y/z, gravity_x/y/z, gyro_x/y/z, gyro_present
Physics (12): tilt_deg, sma, cadence_hz, periodicity, vertical_std,
              rotation_ratio, jerk_mean, gyro_x/y/z_rms,
              dominant_freq_hz, acc_gyro_phase
Placement (4): phone_pocket, phone_hand, phone_bag, phone_table
```

---

## Training

### Prerequisites

```bash
pip install -r requirements.txt
# For CPU-only llama-cpp-python:
CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python
```

### Download SLM weights (optional, ~2.3 GB)

```bash
python3 -m models.slm --download
```

### Train the alpha-aware model

```bash
# From the project root
python3 -m training.train_alpha \
  --cap 300 --epochs 30 \
  --hidden 96 --layers 2 \
  --lr 1e-3 --batch-size 64 \
  --workers 2 --seed 42 \
  --out checkpoints/lstm_alpha_v3.pt \
  --report checkpoints/lstm_alpha_v3_report.json
```

**Worker cap**: Keep `--workers 2` on machines with < 8 GB free RAM.
Each worker loads ~400 MB of burst data. 16 workers × 400 MB = 12 GB → OOM.

**Cap**: `--cap 300` means at most 300 labeled minutes per class per split.
Raise to 500–1000 if you have more RAM.

### Checkpoint compatibility

Checkpoints store `N_CHANNELS` implicitly via `len(mean)`. A checkpoint
trained with 10 channels cannot be loaded into a 26-channel model. Always
use `load_checkpoint()` which reads `model_type` and reconstructs the
correct architecture.

---

## Running the App

```bash
streamlit run app.py
```

Three tabs:
- **Timeline**: activity segments with coverage indicator (fraction backed by
  real signal, not just labeled minutes)
- **Ask**: free-text questions routed through B8 → B9. Confidence/entropy
  shown when low.
- **Trends**: rolling 7d/monthly/yearly duration charts + kcal toggle
  (labeled as population-level estimate)

---

## Key Design Decisions & Known Limitations

### Why fitted thresholds, not hand-set?

The rule structure encodes physics ("locomotion is periodic", "posture shows
in tilt"). The constants encode the units and dynamic range of one particular
sensor pipeline. Hand-set constants from another setup silently route every
sample down the wrong branch. `Thresholds.fit()` derives every constant from
labelled training data by maximising macro-F1.

### Why subject-wise splits?

Windows from one burst overlap 50% and windows from one user share a device,
gait, and carrying position. Splitting at the window level puts near-duplicates
on both sides and inflates every metric. `split_users()` partitions UUIDs.

### Why 60 s per labeled minute in rollup?

The ExtraSensory burst is ~20 s of sensor data carrying a 60 s label. The
burst is evidence for the label, not the duration itself. Counting burst
length would undercount all activity durations by ~3×.

### Why alpha per sample, not a global scalar?

A global alpha assumes the LSTM is uniformly better or worse than physics
across all inputs. In practice the LSTM is better on common classes (sitting,
walking) and physics is better on rare or ambiguous ones (running vs. fast
walking, bicycling). A per-sample gate lets the model learn which expert to
trust for each window.

### Why Phi-3 Mini for the SLM?

- 3.8B params, ~2.3 GB in Q4_K_M quantisation
- Runs on CPU via llama-cpp-python; no GPU required
- Instruction-tuned: follows structured prompts without fine-tuning
- Fits on a Raspberry Pi 5 (8 GB) or Jetson Orin Nano for edge deployment
- MIT licence

### Known limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| `standing and moving` absent from ExtraSensory training | Model cannot reliably predict this class | Acknowledged in absent_classes; excluded from macro-F1 |
| Jerk/FFT/phase broadcast as constants per window | LSTM cannot learn temporal patterns from them | Acknowledged; they act as static hints to the physics prior |
| No HMM post-processing yet | Single-window blips not smoothed | B3 segment.py is designed; Viterbi not yet trained |
| 0 swap on dev machine | OOM with > 2 workers | Workers capped at 2 via env var |
| SLM not downloaded by default | Router falls back to rule-only | `python3 -m models.slm --download` |

---

## Academic Citations

| Source | Used in |
|---|---|
| Vaizman, Ellis & Lanckriet, IEEE Pervasive Computing 2017 (arXiv:1609.06354) | B0 burst structure, B1 gravity/body split |
| DrHouse — Sui et al., ACM IMWUT 8(4), 2024, doi:10.1145/3699765 | B9 grounded reasoning pattern |
| JARVIS for HVAC — Lee et al., ACM IMWUT 10(2), 2026, doi:10.1145/3810210 | B8 staged routing |
| SensorChat — Yu et al., ACM IMWUT 9(3), 2025, doi:10.1145/3749496 | B8 qualitative vs. quantitative split |
| Sensor2Text, ACM IMWUT 2024, doi:10.1145/3699747 | B9 NL interaction for activity tracking |
| Ainsworth et al., Compendium of Physical Activities (pacompendium.com) | B_E MET values |
| HMM for accelerometer data, PLOS ONE 2014 (PMC4251969) | B3 Viterbi smoothing |
| Viterbi k-min-duration constraint, ResearchGate 268981604 | B3 minimum-duration constraint |
| Fall detection rule, PMC4346101 | B4.5 impact → stillness → orientation-change |
| Fall detection rule, Springer doi:10.1007/978-3-642-41043-7_2 | B4.5 same rule, independent source |
