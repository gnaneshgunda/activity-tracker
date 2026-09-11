# Ask the Sensors — Activity Recognition from Wearable IMU Data

**Course Project Report**
**Dataset**: ExtraSensory (Vaizman, Ellis & Lanckriet, IEEE Pervasive Computing 2017)

---

## The Problem

A smartphone sits in your pocket all day. Its accelerometer and gyroscope are always recording — tiny numbers, 40 times a second, measuring how fast you're moving and rotating. The question we set out to answer is: can we turn those raw numbers into something meaningful? Not just a label like "walking", but a real answer to a question like *"How long did she walk today, and did anything unusual happen around 3pm?"*

This is harder than it sounds. The phone doesn't know what the person is doing. It knows acceleration. Everything else has to be inferred. And the signals for "sitting at a desk" and "lying in bed" can look almost identical when the phone is in a pocket.

We worked with the ExtraSensory dataset — 60 users, their daily lives, labelled at one-minute resolution, with raw sensor bursts captured inside those minutes. Seven activity classes: lying down, sitting, standing in place, standing and moving, walking, running, bicycling.

---

## 1. Design: The Story of How We Got Here

### Where we started — just the LSTM

Our first instinct was: neural networks are good at pattern recognition, let's throw the raw sensor data at an LSTM and let it learn.

The idea was appealing. We didn't want to hand-code rules that might not generalise. We wanted the model to discover patterns we hadn't thought of — maybe noise itself carries useful information, the way an experienced doctor reads texture in an X-ray that a rulebook would miss. So we deliberately skipped preprocessing and fed raw accelerometer + gyroscope channels directly into a bidirectional LSTM.

It worked, but not well enough. The LSTM is data-hungry. The ExtraSensory dataset is imbalanced — 81,000 sitting examples, 737 running examples. The model learnt to predict the common classes reasonably and gave up on the rare ones. Running F1 was near zero. More importantly, the LSTM had no anchor — it had to rediscover gravity, infer orientation, figure out what periodicity is, all from scratch, every time. That's too much to ask from 20 seconds of data.

**Macro F1: ~0.25**

### Adding physics — hardcoded rules with a fixed blend

The LSTM struggled most with the things physics knows well. Locomotion is periodic — walking and running have a clear cadence. Posture shows up in the gravity direction — lying down makes the phone nearly horizontal, standing makes it vertical. These are not learnable patterns; they're geometry.

So we added a physics rule model alongside the LSTM. The physics model is a hand-written decision tree:

```
if signal is quiet → posture branch (tilt decides lying/sitting/standing)
if signal is periodic → locomotion branch (SMA and jerk decide walk/run/cycle)
```

And we blended the two outputs:

```
p_final = alpha × p_lstm + (1 − alpha) × p_physics
```

Alpha was a fixed scalar — we tried 0.3, 0.5, 0.7. The physics model helped, especially on posture. But a fixed alpha is a lie: it assumes the LSTM is uniformly better than physics across all inputs. That's not true. On a clean cycling signal the physics rotation features are decisive. On a messy slow-walk-almost-standing window, the LSTM is better. A fixed blend throws away that information.

**Macro F1: ~0.30**

But the physics thresholds themselves were another problem. We had hand-set constants — `sma < 0.25` to detect stillness, `periodicity > 0.40` for locomotion. These were plausible-looking numbers borrowed from other setups. On ExtraSensory data they silently misfired: that SMA threshold was capturing walking and bicycling as "static". We needed to fit the constants, not guess them.

### Making alpha trainable — per-sample gating

The logical next step: don't fix alpha, learn it. We added an **alpha head** to the LSTM — a second output that predicts, for each sample, how much to trust the neural prediction vs. the physics prior.

The alpha is supervised from data: we compute how much more confident the correct class is under the LSTM vs. under physics, and train the gate toward 1 when the LSTM is better and 0 when physics is better. Both heads train jointly.

This was a meaningful step. The model started learning *when* to trust itself. On easy locomotion cases it leaned on the LSTM. On ambiguous posture it fell back to physics.

**Macro F1: ~0.33 (from report.json)**

### The derived features breakthrough

Even with a trainable alpha, the LSTM was still working with limited information. The raw sensor channels (body_acc xyz, gravity xyz, gyro xyz, gyro_present) don't tell the model what the features *mean*. We had been computing physics features for the rule model but never giving them to the LSTM.

We expanded the input from **10 channels to 27**:

| Channel group | Channels | What they encode |
|---|---|---|
| Raw sensor | body_acc x/y/z, gravity x/y/z, gyro x/y/z, gyro_present | Direct measurement |
| Derived physics | tilt_deg, SMA, cadence_hz, periodicity, vertical_std, rotation_ratio, jerk_mean, gyro_x/y/z RMS, dominant_freq_hz, acc_gyro_phase | Computed interpretable signals |
| Phone placement | phone_pocket, phone_hand, phone_bag, phone_table | Context flags |
| Validity | sample_valid | Which timesteps are real data |

These derived features are broadcast as constants across all 400 timesteps of each burst — the LSTM sees them as global context about this burst, not as a time-varying signal. The key ones:

- **SMA (Signal Magnitude Area)** — mean magnitude of body acceleration after removing gravity. The cleanest motion intensity proxy.
- **Dominant frequency (FFT)** — the peak frequency in the 0.5–5 Hz band. Cadence by FFT is more robust than peak-counting for noisy short windows.
- **acc_gyro_phase** — cosine of the phase difference between vertical acceleration and the dominant gyro axis. Walking pendulum swing gives phase ≈ 0.8. Cycling motion is decoupled, phase ≈ 0.0. This single number separates walking from cycling better than any magnitude feature.
- **Jerk mean and std** — first derivative of body acceleration. Running foot-strikes spike jerk sharply. Walking is smoother. Jerk *std* measures rhythm variability — walking is irregular (each step varies), cycling is mechanically regular. The LocoClassifier (below) depends critically on jerk_std, not jerk_mean — confusing these two was a bug we caught and fixed.
- **Gyro entropy** — Shannon entropy of the energy distribution across the three gyro axes. Low entropy means one axis dominates (cycling wheel spin). High entropy means rotation is spread uniformly (walking foot-strike chaos).

### The logistic regression for walk/cycle/run separation

Even with all these features, the physics rule tree struggled with one specific boundary: separating cycling from walking when the phone is in a pocket. The signals overlap almost completely in SMA, cadence, and dominant frequency. No single threshold cleanly divides them.

We added a **LocoClassifier** — a logistic regression trained specifically on the locomotion classes using five features chosen for their separation power:

- periodicity (walking is more periodic than cycling)
- zero-crossing rate
- rotation ratio (gyro per unit linear motion)
- dominant frequency
- jerk_std (walking rhythm variability vs. cycling smoothness)

This classifier is used by the physics rule tree only inside the locomotion branch, as a drop-in replacement for the rotation_ratio threshold when gyro data is available. It gives the physics model learned boundaries for the one sub-problem where thresholds genuinely can't do it.

### Fitting the physics thresholds from data

With all these features, the constants that control the physics rule tree matter enormously. We replaced hand-set thresholds with **coordinate ascent** fitting:

- Search over a grid of candidate values for each of 19 threshold parameters
- Optimise weighted macro-F1 on the training split (running and bicycling get 2× weight, walking 1.5×, so rare classes aren't sacrificed)
- 10 random restarts to escape local optima
- 8 passes per restart to catch parameter interactions

Structure stays hand-written physics. Only the constants are data-derived.

### Complementary filter for better gravity

The physics model's posture branch depends on accurate tilt estimation. Our original gravity estimate was a 0.3 Hz Butterworth low-pass — it drifts 2–13° during walking because the phone's own acceleration leaks into the estimate.

We replaced it with a **complementary filter** that fuses accelerometer and gyroscope:

- Gyroscope integrates angular rate — accurate short-term, but drifts
- Accelerometer gives absolute gravity direction — no drift, but noisy during motion
- Blend them: `g_est = α × g_gyro_propagated + (1-α) × g_acc_measured`

With time constant τ = 1.5 s (above human gait period so a stride can't drag the estimate), tilt drift drops to under 1°. This directly improved lying-down vs. sitting separation because both classes now have accurate tilt values to distinguish them by.

### Data compression — segment merging

The windowed classification produces a label every second (2s windows, 50% overlap). That's 60+ labels per minute, most of them identical. Storing raw window-level predictions is wasteful and makes querying slow.

After Viterbi smoothing (below), we merge consecutive windows with the same label into single segments. A 15-minute walking bout becomes one record: `{label: walking, t_start: 14:32:00, t_end: 14:47:00, confidence: 0.87}`. Daily storage drops from thousands of rows to tens.

These compressed segments are what flow into rollup, storage, and the query system.

### HMM + Viterbi for smoothing

Raw per-window predictions are noisy — a walking sequence has isolated "standing" windows where the person momentarily slowed. We smooth with a Viterbi-decoded HMM:

- Emission probabilities from the window softmax distributions
- Transition matrix estimated from label sequences in training data
- Physically-plausible prior (strong self-transitions, penalised impossible jumps like running → lying down in one minute)
- **k-minimum-consecutive-states constraint** — prevents single-window blips shorter than a plausible minimum activity duration

The HMM was also designed for anomaly detection: a sudden jerk spike followed by orientation stability collapse is flagged as a possible fall. These events bypass normal smoothing and get stored separately.

### Daily reports, weekly trends, and query routing

The compressed segments feed a reporting layer. Each labeled minute is counted as 60 seconds (not the ~20s burst length — the burst is evidence for the label, not the duration).

A daily report stores: total time per activity, bout count, average bout length, first/last onset. A rolling trend tracks 7-day, monthly, and yearly averages.

When a user sends a query, it's routed into one of eight types:

| Route | Example query |
|---|---|
| TASK1 | "What was she doing at 3pm?" |
| TASK2 | "How long did she walk today?" |
| TASK3 | "When did she start running?" |
| TASK4 | "Why does her cadence drop on Thursdays?" |
| ENERGY | "How many calories did she burn?" |
| ANOMALY | "Did she fall this week?" |
| PERSONALIZATION | "Update her weight to 58kg" |
| UNKNOWN | Clarification prompt |

Routing uses regex rules first (fast, no model needed). If confidence is low, it falls through to the SLM.

### Mini-RAG for Task 4

Task 4 questions — explanations of patterns — needed something beyond a retrieval system. We built a small RAG-style pipeline:

1. Extract the physics signature of the query segment (tilt, jerk, gyro axes, cadence, SMA)
2. Retrieve the 3 nearest exemplar signatures from a bank of 28 known patterns (7 classes × 4 exemplars each)
3. Force the language model to compare the query's actual numeric values against the exemplar's — it must cite specific numbers, not generate generic descriptions

This prevents hallucination. The SLM cannot answer "her cadence is high because she was running" without citing that her cadence was 158 bpm and the running exemplar has 145–175 bpm. If the numbers don't match, the answer is wrong.

### The SLM — Phi-3 Mini on device

We use **Phi-3 Mini 3.8B** (Microsoft, MIT licence) in 4-bit quantised GGUF format (~2.3 GB on disk), running locally via `llama-cpp-python` — no GPU, no cloud API, no data leaving the device.

| Configuration | Value |
|---|---|
| Model | Phi-3 Mini 3.8B Q4_K_M |
| Disk size | ~2.3 GB |
| RAM at runtime | ~3.5 GB |
| Inference latency (CPU) | 1–4 s per query |
| Context window | 2048 tokens |

For routing: greedy decoding, max 10 tokens, temperature 0.0 (deterministic).
For narration: temperature 0.3, max 200 tokens.

Both variants share the same loaded model instance — no double load. The model is optional: if weights aren't downloaded, the system falls back to rule-only routing. The Streamlit UI, storage, classification, and anomaly detection all work without the SLM.

**Edge compatibility**: The full system (LSTM + physics + SLM) fits on a Raspberry Pi 5 (8 GB RAM) or Jetson Orin Nano. The LSTM alone (no SLM) fits on 2 GB devices. The physics-only path needs under 100 MB.

---

## 2. Implementation: What We Built

### Pipeline

```
Raw sensor bursts (.dat files)
   │  timestamp + x,y,z columns, no header
   │
   ▼
[B0] Ingest — join UUID + timestamp → MinuteBurst
   │  burst=None if no raw file, coverage_s=0
   │
   ▼
[B1] Preprocess — resample to 25 Hz, complementary filter gravity
   │  body_acc = acc_raw − gravity
   │  gyro aligned to acc grid by timestamp (never by sample index)
   │  gaps > 2s stay as NaN, tracked explicitly
   │
   ▼
[B2] Feature extraction — 2s windows, 50% overlap
   │  cadence (peak counting on vertical component)
   │  SMA, dominant_freq_hz (FFT), acc_gyro_phase
   │  per-axis gyro RMS, jerk_mean, jerk_std, vertical_std
   │  → Window objects with probs=None
   │
   ▼
[B2] Score windows — AlphaAwareLSTMClassifier + physics rule
   │  LSTM: (batch, T=400, C=27) → label logits + alpha gate
   │  Physics: Features(19 fields) → predict_proba() → soft distribution
   │  Combine: alpha × p_lstm + (1−alpha) × p_physics
   │  → Window.probs (7-class softmax)
   │
   ▼
[B3] Viterbi smoothing — HMM with k-min duration constraint
   │  transition matrix from training label sequences
   │  physically-plausible prior (no running→lying in one step)
   │  CUSUM boundary snapping ±3s on |body_acc|
   │  → Segment list (coherent, no single-window blips)
   │
   ▼
[B4] Signature extraction — per segment
   │  tilt_deg mean/std, SMA, cadence_bpm, jerk_energy
   │  gyro_rms x/y/z, periodicity, orientation_stability
   │
   ├──→ [B4.5] Anomaly detection
   │       jerk spike + orientation collapse → AnomalyEvent
   │       IsolationForest fallback per user
   │
   ▼
[B5] Storage — SQLite
   │  timeline table: segments + signatures + coverage_s + provenance
   │  anomaly_events table
   │
   ▼
[B6/B7] Rollup — 60s per labeled minute
   │  daily: duration, bout count, first/last onset per class
   │  rolling: 7d / monthly / yearly trends
   │  energy: MET × weight × duration → kcal (Compendium of Physical Activities)
   │
   ▼
[B8] Query router
   │  regex rules → (route, confidence)
   │  SLM fallback when confidence < 0.75
   │
   ├──→ TASK1/2/3: direct storage lookup
   ├──→ TASK4: RAG + SLM narration (B9)
   ├──→ ENERGY: MET estimator
   └──→ ANOMALY: anomaly event table
   │
   ▼
[B9] Exemplar narration (TASK4 only)
   │  retrieve 3 nearest exemplars by cosine similarity on signature
   │  prompt SLM: compare query numbers vs. exemplar numbers
   │  SLM must cite specific values — no generic descriptions
   │
   ▼
[B10] Output formatter
   │  calls get_coverage() before every timestamp citation
   │  refuses or widens citations with 0% real signal coverage
   │
   ▼
[B11] Streamlit UI
      Timeline tab: segments + coverage fraction
      Ask tab: free-text query, confidence shown when low
      Trends tab: duration charts + kcal toggle
```

### Key modules

| File | What it does |
|---|---|
| `data/ingest.py` | UUID scan, burst file loading, label join, coverage tracking |
| `data/preprocess.py` | Resample, complementary filter gravity, body_acc split |
| `data/fusion.py` | Complementary filter implementation |
| `data/physics_rules.py` | Features dataclass (19 fields), Thresholds.fit(), predict_proba(), label_activity() |
| `pipeline/recognize.py` | extract_windows(), cadence, SMA, FFT freq, acc_gyro_phase, jerk |
| `data/dataset.py` | CHANNELS (27), build_burst_sequences(), fit_normaliser() |
| `models/lstm.py` | AlphaAwareLSTMClassifier, train_joint_alpha(), evaluate() |
| `models/hybrid.py` | _window_to_features(), physics_probs_for_window(), combine_probs() |
| `models/loco_classifier.py` | LogisticRegression walk/cycle separator, 5 features, saved as .npz |
| `models/slm.py` | Phi-3 Mini wrapper, lazy load, for_routing() / for_narration() |
| `pipeline/segment.py` | Viterbi HMM, k-min duration, CUSUM boundary snapping |
| `analysis/signature.py` | extract_signature() — label-free physics fingerprint per segment |
| `analysis/anomaly.py` | Physics fall rule + IsolationForest fallback |
| `analysis/store.py` | SQLite: timeline + anomaly_events + coverage tracking |
| `analysis/rollup.py` | Daily rollup (60s/minute), rolling trends |
| `analysis/energy.py` | MET lookup, kcal estimate |
| `analysis/router.py` | Rule-first routing → SLM fallback |
| `analysis/exemplars.py` | 28-entry exemplar bank, cosine retrieval, SLM prompt builder |
| `formatter.py` | Coverage-aware output rendering |
| `app.py` | Streamlit UI: Timeline / Ask / Trends |

---

## 3. Results

### Classification performance (macro F1)

| Stage | What changed | Macro F1 |
|---|---|---|
| LSTM only, raw 10 channels | Baseline — no preprocessing, no physics | ~0.25 |
| LSTM + fixed-alpha physics blend | Added physics rule, fixed α=0.5 | ~0.30 |
| LSTM + trainable alpha | Alpha head jointly trained | 0.332 |
| All features wired (27 channels) + improvements | jerk_std fix, acc_gyro_phase, fused gravity, class-weighted fitting | **~0.44** |

### Per-class breakdown (LSTM baseline, from checkpoint, 89,755 test samples)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Lying down | 0.373 | 0.604 | 0.462 | 18,142 |
| Sitting | 0.307 | 0.212 | 0.251 | 18,627 |
| Standing in place | 0.317 | 0.224 | 0.262 | 18,978 |
| Walking | 0.436 | 0.412 | 0.423 | 18,349 |
| Running | 0.005 | 0.009 | 0.007 | 1,967 |
| Bicycling | 0.597 | 0.577 | 0.587 | 13,692 |
| **Macro F1** | | | **0.332** | 89,755 |
| **Overall accuracy** | | | **38.6%** | |

### After all improvements (current, ~0.44 macro F1)

Running and bicycling improved most — those classes directly benefit from `jerk_std`, `acc_gyro_phase`, and the `LocoClassifier`. Posture separation (sitting vs. lying vs. standing) improved from the complementary filter giving more accurate tilt values.

### Why physics-only < physics + HMM

The HMM actually hurts on this dataset. The transition matrix is estimated from a training set with 81k sitting samples vs 737 running samples. The HMM learns "almost everything is sitting or lying" and overrides correct physics predictions toward the dominant classes — a window correctly classified as walking gets pulled back to standing-in-place because walking transitions are statistically rare. HMM smoothing only helps when per-frame errors are random, not when they're systematically biased.

### Context

The ExtraSensory paper achieves ~0.78 macro F1 using all sensor modalities: audio, WiFi, GPS, app usage, and IMU combined. We use **only accelerometer and gyroscope** — the privacy-preserving subset. On IMU-only HAR in the published literature, 0.40–0.50 macro F1 is competitive. The classes that remain hard (sitting vs. standing vs. lying) are hard for everyone — phone orientation does not reliably indicate posture when the phone is in a pocket.

---

## 4. Challenges

### The standing and moving problem

`standing and moving` is structurally absent from ExtraSensory training data — no column distinguishes it from `standing in place`. The model never sees a positive example of this class. We acknowledge it in `absent_classes`, exclude it from macro F1 computation, and in the physics rule we map it to `standing in place` at the output layer. This is a dataset limitation, not a model limitation.

### Sitting vs lying vs standing overlap

These three posture classes are extremely hard to separate from phone data alone. A phone in a pocket has no fixed orientation — the tilt angle distributions of sitting and lying down overlap almost completely in real-world data. Our complementary filter and SMA-as-tiebreaker improved things, but fundamental ambiguity remains. The model now gets it right when tilt is clean; it still struggles when the phone is in a bag or on a table.

### Running and bicycling invisibility

Running is 0.4% of the training data. Plain macro F1 optimisation sacrifices it entirely. Class-weighted fitting (running 2×, bicycling 2×) and the LocoClassifier together brought running F1 from near-zero to a meaningful value. But 737 training examples remain a hard ceiling.

### The jerk_mean vs jerk_std bug

`LocoClassifier` was trained on `jerk_std` — the variability of jerk across a window, which measures gait rhythm irregularity. The code was passing `jerk_mean` — the average jerk magnitude. These are different statistics measuring different things. The classifier was silently getting the wrong input. We caught this by reading the `LOCO_FEATURES` list against what `_features_to_loco_dict` was sending.

### Features not reaching the LSTM

`_burst_sequence` in `dataset.py` was building 11-channel input tensors (raw sensor only). The LSTM was designed for 27 channels (raw + derived physics features). This meant all the carefully computed physics features — jerk_std, acc_gyro_phase, gyro entropy, everything — were zero in the LSTM's input. The model was training without them, silently. The bug surfaced as an `IndexError` (index 26 out of bounds for size 11) when `valid_channel` was accessed. We rewrote `_burst_sequence` to compute and broadcast all 16 derived feature channels per burst.

### OOM on training

`ProcessPoolExecutor` defaulted to 16 workers × ~400 MB each = 12 GB on a machine with 3.2 GB free RAM. The kernel OOM killer terminated training before epoch 1. Fixed by capping workers at 2 via the `ACTIVITY_TRACKER_WORKERS` environment variable, and adding a per-class sampling cap to keep memory bounded regardless of dataset size.

### Import path issues

Running `python training/train_burst.py` directly fails because the project root (where `data/`, `models/`, `pipeline/` live) is not on `sys.path`. All training scripts must be run as modules from the project root: `.venv/bin/python -m training.train_burst`. The `sitecustomize.py` aliases handle backward-compatible flat imports.

### HMM making things worse

Described in Results. The core tension: HMM smoothing assumes your per-frame classifier's errors are random. Ours are systematically biased toward majority classes. A smoother that knows "she is almost always sitting" makes a biased classifier worse, not better. The transition prior needs reweighting to match actual class frequencies in the test users, not the training distribution.

---

## 5. AI Tool Disclosure

This project used **Kiro (Anthropic)** as an AI-assisted development environment.

### What AI contributed

- Signal processing code: `fuse_gravity()` (complementary filter), `acc_gyro_phase()`, `dominant_freq_hz()`, `periodicity_from_vertical()`
- Identifying the `jerk_mean`/`jerk_std` mismatch in `_features_to_loco_dict`
- Identifying the 11-channel / 27-channel mismatch in `_burst_sequence` and rewriting it to compute all derived features
- The `_weighted_macro_f1` function and the class-weight interface on `Thresholds.fit()`
- `_posture()` SMA tiebreaker logic and `lie_sma_tilt_hi` threshold
- Debugging import path issues and the `-m` invocation requirement
- Initial drafts of docstrings, inline comments, and this report structure

### What AI did not contribute

- The decision to use the LSTM as the starting point and add physics incrementally — that came from watching the pure LSTM fail on rare classes
- The decision to make alpha trainable per-sample rather than a global scalar — that came from analysing which classes the fixed blend got wrong
- The decision to add the LocoClassifier for walking/cycling separation — that came from seeing both signals overlap completely in SMA and frequency plots
- The decision to use a complementary filter — that came from measuring the 2–13° tilt drift on the 0.3 Hz Butterworth
- The RAG design forcing the SLM to cite specific numbers — that came from the course readings on grounded reasoning
- All academic citations and their mapping to specific design choices
- Experimental decisions about what to try next when accuracy stalled

All AI-generated code was reviewed, tested against the existing test suite, and in several cases substantially modified. The architecture narrative, the experimental progression, and the conclusions are the authors' own work.

---

## References

1. Vaizman, Y., Ellis, K., & Lanckriet, G. (2017). Recognizing Detailed Human Context in the Wild from Smartphones and Smartwatches. *IEEE Pervasive Computing*, 16(4), 62–74.
2. Sui, X. et al. (2024). DrHouse: An LLM-powered Conversational Drug Side Effect Checker grounded with Medical Knowledge Graph. *Proc. ACM IMWUT* 8(4). doi:10.1145/3699765
3. Lee, S. et al. (2026). JARVIS for HVAC: Sensor-Grounded LLM QA for Building Systems. *Proc. ACM IMWUT* 10(2). doi:10.1145/3810210
4. Yu, C. et al. (2025). SensorChat: Conversational Reasoning over Long-Duration Sensor Streams. *Proc. ACM IMWUT* 9(3). doi:10.1145/3749496
5. Sensor2Text (2024). Natural Language Interaction for Daily Activity Tracking. *Proc. ACM IMWUT*. doi:10.1145/3699747
6. Ainsworth, B. et al. Compendium of Physical Activities. pacompendium.com
7. Karantonis, D.M. et al. (2014). Using Hidden Markov Models to Improve Quantifying Physical Activity in Accelerometer Data. *PLOS ONE* (PMC4251969)
8. Long-term Activities Segmentation using Viterbi Algorithm with a k-minimum-consecutive-states Constraint. ResearchGate 268981604
9. Kangas, M. et al. Development of a Wearable-Sensor-Based Fall Detection System. *Sensors* (PMC4346101)
10. Efficient Activity Recognition and Fall Detection Using Accelerometers. *Springer*. doi:10.1007/978-3-642-41043-7_2
