# Changes Log — Activity Tracker

What changed, why it changed, and what each thing did to the F1 score.
Written in the order things actually happened.

---

## Where we started

- Basic LSTM trained on 10 raw sensor channels (body_acc xyz, gravity xyz, gyro xyz, gyro_present)
- Hand-set physics thresholds that silently misfired on ExtraSensory data (`sma < 0.25` was capturing walking as static)
- Physics and LSTM were run separately — no fusion
- No jerk, no FFT frequency, no per-axis gyro RMS
- `build_windows` used as data pipeline — 2s windows, 50 timesteps per sample
- Fixed alpha blend of 0.5 between LSTM and physics
- **Macro F1: ~0.25**

---

## Phase 1 — Physics model overhaul

### Problem
The physics rules had hand-written thresholds that were plausible-looking but wrong for this dataset. Locomotion and posture classes were misclassified at the boundary because the constants were copied from other setups.

### What changed (`data/physics_rules.py`)

**New features added to `Features` dataclass:**
- `jerk_mean` — mean magnitude of frame-to-frame body_acc derivative. Running foot-strikes spike jerk sharply; walking is smoother
- `jerk_std` — **rhythm variability** of jerk. Walking is irregular (each step varies); cycling is mechanically smooth. This is what `LocoClassifier` was actually trained on — the old code was passing `jerk_mean` by mistake (silent bug, now fixed)
- `dominant_freq_hz` — FFT peak in 0.5–5 Hz band. More robust than autocorrelation cadence for short/noisy windows
- `gyro_x_rms`, `gyro_y_rms`, `gyro_z_rms` — kept axis-resolved, not collapsed to magnitude. Walking has dominant pitch (y-axis); cycling concentrates rotation in one axis
- `gyro_entropy` — Shannon entropy of gyro energy across axes. Low = one axis dominates (cycling wheel spin). High = chaotic spread (walking foot-strikes)
- `gyro_dominance` — max single-axis RMS / vector magnitude. Near 1.0 = cycling. Near 0.577 = uniform (walking/running)
- `acc_gyro_phase` — cosine of phase difference between vertical acc and dominant gyro axis at cadence frequency. Walking pendulum swing ≈ 0.8 (in-phase). Cycling ≈ 0.0 (decoupled). This directly separates walking from cycling better than any magnitude feature
- `vertical_std` — std of gravity-projected body acceleration. Separates lying from walking
- `zcr` — zero-crossing rate of acc magnitude

**`Thresholds.fit()` — coordinate ascent instead of hand-set constants:**
- Structure stays hand-written physics (interpretable)
- Constants fitted by coordinate ascent over a grid maximising macro-F1 on training data
- n_restarts: 2 → 10 (more random restarts to escape local optima)
- Passes: 6 → 8 per restart
- Added `class_weights` parameter: running×2, bicycling×2, walking×1.5 so rare classes aren't sacrificed
- Added `_weighted_macro_f1` replacing `_macro_f1` — backward compatible

**`predict_proba()` improvements:**
- `acc_gyro_phase` added to walking score (+margin) and bicycling score (-margin)
- No-gyro cycling fallback: replaced flat `-1.0` penalty with `-margin(cadence_bpm, walk_cadence_lo) * 0.5` — proportional penalty based on cadence evidence instead of collapsing bicycling to zero
- Added `phase_walk_lo` threshold to `Thresholds` and default fitting grid

**`_posture()` improvements:**
- Added `lie_sma_tilt_hi` SMA tiebreaker in ambiguous tilt zone
- Lying down (SMA median ~0.002g) is genuinely quieter than sitting (0.003g) — used as secondary discriminator when tilt overlaps

**`LocoClassifier` fix:**
- `_features_to_loco_dict` was passing `jerk_mean` to a model trained on `jerk_std`
- Fixed to pass `jerk_std`, `zcr`, actual `rotation_ratio`, `dominant_freq_hz`

---

## Phase 2 — Input channels: 10 → 27 → 35

### Problem
The LSTM was only seeing raw sensor channels. All the physics features we computed were never given to the neural network. The model had to re-derive everything from scratch.

### What changed (`data/dataset.py`)

**Channels expanded from 10 to 27:**
- Raw sensor (10): unchanged
- Derived physics (12): tilt_deg, SMA, cadence_hz, periodicity, vertical_std, rotation_ratio, jerk_mean, gyro xyz RMS, dominant_freq_hz, acc_gyro_phase
- Placement flags (4): phone_pocket, phone_hand, phone_bag, phone_table

**Channels further expanded from 27 to 35 (temporal context):**
- 8 new channels: rolling mean/std of SMA and jerk_mean over ±2 and ±5 burst neighbours
- `sma_roll2_mean/std`, `jerk_roll2_mean/std`, `sma_roll5_mean/std`, `jerk_roll5_mean/std`
- Why: a single 20s burst can't distinguish sitting from lying — they're both quiet and near-horizontal. But a person lying down all morning has consistently low SMA neighbours; someone who just sat down from walking still has high-SMA neighbours. Context across minutes solves what context within a window cannot

**`_burst_sequence` rewrite:**
- Was building 11-channel tensors (raw sensor only)
- Now computes all 26 physics/placement channels per burst
- `sample_valid` flag moved to be appended last (index 34) so it always matches `SEQ_CHANNELS[-1]` — this was causing a critical bug where the LSTM's masked pooling couldn't find the validity flag

**`_attach_temporal_context` added:**
- Runs after all bursts are loaded, before stacking
- Uses centred rolling window across burst sequence
- SMA index 11, jerk_mean index 16 — correct channel positions

---

## Phase 3 — Data pipeline: windows → burst sequences

### Problem
`build_windows` was creating ~18 overlapping 2s windows per burst (50 timesteps each). This was:
1. Slow — 8240 minutes × 18 windows = 148k sequences, 30+ min to load
2. Wrong for the AlphaAwareLSTMClassifier which expects 400-timestep burst sequences
3. Inflating sample count 18× without adding real information (all windows of one burst share one label)

### What changed (`training/train_common.py`)
- `build_windows` → `build_burst_sequences` in `build_split_sets`
- Now: one sequence of (400, 35) per labeled minute
- Data loading: 30 min → 2 min for cap=1500
- LSTM sees the full 16-second burst (400 timesteps at 25 Hz), not a 2-second slice

---

## Phase 4 — Complementary filter gravity

### Problem
The gravity estimator used a 0.3 Hz Butterworth low-pass. During walking, the phone's own acceleration leaks into the estimate, causing 2–13° of tilt drift per burst. This made posture separation less reliable.

### What changed (`data/preprocess.py`, `data/fusion.py`)
- Added complementary filter (`fuse_gravity`) that blends:
  - Gyroscope: accurate short-term orientation, but drifts
  - Accelerometer: absolute gravity reference, noisy during motion
  - Formula: `g = normalise(α × g_gyro_propagated + (1-α) × g_acc)`
  - Time constant τ = 1.5s (above gait period so a stride can't drag the estimate)
- Added `use_fusion=True` parameter to `preprocess_burst` — falls back to LP when no gyro
- Added `PreprocessFlag.GRAVITY_FUSED` marker
- Tilt drift: 2–13° → under 1° during active motion

---

## Phase 5 — Alpha-aware LSTM

### Problem
Fixed-alpha blend (α=0.5) assumed LSTM and physics are equally good on all inputs. Not true — physics is better on clean locomotion signals, LSTM is better on ambiguous posture.

### What changed (`models/lstm.py`)

**`AlphaAwareLSTMClassifier`:**
- Two output heads: label logits (7-class) + alpha gate (7-dim vector, one per class)
- Alpha is a learned per-class weight — α_bicycling can be 1.0 (trust physics rotation features) while α_sitting is 0.0 (trust LSTM posture context)
- Supervised: target α derived from gap between LSTM and physics log-likelihoods per class
- Loss = focal_loss(label) + 0.1 × MSE(alpha_pred, alpha_target)

**Weight initialisation fix (NaN loss on first batch):**
- Default PyTorch random init caused NaN loss immediately with T=400 sequences
- Fixed with orthogonal init for recurrent weights (standard for long sequences)
- Forget gate bias set to 1.0 (helps gradient flow in early training)

**LR schedule fix:**
- Was: lr=1e-3, ReduceLROnPlateau(patience=2)
- Model peaked at epoch 3 then oscillated — typical sign of LR too high
- Now: lr=3e-4, 3-epoch warmup (ramps from lr/10 → lr), ReduceLROnPlateau(patience=3)

**Best-checkpoint save on improvement:**
- Was: only saved checkpoint at end of training
- Now: saves to `--out` every time val F1 improves
- Also saves every epoch to `--resume-out` for crash recovery / session resumption

**Trainable config defaults:**
- epochs: 30 → 60
- patience: 6 → 10
- batch_size: 256 → 128 (more gradient updates per epoch)
- warmup_epochs: 3 (new)

---

## Phase 6 — Normaliser robustness

### Problem
Raw physics features had extreme outliers (jerk_mean up to 34 g/s, rotation_ratio up to 22). After standard normalisation (`x - mean) / std`), normalised values reached ±22. LSTM with random weights explodes to NaN immediately on inputs this large.

### What changed (`data/dataset.py`)

**Robust normaliser:**
- Mean → median (outlier-resistant)
- Std → IQR/1.349 (robust equivalent of std for Gaussian)
- Clip to ±5σ after standardising — outlier spikes become ±5, not ±22
- `clip_sigma` saved in checkpoint so inference clips identically

---

## Phase 7 — HMM improvements

### Problem 1: Uncalibrated emissions
The LSTM trained with focal loss on imbalanced data is overconfident on majority classes. Viterbi treats these as likelihoods, so overconfident majority emissions drown out minority-class evidence. Documented failure: HMM made macro-F1 **worse** than no smoothing (0.2442 vs 0.2528).

### What changed (`pipeline/segment.py`)

**Temperature scaling:**
- `fit_temperature(val_probs, val_labels)` — grid search over T ∈ [0.5, 2.0] minimising NLL
- `calibrate_emissions(log_E, T)` — divides log-emissions by T then re-normalises
- Applied before every Viterbi decoding step
- T > 1 softens overconfident distributions; minority classes can compete

### Problem 2: Fixed transition matrix ignores time gaps
ExtraSensory captures ~20s per minute — leaving ~40s of unobserved time between bursts. The old HMM used one fixed transition matrix per step, assuming the same activity continued through the gap.

**Time-aware transitions (CTMC):**
- `generator_matrix(A, dt)` — converts discrete A to generator Q via matrix logarithm
- `transition_for_dt(Q, dt)` — computes `expm(Q × Δt)` for actual elapsed time between windows
- Within-burst step (Δt ≈ 1s): matrix ≈ A (no change)
- Cross-burst gap (Δt ≈ 40s): rows relax toward stationary distribution
- After 1h gap: rows converge to near-identical (activity could be anything)

---

## Phase 8 — Unified eval split

### Problem
`eval_local.py` used its own `split_users()` which shuffled from glob order. `data.dataset.split_users()` sorted first then shuffled. Both had seed=42 but produced different test users — physics eval and LSTM training were measuring on different people.

### What changed (`eval_local.py`)
- `eval_local.split_users()` now delegates to `data.dataset.split_users(seed=42)`
- Both paths evaluate on the exact same 12 test users
- Verified user-disjoint: Train∩Test=0, Val∩Test=0, Train∩Val=0

---

## Phase 9 — Pipeline bug fixes

These were silent bugs causing wrong behaviour, not crashes:

| Bug | Location | Effect | Fix |
|---|---|---|---|
| `jerk_mean` passed as `jerk_std` to LocoClassifier | `physics_rules.py` | Walk/cycle separator getting wrong feature | Fixed field name |
| 11-channel LSTM input (physics features were zero) | `dataset.py` | LSTM trained without jerk, phase, gyro entropy | Rewrote `_burst_sequence` to populate all 26 channels |
| `sample_valid` at wrong index (26 instead of 34) | `dataset.py` | LSTM masked pooling found no valid timesteps → all outputs NaN | Moved `sample_valid` to last position |
| `build_windows` instead of `build_burst_sequences` | `train_common.py` | Wrong sequence length (50 vs 400 timesteps) → IndexError | Swapped to `build_burst_sequences` |
| `SensorBurst(timestamps=...)` wrong kwarg | `run_pipeline.py` | Pipeline crashed on every file upload | Fixed to `t=` |
| `MinuteBurst(uuid=..., timestamp=...)` wrong kwargs | `run_pipeline.py` | Pipeline crashed on every file upload | Moved to `preprocess_burst()` kwargs |
| Space-separated .dat files parsed as one CSV column | `run_pipeline.py` | "No timestamp column found" error on upload | `_read_csv` auto-detects delimiter and headerless files |
| Stale tool JSON appended to `run_pipeline.py` | `run_pipeline.py` | SyntaxError on import | Removed trailing metadata |
| SQLite connection not thread-safe | `analysis/store.py` | "SQLite objects created in thread X" error in Streamlit | Added `check_same_thread=False` + `threading.Lock` |
| TASK2 response had no timestamps | `app.py`, `formatter.py` | Evidence showed "N/A" for time range on aggregation queries | Added `t_start`/`t_end` to query responses; formatter reads them for non-coverage-checked routes |

---

## Current results

```
cap=100, epochs=5 (smoke test, 5.7 min):

Class                    Prec    Rec     F1    Support
lying down              0.378  0.540  0.444      300
sitting                 0.315  0.190  0.237      300
standing in place       0.308  0.307  0.307      300
walking                 0.444  0.400  0.421      300
running                 0.509  0.672  0.579      131
bicycling               0.541  0.503  0.522      300

val macro-F1: 0.4497   (epoch 5)
```

| Stage | Macro F1 |
|---|---|
| LSTM only, 10 channels, hand-set thresholds | ~0.25 |
| + Fitted thresholds, fixed-alpha blend | ~0.30 |
| + Trainable alpha (AlphaAware) | 0.332 |
| + All 35 channels, burst sequences, NaN fixes | **0.4497** |
| + cap=1500, 60 epochs (projected) | ~0.48–0.55 |

---

## Training commands

**First run:**
```bash
.venv/bin/python -m training.train_full \
  --cap 1500 --epochs 60 \
  --out checkpoints/lstm_alpha_v4.pt \
  --resume-out checkpoints/lstm_alpha_latest.pt \
  --loco-out checkpoints/loco.npz \
  --report checkpoints/report_full.json
```

**Resume after stopping:**
```bash
.venv/bin/python -m training.train_full \
  --cap 1500 --epochs 60 \
  --resume checkpoints/lstm_alpha_latest.pt \
  --skip-thresholds --skip-loco \
  --out checkpoints/lstm_alpha_v4.pt \
  --resume-out checkpoints/lstm_alpha_latest.pt \
  --report checkpoints/report_full.json
```

**K-fold eval for report (physics only, ~30 min):**
```bash
.venv/bin/python -m training.eval_kfold --folds 5 --restarts 3
```

---

## Phase 10 — TASK4 properly wired to exemplar retrieval + SLM

### Problem
TASK4 queries ("why does his cadence drop?", "explain this pattern") were routing to `_query_generic()` which just returned "Most common activity: walking (4.2 min)." — no SLM, no exemplar retrieval, none of the grounded narration that `exemplars.py` implements.

### What changed (`app.py`)

`_query_generic()` now runs the full TASK4 path:

1. Find the dominant activity segment from DB
2. Build a signature proxy (tilt, SMA, cadence, gyro axes) from the segment row
3. `nearest_exemplars()` — cosine similarity on 9 physics features against 28 hand-curated exemplars (walking flat/uphill/slow, running jog/sprint, cycling flat/rough/uphill, postures, anomaly events)
4. `build_explain_prompt()` — 2800-char structured prompt with:
   - Measured feature table (actual numbers from the segment)
   - Closest exemplar feature table + description
   - System rules: cite specific numbers, compare against exemplar, don't invent values
5. SLM narration (Phi-3 Mini if downloaded) or exemplar description fallback

The SLM **must** cite the actual measured values and compare them against the exemplar. It cannot write a generic activity description from its own training knowledge. This follows the DrHouse grounded-reasoning pattern (Sui et al. 2024).

---

## Phase 11 — Overfitting fix: smaller model + stronger dropout

### Problem
With cap=1500 (~9,000 sequences) and hidden=96 (331k parameters), the model showed textbook overfitting:

```
Epoch 8:  train_loss=0.929  val_f1=0.435  ← best
Epoch 18: train_loss=0.532  val_f1=0.410  ← train keeps falling, val stuck/declining
```

Param/sample ratio was 37 — too large. The model was memorising the 9k training sequences rather than generalising.

### What changed (`models/lstm.py`)

- `hidden`: 96 → 64 (parameters: 331k → 155k)
- `dropout`: 0.3 → 0.5 (stronger regularisation)
- Removed duplicate `dropout` field in `TrainConfig` (was defined twice, second shadowing first)

New param/sample ratio at cap=1500: **17** (was 37). Still high but much more manageable with dropout=0.5.

---

## Phase 12 — Temporal context feature index bug fix

### Problem
`_attach_temporal_context()` was reading `x[:, -1] > 0.5` to find valid timesteps (looking for `sample_valid` flag at the last column). But `_burst_sequence` was changed to NOT include `sample_valid` in the raw block — it's now appended separately by `build_burst_sequences` after temporal context is added. So `x[:, -1]` was reading whatever the last physics channel was, not the validity flag, giving wrong SMA/jerk values for the rolling context.

### What changed (`data/dataset.py`)

- `_attach_temporal_context()`: switched from `x[:, -1] > 0.5` to `x[0, 11]` (first timestep, SMA channel) — the physics features are broadcast constants so any timestep gives the same value, and index 0 is always a real sample (padding is added after)
- `_burst_sequence()`: now returns a 5-tuple `(block, label_index, uuid, ts, valid_col)` — `sample_valid` stored separately
- `build_burst_sequences()`: unpacks the 5-tuple, calls `_attach_temporal_context()` on the 26-channel blocks, then appends `sample_valid` as the last column (index 34 = `SEQ_CHANNELS[-1]`)

This guarantees `sample_valid` is always at index 34, matching `SEQ_CHANNELS.index("sample_valid")`, so the LSTM's masked pooling always finds the correct validity flag.

---

## Phase 13 — Robust normaliser (NaN loss fix)

### Problem
With the correct 35-channel input, `fit_normaliser` used mean/std which was dominated by outlier spikes (jerk_mean raw max = 34 g/s, rotation_ratio max = 22). After standard normalisation, values reached ±22. The LSTM with random initialisation produced NaN on the first forward pass.

### What changed (`data/dataset.py`)

`fit_normaliser()` now uses:
- **Median** instead of mean (outlier-resistant center)
- **IQR/1.349** instead of std (robust equivalent for Gaussian distributions)
- **±5σ clipping** applied after standardising — outlier spikes become ±5 instead of ±22
- `clip_sigma` saved in checkpoint so inference clips identically to training

```
Before: normalised range ±22  → NaN loss on first batch
After:  normalised range ±5   → loss=1.64 on first batch (correct for 7 classes)
```

---

## Phase 14 — LSTM weight initialisation (NaN loss fix)

### Problem
Even with normalised inputs in ±5, the BiLSTM with default PyTorch random initialisation produced NaN on the first forward pass with T=400 timesteps. Default uniform random weights cause activation explosion over long sequences.

### What changed (`models/lstm.py`)

Added `_init_weights()` to `AlphaAwareLSTMClassifier`:
- Input weights: Xavier uniform (controls input-to-hidden scale)
- Recurrent weights: **orthogonal initialisation** — standard practice for long sequences (prevents hidden state from growing/shrinking exponentially)
- Biases: zero-initialised, with **forget gate bias = 1.0** (helps gradient flow in early training by keeping the forget gate open)

```
Before: first forward pass → NaN logits → NaN loss
After:  first forward pass → logits range [-1.3, 1.6] → loss=1.64
```

---

## Phase 15 — Training data pipeline: windows → burst sequences

### Problem
`train_common.py` was calling `build_windows()` which creates ~18 overlapping 2s windows per burst (50 timesteps, shape (50, 35)). But `AlphaAwareLSTMClassifier` expects 400-timestep burst sequences. This caused:
- Data loading: 30+ minutes for cap=1500 (148k windows to process)
- Shape mismatch: `IndexError: index 26 out of bounds for dimension 1 with size 11`
- Wrong granularity: 18 windows sharing one label inflates sample count 18× without adding information

### What changed (`training/train_common.py`)

`build_windows()` → `build_burst_sequences()`:
- One sequence of (400, 35) per labeled minute (not 18 windows)
- Data loading: 30 min → 2 min for cap=1500
- LSTM sees the full 16-second burst, not a 2-second slice
- 9,000 training sequences at cap=1500 (not 162,000 windows)

---

## Summary of all model changes

| What | Before | After | Why |
|---|---|---|---|
| Input channels | 10 | 35 | Physics features + temporal context wired in |
| Sequence length | 50 timesteps (2s windows) | 400 timesteps (16s bursts) | Full burst context for LSTM |
| Hidden units | 96 | 64 | Reduce overfitting (331k→155k params) |
| Dropout | 0.3 | 0.5 | Main regulariser for small dataset |
| LR | 1e-3 | 3e-4 | Prevent early oscillation |
| Warmup | none | 3 epochs | Stable start |
| Patience | 6 | 10 | More room to converge |
| Gravity | 0.3Hz Butterworth | Complementary filter | 2-13° → <1° tilt drift |
| Normaliser | mean/std | median/IQR + ±5σ clip | Outlier robustness, NaN fix |
| LSTM init | PyTorch default | Orthogonal + forget-gate bias=1 | NaN on first batch fixed |
| HMM transitions | Fixed matrix | Time-aware CTMC expm(Q×Δt) | Cross-burst gaps relax to stationary |
| Calibration | None | Temperature scaling before Viterbi | Minority class HMM override fixed |
| TASK4 | Generic summary | Exemplar retrieval + SLM narration | Actually uses the exemplar bank |

## Current best result

```
val macro-F1: 0.4352  (epoch 8, cap=1500, hidden=96 — now overfitting)
```

Expected with hidden=64, dropout=0.5, cap=1500, 60 epochs: **~0.46–0.52**
