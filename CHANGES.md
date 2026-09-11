# Model Improvements — Change Log

Baseline result before these changes:

```
accuracy  : 0.3972
macro F1  : 0.3953

lying down          prec=0.366  rec=0.530  f1=0.433
sitting             prec=0.245  rec=0.378  f1=0.298   ← worst recall
standing in place   prec=0.265  rec=0.150  f1=0.192   ← nearly invisible
standing and moving prec=0.000  rec=0.000  f1=0.000   (absent in train)
walking             prec=0.467  rec=0.481  f1=0.474
running             prec=0.621  rec=0.333  f1=0.433   ← good precision, bad recall
bicycling           prec=0.575  rec=0.514  f1=0.542
```

Root causes identified:
- Data imbalance: sitting/lying dominate; running/bicycling/standing-and-moving are rare
- Scalar alpha gate collapses early — one gate for all 7 classes simultaneously
- Plain CrossEntropy + inverse-frequency weights: majority class gradients still dominate
- Physics probs fed raw into alpha target: hard rule outputs (near 0/1) kill alpha gradients

---

## Change 1 — Dynamic Window Stride by Class

**File**: `data/dataset.py`

**What**: Added `_CLASS_STRIDE_S` table. `_minute_windows()` now passes a
per-class overlap to `extract_windows()` instead of a fixed 50%.

| Class | Stride | Overlap | Reason |
|---|---|---|---|
| lying down | 2.0 s | 0% | Already overrepresented — fewer windows per burst |
| sitting | 2.0 s | 0% | Same |
| standing in place | 1.0 s | 50% | Moderate — keep existing behaviour |
| walking | 1.0 s | 50% | Same |
| running | 0.2 s | 90% | Minority — ~9× more windows per burst |
| bicycling | 0.2 s | 90% | Same |
| standing and moving | 0.2 s | 90% | Absent from ExtraSensory but prepared |

**Why not SMOTE**: SMOTE interpolates between sensor windows. Interpolating
between a running window and a walking window produces a signal that never
occurred in nature and teaches the model a fake boundary. Stride resampling
produces more views of real signals only.

**Expected effect**: running and bicycling window counts increase ~9×, sitting
and lying decrease ~2×. The training distribution becomes closer to uniform
without any synthetic data.

---

## Change 2 — Effective-Number Class Weights + Focal Loss

**File**: `models/lstm.py`

### 2a — Effective-number weights (replaces inverse-frequency)

Old formula: `w_c = total / (n_present × count_c)`

New formula (Cui et al. 2019, β=0.999):
```
effective_c = (1 - β^N_c) / (1 - β)
w_c = 1 / effective_c
```

For large N_c (sitting: ~28k windows) the effective number saturates near
`1/(1-β) = 1000`, so the weight stops growing. For small N_c (running: ~7k)
the weight is meaningfully larger. This prevents the inverse-frequency formula
from assigning extreme weights that destabilise training on very rare classes.

### 2b — Focal loss (replaces CrossEntropyLoss)

```
L_focal = -Σ_c  w_c · (1 - p_{t,c})^γ · log(p_{t,c})     γ = 2.0
```

The `(1 - p_t)^γ` term is the key addition. Once the model predicts sitting
correctly with high confidence (p_t → 1), that sample's gradient contribution
shrinks toward zero. The optimizer is forced to keep working on the hard
examples (running, bicycling) rather than over-fitting the easy majority.

**Expected effect**: sitting recall may drop slightly (it was inflated by
gradient dominance); running and bicycling recall should increase.

---

## Change 3 — Vector Alpha Gate (7-dim, replaces scalar)

**File**: `models/lstm.py` — `AlphaAwareLSTMClassifier`

**Old**: `alpha_head` output shape `(B, 1)` — one scalar gate per window.

**New**: `alpha_head` output shape `(B, 7)` — one gate per class per window.

```
fused_probs = α_t ⊙ P_physics + (1 - α_t) ⊙ P_lstm
```

**Why the scalar fails**: a scalar gate forces the model to choose one expert
for all 7 classes simultaneously. In practice:
- Physics is strong for bicycling (yaw rotation ratio is a reliable rule)
- LSTM is stronger for sitting vs. standing (temporal context matters)
- A scalar set to trust physics for bicycling also trusts physics for sitting,
  where the rule is weakest

**Why the vector fixes it**: each class independently learns its own trust
weight. The model can learn `α_bicycling → 0.9` and `α_sitting → 0.1` at the
same time.

**Alpha target** is now also per-class:
```python
alpha_target = sigmoid((ml_logp - physics_logp) / 1.0)   # shape (B, 7)
```

Where `ml_logp` and `physics_logp` are the full log-probability vectors, not
just the score at the true class. This gives the gate a signal for every class
on every window, not just the ground-truth class.

---

## Change 4 — Physics Temperature Scaling (T = 2.0)

**File**: `models/lstm.py` — `train_joint_alpha()`

Before computing the alpha target, physics log-probs are re-softmaxed at
temperature T=2.0:

```python
physics_probs = softmax(log(physics_raw) / 2.0)
```

**Why**: the physics rule outputs can be near-deterministic (e.g. tilt clearly
indicates lying down → P_lying ≈ 0.95, all others ≈ 0.01). When fed raw into
the alpha target formula, this makes `alpha_target ≈ 0` for almost every
window where physics is confident, which drives the alpha head to always output
0 early in training. Once `α → 0`, gradients stop flowing into the alpha head
parameters — gate collapse.

T=2.0 softens the physics distribution enough that the alpha target stays in
a useful gradient range without destroying the physics signal entirely.

---

## Checkpoint Compatibility

The old scalar alpha head had shape `Linear(pool_dim, 1)`.
The new vector alpha head has shape `Linear(pool_dim, 7)`.

**These are incompatible.** Any checkpoint trained before these changes
(`lstm_alpha.pt`, `lstm_alpha_ft.pt`) cannot be fine-tuned — the state dict
will fail to load due to the shape mismatch on `alpha_head.2.weight` and
`alpha_head.2.bias`.

Train from scratch with the command below.

---

## Training Command

```bash
python3 -m training.train_alpha \
  --cap 300 \
  --epochs 40 \
  --hidden 96 \
  --layers 2 \
  --lr 1e-3 \
  --batch-size 64 \
  --workers 2 \
  --seed 42 \
  --out checkpoints/lstm_alpha_v3.pt \
  --report checkpoints/lstm_alpha_v3_report.json
```

**Why `--cap 300` and not higher**: with 90% overlap on minority classes,
300 minutes of running produces ~2700 windows (vs ~300 before). Effective
window count is already much higher. Raising cap further risks OOM.

**Why `--batch-size 64`**: focal loss gradients are noisier than CE gradients
(the `(1-p)^γ` weighting makes batch statistics less stable). Smaller batches
help.

**Why `--epochs 40`**: the new loss landscape takes longer to converge. The
patience=6 early stopping will cut it short if val macro-F1 plateaus.
