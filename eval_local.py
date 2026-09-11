"""Evaluate physics rules on ExtraSensory pre-computed features (no burst files).

Runs entirely from the .csv.gz label files that are already local.
Fits Thresholds on train split, evaluates on test split, prints per-class
accuracy + macro-F1. Also evaluates with HMM prior smoothing per user sequence.

Usage:
    python3 eval_local.py
"""
from __future__ import annotations
import gzip, csv, math, collections
from pathlib import Path
import numpy as np

LABELS_DIR = Path("ExtraSensory.per_uuid_features_labels")
SEED = 42

COL_MAP = {
    "label:LYING_DOWN":  "lying down",
    "label:SITTING":     "sitting",
    "label:OR_standing": "standing in place",
    "label:FIX_walking": "walking",
    "label:FIX_running": "running",
    "label:BICYCLING":   "bicycling",
}
CLASSES = ["lying down", "sitting", "standing in place", "standing and moving", "walking", "running", "bicycling"]
REPORT_CLASSES = ["lying down", "sitting", "standing in place", "walking", "running", "bicycling"]


def _f(row, col):
    v = row.get(col, "")
    if v is None or v.strip() == "": return float("nan")
    try: return float(v)
    except: return float("nan")


def row_to_features(row):
    """Map pre-computed ExtraSensory columns to our Features fields.

    raw_acc:magnitude_stats:mean includes gravity (~1 g) so it is useless as
    a motion proxy. We use raw_acc:magnitude_stats:std instead -- it measures
    variability around the mean, which is zero when still and large when moving.
    Tilt comes from raw_acc:3d:mean_z (gravity projection on phone z-axis).
    """
    # Motion intensity: std of acc magnitude (body-acc proxy, gravity-free)
    sma         = _f(row, "raw_acc:magnitude_stats:std")
    periodicity = _f(row, "raw_acc:magnitude_autocorrelation:normalized_ac")
    gyro_rms    = _f(row, "proc_gyro:magnitude_stats:mean")
    cadence_period = _f(row, "raw_acc:magnitude_autocorrelation:period")  # seconds

    # dominant freq: ExtraSensory band centres ~[0-1, 1-2, 2-4, 4-8, 8-16] Hz
    bands = [_f(row, f"raw_acc:magnitude_spectrum:log_energy_band{i}") for i in range(5)]
    band_centers = [0.5, 1.5, 3.0, 6.0, 12.0]
    valid_bands = [(e, fc) for e, fc in zip(bands, band_centers) if math.isfinite(e)]
    dom_hz = max(valid_bands, key=lambda x: x[0])[1] if valid_bands else 0.0

    # tilt: angle between mean gravity vector and vertical (screen-up = 0 deg)
    mz = _f(row, "raw_acc:3d:mean_z")
    mx = _f(row, "raw_acc:3d:mean_x")
    my = _f(row, "raw_acc:3d:mean_y")
    if math.isfinite(mz) and math.isfinite(mx) and math.isfinite(my):
        gnorm = math.sqrt(mx**2 + my**2 + mz**2)
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, -mz / gnorm)))) if gnorm > 0.01 else 90.0
    else:
        tilt = 90.0

    cadence_bpm = 60.0 / cadence_period if math.isfinite(cadence_period) and cadence_period > 0 else 0.0
    gyro_x = _f(row, "proc_gyro:3d:mean_x")
    gyro_y = _f(row, "proc_gyro:3d:mean_y")
    gyro_z = _f(row, "proc_gyro:3d:mean_z")
    has_gyro = math.isfinite(gyro_rms) and gyro_rms > 0

    # vertical_std: max per-axis acc std — placement-robust proxy for the most
    # active acceleration axis. Using the max rather than a fixed axis avoids
    # phone-orientation dependence (the "vertical" axis changes with placement).
    acc_std_x = _f(row, "raw_acc:3d:std_x")
    acc_std_y = _f(row, "raw_acc:3d:std_y")
    acc_std_z = _f(row, "raw_acc:3d:std_z")
    vert_std_candidates = [v for v in [acc_std_x, acc_std_y, acc_std_z] if math.isfinite(v) and v >= 0]
    vertical_std = max(vert_std_candidates) if vert_std_candidates else 0.0

    # zcr: zero-crossing rate of acc magnitude. Walking produces more rapid
    # direction reversals per second than cycling.
    zcr_raw = _f(row, "raw_acc:magnitude_stats:zero_crossing_rate")
    zcr = zcr_raw if math.isfinite(zcr_raw) and zcr_raw >= 0 else 0.0

    from data.physics_rules import Features
    return Features(
        tilt_deg=tilt if math.isfinite(tilt) else 90.0,
        sma=sma if math.isfinite(sma) else 0.0,
        gyro_rms=gyro_rms if math.isfinite(gyro_rms) else 0.0,
        cadence_bpm=cadence_bpm,
        periodicity=periodicity if math.isfinite(periodicity) else 0.0,
        has_gyro=has_gyro,
        dominant_freq_hz=dom_hz,
        gyro_x_rms=abs(gyro_x) if math.isfinite(gyro_x) else 0.0,
        gyro_y_rms=abs(gyro_y) if math.isfinite(gyro_y) else 0.0,
        gyro_z_rms=abs(gyro_z) if math.isfinite(gyro_z) else 0.0,
        vertical_std=vertical_std,
        zcr=zcr,
    )


def load_user(path):
    """Returns list of (features, label, timestamp) for one user."""
    rows = []
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="") as fh:
        reader = csv.reader(fh)
        header = [h.strip() for h in next(reader)]
        for raw in reader:
            if not raw: continue
            row = {h: v.strip() for h, v in zip(header, raw)}
            positives = [cls for col, cls in COL_MAP.items()
                         if row.get(col, "").strip() == "1"]
            if len(positives) != 1: continue
            truth = positives[0]
            try: ts = int(float(row.get("timestamp", "")))
            except: continue
            f = row_to_features(row)
            rows.append((f, truth, ts))
    return rows


def split_users(uuids, seed=42):
    rng = np.random.default_rng(seed)
    arr = np.array(sorted(uuids))
    rng.shuffle(arr)
    n = len(arr)
    t1, t2 = int(0.6 * n), int(0.8 * n)
    return arr[:t1].tolist(), arr[t1:t2].tolist(), arr[t2:].tolist()


def macro_f1_report(confusion, classes):
    f1s = []
    print(f"\n{'Class':<22} {'Acc%':>6}  {'F1':>5}  top-error")
    for cls in classes:
        total = sum(confusion[cls].values())
        if total == 0:
            print(f"{cls:<22}  no data")
            continue
        tp = confusion[cls][cls]
        fp = sum(confusion[o][cls] for o in classes if o != cls)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / total
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        f1s.append(f1)
        wrong = sorted([(v, k) for k, v in confusion[cls].items() if k != cls], reverse=True)
        top = f"{wrong[0][1]} ({wrong[0][0]})" if wrong else "—"
        print(f"{cls:<22} {100*r:>5.1f}%  {f1:.3f}  {top}")
    mf1 = float(np.mean(f1s)) if f1s else 0.0
    print(f"\nmacro-F1: {mf1:.4f}")
    return mf1


# ── HMM prior smoothing ────────────────────────────────────────────────────

def hmm_smooth_sequence(probs_seq, log_A):
    """Viterbi decode a sequence of per-row softmax distributions."""
    T = len(probs_seq)
    S = len(CLASSES)
    if T == 0: return []
    NEG = -np.inf
    log_E = np.log(np.clip(np.array(probs_seq), 1e-12, None))  # (T, S)
    delta = np.full(S, -math.log(S)) + log_E[0]
    back = np.full((T, S), -1, dtype=int)
    for t in range(1, T):
        trans = delta[:, None] + log_A          # (S, S)
        best = np.argmax(trans, axis=0)         # (S,)
        delta = trans[best, np.arange(S)] + log_E[t]
        back[t] = best
    path = np.empty(T, dtype=int)
    path[-1] = int(np.argmax(delta))
    for t in range(T - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    return [CLASSES[i] for i in path]


def build_log_transition(label_seqs, classes):
    """Estimate transition matrix from label sequences, with self-transition prior."""
    S = len(classes)
    idx = {c: i for i, c in enumerate(classes)}
    counts = np.zeros((S, S))
    for seq in label_seqs:
        for a, b in zip(seq, seq[1:]):
            if a in idx and b in idx:
                counts[idx[a], idx[b]] += 1
    # Asymmetric physics-plausible prior:
    #   - Strong self-transition (activities persist in time)
    #   - Uniform weak cross-transitions as base
    #   - Penalise physically impossible 1-minute jumps
    idx_map = {c: i for i, c in enumerate(classes)}
    prior = np.eye(S) * 30.0 + np.ones((S, S))  # strong self, weak cross
    _impossible = [
        ("running",   "lying down"),
        ("running",   "sitting"),
        ("bicycling", "lying down"),
        ("bicycling", "standing in place"),
    ]
    for src, dst in _impossible:
        if src in idx_map and dst in idx_map:
            prior[idx_map[src], idx_map[dst]] *= 0.05
    m = counts + prior
    m = m / m.sum(axis=1, keepdims=True)
    return np.log(m)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    from data.physics_rules import Thresholds, label_activity, predict_proba

    uuids = [p.name.split(".")[0] for p in LABELS_DIR.glob("*.features_labels.csv*")]
    train_u, val_u, test_u = split_users(uuids)
    print(f"Split: train={len(train_u)} val={len(val_u)} test={len(test_u)}")

    # ── 1. Load train data ─────────────────────────────────────────────────
    print("\nLoading train split...")
    train_rows, train_seqs = [], []
    for uuid in train_u:
        path = next(LABELS_DIR.glob(f"{uuid}.features_labels.csv*"), None)
        if path is None: continue
        user_rows = load_user(path)
        train_rows.extend([(f, lbl) for f, lbl, _ in user_rows])
        train_seqs.append([lbl for _, lbl, _ in user_rows])
    print(f"  {len(train_rows)} labeled rows")
    counts = collections.Counter(lbl for _, lbl in train_rows)
    for cls in CLASSES: print(f"    {cls:<22} {counts.get(cls,0)}")

    # ── 2. Fit thresholds ──────────────────────────────────────────────────
    print("\nFitting thresholds (coordinate ascent)...")
    from data.physics_rules import _DEFAULT_GRID
    grid = dict(_DEFAULT_GRID)
    # Calibrate SMA ranges for ExtraSensory's pre-computed std feature
    # (raw_acc:magnitude_stats:std — gravity-contaminated but useful).
    grid["static_sma"] = [0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07]
    grid["run_sma"]    = [0.06, 0.08, 0.10, 0.14, 0.18, 0.22, 0.28, 0.35]
    th = Thresholds.fit(train_rows, grid=grid, n_restarts=2)
    print(f"  {th.as_dict()}")

    # ── 3. Build HMM transition matrix from train sequences ────────────────
    log_A = build_log_transition(train_seqs, CLASSES)

    # ── 4. Evaluate on test split ──────────────────────────────────────────
    print("\nLoading test split...")
    test_rows_by_user = {}
    for uuid in test_u:
        path = next(LABELS_DIR.glob(f"{uuid}.features_labels.csv*"), None)
        if path is None: continue
        test_rows_by_user[uuid] = load_user(path)

    total_test = sum(len(v) for v in test_rows_by_user.values())
    print(f"  {total_test} labeled rows")

    def collapse(pred):
        return "standing in place" if pred == "standing and moving" else pred

    # ── 4a. Physics-only (no smoothing) ───────────────────────────────────
    print("\n=== Physics rules only (no HMM) ===")
    conf_plain = collections.defaultdict(collections.Counter)
    for rows in test_rows_by_user.values():
        for f, truth, _ in rows:
            conf_plain[truth][collapse(label_activity(f, th))] += 1
    macro_f1_report(conf_plain, REPORT_CLASSES)

    # ── 4b. Physics + HMM Viterbi smoothing ───────────────────────────────
    print("\n=== Physics + HMM Viterbi smoothing ===")
    conf_hmm = collections.defaultdict(collections.Counter)
    for rows in test_rows_by_user.values():
        if not rows: continue
        probs_seq = [predict_proba(f, th, classes=CLASSES) for f, _, _ in rows]
        smoothed = hmm_smooth_sequence(probs_seq, log_A)
        for (_, truth, _), pred in zip(rows, smoothed):
            conf_hmm[truth][collapse(pred)] += 1
    macro_f1_report(conf_hmm, REPORT_CLASSES)


if __name__ == "__main__":
    main()
