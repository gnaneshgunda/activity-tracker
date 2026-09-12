"""K-fold cross-validation eval — physics rules only (no LSTM training).

Runs 5-fold user-disjoint evaluation of the physics rule model to produce
an honest macro-F1 estimate with confidence interval. Uses the same
coordinate-ascent threshold fitting as the main eval, one fold at a time.

This is NOT used to build the deployed model — train_full.py does that.
This is for reporting: "our physics model achieves X.XX ± Y.YY macro-F1
across 5 folds" is a more credible number than a single-fold result.

Usage:
    python -m training.eval_kfold
    python -m training.eval_kfold --folds 5 --restarts 3
"""
from __future__ import annotations

import argparse
import collections
import gzip
import csv
import math
import json
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
CLASSES = ["lying down", "sitting", "standing in place",
           "standing and moving", "walking", "running", "bicycling"]
REPORT_CLASSES = ["lying down", "sitting", "standing in place",
                  "walking", "running", "bicycling"]


# ── feature extraction (same as eval_local.py) ────────────────────────────

def _f(row, col):
    v = row.get(col, "")
    if not v or v.strip() == "":
        return float("nan")
    try:
        return float(v)
    except ValueError:
        return float("nan")


def row_to_features(row):
    from eval_local import row_to_features as _rtf
    return _rtf(row)


RAW_DIR = "."   # project root — raw_acc/ and proc_gyro/ live here


# ── feature extraction from raw bursts ───────────────────────────────────

def burst_to_features(uuid: str, ts: int):
    """Load raw burst files and compute the full physics feature set.
    Returns a Features object or None if files are missing/unusable.
    """
    from data.ingest import burst_paths, load_burst_file, MinuteBurst, SensorBurst
    from data.preprocess import preprocess_burst
    from data.physics_rules import Features, periodicity_from_vertical
    from pipeline.recognize import (vertical_component, cadence_from_vertical,
                                    dominant_freq_hz, acc_gyro_phase)
    import math

    acc_p, gyro_p = burst_paths(RAW_DIR, uuid, ts)
    if not acc_p.is_file():
        return None
    try:
        acc  = load_burst_file(acc_p)
        gyro = load_burst_file(gyro_p) if gyro_p.is_file() else None
    except Exception:
        return None

    sig = preprocess_burst(MinuteBurst(acc=acc, gyro=gyro), uuid=uuid, timestamp=ts)
    if sig.n_samples < 25:
        return None

    vertical = vertical_component(sig.body_acc, sig.gravity)
    sma = float(np.nanmean(np.linalg.norm(sig.body_acc, axis=1)))
    cadence_hz, _, _ = cadence_from_vertical(vertical, fs=sig.fs, sma=sma)
    cadence_bpm = float(cadence_hz * 60) if math.isfinite(cadence_hz) else 0.0

    fin_v = vertical[np.isfinite(vertical)]
    vert_std = float(np.std(fin_v)) if fin_v.size > 1 else 0.0
    periodicity = float(periodicity_from_vertical(vertical, sig.fs))
    dom_hz = float(dominant_freq_hz(vertical, fs=sig.fs))

    diff = np.diff(sig.body_acc, axis=0) * sig.fs
    jerk_mags = np.linalg.norm(diff, axis=1)
    fin_j = jerk_mags[np.isfinite(jerk_mags)]
    jerk_mean = float(np.mean(fin_j)) if fin_j.size > 0 else 0.0
    jerk_std  = float(np.std(fin_j))  if fin_j.size > 1 else 0.0

    g_mean = np.nanmean(sig.gravity, axis=0)
    g_norm = float(np.linalg.norm(g_mean))
    tilt = float(math.degrees(math.acos(
        max(-1.0, min(1.0, float(np.dot(g_mean / g_norm, [0., 0., -1.]))))
    ))) if g_norm > 1e-9 else 90.0

    has_gyro = sig.gyro_raw is not None
    gy = sig.gyro_raw if has_gyro else np.zeros((sig.n_samples, 3))
    with np.errstate(invalid="ignore"):
        gx_rms = float(np.sqrt(np.nanmean(np.square(gy[:, 0])))) if has_gyro else 0.0
        gy_rms = float(np.sqrt(np.nanmean(np.square(gy[:, 1])))) if has_gyro else 0.0
        gz_rms = float(np.sqrt(np.nanmean(np.square(gy[:, 2])))) if has_gyro else 0.0
    gyro_rms = float(np.sqrt(gx_rms**2 + gy_rms**2 + gz_rms**2))

    mags = np.linalg.norm(sig.body_acc, axis=1)
    fin_m = mags[np.isfinite(mags)]
    if fin_m.size > 1:
        mean_m = float(np.mean(fin_m))
        crossings = int(np.sum(np.diff((fin_m > mean_m).astype(np.int8)) != 0))
        zcr = float(crossings) / (fin_m.size / sig.fs)
    else:
        zcr = 0.0

    phase = 0.0
    if has_gyro:
        rms_axes = np.array([gx_rms, gy_rms, gz_rms])
        dom_ax = int(np.argmax(rms_axes))
        phase = float(acc_gyro_phase(vertical, gy[:, dom_ax], fs=sig.fs))

    return Features(
        tilt_deg=tilt, sma=sma, gyro_rms=gyro_rms,
        cadence_bpm=cadence_bpm, periodicity=periodicity,
        has_gyro=has_gyro, jerk_mean=jerk_mean, jerk_std=jerk_std,
        dominant_freq_hz=dom_hz, gyro_x_rms=gx_rms,
        gyro_y_rms=gy_rms, gyro_z_rms=gz_rms,
        vertical_std=vert_std, zcr=zcr, acc_gyro_phase=phase,
    )


def load_user(path):
    """Load labeled rows for one user using raw burst files for full features.
    Falls back to pre-computed CSV features if no raw burst exists.
    """
    from eval_local import row_to_features as _csv_features
    rows = []
    uuid = Path(path).name.split(".")[0]
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="") as fh:
        reader = csv.reader(fh)
        header = [h.strip() for h in next(reader)]
        for raw in reader:
            if not raw:
                continue
            row = {h: v.strip() for h, v in zip(header, raw)}
            positives = [cls for col, cls in COL_MAP.items()
                         if row.get(col, "").strip() == "1"]
            if len(positives) != 1:
                continue
            try:
                ts = int(float(row.get("timestamp", "")))
            except (ValueError, TypeError):
                continue
            # Try raw burst first (full features), fall back to CSV
            f = burst_to_features(uuid, ts)
            if f is None:
                f = _csv_features(row)
            rows.append((f, positives[0]))
    return rows


# ── k-fold split ──────────────────────────────────────────────────────────

def kfold_splits(uuids: list[str], k: int = 5, seed: int = SEED):
    """Yield (train_uuids, test_uuids) for each of k folds."""
    rng = np.random.default_rng(seed)
    arr = np.array(sorted(uuids))
    rng.shuffle(arr)
    n = len(arr)
    fold_size = n // k
    for i in range(k):
        lo = i * fold_size
        hi = lo + fold_size if i < k - 1 else n
        test = arr[lo:hi].tolist()
        train = arr[:lo].tolist() + arr[hi:].tolist()
        yield train, test


# ── per-fold eval ─────────────────────────────────────────────────────────

def eval_fold(train_uuids, test_uuids, *, n_restarts: int = 3) -> dict:
    from data.physics_rules import Thresholds, label_activity, _DEFAULT_GRID

    # Load train
    train_rows = []
    for uuid in train_uuids:
        path = next(LABELS_DIR.glob(f"{uuid}.features_labels.csv*"), None)
        if path is None:
            continue
        train_rows.extend(load_user(path))

    # Fit thresholds
    grid = dict(_DEFAULT_GRID)
    grid["static_sma"] = [0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07]
    grid["run_sma"]    = [0.06, 0.08, 0.10, 0.14, 0.18, 0.22, 0.28, 0.35, 0.45, 0.60]
    class_weights = {"running": 2.0, "bicycling": 2.0, "walking": 1.5}
    th = Thresholds.fit(train_rows, grid=grid, n_restarts=n_restarts,
                        class_weights=class_weights)

    # Eval on test
    def collapse(pred):
        return "standing in place" if pred == "standing and moving" else pred

    cm = collections.defaultdict(collections.Counter)
    for uuid in test_uuids:
        path = next(LABELS_DIR.glob(f"{uuid}.features_labels.csv*"), None)
        if path is None:
            continue
        for f, truth in load_user(path):
            pred = collapse(label_activity(f, th))
            cm[truth][pred] += 1

    f1s = {}
    for cls in REPORT_CLASSES:
        total = sum(cm[cls].values())
        if total == 0:
            f1s[cls] = float("nan")
            continue
        tp = cm[cls][cls]
        fp = sum(cm[o][cls] for o in REPORT_CLASSES if o != cls)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / total
        f1s[cls] = 2 * p * r / (p + r) if p + r else 0.0

    valid_f1s = [v for v in f1s.values() if math.isfinite(v)]
    macro_f1 = float(np.mean(valid_f1s)) if valid_f1s else 0.0
    return {"macro_f1": macro_f1, "per_class": f1s, "thresholds": th.as_dict()}


# ── main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds",    type=int, default=5)
    ap.add_argument("--restarts", type=int, default=3,
                    help="coordinate-ascent restarts per fold")
    ap.add_argument("--seed",     type=int, default=SEED)
    ap.add_argument("--out",      default="checkpoints/kfold_report.json")
    args = ap.parse_args()

    uuids = [p.name.split(".")[0]
             for p in sorted(LABELS_DIR.glob("*.features_labels.csv*"))]
    print(f"K-fold eval: {args.folds} folds, {len(uuids)} users, "
          f"{args.restarts} restarts/fold\n")

    fold_results = []
    for fold_i, (train_u, test_u) in enumerate(
            kfold_splits(uuids, k=args.folds, seed=args.seed)):
        print(f"── Fold {fold_i + 1}/{args.folds}  "
              f"(train={len(train_u)} test={len(test_u)}) ──")
        result = eval_fold(train_u, test_u, n_restarts=args.restarts)
        fold_results.append(result)
        print(f"   macro-F1: {result['macro_f1']:.4f}")
        for cls in REPORT_CLASSES:
            f1 = result["per_class"].get(cls, float("nan"))
            tag = f"{f1:.3f}" if math.isfinite(f1) else "  N/A"
            print(f"   {cls:<22} {tag}")
        print()

    # Summary
    macro_f1s = [r["macro_f1"] for r in fold_results]
    mean_f1   = float(np.mean(macro_f1s))
    std_f1    = float(np.std(macro_f1s))
    print("═" * 50)
    print(f"Macro-F1 across {args.folds} folds: "
          f"{mean_f1:.4f} ± {std_f1:.4f}")
    print(f"Min: {min(macro_f1s):.4f}   Max: {max(macro_f1s):.4f}")
    print()

    # Per-class mean ± std
    print(f"{'Class':<22} {'mean F1':>8} {'± std':>7}")
    for cls in REPORT_CLASSES:
        vals = [r["per_class"].get(cls, float("nan")) for r in fold_results]
        valid = [v for v in vals if math.isfinite(v)]
        if valid:
            m, s = float(np.mean(valid)), float(np.std(valid))
            print(f"{cls:<22} {m:>8.3f} {s:>7.3f}")
        else:
            print(f"{cls:<22}      N/A")

    # Save report
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "folds": args.folds,
        "seed": args.seed,
        "mean_macro_f1": mean_f1,
        "std_macro_f1": std_f1,
        "fold_macro_f1": macro_f1s,
        "fold_results": fold_results,
    }, indent=1))
    print(f"\nReport saved → {args.out}")


if __name__ == "__main__":
    main()
