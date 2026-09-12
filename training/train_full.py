"""Full training pipeline — trains all inference-time models in one command.

Trains in this order:
  1. Thresholds (physics rule constants) — coordinate ascent on train split
  2. LocoClassifier (logistic regression, walking vs cycling) — on locomotion windows
  3. AlphaAwareLSTMClassifier — joint label + alpha gate training

All three are needed at inference time:
  - Physics rules use Thresholds + LocoClassifier to produce p_physics
  - LSTM produces p_lstm + alpha
  - Final: alpha * p_lstm + (1-alpha) * p_physics

Usage:
    python -m training.train_full --epochs 40 --cap 3000 \\
        --out checkpoints/lstm_alpha_v4.pt \\
        --loco-out checkpoints/loco.npz \\
        --report checkpoints/report_full.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np

from data import dataset as ds
from data.ingest import TARGET_CLASSES
from models.lstm import TrainConfig, evaluate, save_checkpoint, train_joint_alpha
from training.train_common import (
    add_split_summary,
    build_split_sets,
    load_resume_checkpoint,
    write_report,
)

RAW = "."
LABELS = "ExtraSensory.per_uuid_features_labels"

log = logging.getLogger("train_full")


# ---------------------------------------------------------------------------
# Step 1 — Fit physics Thresholds
# ---------------------------------------------------------------------------

def train_thresholds(sets: dict, *, n_restarts: int = 5) -> "Thresholds":
    from data.physics_rules import Thresholds, _DEFAULT_GRID

    print("\n" + "="*60)
    print("STEP 1/3 — Fitting physics Thresholds (coordinate ascent)")
    print("="*60)

    t0 = time.time()
    # Build (Features, label) rows from train windows
    rows = _windows_to_feature_rows(sets["train"])
    print(f"  Train rows for fitting: {len(rows)}")

    # Up-weight rare locomotion classes so coordinate ascent doesn't sacrifice them
    class_weights = {"running": 2.0, "bicycling": 2.0, "walking": 1.5}
    th = Thresholds.fit(rows, n_restarts=n_restarts, class_weights=class_weights)
    print(f"  Fitted in {time.time()-t0:.0f}s — best thresholds: {th.as_dict()}")
    return th


def _windows_to_feature_rows(ws: ds.WindowSet):
    """Convert a WindowSet back into (Features, label_str) pairs for threshold fitting."""
    from data.physics_rules import Features
    from data.ingest import PhonePlacement

    LOCO_CLASSES = {"walking", "running", "bicycling"}
    rows = []
    for i in range(len(ws.X)):
        x = ws.X[i]          # (T, C) — use the physics feature channels (indices 10-21)
        label_idx = int(ws.y[i])
        label = TARGET_CLASSES[label_idx]

        # Physics features are broadcast as constants; take first valid timestep
        valid_mask = x[:, -1] > 0.5   # sample_valid channel (last)
        if not valid_mask.any():
            continue
        row = x[valid_mask][0]         # first valid timestep

        # Channel layout: body_acc(0-2) grav(3-5) gyro(6-8) gyro_present(9)
        # derived: tilt(10) sma(11) cadence(12) periodicity(13) vert_std(14)
        # rot_ratio(15) jerk_mean(16) gx_rms(17) gy_rms(18) gz_rms(19)
        # dom_hz(20) phase(21) placement(22-25)
        tilt_deg   = float(row[10]) * 90.0   # was normalised by /90
        sma        = float(row[11])
        cadence_hz = float(row[12])
        periodicity= float(row[13])
        vert_std   = float(row[14])
        rot_ratio  = float(row[15])
        jerk_mean  = float(row[16])
        gx_rms     = float(row[17])
        gy_rms     = float(row[18])
        gz_rms     = float(row[19])
        dom_hz     = float(row[20])
        phase      = float(row[21])
        has_gyro   = float(row[9]) > 0.5

        gyro_rms = float(np.sqrt(gx_rms**2 + gy_rms**2 + gz_rms**2))

        import math
        f = Features(
            tilt_deg=tilt_deg,
            sma=sma,
            gyro_rms=gyro_rms,
            cadence_bpm=cadence_hz * 60.0,
            periodicity=periodicity,
            has_gyro=has_gyro,
            jerk_mean=jerk_mean,
            jerk_std=0.0,      # not stored as a channel; jerk_mean used as proxy
            dominant_freq_hz=dom_hz,
            gyro_x_rms=gx_rms,
            gyro_y_rms=gy_rms,
            gyro_z_rms=gz_rms,
            vertical_std=vert_std,
            acc_gyro_phase=phase,
        )
        rows.append((f, label))
    return rows


# ---------------------------------------------------------------------------
# Step 2 — Train LocoClassifier
# ---------------------------------------------------------------------------

def train_loco(sets: dict, *, loco_out: str) -> "LocoClassifier":
    from models.loco_classifier import LocoClassifier, LOCO_FEATURES

    print("\n" + "="*60)
    print("STEP 2/3 — Training LocoClassifier (walking vs cycling logistic regression)")
    print("="*60)

    LOCO_LABELS = {"walking": 0, "bicycling": 1}
    rows = _windows_to_feature_rows(sets["train"])

    X_loco, y_loco = [], []
    for f, label in rows:
        if label not in LOCO_LABELS:
            continue
        from data.physics_rules import _features_to_loco_dict
        feat_dict = _features_to_loco_dict(f)
        X_loco.append([feat_dict.get(k, 0.0) for k in LOCO_FEATURES])
        y_loco.append(LOCO_LABELS[label])

    if len(X_loco) < 10:
        print(f"  WARNING: only {len(X_loco)} loco samples — skipping LocoClassifier training")
        return None

    X_arr = np.array(X_loco, dtype=np.float64)
    y_arr = np.array(y_loco, dtype=np.int32)
    n_walk = int((y_arr == 0).sum())
    n_bike = int((y_arr == 1).sum())
    print(f"  Loco samples: walking={n_walk}, bicycling={n_bike}")

    clf = LocoClassifier.fit(X_arr, y_arr)
    Path(loco_out).parent.mkdir(parents=True, exist_ok=True)
    clf.save(loco_out)
    print(f"  LocoClassifier saved → {loco_out}")

    # Quick eval on val set
    val_rows = _windows_to_feature_rows(sets["val"])
    correct, total = 0, 0
    for f, label in val_rows:
        if label not in LOCO_LABELS:
            continue
        from data.physics_rules import _features_to_loco_dict
        pred = clf.predict(_features_to_loco_dict(f))
        correct += int(pred == label)
        total += 1
    if total > 0:
        print(f"  Val accuracy (walk vs bike): {correct}/{total} = {correct/total:.1%}")

    return clf


# ---------------------------------------------------------------------------
# Step 3 — Train AlphaAwareLSTMClassifier
# ---------------------------------------------------------------------------

def train_lstm(sets: dict, norm, cfg: TrainConfig, *, out: str,
               resume_checkpoint: str,
               resume_model=None, resume_epoch: int = 0) -> tuple:
    print("\n" + "="*60)
    print("STEP 3/3 — Training AlphaAwareLSTMClassifier (label + alpha gate)")
    print("="*60)

    model, result = train_joint_alpha(
        sets["train"], sets["val"], norm, cfg,
        model=resume_model,
        start_epoch=resume_epoch,
        checkpoint_path=out,
        resume_checkpoint_path=resume_checkpoint,
    )
    return model, result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train all inference-time models: Thresholds + LocoClassifier + AlphaLSTM"
    )
    ap.add_argument("--cap", type=int, default=1500,
                    help="labeled minutes per class per split. "
                         "1500 ≈ 9k sequences, ~45 min/epoch on CPU. "
                         "3000 ≈ 18k sequences, ~90 min/epoch. "
                         "Use 1500 for 2-3h sessions.")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=128,
                    help="sequences per gradient update. 128 = good balance of "
                         "speed and learning. Lower (64) uses less RAM.")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--alpha-loss-weight", type=float, default=0.1)
    ap.add_argument("--restarts", type=int, default=3,
                    help="random restarts for Thresholds.fit")
    ap.add_argument("--workers", type=int, default=2,
                    help="data loading workers. Keep ≤2 on <8 GB RAM machines.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None,
                    help="resume LSTM training from this checkpoint (best or latest)")
    ap.add_argument("--skip-thresholds", action="store_true")
    ap.add_argument("--skip-loco", action="store_true")
    ap.add_argument("--out", default="checkpoints/lstm_alpha_v4.pt",
                    help="best checkpoint output path")
    ap.add_argument("--resume-out", default="checkpoints/lstm_alpha_latest.pt",
                    help="latest-epoch checkpoint (for resuming). Written every epoch.")
    ap.add_argument("--loco-out", default="checkpoints/loco.npz")
    ap.add_argument("--report", default="checkpoints/report_full.json")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    os.environ["ACTIVITY_TRACKER_WORKERS"] = str(args.workers)

    t_total = time.time()
    print(f"\n{'='*60}")
    print(f"Full training pipeline")
    print(f"  cap={args.cap}  epochs={args.epochs}  batch={args.batch_size}  hidden={args.hidden}  workers={args.workers}")
    print(f"  out (best)   = {args.out}")
    print(f"  out (latest) = {args.resume_out}  ← resume from here after stopping")
    print(f"  loco         = {args.loco_out}")
    print(f"{'='*60}")

    # ── Load data (shared across all three steps) ──────────────────────
    print("\n[data] Loading train/val/test splits...")
    sets, plan = build_split_sets(
        cap=args.cap, seed=args.seed, balance_users=True,
    )
    log.info("users: %d train / %d val / %d test",
             len(plan.train), len(plan.val), len(plan.test))
    for name, ws in sets.items():
        add_split_summary(name, ws)

    norm = ds.fit_normaliser(sets["train"].X)

    # ── Step 1: Thresholds ─────────────────────────────────────────────
    if not args.skip_thresholds:
        th = train_thresholds(sets, n_restarts=args.restarts)
    else:
        from data.physics_rules import Thresholds
        th = Thresholds()
        print("\nSTEP 1/3 — Thresholds: skipped (using defaults)")

    # ── Step 2: LocoClassifier ─────────────────────────────────────────
    if not args.skip_loco:
        loco_clf = train_loco(sets, loco_out=args.loco_out)
    else:
        loco_clf = None
        print("\nSTEP 2/3 — LocoClassifier: skipped")

    # ── Step 3: AlphaAwareLSTM ─────────────────────────────────────────
    resume_model, resume_norm, resume_epoch = load_resume_checkpoint(
        resume_path=args.resume, epochs=args.epochs,
    ) if args.resume else (None, None, 0)

    cfg = TrainConfig(
        hidden=args.hidden,
        layers=args.layers,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
        alpha_loss_weight=args.alpha_loss_weight,
        valid_channel=ds.SEQ_CHANNELS.index("sample_valid"),
    )

    model, result = train_lstm(
        sets, norm, cfg,
        out=args.out,
        resume_checkpoint=args.resume_out,
        resume_model=resume_model,
        resume_epoch=resume_epoch,
    )

    # ── Final eval on test set ─────────────────────────────────────────
    print("\n" + "="*60)
    print("FINAL TEST EVALUATION")
    print("="*60)
    absent = tuple(result.absent_classes)
    test_X = norm.apply(sets["test"].X).astype(np.float32)
    m = evaluate(model, test_X, sets["test"].y,
                 classes=TARGET_CLASSES, absent=absent)

    print(f"test accuracy : {m['accuracy']:.4f}")
    print(f"test macro F1 : {m['macro_f1']:.4f}")
    print(f"\n{'class':22} {'prec':>6} {'rec':>6} {'f1':>6} {'support':>8}")
    for c, v in m["per_class"].items():
        if v["support"] == 0 and c in absent:
            continue
        print(f"{c:22} {v['precision']:6.3f} {v['recall']:6.3f} {v['f1']:6.3f} {v['support']:8d}")

    print(f"\nTotal training time: {(time.time()-t_total)/60:.1f} min")
    print(f"Best checkpoint: {args.out}  (epoch {result.best_epoch}, val_F1={result.best_val_macro_f1:.4f})")
    if loco_clf is not None:
        print(f"LocoClassifier:  {args.loco_out}")

    # ── Write report ───────────────────────────────────────────────────
    write_report(args.report, {
        "mode": "full",
        "thresholds": th.as_dict() if not args.skip_thresholds else "defaults",
        "loco_out": args.loco_out if not args.skip_loco else None,
        "lstm_best_out": args.out,
        "lstm_latest_out": args.resume_out,
        "best_epoch": result.best_epoch,
        "best_val_macro_f1": result.best_val_macro_f1,
        "history": result.history,
        "test": m,
    })
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
