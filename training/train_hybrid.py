"""Train the LSTM and learn the hybrid mixing weight alpha.

Usage:
    python train_hybrid.py --cap 1500 --epochs 30 --out checkpoints/hybrid.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from data import dataset as ds
from data.ingest import TARGET_CLASSES, MinuteBurst, burst_paths, load_burst_file
from data.preprocess import preprocess_burst
from models.hybrid import fit_alpha, physics_probs_for_window
from models.lstm import TrainConfig, evaluate, save_checkpoint, train
from pipeline.recognize import extract_windows, softmax
from training.train_common import add_split_summary, build_split_sets, load_resume_checkpoint, write_report

RAW = "."
LABELS = "ExtraSensory.per_uuid_features_labels"


def _window_rows(minutes, raw_root):
    rows = []
    for minute in minutes:
        acc_p, gyro_p = burst_paths(raw_root, minute.uuid, minute.timestamp)
        if not acc_p.is_file():
            continue
        try:
            acc = load_burst_file(acc_p)
            gyro = load_burst_file(gyro_p) if gyro_p.is_file() else None
        except (ValueError, OSError):
            continue
        sig = preprocess_burst(
            MinuteBurst(acc=acc, gyro=gyro), uuid=minute.uuid, timestamp=minute.timestamp
        )
        for w in extract_windows(sig):
            if not w.is_usable:
                continue
            rows.append((w, sig, minute.label_index))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=1500, help="minutes per class per split")
    ap.add_argument("--no-balance-users", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None, help="checkpoint path to resume/fine-tune from")
    ap.add_argument("--out", default="checkpoints/hybrid.pt")
    ap.add_argument("--report", default="checkpoints/hybrid_report.json")
    ap.add_argument("--alpha-grid", nargs="*", type=float, default=None,
                    help="grid of alpha values to test, default is 0..1 in 101 steps")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("train_hybrid")

    sets, plan = build_split_sets(
        cap=args.cap,
        seed=args.seed,
        balance_users=not args.no_balance_users,
    )
    log.info("users: %d train / %d val / %d test", len(plan.train), len(plan.val), len(plan.test))

    minutes_by_split = {}
    for name, group in plan.as_dict().items():
        minutes_by_split[name] = ds.scan_all(group, LABELS, RAW)
        if name in sets:
            add_split_summary(name, sets[name])

    norm = ds.fit_normaliser(sets["train"].X)
    cfg = TrainConfig(
        hidden=args.hidden, layers=args.layers, lr=args.lr,
        batch_size=args.batch_size, epochs=args.epochs, seed=args.seed,
    )

    resume_model = None
    resume_epoch = 0
    if args.resume:
        resume_model, resume_norm, resume_epoch = load_resume_checkpoint(
            resume_path=args.resume,
            epochs=args.epochs,
        )
        norm = resume_norm
        print(f"resuming from {args.resume} at epoch {resume_epoch}")

    model, result = train(
        sets["train"], sets["val"], norm, cfg,
        model=resume_model,
        start_epoch=resume_epoch,
    )

    val_rows = _window_rows(minutes_by_split["val"], RAW)
    if len(val_rows) != len(sets["val"]):
        raise ValueError(f"validation window count mismatch: {len(val_rows)} vs {len(sets['val'])}")

    Xva = norm.apply(sets["val"].X).astype(np.float32)
    with np.errstate(all="ignore"):
        ml_probs = softmax(model(torch.from_numpy(Xva).float()).detach().numpy(), axis=1)
    phys_probs = np.stack([physics_probs_for_window(w, s) for w, s, _ in val_rows])
    alpha = fit_alpha(ml_probs[: len(phys_probs)], phys_probs, sets["val"].y[: len(phys_probs)], grid=args.alpha_grid)

    test_metrics = evaluate(
        model, Xva, sets["val"].y,
        classes=TARGET_CLASSES, absent=result.absent_classes,
    )

    print(f"\n=== VALIDATION (held-out users) ===")
    print(f"alpha     : {alpha:.4f}")
    print(f"accuracy  : {test_metrics['accuracy']:.4f}")
    print(f"macro F1  : {test_metrics['macro_f1']:.4f}")

    save_checkpoint(args.out, model, norm, result)
    write_report(args.report, {
        "alpha": alpha,
        "split": {k: list(v) for k, v in plan.as_dict().items()},
        "train_counts": result.train_counts,
        "best_epoch": result.best_epoch,
        "best_val_macro_f1": result.best_val_macro_f1,
        "history": result.history,
        "validation": test_metrics,
    })
    print(f"\nsaved {args.out} and {args.report}")


if __name__ == "__main__":
    main()
