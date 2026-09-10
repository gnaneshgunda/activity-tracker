"""Train the B2 LSTM window classifier end to end.

Usage:
    python train_lstm.py --cap 1500 --epochs 30 --out checkpoints/lstm.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

import dataset as ds
from ingest import TARGET_CLASSES
from lstm import TrainConfig, evaluate, save_checkpoint, train

RAW = "."
LABELS = "ExtraSensory.per_uuid_features_labels"


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
    ap.add_argument("--out", default="checkpoints/lstm.pt")
    ap.add_argument("--report", default="checkpoints/report.json")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("train")

    uuids = ds.list_uuids(LABELS)
    plan = ds.split_users(uuids, seed=args.seed)
    log.info("users: %d train / %d val / %d test", len(plan.train), len(plan.val), len(plan.test))

    sets = {}
    for name, group in plan.as_dict().items():
        t0 = time.time()
        minutes = ds.scan_all(group, LABELS, RAW)
        sampled = ds.sample_minutes(minutes, per_class_cap=args.cap, seed=args.seed,
                                    balance_users=not args.no_balance_users)
        ws = ds.build_windows(sampled, RAW)
        sets[name] = ws
        log.info(
            "%s: %d users, %d labeled minutes -> %d sampled -> %d windows (%.0fs)",
            name, len(group), len(minutes), len(sampled), len(ws), time.time() - t0,
        )
        log.info("   %s counts: %s", name, ws.class_counts())
        # user diversity per class: a class carried by few users cannot
        # generalise cross-subject, however many windows it contributes.
        div = {}
        for cls_i in np.unique(ws.y):
            cls = TARGET_CLASSES[cls_i]
            div[cls] = len(set(ws.uuids[ws.y == cls_i].tolist()))
        log.info("   %s distinct users per class: %s", name, div)

    norm = ds.fit_normaliser(sets["train"].X)
    cfg = TrainConfig(
        hidden=args.hidden, layers=args.layers, lr=args.lr,
        batch_size=args.batch_size, epochs=args.epochs, seed=args.seed,
    )
    model, result = train(sets["train"], sets["val"], norm, cfg)

    test_metrics = evaluate(
        model, norm.apply(sets["test"].X).astype(np.float32), sets["test"].y,
        classes=TARGET_CLASSES, absent=result.absent_classes,
    )

    print("\n=== TEST (held-out users) ===")
    print(f"accuracy  : {test_metrics['accuracy']:.4f}")
    print(f"macro F1  : {test_metrics['macro_f1']:.4f}")
    print(f"{'class':22} {'prec':>6} {'rec':>6} {'f1':>6} {'support':>8}")
    for c, m in test_metrics["per_class"].items():
        tag = "  (absent in train)" if c in result.absent_classes else ""
        print(f"{c:22} {m['precision']:6.3f} {m['recall']:6.3f} {m['f1']:6.3f} {m['support']:8d}{tag}")

    save_checkpoint(args.out, model, norm, result)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps({
        "split": {k: list(v) for k, v in plan.as_dict().items()},
        "train_counts": result.train_counts,
        "absent_classes": list(result.absent_classes),
        "best_epoch": result.best_epoch,
        "best_val_macro_f1": result.best_val_macro_f1,
        "history": result.history,
        "test": test_metrics,
    }, indent=1))
    print(f"\nsaved {args.out} and {args.report}")


if __name__ == "__main__":
    main()
