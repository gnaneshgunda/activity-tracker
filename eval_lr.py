"""Logistic Regression classifier on ExtraSensory pre-computed features.

Reads directly from .csv.gz files — no burst files, no GPU, no cluster.
Subject-wise train/val/test split (same seed as eval_local.py).
Prints per-class accuracy, F1, and macro-F1 for both LR and LR+HMM smoothing.

Usage:
    python3 eval_lr.py
"""
from __future__ import annotations
import gzip, csv, math, collections
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

LABELS_DIR = Path("ExtraSensory.per_uuid_features_labels")
SEED = 42
CLASSES = ["lying down", "sitting", "standing in place", "walking", "running", "bicycling"]
COL_MAP = {
    "label:LYING_DOWN":  "lying down",
    "label:SITTING":     "sitting",
    "label:OR_standing": "standing in place",
    "label:FIX_walking": "walking",
    "label:FIX_running": "running",
    "label:BICYCLING":   "bicycling",
}


def get_feature_cols(header):
    skip = {"timestamp", "label_source"}
    return [c for c in header
            if c not in skip and not c.startswith("label")]


def load_user(path, feat_cols):
    """Returns (X, y) arrays for one user. Skips rows with ambiguous labels."""
    X_rows, y_rows = [], []
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="") as fh:
        reader = csv.reader(fh)
        header = [h.strip() for h in next(reader)]
        col_idx = {h: i for i, h in enumerate(header)}
        feat_idx = [col_idx[c] for c in feat_cols if c in col_idx]
        for raw in reader:
            if not raw: continue
            row = raw
            pos = [cls for col, cls in COL_MAP.items()
                   if col in col_idx and row[col_idx[col]].strip() == "1"]
            if len(pos) != 1: continue
            vals = []
            for i in feat_idx:
                v = row[i].strip() if i < len(row) else ""
                try: vals.append(float(v))
                except: vals.append(float("nan"))
            X_rows.append(vals)
            y_rows.append(pos[0])
    return np.array(X_rows, dtype=np.float32), y_rows


def split_users(uuids, seed=42):
    rng = np.random.default_rng(seed)
    arr = np.array(sorted(uuids))
    rng.shuffle(arr)
    n = len(arr)
    t1, t2 = int(0.6 * n), int(0.8 * n)
    return arr[:t1].tolist(), arr[t1:t2].tolist(), arr[t2:].tolist()


def impute(X):
    """Column-wise median imputation for NaNs."""
    for j in range(X.shape[1]):
        col = X[:, j]
        mask = np.isfinite(col)
        if mask.sum() == 0: X[:, j] = 0.0
        else: X[~mask, j] = np.median(col[mask])
    return X


def macro_f1_report(y_true, y_pred, classes):
    conf = collections.defaultdict(collections.Counter)
    for t, p in zip(y_true, y_pred):
        conf[t][p] += 1
    f1s = []
    print(f"\n{'Class':<22} {'Acc%':>6}  {'F1':>5}  top-error")
    for cls in classes:
        total = sum(conf[cls].values())
        if total == 0:
            print(f"{cls:<22}  no data"); continue
        tp = conf[cls][cls]
        fp = sum(conf[o][cls] for o in classes if o != cls)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / total
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        f1s.append(f1)
        wrong = sorted([(v, k) for k, v in conf[cls].items() if k != cls], reverse=True)
        top = f"{wrong[0][1]} ({wrong[0][0]})" if wrong else "—"
        print(f"{cls:<22} {100*r:>5.1f}%  {f1:.3f}  {top}")
    mf1 = float(np.mean(f1s)) if f1s else 0.0
    print(f"\nmacro-F1: {mf1:.4f}")
    return mf1


def build_log_transition(label_seqs, classes):
    S = len(classes)
    idx = {c: i for i, c in enumerate(classes)}
    counts = np.zeros((S, S))
    for seq in label_seqs:
        for a, b in zip(seq, seq[1:]):
            if a in idx and b in idx:
                counts[idx[a], idx[b]] += 1
    prior = np.eye(S) * 30.0 + np.ones((S, S))
    m = counts + prior
    m = m / m.sum(axis=1, keepdims=True)
    return np.log(m)


def viterbi(log_probs, log_A):
    """Simple Viterbi decode. log_probs: (T, S), log_A: (S, S)."""
    T, S = log_probs.shape
    delta = log_probs[0] - math.log(S)
    back = np.zeros((T, S), dtype=int)
    for t in range(1, T):
        trans = delta[:, None] + log_A
        back[t] = np.argmax(trans, axis=0)
        delta = trans[back[t], np.arange(S)] + log_probs[t]
    path = np.empty(T, dtype=int)
    path[-1] = int(np.argmax(delta))
    for t in range(T - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    return path


def main():
    # ── discover feature columns from first file ───────────────────────────
    first = next(LABELS_DIR.glob("*.features_labels.csv*"))
    opener = gzip.open if str(first).endswith(".gz") else open
    with opener(first, "rt") as f:
        header = [h.strip() for h in next(csv.reader(f))]
    feat_cols = get_feature_cols(header)
    print(f"Feature columns: {len(feat_cols)}")

    uuids = [p.name.split(".")[0] for p in LABELS_DIR.glob("*.features_labels.csv*")]
    train_u, val_u, test_u = split_users(uuids)
    print(f"Split: train={len(train_u)} val={len(val_u)} test={len(test_u)}")

    # ── load train ─────────────────────────────────────────────────────────
    print("\nLoading train split...")
    X_train_list, y_train, train_seqs = [], [], []
    for uuid in train_u:
        path = next(LABELS_DIR.glob(f"{uuid}.features_labels.csv*"), None)
        if path is None: continue
        X, y = load_user(path, feat_cols)
        if len(y) == 0: continue
        X_train_list.append(X)
        y_train.extend(y)
        train_seqs.append(y)
    X_train = np.vstack(X_train_list)
    print(f"  {len(y_train)} rows, {X_train.shape[1]} features")
    counts = collections.Counter(y_train)
    for cls in CLASSES: print(f"    {cls:<22} {counts.get(cls, 0)}")

    # ── impute + scale ─────────────────────────────────────────────────────
    X_train = impute(X_train)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # ── train ──────────────────────────────────────────────────────────────
    print("\nTraining Logistic Regression...")
    clf = LogisticRegression(
        max_iter=1000, class_weight="balanced",
        solver="lbfgs", C=1.0, random_state=SEED
    )
    clf.fit(X_train, y_train)
    print("  Done.")

    # ── build HMM transition from train sequences ──────────────────────────
    log_A = build_log_transition(train_seqs, CLASSES)

    # ── load test ──────────────────────────────────────────────────────────
    print("\nLoading test split...")
    test_by_user = {}
    for uuid in test_u:
        path = next(LABELS_DIR.glob(f"{uuid}.features_labels.csv*"), None)
        if path is None: continue
        X, y = load_user(path, feat_cols)
        if len(y) == 0: continue
        test_by_user[uuid] = (X, y)
    total = sum(len(v[1]) for v in test_by_user.values())
    print(f"  {total} rows")

    # ── evaluate: LR only ─────────────────────────────────────────────────
    print("\n=== Logistic Regression only ===")
    y_true_all, y_pred_all = [], []
    for X, y in test_by_user.values():
        X = impute(X.copy())
        X = scaler.transform(X)
        preds = clf.predict(X)
        y_true_all.extend(y)
        y_pred_all.extend(preds)
    macro_f1_report(y_true_all, y_pred_all, CLASSES)

    # ── evaluate: LR + HMM Viterbi ────────────────────────────────────────
    print("\n=== Logistic Regression + HMM Viterbi smoothing ===")
    idx_map = {c: i for i, c in enumerate(CLASSES)}
    y_true_hmm, y_pred_hmm = [], []
    for X, y in test_by_user.values():
        X = impute(X.copy())
        X = scaler.transform(X)
        log_probs = np.log(np.clip(clf.predict_proba(X), 1e-12, None))
        # reorder columns to match CLASSES order
        clf_classes = list(clf.classes_)
        reorder = [clf_classes.index(c) if c in clf_classes else 0 for c in CLASSES]
        log_probs = log_probs[:, reorder]
        path = viterbi(log_probs, log_A)
        y_true_hmm.extend(y)
        y_pred_hmm.extend([CLASSES[i] for i in path])
    macro_f1_report(y_true_hmm, y_pred_hmm, CLASSES)


if __name__ == "__main__":
    main()
