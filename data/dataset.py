"""Assemble training tensors for the B2 window classifier.

Turns labeled ExtraSensory minutes into ``(n_windows, T, C)`` float arrays plus
integer class targets, ready for :mod:`lstm`.

Two things this module refuses to do, because both would produce dishonest
accuracy numbers:

*Subject leakage.* Splits are **by user**, never by window. Windows from one
burst overlap by 50 % and windows from one user share a device, a gait and a
carrying position; splitting at the window level would put near-duplicates on
both sides and inflate every metric. :func:`split_users` partitions UUIDs.

*Normalising on everything.* Channel statistics are computed on the training
split alone and applied to val/test, so no test-set information reaches the
model. See :func:`fit_normaliser`.

Sampling
--------
The full label set is ~306 k labeled minutes, which at ~22 windows per burst is
~6.7 M windows -- far more than fits comfortably on this machine. Minutes are
therefore sampled *before* any signal processing happens, with a per-class cap,
which also directly addresses the 125:1 class imbalance (sitting 44.5 % vs
running 0.4 %). Sampling minutes rather than windows keeps every burst intact.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from data.ingest import (
    TARGET_CLASSES,
    MinuteBurst,
    PhonePlacement,
    PHONE_PLACEMENT_COLUMNS,
    burst_paths,
    iter_label_rows,
    labels_path,
    load_burst_file,
    read_phone_placement,
    resolve_label,
)
from data.preprocess import preprocess_burst
from pipeline.recognize import WINDOW_S, cadence_from_vertical, extract_windows, vertical_component, dominant_freq_hz, acc_gyro_phase

__all__ = [
    "CHANNELS",
    "LabeledMinute",
    "SplitPlan",
    "WindowSet",
    "Normaliser",
    "scan_labels",
    "split_users",
    "sample_minutes",
    "build_windows",
    "fit_normaliser",
    "list_uuids",
]

log = logging.getLogger(__name__)

LABELS = "ExtraSensory.per_uuid_features_labels"
LABELS_ROOT_SENTINEL = Path(LABELS)  # default labels root, relative to cwd

#: Input channels, in order.
#: Raw sensor channels (10) + derived physics features (12) + phone placement flags (4).
#: Derived: tilt_deg, sma, cadence_hz, periodicity, vertical_std, rotation_ratio,
#:          jerk_mean, gyro_x_rms, gyro_y_rms, gyro_z_rms, dominant_freq_hz, acc_gyro_phase.
#: Placement: phone_pocket, phone_hand, phone_bag, phone_table (0/1 per timestep).
CHANNELS: tuple[str, ...] = (
    # raw sensor
    "body_acc_x", "body_acc_y", "body_acc_z",
    "gravity_x", "gravity_y", "gravity_z",
    "gyro_x", "gyro_y", "gyro_z",
    "gyro_present",
    # derived physics features (broadcast as constant per window)
    "tilt_deg",
    "sma",
    "cadence_hz",
    "periodicity",
    "vertical_std",
    "rotation_ratio",
    "jerk_mean",
    "gyro_x_rms",
    "gyro_y_rms",
    "gyro_z_rms",
    "dominant_freq_hz",
    "acc_gyro_phase",
    # phone placement flags (broadcast as constant per window)
    "phone_pocket",
    "phone_hand",
    "phone_bag",
    "phone_table",
)

N_CHANNELS = len(CHANNELS)


@dataclass(frozen=True)
class LabeledMinute:
    uuid: str
    timestamp: int
    label: str
    label_index: int


@dataclass(frozen=True)
class SplitPlan:
    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]

    def as_dict(self) -> dict[str, tuple[str, ...]]:
        return {"train": self.train, "val": self.val, "test": self.test}


@dataclass
class WindowSet:
    """Windows for one split."""

    X: np.ndarray           # (n, T, C) float32
    y: np.ndarray           # (n,) int64
    uuids: np.ndarray       # (n,) object -- for per-subject reporting
    timestamps: np.ndarray  # (n,) int64
    classes: tuple[str, ...] = TARGET_CLASSES

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def class_counts(self) -> dict[str, int]:
        c = np.bincount(self.y, minlength=len(self.classes))
        return {self.classes[i]: int(c[i]) for i in range(len(self.classes))}


@dataclass(frozen=True)
class Normaliser:
    """Per-channel mean/std, fitted on the training split only."""

    mean: np.ndarray
    std: np.ndarray

    def apply(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def save(self, path: os.PathLike | str) -> None:
        np.savez(path, mean=self.mean, std=self.std)

    @staticmethod
    def load(path: os.PathLike | str) -> "Normaliser":
        d = np.load(path)
        return Normaliser(mean=d["mean"], std=d["std"])


# --------------------------------------------------------------------------
# Label scanning (no signal processing -- deliberately cheap)
# --------------------------------------------------------------------------


def list_uuids(labels_root: os.PathLike | str) -> list[str]:
    """UUIDs that have a label file, compressed or not."""
    root = Path(labels_root)
    seen: set[str] = set()
    for p in root.iterdir():
        name = p.name
        for suf in (".features_labels.csv.gz", ".features_labels.csv"):
            if name.endswith(suf):
                seen.add(name[: -len(suf)])
    return sorted(seen)


def scan_labels(
    uuid: str,
    labels_root: os.PathLike | str,
    raw_root: os.PathLike | str,
    *,
    standing_split: str = "flag",
    require_burst: bool = True,
) -> list[LabeledMinute]:
    """Resolve one user's labeled minutes without loading any burst data.

    Only minutes that resolve to exactly one class are returned; ambiguous and
    all-missing minutes are dropped here because they cannot supply a target.
    """
    path = labels_path(labels_root, uuid)
    if not path.is_file():
        return []

    out: list[LabeledMinute] = []
    for row in iter_label_rows(path):
        try:
            ts = int(float(row.get("timestamp", "")))
        except (TypeError, ValueError):
            continue
        res = resolve_label(row, standing_split=standing_split)
        if res.label is None or not res.in_subset:
            continue
        if require_burst and not burst_paths(raw_root, uuid, ts)[0].is_file():
            continue
        out.append(
            LabeledMinute(
                uuid=uuid, timestamp=ts, label=res.label,
                label_index=TARGET_CLASSES.index(res.label),
            )
        )
    return out


def _scan_one(args) -> list[LabeledMinute]:
    return scan_labels(*args[:3], standing_split=args[3])


def scan_all(
    uuids: Sequence[str],
    labels_root: os.PathLike | str,
    raw_root: os.PathLike | str,
    *,
    standing_split: str = "flag",
    workers: Optional[int] = None,
) -> list[LabeledMinute]:
    """Scan many users in parallel."""
    workers = workers or min(16, (os.cpu_count() or 4))
    args = [(u, str(labels_root), str(raw_root), standing_split) for u in uuids]
    out: list[LabeledMinute] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for res in pool.map(_scan_one, args, chunksize=1):
            out.extend(res)
    return out


# --------------------------------------------------------------------------
# Splitting and sampling
# --------------------------------------------------------------------------


def split_users(
    uuids: Sequence[str],
    *,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    seed: int = 0,
) -> SplitPlan:
    """Partition users into train/val/test. Subject-wise, never window-wise."""
    if not 0 <= val_frac + test_frac < 1:
        raise ValueError("val_frac + test_frac must be in [0, 1)")
    rng = np.random.default_rng(seed)
    shuffled = list(uuids)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    test = shuffled[:n_test]
    val = shuffled[n_test : n_test + n_val]
    train = shuffled[n_test + n_val :]
    return SplitPlan(train=tuple(sorted(train)), val=tuple(sorted(val)), test=tuple(sorted(test)))


def sample_minutes(
    minutes: Sequence[LabeledMinute],
    *,
    per_class_cap: Optional[int] = None,
    seed: int = 0,
    balance_users: bool = True,
) -> list[LabeledMinute]:
    """Cap each class, sampling without replacement.

    Applied per split, so the cap never moves a minute across a split boundary.

    With ``balance_users`` (the default) each class's budget is filled
    **round-robin across the users who have that class**, rather than drawn
    uniformly from the pooled minutes. This matters: activity minutes are very
    unevenly distributed between subjects -- one user contributes 26% of all
    running in the dataset -- so pooled sampling quietly turns a rare class into
    "that one person's gait" and the model cannot generalise to new users.
    Round-robin spends the budget on as many distinct users as possible, and
    only takes a second minute from a user once every other user has had one.
    """
    if per_class_cap is None:
        return list(minutes)
    rng = np.random.default_rng(seed)

    by_class: dict[int, list[LabeledMinute]] = {}
    for m in minutes:
        by_class.setdefault(m.label_index, []).append(m)

    out: list[LabeledMinute] = []
    for _, group in sorted(by_class.items()):
        if len(group) <= per_class_cap:
            out.extend(group)
            continue
        if not balance_users:
            idx = rng.choice(len(group), size=per_class_cap, replace=False)
            out.extend(group[i] for i in idx)
            continue

        by_user: dict[str, list[LabeledMinute]] = {}
        for m in group:
            by_user.setdefault(m.uuid, []).append(m)
        for lst in by_user.values():
            rng.shuffle(lst)

        users = sorted(by_user)
        rng.shuffle(users)
        picked: list[LabeledMinute] = []
        cursor = {u: 0 for u in users}
        while len(picked) < per_class_cap:
            progressed = False
            for u in users:
                if len(picked) >= per_class_cap:
                    break
                i = cursor[u]
                if i < len(by_user[u]):
                    picked.append(by_user[u][i])
                    cursor[u] = i + 1
                    progressed = True
            if not progressed:
                break
        out.extend(picked)

    out.sort(key=lambda m: (m.uuid, m.timestamp))
    return out


# --------------------------------------------------------------------------
# Window construction
# --------------------------------------------------------------------------


def _placement_flags(placement: str) -> tuple[float, float, float, float]:
    """Return (pocket, hand, bag, table) one-hot from a PhonePlacement value."""
    return (
        1.0 if placement == PhonePlacement.POCKET else 0.0,
        1.0 if placement == PhonePlacement.HAND   else 0.0,
        1.0 if placement == PhonePlacement.BAG    else 0.0,
        1.0 if placement == PhonePlacement.TABLE  else 0.0,
    )


# Per-class window stride: minority classes get high overlap to produce more
# windows per burst without generating fake signals.
# majority (lying/sitting): 0% overlap = 2.0s stride
# moderate (walking/standing): 50% overlap = 1.0s stride
# minority (running/bicycling/standing-and-moving): 90% overlap = 0.2s stride
_CLASS_STRIDE_S: dict[int, float] = {
    TARGET_CLASSES.index("lying down"):          2.0,
    TARGET_CLASSES.index("sitting"):             2.0,
    TARGET_CLASSES.index("standing in place"):   1.0,
    TARGET_CLASSES.index("standing and moving"): 0.2,
    TARGET_CLASSES.index("walking"):             1.0,
    TARGET_CLASSES.index("running"):             0.2,
    TARGET_CLASSES.index("bicycling"):           0.2,
}


def _minute_windows(args) -> Optional[tuple[np.ndarray, int, str, int]]:
    uuid, ts, label_index, raw_root, window_s, placement = args
    acc_p, gyro_p = burst_paths(raw_root, uuid, ts)
    if not acc_p.is_file():
        return None
    try:
        acc = load_burst_file(acc_p)
        gyro = load_burst_file(gyro_p) if gyro_p.is_file() else None
    except (ValueError, OSError):
        return None

    sig = preprocess_burst(MinuteBurst(acc=acc, gyro=gyro), uuid=uuid, timestamp=ts)
    if sig.n_samples == 0 or "unit_ambiguous" in sig.flags:
        return None

    wins = extract_windows(sig, window_s=window_s,
                            overlap=1.0 - _CLASS_STRIDE_S[label_index] / window_s)
    n_win = int(round(window_s * sig.fs))
    pl_flags = np.array(_placement_flags(placement), dtype=np.float32)  # (4,)
    vertical_all = vertical_component(sig.body_acc, sig.gravity)
    keep: list[np.ndarray] = []
    for w in wins:
        if not w.is_usable:
            continue
        i = int(round(w.t_start_s * sig.fs))
        if i + n_win > sig.n_samples:
            continue
        body = sig.body_acc[i : i + n_win]
        grav = sig.gravity[i : i + n_win]
        vert = vertical_all[i : i + n_win]

        if not np.all(np.isfinite(body)) or not np.all(np.isfinite(grav)):
            continue

        if sig.gyro_raw is None:
            gy = np.zeros((n_win, 3), dtype=np.float64)
            present = np.zeros((n_win, 1), dtype=np.float64)
        else:
            gy = sig.gyro_raw[i : i + n_win]
            ok = np.all(np.isfinite(gy), axis=1)
            present = ok.astype(np.float64)[:, None]
            gy = np.where(ok[:, None], gy, 0.0)

        # --- derived physics features (scalar per window, broadcast) ---
        sma = float(np.nanmean(np.linalg.norm(body, axis=1)))
        g_mean = np.nanmean(grav, axis=0)
        g_norm = float(np.linalg.norm(g_mean))
        if g_norm > 1e-9:
            cos = float(np.dot(g_mean / g_norm, np.array([0.0, 0.0, -1.0])))
            tilt_deg = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
        else:
            tilt_deg = 90.0
        cadence_hz, _, _ = cadence_from_vertical(vert, fs=sig.fs, sma=sma)
        cadence_hz = 0.0 if not np.isfinite(cadence_hz) else cadence_hz
        fin_vert = vert[np.isfinite(vert)]
        periodicity = float(np.std(fin_vert)) if fin_vert.size > 1 else 0.0
        vertical_std = periodicity
        gyro_rms = float(np.sqrt(np.nanmean(np.square(gy)))) if sig.gyro_raw is not None else 0.0
        rotation_ratio = gyro_rms / (sma + 1e-6)

        # jerk: mean magnitude of frame-to-frame acceleration derivative
        diff_body = np.diff(body, axis=0) * sig.fs  # (T-1, 3) in g/s
        jerk_mean = float(np.nanmean(np.linalg.norm(diff_body, axis=1))) if diff_body.size > 0 else 0.0

        # axis-resolved gyro RMS
        gyro_x_rms = float(np.sqrt(np.nanmean(np.square(gy[:, 0])))) if sig.gyro_raw is not None else 0.0
        gyro_y_rms = float(np.sqrt(np.nanmean(np.square(gy[:, 1])))) if sig.gyro_raw is not None else 0.0
        gyro_z_rms = float(np.sqrt(np.nanmean(np.square(gy[:, 2])))) if sig.gyro_raw is not None else 0.0

        # FFT dominant frequency and acc-gyro phase alignment
        dom_freq = dominant_freq_hz(vert, fs=sig.fs)
        phase = acc_gyro_phase(vert, gy[:, 1], fs=sig.fs) if sig.gyro_raw is not None else 0.0

        derived = np.array(
            [tilt_deg, sma, cadence_hz, periodicity, vertical_std, rotation_ratio,
             jerk_mean, gyro_x_rms, gyro_y_rms, gyro_z_rms, dom_freq, phase],
            dtype=np.float32,
        )  # (12,)

        # broadcast scalars across timesteps
        derived_block = np.tile(derived, (n_win, 1))       # (T, 6)
        placement_block = np.tile(pl_flags, (n_win, 1))    # (T, 4)

        block = np.concatenate([body, grav, gy, present, derived_block, placement_block], axis=1)
        keep.append(block.astype(np.float32))

    if not keep:
        return None
    return np.stack(keep), label_index, uuid, ts


def build_windows(
    minutes: Sequence[LabeledMinute],
    raw_root: os.PathLike | str,
    *,
    labels_root: os.PathLike | str = LABELS,
    window_s: float = WINDOW_S,
    workers: Optional[int] = None,
) -> WindowSet:
    """Preprocess the selected minutes and stack their windows.

    Minutes whose burst is missing, unreadable, unit-ambiguous or entirely
    gap-filled are skipped -- nothing is imputed to keep a row.
    """
    workers = workers or min(16, (os.cpu_count() or 4))
    labels_root_path = Path(labels_root)
    # read placement from label file for each minute
    placement_map: dict[tuple[str, int], str] = {}
    for m in minutes:
        lpath = labels_path(labels_root_path, m.uuid)
        if lpath.is_file():
            for row in iter_label_rows(lpath):
                try:
                    rts = int(float(row.get("timestamp", "")))
                except (TypeError, ValueError):
                    continue
                if rts == m.timestamp:
                    placement_map[(m.uuid, m.timestamp)] = read_phone_placement(row)
                    break

    args = [
        (m.uuid, m.timestamp, m.label_index, str(raw_root), window_s,
         placement_map.get((m.uuid, m.timestamp), PhonePlacement.UNKNOWN))
        for m in minutes
    ]

    Xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    us: list[np.ndarray] = []
    ts_: list[np.ndarray] = []

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for res in pool.map(_minute_windows, args, chunksize=32):
            if res is None:
                continue
            block, label_index, uuid, ts = res
            Xs.append(block)
            n = block.shape[0]
            ys.append(np.full(n, label_index, dtype=np.int64))
            us.append(np.full(n, uuid, dtype=object))
            ts_.append(np.full(n, ts, dtype=np.int64))

    if not Xs:
        return WindowSet(
            X=np.zeros((0, 0, N_CHANNELS), np.float32),
            y=np.zeros(0, np.int64),
            uuids=np.zeros(0, dtype=object),
            timestamps=np.zeros(0, np.int64),
        )

    return WindowSet(
        X=np.concatenate(Xs).astype(np.float32),
        y=np.concatenate(ys),
        uuids=np.concatenate(us),
        timestamps=np.concatenate(ts_),
    )


#: Fixed sequence length for per-burst classification (see build_burst_sequences).
BURST_TIMESTEPS = 400

#: Channels for per-burst sequences: the window channels plus a padding mask.
SEQ_CHANNELS: tuple[str, ...] = CHANNELS + ("sample_valid",)


#: Channels that are 0/1 indicators, not measurements. They must survive
#: normalisation unchanged: downstream code tests them with ``> 0.5``, and
#: standardising a mostly-1 flag maps 1 to ~0.25, silently inverting the test.
FLAG_CHANNELS: tuple[str, ...] = (
    "gyro_present", "sample_valid",
    "phone_pocket", "phone_hand", "phone_bag", "phone_table",
)


def fit_normaliser(X: np.ndarray, *, eps: float = 1e-6) -> Normaliser:
    """Per-channel mean/std over ``(n, T, C)``. Fit on the training split only.

    Indicator channels (:data:`FLAG_CHANNELS`) are left untouched (mean 0,
    std 1) so they stay interpretable 0/1 flags.
    """
    mean = X.reshape(-1, X.shape[-1]).mean(axis=0)
    std = X.reshape(-1, X.shape[-1]).std(axis=0)
    std = np.where(std < eps, 1.0, std)
    n_ch = X.shape[-1]
    names = SEQ_CHANNELS if n_ch == len(SEQ_CHANNELS) else CHANNELS
    for name in FLAG_CHANNELS:
        if name in names:
            i = names.index(name)
            if i < n_ch:
                mean[i], std[i] = 0.0, 1.0
    return Normaliser(mean=mean.astype(np.float32), std=std.astype(np.float32))


# --------------------------------------------------------------------------
# Per-burst sequences
# --------------------------------------------------------------------------

def _burst_sequence(args) -> Optional[tuple[np.ndarray, int, str, int]]:
    uuid, ts, label_index, raw_root, timesteps = args
    acc_p, gyro_p = burst_paths(raw_root, uuid, ts)
    if not acc_p.is_file():
        return None
    try:
        acc = load_burst_file(acc_p)
        gyro = load_burst_file(gyro_p) if gyro_p.is_file() else None
    except (ValueError, OSError):
        return None

    sig = preprocess_burst(MinuteBurst(acc=acc, gyro=gyro), uuid=uuid, timestamp=ts)
    if sig.n_samples < 25 or "unit_ambiguous" in sig.flags:
        return None

    n = min(sig.n_samples, timesteps)
    body = sig.body_acc[:n]
    grav = sig.gravity[:n]
    if sig.gyro_raw is None:
        gy = np.zeros((n, 3)); present = np.zeros((n, 1))
    else:
        gy = sig.gyro_raw[:n]
        ok = np.all(np.isfinite(gy), axis=1)
        present = ok.astype(np.float64)[:, None]
        gy = np.where(ok[:, None], gy, 0.0)

    valid = np.all(np.isfinite(body), axis=1) & np.all(np.isfinite(grav), axis=1)
    if valid.sum() < 25:
        return None
    body = np.where(valid[:, None], body, 0.0)
    grav = np.where(valid[:, None], grav, 0.0)

    # ---- derived physics features (scalar per burst, broadcast to all timesteps) ----
    import math
    vertical = vertical_component(body, grav)
    sma = float(np.nanmean(np.linalg.norm(body, axis=1)))

    # tilt: angle between mean gravity and screen-up reference (0,0,-1)
    g_mean = np.nanmean(grav, axis=0)
    g_norm = float(np.linalg.norm(g_mean))
    if g_norm > 1e-9:
        cos_t = float(np.dot(g_mean / g_norm, [0.0, 0.0, -1.0]))
        tilt_deg = float(math.degrees(math.acos(max(-1.0, min(1.0, cos_t)))))
    else:
        tilt_deg = 90.0

    fin_vert = vertical[np.isfinite(vertical)]
    vert_std = float(np.nanstd(fin_vert)) if fin_vert.size > 1 else 0.0
    cadence_hz, _, _ = cadence_from_vertical(vertical, fs=sig.fs, sma=sma)
    cadence_hz = float(cadence_hz) if math.isfinite(cadence_hz) else 0.0

    from data.physics_rules import periodicity_from_vertical
    periodicity = float(periodicity_from_vertical(vertical, sig.fs))

    # jerk mean
    diff_body = np.diff(body, axis=0) * sig.fs
    fin_jerk = np.linalg.norm(diff_body, axis=1)
    jerk_mean = float(np.nanmean(fin_jerk)) if fin_jerk.size > 0 else 0.0

    # per-axis gyro RMS
    gyro_x_rms = float(np.sqrt(np.nanmean(np.square(gy[:, 0])))) if sig.has_gyro else 0.0
    gyro_y_rms = float(np.sqrt(np.nanmean(np.square(gy[:, 1])))) if sig.has_gyro else 0.0
    gyro_z_rms = float(np.sqrt(np.nanmean(np.square(gy[:, 2])))) if sig.has_gyro else 0.0
    gyro_vec_rms = math.sqrt(gyro_x_rms**2 + gyro_y_rms**2 + gyro_z_rms**2)

    # rotation_ratio = gyro_rms / (sma + eps)
    rotation_ratio = float(gyro_vec_rms / (sma + 1e-6))

    # dominant freq and acc_gyro_phase
    dom_hz = float(dominant_freq_hz(vertical, fs=sig.fs))
    if sig.has_gyro and gyro_vec_rms > 1e-9:
        rms_axes = np.array([gyro_x_rms, gyro_y_rms, gyro_z_rms])
        dom_ax = int(np.argmax(rms_axes))
        phase = float(acc_gyro_phase(vertical, gy[:, dom_ax], fs=sig.fs))
    else:
        phase = 0.0

    # phone placement (all zeros if not in label file — unknown placement)
    phys_features = np.array([
        tilt_deg / 90.0,   # normalise to ~[-2, 2] range
        sma,
        cadence_hz,
        periodicity,
        vert_std,
        rotation_ratio,
        jerk_mean,
        gyro_x_rms,
        gyro_y_rms,
        gyro_z_rms,
        dom_hz,
        phase,
        0.0, 0.0, 0.0, 0.0,  # phone placement flags (unknown)
    ], dtype=np.float64)  # 16 values = 12 physics + 4 placement

    # broadcast physics features as constant columns across all timesteps
    phys_block = np.tile(phys_features[None, :], (n, 1))  # (n, 16)

    block = np.concatenate([
        body, grav, gy, present,   # 10 raw channels
        phys_block,                # 16 derived + placement channels
        valid.astype(np.float64)[:, None],  # sample_valid (1)
    ], axis=1)  # total: 27 = len(SEQ_CHANNELS)

    if n < timesteps:
        block = np.vstack([block, np.zeros((timesteps - n, block.shape[1]))])
    return block.astype(np.float32), label_index, uuid, ts


def build_burst_sequences(
    minutes: Sequence[LabeledMinute],
    raw_root: os.PathLike | str,
    *,
    timesteps: int = BURST_TIMESTEPS,
    workers: Optional[int] = None,
) -> WindowSet:
    """One fixed-length sequence per labeled burst -- **one label, one sample**.

    This is the granularity ExtraSensory actually labels at: the released
    examples are ~20 s of sensor data carrying one label set. Slicing a burst
    into 2 s windows and copying its label onto all ~18 of them invents
    supervision that was never collected (a "walking" burst contains moments of
    standing still), and because those windows overlap 50 % it also inflates the
    apparent sample count roughly 18-fold without adding information.
    """
    workers = workers or min(16, (os.cpu_count() or 4))
    args = [(m.uuid, m.timestamp, m.label_index, str(raw_root), timesteps) for m in minutes]
    Xs, ys, us, tss = [], [], [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for res in pool.map(_burst_sequence, args, chunksize=32):
            if res is None:
                continue
            block, li, uuid, ts = res
            Xs.append(block); ys.append(li); us.append(uuid); tss.append(ts)
    if not Xs:
        return WindowSet(
            X=np.zeros((0, timesteps, len(SEQ_CHANNELS)), np.float32),
            y=np.zeros(0, np.int64), uuids=np.zeros(0, dtype=object),
            timestamps=np.zeros(0, np.int64),
        )
    return WindowSet(
        X=np.stack(Xs), y=np.array(ys, dtype=np.int64),
        uuids=np.array(us, dtype=object), timestamps=np.array(tss, dtype=np.int64),
    )


def drop_classes(ws: WindowSet, classes_to_drop: Sequence[str]) -> WindowSet:
    """Remove samples of the named classes. Class indices are preserved."""
    if not classes_to_drop:
        return ws
    drop = {ws.classes.index(c) for c in classes_to_drop}
    keep = ~np.isin(ws.y, list(drop))
    return WindowSet(X=ws.X[keep], y=ws.y[keep], uuids=ws.uuids[keep],
                     timestamps=ws.timestamps[keep], classes=ws.classes)
