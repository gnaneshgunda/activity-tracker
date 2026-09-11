"""End-to-end pipeline runner: CSV → preprocess → physics label → segment → store.

Takes raw accelerometer and (optionally) gyroscope CSV files, runs them through
the full B0→B1→B2→B3→B5 pipeline, and writes the results to a SQLite database
that the Streamlit app can open.

Usage
-----
    # Separate acc and gyro CSVs:
    python run_pipeline.py --acc acc.csv --gyro gyro.csv --db pipeline.db

    # Single combined CSV (columns: timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z):
    python run_pipeline.py --combined sensor_data.csv --db pipeline.db

    # Programmatic:
    from run_pipeline import run_pipeline
    db_path = run_pipeline("acc.csv", gyro_csv="gyro.csv")

CSV Format
----------
Accelerometer CSV: ``timestamp, x, y, z`` (or ``timestamp, acc_x, acc_y, acc_z``)
Gyroscope CSV:     ``timestamp, x, y, z`` (or ``timestamp, gyro_x, gyro_y, gyro_z``)
Combined CSV:      ``timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z``

Timestamps may be Unix seconds (float) or ISO-8601 strings.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# CSV parsing
# --------------------------------------------------------------------------

_ACC_X_NAMES = {"x", "acc_x", "accel_x", "ax"}
_ACC_Y_NAMES = {"y", "acc_y", "accel_y", "ay"}
_ACC_Z_NAMES = {"z", "acc_z", "accel_z", "az"}
_GYRO_X_NAMES = {"gyro_x", "gx", "gyr_x"}
_GYRO_Y_NAMES = {"gyro_y", "gy", "gyr_y"}
_GYRO_Z_NAMES = {"gyro_z", "gz", "gyr_z"}
_TIME_NAMES = {"timestamp", "time", "t", "epoch", "ts"}


def _find_col(header: list[str], candidates: set[str]) -> Optional[int]:
    """Find the first column in header matching any candidate name (case-insensitive)."""
    for i, h in enumerate(header):
        if h.strip().lower() in candidates:
            return i
    return None


def _parse_timestamp(val: str, first_ts: Optional[float] = None) -> float:
    """Parse a timestamp value — supports Unix seconds or ISO-8601."""
    try:
        return float(val)
    except ValueError:
        pass
    # Try ISO-8601
    import datetime as _dt
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = _dt.datetime.strptime(val, fmt)
            return dt.timestamp()
        except ValueError:
            continue
    raise ValueError(f"Cannot parse timestamp: {val!r}")


def _read_csv(path: str | Path) -> tuple[np.ndarray, list[str]]:
    """Read a CSV/DAT file, returning (rows, header_names).

    Handles:
    - Comma-separated CSV with a header row
    - Space/tab-separated .dat files with NO header (ExtraSensory raw format)
    Auto-detects delimiter and whether the first row is a header.
    """
    import csv as _csv

    path = Path(path)
    with open(path, "r", newline="") as f:
        raw = f.read()

    # Auto-detect delimiter: try comma first, fall back to whitespace
    first_line = raw.lstrip().split("\n")[0].strip()
    if "," in first_line:
        delimiter = ","
        lines = [l for l in raw.splitlines() if l.strip()]
        reader = _csv.reader(lines, delimiter=",")
        all_rows = [row for row in reader if row and any(c.strip() for c in row)]
    else:
        # Space/tab separated — split on whitespace
        all_rows = [line.split() for line in raw.splitlines()
                    if line.strip() and not line.strip().startswith("#")]

    if not all_rows:
        return [], []

    # Decide if first row is a header: if any cell is non-numeric, it's a header
    def _is_numeric(s: str) -> bool:
        try:
            float(s)
            return True
        except ValueError:
            return False

    first_row = all_rows[0]
    if all(_is_numeric(c.strip()) for c in first_row if c.strip()):
        # No header — synthesise column names: timestamp, x, y, z, ...
        n_cols = len(first_row)
        if n_cols == 4:
            header = ["timestamp", "x", "y", "z"]
        elif n_cols == 7:
            header = ["timestamp", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
        else:
            header = ["timestamp"] + [f"col{i}" for i in range(1, n_cols)]
        rows = all_rows
    else:
        header = [h.strip().lower() for h in first_row]
        rows = all_rows[1:]

    rows = [r for r in rows if r and any(c.strip() for c in r)]
    return rows, header


def parse_sensor_csvs(
    acc_csv: str | Path,
    gyro_csv: Optional[str | Path] = None,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Parse acc (and optional gyro) CSVs into numpy arrays.

    Returns
    -------
    acc_t : (n,) timestamps in Unix seconds
    acc_xyz : (n, 3) accelerometer readings
    gyro_t : (m,) timestamps or None
    gyro_xyz : (m, 3) gyroscope readings or None
    """
    rows, header = _read_csv(acc_csv)

    ti = _find_col(header, _TIME_NAMES)
    xi = _find_col(header, _ACC_X_NAMES)
    yi = _find_col(header, _ACC_Y_NAMES)
    zi = _find_col(header, _ACC_Z_NAMES)

    if ti is None:
        raise ValueError(f"No timestamp column found in {acc_csv}. Header: {header}")
    if xi is None or yi is None or zi is None:
        # Try combined CSV: check for gyro columns too
        gxi = _find_col(header, _GYRO_X_NAMES)
        if gxi is not None:
            return _parse_combined_csv(rows, header)
        raise ValueError(f"Cannot find acc x/y/z columns in {acc_csv}. Header: {header}")

    acc_t_list, acc_xyz_list = [], []
    for row in rows:
        try:
            t = _parse_timestamp(row[ti])
            x, y, z = float(row[xi]), float(row[yi]), float(row[zi])
            acc_t_list.append(t)
            acc_xyz_list.append([x, y, z])
        except (ValueError, IndexError):
            continue

    acc_t = np.array(acc_t_list, dtype=np.float64)
    acc_xyz = np.array(acc_xyz_list, dtype=np.float64)

    gyro_t, gyro_xyz = None, None
    if gyro_csv is not None:
        g_rows, g_header = _read_csv(gyro_csv)
        gti = _find_col(g_header, _TIME_NAMES)
        gxi = _find_col(g_header, _GYRO_X_NAMES) or _find_col(g_header, _ACC_X_NAMES)
        gyi = _find_col(g_header, _GYRO_Y_NAMES) or _find_col(g_header, _ACC_Y_NAMES)
        gzi = _find_col(g_header, _GYRO_Z_NAMES) or _find_col(g_header, _ACC_Z_NAMES)

        if gti is not None and gxi is not None and gyi is not None and gzi is not None:
            gt_list, gxyz_list = [], []
            for row in g_rows:
                try:
                    t = _parse_timestamp(row[gti])
                    x, y, z = float(row[gxi]), float(row[gyi]), float(row[gzi])
                    gt_list.append(t)
                    gxyz_list.append([x, y, z])
                except (ValueError, IndexError):
                    continue
            gyro_t = np.array(gt_list, dtype=np.float64)
            gyro_xyz = np.array(gxyz_list, dtype=np.float64)

    return acc_t, acc_xyz, gyro_t, gyro_xyz


def _parse_combined_csv(
    rows: list, header: list[str]
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Parse a combined acc+gyro CSV."""
    ti = _find_col(header, _TIME_NAMES)
    axi = _find_col(header, _ACC_X_NAMES)
    ayi = _find_col(header, _ACC_Y_NAMES)
    azi = _find_col(header, _ACC_Z_NAMES)
    gxi = _find_col(header, _GYRO_X_NAMES)
    gyi = _find_col(header, _GYRO_Y_NAMES)
    gzi = _find_col(header, _GYRO_Z_NAMES)

    if ti is None or axi is None or ayi is None or azi is None:
        raise ValueError(f"Cannot find acc columns in combined CSV. Header: {header}")

    acc_t_list, acc_xyz_list = [], []
    gyro_t_list, gyro_xyz_list = [], []
    has_gyro = gxi is not None and gyi is not None and gzi is not None

    for row in rows:
        try:
            t = _parse_timestamp(row[ti])
            ax, ay, az = float(row[axi]), float(row[ayi]), float(row[azi])
            acc_t_list.append(t)
            acc_xyz_list.append([ax, ay, az])
            if has_gyro:
                gx, gy, gz = float(row[gxi]), float(row[gyi]), float(row[gzi])
                if math.isfinite(gx) and math.isfinite(gy) and math.isfinite(gz):
                    gyro_t_list.append(t)
                    gyro_xyz_list.append([gx, gy, gz])
        except (ValueError, IndexError):
            continue

    acc_t = np.array(acc_t_list, dtype=np.float64)
    acc_xyz = np.array(acc_xyz_list, dtype=np.float64)
    gyro_t = np.array(gyro_t_list, dtype=np.float64) if gyro_t_list else None
    gyro_xyz = np.array(gyro_xyz_list, dtype=np.float64) if gyro_xyz_list else None

    return acc_t, acc_xyz, gyro_t, gyro_xyz


# --------------------------------------------------------------------------
# Feature bridge: Window → physics_rules.Features
# --------------------------------------------------------------------------

def _window_to_features(window) -> "Features":
    """Convert a pipeline Window to a physics_rules.Features dataclass."""
    from data.physics_rules import Features
    from data.ingest import PhonePlacement
    from data.preprocess import TARGET_RATE_HZ

    has_gyro = window.gyro_rms is not None
    gyro_rms = window.gyro_rms.magnitude if has_gyro else 0.0
    gyro_x = window.gyro_rms.x if has_gyro else 0.0
    gyro_y = window.gyro_rms.y if has_gyro else 0.0
    gyro_z = window.gyro_rms.z if has_gyro else 0.0

    vstd = window.vertical_std if math.isfinite(window.vertical_std) else 0.0
    sma = window.sma if math.isfinite(window.sma) else 0.0
    cad_hz = window.cadence_hz if math.isfinite(window.cadence_hz) else 0.0

    return Features(
        tilt_deg=90.0,  # Unknown without gravity reference orientation
        sma=sma,
        gyro_rms=gyro_rms,
        cadence_bpm=cad_hz * 60.0,
        periodicity=0.0,  # Would need autocorrelation; use cadence instead
        has_gyro=has_gyro,
        phone_placement=PhonePlacement.UNKNOWN,
        jerk_mean=0.0,
        jerk_std=0.0,
        dominant_freq_hz=0.0,
        gyro_x_rms=gyro_x,
        gyro_y_rms=gyro_y,
        gyro_z_rms=gyro_z,
        vertical_std=vstd,
        zcr=0.0,
        acc_gyro_phase=0.0,
    )


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------

def run_pipeline(
    acc_csv: str | Path,
    gyro_csv: Optional[str | Path] = None,
    *,
    db_path: str | Path = "pipeline.db",
    user_id: str = "user_01",
    progress_fn: Optional[Callable[[str, float], None]] = None,
) -> Path:
    """Run the full pipeline: CSV → preprocess → physics label → segment → store.

    Parameters
    ----------
    acc_csv : path to accelerometer CSV
    gyro_csv : optional path to gyroscope CSV
    db_path : output SQLite database path
    user_id : user identifier for provenance
    progress_fn : optional callback(step_name, fraction) for UI progress bars

    Returns
    -------
    Path to the populated SQLite database.
    """
    db_path = Path(db_path)

    def _progress(step: str, frac: float):
        if progress_fn:
            progress_fn(step, frac)
        log.info("Pipeline: %s (%.0f%%)", step, frac * 100)

    # ── Step 0: Parse CSVs ─────────────────────────────────────────────
    _progress("Parsing CSV files", 0.0)
    acc_t, acc_xyz, gyro_t, gyro_xyz = parse_sensor_csvs(acc_csv, gyro_csv)
    n_samples = len(acc_t)
    if n_samples < 10:
        raise ValueError(f"Too few samples ({n_samples}) in accelerometer CSV")

    # Detect sample rate from timestamps
    dt = np.diff(acc_t)
    dt_pos = dt[dt > 0]
    if len(dt_pos) == 0:
        raise ValueError("Timestamps are not increasing in accelerometer CSV")
    source_rate = 1.0 / float(np.median(dt_pos))
    log.info("Detected source rate: %.1f Hz, %d acc samples, gyro: %s",
             source_rate, n_samples, "yes" if gyro_t is not None else "no")

    _progress("Parsing CSV files", 1.0)

    # ── Step 1: Chunk into minute-sized bursts ─────────────────────────
    _progress("Chunking into bursts", 0.1)
    from data.preprocess import (
        PreprocessedSignal, TARGET_RATE_HZ, Gap,
        detect_accel_unit, resample_uniform, lowpass, estimate_gravity,
        LOWPASS_CUTOFF_HZ, GRAVITY_CUTOFF_HZ, FILTER_ORDER, MAX_INTERP_GAP_S,
    )

    # Split data into ~60-second chunks to match ExtraSensory burst geometry
    chunk_duration = 60.0  # seconds
    recording_start = float(acc_t[0])
    chunks = []
    chunk_start_idx = 0

    while chunk_start_idx < n_samples:
        t0 = acc_t[chunk_start_idx]
        # Find end of this chunk
        chunk_end_idx = chunk_start_idx
        while chunk_end_idx < n_samples and acc_t[chunk_end_idx] - t0 < chunk_duration:
            chunk_end_idx += 1
        if chunk_end_idx - chunk_start_idx < 10:
            break

        chunk_acc_t = acc_t[chunk_start_idx:chunk_end_idx]
        chunk_acc = acc_xyz[chunk_start_idx:chunk_end_idx]

        # Match gyro samples to this time range
        chunk_gyro_t, chunk_gyro = None, None
        if gyro_t is not None and gyro_xyz is not None:
            mask = (gyro_t >= chunk_acc_t[0]) & (gyro_t <= chunk_acc_t[-1])
            if mask.sum() >= 2:
                chunk_gyro_t = gyro_t[mask]
                chunk_gyro = gyro_xyz[mask]

        chunks.append((chunk_acc_t, chunk_acc, chunk_gyro_t, chunk_gyro))
        chunk_start_idx = chunk_end_idx

    log.info("Split into %d chunks", len(chunks))

    # ── Step 2: Preprocess each chunk ──────────────────────────────────
    _progress("Preprocessing", 0.2)
    from data.preprocess import preprocess_burst
    from data.ingest import MinuteBurst, SensorBurst

    signals: dict[int, PreprocessedSignal] = {}
    all_windows = []
    total_chunks = len(chunks)

    for ci, (c_acc_t, c_acc, c_gyro_t, c_gyro) in enumerate(chunks):
        _progress("Preprocessing", 0.2 + 0.3 * (ci / total_chunks))

        # Generate a UUID-like identifier
        chunk_ts = int(c_acc_t[0])
        uuid = user_id

        # Build SensorBurst objects
        acc_burst = SensorBurst(t=c_acc_t, xyz=c_acc)
        gyro_burst = None
        if c_gyro_t is not None and c_gyro is not None:
            gyro_burst = SensorBurst(t=c_gyro_t, xyz=c_gyro)

        burst = MinuteBurst(
            acc=acc_burst,
            gyro=gyro_burst,
        )

        try:
            signal = preprocess_burst(burst, uuid=uuid, timestamp=chunk_ts)
        except Exception as exc:
            log.warning("Chunk %d preprocess failed: %s", ci, exc)
            continue

        if signal.n_samples < 10:
            continue

        signals[chunk_ts] = signal

        # ── Step 3: Extract windows ────────────────────────────────────
        from pipeline.recognize import extract_windows
        windows = extract_windows(signal)
        if not windows:
            continue

        # ── Step 4: Physics scoring ────────────────────────────────────
        from data.physics_rules import Thresholds, predict_proba, make_scorer
        from pipeline.recognize import attach_probs

        th = Thresholds()

        def _scorer(window):
            feats = _window_to_features(window)
            probs = predict_proba(feats, th)
            return np.log(np.clip(probs, 1e-12, None))

        scored_windows = attach_probs(windows, _scorer)
        all_windows.extend(scored_windows)

    if not all_windows:
        raise ValueError("No usable windows after preprocessing. Check CSV data quality.")

    log.info("Total scored windows: %d", len(all_windows))

    # ── Step 5: Segment (HMM + Viterbi) ───────────────────────────────
    _progress("Segmenting (HMM Viterbi)", 0.6)
    from pipeline.segment import segment_windows

    segments = segment_windows(all_windows, signals=signals)
    log.info("Produced %d segments", len(segments))

    if not segments:
        raise ValueError("No segments produced. Data may be too short or too noisy.")

    # ── Step 6: Store in SQLite ────────────────────────────────────────
    _progress("Writing to database", 0.8)
    from analysis.store import ActivityStore, TimelineRow

    with ActivityStore(db_path) as store:
        for seg in segments:
            row = TimelineRow.from_segment(seg)
            sig = signals.get(int(seg.t_start), None)
            store.upsert_segment(row, raw_signal=sig)

    log.info("Pipeline complete → %s (%d segments)", db_path, len(segments))
    _progress("Complete", 1.0)

    return db_path


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Run the activity-tracker pipeline: CSV → SQLite",
        epilog="Example: python run_pipeline.py --acc acc.csv --gyro gyro.csv --db pipeline.db",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--acc", help="Path to accelerometer CSV")
    group.add_argument("--combined", help="Path to combined acc+gyro CSV")
    parser.add_argument("--gyro", help="Path to gyroscope CSV (with --acc)")
    parser.add_argument("--db", default="pipeline.db", help="Output SQLite path")
    parser.add_argument("--user-id", default="user_01", help="User identifier")

    args = parser.parse_args()

    if args.combined:
        acc_path = args.combined
        gyro_path = None
    else:
        acc_path = args.acc
        gyro_path = args.gyro

    db_path = run_pipeline(
        acc_csv=acc_path,
        gyro_csv=gyro_path,
        db_path=args.db,
        user_id=args.user_id,
    )
    print(f"\n✅ Pipeline complete → {db_path}")


if __name__ == "__main__":
    main()
