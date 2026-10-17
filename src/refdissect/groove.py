"""How the parts lock to the kick: the checks a producer hears first.

A rebuilt beat can match a reference's sounds, notes and loudness and still
sound unrelated to its drums. On HAYIR DEMELIYDIM (Motive) the first rebuild
failed by ear for three reasons none of the other measures saw:

- **ducking** - the reference's 808 falls ~20 dB within 20 ms of every kick and
  ~45 dB by 200 ms, back by ~500 ms; its chords fall ~13 dB and are back by
  270 ms. The rebuild did not move at all (`kick_curve`).
- **drum balance** - matched as one group, the hats and percussion came out as
  loud as kick and clap; in the reference they sit 19 and 31 dB under them,
  each measured in its own band (`hit_levels`, `role_offsets`).
- **one-shot length** - notes one 16th long made Simpler cut a 280 ms kick at
  130 ms (`cut_one_shots`).
"""

from __future__ import annotations

import numpy as np
import scipy.signal as ss

from .audio import SR

CURVE_POINTS = (0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)


def band_envelope(x: np.ndarray, lo: float, hi: float, window_ms: float = 5.0, sr: int = SR) -> np.ndarray:
    """Amplitude envelope (linear) of x in the lo-hi band, one value per sample:
    the analytic signal's magnitude, smoothed over window_ms. (An RMS over 5 ms
    follows a 60 Hz wave itself, not its level.)"""
    from scipy.ndimage import uniform_filter1d
    sos = ss.butter(4, (lo, hi), btype="band", fs=sr, output="sos")
    y = np.abs(ss.hilbert(ss.sosfiltfilt(sos, np.asarray(x, dtype=np.float64))))
    w = max(1, int(window_ms / 1000 * sr))
    return np.maximum(uniform_filter1d(y, w, mode="constant"), 0)


def kick_curve(env: np.ndarray, kicks: list[float], pre: float = 0.05, post: float = 0.6,
               sr: int = SR) -> dict:
    """The median level around the kicks, in dB re the 100-20 ms before each.

    Returns how deep it falls, when it is lowest, when it is back within 3 dB,
    and the curve at CURVE_POINTS (seconds after the kick)."""
    a, b = int(pre * sr), int(post * sr)
    rows = []
    for t in kicks:
        k = int(round(t * sr))
        if k - int(0.1 * sr) < 0 or k + b > len(env):
            continue
        ref = float(np.median(env[k - int(0.1 * sr):k - int(0.02 * sr)]))
        if ref <= 0:
            continue
        rows.append(20 * np.log10(env[k - a:k + b] / ref + 1e-9))
    if not rows:
        raise ValueError("No kick has 100 ms before it and the curve's length after it")
    c = np.median(rows, axis=0)
    t = (np.arange(len(c)) - a) / sr
    low = int(np.argmin(np.where(t >= 0, c, np.inf)))
    back = next((float(t[i]) for i in range(low, len(c)) if c[i] >= -3), None)
    return {"kicks": len(rows), "depth_db": round(float(c[low]), 1), "lowest_at_s": round(float(t[low]), 3),
            "back_to_-3db_s": None if back is None else round(back, 3),
            "curve_db": {f"{p:.2f}": round(float(np.interp(p, t, c)), 1) for p in CURVE_POINTS}}


def curve_distance(a: dict, b: dict, points=CURVE_POINTS[1:]) -> float:
    """Mean absolute dB between two kick curves (from kick_curve) at points."""
    return round(float(np.mean([abs(a["curve_db"][f"{p:.2f}"] - b["curve_db"][f"{p:.2f}"]) for p in points])), 2)


def hit_levels(x: np.ndarray, times: list[float], lo: float, hi: float, window_ms: float = 30.0,
               sr: int = SR) -> dict:
    """Peak level (dBFS) of each hit within window_ms of its time, in its band;
    the median and the spread over the hits."""
    sos = ss.butter(4, (lo, hi), btype="band", fs=sr, output="sos")
    y = np.abs(ss.sosfiltfilt(sos, np.asarray(x, dtype=np.float64)))
    w = int(window_ms / 1000 * sr)
    peaks = [float(y[int(t * sr):int(t * sr) + w].max()) for t in times if 0 <= int(t * sr) < len(y) - w]
    db = 20 * np.log10(np.array(peaks) + 1e-12)
    if not len(db):
        return {"hits": 0, "median_db": None, "iqr_db": None}
    q1, q3 = np.percentile(db, [25, 75])
    return {"hits": len(db), "median_db": round(float(np.median(db)), 1), "iqr_db": round(float(q3 - q1), 1)}


def role_offsets(levels: dict[str, dict]) -> dict[str, float]:
    """Each role's median hit level re the loudest role (0 dB), from hit_levels."""
    med = {k: v["median_db"] for k, v in levels.items() if v.get("median_db") is not None}
    top = max(med.values())
    return {k: round(v - top, 1) for k, v in med.items()}


def balance_moves(reference: dict[str, float], rebuild: dict[str, float]) -> dict[str, float]:
    """dB to move each role's fader so its offset (role_offsets) matches the
    reference's, keeping the loudest reference role where it is."""
    anchor = max(reference, key=reference.get)
    shift = rebuild[anchor] - reference[anchor]
    return {k: round(reference[k] - (rebuild[k] - shift), 1) for k in reference if k in rebuild}


def cut_one_shots(notes: list[dict], sample_seconds: float, bpm: float) -> list[dict]:
    """The notes shorter than their one-shot: Simpler (one-shot off) ends the
    sound with the note. notes: {start_time, duration} in beats."""
    need = sample_seconds * bpm / 60
    return [n for n in notes if n["duration"] + 1e-6 < need]
