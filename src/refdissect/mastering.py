"""Mastering a rebuild the way its reference is mastered, from measurements
of the reference alone: its integrated loudness, its spectral tilt and its
peak ceiling.

A song reaches the pipeline mixed, mastered and often encoded; a rebuild's
render does not. Separated and measured as it is, the rebuild would differ
from the reference in level, brightness and density before any sound was
wrong. master_like() moves only those three things:

- tilt: the slope of the long-term spectrum in dB per octave, fitted over
  third-octave band levels from 50 Hz to 16 kHz (one parameter, not a full
  EQ match: a match would also copy the reference's balance between sounds
  onto the rebuild and hide what is wrong with it);
- loudness: BS.1770 integrated LUFS (loudness.integrated_lufs);
- limiting: a look-ahead peak limiter that holds the reference's sample
  peak, its gain brought down over `lookahead_ms` before a peak and back
  over as long after it; the gain into it is iterated until the limited
  signal reaches the reference's loudness, so a dense reference (a small
  peak-to-loudness ratio) gets a rebuild limited as hard.

encode() runs a file through an mp3 round trip (ffmpeg), as the suite's
reference mixes were, aligned back by the decoder's delay.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .loudness import integrated_lufs

TILT_RANGE = (50.0, 16000.0)


def _stereo(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.stack([x, x], 1) if x.ndim == 1 else x


def third_octaves(lo: float = TILT_RANGE[0], hi: float = TILT_RANGE[1]) -> np.ndarray:
    """Third-octave centres from lo to hi (base 1 kHz)."""
    k = np.arange(np.ceil(3 * np.log2(lo / 1000)), np.floor(3 * np.log2(hi / 1000)) + 1)
    return 1000 * 2 ** (k / 3)


def band_levels(x: np.ndarray, sr: int, centres: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Third-octave band levels (dB) of the mono sum, from Welch's spectrum."""
    import scipy.signal as ss
    m = _stereo(x).mean(axis=1)
    f, p = ss.welch(m, fs=sr, nperseg=8192)
    c = third_octaves() if centres is None else centres
    lv = np.array([10 * np.log10(p[(f >= a / 2 ** (1 / 6)) & (f < a * 2 ** (1 / 6))].sum() + 1e-20) for a in c])
    return c, lv


def tilt(x: np.ndarray, sr: int) -> float:
    """Spectral tilt in dB per octave: the least-squares slope of the
    third-octave band levels against log2 frequency. Pink noise is ~0, white
    noise ~+3, a mix ~-3 to -6."""
    c, lv = band_levels(x, sr)
    keep = lv > lv.max() - 80  # empty bands (a band-limited signal) do not pull the line
    if keep.sum() < 3:
        return 0.0
    return float(np.polyfit(np.log2(c[keep] / 1000), lv[keep], 1)[0])


def apply_tilt(x: np.ndarray, sr: int, db_per_octave: float, pivot: float = 1000.0) -> np.ndarray:
    """x through a zero-phase tilt of db_per_octave about pivot (flat outside
    TILT_RANGE, so DC and the top octave do not run away)."""
    if not db_per_octave:
        return _stereo(x).copy()
    y = _stereo(x)
    n = len(y)
    f = np.clip(np.fft.rfftfreq(n, 1 / sr), *TILT_RANGE)
    g = 10 ** (db_per_octave * np.log2(f / pivot) / 20)
    return np.fft.irfft(np.fft.rfft(y, axis=0) * g[:, None], n=n, axis=0)


def limit(x: np.ndarray, sr: int, ceiling: float, lookahead_ms: float = 5.0) -> np.ndarray:
    """A look-ahead peak limiter: no sample over `ceiling`. The gain each
    sample needs (min over channels) is spread by a minimum over twice the
    look-ahead and smoothed by a moving mean over it, so it is reached before
    a peak and released as long after it; a moving mean of values at or
    under a peak's gain cannot exceed it, and a final clip catches rounding."""
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    y = _stereo(x)
    peak = np.abs(y).max(axis=1)
    need = np.minimum(1.0, ceiling / np.maximum(peak, 1e-12))
    if need.min() >= 1.0:
        return y.copy()
    w = max(1, int(round(lookahead_ms * sr / 1000)))
    g = uniform_filter1d(minimum_filter1d(need, 2 * w + 1, mode="nearest"), w | 1, mode="nearest")
    return np.clip(y * g[:, None], -ceiling, ceiling)


def profile(x: np.ndarray, sr: int) -> dict:
    """What master_like matches: loudness, sample peak, their difference
    (the peak-to-loudness ratio) and the tilt."""
    y = _stereo(x)
    lufs = integrated_lufs(y, sr)
    peak = float(np.abs(y).max()) if len(y) else 0.0
    peak_db = 20 * np.log10(peak + 1e-12)
    return {"lufs": round(lufs, 2) if np.isfinite(lufs) else None, "peak": round(peak, 5),
            "peak_dbfs": round(peak_db, 2), "plr_db": round(peak_db - lufs, 2) if np.isfinite(lufs) else None,
            "tilt_db_per_octave": round(tilt(y, sr), 3)}


def master_like(x: np.ndarray, sr: int, reference: dict, lookahead_ms: float = 5.0, iterations: int = 6,
                tolerance_db: float = 0.05) -> tuple[np.ndarray, dict]:
    """x mastered to a reference's profile() (a dict, so a song's profile
    can be kept without its audio): tilted to the reference's tilt, then
    gained and limited at the reference's peak until its loudness is the
    reference's within tolerance_db. Returns the master and a report: the
    profiles before and after, the tilt and the gain applied, and the gain
    the limiter took at its deepest (dB)."""
    y = _stereo(x)
    before = profile(y, sr)
    if before["lufs"] is None or reference.get("lufs") is None:
        return y.copy(), {"reference": reference, "before": before, "after": before, "skipped": "silent"}
    d_tilt = reference["tilt_db_per_octave"] - before["tilt_db_per_octave"]
    tilted = apply_tilt(y, sr, d_tilt)
    ceiling = reference["peak"]
    gain_db = reference["lufs"] - integrated_lufs(tilted, sr)
    out = tilted
    for _ in range(iterations):
        out = limit(tilted * 10 ** (gain_db / 20), sr, ceiling, lookahead_ms)
        err = reference["lufs"] - integrated_lufs(out, sr)
        if abs(err) <= tolerance_db:
            break
        gain_db += err
    pre = np.abs(tilted * 10 ** (gain_db / 20)).max(axis=1)
    post = np.abs(out).max(axis=1)
    reduction = float(np.min(np.where(pre > 0, post / np.maximum(pre, 1e-12), 1.0))) if len(pre) else 1.0
    return out, {"reference": reference, "before": before, "after": profile(out, sr),
                 "tilt_applied_db_per_octave": round(d_tilt, 3), "gain_db": round(gain_db, 2),
                 "max_reduction_db": round(20 * np.log10(max(reduction, 1e-12)), 2)}


def encode(x: np.ndarray, sr: int, kbps: int = 256) -> np.ndarray:
    """x through an mp3 round trip at kbps (ffmpeg), moved back by the
    decoder's delay (scenarios.mp3_lag) and cut or padded to x's length."""
    import soundfile as sf

    from .audio import shift
    from .scenarios import mp3_lag
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is needed for the mp3 round trip")
    y = _stereo(x)
    with tempfile.TemporaryDirectory() as d:
        wav, mp3, dec = Path(d) / "in.wav", Path(d) / "x.mp3", Path(d) / "out.wav"
        sf.write(wav, y, sr, subtype="FLOAT")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-b:a", f"{kbps}k", str(mp3)], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(mp3), "-ar", str(sr), str(dec)], check=True)
        z, _ = sf.read(dec, dtype="float64", always_2d=True)
    lag = mp3_lag(y.mean(axis=1), z.mean(axis=1))
    n = len(y)
    z = np.pad(z, ((0, max(0, n - len(z))), (0, 0)))
    return np.stack([shift(c, -lag)[:n] for c in z.T], 1)
