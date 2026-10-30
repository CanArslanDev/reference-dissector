"""Numbers that describe a sound the way a producer would adjust it: pitch
and cents, pitch drop, decay, brightness, harmonics (drive), envelope.

These are what a model that cannot hear can read and act on ("decay 190 ms
too long", "3rd harmonic 9 dB too loud: less drive"), and what catches what
a single similarity score hides: a synthesized 808 can correlate 0.97 with
the reference and still miss its distortion by 25 dB.
"""

from __future__ import annotations

import functools

import librosa
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import hilbert

from .audio import SR, samples


def envelope_db(x: np.ndarray, window_ms: float) -> np.ndarray:
    """Loudness envelope in dB re its peak. A low tone needs a window of a
    period or more (25 ms for an 808), or the envelope follows the wave."""
    env = np.abs(hilbert(x))
    w = max(1, samples(window_ms))
    # the moving mean of np.convolve(..., mode="same") in O(n) (the 808 fit calls
    # this ~24,000 times; the convolution was half of each call)
    env = np.sqrt(np.maximum(uniform_filter1d(env ** 2, w, mode="constant", cval=0.0), 0))
    return 20 * np.log10(env / (env.max() or 1.0) + 1e-9)


def pitch_track(x: np.ndarray, fmin: float = 25, fmax: float = 400) -> np.ndarray:
    """f0 per 256-sample frame (NaN where unvoiced)."""
    f0, voiced, _ = librosa.pyin(x, fmin=fmin, fmax=fmax, sr=SR, frame_length=4096, hop_length=256)
    return np.where(voiced, f0, np.nan)


def note_name(hz: float) -> str:
    midi = float(librosa.hz_to_midi(hz))
    return f"{librosa.midi_to_note(round(midi), unicode=False)} {100 * (midi - round(midi)):+.0f} cents"


_FREQS = np.fft.rfftfreq(1 << 16, 1 / SR)


@functools.lru_cache(maxsize=8)
def _hann(n: int) -> np.ndarray:
    """np.hanning(n), made once per length (read-only): 70 us of the 808
    fit's 2.6 ms a candidate."""
    w = np.hanning(n)
    w.setflags(write=False)
    return w


def harmonics(x: np.ndarray, f1: float, upto: int = 12, start_s: float = 0.15,
              end_s: float = 0.6) -> np.ndarray:
    """Levels of harmonics 2..upto relative to the fundamental (dB, floor -60),
    from the steady part of a tone."""
    seg = x[int(start_s * SR):int(end_s * SR)]
    seg = seg * _hann(len(seg))
    p = np.abs(np.fft.rfft(seg, 1 << 16)) ** 2
    # each harmonic's ±3 % band as a slice of the sorted frequency axis
    lo = np.searchsorted(_FREQS, np.arange(1, upto + 1) * f1 * 0.97, side="right")
    hi = np.searchsorted(_FREQS, np.arange(1, upto + 1) * f1 * 1.03, side="left")
    level = [p[a:b].max() if b > a else 0.0 for a, b in zip(lo, hi)]
    return np.maximum(10 * np.log10(np.array(level[1:]) / (level[0] or 1e-12) + 1e-12), -60)


def describe(x: np.ndarray, tonal: bool = False) -> dict:
    """Producer-facing descriptors of one sound (peak-normalized first)."""
    x = x / (np.max(np.abs(x)) or 1.0)
    env = envelope_db(x, 25 if tonal else 3)
    peak = int(np.argmax(env))
    after = env[peak:]

    def decay_to(db: float):
        below = np.where(after < db)[0]
        return round(float(below[0]) * 1000 / SR, 1) if len(below) else None
    seg = x[peak:peak + 8192] if tonal else x[:4096]
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    freqs = np.fft.rfftfreq(len(seg), 1 / SR)
    power = spec ** 2
    total = float(np.sum(power)) or 1.0
    cumulative = np.cumsum(power) / total
    d = {"peak_ms": round(peak * 1000 / SR, 1), "decay_-6dB_ms": decay_to(-6),
         "decay_-20dB_ms": decay_to(-20),
         "centroid_hz": round(float(np.sum(freqs * power) / total)),
         "rolloff85_hz": round(float(freqs[min(len(freqs) - 1, np.searchsorted(cumulative, 0.85))])),
         "noisiness": round(float(np.exp(np.mean(np.log(spec + 1e-12))) / (np.mean(spec) or 1.0)), 3)}
    if tonal:
        f = pitch_track(x[: int(0.6 * SR)])
        f = f[np.isfinite(f)]
        if len(f):
            d["pitch_start_hz"] = round(float(np.median(f[:3])), 1)
            d["pitch_hz"] = round(float(np.median(f[len(f) // 2:])), 1)
            d["pitch"] = note_name(d["pitch_hz"])
            d["harmonics_2_6_db"] = [round(float(v), 1) for v in harmonics(x, d["pitch_hz"], upto=6)]
    return d


def mel_db(x: np.ndarray, n_mels: int = 64, hop: int = 256) -> np.ndarray:
    m = librosa.feature.melspectrogram(y=x, sr=SR, n_fft=2048, hop_length=hop, n_mels=n_mels,
                                       fmin=20, fmax=16000)
    return librosa.power_to_db(m, ref=np.max(m) or 1.0, top_db=80)


def mel_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute log-mel difference (dB) where either sound has energy."""
    n = min(len(a), len(b))
    A, B = mel_db(a[:n]), mel_db(b[:n])
    mask = (A > -60) | (B > -60)
    return round(float(np.mean(np.abs(A - B)[mask])), 2) if mask.any() else 0.0
