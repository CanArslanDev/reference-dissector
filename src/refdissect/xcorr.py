"""Normalized cross-correlation: the "is this the same waveform?" measure.

ncc(t, c) answers for two short sounds (over a few milliseconds of lag);
sliding_ncc(x, t) slides one sound along a whole stem (a matched filter) and
answers "where does it play, and how loud?".

Lag convention (checked by a test, an easy thing to get backwards): ncc
returns lag d when t is c delayed by d samples, so shift(c, d) lines c up
with t. A window read at s - d lines up with a template read at s.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

from .audio import band


def _next_pow2(n: int) -> int:
    return 1 << (int(n) - 1).bit_length()


def ncc(t: np.ndarray, c: np.ndarray, max_lag: int) -> tuple[float, int]:
    """(normalized correlation with the largest magnitude, its lag) within
    ±max_lag samples. The value is signed: -1 means the same sound inverted."""
    n = _next_pow2(2 * len(t))
    corr = np.fft.irfft(np.fft.rfft(t, n) * np.conj(np.fft.rfft(c, n)), n)
    lags = np.r_[corr[: max_lag + 1], corr[-max_lag:]] if max_lag else corr[:1]
    norm = float(np.linalg.norm(t) * np.linalg.norm(c)) or 1.0
    v = lags / norm
    i = int(np.argmax(np.abs(v)))
    return float(v[i]), (i if i <= max_lag else i - len(v))


def energy(spectra: np.ndarray, n: int) -> np.ndarray:
    """Signal energy from rfft spectra of length n (Parseval), along the last axis."""
    p = np.abs(spectra) ** 2
    e = 2 * p.sum(axis=-1) - p[..., 0] - (p[..., -1] if n % 2 == 0 else 0)
    return e / n


def sliding_ncc(x: np.ndarray, t: np.ndarray, role: str = "full") -> tuple[np.ndarray, np.ndarray]:
    """(ncc[k], gain[k]) of template t against x[k:k+len(t)], both filtered to
    the role's band. gain is the least-squares level of t at k (1 = as loud
    as the template)."""
    xb = band(x, role).astype(np.float64)
    tb = band(t, role).astype(np.float64)
    n = len(tb)
    num = fftconvolve(xb, tb[::-1], mode="valid")
    c = np.cumsum(np.r_[0.0, xb ** 2])
    window = np.maximum(c[n:] - c[:-n], 1e-12)
    tn = float(np.dot(tb, tb)) or 1e-12
    return num / (np.sqrt(tn) * np.sqrt(window)), num / tn
