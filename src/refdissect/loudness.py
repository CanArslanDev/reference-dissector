"""Integrated loudness (ITU-R BS.1770-4, LUFS) without extra dependencies.

K-weighting as two biquads (a +4 dB high shelf near 1.7 kHz and a high-pass at
38 Hz, recomputed for the sample rate), 400 ms blocks
with 75 % overlap, the absolute gate at -70 LUFS and the relative gate 10 LU
under the ungated mean.
"""

from __future__ import annotations

import numpy as np
import scipy.signal as ss


def k_weight(x: np.ndarray, sr: int) -> np.ndarray:
    """BS.1770 K-weighting at any rate: the standard's 48 kHz filters re-derived
    from their analogue parameters (as libebur128 does)."""
    f0, gain_db, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    k = np.tan(np.pi * f0 / sr)
    vh = 10 ** (gain_db / 20)
    vb = vh ** 0.4996667741545416
    a0 = 1 + k / q + k * k
    b = [(vh + vb * k / q + k * k) / a0, 2 * (k * k - vh) / a0, (vh - vb * k / q + k * k) / a0]
    a = [1, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0]
    y = ss.lfilter(b, a, x, axis=0)
    f0, q = 38.13547087602444, 0.5003270373238773
    k = np.tan(np.pi * f0 / sr)
    a0 = 1 + k / q + k * k
    return ss.lfilter([1, -2, 1], [1, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0], y, axis=0)


def integrated_lufs(x: np.ndarray, sr: int) -> float:
    """Integrated loudness of a mono (n,) or multichannel (n, ch) signal; -inf if silent."""
    x = np.asarray(x, dtype=np.float64)
    x = x[:, None] if x.ndim == 1 else x
    y = k_weight(x, sr)
    block, step = int(0.4 * sr), int(0.1 * sr)
    if len(y) < block:
        z = np.mean(y ** 2, axis=0)[None, :]
    else:
        starts = range(0, len(y) - block + 1, step)
        z = np.array([np.mean(y[s:s + block] ** 2, axis=0) for s in starts])
    power = z.sum(axis=1)  # channel weights 1 (left, right, centre)
    loud = -0.691 + 10 * np.log10(power + 1e-20)
    keep = loud > -70
    if not keep.any():
        return float("-inf")
    rel = -0.691 + 10 * np.log10(power[keep].mean()) - 10
    keep &= loud > rel
    return float(-0.691 + 10 * np.log10(power[keep].mean()))


def to_lufs(x: np.ndarray, sr: int, target: float = -14.0) -> np.ndarray:
    """x scaled to the target integrated loudness (unchanged if silent)."""
    lv = integrated_lufs(x, sr)
    return x if not np.isfinite(lv) else x * 10 ** ((target - lv) / 20)
