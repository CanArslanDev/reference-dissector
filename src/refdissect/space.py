"""The space around a layer: width, reverb tail, and a band-pass it went through.

Compared dry, every chord candidate for HAYIR DEMELIYDIM lost by ear: "the
song's chords come from far away". The reference had a long reverb, a chorus
before it, and in its B sections the same chords through a band-pass (heard as
a second, thinner pad that was not there). These read that from a stem.

- `band_correlation`: L/R correlation per octave band. A centred dry sound with
  a fully decorrelated reverb gives r = D / (D + R), so 1/r - 1 bounds the
  reverb's share from above.
- `tail`: what happens after a stop (the whole layer ends): how far the level
  drops in 40 ms (the direct sound), then its decay slope as a reverb time, and
  how decorrelated the tail is.
- `fit_bandpass`: the high- and low-pass that turn one section's spectrum into
  another's (EQ Eight-style slopes, 6 dB/octave per order).
"""

from __future__ import annotations

import numpy as np
import scipy.signal as ss

from .audio import SR

OCTAVES = (125, 250, 500, 1000, 2000, 4000, 8000)


def _band(x: np.ndarray, lo: float, hi: float, sr: int) -> np.ndarray:
    sos = ss.butter(4, (lo, min(hi, sr / 2 * 0.99)), btype="band", fs=sr, output="sos")
    return ss.sosfiltfilt(sos, x, axis=0)


def band_correlation(x: np.ndarray, sr: int = SR, centres=OCTAVES) -> dict[str, dict]:
    """L/R correlation per octave band of a stereo signal (n, 2), and the upper
    bound of the reverb's power share it implies (1/r - 1, for r > 0)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError("band_correlation needs a stereo signal (n, 2)")
    out = {}
    for c in centres:
        y = _band(x[:, :2], c / np.sqrt(2), c * np.sqrt(2), sr)
        l, r = y[:, 0], y[:, 1]
        d = float(np.sqrt((l ** 2).sum() * (r ** 2).sum()))
        rho = float((l * r).sum() / d) if d else 1.0
        out[str(c)] = {"r": round(rho, 3), "reverb_share_max": round(1 / rho - 1, 2) if rho > 0.01 else None}
    return out


def tail(x: np.ndarray, stop_s: float, sr: int = SR, before: float = 0.2, after: float = 1.2,
         fit=(0.06, 0.6)) -> dict:
    """The level after a stop at stop_s (dB re the median of the `before`
    seconds before it, 5 ms RMS), mono or stereo:

    - drop_db: the fall within 40 ms (the direct sound ending);
    - decay_db_s and rt60_s: the slope of the rest, fitted over `fit`;
    - lr_r_before / lr_r_after: L/R correlation before the stop and in the tail
      (stereo only); a reverb tail is decorrelated, a released dry note is not."""
    x = np.asarray(x, dtype=np.float64)
    mono = x.mean(axis=1) if x.ndim == 2 else x
    w = int(0.005 * sr)
    from scipy.ndimage import uniform_filter1d
    env = np.sqrt(np.maximum(uniform_filter1d(mono ** 2, w, mode="constant"), 0))
    k = int(stop_s * sr)
    ref = float(np.median(env[k - int(before * sr):k])) or 1e-12
    seg = 20 * np.log10(env[k:k + int(after * sr)] / ref + 1e-12)
    t = np.arange(len(seg)) / sr
    m = (t >= fit[0]) & (t <= fit[1])
    slope = float(np.polyfit(t[m], seg[m], 1)[0]) if m.sum() > 2 else float("nan")
    out = {"drop_db": round(float(np.interp(0.04, t, seg)), 1), "decay_db_s": round(slope, 1),
           "rt60_s": round(-60 / slope, 2) if slope < 0 else None}
    if x.ndim == 2 and x.shape[1] >= 2:
        def r(a, b):
            d = float(np.sqrt((a ** 2).sum() * (b ** 2).sum()))
            return round(float((a * b).sum() / d), 3) if d else None
        pre, post = x[k - int(before * sr):k], x[k + int(fit[0] * sr):k + int(fit[1] * sr)]
        out.update(lr_r_before=r(pre[:, 0], pre[:, 1]), lr_r_after=r(post[:, 0], post[:, 1]))
    return out


def median_spectrum_db(x: np.ndarray, sr: int = SR, n_fft: int = 16384) -> tuple[np.ndarray, np.ndarray]:
    """(frequencies, median power spectrum in dB) of a mono signal."""
    f, _t, Z = ss.stft(np.asarray(x, dtype=np.float64), fs=sr, nperseg=n_fft, noverlap=n_fft * 3 // 4)
    return f, 10 * np.log10(np.median(np.abs(Z) ** 2, axis=1) + 1e-14)


def filter_response_db(f: np.ndarray, highpass: float | None, lowpass: float | None, order: int,
                       sr: int = SR) -> np.ndarray:
    """Magnitude (dB) of a Butterworth high-pass and/or low-pass of `order` at f."""
    r = np.zeros_like(f, dtype=np.float64)
    for fc, kind in ((highpass, "high"), (lowpass, "low")):
        if fc:
            sos = ss.butter(order, fc, btype=kind, fs=sr, output="sos")
            _, h = ss.sosfreqz(sos, worN=f, fs=sr)
            r += 20 * np.log10(np.abs(h) + 1e-12)
    return r


def fit_bandpass(f: np.ndarray, diff_db: np.ndarray, lo: float = 150.0, hi: float = 6000.0,
                 orders=(1, 2, 3, 4), sr: int = SR) -> dict:
    """The high-/low-pass (cutoffs, order) and gain that best explain diff_db
    (one section's spectrum minus another's, at f) between lo and hi, by RMS."""
    grid = np.geomspace(lo, hi, 200)
    d = np.interp(grid, f, diff_db)
    flat = float(np.sqrt(np.mean((d - d.mean()) ** 2)))
    best = None
    for order in orders:
        for hp in [None, *np.geomspace(lo, hi / 2, 50)]:
            for lp in [None, *np.geomspace(lo * 2, min(hi * 4, sr / 2 * 0.95), 50)]:
                if hp and lp and lp <= hp:
                    continue
                r = filter_response_db(grid, hp, lp, order, sr)
                gain = float(np.mean(d - r))
                err = float(np.sqrt(np.mean((d - r - gain) ** 2)))
                if best is None or err < best[0]:
                    best = (err, order, hp, lp, gain)
    err, order, hp, lp, gain = best
    return {"highpass_hz": round(float(hp)) if hp else None, "lowpass_hz": round(float(lp)) if lp else None,
            "order": order, "slope_db_per_octave": 6 * order, "gain_db": round(gain, 1),
            "rms_error_db": round(err, 2), "rms_error_db_without_filter": round(flat, 2)}
