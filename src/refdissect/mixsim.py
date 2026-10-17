"""Mix decisions on recorded stems, without recording again.

Live only renders in real time. In the HAYIR DEMELIYDIM mix loop every fader
move was checked by recording all 13 tracks for 100 bars (~3 minutes), twice:
14 rounds took ~98 minutes. A fader is a gain, and a track's stem recorded
after its devices (capture_stems uses each track's "Post Mixer" output) scales
exactly with it. So: record the stems once, decide the gains here, set them in
Live, and record again only after something non-linear changed (a compressor,
a limiter, a sidechain, a saturator) or before the final check.

- `level_db`: a stem's level in some bars (integrated LUFS, or RMS dB).
- `gains_to_targets`: the gain per stem that puts it at a target level.
- `mix`: the stems summed with gains, and what that does to the peak (a limiter
  on the master is not simulated: `clipped_samples` says when one would act).
"""

from __future__ import annotations

import numpy as np

from .loudness import integrated_lufs


def segments(x: np.ndarray, sr: int, spans: list[tuple[float, float]]) -> np.ndarray:
    """The parts of x between (start, end) seconds, joined."""
    parts = [x[int(a * sr):int(b * sr)] for a, b in spans]
    return np.concatenate(parts) if parts else x[:0]


def level_db(x: np.ndarray, sr: int, measure: str = "lufs") -> float:
    """Integrated LUFS ("lufs") or RMS in dBFS ("rms") of x."""
    if measure == "rms":
        return float(10 * np.log10(np.mean(np.asarray(x, dtype=np.float64) ** 2) + 1e-20))
    return integrated_lufs(x, sr)


def gains_to_targets(stems: dict[str, np.ndarray], targets: dict[str, float], sr: int,
                     spans: dict[str, list[tuple[float, float]]] | None = None,
                     measure: str = "lufs") -> dict[str, dict]:
    """For each stem with a target level (dB), the gain that reaches it, measured
    over that stem's spans (seconds; default the whole stem). Exact for a gain:
    every measure here moves dB for dB with it."""
    out = {}
    for name, target in targets.items():
        x = stems[name] if not spans or name not in spans else segments(stems[name], sr, spans[name])
        now = level_db(x, sr, measure)
        out[name] = {"level_db": round(now, 2) if np.isfinite(now) else None,
                     "gain_db": round(target - now, 2) if np.isfinite(now) else None}
    return out


def mix(stems: dict[str, np.ndarray], gains_db: dict[str, float] | None = None, ceiling: float = 1.0) -> dict:
    """The stems summed (shorter ones zero-padded) with gains in dB, and its peak."""
    gains_db = gains_db or {}
    n = max(len(x) for x in stems.values())
    shape = (n,) + np.asarray(next(iter(stems.values()))).shape[1:]
    y = np.zeros(shape)
    for name, x in stems.items():
        x = np.asarray(x, dtype=np.float64)
        y[:len(x)] += x * 10 ** (gains_db.get(name, 0.0) / 20)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    return {"audio": y, "peak_dbfs": round(20 * np.log10(peak), 2) if peak else None,
            "clipped_samples": int(np.count_nonzero(np.abs(y) >= ceiling))}
