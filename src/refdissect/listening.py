"""Listening files: the reference, a beep, then the candidate, at equal loudness.

What the user could judge on HAYIR DEMELIYDIM was always a two-part file per
question (song -> beep -> one candidate); ten multi-part files at once were
"too many sounds, I did not understand which to test". Both halves are set to
the same integrated loudness, so the louder one does not win by being louder,
and a file that would clip is scaled down as a whole and says so.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from .loudness import to_lufs


def beep(sr: int, seconds: float = 0.25, hz: float = 1000.0, level_db: float = -20.0, gap: float = 0.35) -> np.ndarray:
    """A short sine with silence around it, to mark the switch."""
    t = np.arange(int(seconds * sr)) / sr
    tone = 10 ** (level_db / 20) * np.sin(2 * np.pi * hz * t) * np.hanning(len(t))
    pad = np.zeros(int(gap * sr))
    return np.r_[pad, tone, pad]


def ab_file(reference: np.ndarray, candidate: np.ndarray, sr: int, path: str | Path | None = None,
            target_lufs: float = -14.0, ceiling: float = 0.98) -> dict:
    """reference -> beep -> candidate, each at target_lufs, stereo (n, 2).
    Written as 16-bit WAV when path is given. Reports the scale applied when the
    file would have gone over `ceiling` (both halves move together)."""
    def stereo(x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 2 and x.shape[1] == 1:
            x = x[:, 0]
        return np.stack([x, x], 1) if x.ndim == 1 else x[:, :2]
    b = beep(sr)
    y = np.concatenate([to_lufs(stereo(reference), sr, target_lufs), np.stack([b, b], 1),
                        to_lufs(stereo(candidate), sr, target_lufs)])
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    scale = min(1.0, ceiling / peak) if peak else 1.0
    y = y * scale
    out = {"seconds": round(len(y) / sr, 2), "switch_at_s": round((len(reference) + len(b) / 2) / sr, 2),
           "scaled_db": round(20 * np.log10(scale), 1) if scale < 1 else 0.0}
    if path is not None:
        sf.write(path, y.astype(np.float32), sr, subtype="PCM_16")
        out["file"] = str(path)
    return out
