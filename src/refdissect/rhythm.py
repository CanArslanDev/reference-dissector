"""A layer's rhythm read from its onsets, not from transcribed notes.

Basic Pitch placed the reverb-heavy layers of HAYIR DEMELIYDIM up to a third of
a 16th off the grid; their note statistics ("triplet share", "distinct bars")
then measured the transcription's noise: a straight block-chord rebuild gave
the same numbers as the reference. The onset strength in the layer's register,
read at 48 points per bar (every 3rd a 16th, every 4th a 16th triplet), showed
what the notes did not: the intro's bass and riff hitting together on
0-4-7-10-14, the harmony moving on step 7.
"""

from __future__ import annotations

from collections import Counter

import librosa
import numpy as np

from .audio import SR
from .grid import Grid

POSITIONS = 48


def label(p: int) -> str:
    """'7' for a 16th (step 7), 't5' for a 16th-triplet slot, '~8.67' otherwise."""
    if p % 3 == 0:
        return f"{p // 3}"
    if p % 4 == 0:
        return f"t{p // 4}"
    return f"~{p / 3:.2f}"


def onset_profile(x: np.ndarray, grid: Grid, bars: list[int], fmin: float, fmax: float, rel: float = 0.35,
                  hop: int = 64, sr: int = SR) -> dict:
    """Per bar (0-based, as Grid.bar_of counts them), the positions whose onset strength is a
    local peak at or above `rel` times the bar's strongest; over the bars, the
    share of bars with an onset at each position, and how many onsets fall on
    16ths, on triplet slots only, or in between."""
    env = librosa.onset.onset_strength(y=np.asarray(x, dtype=np.float32), sr=sr, hop_length=hop, fmin=fmin,
                                       fmax=fmax, n_mels=40, lag=1, max_size=1)
    t = librosa.frames_to_time(np.arange(len(env)), sr=sr, hop_length=hop)
    rows, hist = {}, Counter()
    for b in bars:
        q = grid.phase + b * grid.bar + np.arange(POSITIONS) / POSITIONS * grid.bar
        v = np.array([env[(t >= a - 0.012) & (t < a + 0.012)].max(initial=0) for a in q])
        top = v.max() or 1.0
        on = [p for p in range(POSITIONS) if v[p] >= rel * top and v[p] >= v[(p - 1) % POSITIONS]
              and v[p] >= v[(p + 1) % POSITIONS]]
        rows[b] = [label(p) for p in on]
        hist.update(rows[b])
    n = max(1, len(bars))
    on16 = sum(c for k, c in hist.items() if k[0] not in "t~")
    trip = sum(c for k, c in hist.items() if k.startswith("t"))
    return {"bars": rows, "share_per_bar": {k: round(c / n, 2) for k, c in hist.most_common()},
            "on_16th": on16, "triplet_only": trip, "between": sum(hist.values()) - on16 - trip}


def common_pattern(profile: dict, min_share: float = 0.5) -> list[str]:
    """The positions present in at least min_share of the bars, in bar order."""
    keep = [k for k, s in profile["share_per_bar"].items() if s >= min_share]

    def pos(k):
        return float(k.lstrip("t~")) * (4 / 3 if k.startswith("t") else 1)
    return sorted(keep, key=pos)
