"""Sections from what plays in each bar.

Every bar gets a signature: which detected sounds play in it (snare,
hats, 808 ...) and which stems are clearly active. A section is a run of
bars with the same signature; runs shorter than `min_bars` are absorbed
into their neighbor. On PIRAMIT this finds the bars without a snare
(25-31), the breakdown without drums or 808 (49-56) and the parts around.
"""

from __future__ import annotations

import numpy as np

from .audio import SR
from .grid import Grid


def bar_activity(grid: Grid, n_bars: int, events: dict[str, list[dict]], notes_808: list[dict],
                 stems: dict[str, np.ndarray] | None = None, active_db: float = -20.0,
                 min_hits: int = 16) -> list[dict]:
    """Per bar: the sounds that play in it (those heard at least min_hits
    times in the track) and the active stems."""
    rows = [{"bar": b + 1, "sounds": set(), "stems": set()} for b in range(n_bars)]
    for name, hits in events.items():
        if len(hits) < min_hits:
            continue
        for b in grid.bar_of([e["t"] for e in hits]):
            if 0 <= b < n_bars:
                rows[b]["sounds"].add(name)
    for b in grid.bar_of([n["t"] for n in notes_808]):
        if 0 <= b < n_bars:
            rows[b]["sounds"].add("808")
    for name, x in (stems or {}).items():
        bar = int(grid.bar * SR)
        levels = []
        for b in range(n_bars):
            a = int((grid.phase + b * grid.bar) * SR)
            seg = x[max(0, a):a + bar]
            levels.append(20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-9) if len(seg) else -120)
        top = np.percentile(levels, 90)
        for b, level in enumerate(levels):
            if level > top + active_db:
                rows[b]["stems"].add(name)
    return rows


def sections(rows: list[dict], min_bars: int = 2, smooth: int = 3, use_stems: bool = False) -> list[dict]:
    """Runs of bars that play the same sounds.

    Each sound's presence is smoothed first (majority over `smooth` bars),
    so one bar with an extra perc does not start a section; runs shorter
    than min_bars join the neighbour whose sounds are closest."""
    names = sorted(set().union(*[r["sounds"] | (r["stems"] if use_stems else set()) for r in rows])) if rows else []
    if not names:
        return [{"start_bar": 1, "end_bar": len(rows), "name": "A", "plays": []}] if rows else []
    m = np.array([[n in (r["sounds"] | (r["stems"] if use_stems else set())) for n in names] for r in rows], float)
    half = smooth // 2
    sig_rows = [tuple((m[max(0, i - half):i + half + 1].mean(axis=0) >= 0.5).astype(int)) for i in range(len(rows))]
    runs: list[list] = []
    for i, sig in enumerate(sig_rows):
        if runs and runs[-1][2] == sig:
            runs[-1][1] = i
        else:
            runs.append([i, i, sig])

    def distance(a, b):
        return sum(x != y for x, y in zip(a, b))
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for j, (a, b, sig) in enumerate(runs):
            if b - a + 1 < min_bars:
                into = min((x for x in (j - 1, j + 1) if 0 <= x < len(runs)), key=lambda x: distance(runs[x][2], sig))
                lo, hi = min(j, into), max(j, into)
                runs[lo] = [runs[lo][0], runs[hi][1], runs[into][2]]
                del runs[hi]
                changed = True
                break
    merged: list[list] = []
    for run in runs:
        if merged and merged[-1][2] == run[2]:
            merged[-1][1] = run[1]
        else:
            merged.append(list(run))
    letters = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    label: dict[tuple, str] = {}
    out = []
    for a, b, sig in merged:
        if sig not in label:
            label[sig] = next(letters)
        out.append({"start_bar": rows[a]["bar"], "end_bar": rows[b]["bar"], "name": label[sig],
                    "plays": [n for n, on in zip(names, sig) if on]})
    return out
