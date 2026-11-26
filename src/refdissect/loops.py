"""Loop-like layers among the drum sounds.

discover() makes a template of every sound that recurs, and a sampled loop
(a break, a hat or percussion loop) recurs too: each of its hits comes back
once a cycle, at one place. Read as one-shots, a four-bar hat loop is a
dozen "hats" of a hit or two a cycle each, none with a close library match. Here such
sounds are found and named as one layer, and a cycle of it is cut out to
search a loop library with.

A sound is periodic at P bars when its hits keep to one or two places of a
P-bar cycle, cycle after cycle, and do not at any shorter period (a snare
on every bar's 2 and 4 keeps its places at one bar: an ordinary part).
Several such sounds at one period make a layer. That says "loop-like", not
"sampled": a part programmed with a different sample on every step of four
bars reads the same.

Set on one song (99 BPM boom bap, 2026-11-26): 12 of its 26 sounds keep one
or two places of a four-bar cycle, 12-31 hits each, and hold 34 % of the
hat band's hits. The thresholds are that song's and synthetic events'; no
beat with a known loop in its drums has scored them.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from .audio import SR, band, write
from .grid import Grid

PERIODS = (1, 2, 4, 8)  # bars; a sound's period is the shortest it keeps its places at
PLACES = 2  # a periodic sound keeps to this many places of its cycle
SHARE = 0.8  # with this share of its hits
MIN_CYCLES = 4  # in this many cycles
COVER = 0.75  # and in this share of the cycles between its first and last hit (once in four bars is 0.5 at two)
MIN_SOUNDS = 4  # sounds at one period that make a layer


def period(times, grid: Grid) -> dict | None:
    """The shortest of PERIODS at which a sound's hits keep to PLACES places
    of the cycle: {"bars", "places": [[bar in the cycle (from 1), step in
    the bar]], "hits": those at the places, "cycles": the cycles they fall
    in}; None when it has no such period."""
    t = np.asarray(times, dtype=float)
    if len(t) < MIN_CYCLES:
        return None
    steps = np.round((t - grid.phase) / grid.step).astype(int)
    per_bar = grid.steps_per_bar
    for bars in PERIODS:
        n = bars * per_bar
        top = Counter((steps % n).tolist()).most_common(PLACES)
        if len(top) == 2 and bars > 1 and (top[0][0] - top[1][0]) % (n // 2) == 0:
            continue  # one place of half this cycle, in some of its turns only: no period of its own
        at = np.isin(steps % n, [p for p, _ in top])
        cycles = np.unique(steps[at] // n)
        span = int(steps.max() // n - steps.min() // n) + 1
        if at.mean() >= SHARE and len(cycles) >= MIN_CYCLES and len(cycles) >= COVER * span:
            return {"bars": bars, "places": [[p // per_bar + 1, p % per_bar] for p, _ in sorted(top)],
                    "hits": int(at.sum()), "cycles": int(len(cycles))}
    return None


def layers(events: dict[str, list[dict]], templates: dict[str, dict], grid: Grid) -> list[dict]:
    """The loop-like layers: per period over one bar with MIN_SOUNDS
    periodic sounds or more, {"bars", "band" (the commonest among its
    sounds), "sounds": {name: its period() and band}, "hits", "share_of_band"
    (of the hits of that band's sounds), "cycle": the cycle to cut (cycle):
    {"start_bar", "start_s", "sounds": how many of the layer's play in it}}.
    templates: analysis.json's, for each sound's band."""
    found = {}
    for name, hits in events.items():
        p = period([e["t"] for e in hits], grid)
        if p and p["bars"] > 1:
            found[name] = {**p, "band": templates[name]["band"]}
    out = []
    for bars in sorted({p["bars"] for p in found.values()}):
        sounds = {k: v for k, v in found.items() if v["bars"] == bars}
        if len(sounds) < MIN_SOUNDS:
            continue
        b = Counter(v["band"] for v in sounds.values()).most_common(1)[0][0]
        in_band = sum(len(events[k]) for k, t in templates.items() if t["band"] == b and k in events)
        hits = sum(v["hits"] for v in sounds.values())
        mine = sum(v["hits"] for v in sounds.values() if v["band"] == b)
        out.append({"bars": bars, "band": b, "sounds": sounds, "hits": hits,
                    "share_of_band": round(mine / in_band, 3) if in_band else 0.0,
                    "cycle": cycle({k: events[k] for k in sounds}, grid, bars)})
    return out


def cycle(events: dict[str, list[dict]], grid: Grid, bars: int) -> dict:
    """The cycle of a layer to cut out: the one most of its sounds play in,
    then the one they match their templates best in (mean NCC), then the
    first."""
    n = bars * grid.steps_per_bar
    seen: dict[int, dict[str, float]] = {}
    for name, hits in events.items():
        for e in hits:
            c = int(np.round((e["t"] - grid.phase) / grid.step)) // n
            seen.setdefault(c, {})[name] = max(seen.get(c, {}).get(name, 0.0), float(e.get("ncc", 0.0)))
    best = max((c for c in seen if c >= 0), key=lambda c: (len(seen[c]), np.mean(list(seen[c].values())), -c))
    return {"start_bar": best * bars + 1, "start_s": round(grid.phase + best * bars * grid.bar, 5),
            "sounds": len(seen[best])}


def cut(drums: np.ndarray, grid: Grid, layer: dict, folder: str | Path) -> dict[str, str]:
    """One cycle of a layer as audio, to search a loop library with: the
    drum stem in the layer's band (what the loop adds over the kick and
    snare) and the whole drum stem over the same bars. Part of the song:
    under the work folder, local."""
    a = max(0, int(round(layer["cycle"]["start_s"] * SR)))
    seg = drums[a:a + int(round(layer["bars"] * grid.bar * SR))]
    folder = Path(folder)
    stem = f"loop_{layer['bars']}bars"
    return {layer["band"]: str(write(folder / f"{stem}_{layer['band']}.wav", band(seg, layer["band"]))),
            "drums": str(write(folder / f"{stem}_drums.wav", seg))}
