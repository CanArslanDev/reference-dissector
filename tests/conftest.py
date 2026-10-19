"""Synthetic drums for the tests: no song, no sample library needed."""

import numpy as np
import pytest
from scipy.signal import butter, sosfilt

from refdissect.audio import SR


def burst(seed: int, ms: float, lo: float, hi: float, decay_ms: float, tone_hz: float = 0.0) -> np.ndarray:
    """A drum-like one-shot: filtered noise (and a tone) with an exponential decay."""
    rng = np.random.default_rng(seed)
    n = int(ms * SR / 1000)
    t = np.arange(n) / SR
    sos = butter(4, [lo, hi], btype="band", fs=SR, output="sos")
    x = sosfilt(sos, rng.standard_normal(n))
    if tone_hz:
        x = x * 0.3 + np.sin(2 * np.pi * tone_hz * t) * np.max(np.abs(x))
    x *= np.exp(-t / (decay_ms / 1000))
    return (x / np.max(np.abs(x))).astype(np.float32)


@pytest.fixture(scope="session")
def kit():
    return {"kick": burst(1, 300, 30, 2000, 120, tone_hz=55.0),
            "snare": burst(2, 250, 200, 9000, 60),
            "hat": burst(3, 80, 6000, 16000, 15)}


@pytest.fixture(scope="session")
def beat(kit):
    """20 bars at 137 BPM, first beat 30 ms in: kick on 1 and the 'a' of 3,
    snare on 3 (half time), hats on every eighth; plus a little noise."""
    bpm, phase, bars = 137.0, 0.030, 20
    step = 60 / bpm / 4
    n = int((phase + bars * 16 * step + 1) * SR)
    x = np.random.default_rng(9).standard_normal(n).astype(np.float32) * 0.003
    truth = {"kick": [], "snare": [], "hat": []}
    rng = np.random.default_rng(5)
    for b in range(bars):
        for s in range(16):
            t = phase + (b * 16 + s) * step
            plays = []
            if s in (0, 10):
                plays.append(("kick", 0.9))
            if s == 8:
                plays.append(("snare", 0.8))
            if s % 2 == 0:
                plays.append(("hat", float(rng.uniform(0.25, 0.4))))
            for name, gain in plays:
                a = int(round(t * SR))
                y = kit[name]
                x[a:a + len(y)] += gain * y[: max(0, min(len(y), n - a))]
                truth[name].append(t)
    return {"audio": x, "bpm": bpm, "phase": phase, "truth": truth}


def drum_beat(kit: dict, bpm: float, bars: int, beats_per_bar: int = 4, swing: float = 0.5, lead: float = 0.03,
              crop: float = 0.0, ramp: float = 0.0, hats: int = 1) -> dict:
    """A beat whose grid is known: kick on beat 1 and two 16ths before the
    last beat, snare on the middle beat, a hat on every 16th (every 2nd with
    hats=2), accented on the beats; the odd 16ths at `swing` of their 8th.
    ramp: the tempo rises by that share over the beat. crop: seconds cut from
    the start (the truth moves with it)."""
    steps = 4 * beats_per_bar
    total = bars * steps

    def at(s):  # seconds of 16th step s
        q = s / 4
        if ramp:
            return lead + 60.0 / bpm * (bars * beats_per_bar / ramp) * np.log1p(ramp * q / (bars * beats_per_bar))
        return lead + q * 60.0 / bpm

    def placed(s):
        return at(s - 1) + swing * (at(s + 1) - at(s - 1)) if s % 2 else at(s)

    n = int((at(total) + 1.0) * SR)
    x = np.random.default_rng(9).standard_normal(n).astype(np.float32) * 0.003
    for s in range(total):
        k = s % steps
        plays = [("hat", 0.4 if k % 4 == 0 else 0.3 if k % 2 == 0 else 0.22)] if s % hats == 0 else []
        if k in (0, steps - 6):
            plays.append(("kick", 0.9))
        if k == 4 * (beats_per_bar // 2):
            plays.append(("snare", 0.8))
        for name, gain in plays:
            a = int(round(placed(s) * SR))
            y = kit[name]
            x[a:a + len(y)] += gain * y[: max(0, min(len(y), n - a))]
    beats = np.array([at(s) for s in range(0, total, 4)])
    downbeats = beats[::beats_per_bar]
    a = int(round(crop * SR))
    return {"audio": x[a:], "bpm": bpm, "beats": beats[beats >= crop] - crop,
            "downbeats": downbeats[downbeats >= crop] - crop, "bar": beats_per_bar * 60.0 / bpm}


def tracked(truth: dict, half: bool = False, delay: float = 0.005, every: int | None = None) -> dict:
    """What beat_this gives for a beat: its beats (every other one at half
    tempo, as it read PIRAMIT), `delay` late on 20 ms frames, and a downbeat
    every `every` of its beats (default: on the true downbeats)."""
    beats = truth["beats"][::2] if half else truth["beats"]
    beats = np.round((beats + delay) / 0.02) * 0.02
    if every:
        downbeats = beats[::every]
    else:
        downbeats = np.round((truth["downbeats"] + delay) / 0.02) * 0.02
    return {"beats": beats.tolist(), "downbeats": downbeats.tolist(), "name": "beat_this", "version": "1.1.0",
            "checkpoint": "final0", "device": "cpu", "seconds": 0.5}
