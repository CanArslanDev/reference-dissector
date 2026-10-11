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
