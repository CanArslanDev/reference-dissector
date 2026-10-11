"""The beat grid and the tuning.

A produced track sits on a fixed grid, so the tempo can be fitted directly:
for each candidate tempo and phase, how strong is the onset envelope on the
grid lines? Searched coarse, then fine around the best candidates. On
PIRAMIT this gave 137.000 BPM with every bar's onsets within a few ms,
where a beat tracker's median inter-beat interval (68.18 BPM, i.e. 136.36)
drifted by up to 49 ms over two minutes.

Half and double time fit almost equally well; bpm_range decides, so set it
for the genre (trap: 100-180 counts the hats' grid, not the half-time feel).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import librosa
import numpy as np

from .audio import SR
from .hits import HOP, strength_peaks


@dataclass
class Grid:
    bpm: float
    phase: float  # time of the first grid beat, in seconds
    beats_per_bar: int = 4

    @property
    def beat(self) -> float:
        return 60.0 / self.bpm

    @property
    def step(self) -> float:
        """A sixteenth note, in seconds."""
        return self.beat / 4

    @property
    def bar(self) -> float:
        return self.beat * self.beats_per_bar

    def bar_of(self, t):
        return np.floor((np.asarray(t) - self.phase + 0.05) / self.bar).astype(int)

    def step_in_bar(self, t):
        """Nearest sixteenth within the bar, 0..(4*beats_per_bar - 1)."""
        steps = 4 * self.beats_per_bar
        return np.round((np.asarray(t) - self.phase) / self.step).astype(int) % steps

    def beats(self, t):
        """Seconds to beats from the start of the audio (the Arrangement places
        the audio at beat 0, so the first grid beat lands at phase * bpm / 60)."""
        return np.asarray(t) * self.bpm / 60.0

    def to_dict(self) -> dict:
        return asdict(self)


def _scores(env: np.ndarray, bpms: np.ndarray, fps: float, phase_step: float = 0.5) -> np.ndarray:
    """(best score, best phase in frames) for each tempo."""
    out = np.zeros((len(bpms), 2))
    for i, bpm in enumerate(bpms):
        period = 60.0 / bpm * fps
        phases = np.arange(0, period, phase_step)
        n_beats = int((len(env) - 1 - period) / period)
        idx = np.round(phases[:, None] + np.arange(n_beats)[None, :] * period).astype(int)
        values = env[np.minimum(idx, len(env) - 1)].mean(axis=1)
        j = int(np.argmax(values))
        out[i] = values[j], phases[j]
    return out


def fit_grid(x: np.ndarray, bpm_range: tuple[float, float] = (100.0, 180.0)) -> tuple[Grid, dict]:
    """The tempo and phase whose grid lines carry the most onset strength."""
    env = librosa.onset.onset_strength(y=x, sr=SR, hop_length=HOP)
    fps = SR / HOP
    coarse = np.arange(bpm_range[0], bpm_range[1], 0.05)
    s = _scores(env, coarse, fps, phase_step=1.0)
    best = []
    for i in np.argsort(-s[:, 0])[:3]:
        fine = np.arange(coarse[i] - 0.1, coarse[i] + 0.1, 0.002)
        f = _scores(env, fine, fps)
        j = int(np.argmax(f[:, 0]))
        best.append((f[j, 0], fine[j], f[j, 1] / fps))
    score, bpm, phase = max(best)
    grid = Grid(bpm=round(float(bpm), 3), phase=round(float(phase), 4))
    return grid, {"score": round(float(score), 3), "mean_onset_strength": round(float(env.mean()), 3)}


def offsets(x: np.ndarray, grid: Grid, role: str = "snare", tolerance: float = 0.05) -> dict:
    """How far strong onsets sit from the grid, per beat of the bar (ms): the
    check that a tempo is right over the whole track, not just on average."""
    t, strength = strength_peaks(x, role)
    out = {}
    for slot in range(grid.beats_per_bar):
        d = []
        k = 0
        while True:
            g = grid.phase + (k * grid.beats_per_bar + slot) * grid.beat
            if g > len(x) / SR:
                break
            near = np.abs(t - g) < tolerance
            if near.any():
                j = np.argmax(np.where(near, strength, -1))
                d.append((t[j] - g) * 1000)
            k += 1
        d = np.array(d)
        out[f"beat {slot + 1}"] = {"bars": len(d),
                                   "median_ms": round(float(np.median(d)), 1) if len(d) else None,
                                   "p90_abs_ms": round(float(np.percentile(np.abs(d), 90)), 1)
                                   if len(d) else None}
    return out


def tuning(x: np.ndarray, seconds: float = 60.0) -> float:
    """Deviation from A440 in semitones (-0.5..0.5) of the pitched content."""
    x = x[: int(seconds * SR)]
    return round(float(librosa.estimate_tuning(y=x, sr=SR, n_fft=8192, bins_per_octave=60)), 2)
