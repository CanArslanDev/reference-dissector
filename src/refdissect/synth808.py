"""An 808 voice fitted to a reference: sine + pitch drop + envelope + drive.

    f(t)   = f_end + (f_start - f_end) * exp(-t / glide)
    amp(t) = (1 - exp(-t / attack)) * exp(-t / decay)
    y(t)   = tanh(drive * (amp(t) * sin(phase + 2*pi*integral f) + asym)) - tanh(drive * asym)

Everything a producer would set on Operator/Simpler plus Saturator. The fit
(differential evolution, about half a minute to a minute and a half) scores a candidate the
way the comparison does: waveform match in the 808 band, plus harmonics
(the grit) and the loudness envelope. Scored on the waveform alone, the fit
gets the highest correlation and loses the distortion entirely (harmonics
25 dB off on PIRAMIT): you get what you measure.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import differential_evolution

from .audio import PRE, SR, band, samples
from .describe import envelope_db, harmonics, mel_distance
from .xcorr import ncc

NAMES = ("f_end_hz", "f_start_hz", "glide_ms", "decay_s", "attack_ms", "drive", "asym", "phase")
BOUNDS = [(25, 60), (25, 120), (2, 200), (0.1, 6), (0.3, 200), (0.1, 8), (-0.6, 0.6), (0, 2 * np.pi)]


@dataclass
class Params:
    f_end_hz: float
    f_start_hz: float
    glide_ms: float
    decay_s: float
    attack_ms: float
    drive: float
    asym: float
    phase: float

    @classmethod
    def of(cls, v) -> Params:
        return cls(*[float(x) for x in v])

    def to_dict(self) -> dict:
        return {k: round(v, 4) for k, v in asdict(self).items()}

    @property
    def glide_semitones(self) -> float:
        return float(12 * np.log2(self.f_start_hz / self.f_end_hz))


def voice(p: Params, seconds: float, pitch_hz: float | None = None) -> np.ndarray:
    """The voice, peak-normalized; pitch_hz moves the whole pitch curve (the
    drop keeps its size in semitones)."""
    n = int(seconds * SR)
    ratio = (pitch_hz / p.f_end_hz) if pitch_hz else 1.0
    t = np.arange(n) / SR
    f = (p.f_end_hz + (p.f_start_hz - p.f_end_hz) * np.exp(-t / (p.glide_ms / 1000))) * ratio
    phi = p.phase + 2 * np.pi * np.cumsum(f) / SR
    amp = (1 - np.exp(-t / (p.attack_ms / 1000))) * np.exp(-t / p.decay_s)
    y = np.tanh(p.drive * (amp * np.sin(phi) + p.asym)) - np.tanh(p.drive * p.asym)
    return (y / (np.max(np.abs(y)) or 1.0)).astype(np.float32)


def laid_out(p: Params, length: int) -> np.ndarray:
    """The voice positioned like a template (onset PRE in)."""
    return np.r_[np.zeros(PRE, np.float32), voice(p, (length - PRE) / SR)][:length]


def compare(reference: np.ndarray, x: np.ndarray, f1: float) -> dict:
    """How close x is to the reference, measure by measure (lower dB is closer)."""
    target = band(reference, "808")
    h_ref, h = harmonics(reference, f1), harmonics(x, f1)
    e_ref = np.maximum(envelope_db(reference, 25), -40)
    n = min(len(reference), len(x))
    return {"ncc": round(abs(ncc(target, band(x, "808"), samples(8))[0]), 3),
            "harmonics_2_5_db": round(float(np.mean(np.abs(h[:4] - h_ref[:4]))), 1),
            "harmonics_6_12_db": round(float(np.mean(np.abs(h[4:] - h_ref[4:]))), 1),
            "envelope_db": round(float(np.mean(np.abs(np.maximum(envelope_db(x[:n], 25), -40) - e_ref[:n]))), 1),
            "mel_db": mel_distance(band(reference, "808"), band(x, "808"))}


def fit(reference: np.ndarray, f1: float, waveform_only: bool = False, seed: int = 1,
        maxiter: int = 150, popsize: int = 20) -> tuple[Params, dict]:
    """Params whose voice matches the reference template (an averaged 808
    note laid out with its onset PRE in), and how long the search took."""
    target = band(reference, "808")
    h_ref = harmonics(reference, f1)
    e_ref = np.maximum(envelope_db(reference, 25), -40)
    n = len(reference)
    # ncc(target, ·) with the target's spectrum and norm computed once
    size = 1 << int(np.ceil(np.log2(2 * len(target))))
    spectrum, t_norm, lag = np.fft.rfft(target, size), float(np.linalg.norm(target)) or 1.0, samples(8)

    def wave_match(c):
        corr = np.fft.irfft(spectrum * np.conj(np.fft.rfft(c, size)), size)
        v = np.r_[corr[: lag + 1], corr[-lag:]] / (t_norm * (float(np.linalg.norm(c)) or 1.0))
        return float(np.max(np.abs(v)))

    def loss(v):
        p = Params.of(v)
        if p.f_start_hz < p.f_end_hz:
            return 2.0
        y = laid_out(p, n)
        wave = 1 - wave_match(band(y, "808"))
        if waveform_only:
            return wave
        harm = np.mean(np.abs(harmonics(y, p.f_end_hz) - h_ref)) / 20
        env = np.mean(np.abs(np.maximum(envelope_db(y, 25), -40) - e_ref)) / 20
        return wave + 0.1 * harm + 0.1 * env

    bounds = list(BOUNDS)
    bounds[0] = (f1 * 0.9, f1 * 1.1)
    started = time.time()
    r = differential_evolution(loss, bounds, popsize=popsize, maxiter=maxiter, tol=1e-7, seed=seed,
                               polish=True)
    p = Params.of(r.x)
    return p, {"loss": round(float(r.fun), 4), "seconds": round(time.time() - started, 1)}


def one_shot(p: Params, pitch_hz: float, seconds: float = 3.0, fade_ms: float = 10) -> np.ndarray:
    """A sample for Simpler: the voice at pitch_hz, -1 dBFS, faded out at the end."""
    y = voice(p, seconds, pitch_hz)
    fade = samples(fade_ms)
    y[-fade:] *= np.linspace(1, 0, fade)
    return (y * 10 ** (-1 / 20) / (np.max(np.abs(y)) or 1.0)).astype(np.float32)
