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

One exponential falls too fast: the suite's 808s hold 100 cents or so over
their settled pitch until 150-200 ms, where the fitted drop (bass_low-1:
25.9 semitones, 19 ms) has long settled. tracked_shot() makes the one-shot
with the template's own periods instead: the fitted voice's envelope, drive
and attack, its pitch crossing zero where the template's does
(periods(), from bass808's pitch curve) until the drop has settled.
"""

from __future__ import annotations

import functools
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import differential_evolution, least_squares

from .audio import PRE, SR, band, samples
from .cache import keep, kept, key
from .describe import envelope_db, harmonics, mel_distance
from .xcorr import ncc

NAMES = ("f_end_hz", "f_start_hz", "glide_ms", "decay_s", "attack_ms", "drive", "asym", "phase")
BOUNDS = [(25, 60), (25, 120), (2, 200), (0.1, 6), (0.3, 200), (0.1, 8), (-0.6, 0.6), (0, 2 * np.pi)]
# tracked_shot(): the template's periods are followed until its drop has settled, SETTLED_CENTS or less from
# its settled pitch for SETTLED_S (the suite's 808s stay within 5 cents of it from 0.14-0.98 s on:
# scenarios.pitch_curve), then the pitch eases there over EASE_S; the crossings the reader takes off the voice
# are moved onto the template's in up to ROUNDS rounds, and within STEADY_S of them a round is final
SETTLED_CENTS, SETTLED_S, EASE_S, ROUNDS, STEADY_S = 5.0, 0.1, 0.05, 3, 5e-5
# an 808's own drop only falls: the periods are followed from the first one past which the template's curve
# never reads RISE_CENTS over its lowest point since. Of the suite's 96 templates 12 rise that much, by
# 106-456 cents at 50-80 ms, where a kick plays (11 of them are one 808's on htdemucs, whose truth falls
# from 995 to 67 cents over 0-80 ms); 9 others rise 54-88 cents, among them a 58-cent hold at 140-170 ms
# that a 50-cent line would leave out (that run's plan |cents| 1.9 -> 15.1)
RISE_CENTS = 100.0


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


@functools.lru_cache(maxsize=8)
def _times(n: int) -> np.ndarray:
    """The sample times of n samples (s), made once per length (read-only):
    the fit's every candidate is as long."""
    t = np.arange(n) / SR
    t.setflags(write=False)
    return t


def _drop_hz(p: Params, t: np.ndarray, ratio: float = 1.0) -> np.ndarray:
    """The fitted pitch (Hz) at times t, moved by ratio."""
    return (p.f_end_hz + (p.f_start_hz - p.f_end_hz) * np.exp(-t / (p.glide_ms / 1000))) * ratio


def _sound(p: Params, t: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """The voice at phase phi (radians) at times t: its envelope and drive, peak-normalized."""
    amp = (1 - np.exp(-t / (p.attack_ms / 1000))) * np.exp(-t / p.decay_s)
    y = np.tanh(p.drive * (amp * np.sin(phi) + p.asym)) - np.tanh(p.drive * p.asym)
    return (y / (np.max(np.abs(y)) or 1.0)).astype(np.float32)


def voice(p: Params, seconds: float, pitch_hz: float | None = None) -> np.ndarray:
    """The voice, peak-normalized; pitch_hz moves the whole pitch curve (the
    drop keeps its size in semitones)."""
    t = _times(int(seconds * SR))
    f = _drop_hz(p, t, (pitch_hz / p.f_end_hz) if pitch_hz else 1.0)
    return _sound(p, t, p.phase + 2 * np.pi * np.cumsum(f) / SR)


def laid_out(p: Params, length: int) -> np.ndarray:
    """The voice positioned like a template (onset PRE in), exactly length
    samples (int(seconds * SR) can lose one to rounding)."""
    y = np.r_[np.zeros(PRE, np.float32), voice(p, (length - PRE + 1) / SR)][:length]
    return np.pad(y, (0, length - len(y)))


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


def _read_back(path: Path | None, log) -> tuple[Params, dict] | None:
    """The fit kept at path (cache.kept), with info cached naming its file."""
    found = kept(path, log)
    try:
        return Params.of(found["x"]), {"loss": found["loss"], "seconds": found["seconds"], "cached": path.name}
    except (KeyError, TypeError, ValueError):  # nothing kept (None), or not a fit
        return None


def fit(reference: np.ndarray, f1: float, waveform_only: bool = False, seed: int = 1,
        maxiter: int = 150, popsize: int = 20, cache: str | Path | None = None, log=print) -> tuple[Params, dict]:
    """Params whose voice matches the reference template (an averaged 808
    note laid out with its onset PRE in), and how long the search took.
    With a cache folder, a fit is kept there under the template's samples,
    f1, the settings, the versions and the package's code (cache.key) and
    read back by a fit that has the same ones (info's cached names the file;
    seconds is the search's). The search is seeded, so it would find the
    same Params again: 50-140 s a suite beat, and a few ms read back. log
    gets what the cache says (code changed on disk: not read back or kept)."""
    settings = {"f1": float(f1), "waveform_only": waveform_only, "seed": seed, "maxiter": maxiter, "popsize": popsize}
    path = Path(cache) / f"fit-808-{key((reference,), settings)}.json" if cache else None
    if found := _read_back(path, log):
        return found
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
    info = {"loss": round(float(r.fun), 4), "seconds": round(time.time() - started, 1)}
    keep(path, {"x": [float(v) for v in r.x], **info}, log)
    return p, info


def _finished(y: np.ndarray, fade_ms: float) -> np.ndarray:
    """A voice as a sample for Simpler: -1 dBFS, faded out at the end."""
    fade = samples(fade_ms)
    y[-fade:] *= np.linspace(1, 0, fade)
    return (y * 10 ** (-1 / 20) / (np.max(np.abs(y)) or 1.0)).astype(np.float32)


def one_shot(p: Params, pitch_hz: float, seconds: float = 3.0, fade_ms: float = 10) -> np.ndarray:
    """A sample for Simpler: the voice at pitch_hz, -1 dBFS, faded out at the end."""
    return _finished(voice(p, seconds, pitch_hz), fade_ms)


# ── The template's own periods ──

def _slope_cents(z: np.ndarray, t: np.ndarray, hz: float) -> np.ndarray:
    """What scenarios.pitch_track reads at times t off crossings z (the
    slope of a PCHIP through them, one period each, held outside them),
    in cents re hz."""
    slope = PchipInterpolator(z, np.arange(len(z), dtype=float)).derivative()
    return 1200 * np.log2(np.maximum(slope(np.clip(t, z[0], z[-1])), 1e-9) / hz)


def _guess(t: np.ndarray, c: np.ndarray, hz: float, extra: int) -> np.ndarray:
    """Crossings from the curve integrated between its plateaus (the first
    ends at the first crossing, the last starts at the last), as many
    periods as that holds, `extra` more or fewer, evenly in cycles."""
    moved = np.flatnonzero(np.abs(c - c[0]) > 0.05)
    first = (t[moved[0] - 1] + t[moved[0]]) / 2 if len(moved) else t[0]
    moved = np.flatnonzero(np.abs(c - c[-1]) > 0.05)
    last = (t[moved[-1]] + t[moved[-1] + 1]) / 2 if len(moved) else t[-1]
    fine = np.arange(first, last + 1e-4, 1e-4)
    cycles = np.r_[0.0, np.cumsum(hz * 2 ** (np.interp(fine[:-1], t, c) / 1200)) * 1e-4]
    n = max(2, int(round(cycles[-1])) + extra)
    return np.interp(np.arange(n + 1) * cycles[-1] / n, cycles, fine)


def periods(curve, hz: float) -> tuple[np.ndarray, float] | None:
    """The upward zero crossings (s) a pitch curve was read off, found back
    from the curve ([s, cents re hz], as bass808.json keeps the template's:
    every 10 ms of scenarios.pitch_curve), and how far (rms cents) their
    reading lies from it: the curve integrated between its plateaus
    (_guess: as many periods as it holds, one more or one fewer), then
    moved by least squares until _slope_cents reads the curve. On the
    suite's 96 template curves (both models) the crossings found read them
    within 0.03 cents (rms); off the 10 ms points of a tone whose crossings
    are known they land within 0.04 ms of them. None without three."""
    c = np.asarray(curve, dtype=float)
    if c.ndim != 2 or len(c) < 4:
        return None
    t, v = c[:, 0], c[:, 1]

    def crossings(x):  # the first crossing, then the periods' logs (each 4-200 ms long)
        return x[0] + np.r_[0.0, np.cumsum(np.exp(x[1:]))]
    best = None
    for extra in (0, -1, 1):
        z = _guess(t, v, hz, extra)
        if len(z) < 3:
            continue
        lo = np.r_[z[0] - 0.01, np.full(len(z) - 1, np.log(0.004))]
        hi = np.r_[z[0] + 0.01, np.full(len(z) - 1, np.log(0.2))]
        x0 = np.clip(np.r_[z[0], np.log(np.diff(z))], lo + 1e-9, hi - 1e-9)
        fit_ = least_squares(lambda x: _slope_cents(crossings(x), t, hz) - v, x0, bounds=(lo, hi), xtol=1e-12,
                             ftol=1e-12, x_scale="jac")
        rms = float(np.sqrt(np.mean(fit_.fun ** 2)))
        if best is None or rms < best[1]:
            best = (crossings(fit_.x), rms)
    return best


def _settled_at(z: np.ndarray, curve) -> int:
    """The last of the crossings z the one-shot follows: the first one past
    the point of the curve from which it stays within SETTLED_CENTS of the
    settled pitch for SETTLED_S; on a curve that never does, the last one
    before it first reads more than SETTLED_CENTS under that pitch after
    coming that close; else the last one. An 808's own drop ends at its
    settled pitch: past that, a template holds its ripple and, near its
    end, the next note coming in (one reads -109 cents over its last 50 ms,
    one -78 from 230 ms on, one drops an octave at 1.1 s)."""
    c = np.asarray(curve, dtype=float)
    t, v = c[:, 0], c[:, 1]
    near = np.abs(v) <= SETTLED_CENTS
    reached = np.flatnonzero(near & (t > z[0]))
    for i in reached:
        if t[-1] >= t[i] + SETTLED_S and near[(t >= t[i]) & (t <= t[i] + SETTLED_S)].all():
            return int(min(len(z) - 1, max(2, np.searchsorted(z, t[i], side="right"))))
    under = np.flatnonzero((t > t[reached[0]]) & (v < -SETTLED_CENTS)) if len(reached) else []
    if len(under):
        return int(min(len(z) - 1, max(2, np.searchsorted(z, t[under[0]], side="left") - 1)))
    return len(z) - 1


def _falls_from(z: np.ndarray, curve, last: int) -> int:
    """The first of the crossings z from which the curve, up to z[last],
    never reads RISE_CENTS over its lowest point since (the fitted drop
    plays up to it)."""
    c = np.asarray(curve, dtype=float)
    for s in range(last - 1):
        span = (c[:, 0] > z[s]) & (c[:, 0] <= z[last])
        v = np.r_[np.interp(z[s], c[:, 0], c[:, 1]), c[span, 1]]
        if np.max(v - np.minimum.accumulate(v)) <= RISE_CENTS:
            return s
    return last - 1


def _cycles(p: Params, t: np.ndarray, pitch_hz: float, z: np.ndarray, last: int, m: int | None,
            blend: float = 0.5) -> tuple[np.ndarray, int, float]:
    """(cycles at times t, m, scale): the fitted drop at pitch_hz up to
    z[0], blended into the first period's slope over its last `blend`
    period and scaled so that it starts at the fit's phase and m cycles
    end there (by default the whole number nearest the drop's own), then a
    PCHIP through z[:last + 1] (a cycle each), then the slope there eased
    to pitch_hz over EASE_S."""
    pchip = PchipInterpolator(z, np.arange(len(z), dtype=float))
    slope = pchip.derivative()
    d0, i0 = float(slope(z[0])), int(np.ceil(z[0] * SR))
    w = np.clip((t[:i0] - z[0]) * d0 / blend + 1, 0.0, 1.0)
    w = w * w * (3 - 2 * w)
    f = (1 - w) * _drop_hz(p, t[:i0], pitch_hz / p.f_end_hz) + w * d0
    so_far = np.r_[0.0, np.cumsum(f[:-1])] / SR
    reach = so_far[-1] + f[-1] * (z[0] - t[i0 - 1])  # cycles from 0 to z[0]
    a = p.phase / (2 * np.pi)
    m = int(round(a + reach)) if m is None else m
    scale = (m - a) / reach
    u = np.empty(len(t))
    u[:i0] = a + so_far * scale
    inside = (t >= z[0]) & (t <= z[last])
    u[inside] = m + pchip(t[inside])
    after = t > z[last]
    if after.any():
        ta = t[after]
        g = pitch_hz * 2 ** (np.log2(float(slope(z[last])) / pitch_hz) * np.exp(-(ta - z[last]) / EASE_S))
        u[after] = m + last + g[0] * (ta[0] - z[last]) + np.r_[0.0, np.cumsum(g[:-1])] / SR
    return u, m, scale


def _along(p: Params, t: np.ndarray, pitch_hz: float, z: np.ndarray, last: int, m: int) -> dict:
    """The voice whose crossings, as scenarios.tuning reads them, land on
    z[:last + 1]: in each round the crossings it is made through move by
    how far the reader's nearest ones land off them (the low-pass, and the
    DC the drive's asymmetry adds, move them 0.2-0.7 ms), the closest kept.
    j: how many crossings the reader takes before z[0] (on the template,
    one: the attack's, which it leaves out)."""
    from .scenarios import crossings, native_f0
    made, best = z.copy(), None
    for _ in range(ROUNDS + 1):
        u, m, scale = _cycles(p, t, pitch_hz, made, last, m)
        y = _sound(p, t, 2 * np.pi * u)
        read = crossings(y, native_f0(y) or pitch_hz)
        if len(read) < 2:
            break
        i = np.clip(np.searchsorted(read, z[:last + 1]), 1, len(read) - 1)
        i -= np.abs(read[i - 1] - z[:last + 1]) < np.abs(read[i] - z[:last + 1])
        off = read[i] - z[:last + 1]
        worst = float(np.max(np.abs(off)))
        if best is None or worst < best["worst_s"]:
            best = {"y": y, "j": int(i[0]), "worst_s": worst, "scale": scale}
        made[:last + 1] -= off
        if worst <= STEADY_S or np.any(np.diff(made) < 0.002):
            break
    return best


def tracked_shot(p: Params, pitch_hz: float, curve, curve_hz: float, seconds: float = 3.0,
                 fade_ms: float = 10) -> tuple[np.ndarray, dict] | None:
    """A sample for Simpler (one_shot's level and fade) whose pitch follows
    the template's own periods, and what synth808.json records of it: the
    fitted voice's envelope, drive and attack, crossing zero where the
    template does (periods() of its pitch curve, cents re curve_hz) from
    its second crossing, or from the first past which the curve only falls
    (_falls_from), until its drop has settled (_settled_at), then eased to
    pitch_hz. Before that, the fitted drop, started at the fit's phase and
    scaled to end a whole cycle there: of the two counts nearest its own,
    one whose crossings the rounds put in place (_along), the one scaled
    least (0.76-2.1 on the suite's beats, median 1.03-1.06). None where the
    curve holds under three crossings (one_shot's voice then).

    scenarios.tuning reads the result as the analysis reads the template:
    on the suite's 96 beat runs, from 50 ms to the end of the periods
    followed (450 ms at most), within 0.35 cents (median per one-shot; 5.0
    at most) where it takes one crossing before the first period, as on the
    template (30 one-shots on htdemucs, 29 on htdemucs_ft). Where it takes
    none or two (the attack's low-passed start), it reads up to 122 cents
    apart before the third crossing (55-70 ms; a fast note is read later):
    preferring one there scaled the attack by up to 2.5 for the same plan
    scores. heard_808's Detune correction at the common pitch then no
    longer follows the length of the notes: on bass_low-1 47.1 / 123.4 /
    82.6 cents (notes under 120 ms, 120-250, 250-600) become 0.1 / -0.1 /
    -0.1 (htdemucs_ft: 38.4 / 114.5 / 84.0 -> 3.9 / 0.0 / -0.2). About 0.2 s
    a one-shot."""
    found = periods(curve, curve_hz)
    if found is None or found[0][0] < 0.005:
        return None
    z, rms = found
    last = _settled_at(z, curve)
    first = _falls_from(z, curve, last)
    z, last = z[first:], last - first
    if last < 2 or z[0] < 0.005:
        return None
    t = _times(int(seconds * SR))
    _, m0, scale0 = _cycles(p, t, pitch_hz, z, last, None)
    a = p.phase / (2 * np.pi)
    natural = a + (m0 - a) / scale0
    tries = [r for m in (int(np.floor(natural)), int(np.floor(natural)) + 1)
             if m > a and (r := _along(p, t, pitch_hz, z, last, m))]
    if not tries:
        return None
    pick = min(tries, key=lambda r: (r["worst_s"] > STEADY_S, abs(np.log(r["scale"]))))
    info = {"periods": len(z) + first, "from": first, "followed": last + 1, "until_s": round(float(z[last]), 4),
            "settled": last + 1 < len(z), "curve_rms_cents": round(rms, 3),
            "attack_scale": round(float(pick["scale"]), 3), "read_as_template": pick["j"] == 1,
            "worst_ms": round(1000 * pick["worst_s"], 3)}
    return _finished(pick["y"], fade_ms), info
