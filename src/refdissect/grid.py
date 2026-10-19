"""The beat grid and the tuning.

A produced track sits on a fixed grid, so the tempo can be fitted directly:
for each candidate tempo and phase, how strong is the onset envelope on the
grid lines? Searched coarse, then fine around the best candidates. On
PIRAMIT this gave 137.000 BPM with every bar's onsets within a few ms,
where a beat tracker's median inter-beat interval (68.18 BPM, i.e. 136.36)
drifted by up to 49 ms over two minutes (a line through the same beats gives
68.501: tracker_line).

Half and double time fit almost equally well; bpm_range decides, so set it
for the genre (trap: 100-180 counts the hats' grid, not the half-time feel).

estimate() takes from a beat tracker's beats and downbeats (beats.py) what
the comb cannot know, and moves the comb's phase where it is slightly off.
On 26 constant-tempo test beats (72-160 BPM, 4/4 and 3/4, swing up to 66 %,
two kits; the comb on the drum bus, beat_this on the mp3 mix):
- the octave: fit_grid in 100-180 read 9 at x4/3, x3/2 or x2/3 (no octave
  of 95 BPM lies in it); refined around the tracker's tempo, none;
- beat 1: fit_grid's first grid beat was a downbeat in 16 of 22 beats that
  start on one and in none of 4 that start mid-bar; with the tracker's
  downbeats the first bar line was beat 1 in all 26;
- the phase: the comb's lines sat 8.0 ms (median) after the sample starts,
  and on 4 drum buses half a beat or a swung 16th off; moved onto the
  attacks, 0.1 ms after them (largest 3.2 ms off);
- the meter, the swing of the off-beat 16ths and 8ths, and how far the
  beats drift from a constant tempo.
Without a tracker it is fit_grid in bpm_range, exactly as before.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
from fractions import Fraction

import librosa
import numpy as np
from scipy.signal import savgol_filter

from .audio import SR, attack_time
from .hits import HOP, strength_peaks

# The tracker's tempo times these: its own level, the octaves, and the
# ternary relatives (reported; see _choose)
LEVELS = (("x1", 1.0), ("x2", 2.0), ("x1/2", 0.5), ("x3/2", 1.5), ("x2/3", 2 / 3))
WINDOW = 0.03  # each level is refined by the comb within +-3 %
DRIFT_MS = 10.0  # beyond this the tracker's beats are kept as a tempo map
BAR_SLACK = 0.05  # a bar line up to 50 ms before the audio starts is the first one (as Grid.bar_of counts)


@dataclass
class Grid:
    bpm: float  # quarter notes per minute, as Live counts its tempo
    phase: float  # time of the first bar line (beat 1), in seconds; fit_grid's: the first grid beat
    beats_per_bar: int = 4
    beat_unit: int = 4  # the meter's denominator: 4 in 3/4 and 4/4, 8 in 6/8

    @property
    def beat(self) -> float:
        """A quarter note, in seconds."""
        return 60.0 / self.bpm

    @property
    def step(self) -> float:
        """A sixteenth note, in seconds."""
        return self.beat / 4

    @property
    def steps_per_bar(self) -> int:
        """Sixteenths in a bar: 16 in 4/4, 12 in 3/4 and 6/8, 14 in 7/8."""
        return self.beats_per_bar * 16 // self.beat_unit

    @property
    def bar(self) -> float:
        return self.beat * self.beats_per_bar * 4 / self.beat_unit

    def bar_of(self, t):
        return np.floor((np.asarray(t) - self.phase + 0.05) / self.bar).astype(int)

    def step_in_bar(self, t):
        """Nearest sixteenth within the bar, 0..(steps_per_bar - 1)."""
        return np.round((np.asarray(t) - self.phase) / self.step).astype(int) % self.steps_per_bar

    def beats(self, t):
        """Seconds to beats from the start of the audio (the Arrangement places
        the audio at beat 0, so the first bar line lands at phase * bpm / 60)."""
        return np.asarray(t) * self.bpm / 60.0

    def to_dict(self) -> dict:
        return asdict(self)


def parse_meter(text: str) -> tuple[int, int]:
    """'6/8' -> (6, 8)."""
    n, d = (int(v) for v in text.split("/"))
    if not 1 <= n <= 32 or d not in (2, 4, 8, 16):
        raise ValueError(f"Not a meter: {text}")
    return n, d


def _comb(env: np.ndarray, bpm: float, fps: float, phase_step: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """(phases in frames, mean onset strength on the lines) at one tempo."""
    period = 60.0 / bpm * fps
    phases = np.arange(0, period, phase_step)
    n_beats = int((len(env) - 1 - period) / period)
    idx = np.round(phases[:, None] + np.arange(n_beats)[None, :] * period).astype(int)
    return phases, env[np.minimum(idx, len(env) - 1)].mean(axis=1)


def _scores(env: np.ndarray, bpms: np.ndarray, fps: float, phase_step: float = 0.5) -> np.ndarray:
    """(best score, best phase in frames) for each tempo."""
    out = np.zeros((len(bpms), 2))
    for i, bpm in enumerate(bpms):
        phases, values = _comb(env, bpm, fps, phase_step)
        j = int(np.argmax(values))
        out[i] = values[j], phases[j]
    return out


def _envelope(x: np.ndarray) -> tuple[np.ndarray, float]:
    return librosa.onset.onset_strength(y=x, sr=SR, hop_length=HOP), SR / HOP


def _fit(env: np.ndarray, fps: float, bpm_range: tuple[float, float]) -> tuple[float, float, float]:
    """(score, bpm, phase in s) of the comb that carries the most onset
    strength in bpm_range: 0.05 BPM steps, then 0.002 around the best three."""
    coarse = np.arange(bpm_range[0], bpm_range[1], 0.05)
    s = _scores(env, coarse, fps, phase_step=1.0)
    best = []
    for i in np.argsort(-s[:, 0])[:3]:
        fine = np.arange(coarse[i] - 0.1, coarse[i] + 0.1, 0.002)
        f = _scores(env, fine, fps)
        j = int(np.argmax(f[:, 0]))
        best.append((f[j, 0], fine[j], f[j, 1] / fps))
    return max(best)


def fit_grid(x: np.ndarray, bpm_range: tuple[float, float] = (100.0, 180.0)) -> tuple[Grid, dict]:
    """The tempo and phase whose grid lines carry the most onset strength."""
    env, fps = _envelope(x)
    score, bpm, phase = _fit(env, fps, bpm_range)
    grid = Grid(bpm=round(float(bpm), 3), phase=round(float(phase), 4))
    return grid, {"score": round(float(score), 3), "mean_onset_strength": round(float(env.mean()), 3)}


def _numbered(beats: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    """(beat numbers, beats): intervals counted in half periods (a skipped beat
    counts 2, an extra off-beat 0.5) and the beats on half numbers left out."""
    k = np.r_[0.0, np.cumsum(np.maximum(0.5, np.round(np.diff(beats) / period * 2) / 2))]
    half = np.abs(k - np.round(k)) > 0.25
    if half.sum() > (~half).sum():
        return np.round(k[half] - 0.5).astype(int), beats[half]
    return np.round(k[~half]).astype(int), beats[~half]


def tracker_line(beats) -> tuple[float, np.ndarray, np.ndarray] | None:
    """(BPM, beat numbers, beats) of a line through a tracker's beats, with
    every beat that falls on a whole number (_numbered). beat_this's beats
    come on 20 ms frames, so their median interval is a whole number of
    frames: 0.88 s (68.18 BPM) on PIRAMIT, where the line gives 68.498. Beats
    more than a quarter period off the line are left out of the fit."""
    b = np.asarray(beats, dtype=float)
    if len(b) < 4:
        return None
    period = float(np.median(np.diff(b)))
    for _ in range(3):
        k, kb = _numbered(b, period)
        for _ in range(4):
            if len(k) < 3:
                return None
            period, t0 = np.polyfit(k, kb, 1)
            good = np.abs(kb - (t0 + period * k)) < 0.25 * period
            if good.all():
                break
            k, kb = k[good], kb[good]
    return (60.0 / float(period), *_numbered(b, float(period)))


def _on_16ths(t: np.ndarray, bpm: float, phase: float, tol: float = 0.012) -> float:
    """How much better than chance onsets fall within tol of a grid's 16th
    lines: 1 when all do, 0 when no more than random times would."""
    step = 60.0 / bpm / 4
    tol = min(tol, step / 4)
    rel = (np.asarray(t) - phase) / step
    share = float(np.mean(np.abs(rel - np.round(rel)) * step < tol)) if len(rel) else 0.0
    chance = 2 * tol / step
    return (share - chance) / (1 - chance)


def octave_candidates(env: np.ndarray, fps: float, onsets: np.ndarray, tracker_bpm: float,
                      bpm_range: tuple[float, float] | None) -> list[dict]:
    """Each level of the tracker's tempo, refined by the comb within +-3 %:
    its tempo, comb phase and score, and how well the onsets sit on its 16ths."""
    rows = []
    for level, ratio in LEVELS:
        c = tracker_bpm * ratio
        score, bpm, phase = _fit(env, fps, (c * (1 - WINDOW), c * (1 + WINDOW)))
        rows.append({"level": level, "bpm": round(float(bpm), 3), "phase": float(phase),
                     "score": round(float(score), 3), "on_16ths": round(_on_16ths(onsets, bpm, phase), 2),
                     "in_range": bool(bpm_range is not None and bpm_range[0] <= bpm <= bpm_range[1])})
    return rows


def _choose(rows: list[dict], bpm_range: tuple[float, float] | None) -> tuple[dict, str]:
    """The tracker's level unless bpm_range holds an octave of it instead.
    x3/2 and x2/3 are reported but never chosen here (--bpm can): a straight
    beat's 16ths miss their lines, and a swung one fits x3/2's better than its
    own (test beats at 66 % swing: on_16ths 1.00 against 0.52)."""
    by = {r["level"]: r for r in rows}
    if bpm_range is None:
        return by["x1"], "the tracker's level"
    for level in ("x1", "x2", "x1/2"):
        if by[level]["in_range"]:
            return by[level], f"bpm_range {bpm_range[0]:g}-{bpm_range[1]:g}"
    return by["x1"], f"no octave of the tracker's tempo in bpm_range {bpm_range[0]:g}-{bpm_range[1]:g}"


def _level(ratio: float) -> Fraction:
    """The grid's tempo over the tracker's as a simple fraction (2, 1/2, 3/2...)."""
    return Fraction(ratio).limit_denominator(4)


def phase_near(env: np.ndarray, fps: float, bpm: float, beats: np.ndarray, ratio: float,
               width: float = 0.03) -> tuple[float, float, dict]:
    """(phase in s, score, evidence) of the comb at bpm, its phase searched only
    within `width` of where the tracker's beats sit. Those repeat against the
    grid's lines every beat at x1 and x2, every half beat at x1/2 and x3/2,
    every third at x2/3; where they agree on no place (resultant < 0.5, a
    tempo forced far from the tracker's) the whole beat is searched. Alone,
    the comb sat half a beat or a swung 16th off on 4 of 26 test drum buses:
    hats and kicks fill the off-beats too."""
    beat = 60.0 / bpm
    fold = beat / _level(ratio).denominator
    z = np.mean(np.exp(2j * np.pi * np.asarray(beats) / fold))
    centre = float(np.angle(z)) / (2 * np.pi) * fold
    phases, values = _comb(env, bpm, fps)
    free = int(np.argmax(values))
    if abs(z) >= 0.5:
        off = (phases / fps - centre + fold / 2) % fold - fold / 2
        values = np.where(np.abs(off) <= width, values, -np.inf)
    j = int(np.argmax(values))
    moved = float((phases[j] - phases[free]) / fps + beat / 2) % beat - beat / 2
    return float(phases[j] / fps), float(values[j]), {"agreement": round(float(abs(z)), 2),
                                                      "comb_moved_ms": round(1000 * moved, 1)}


def _attack(x: np.ndarray, t: float, role: str) -> float | None:
    """The attack (audio.attack_time in the role's band) of an onset-strength
    peak at t: searched from 40 ms before it over 60 ms. None for a hit still
    ringing from the sound before (no rise in the window)."""
    a = int((t - 0.04) * SR)
    if a < 0:
        return None
    at = attack_time(x[a:a + int(0.1 * SR)], role, search_s=0.06)
    return a / SR + at if at > 0.001 else None


def attack_offset(x: np.ndarray, bpm: float, phase: float, near: float = 0.035) -> tuple[float, dict]:
    """Median seconds from the beat lines to the attacks of the stronger half
    of the kick- and snare-band onsets within `near` of them; negative when
    the attacks come first. The comb's lines (onset-strength peaks) came
    5.5-12.7 ms after them on the test beats, 7.1 ms on PIRAMIT's drums."""
    beat = 60.0 / bpm
    d = []
    for role in ("kick", "snare"):
        t, s = strength_peaks(x, role)
        if not len(t):
            continue
        line = phase + np.round((t - phase) / beat) * beat
        keep = (np.abs(t - line) < near) & (s >= np.median(s))
        d += [a - li for ti, li in zip(t[keep], line[keep]) if (a := _attack(x, ti, role)) is not None]
    if len(d) < 4:
        return 0.0, {"attacks": len(d)}
    q1, med, q3 = np.percentile(d, [25, 50, 75])
    return float(med), {"attacks": len(d), "iqr_ms": round(1000 * float(q3 - q1), 1)}


def tracker_meter(beats, downbeats, ratio: float = 1.0, min_share: float = 0.6,
                  min_bars: int = 4) -> tuple[int, dict]:
    """Beats per bar from the tracker beats between its downbeats: 3 or 4 when
    at least min_share of min_bars or more bars agree, else 4 (unsure). On the
    3/4 test beats beat_this agreed on 3 at 72 BPM (in 76 and 88 % of the
    bars), not at 136 (38 and 59 %: 4/4); it never read a 4/4 beat as 3. A
    3/4 read at twice the tracker's tempo keeps the tracker's bar (6/4); a
    4/4 is split in two, trap's half-time convention."""
    b, d = np.asarray(beats, dtype=float), np.asarray(downbeats, dtype=float)
    counts = [int(np.sum((b >= d[i] - 1e-3) & (b < d[i + 1] - 1e-3))) for i in range(len(d) - 1)]
    hist = Counter(counts)
    evidence = {"source": "tracker", "tracker_beats_per_bar": {str(k): v for k, v in sorted(hist.items())},
                "bars": len(counts)}
    n, agree = hist.most_common(1)[0] if counts else (4, 0)
    share = agree / len(counts) if counts else 0.0
    accepted = n in (3, 4) and share >= min_share and len(counts) >= min_bars
    evidence.update(share=round(share, 2), accepted=accepted)
    if not accepted:
        return 4, evidence
    return (n * 2 if n % 2 and round(ratio) == 2 else n), evidence


def bar_line(phase: float, grid: Grid, downbeats) -> tuple[float, dict]:
    """phase moved to the 16th of the bar that most of the tracker's downbeats
    fall on (ties: the earliest downbeat's), then to the first bar line."""
    d = np.asarray(downbeats, dtype=float)
    if not len(d):
        return phase, {"downbeats": 0}
    residues = np.round((d - phase) / grid.step).astype(int) % grid.steps_per_bar
    votes = Counter(residues.tolist())
    top = max(votes.values())
    winner = next(int(r) for r in residues if votes[r] == top)
    second = max([v for r, v in votes.items() if r != winner], default=0)
    phase = (phase + winner * grid.step + BAR_SLACK) % grid.bar - BAR_SLACK
    return phase, {"downbeats": len(d), "votes_16th_of_bar": {str(r): v for r, v in votes.most_common(4)},
                   "ambiguous": second >= 0.8 * top}


def swing(x: np.ndarray, grid: Grid, role: str = "hat") -> dict:
    """Where the off-beat 16ths and 8ths sit, from onsets in the hat band
    against the grid's beats. ratio_16: the odd 16th's place in its 8th (0.5
    straight, 0.66 a triplet swing); ratio_8: the off-beat 8th's place in the
    beat. Each is the median place, less the median place of the onsets on
    the lines (the onset detector's delay); None from fewer than 8 onsets, and
    ratio_16 when the 8ths themselves swing. On the test beats ratio_16 was
    within 0.6 and 0.75 points (median; largest 3.3) of 58 and 66 %, and
    read straight 16ths as 50.2-51.4 %."""
    t, _ = strength_peaks(x, role)
    u = ((t - grid.phase) / grid.beat) % 1.0  # place in the beat
    v = (2 * u) % 1.0  # place in the 8th
    wrap_u, wrap_v = np.where(u > 0.5, u - 1, u), np.where(v > 0.5, v - 1, v)
    on_beat, on_8th = np.abs(wrap_u) < 0.06, np.abs(wrap_v) < 0.12
    off_8, off_16 = (u >= 0.4) & (u <= 0.72), (v >= 0.3) & (v <= 0.8)
    delay_u = float(np.median(wrap_u[on_beat])) if on_beat.sum() >= 8 else 0.0
    delay_v = float(np.median(wrap_v[on_8th])) if on_8th.sum() >= 8 else 0.0
    r8 = float(np.median(u[off_8])) - delay_u if off_8.sum() >= 8 else None
    r16 = float(np.median(v[off_16])) - delay_v if off_16.sum() >= 8 and (r8 is None or r8 < 0.55) else None
    q16 = np.percentile(v[off_16], [25, 75]) if off_16.sum() >= 8 else (0.0, 0.0)
    return {"ratio_16": None if r16 is None else round(r16, 3), "ratio_8": None if r8 is None else round(r8, 3),
            "evidence": {"band": role, "onsets": len(t), "on_8ths": int(on_8th.sum()),
                         "off_16ths": int(off_16.sum()), "off_8ths": int(off_8.sum()),
                         "iqr_16_pp": round(100 * float(q16[1] - q16[0]), 1),
                         "delay_ms": round(1000 * delay_v * grid.beat / 2, 1)}}


def _running(r: np.ndarray, size: int, how) -> np.ndarray:
    """how (np.nanmedian...) over `size` values centred on each, fewer at the ends."""
    h = size // 2
    return how(np.lib.stride_tricks.sliding_window_view(np.pad(r, h, constant_values=np.nan), 2 * h + 1), axis=1)


def drift(x: np.ndarray, grid: Grid, numbers: np.ndarray, beats: np.ndarray, ratio: float,
          onsets: np.ndarray | None = None, near: float = 0.04) -> tuple[float, list[float] | None]:
    """How far the beats move from the constant grid (ms). Each numbered
    tracker beat is replaced by the attack of the onset nearest it (within
    `near`): beat_this's beats come on 20 ms frames and run a few ms late. Its
    place against the grid line it should fall on is smoothed: a running
    median over 5 beats (a stray one), then a quadratic Savitzky-Golay fit
    over 9 (17 for the tracker's own beats, used less their median offset
    where fewer than 8 beats have an onset), which follows a ramp to its ends
    where a running mean lags. Beyond DRIFT_MS, the grid's beats moved by
    that curve are kept: a tempo map. Constant-tempo test beats: at most
    5.3 ms (from the tracker's own beats: up to 15.3). A tempo rising 3 % over
    16 bars: 198-220 ms, and the map within 1.4-2.6 ms of the beats (median)
    where the constant grid is 17-24 ms off."""
    b, numbers = np.asarray(beats, dtype=float), np.asarray(numbers)
    if len(b) < 2:
        return 0.0, None
    den = _level(ratio).denominator
    attacks = []
    if onsets is not None and len(onsets) >= 2:
        j = np.clip(np.searchsorted(onsets, b), 1, len(onsets) - 1)
        nearest = np.where(b - onsets[j - 1] < onsets[j] - b, onsets[j - 1], onsets[j])
        attacks = [(k, a) for k, t, bt in zip(numbers, nearest, b)
                   if abs(t - bt) < near and (a := _attack(x, t, "full")) is not None]
    if len(attacks) >= 8:
        numbers, b = (np.array(v) for v in zip(*attacks))
    r = (b - grid.phase) / grid.beat - float(_level(ratio)) * numbers
    r -= np.round(np.median(r) * den) / den if len(attacks) >= 8 else np.median(r)
    curve = _running(r, 5, np.nanmedian)
    window = 9 if len(attacks) >= 8 else 17
    if len(curve) >= window:
        curve = savgol_filter(curve, window, 2, mode="interp")
    curve = curve * grid.beat
    drift_ms = round(1000 * float(np.max(np.abs(curve))), 1)
    if drift_ms <= DRIFT_MS:
        return drift_ms, None
    k = np.arange(np.ceil((b[0] - grid.phase) / grid.beat), np.floor((b[-1] - grid.phase) / grid.beat) + 1)
    lines = grid.phase + k * grid.beat
    return drift_ms, np.round(lines + np.interp(lines, b, curve), 4).tolist()


def onset_grid(grid: Grid, info: dict) -> Grid:
    """The grid on the onset-strength time base (hits.strength_peaks, which
    detect.discover and offsets read): moved back by the phase correction."""
    return replace(grid, phase=round(grid.phase - info.get("phase_correction_ms", 0.0) / 1000, 4))


def estimate(x: np.ndarray, tracker: dict | None = None, bpm_range: tuple[float, float] | None = (100.0, 180.0),
             bpm: float | None = None, meter: str | None = None) -> tuple[Grid, dict]:
    """The grid of a drum stem, with what decided it.

    tracker: beats.track()'s output, or None. Without one (or with fewer than
    4 beats) this is fit_grid in bpm_range, as before, or the comb at a forced
    bpm; the meter is --meter or 4/4 and phase the first grid beat. With one:
    1. the octave: the tracker's tempo (a line through its beats) and its
       levels x2, x1/2, x3/2, x2/3, each refined by the comb within +-3 %
       (alternatives); bpm_range picks the octave (_choose), bpm forces one;
    2. the phase: the comb's, within 30 ms of the tracker's beats (phase_near),
       moved onto the attacks (phase_correction_ms), then to the 16th of the
       bar the tracker's downbeats vote for, so phase is the first bar line;
    3. the meter: --meter, else the tracker's (tracker_meter);
    4. swing against the grid, and drift_ms of the tracker's beats from it
       (beyond 10 ms they are kept as tempo_map; the grid stays constant).
    """
    n, unit = parse_meter(meter) if meter else (4, 4)
    line = tracker_line(tracker["beats"]) if tracker else None
    if line is None:
        return _without_tracker(x, bpm_range, bpm, meter, n, unit)
    tempo, numbers, beats = line
    env, fps = _envelope(x)
    t, s = strength_peaks(x, "full")
    strong = t[s >= np.median(s)] if len(s) else t
    rows = octave_candidates(env, fps, strong, tempo, bpm_range)
    chosen, why = ({"level": "forced", "bpm": round(float(bpm), 3)}, "--bpm") if bpm else _choose(rows, bpm_range)
    downbeats = tracker.get("downbeats") or []
    ratio = chosen["bpm"] / tempo
    comb_phase, score, near = phase_near(env, fps, chosen["bpm"], beats, ratio)
    correction, attacks = attack_offset(x, chosen["bpm"], comb_phase)
    if meter:
        meter_info = {"source": "--meter"}
    else:
        n, meter_info = tracker_meter(beats, downbeats, ratio)
    grid = Grid(chosen["bpm"], 0.0, n, unit)
    phase, down = bar_line(comb_phase + correction, grid, downbeats)
    grid = replace(grid, phase=round(float(phase), 4))
    drift_ms, tempo_map = drift(x, grid, numbers, beats, ratio, strong)
    info = {"score": round(score, 3), "mean_onset_strength": round(float(env.mean()), 3),
            "level": {"level": chosen["level"], "why": why, "tracker_bpm": round(tempo, 3)},
            "alternatives": [{k: v for k, v in r.items() if k != "phase"} | {"chosen": r is chosen} for r in rows],
            "phase_correction_ms": round(1000 * correction, 1), "attacks": attacks, "comb": near,
            "downbeat": down, "meter": meter_info,
            "swing": swing(x, grid), "drift_ms": drift_ms, "tempo_map": tempo_map,
            "tracker": {k: tracker.get(k) for k in ("name", "version", "checkpoint", "device")}
            | {"beats": len(tracker["beats"]), "on_its_line": len(beats), "downbeats": len(downbeats)}}
    return grid, info


def _without_tracker(x, bpm_range, bpm, meter, n, unit) -> tuple[Grid, dict]:
    if bpm:
        env, fps = _envelope(x)
        score, phase = _scores(env, np.array([bpm]), fps)[0]
        grid, info = Grid(round(float(bpm), 3), round(float(phase / fps), 4)), {
            "score": round(float(score), 3), "mean_onset_strength": round(float(env.mean()), 3)}
    else:
        bpm_range = bpm_range or (100.0, 180.0)
        grid, info = fit_grid(x, bpm_range)
    grid = replace(grid, beats_per_bar=n, beat_unit=unit)
    level = "forced" if bpm else f"bpm_range {bpm_range[0]:g}-{bpm_range[1]:g}"
    return grid, {**info, "level": {"level": level, "why": "no tracker"}, "alternatives": [],
                  "phase_correction_ms": 0.0, "meter": {"source": "--meter" if meter else "default"},
                  "downbeat": None, "swing": swing(x, grid), "drift_ms": None, "tempo_map": None, "tracker": None}


def offsets(x: np.ndarray, grid: Grid, role: str = "snare", tolerance: float = 0.05) -> dict:
    """How far strong onsets sit from the grid, per beat of the bar (ms): the
    check that a tempo is right over the whole track, not just on average."""
    t, strength = strength_peaks(x, role)
    out = {}
    for slot in range(grid.beats_per_bar):
        d = []
        k = 0
        while True:
            g = grid.phase + (k * grid.beats_per_bar + slot) * grid.beat * 4 / grid.beat_unit
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


def tuning(x: np.ndarray, seconds: float = 60.0, fmin: float = 25.0, fmax: float = 2000.0) -> float:
    """Deviation from A440 in semitones (-0.5..0.5) of the pitched content.

    From stable spectral peaks: each frame's peaks (16384-point FFT, parabolic
    interpolation) that stay within ~5 cents over five frames, their offsets from
    equal temperament averaged on the circle (one semitone = one turn) and
    weighted by level. librosa.estimate_tuning was off by up to 60 cents on
    synthetic tones (it histograms every bin, sidelobes and noise included);
    this stays within 1 cent of them.
    """
    import scipy.signal as ss

    x = np.asarray(x, dtype=np.float64)[: int(seconds * SR)]
    n, hop = 16384, 2048
    if len(x) < n:
        return 0.0
    f, _t, Z = ss.stft(x, fs=SR, nperseg=n, noverlap=n - hop, window="hann", boundary=None)
    L = 20 * np.log10(np.abs(Z) + 1e-12)
    frames = []
    for col in L.T:
        top = col.max()
        cur = {}
        if top > -100:
            pk, _ = ss.find_peaks(col, height=top - 30, distance=3)
            pk = pk[(pk > 0) & (pk < len(col) - 1) & (f[pk] > fmin) & (f[pk] < fmax)]
            a, b, c = col[pk - 1], col[pk], col[pk + 1]
            d = 0.5 * (a - c) / np.where(a - 2 * b + c == 0, -1e-12, a - 2 * b + c)
            cents = 1200 * np.log2((pk + d) * SR / n / 440.0)
            for ct, lv in zip(cents, b):
                cur[int(round(ct / 5))] = (float(ct), float(lv))
        frames.append(cur)
    keys = [set(fr) for fr in frames]
    num, den = 0j, 0.0
    for j in range(2, len(frames) - 2):
        for key, (ct, lv) in frames[j].items():
            near = {key - 1, key, key + 1}
            if all(near & keys[jj] for jj in (j - 2, j - 1, j + 1, j + 2)):
                w = 10 ** (lv / 20)
                num += w * np.exp(2j * np.pi * ct / 100)
                den += w
    if den == 0:
        return 0.0
    return round(float(np.angle(num / den) / (2 * np.pi)), 3)
