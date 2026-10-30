"""Calibration v2: beats with every answer kept, one change at a time.

build() renders one scenario of SCENARIOS from library one-shots and writes,
into its folder:
- mix.wav (before the codec), mix.mp3 (256 kb/s, as v1) and mix_mp3.wav (the
  mp3 decoded, moved back by the decoder's delay and cut to mix.wav's length:
  what the pipeline gets);
- stems_truth/<sound>.flac: every sound on its own track at the master's
  small-signal gain, so the stems sum to the mix without the clipper, plus
  the four groups Demucs separates (drums, bass, other, vocals);
- events.json (schema refdissect.events/1, kind "truth"): the grid, every drum
  hit with its gain, velocity and layer, every 808 note with the pitch it
  sounds at and the pitch it settles to (cents included), its cut and its
  glide, the melodic notes and chords (voices.py), and each sound's file,
  hash and attack offset.

A seed's kit is drawn as calibrate.build_beat draws it, so scenario v1 is
build_beat's beat and the other scenarios change one thing against it with
the same sounds ("full" combines several, "producer" plays producer MIDI).
convert_v1() brings the six published v1 beats, rebuilt with truth stems,
into the same layout; verify() checks a folder against its own truth.

t is where a played file starts; t_attack adds audio.attack_time() of the
file as played (lead-in, transposition and cut included): the one definition
of when a sound happens. The audio is made of your library and stays local;
only scores are published (docs/09-data-and-copyright.md).
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.interpolate import PchipInterpolator
from scipy.signal import butter, sosfiltfilt

from . import __version__
from .audio import ATTACK_BAND, BANDS, SR, attack_time, band, read, rms_db, samples, shift, transpose, trim_onset, write
from .calibrate import LEVELS as V1_LEVELS
from .calibrate import NOTES_808, STEP_PATTERN, _pick
from .library import role_of
from .voices import PROGRESSIONS, chord_at, chord_plan, lead_notes, pad_notes, pluck_notes, voice
from .xcorr import ncc

SCHEMA = "refdissect.events/1"
BARS, SLOW_BARS, SLOW_BPM = 16, 8, 100.0  # 8 bars under 100 BPM keep every beat under 60 s (72 BPM: 26.7 s)
TAIL_S = 2.0  # after the last bar, as build_beat: the last tails ring out
MASTER = {"rms_db": -12.0, "drive_db": 5.0, "ceiling": 0.95}  # build_beat's chain: about -8.6 dB RMS out
FADE = samples(5)  # a cut sound fades out over its last 5 ms, as build_beat cuts its 808
# v1's levels; the roles v1 did not have sit under its snare
LEVELS = {**V1_LEVELS, "clap": 0.55, "rim": 0.35, "perc": 0.3, "tom": 0.45, "crash": 0.25}
MELODIC_DB = {"pad": -9.0, "pluck": -10.0, "lead": -6.0}  # each layer's RMS re the 808 track's, under it
GROUPS = {"drums": ("kick", "snare", "clap", "rim", "hat", "open_hat", "perc", "tom", "crash", "ride"),
          "bass": ("808",), "other": ("melodic",), "vocals": ("vocal",)}

# What each scenario changes against v1 (DEFAULTS); the numbers are below
DEFAULTS = {"bpm": 136.0, "meter": "4/4", "swing": None, "start_s": 0.0, "pattern": "v1", "bass": "v1",
            "rolls": 0, "chokes": False, "clap_layer": False, "kick_layer": 0.0, "kick_808": False,
            "drum_pitch": False, "lead_in": False, "rare": False, "flams": False, "dynamics": False,
            "melodic": [], "loop": True, "vocal": True}
SCENARIOS: dict[str, dict] = {
    "v1": {},
    "tempo72": {"bpm": 72.0},  # slow trap and boom bap (8 bars)
    "tempo95": {"bpm": 95.0},  # just under the 100-180 BPM grid search range (8 bars)
    "tempo160": {"bpm": 160.0},  # fast trap
    "meter34": {"meter": "3/4"},
    "meter68": {"meter": "6/8"},
    "swing58": {"swing": 0.58},  # a light 16th swing
    "swing66": {"swing": 0.66},  # close to triplets (2/3)
    "rolls": {"rolls": 2, "chokes": True},
    "layers": {"clap_layer": True, "kick_layer": 0.5},
    "kick808": {"kick_808": True},
    "bass_low": {"bass": "low"},
    "glides": {"bass": "glides"},
    "drum_pitch": {"drum_pitch": True},
    "leadin": {"lead_in": True},
    "rare": {"rare": True},
    "flams": {"flams": True},
    "dynamics": {"dynamics": True},
    "melodic": {"bass": "chords", "melodic": ["pad", "pluck", "lead"], "loop": False},
    "producer": {"pattern": "producer", "bass": "producer", "chokes": True},
    "full": {"swing": 0.54, "rolls": 1, "chokes": True, "clap_layer": True, "bass": "full",
             "melodic": ["pad", "lead"], "loop": False, "start_s": 0.4},
}

# meter: beats per bar, beat unit, sixteenths per bar
METERS = {"4/4": (4, 4, 16), "3/4": (3, 4, 12), "6/8": (6, 8, 12)}
# two bars of sixteenths: kick, snare and open hat steps, and the 16th fills of the hats (on
# eighths) in even and odd two-bar phrases. 4/4 is build_beat's (STEP_PATTERN); 3/4 and 6/8 follow it
PATTERNS = {"4/4": {**STEP_PATTERN, "fills": ({13, 15, 29, 30, 31}, {27, 29})},
            "3/4": {"kick": [0, 6, 12, 15, 18], "snare": [8, 20], "open_hat": [10],
                    "fills": ({9, 11, 21, 22, 23}, {21, 23})},
            "6/8": {"kick": [0, 10, 12, 16], "snare": [6, 18], "open_hat": [22],
                    "fills": ({9, 11, 19, 20, 21}, {19, 21})}}
CLAP_MS, CLAP_GAIN = (0.0, 8.0), (0.5, 1.0)  # a clap layered on the snare, nudged up to 8 ms later, -6..0 dB
KICK_LAYER_GAIN = (0.4, 0.7)  # the layered kick 3-8 dB under the main one
KICK_808_SHARE, KICK_808_MS = 0.3, (0.0, 20.0)  # 30 % each way; 808s nudged late, or late by a lead-in
LOW_ROOTS = (16, 22)  # MIDI E0..A#0: held roots of 20.6-29.1 Hz
LOW_DETUNE = (11.0, -20.0)  # cents on odd and even seeds: the six v1 808s settle -25..+14 cents off A440
GLIDE_SEMIS, GLIDE_MS = (1, 12), (50.0, 300.0)  # a semitone to an octave, over 50-300 ms
GLIDE_TOL_CENTS = 10.0  # an exponential start glide is within 10 cents of its note after `ms`
SHORT_808_S = 0.14  # an 808 note under 140 ms is "short" (a 16th at 107 BPM)
# where an 808 note's pitch is read (s): from 60 ms after it starts, over at most 450 ms, to 20 ms before it
# ends (bass808.sounding hears a transcribed note by this rule too)
SOUNDING = (0.060, 0.450, 0.020)
DRUM_PITCH = {"kick": (2, 0), "snare": (-1, 30), "hat": (0, 25), "open_hat": (-2, 0)}  # semitones, cents
LEAD_IN_MS, LEAD_IN_DB = (5.0, 30.0), -60.0  # quiet noise in front of the attack, as untrimmed files have
RARE_HITS = (1, 12)
FLAM_MS, FLAM_DB = (10.0, 30.0), -6.0  # the grace hit, before its main hit
GHOST_DB, GHOST_STEPS = (-18.0, -12.0), (5, 6, 13, 15, 21, 22, 29, 31)  # before the backbeats, not right after
HAT_GAINS = (0.3, 0.9)  # hats off the beat; the ones on a beat are accents at 1.0
# hat rolls as producers write them: hits per sixteenth and sixteenths long
ROLLS = {"roll32": (2, 4), "roll16t": (1.5, 4), "roll32t": (3, 2), "roll64": (4, 2)}
ROLL_STARTS = (4, 18, 28)  # where in a two-bar phrase a roll replaces the hats (clear of the open hat at 14)
PITCHED_ROLLS, ROLL_SEMIS = 0.4, (1, 5)  # share of rolls whose hits climb or fall, and by how much
PICKUPS = (6, 15, 29)  # "full": 16th-note 808 pickups into the kicks at 7, 16 and 30
LOW_LINES = (  # two bars of (step, sixteenths, interval): 16th notes and same-pitch retriggers on v1's kicks
    ((0, 2, 0), (2, 1, 0), (3, 1, 0), (4, 3, 0), (7, 2, 0), (9, 1, 0), (10, 2, 3), (12, 1, 3), (13, 1, 0),
     (14, 2, 0), (16, 3, 0), (19, 1, 0), (20, 2, 5), (22, 1, 5), (23, 3, 7), (26, 2, 0), (28, 1, 0), (29, 1, 0),
     (30, 2, 10)),
    ((0, 3, 0), (3, 1, 0), (4, 1, 0), (5, 2, 0), (7, 3, 0), (10, 1, 0), (11, 1, 0), (12, 2, 3), (14, 2, 3),
     (16, 2, 0), (18, 1, 0), (19, 1, 0), (20, 3, 0), (23, 2, 5), (25, 1, 5), (26, 2, 7), (28, 1, 0), (29, 1, 0),
     (30, 1, 0), (31, 1, 12)))

MIDI_ROLES = ("kick", "snare", "clap", "hat", "open_hat", "808")
MAX_MIDI_NOTES = 2000  # longer files are songs: the single-role patterns measured hold 4-210 notes
MAX_808_PITCHES = 12  # an 808 line; a scale or melody file can carry the word "808" too
PRODUCER_GLIDE_MS = (60.0, 90.0, 120.0)  # overlapping 808 notes slide over this (by seed)


# ── time and sounds ──────────────────────────────────────────────────────────

@dataclass
class Clock:
    """Where a position in sixteenths from bar 1 sounds: tempo (quarter notes
    per minute, as Live counts), the first bar line, and 16th swing (the second
    sixteenth of each eighth at `swing` of it; positions between move in
    proportion)."""
    bpm: float
    start_s: float = 0.0
    swing: float | None = None

    @property
    def step_s(self) -> float:
        return 60.0 / self.bpm / 4

    def steps(self, p: float) -> float:
        if not self.swing:
            return p
        k = math.floor(p / 2)
        f = p - 2 * k
        r = self.swing
        return 2 * k + (2 * r * f if f <= 1 else 2 * r + 2 * (1 - r) * (f - 1))

    def time(self, p: float) -> float:
        return self.start_s + self.steps(p) * self.step_s

    def sample(self, p: float, dt: float = 0.0) -> int:
        return int(round((self.time(p) + dt) * SR))


@functools.cache
def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Sound:
    """A sound as played: its file, its audio after transposition and lead-in,
    its attack (samples from the start) and its track's level. source is the
    file as loaded, for pitched copies."""
    role: str
    path: str
    audio: np.ndarray
    level: float
    source: np.ndarray | None = None
    semitones: float = 0.0
    cents: float = 0.0
    attack: int = 0
    lead_in: int = 0
    f0_hz: float | None = None
    f0_curve: list | None = None
    digest: str | None = None  # for a file written next to the beat (path relative to its folder)

    def entry(self) -> dict:
        e = {"role": self.role, "path": self.path, "sha256": self.digest or sha256(self.path),
             "semitones": float(self.semitones), "cents": float(self.cents), "attack_s": round(self.attack / SR, 6)}
        if self.lead_in:
            e["lead_in_s"] = round(self.lead_in / SR, 6)
        if self.f0_hz:
            e["f0_hz"] = round(self.f0_hz, 3)
        if self.f0_curve:
            e["f0_curve"] = self.f0_curve
        return e


def make_sound(role: str, path: str, x: np.ndarray, level: float, semitones: float = 0.0, cents: float = 0.0,
               lead_in: np.ndarray | None = None) -> Sound:
    y = transpose(x, semitones + cents / 100) if semitones or cents else x
    if lead_in is not None:
        y = np.r_[lead_in, y].astype(np.float32)
    return Sound(role, path, y, level, x, semitones, cents, round(attack_time(y, role) * SR),
                 0 if lead_in is None else len(lead_in))


def native_f0(x: np.ndarray) -> float | None:
    """The settled pitch (Hz) of a tonal one-shot: the strongest peak between
    16 and 300 Hz over 0.15-1.2 s (a peak an octave or a twelfth lower within
    20 dB wins: the fundamental under a louder 2nd harmonic), refined by the
    spacing of upward zero crossings (zero_cross_hz), over the
    second half of the span where the 25 ms level is within 25 dB of its peak
    (an 808 sample's pitch drop has settled there). None without 0.1 s of
    such a span."""
    lo = band(x, "808").astype(np.float64)
    a, b = int(0.15 * SR), min(len(lo), int(1.2 * SR))
    if b - a < int(0.2 * SR):
        return None
    n_fft = 1 << 18
    spec = np.abs(np.fft.rfft(lo[a:b] * np.hanning(b - a), n_fft))
    f = np.fft.rfftfreq(n_fft, 1 / SR)
    k = int(np.argmax(np.where((f >= 16) & (f <= 300), spec, 0)))
    fc = float(f[k])
    for d in (3, 2):
        near = (f >= fc / d * 0.97) & (f <= fc / d * 1.03)
        if fc / d >= 16 and spec[near].max() >= spec[k] * 0.1:
            fc = float(f[near][np.argmax(spec[near])])
            break
    w = samples(25)
    env = np.sqrt(np.convolve(lo ** 2, np.ones(w) / w, mode="same"))
    quiet = np.flatnonzero(env[int(0.12 * SR):] < env.max() * 10 ** (-25 / 20))
    end = min(len(lo) - int(0.15 * SR), int(0.12 * SR) + (int(quiet[0]) if len(quiet) else len(lo)))
    start = (int(0.12 * SR) + end) // 2
    return zero_cross_hz(lo, fc, start, end) if end - start >= int(0.1 * SR) else None


def zero_cross_hz(x: np.ndarray, near_hz: float, start: int = 0, end: int | None = None) -> float | None:
    """The pitch of a tone near near_hz from x[start:end]: the median spacing
    of upward zero crossings (linearly interpolated, at least 0.6 period
    apart, so a strong 2nd harmonic adds none) after a zero-phase low-pass at
    twice near_hz over all of x. Measured on 0.2-0.33 s synthetic 808 notes
    at 26-47 Hz: within 0.5 cents (a band-pass around the pitch read 5-39
    cents flat there: its ringing reaches the middle of a short note)."""
    y = sosfiltfilt(butter(4, 2 * near_hz, btype="low", fs=SR, output="sos"), np.asarray(x, dtype=np.float64))
    y = y[start:end]
    neg = y < 0
    up = np.flatnonzero(neg[:-1] & ~neg[1:])
    tz = up + y[up] / (y[up] - y[up + 1])
    kept = []
    for z in tz:
        if not kept or z - kept[-1] >= 0.6 * SR / near_hz:
            kept.append(z)
    return float(SR / np.median(np.diff(kept))) if len(kept) >= 3 else None


def crossings(x: np.ndarray, near_hz: float) -> np.ndarray:
    """Seconds of the upward zero crossings of a tone near near_hz (linearly
    interpolated) after a zero-phase low-pass at twice near_hz, one per
    period: a crossing counts when it comes at least 0.6 of the last period
    after the one before (0.3 of near_hz's before there is one), so a strong
    2nd harmonic adds none and the short periods of an 808's pitch drop are
    kept. zero_cross_hz's fixed rule (0.6 of near_hz's period) merges those:
    on four of the suite's eight 808s it reads their first one or two periods
    3.0-5.2 semitones flat where they sit 2.7-12.1 semitones sharp."""
    y = sosfiltfilt(butter(4, 2 * near_hz, btype="low", fs=SR, output="sos"), np.asarray(x, dtype=np.float64))
    neg = y < 0
    up = np.flatnonzero(neg[:-1] & ~neg[1:])
    kept: list[float] = []
    for z in up + y[up] / (y[up] - y[up + 1]):
        gap = 0.6 * (kept[-1] - kept[-2]) if len(kept) >= 2 else 0.3 * SR / near_hz
        if not kept or z - kept[-1] >= gap:
            kept.append(float(z))
    return np.array(kept) / SR


def pitch_track(x: np.ndarray, near_hz: float) -> tuple[Callable, tuple[float, float]] | None:
    """(f, span): f(t) is the frequency (Hz) of a tone near near_hz t seconds
    into x, the slope of a monotone cubic (PCHIP) through its periods counted
    at each crossing() from the second on (the first holds the attack: on
    one v1 808 its period reads 244 cents where the next reads 757), held
    outside span, the first and last of those crossings; None without three.
    Each period's mean frequency at its middle, joined by straight lines,
    puts a steep drop's points too far apart: two readings of the same 110 ms
    bass_low notes (the sample's curve at the note's rate, the note in the
    stem) then differ by up to 54 cents, 11 cents with the cubic."""
    z = crossings(x, near_hz)[1:]
    if len(z) < 3:
        return None
    slope = PchipInterpolator(z, np.arange(len(z), dtype=float)).derivative()
    return (lambda t: slope(np.clip(t, z[0], z[-1]))), (float(z[0]), float(z[-1]))


def pitch_curve(x: np.ndarray, f0: float, upto_s: float = 1.0, step_s: float = 0.005) -> list[list[float]]:
    """[seconds, cents re f0] every step_s over the first upto_s of a tonal
    one-shot (pitch_track, from its second period on): how far a note sits
    above its settled pitch t seconds into the sample (at rate r, t / r
    seconds into the note); before the second crossing (17-65 ms into the
    suite's eight 808s) it holds the first slope. Those 808s start 0-10
    semitones above it there and stay within 5 cents of it after 0.14-0.98 s
    (the 100 ms windows this curve replaced began at 0.1 s, past most of the
    drop)."""
    track = pitch_track(x, f0)
    if track is None:
        return []
    t = np.arange(0.0, min(upto_s, len(x) / SR) + 1e-9, step_s)
    return [[round(float(a), 4), round(float(c), 1)] for a, c in zip(t, 1200 * np.log2(track[0](t) / f0))]


def drop(curve: list, s) -> np.ndarray:
    """A pitch_curve read at sample times s (seconds): cents above the
    settled pitch, linear between its points, held before the first and
    after the last."""
    s = np.asarray(s, dtype=float)
    if not curve:
        return np.zeros_like(s)
    c = np.asarray(curve, dtype=float)
    return np.interp(s, c[:, 0], c[:, 1])


def tuning(x: np.ndarray) -> tuple[float, list]:
    """A tonal one-shot's settled pitch (Hz: native_f0, or with no settled
    stretch the strongest peak between 16 and 300 Hz of the whole file) and
    its pitch_curve."""
    f0 = native_f0(x)
    if f0 is None:
        spec = np.abs(np.fft.rfft(x, 1 << 18))
        f = np.fft.rfftfreq(1 << 18, 1 / SR)
        f0 = float(f[np.argmax(np.where((f >= 16) & (f <= 300), spec, 0))])
    return f0, pitch_curve(x, f0)


def read_window(dur: float) -> tuple[float, float]:
    """Seconds into a note of dur seconds over which its pitch is read
    (SOUNDING); a note under 80 ms, which has no such stretch, is read at
    its middle."""
    a, b = SOUNDING[0], min(SOUNDING[1], dur - SOUNDING[2])
    return (a, b) if b > a else (dur / 2, dur / 2)


def sounding(pitch_at, dur: float) -> float:
    """A note's sounding pitch: the median of pitch_at(seconds into the note)
    over its read_window, every millisecond. An 808 note sits above the
    pitch its sample settles to for as long as the sample's drop lasts, and
    a short note never gets there: a 110 ms bass_low note reads 0.7-2.4
    semitones above it."""
    a, b = read_window(dur)
    return float(np.median(pitch_at(np.arange(a, b + 5e-4, 0.001))))


def hz_to_midi(f: float) -> float:
    return 69 + 12 * math.log2(f / 440.0)


def _draw(rows: list[dict], rng: random.Random, role: str, lo: float, hi: float, exclude=(),
          pattern: str | None = None) -> tuple[str, np.ndarray]:
    """A file of the role (its path matching pattern) loaded as calibrate._pick
    loads it, other than the ones excluded."""
    pool = sorted(r["path"] for r in rows if r["role"] == role and r["duration"] and lo <= r["duration"] <= hi
                  and r["path"] not in exclude and (not pattern or re.search(pattern, r["path"].lower())))
    for path in rng.sample(pool, len(pool)):
        try:
            x = read(path)
        except Exception:
            continue
        if np.max(np.abs(x)) > 0.05:
            return path, trim_onset(x) / np.max(np.abs(x))
    raise ValueError(f"No {role} between {lo} and {hi} s in the inventory")


# sounds a scenario adds to the kit: role, inventory role, duration range (s), path pattern
EXTRA = {"kick2": ("kick", "kick", 0.15, 1.5, None), "snare": ("snare", "snare", 0.1, 1.5, None),
         "clap": ("clap", "clap", 0.1, 1.5, None), "perc": ("perc", "perc", 0.03, 1.5, r"^(?!.*\btoms?\b)"),
         "tom": ("tom", "perc", 0.1, 2.0, r"\btoms?\b"), "crash": ("crash", "crash/ride", 0.5, 6.0, r"crash"),
         "rim": ("rim", "rim", 0.03, 1.5, None)}


def draw_kit(rows: list[dict], rng: random.Random) -> dict[str, tuple[str, str, np.ndarray]]:
    """build_beat's draw in its order, so a seed gives build_beat's files:
    sound id -> (role, path, audio onset-trimmed and peak-normalized). The
    backbeat is a snare or a clap, named by its role."""
    kit = {"kick": ("kick", *_pick(rows, rng, {"kick"}, 0.15, 1.5))}
    role = rng.choice(["snare", "clap"])
    kit[role] = (role, *_pick(rows, rng, {role}, 0.1, 1.5))
    kit["hat"] = ("hat", *_pick(rows, rng, {"hat"}, 0.03, 0.6))
    kit["open_hat"] = ("open_hat", *_pick(rows, rng, {"open_hat"}, 0.2, 1.5))
    kit["808"] = ("808", *_pick(rows, rng, {"808"}, 0.8, 4.0))
    kit["loop"] = ("melodic", *_pick(rows, rng, {"melodic/loop"}, 8, 120))
    kit["vocal"] = ("vocal", *_pick(rows, rng, {"vox"}, 8, 300))
    return kit


class Beat:
    """One scenario being built: its clock, sounds, tracks (one per sound) and
    truth."""

    def __init__(self, clock: Clock, n: int, bars: int, meter: str):
        self.clock, self.n, self.bars, self.meter = clock, n, bars, meter
        self.steps_per_bar = METERS[meter][2]
        self.sounds: dict[str, Sound] = {}
        self.tracks: dict[str, np.ndarray] = {}
        self.backbeat = "snare"
        self.native_midi = 0.0
        self.drums: list[dict] = []
        self.bass808: list[dict] = []
        self.notes: list[dict] = []
        self.chords: list[dict] = []

    def add(self, sid: str, x: np.ndarray, a: int, gain: float = 1.0) -> None:
        track = self.tracks.setdefault(sid, np.zeros(self.n))
        if a < 0:
            x, a = x[-a:], 0
        b = min(self.n, a + len(x))
        if b > a:
            track[a:b] += gain * x[: b - a]

    def pitched(self, sid: str, semitones: int) -> str:
        """The id of sid played `semitones` higher (created on first use)."""
        if not semitones:
            return sid
        name = f"{sid}{semitones:+d}"
        if name not in self.sounds:
            s = self.sounds[sid]
            self.sounds[name] = make_sound(s.role, s.path, s.source, s.level, s.semitones + semitones, s.cents)
        return name


def _t(a: int) -> float:
    return round(a / SR, 6)


def _cut(x: np.ndarray, length: int) -> np.ndarray:
    """x cut to length samples, its last 5 ms faded out (as build_beat cuts its 808)."""
    x = x[: max(0, length)].copy()
    if len(x) == length and length >= FADE:
        x[-FADE:] *= np.linspace(1, 0, FADE)
    return x


def velocity(gain: float) -> int:
    """Our velocity scale: a hit's gain (re its track's level) is velocity / 127."""
    return int(np.clip(round(127 * gain), 1, 127))


# ── patterns ─────────────────────────────────────────────────────────────────

def _hit(sound: str, p: float, gain: float = 1.0, *flags: str, dt: float = 0.0, layer_of: str | None = None) -> dict:
    return {"sound": sound, "p": p, "gain": gain, "dt": dt, "flags": set(flags), "layer_of": layer_of, "cut": None}


def _note(p: float, end: float, semis: float, with_kick: bool = True, *flags: str) -> dict:
    return {"p": p, "end": end, "semis": semis, "gain": 1.0, "dt": 0.0, "with_kick": with_kick, "glide": None,
            "flags": set(flags)}


def pattern(beat: Beat, rng: random.Random) -> tuple[list[dict], list[dict]]:
    """Two bars of PATTERNS[meter], repeated: kick, backbeat, hats on every
    eighth with 16th fills (gain 0.55-1.0, drawn in build_beat's order), the
    open hat, an 808 note on every kick (NOTES_808) cut at the next. In 4/4
    this is build_beat's beat."""
    pat, span = PATTERNS[beat.meter], 2 * beat.steps_per_bar
    drums, bass = [], []
    for pair in range(beat.bars // 2):
        o = pair * span
        drums += [_hit("kick", o + s) for s in pat["kick"]]
        drums += [_hit(beat.backbeat, o + s) for s in pat["snare"]]
        hats = set(range(0, span, 2)) | pat["fills"][pair % 2]
        drums += [_hit("hat", o + s, rng.uniform(0.55, 1.0)) for s in sorted(hats - set(pat["open_hat"]))]
        drums += [_hit("open_hat", o + s, 0.9) for s in pat["open_hat"]]
        kicks = pat["kick"]
        bass += [_note(o + s, o + e, st) for s, e, st in zip(kicks, kicks[1:] + [span], NOTES_808)]
    return drums, bass


def add_rolls(beat: Beat, drums: list[dict], rng: random.Random, per_pair: int) -> None:
    """Hat rolls in place of the hats they cover: 32nds, 16th and 32nd
    triplets, 64th bursts, each as often; PITCHED_ROLLS of them climb or fall
    1-5 semitones over the roll; velocity rises or falls 0.55-1.0."""
    count = beat.bars // 2 * per_pair
    figures = (list(ROLLS) * (count // len(ROLLS) + 1))[:count]
    rng.shuffle(figures)
    pitched = set(rng.sample(range(count), round(PITCHED_ROLLS * count)))
    k = 0
    for pair in range(beat.bars // 2):
        o = pair * 32
        for start in sorted(rng.sample(ROLL_STARTS, per_pair)):
            per, length = ROLLS[figures[k]]
            a, b = o + start, o + start + length
            drums[:] = [d for d in drums if not (d["sound"] == "hat" and a <= d["p"] < b)]
            hits = int(per * length)
            ramp = rng.choice([1, -1]) * rng.randint(*ROLL_SEMIS) if k in pitched else 0
            up = rng.random() < 0.5
            for i in range(hits):
                x = i / (hits - 1)
                st = round(ramp * x)
                gain = 0.55 + 0.45 * (x if up else 1 - x)
                flags = ["roll", figures[k]] + (["pitched"] if ramp else [])
                drums.append(_hit(beat.pitched("hat", st), a + i / per, gain, *flags))
            k += 1


def choke(beat: Beat, drums: list[dict]) -> None:
    """An open hat stops where the next closed hat starts (a choke group)."""
    closed = sorted((d["p"], d["dt"]) for d in drums if beat.sounds[d["sound"]].role == "hat")
    for d in drums:
        if beat.sounds[d["sound"]].role != "open_hat":
            continue
        nxt = next(((p, dt) for p, dt in closed if (p, dt) > (d["p"], d["dt"])), None)
        if nxt is None:
            continue
        length = beat.clock.sample(*nxt) - beat.clock.sample(d["p"], d["dt"])
        if length < len(beat.sounds[d["sound"]].audio):
            d["cut"] = nxt
            d["flags"].add("choked")


def add_layers(beat: Beat, drums: list[dict], rng: random.Random, clap: bool, kick_share: float) -> None:
    """A clap on every snare (0-8 ms behind, gain 0.5-1.0) and a second kick
    under kick_share of the kicks (gain 0.4-0.7), each its own sound."""
    if clap:
        for d in [d for d in drums if d["sound"] == "snare" and "grace" not in d["flags"]]:
            drums.append(_hit("clap", d["p"], rng.uniform(*CLAP_GAIN), dt=d["dt"] + rng.uniform(*CLAP_MS) / 1000,
                              layer_of="snare"))
    if kick_share:
        kicks = [d for d in drums if d["sound"] == "kick"]
        for d in sorted(rng.sample(kicks, round(kick_share * len(kicks))), key=lambda d: d["p"]):
            drums.append(_hit("kick2", d["p"], rng.uniform(*KICK_LAYER_GAIN), dt=d["dt"], layer_of="kick"))


def kick_808(drums: list[dict], bass: list[dict], rng: random.Random) -> None:
    """KICK_808_SHARE of the 808 notes lose their kick, as many kicks lose their
    808 (the note before rings on to the next), and the 808s that keep their
    kick start 0-20 ms after it."""
    k = round(KICK_808_SHARE * len(bass))
    chosen = rng.sample(range(len(bass)), 2 * k)
    no_kick, no_808 = set(chosen[:k]), set(chosen[k:])
    kick_at = {d["p"]: d for d in drums if d["sound"] == "kick"}
    for i in sorted(no_kick):
        drums.remove(kick_at[bass[i]["p"]])
        bass[i]["with_kick"] = False
    for i in sorted(no_808):
        kick_at[bass[i]["p"]]["flags"].add("no_808")
    keep = [b for i, b in enumerate(bass) if i not in no_808]
    for b, nxt in zip(keep, keep[1:]):
        b["end"] = nxt["p"]
    for b in keep:
        if b["with_kick"]:
            b["dt"] = rng.uniform(*KICK_808_MS) / 1000
    bass[:] = keep


def add_rare(beat: Beat, drums: list[dict], rng: random.Random) -> None:
    """Perc, tom, crash and rim, 1-12 hits each, where they usually play:
    crashes on bar lines, toms in the last beat of a bar, perc off the beat,
    rims on beats 2 and 4."""
    spb, steps = beat.steps_per_bar, range(beat.bars * beat.steps_per_bar)
    where = {"crash": [p for p in steps if p % spb == 0], "tom": [p for p in steps if p % spb >= spb - 4],
             "perc": [p for p in steps if p % 2], "rim": [p for p in steps if p % spb in (4, 12)]}
    for sid, cands in where.items():
        for p in sorted(rng.sample(cands, min(len(cands), rng.randint(*RARE_HITS)))):
            drums.append(_hit(sid, p, rng.uniform(0.7, 1.0)))


def add_flams(beat: Beat, drums: list[dict], rng: random.Random) -> None:
    """Every backbeat hit becomes a flam: the same sound 10-30 ms earlier at -6 dB."""
    for d in [d for d in drums if d["sound"] == beat.backbeat]:
        d["flags"].add("flam")
        drums.append(_hit(d["sound"], d["p"], d["gain"] * 10 ** (FLAM_DB / 20), "flam", "grace",
                          dt=d["dt"] - rng.uniform(*FLAM_MS) / 1000))


def dynamics(beat: Beat, drums: list[dict], rng: random.Random) -> None:
    """Hats on a beat accented (1.0), the others 0.3-0.9; backbeats accented;
    three ghost snares per two bars at -18..-12 dB."""
    for d in drums:
        if d["sound"] == "hat":
            accent = d["p"] % 4 == 0
            d["gain"] = 1.0 if accent else rng.uniform(*HAT_GAINS)
            d["flags"] |= {"accent"} if accent else set()
        elif d["sound"] == beat.backbeat:
            d["flags"].add("accent")
    for pair in range(beat.bars // 2):
        for s in sorted(rng.sample(GHOST_STEPS, 3)):
            drums.append(_hit(beat.backbeat, pair * 32 + s, 10 ** (rng.uniform(*GHOST_DB) / 20), "ghost"))


def mark(beat: Beat, drums: list[dict]) -> None:
    """Flags that follow from the pattern: hats under a backbeat, swung hits."""
    backbeats = {(d["p"], d["dt"]) for d in drums if d["sound"] == beat.backbeat and "grace" not in d["flags"]}
    for d in drums:
        if beat.sounds[d["sound"]].role == "hat" and (d["p"], d["dt"]) in backbeats:
            d["flags"].add("under_snare")
        if beat.clock.swing and d["p"] % 2 == 1:
            d["flags"].add("swung")


# ── 808 lines ────────────────────────────────────────────────────────────────

def low_line(beat: Beat, rng: random.Random, cents: float) -> list[dict]:
    """LOW_LINES on a root of 20-30 Hz, every note `cents` off A440."""
    root, kicks = rng.randint(*LOW_ROOTS), set(STEP_PATTERN["kick"])
    bass = []
    for pair in range(beat.bars // 2):
        for s, length, iv in LOW_LINES[pair % 2]:
            p = pair * 32 + s
            bass.append(_note(p, p + length, root + iv + cents / 100 - beat.native_midi, s in kicks))
    return bass


def add_glides(beat: Beat, bass: list[dict], rng: random.Random, share: float = 1.0) -> None:
    """Glides on the 808: a quarter of the notes start 1-12 semitones off and
    fall or rise into their pitch (exponential), a sixth end on a linear slide
    away from it, a sixth slide in from the note before without a new attack
    (legato). share scales the counts."""
    k = len(bass)
    kinds = ["start"] * round(share * k / 4) + ["end"] * round(share * k / 6) + ["legato"] * round(share * k / 6)
    kinds += [None] * (k - len(kinds))
    rng.shuffle(kinds)
    for i, (b, kind) in enumerate(zip(bass, kinds)):
        dur = (beat.clock.steps(b["end"]) - beat.clock.steps(b["p"])) * beat.clock.step_s * 1000
        prev = bass[i - 1] if i else None
        if kind == "legato" and (prev is None or prev["semis"] == b["semis"] or prev["end"] != b["p"]
                                 or (prev["glide"] or {}).get("kind") == "end" or dur < 80):
            kind = None
        if kind == "end" and (dur < 200 or (i + 1 < k and kinds[i + 1] == "legato")):
            kind = None
        if kind == "start":
            g = rng.randint(*GLIDE_SEMIS) * (1 if rng.random() < 0.6 else -1)
            b["glide"] = {"kind": "start", "semis": g, "ms": round(rng.uniform(*GLIDE_MS), 1)}
        elif kind == "end":
            g = rng.randint(*GLIDE_SEMIS) * rng.choice([1, -1])
            ms = round(rng.uniform(GLIDE_MS[0], min(GLIDE_MS[1], dur - 40)), 1)
            b["glide"] = {"kind": "end", "semis": g, "ms": ms}
        elif kind == "legato":
            ms = round(rng.uniform(GLIDE_MS[0], min(GLIDE_MS[1], dur - 20)), 1)
            b["glide"] = {"kind": "legato", "semis": 0, "ms": ms}


def chord_line(beat: Beat, chords: list[dict], pickups: tuple = ()) -> list[dict]:
    """The 808 on v1's kicks (and 16th pickups into them), on the root of the
    chord that sounds there, MIDI 24-35 (33-62 Hz)."""
    kicks = STEP_PATTERN["kick"]
    steps = sorted(set(kicks) | set(pickups))
    bass = []
    for pair in range(beat.bars // 2):
        o = pair * 32
        for s, e in zip(steps, steps[1:] + [32]):
            at = o + e if s in pickups else o + s  # a pickup plays the note it leads into
            root = chord_at(chords, min(at, beat.bars * 16 - 1))["root"]
            bass.append(_note(o + s, o + e, 24 + (root - 24) % 12 - beat.native_midi, s in kicks))
    return bass


def mark_808(beat: Beat, bass: list[dict]) -> None:
    for prev, b in zip([None] + bass[:-1], bass):
        if (beat.clock.steps(b["end"]) - beat.clock.steps(b["p"])) * beat.clock.step_s < SHORT_808_S:
            b["flags"].add("short")
        legato = (b["glide"] or {}).get("kind") == "legato"
        if prev is not None and not legato and prev["semis"] == b["semis"] and prev["end"] == b["p"]:
            b["flags"].add("retrigger")


# ── melodic layers ───────────────────────────────────────────────────────────

def play_melodic(beat: Beat, layer: str, notes: list[dict], rng: random.Random) -> None:
    """Render a layer's notes into its track (velocity -> amplitude (v/127)^2)."""
    clock = beat.clock
    groups: list[list[dict]] = []
    for nt in notes:
        if nt.get("glide_ms") and groups:
            groups[-1].append(nt)
        else:
            groups.append([nt])
    for g in groups:
        a = clock.sample(g[0]["p"])
        n_on = clock.sample(g[-1]["end"]) - a
        midi = np.full(n_on, float(g[0]["midi"]))
        for prev, nt in zip(g, g[1:]):
            s = clock.sample(nt["p"]) - a
            ramp = np.clip(np.arange(n_on - s) / max(1, samples(nt["glide_ms"])), 0, 1)
            midi[s:] = prev["midi"] + (nt["midi"] - prev["midi"]) * ramp
        beat.add(layer, voice(layer, midi, n_on, rng.random()), a, (g[0]["velocity"] / 127) ** 2)
    for nt in notes:
        beat.notes.append({"layer": layer, "t": _t(clock.sample(nt["p"])), "end": _t(clock.sample(nt["end"])),
                           "midi": int(nt["midi"]), "cents": 0.0, "velocity": int(nt["velocity"]),
                           "flags": sorted(nt["flags"])})


def melodic_sound(layer: str, out: Path) -> Sound:
    """The layer's voice as a one-shot file (C4 or C5) in out/sounds, so it has
    a path (relative to out) and a hash like any sound."""
    midi = 60.0 if layer == "pad" else 72.0
    n_on = int((1.0 if layer == "pad" else 0.15 if layer == "pluck" else 0.8) * SR)
    path = write(out / "sounds" / f"{layer}.flac", voice(layer, np.full(n_on, midi), n_on, 0.0), peak=0.9)
    x = read(path)
    return Sound("melodic", f"sounds/{layer}.flac", x, 1.0, attack=round(attack_time(x, "melodic") * SR),
                 digest=sha256(str(path)))


# ── producer MIDI ────────────────────────────────────────────────────────────

def read_midi(path: str | Path) -> dict | None:
    """Notes of a MIDI file in beats: (start, length, pitch, velocity) sorted,
    its first tempo and the channels its notes use; None if it does not parse."""
    import mido
    try:
        m = mido.MidiFile(str(path))
    except Exception:
        return None
    notes, channels, bpm = [], set(), None
    for track in m.tracks:
        t, on = 0, {}
        for msg in track:
            t += msg.time
            if msg.type == "set_tempo" and bpm is None:
                bpm = round(mido.tempo2bpm(msg.tempo), 2)
            elif msg.type == "note_on" and msg.velocity > 0:
                on.setdefault((msg.channel, msg.note), []).append((t, msg.velocity))
                channels.add(msg.channel)
            elif msg.type in ("note_on", "note_off") and on.get((msg.channel, msg.note)):
                t0, v = on[(msg.channel, msg.note)].pop(0)
                notes.append((t0 / m.ticks_per_beat, (t - t0) / m.ticks_per_beat, msg.note, v))
    return {"notes": sorted(notes), "bpm": bpm, "channels": channels}


def producer_files(midi_dir: str | Path) -> dict[str, list[str]]:
    """Single-role pattern files under midi_dir, by the role words in their
    names (library.role_of): one channel, not GM drums (channel 10), at most
    MAX_MIDI_NOTES notes; an 808 file at most MAX_808_PITCHES pitches. macOS
    AppleDouble stubs (._name) are skipped."""
    found: dict[str, list[str]] = {r: [] for r in MIDI_ROLES}
    for path in sorted(Path(midi_dir).expanduser().rglob("*")):
        if path.name.startswith("._") or path.suffix.lower() not in (".mid", ".midi"):
            continue
        role = role_of(re.sub(r"[_\W]+", " ", path.stem))
        if role not in found:
            continue
        m = read_midi(path)
        if (not m or len(m["notes"]) < 2 or len(m["notes"]) > MAX_MIDI_NOTES or len(m["channels"]) != 1
                or 9 in m["channels"]):
            continue
        if role == "808" and len({q[2] for q in m["notes"]}) > MAX_808_PITCHES:
            continue
        found[role].append(str(path))
    return found


def looped(notes: list[tuple], total: float) -> list[tuple]:
    """A pattern's notes repeated to fill `total` beats, the loop a whole
    number of bars; the same pitch at the same time once."""
    end = max(s + d for s, d, _, _ in notes)
    loop = 4.0 * max(1, math.ceil((end - 1e-6) / 4.0), math.floor(max(s for s, *_ in notes) / 4.0 + 1e-9) + 1)
    out, seen = [], set()
    for k in range(math.ceil(total / loop)):
        for s, d, pitch, v in notes:
            s += k * loop
            if s < total - 1e-9 and (round(s, 4), pitch) not in seen:
                seen.add((round(s, 4), pitch))
                out.append((s, min(d, total - s), pitch, v))
    return sorted(out)


def _fold(semis: float) -> int:
    """Semitones folded into -12..12 (pattern files are written around C3)."""
    semis = int(semis)
    while semis > 12:
        semis -= 12
    while semis < -12:
        semis += 12
    return semis


def _roll_flag(gap_beats: float) -> str | None:
    for flag, gap in (("roll64", 1 / 16), ("roll32t", 1 / 12), ("roll32", 1 / 8), ("roll16t", 1 / 6)):
        if gap_beats <= gap + 1e-6:
            return flag
    return None


def producer_drums(beat: Beat, files: dict[str, str], total: float) -> list[dict]:
    """Each role's file looped over the bars; pitched notes (re the file's most
    common note) play pitched copies; hits of a role within 5 ms merge (the
    loudest stays); a closed hat under an open hat is dropped."""
    drums = []
    for role in ("kick", "snare", "clap", "hat", "open_hat"):
        if role not in files:
            continue
        notes = looped(read_midi(files[role])["notes"], total)
        common = max(set(q[2] for q in notes), key=[q[2] for q in notes].count)
        merged: list[tuple] = []
        for q in sorted(notes, key=lambda q: (q[0], -q[3])):
            if not merged or (q[0] - merged[-1][0]) * 60 / beat.clock.bpm >= 0.005:
                merged.append(q)
        starts = [q[0] for q in merged]
        for i, (s, _, pitch, v) in enumerate(merged):
            gaps = [abs(s - starts[j]) for j in (i - 1, i + 1) if 0 <= j < len(starts)]
            flag = _roll_flag(min(gaps)) if gaps else None
            st = _fold(pitch - common)
            flags = ([flag, "roll"] if flag else []) + (["pitched"] if st else [])
            drums.append(_hit(beat.pitched(role, st), 4 * s, v / 127, *flags))
    opens = {d["p"] for d in drums if d["sound"].startswith("open_hat")}
    return [d for d in drums if not (beat.sounds[d["sound"]].role == "hat" and d["p"] in opens)]


def producer_808(beat: Beat, path: str, total: float, root: int, glide_ms: float) -> list[dict]:
    """The 808 file looped, one voice (the highest of notes that start
    together): a note that starts inside the one before at another pitch slides
    to it (legato, linear over glide_ms); the line's most common note plays at
    MIDI `root`."""
    notes = looped(read_midi(path)["notes"], total)
    mono = []
    for q in sorted(notes, key=lambda q: (q[0], -q[2])):
        if not mono or q[0] - mono[-1][0] > 1e-6:
            mono.append(q)
    common = max(set(q[2] for q in mono), key=[q[2] for q in mono].count)
    bass = []
    for i, (s, d, pitch, _) in enumerate(mono):
        nxt = mono[i + 1][0] if i + 1 < len(mono) else total
        b = _note(4 * s, 4 * min(s + max(d, 1 / 16), nxt), root + _fold(pitch - common) - beat.native_midi, False)
        prev = mono[i - 1] if i else None
        if prev and s < prev[0] + prev[1] - 1e-6 and pitch != prev[2]:
            b["glide"] = {"kind": "legato", "semis": 0, "ms": glide_ms}
            bass[-1]["end"] = b["p"]
        bass.append(b)
    attacks = [b["p"] for b in bass if not b["glide"]]
    for b in bass:
        gaps = [abs(b["p"] - a) for a in attacks if a != b["p"]]
        b["flags"] |= {"roll"} if gaps and min(gaps) < 4 * 0.25 - 1e-6 else set()
    return bass


def producer_kicks(bass: list[dict]) -> list[dict]:
    """With no kick file: a kick on every attacked 808 note on an eighth-note
    line and not in a roll (the trap convention of kick and 808 together)."""
    drums = []
    for b in bass:
        if not b["glide"] and abs(b["p"] / 2 - round(b["p"] / 2)) < 1e-6 and "roll" not in b["flags"]:
            b["with_kick"] = True
            drums.append(_hit("kick", b["p"]))
    return drums


# ── rendering ────────────────────────────────────────────────────────────────

def play_drums(beat: Beat, drums: list[dict]) -> None:
    for d in sorted(drums, key=lambda d: (d["p"], d["dt"], d["sound"])):
        s = beat.sounds[d["sound"]]
        a = beat.clock.sample(d["p"], d["dt"])
        x = s.audio if d["cut"] is None else _cut(s.audio, beat.clock.sample(*d["cut"]) - a)
        beat.add(d["sound"], x, a, d["gain"])
        beat.drums.append({"sound": d["sound"], "role": s.role, "t": _t(a), "t_attack": _t(a + s.attack),
                           "gain": round(s.level * d["gain"], 6), "velocity": velocity(d["gain"]),
                           "layer_of": d["layer_of"], "flags": sorted(d["flags"])})


def varispeed(x: np.ndarray, semis: np.ndarray) -> np.ndarray:
    """x read at a rate that follows `semis` (per output sample), as a sampler
    bends a note: pitch and speed move together."""
    pos = np.r_[0.0, np.cumsum(2 ** (semis[:-1] / 12))]
    return np.interp(pos, np.arange(len(x)), x, right=0.0)


def _curve(beat: Beat, chain: list[dict], a: int, length: int) -> np.ndarray:
    """Semitones re the sample's own pitch, per sample, of a legato chain."""
    st = np.zeros(length)
    starts = [beat.clock.sample(b["p"], b["dt"]) - a for b in chain] + [length]
    for i, b in enumerate(chain):
        s, e = max(0, starts[i]), min(length, starts[i + 1])
        t = np.arange(e - s) / SR
        st[s:e] = b["semis"]
        g = b["glide"]
        if g and g["kind"] == "start":
            tau = g["ms"] / 1000 / math.log(abs(g["semis"]) * 100 / GLIDE_TOL_CENTS)
            st[s:e] += g["semis"] * np.exp(-t / tau)
        elif g and g["kind"] == "legato" and i:
            ramp = np.clip(t / (g["ms"] / 1000), 0, 1)
            st[s:e] = chain[i - 1]["semis"] + (b["semis"] - chain[i - 1]["semis"]) * ramp
        if g and g["kind"] == "end" and len(t):
            st[s:e] += g["semis"] * np.clip((t - (t[-1] - g["ms"] / 1000)) / (g["ms"] / 1000), 0, 1)
    return st


def _end(y: np.ndarray, length: int) -> int:
    """Samples until a played sound stops: its cut, or the last sample it has before that."""
    nz = np.flatnonzero(y[:length])
    return int(nz[-1]) + 1 if len(nz) and nz[-1] + 1 < length else length


def play_808(beat: Beat, bass: list[dict]) -> None:
    """Each attacked note (with the legato notes after it) is one voice, cut
    at the next attacked note or its own end, whichever comes first. A
    voice's pitch, per sample, is its transposition (the glide curve when it
    has one) plus the sample's own drop where the voice is reading the
    sample (f0_curve)."""
    s = beat.sounds["808"]
    chains: list[list[dict]] = []
    for b in sorted(bass, key=lambda b: b["p"]):
        if (b["glide"] or {}).get("kind") == "legato" and chains:
            chains[-1].append(b)
        else:
            chains.append([b])
    starts = [beat.clock.sample(c[0]["p"], c[0]["dt"]) for c in chains]
    for i, chain in enumerate(chains):
        a = starts[i]
        own = beat.clock.sample(chain[-1]["end"], chain[-1]["dt"])
        length = (min(own, starts[i + 1]) if i + 1 < len(chains) else own) - a
        if len(chain) == 1 and not chain[0]["glide"]:
            full = transpose(s.audio, chain[0]["semis"])
            semis = np.full(length, float(chain[0]["semis"]))
            read_at = np.arange(length) * 2 ** (chain[0]["semis"] / 12)  # where transpose reads the sample
        else:
            semis = _curve(beat, chain, a, length)
            full = varispeed(s.audio, semis)
            read_at = np.r_[0.0, np.cumsum(2 ** (semis[:-1] / 12))]
        pitch = beat.native_midi + semis + drop(s.f0_curve, read_at / SR) / 100
        y = _cut(full, length)
        beat.add("808", y, a, chain[0]["gain"])
        attack = a + round(attack_time(y, "808") * SR)
        bounds = [beat.clock.sample(b["p"], b["dt"]) for b in chain[1:]] + [a + _end(full, length)]
        for k, (b, stop) in enumerate(zip(chain, bounds)):
            start = beat.clock.sample(b["p"], b["dt"]) if k else a
            heard = sounding(functools.partial(_at, pitch, start - a), (stop - start) / SR)
            beat.bass808.append(_bass_event(beat, b, start, None if k else attack, stop, s.level * b["gain"],
                                            heard, chain[k - 1]["semis"] if k else None))


def _at(pitch: np.ndarray, offset: int, t: np.ndarray) -> np.ndarray:
    """A per-sample pitch read t seconds after sample `offset` (held at its ends)."""
    return pitch[np.clip(offset + np.round(t * SR).astype(int), 0, len(pitch) - 1)]


def _bass_event(beat: Beat, b: dict, a: int, attack: int | None, stop: int, gain: float, heard: float,
                prev_semis: float | None = None) -> dict:
    """An 808 note's truth. midi is its sounding pitch, `heard` (sounding()
    over the voice's pitch: transposition, glide and the sample's own drop);
    midi_settled is the pitch it settles to (the sample's settled pitch, its
    sound's f0_hz, plus the transposition): what a producer wrote, the
    sample's tuning included. A glide goes from from_midi to to_midi (settled
    pitches), starting at t, over ms; a legato note has no attack of its own
    (t_attack = t)."""
    midi = beat.native_midi + b["semis"]
    g, glide, flags = b["glide"], None, set(b["flags"])
    if g and g["kind"] == "start":
        glide = {"from_midi": round(midi + g["semis"], 4), "to_midi": round(midi, 4), "t": _t(a), "ms": g["ms"],
                 "shape": "exp"}
        flags.add("start_glide")
    elif g and g["kind"] == "legato":
        glide = {"from_midi": round(beat.native_midi + prev_semis, 4), "to_midi": round(midi, 4), "t": _t(a),
                 "ms": g["ms"], "shape": "lin"}
        flags.add("legato")
    elif g and g["kind"] == "end":
        glide = {"from_midi": round(midi, 4), "to_midi": round(midi + g["semis"], 4),
                 "t": _t(stop - samples(g["ms"])), "ms": g["ms"], "shape": "lin"}
        flags.add("end_glide")
    return {"sound": "808", "t": _t(a), "t_attack": _t(attack if attack is not None else a), "end": _t(stop),
            "midi": round(heard, 4), "midi_settled": round(midi, 4), "glide": glide, "with_kick": bool(b["with_kick"]),
            "gain": round(gain, 6), "flags": sorted(flags)}


# ── build ────────────────────────────────────────────────────────────────────

def _sounds(beat: Beat, rows: list[dict], kit: dict, sw: dict, rng: random.Random, seed: int,
            midi: dict | None) -> None:
    """The kit as this scenario plays it, plus the extra sounds it needs (each
    drawn from its own seeded stream, so every scenario of a seed gets the same
    ones; never a file of the kit)."""
    used = {path for _, path, _ in kit.values()}
    backbeat = next(k for k in kit if k in ("snare", "clap"))
    need = set()
    if sw["clap_layer"]:
        need |= {"snare", "clap"}
    if sw["kick_layer"]:
        need.add("kick2")
    if sw["rare"]:
        need |= {"perc", "tom", "crash", "rim"}
    if midi:
        need |= {r for r in ("snare", "clap") if r in midi["files"]}
    beat.backbeat = "snare" if "snare" in need else backbeat
    pitch = dict(DRUM_PITCH) if sw["drum_pitch"] else {}
    if pitch:
        pitch["kick"] = (2 if seed % 2 else -2, 0)
    for sid, (role, path, x) in kit.items():
        if sid in ("loop", "vocal"):
            continue
        st, cents = pitch.get("snare" if sid == backbeat else sid, (0, 0))
        lead = None
        if sw["lead_in"] and sid != "808":
            n_lead = samples(rng.uniform(*LEAD_IN_MS))
            lead = np.random.default_rng(rng.getrandbits(32)).standard_normal(n_lead) * 10 ** (LEAD_IN_DB / 20)
        beat.sounds[sid] = make_sound(role, path, x, LEVELS[role], st, cents, lead)
    for sid in sorted(need - set(beat.sounds)):
        role, kind, lo, hi, pattern = EXTRA[sid]
        path, x = _draw(rows, random.Random(f"{sid}/{seed}"), kind, lo, hi, used, pattern)
        used.add(path)
        beat.sounds[sid] = make_sound(role, path, x, LEVELS[role])
    _tune(beat)


def _tune(beat: Beat) -> None:
    """The 808's settled pitch and its pitch curve (tuning())."""
    s808 = beat.sounds["808"]
    s808.f0_hz, s808.f0_curve = tuning(s808.audio)
    beat.native_midi = hz_to_midi(s808.f0_hz)


def _parts(beat: Beat, sw: dict, seed: int, kit_rng: random.Random, rng: random.Random,
           midi: dict | None) -> tuple[list[dict], list[dict], list[dict]]:
    """Drum hits and 808 notes of the scenario (patterns, then its changes), and its chords."""
    total = beat.bars * beat.steps_per_bar / 4
    if midi:
        bass = producer_808(beat, midi["files"]["808"], total, rng.randint(26, 33),
                            PRODUCER_GLIDE_MS[seed % 3]) if "808" in midi["files"] else []
        drums = producer_drums(beat, midi["files"], total)
        if "kick" not in midi["files"]:
            drums += producer_kicks(bass)
        kicks = {d["p"] for d in drums if d["sound"] == "kick"}
        for b in bass:
            b["with_kick"] = b["p"] in kicks
    else:
        drums, bass = pattern(beat, kit_rng)
    chords = []
    if sw["melodic"]:
        prog = PROGRESSIONS[(seed - 1) % len(PROGRESSIONS)]
        chords = chord_plan(beat.bars, random.Random(f"key/{seed}").randrange(12), prog)
    if sw["bass"] == "low":
        bass = low_line(beat, rng, LOW_DETUNE[(seed + 1) % 2])
        beat.sounds["808"].cents = LOW_DETUNE[(seed + 1) % 2]
    elif sw["bass"] in ("chords", "full"):
        bass = chord_line(beat, chords, PICKUPS if sw["bass"] == "full" else ())
    if sw["bass"] in ("glides", "full"):
        add_glides(beat, bass, rng, 1.0 if sw["bass"] == "glides" else 0.5)
    if sw["rolls"]:
        add_rolls(beat, drums, rng, sw["rolls"])
    if sw["clap_layer"] or sw["kick_layer"]:
        add_layers(beat, drums, rng, sw["clap_layer"], sw["kick_layer"])
    if sw["kick_808"]:
        kick_808(drums, bass, rng)
    if sw["rare"]:
        add_rare(beat, drums, rng)
    if sw["flams"]:
        add_flams(beat, drums, rng)
    if sw["dynamics"]:
        dynamics(beat, drums, rng)
    if sw["chokes"]:
        choke(beat, drums)
    mark(beat, drums)
    mark_808(beat, bass)
    return drums, bass, chords


def build(rows: list[dict], name: str, seed: int, out_dir: str | Path, midi_dir: str | Path | None = None,
          bars: int | None = None) -> dict:
    """Render scenario `name` for `seed` into out_dir (the files are in the
    module docstring) and return its events (kind "truth"). The same name and
    seed give the same files. bars (a multiple of 4) shortens a beat for tests."""
    sw = {**DEFAULTS, **SCENARIOS[name]}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    kit_rng, rng = random.Random(seed), random.Random(f"{name}/{seed}")
    kit = draw_kit(rows, kit_rng)
    midi = None
    if sw["pattern"] == "producer":
        if midi_dir is None:
            raise ValueError("The producer scenario needs midi_dir (a folder of producer MIDI patterns)")
        found = producer_files(midi_dir)
        files = {r: rng.choice(v) for r, v in found.items() if v}
        if "hat" not in files:
            raise ValueError(f"No single-role hat pattern under {midi_dir}")
        midi = {"files": files, "bpm": read_midi(files["hat"])["bpm"] or DEFAULTS["bpm"]}
        sw["bpm"] = midi["bpm"]
        sw["midi"] = {r: {"sha256": sha256(p), "notes": len(read_midi(p)["notes"])} for r, p in files.items()}
    bars = bars or (BARS if sw["bpm"] >= SLOW_BPM else SLOW_BARS)
    clock = Clock(sw["bpm"], sw["start_s"], sw["swing"])
    n = int((clock.time(bars * METERS[sw["meter"]][2]) + TAIL_S) * SR)
    beat = Beat(clock, n, bars, sw["meter"])
    _sounds(beat, rows, kit, sw, rng, seed, midi)
    drums, bass, chords = _parts(beat, sw, seed, kit_rng, rng, midi)
    play_drums(beat, drums)
    play_808(beat, bass)
    for layer in sw["melodic"]:
        lrng = random.Random(f"{layer}/{seed}")
        key = random.Random(f"key/{seed}").randrange(12)
        notes = (pad_notes(chords, lrng) if layer == "pad" else pluck_notes(chords, key, bars, lrng)
                 if layer == "pluck" else lead_notes(chords, key, bars, lrng))
        beat.sounds[layer] = melodic_sound(layer, out)
        play_melodic(beat, layer, notes, lrng)
        beat.sounds[layer].level = _level_re_808(beat, layer)
    beat.chords = [{"t": _t(clock.sample(c["p"])), "end": _t(clock.sample(c["end"])), "label": c["label"]}
                   for c in chords]
    _bleed(beat, kit, sw)
    return _write(beat, out, name, seed, sw)


def _level_re_808(beat: Beat, layer: str) -> float:
    """The level that puts a layer's RMS MELODIC_DB[layer] under the 808 track's."""
    rms = {k: float(np.sqrt(np.mean(beat.tracks[k] ** 2))) for k in ("808", layer)}
    return LEVELS["808"] * rms["808"] / (rms[layer] or 1.0) * 10 ** (MELODIC_DB[layer] / 20)


def _bleed(beat: Beat, kit: dict, sw: dict) -> None:
    """build_beat's bleed: the loop tiled from the start, the acapella from 5 s in (when long enough)."""
    n = beat.n
    if sw["loop"]:
        role, path, x = kit["loop"]
        beat.sounds["loop"] = Sound(role, path, x, LEVELS["melodic"])
        beat.add("loop", np.tile(x, n // len(x) + 1)[:n], 0)
    if sw["vocal"]:
        role, path, x = kit["vocal"]
        voc = x[int(5 * SR):] if len(x) > n + 5 * SR else x
        beat.sounds["vocal"] = Sound(role, path, voc, LEVELS["vocal"])
        beat.add("vocal", voc[:n], 0)


def encode_mp3(out: Path, n: int) -> dict:
    """mix.mp3 (256 kb/s, build_beat's command) and mix_mp3.wav: the mp3
    decoded, moved back by the decoder's delay (measured against mix.wav) and
    cut or padded to mix.wav's length."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is needed for the mp3 step")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(out / "mix.wav"), "-b:a", "256k", str(out / "mix.mp3")],
                   check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(out / "mix.mp3"), "-ar", str(SR),
                    str(out / "mix_mp3.wav")], check=True)
    dec, _ = sf.read(str(out / "mix_mp3.wav"), dtype="float32", always_2d=True)
    lag = mp3_lag(read(out / "mix.wav"), dec.mean(axis=1))
    decoded = len(dec)
    if lag or decoded != n:
        dec = np.stack([shift(np.pad(c, (0, max(0, n - len(c))))[: max(n, len(c))], -lag)[:n] for c in dec.T], 1)
        sf.write(str(out / "mix_mp3.wav"), dec, SR, subtype="PCM_16")
    return {"mp3_lag_samples": lag, "mp3_decoded_samples": decoded}


def mp3_lag(ref: np.ndarray, dec: np.ndarray, seconds: float = 20.0, max_lag: int = 4096) -> int:
    """Samples the decoded mp3 runs behind the mix (cross-correlation)."""
    m = min(len(ref), len(dec), int(seconds * SR))
    return ncc(dec[:m].astype(np.float64), ref[:m].astype(np.float64), max_lag)[1]


def _write(beat: Beat, out: Path, name: str, seed: int, sw: dict) -> dict:
    """The master (build_beat's: RMS -12 dBFS, +5 dB into tanh, x0.95), the
    mix files, the stems at the master's small-signal gain, events.json."""
    ids = list(beat.tracks)
    premix = sum(beat.sounds[k].level * beat.tracks[k] for k in ids)
    g = 10 ** (MASTER["rms_db"] / 20) / np.sqrt(np.mean(premix ** 2))
    drive = 10 ** (MASTER["drive_db"] / 20)
    mix = np.tanh(premix * g * drive) * MASTER["ceiling"]
    write(out / "mix.wav", np.stack([mix, mix], 1), peak=None)
    codec = encode_mp3(out, beat.n)
    stem_scale = MASTER["ceiling"] * g * drive
    stems = {k: beat.sounds[k].level * beat.tracks[k] * stem_scale for k in ids}
    for group, roles in GROUPS.items():
        stems[group] = sum((stems[k] for k in ids if beat.sounds[k].role in roles), np.zeros(beat.n))
    linear = sum(stems[k] for k in ids)
    stems_gain = write_stems(out, stems)
    events = {
        "schema": SCHEMA, "kind": "truth", "scenario": name, "seed": seed,
        "generator": {"module": "refdissect.scenarios", "switches": sw, "refdissect": __version__},
        "audio": {"sr": SR, "n_samples": beat.n, "mix": "mix_mp3.wav", "mix_wav": "mix.wav", "mix_mp3": "mix.mp3",
                  "stems": {k: f"stems_truth/{k}.flac" for k in stems},
                  "master": {**MASTER, "stem_scale": round(float(stem_scale), 9), "stems_gain": stems_gain},
                  "measured": {"mix_rms_db": round(rms_db(mix), 2), **codec,
                               "clip_residual_db": round(_db_ratio(mix - linear, linear), 2)}},
        "time": _time(beat), "sounds": {k: s.entry() for k, s in beat.sounds.items()},
        "drums": sorted(beat.drums, key=lambda e: (e["t"], e["sound"])),
        "bass808": sorted(beat.bass808, key=lambda e: e["t"]),
        "notes": sorted(beat.notes, key=lambda e: (e["t"], e["layer"], e["midi"])), "chords": beat.chords}
    (out / "events.json").write_text(json.dumps(events, indent=1), encoding="utf-8")
    return events


def write_stems(out: Path, stems: dict[str, np.ndarray]) -> dict[str, float]:
    """Stems as 24-bit FLAC, which holds -1..1. A stem that would peak there
    (only the Demucs groups ever did: up to 1.2 in the real-data run) is
    written halved as often as needed; the returned gains name those stems
    (normally none)."""
    (out / "stems_truth").mkdir(parents=True, exist_ok=True)
    gains = {}
    for k, x in stems.items():
        peak = float(np.max(np.abs(x)))
        gain = 0.5 ** math.ceil(math.log2(peak / 0.999)) if peak > 0.999 else 1.0
        if gain != 1.0:
            gains[k] = gain
        sf.write(str(out / "stems_truth" / f"{k}.flac"), x * gain, SR, subtype="PCM_24")
    return gains


def _db_ratio(err: np.ndarray, ref: np.ndarray) -> float:
    return float(10 * np.log10(np.sum(np.square(err)) / (np.sum(np.square(ref)) or 1e-20) + 1e-30))


def _time(beat: Beat) -> dict:
    c = beat.clock
    beats_per_bar, unit, spb = METERS[beat.meter]
    pulse = 16 // unit  # sixteenths per beat
    beats = [round(c.time(p), 6) for p in range(0, beat.bars * spb, pulse)]
    return {"bpm": c.bpm, "beats_per_bar": beats_per_bar, "beat_unit": unit, "downbeat_s": round(c.start_s, 6),
            "beats_s": beats, "downbeats_s": [round(c.time(b * spb), 6) for b in range(beat.bars)],
            "swing": {"ratio": c.swing, "level": 16} if c.swing else None, "tempo_map": None}


# ── checks ───────────────────────────────────────────────────────────────────

def _band(x: np.ndarray, role: str) -> np.ndarray:
    """band() in the role's attack band; a stretch too short to filter is kept as it is."""
    return band(x, ATTACK_BAND.get(role, role if role in BANDS else "full")) if len(x) > 64 else x


def find_attack(x: np.ndarray, lo: float, t: float, hi: float, role: str, since: float | None = None,
                rise: float = 0.3) -> float:
    """Where a sound whose file starts at t attacks in x, searched between lo
    and hi seconds: attack_time()'s rule (the 1 ms envelope in the role's band
    reaching 30 % of its peak) counted from what still rings from earlier
    hits: sqrt(2) times the band's RMS over the 30 ms before t (from `since`
    on: past the earlier hit's attack, or its end when it was cut), the upper
    edge of a tail's 1 ms envelope, a low tone's included. Alone after silence, that is
    attack_time() itself. The band is filtered separately before and after
    t: from t on it is what attack_time() sees (the file from its first
    sample), before t whatever is there (a file placed early shows up)."""
    a, b = max(0, int(round(lo * SR))), min(len(x), int(round(hi * SR)))
    s = min(max(int(round(t * SR)), a), b)
    pad = samples(20)
    before = _band(x[max(0, a - pad):s], role)[a - max(0, a - pad):] if s > a else np.zeros(0)
    seg = np.r_[before, _band(x[s:b], role)]
    ring = int(round(t * SR)) - max(int(round(t * SR)) - samples(30), int(round((since or 0.0) * SR)), 0)
    floor = 0.0
    if ring >= samples(2):
        tail = _band(x[max(0, int(round(t * SR)) - ring - pad):int(round(t * SR))], role)[-ring:]
        floor = float(np.sqrt(2 * np.mean(np.square(tail))))
    w = samples(1)
    risen = np.convolve(np.abs(seg), np.ones(w) / w, mode="same") - floor
    return (a + int(np.argmax(risen >= rise * max(float(risen.max()), 1e-12)))) / SR


def attack_errors(events: dict, stems: dict[str, np.ndarray]) -> list[dict]:
    """For every drum hit and attacked 808 note: the attack found in its own
    truth stem (find_attack from 15 ms before t_attack, or 70 % of the way
    from the hit before, to 250 ms after the file starts or the next hit of
    that sound) minus t_attack."""
    by_sound: dict[str, list[dict]] = {}
    for e in events["drums"]:
        by_sound.setdefault(e["sound"], []).append(dict(e, role=e["role"]))
    for e in events["bass808"]:
        by_sound.setdefault("808", []).append(dict(e, role="808"))
    out = []
    for sid, hits in by_sound.items():
        hits.sort(key=lambda e: e["t_attack"])
        for i, e in enumerate(hits):
            if "legato" in e["flags"]:
                continue
            prev, nxt = (hits[i - 1] if i else None), (hits[i + 1] if i + 1 < len(hits) else None)
            since = None
            if prev:  # past the earlier hit's own attack, and past its end when it was cut
                since = max(prev["t_attack"] + 0.7 * (e["t_attack"] - prev["t_attack"]), prev.get("end", 0.0))
            lo = max(e["t_attack"] - 0.015, since or 0.0)
            hi = min(e["t"] + 0.25, nxt["t"] if nxt else e["t"] + 0.25)
            found = find_attack(stems[sid], lo, e["t"], max(hi, e["t_attack"] + 0.002), e["role"], since)
            out.append({"sound": sid, "t_attack": e["t_attack"], "error_ms": round((found - e["t_attack"]) * 1000, 3)})
    return out


def pitch_errors(events: dict, stem: np.ndarray, settle_s: float = 0.3, min_s: float = 0.15) -> list[float]:
    """Cents between each 808 note's settled truth pitch (midi_settled: the
    sample's settled pitch, transposed) and its pitch in the truth stem
    (zero_cross_hz from settle_s after it starts to 20 ms before it ends:
    library 808s drop up to ~150 cents over their first 0.2-0.7 s), for
    notes without a glide that hold at least min_s after settling."""
    out = []
    for e in events["bass808"]:
        a, b = int(round(e["t"] * SR)), int(round(e["end"] * SR))
        if e["glide"] or (b - a) / SR < settle_s + min_s + 0.02:
            continue
        hz = 440 * 2 ** ((e.get("midi_settled", e["midi"]) - 69) / 12)
        f = zero_cross_hz(band(stem[a:b], "808"), hz, samples(settle_s * 1000), b - a - samples(20))
        if f:
            out.append(round(1200 * math.log2(f / hz), 3))
    return out


def sounding_errors(events: dict, stem: np.ndarray) -> list[float]:
    """Cents between each 808 note's truth pitch (midi, from the sample's
    pitch curve) and the same reading of the truth stem: pitch_track over
    the note's own stretch, near its settled pitch, through sounding(); for
    the notes whose read_window lies within the crossings the stretch has
    (the shortest, lowest notes have two or three)."""
    out = []
    for e in events["bass808"]:
        a, b = int(round(e["t"] * SR)), int(round(e["end"] * SR))
        track = pitch_track(stem[a:b], 440 * 2 ** ((e.get("midi_settled", e["midi"]) - 69) / 12))
        lo, hi = read_window((b - a) / SR)
        if track is not None and track[1][0] <= lo and hi <= track[1][1]:
            heard = sounding(lambda t, f=track[0]: 69 + 12 * np.log2(f(t) / 440), (b - a) / SR)
            out.append(round(100 * (heard - e["midi"]), 2))
    return out


DRUM_ROLES = GROUPS["drums"]


def played(sound: dict) -> np.ndarray:
    """A sound entry's file as it was played: loaded as calibrate._pick loads
    it, transposed, behind its lead-in (as silence: the lead-in's -60 dB noise
    is not kept)."""
    x = read(sound["path"])
    x = trim_onset(x) / np.max(np.abs(x))
    st = sound["semitones"] + sound["cents"] / 100
    y = transpose(x, st) if st else x
    return np.r_[np.zeros(round(sound.get("lead_in_s", 0.0) * SR), np.float32), y]


def drum_residuals(events: dict, stems: dict[str, np.ndarray]) -> dict[str, float]:
    """Each drum stem against its truth played again (every hit's file as
    played, at t, with its gain, at the master's small-signal gain; a choked
    hit cut where the next closed hat starts): what is left, in dB re the
    stem. Unlike the attack search, this sees hits under a louder tail."""
    n, scale = events["audio"]["n_samples"], events["audio"]["master"]["stem_scale"]
    hats = sorted(e["t"] for e in events["drums"] if e["role"] == "hat")
    out = {}
    for sid, snd in events["sounds"].items():
        if snd["role"] not in DRUM_ROLES or sid not in stems:
            continue
        y, rebuilt = played(snd), np.zeros(n)
        for e in (e for e in events["drums"] if e["sound"] == sid):
            a = int(round(e["t"] * SR))
            x = y
            if "choked" in e["flags"]:
                x = _cut(y, int(round(next(t for t in hats if t > e["t"]) * SR)) - a)
            b = min(n, a + len(x))
            rebuilt[a:b] += e["gain"] * scale * x[: b - a]
        out[sid] = round(_db_ratio(stems[sid] - rebuilt, stems[sid]), 1)
    return out


def verify(beat_dir: str | Path) -> dict:
    """A built folder against its own truth: every drum hit and attacked 808
    note found in its truth stem (attack_errors), every drum stem played again
    from the truth (drum_residuals, the worst in dB), the 808 notes' settled
    and sounding pitch in their stem (pitch_errors, sounding_errors: p90 and
    worst |cents|), the stems' sum against the mix without the
    clipper (dB, the clipper undone with arctanh), the groups against their
    members, the mix's RMS and the mp3's alignment and length."""
    d = Path(beat_dir)
    ev = json.loads((d / "events.json").read_text(encoding="utf-8"))
    audio = ev["audio"]
    gains = audio["master"].get("stems_gain") or {}
    stems = {k: read(d / p).astype(np.float64) / gains.get(k, 1.0) for k, p in audio["stems"].items()}
    errors = attack_errors(ev, stems)
    nulls = drum_residuals(ev, stems)
    err = np.abs([e["error_ms"] for e in errors]) if errors else np.zeros(1)
    ids = [k for k in stems if k not in GROUPS]
    linear = sum(stems[k] for k in ids)
    mix = read(d / audio["mix_wav"]).astype(np.float64)
    c = audio["master"]["ceiling"]
    unclipped = c * np.arctanh(np.clip(mix / c, -1 + 1e-9, 1 - 1e-9))
    roles = {k: ev["sounds"][k]["role"] for k in ids}
    groups = max(float(np.max(np.abs(stems[g] - sum((stems[k] for k in ids if roles[k] in r), np.zeros(len(mix))))))
                 for g, r in GROUPS.items())
    dec = read(d / audio["mix"])
    cents = pitch_errors(ev, stems["808"]) if "808" in stems else []
    heard = np.abs(sounding_errors(ev, stems["808"])) if "808" in stems else np.zeros(0)
    return {"hits": len(errors), "attack_err_max_ms": round(float(err.max()), 3),
            "attack_err_median_ms": round(float(np.median(err)), 3),
            "within_2ms": round(float(np.mean(err <= 2.0)), 4),
            "worst": sorted(errors, key=lambda e: -abs(e["error_ms"]))[:3],
            "drum_null_db": max(nulls.values()) if nulls else None,
            "stems_vs_mix_db": round(_db_ratio(linear - unclipped, unclipped), 1),
            "groups_max_abs": float(groups), "mix_rms_db": round(rms_db(mix), 2),
            "pitch_notes": len(cents), "pitch_err_max_cents": round(float(np.max(np.abs(cents))), 2) if cents else None,
            "pitch_err_median_cents": round(float(np.median(cents)), 2) if cents else None,
            "sounding_notes": len(heard),
            "sounding_err_p90_cents": round(float(np.percentile(heard, 90)), 2) if len(heard) else None,
            "sounding_err_max_cents": round(float(heard.max()), 2) if len(heard) else None,
            "mp3_lag_samples": mp3_lag(mix, dec), "mp3_samples": len(dec), "mix_samples": len(mix)}


def check_schema(ev: dict) -> list[str]:
    """What an events file lacks or gets wrong against refdissect.events/1
    (empty when it is valid)."""
    problems = []

    def need(obj, keys, where):
        for k in keys:
            if k not in obj:
                problems.append(f"{where}: no {k}")

    need(ev, ("schema", "kind", "scenario", "seed", "generator", "audio", "time", "sounds", "drums", "bass808",
              "notes", "chords"), "top")
    if ev.get("schema") != SCHEMA or ev.get("kind") not in ("truth", "estimate"):
        problems.append("top: schema or kind")
    need(ev.get("audio", {}), ("sr", "n_samples", "mix", "mix_wav", "mix_mp3", "stems", "master"), "audio")
    need(ev.get("audio", {}).get("master", {}), ("rms_db", "drive_db", "ceiling", "stem_scale"), "master")
    t = ev.get("time", {})
    need(t, ("bpm", "beats_per_bar", "beat_unit", "downbeat_s", "beats_s", "downbeats_s", "swing", "tempo_map"), "time")
    if t.get("beat_unit") not in (4, 8):
        problems.append("time: beat_unit")
    roles = {"kick", "snare", "clap", "rim", "hat", "open_hat", "perc", "tom", "crash", "ride", "808", "melodic",
             "vocal"}
    for k, s in ev.get("sounds", {}).items():
        need(s, ("role", "path", "sha256", "semitones", "cents", "attack_s"), f"sounds.{k}")
        if s.get("role") not in roles:
            problems.append(f"sounds.{k}: role {s.get('role')}")
    for i, e in enumerate(ev.get("drums", [])):
        need(e, ("sound", "role", "t", "t_attack", "gain", "velocity", "layer_of", "flags"), f"drums[{i}]")
        if e.get("sound") not in ev.get("sounds", {}) or not e.get("t_attack", 0) >= e.get("t", 0):
            problems.append(f"drums[{i}]: sound or times")
        if e.get("velocity") is not None and not 1 <= e["velocity"] <= 127:
            problems.append(f"drums[{i}]: velocity")
    for i, e in enumerate(ev.get("bass808", [])):
        need(e, ("sound", "t", "t_attack", "end", "midi", "glide", "with_kick", "gain", "flags"), f"bass808[{i}]")
        if e.get("glide"):
            need(e["glide"], ("to_midi", "t", "ms", "shape"), f"bass808[{i}].glide")
        if not e.get("end", 0) > e.get("t", 0):
            problems.append(f"bass808[{i}]: end")
    for i, e in enumerate(ev.get("notes", [])):
        need(e, ("layer", "t", "end", "midi", "cents", "velocity", "flags"), f"notes[{i}]")
    for i, e in enumerate(ev.get("chords", [])):
        need(e, ("t", "end", "label"), f"chords[{i}]")
        if ":" not in e.get("label", ""):
            problems.append(f"chords[{i}]: label")
    return problems


# ── calibration v1 ───────────────────────────────────────────────────────────

V1_STEMS = {"melodic": "loop"}  # the rebuild's stem names that differ from our sound ids


def convert_v1(src_dir: str | Path, out_dir: str | Path, rows: list[dict] | None = None) -> list[dict]:
    """The six v1 calibration beats rebuilt with truth stems (src_dir/beatN with
    truth.json, stems_truth/*.wav and the mix files) as out_dir/v1orig-N in
    this layout: events.json (scenario "v1orig", seed N), the mix files
    hard-linked (copied across devices), the stems as FLAC. rows (the
    inventory) tells a snare from a clap; without it, the file's path does."""
    out = []
    for beat in sorted(Path(src_dir).glob("beat*"), key=lambda p: int(re.sub(r"\D", "", p.name) or 0)):
        if (beat / "truth.json").is_file():
            seed = int(re.sub(r"\D", "", beat.name))
            out.append(convert_beat(beat, Path(out_dir) / f"v1orig-{seed}", seed, rows))
    return out


def convert_beat(src: Path, out: Path, seed: int, rows: list[dict] | None = None) -> dict:
    t = json.loads((src / "truth.json").read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    role_by_path = {r["path"]: r["role"] for r in rows or []}
    snare_path = t["sounds"]["snare"]
    backbeat = role_by_path.get(snare_path) or ("clap" if role_of(snare_path) == "clap" else "snare")
    clock = Clock(t["bpm"])
    n = t["n_samples"]
    total = round((n / SR - TAIL_S) / clock.step_s)  # sixteenths in the beat
    beat = Beat(clock, n, total // 16, "4/4")
    for key, path in t["sounds"].items():
        sid = backbeat if key == "snare" else key
        x = read(path)
        beat.sounds[sid] = make_sound(sid, path, trim_onset(x) / np.max(np.abs(x)), LEVELS[sid])
    _tune(beat)
    s808 = beat.sounds["808"]
    gains = {"hat": t["hat_gains"], "open_hat": [t.get("open_hat_gain", 0.9)] * len(t["positions"]["open_hat"])}
    for key in ("kick", "snare", "hat", "open_hat"):
        sid = backbeat if key == "snare" else key
        s = beat.sounds[sid]
        for i, pos in enumerate(t["positions"][key]):
            a = int(round(pos * SR))
            g = gains[key][i] if key in gains else 1.0
            beat.drums.append({"sound": sid, "role": s.role, "t": _t(a), "t_attack": _t(a + s.attack),
                               "gain": round(s.level * g, 6), "velocity": velocity(g), "layer_of": None, "flags": []})
    # build_beat cut each 808 note at the next kick step: int(sixteenths * step * SR) samples long
    steps = [round(p / clock.step_s) for p, _ in t["positions"]["808"]] + [total]
    specs = [_note(steps[i], steps[i + 1], semis) for i, (_, semis) in enumerate(t["positions"]["808"])]
    mark_808(beat, specs)
    for i, ((pos, semis), b) in enumerate(zip(t["positions"]["808"], specs)):
        a, length = int(round(pos * SR)), int((steps[i + 1] - steps[i]) * clock.step_s * SR)
        full = transpose(s808.audio, semis)
        y = _cut(full, length)
        stop = a + _end(full, length)
        pitch = beat.native_midi + semis + drop(s808.f0_curve, np.arange(length) * 2 ** (semis / 12) / SR) / 100
        beat.bass808.append(_bass_event(beat, b, a, a + round(attack_time(y, "808") * SR), stop, s808.level,
                                        sounding(functools.partial(_at, pitch, 0), (stop - a) / SR)))
    for key, sid in (("melodic", "loop"), ("vocal", "vocal")):
        beat.sounds[sid] = Sound("melodic" if sid == "loop" else "vocal", t[key], np.zeros(0, np.float32),
                                 LEVELS[key])
    for name in ("mix.wav", "mix.mp3", "mix_mp3.wav"):
        _link(src / name, out / name)
    stems = {}
    for name in t["stems_truth"]:
        stems[V1_STEMS.get(name, backbeat if name == "snare" else name)] = read(src / "stems_truth" / f"{name}.wav")
    stems_gain = write_stems(out, stems)
    mix = read(out / "mix.wav")
    sw = {**DEFAULTS, "converted_from": "v1 calibration rebuilt with truth stems"}
    m = t["master"]
    events = {
        "schema": SCHEMA, "kind": "truth", "scenario": "v1orig", "seed": seed,
        "generator": {"module": "refdissect.scenarios", "switches": sw, "refdissect": __version__},
        "audio": {"sr": SR, "n_samples": n, "mix": "mix_mp3.wav", "mix_wav": "mix.wav", "mix_mp3": "mix.mp3",
                  "stems": {k: f"stems_truth/{k}.flac" for k in stems},
                  "master": {"rms_db": MASTER["rms_db"], "drive_db": m["drive_db"], "ceiling": m["out_gain"],
                             "stem_scale": m["stem_scale"], "stems_gain": stems_gain},
                  "measured": {"mix_rms_db": round(rms_db(mix), 2), "mp3_lag_samples": mp3_lag(mix, read(
                      out / "mix_mp3.wav")), "mp3_decoded_samples": t.get("mp3_len"),
                      "clip_residual_db": t.get("clip_residual_db")}},
        "time": _time(beat), "sounds": {k: s.entry() for k, s in beat.sounds.items()},
        "drums": sorted(beat.drums, key=lambda e: (e["t"], e["sound"])), "bass808": beat.bass808,
        "notes": [], "chords": []}
    (out / "events.json").write_text(json.dumps(events, indent=1), encoding="utf-8")
    return events


def _link(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
