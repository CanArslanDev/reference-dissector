"""Melodic parts for the calibration scenarios, every note known.

Chords in Harte syntax (maj, min, 7, maj7, min7, sus2, sus4, dim, maj(9) for
add9, slash inversions) that change inside the bar, voiced in close position
over a bass note; a 16th-note pluck arpeggio with 32nd-note runs and repeats;
a mono lead whose notes are sometimes reached by a glide. The voices are
band-limited saw and square waves under ADSR envelopes.
"""

from __future__ import annotations

import functools
import random

import numpy as np

from .audio import SR

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
CHORDS = {"maj": (0, 4, 7), "min": (0, 3, 7), "7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11), "min7": (0, 3, 7, 10),
          "sus2": (0, 2, 7), "sus4": (0, 5, 7), "dim": (0, 3, 6), "maj(9)": (0, 4, 7, 14)}  # Harte; maj(9) = add9
SLASH = {"3": 4, "b3": 3, "5": 7, "b7": 10}  # Harte bass degrees: semitones over the root
MINOR = (0, 2, 3, 5, 7, 8, 10)
# four bars (64 sixteenths) in a minor key: (degree, chord, sixteenths, slash bass); changes inside bars
PROGRESSIONS = (
    ((0, "min7", 16, None), (8, "maj7", 8, None), (10, "sus4", 4, None), (10, "sus2", 4, None),
     (3, "maj(9)", 8, None), (8, "maj", 8, "3"), (5, "min", 6, None), (7, "7", 6, None), (2, "dim", 4, None)),
    ((0, "min", 12, None), (0, "min7", 4, "b7"), (5, "min7", 8, None), (7, "sus4", 4, None), (7, "7", 4, None),
     (8, "maj7", 6, None), (3, "maj", 10, "5"), (10, "maj(9)", 8, None), (2, "dim", 4, "b3"),
     (7, "sus2", 4, None)))
PLUCK_MASKS = ((1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1), (1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0),
               (1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0), (1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1))
LEAD_RHYTHMS = ((4, 2, 2, 3, 3, 2), (2, 2, 2, 2, 4, -4), (3, 3, 2, 2, 2, -2, 2), (6, 2, -2, 2, 4),
                (1, 1, 2, 4, -4, 4), (2, 1, 1, 2, 2, 4, -4), (3, 1, 2, 2, 4, 2, 2), (8, -2, 2, 2, 2))
LEAD_GLIDE_MS = (50.0, 140.0)


def chord_plan(bars: int, key: int, prog: tuple) -> list[dict]:
    """The progression over the bars, voiced: bass (root or slash bass) in MIDI
    40-51, the chord tones above G3 in close position, each the rotation nearest
    the chord before (smooth voice leading)."""
    chords, prev, p = [], None, 0
    for _ in range(bars // 4):
        for degree, quality, length, slash in prog:
            root = (key + degree) % 12
            pcs = [(root + i) % 12 for i in CHORDS[quality]]
            low = 40 + ((root + SLASH.get(slash, 0)) % 12 - 40) % 12
            rotations = []
            for r in range(len(pcs)):
                cur, tones = 54, []
                for pc in pcs[r:] + pcs[:r]:
                    cur = cur + 1 + (pc - (cur + 1)) % 12
                    tones.append(cur)
                rotations.append(tones)
            target = float(np.mean(prev)) if prev else 64.0
            prev = min(rotations, key=lambda tones: abs(np.mean(tones) - target))
            chords.append({"p": p, "end": p + length, "root": root, "tones": [low, *prev],
                           "label": f"{NOTE_NAMES[root]}:{quality}" + (f"/{slash}" if slash else "")})
            p += length
    return chords


def chord_at(chords: list[dict], p: float) -> dict:
    return next(c for c in chords if c["p"] <= p < c["end"])


def pad_notes(chords: list[dict], rng: random.Random) -> list[dict]:
    notes = []
    for c in chords:
        v = rng.randint(84, 104)
        notes += [{"layer": "pad", "p": c["p"], "end": c["end"], "midi": m, "velocity": v,
                   "flags": {"chord", "bass"} if i == 0 else {"chord"}} for i, m in enumerate(c["tones"])]
    return notes


def pluck_notes(chords: list[dict], key: int, bars: int, rng: random.Random) -> list[dict]:
    """A 16th-note arpeggio over each chord's upper tones (up and down, MIDI
    60-88) with two 32nd-note runs per bar: four 32nds walking the minor scale,
    or one pitch four times."""
    scale = {(key + i) % 12 for i in MINOR}
    notes, idx, step = [], 0, 1
    for bar in range(bars):
        mask, runs = rng.choice(PLUCK_MASKS), set(rng.sample(range(8), 2))
        for s in range(16):
            p = bar * 16 + s
            pool = sorted({u + 12 * o for u in chord_at(chords, p)["tones"][1:] for o in (0, 1, 2)
                           if 60 <= u + 12 * o <= 88})
            if s // 2 in runs:
                if s % 2:
                    continue
                m, seq, d = pool[idx % len(pool)], [], rng.choice([1, -1])
                repeat = rng.random() < 0.5
                for _ in range(4):
                    seq.append(m)
                    m = m if repeat else m + d
                    while not repeat and m % 12 not in scale:
                        m += d
                for j, q in enumerate(seq):
                    flags = {"run32", "repeat"} if repeat and j else {"run32"}
                    notes.append({"layer": "pluck", "p": p + j / 2, "end": p + j / 2 + 0.42, "midi": q,
                                  "velocity": rng.randint(70, 100), "flags": flags})
                continue
            if not mask[s]:
                continue
            idx += step
            if idx >= len(pool) - 1 or idx <= 0:
                step = -step
            idx = max(0, min(idx, len(pool) - 1))
            v = rng.randint(104, 122) if s % 4 == 0 else rng.randint(84, 100) if s % 2 == 0 else rng.randint(60, 84)
            notes.append({"layer": "pluck", "p": p, "end": p + 0.85, "midi": pool[idx], "velocity": v, "flags": set()})
    return notes


def lead_notes(chords: list[dict], key: int, bars: int, rng: random.Random) -> list[dict]:
    """A mono line in the minor scale (MIDI 64-86), chord tones on the beats;
    about 30 % of the notes that follow another directly, 1-7 semitones away,
    are reached by a glide of 50-140 ms (the line's legato groups share one
    velocity)."""
    scale = [m for m in range(64, 87) if (m - key) % 12 in MINOR]
    notes, cur = [], 72 + (key - 72) % 12
    for bar in range(bars):
        pos = 0
        for d in rng.choice(LEAD_RHYTHMS):
            if d < 0:
                pos -= d
                continue
            p = bar * 16 + pos
            tones = {t % 12 for t in chord_at(chords, p)["tones"]}
            cands = [m for m in range(64, 87) if m % 12 in tones] if pos % 4 == 0 else list(scale)
            cands.sort(key=lambda m: abs(m - cur) + rng.random() * 3)
            cur = cands[0] if rng.random() < 0.6 else rng.choice(cands[:4])
            notes.append({"layer": "lead", "p": p, "end": p + d, "midi": cur, "velocity": 0, "flags": set(),
                          "glide_ms": None})
            pos += d
    for prev, nt in zip(notes, notes[1:]):
        if prev["end"] == nt["p"] and 1 <= abs(nt["midi"] - prev["midi"]) <= 7 and rng.random() < 0.3:
            nt["glide_ms"] = round(rng.uniform(*LEAD_GLIDE_MS), 1)
            nt["flags"].add("glide")
    v = 100
    for nt in notes:
        v = v if nt["glide_ms"] else rng.randint(86, 120)
        nt["velocity"] = v
    return notes


def adsr(n_on: int, a: float, d: float, s: float, r: float) -> np.ndarray:
    """Linear attack over a, exponential decay to s, an exponential release
    (-40 dB at r) from the note-off; n_on + 1.5 r long."""
    t = np.arange(n_on) / SR
    a = max(a, 1 / SR)
    on = np.where(t < a, t / a, s + (1 - s) * np.exp(-(t - a) / max(d / 3, 1e-4)))
    level = on[-1] if n_on else 0.0
    return np.r_[on, level * np.exp(-np.arange(int(r * 1.5 * SR)) / SR / max(r / 4.6, 1e-4))]


@functools.lru_cache(maxsize=512)
def _table(wave: str, midi: float, cutoff: float, size: int = 2048) -> np.ndarray:
    """One period of a band-limited saw or square: its harmonics under 4x the
    cutoff (and Nyquist at `midi`), each through a 2nd-order low-pass."""
    f0 = 440 * 2 ** ((midi - 69) / 12)
    k = np.arange(1, max(1, int(min(4 * cutoff, 0.45 * SR) // f0)) + 1)
    amp = 1.0 / k * ((k % 2 == 1) if wave == "square" else 1.0) / np.sqrt(1 + (k * f0 / cutoff) ** 4)
    y = np.sin(np.outer(2 * np.pi * np.arange(size) / size, k)) @ amp
    return y / (np.max(np.abs(y)) or 1.0)


def osc(midi: np.ndarray, wave: str, cutoff: float, phase: float = 0.0) -> np.ndarray:
    """The wave following a pitch curve (MIDI per sample)."""
    tab = _table(wave, round(float(np.max(midi)), 1), cutoff)
    pos = (phase + np.cumsum(440 * 2 ** ((midi - 69) / 12)) / SR) % 1.0 * len(tab)
    return np.interp(pos, np.arange(len(tab) + 1), np.r_[tab, tab[0]])


def voice(layer: str, midi: np.ndarray, n_on: int, phase: float) -> np.ndarray:
    """A note of the layer's voice: pad (two saws 14 cents apart, slow attack),
    pluck (a square whose brightness falls in 70 ms), lead (a saw)."""
    env = adsr(n_on, *{"pad": (0.06, 0.45, 0.75, 0.35), "pluck": (0.002, 0.25, 0.0, 0.08)}.get(
        layer, (0.01, 0.3, 0.8, 0.12)))
    m = np.r_[midi, np.full(max(0, len(env) - len(midi)), midi[-1])][: len(env)]  # the release keeps the last pitch
    if layer == "pad":
        return (osc(m + 0.07, "saw", 1800.0, phase) + osc(m - 0.07, "saw", 1800.0, phase + 0.37)) / 2 * env
    if layer == "pluck":
        w = np.exp(-np.arange(len(env)) / SR / 0.07)
        return (osc(m, "square", 5000.0, phase) * w + osc(m, "square", 700.0, phase) * (1 - w)) * env
    return osc(m, "saw", 3500.0, phase) * env
