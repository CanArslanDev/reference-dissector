"""Where each drum sound plays, and the 808's notes.

Drums, in two passes:

1. discover(): in each band (kick, snare, hat), the strong onsets that sit on
   the sixteenth grid are grouped per grid slot by waveform; groups of at
   least `min_hits` become templates (the averaged hits). Groups that are the
   same sound (in any slot or band) are merged.
2. detect(): each template slides over the whole stem (matched filter); its
   peaks above a threshold are candidate hits, and every moment is fit with
   all the templates that play near it (layers.explain), so a moment can hold
   several sounds (a hat under a snare) and one sound twice (a flam).

The templates come from the song itself, so the second pass finds a sound
even where the first could not see it (under other hits, off the grid).
On PIRAMIT the snare templates separate cleanly: 28 hits at NCC >= 0.8 and
nothing between 0.3 and 0.8.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import librosa
import numpy as np
from scipy.signal import find_peaks

from . import layers
from .audio import PRE, SR, band, samples
from .grid import Grid
from .hits import Group, group, merge, onset_template, onsets, strength_peaks, template
from .xcorr import ncc, sliding_ncc

CLUSTER_MS = {"kick": 80, "snare": 60, "hat": 40, "low": 80}
# How much of a template is compared. Patterns repeat, so whatever always
# follows a hit (the kick two sixteenths after a snare) is in its average
# too; long windows let that later, louder sound decide the match. Detection
# looks at the attack only, library matching at a little more.
DETECT_MS = {"kick": 80, "snare": 60, "hat": 40, "low": 80}
LENGTH_MS = {"kick": 200, "snare": 150, "hat": 80, "low": 200}
MAX_MS = 350  # templates are kept this long and cut where used
THRESHOLD = {"kick": 0.6, "snare": 0.6, "hat": 0.65, "low": 0.6}
# A low tone correlates with itself one period later; under 200 Hz one hit
# would otherwise be found every 35 ms while it rings
MIN_GAP_MS = {"kick": 35, "snare": 35, "hat": 35, "low": 100}


@dataclass
class Template:
    """A sound that repeats: name (sound_1 has the most hits), role (a guess
    from the energy its hits add; matching against a library decides), band
    (where it was found, and where it is detected), the averaged hits
    (MAX_MS long) and the grid slots of those hits."""
    name: str
    role: str
    band: str
    audio: np.ndarray
    hits: int
    slots: dict[int, int] = field(default_factory=dict)
    group: Group | None = None


def new_energy_role(x: np.ndarray, g: Group, gap_ms: float = 10, span_ms: float = 60) -> tuple[str, dict]:
    """A first guess at kick / snare / hat from the energy the hits add: the
    spectrum of the span after each onset minus the span before it (an 808
    ringing under a snare is there before the snare too, so it cancels out).
    Only a guess: bright snares and dull hats overlap; the library decides."""
    span, gap = samples(span_ms), samples(gap_ms)
    n = 1 << (span - 1).bit_length()
    f = np.fft.rfftfreq(n, 1 / SR)
    win = np.hanning(span)
    added = np.zeros(len(f))
    for s, lag, _sign, gain, _ in g.members:
        a = s - lag
        if a - gap - span < 0 or a + span > len(x):
            continue
        after = np.abs(np.fft.rfft(x[a:a + span] * win, n)) ** 2
        before = np.abs(np.fft.rfft(x[a - gap - span:a - gap] * win, n)) ** 2
        added += np.maximum(after - before, 0) * gain ** 2
    total = added.sum() or 1.0
    shares = {"low": float(added[f < 150].sum() / total), "high": float(added[f > 5000].sum() / total)}
    if shares["low"] > 0.35:
        role = "kick"
    elif shares["high"] > 0.35:
        role = "hat"
    else:
        role = "snare"
    return role, {k: round(v, 3) for k, v in shares.items()}


def _times(x: np.ndarray, audio: np.ndarray, band_name: str, threshold: float) -> np.ndarray:
    score, _ = sliding_ncc(x, audio, band_name)
    peaks, _ = find_peaks(score, height=threshold, distance=samples(MIN_GAP_MS[band_name]))
    return (peaks + PRE) / SR


def discover(drums: np.ndarray, grid: Grid, bands=("snare", "hat", "low"), min_hits: int = 8,
             tolerance_ms: float = 12, same_sound: float = 0.6) -> list[Template]:
    """Templates of the sounds that repeat on the grid.

    In each band, strong onsets on the sixteenth grid are grouped per slot by
    waveform; groups of min_hits or more are merged when their averages
    match. A sound usually shows up in more than one band (a snare's top end
    in the hat band): candidates whose hits mostly coincide with an earlier,
    larger candidate's are the same sound and are dropped. Each survivor's
    role comes from the energy its hits add (new_energy_role)."""
    candidates: list[tuple[str, Group]] = []
    tol = tolerance_ms / 1000
    for band_name in bands:
        xb = band(drums, band_name)
        t, _strength = strength_peaks(drums, band_name)
        rel = (t - grid.phase) / grid.step
        on_grid = np.abs(rel - np.round(rel)) * grid.step < tol
        slots = np.round(rel).astype(int) % (4 * grid.beats_per_bar)
        groups: list[Group] = []
        for slot in sorted(set(slots[on_grid].tolist())):
            starts = (t[on_grid & (slots == slot)] * SR).astype(int)
            if len(starts) < min_hits:
                continue
            groups += [g for g in group(xb, starts=starts, threshold=0.6, cluster_ms=CLUSTER_MS[band_name])
                       if g.count >= min_hits]
        candidates += [(band_name, g) for g in merge(xb, groups, threshold=0.8, length_ms=100,
                                                     min_count=min_hits)]
    candidates.sort(key=lambda bg: -bg[1].count)
    kept: list[tuple[str, Group, np.ndarray, np.ndarray]] = []
    for band_name, g in candidates:
        audio = onset_template(drums, g, DETECT_MS[band_name])[0]
        times = _times(drums, audio, band_name, THRESHOLD[band_name])
        if not len(times):
            continue
        short = template(drums, g, 100)[0]
        duplicate = False
        for other_band, _g, other_times, other_short in kept:
            # the same moments AND the same waveform: two sounds that always
            # play together (a hat on every kick) share moments, not waveforms
            shared = np.sum(np.min(np.abs(times[:, None] - other_times[None, :]), axis=1) < 0.015)
            # alike in both bands: a snare that always lands on a hat looks
            # like the hat in the hat band, but not in its own
            alike = min(abs(ncc(band(short, b), band(other_short, b), samples(4))[0])
                        for b in {band_name, other_band})
            if shared >= same_sound * min(len(times), len(other_times)) and alike >= 0.6:
                duplicate = True
                break
        if not duplicate:
            kept.append((band_name, g, times, short))
    out: list[Template] = []
    for k, (band_name, g, _times_found, _short) in enumerate(sorted(kept, key=lambda c: -len(c[2]))):
        role, _shares = new_energy_role(drums, g)
        audio, used = onset_template(drums, g, MAX_MS)
        slot_counts = Counter(grid.step_in_bar(np.array(g.starts()) / SR).tolist())
        out.append(Template(f"sound_{k + 1}", role, band_name, audio, used, dict(sorted(slot_counts.items())), g))
    return out


def explain(drums: np.ndarray, templates: list[Template], thresholds: dict | None = None,
            min_gain: float = layers.MATCHED_GAIN) -> layers.Explained:
    """layers.explain with this module's attack lengths and thresholds: the
    hits and what decided them (duplicates, held sounds, nulls)."""
    return layers.explain(drums, templates, DETECT_MS, {**THRESHOLD, **(thresholds or {})}, min_gain=min_gain)


def detect(drums: np.ndarray, templates: list[Template], thresholds: dict | None = None,
           min_gain: float = layers.MATCHED_GAIN) -> dict[str, list[dict]]:
    """Every hit of every template: {name: [{t, ncc, gain, source, ...}]} (t
    in seconds, PRE after the template's window starts; gain its level, 1 =
    as loud as the template). Each moment is fit with every template that
    plays near it (layers.explain): a hat under a snare, a clap on a snare
    or a flam's grace note are hits of their own, source "layer" where only
    the fit found them (under: the template they were under), with
    semitones for a pitched copy and flam_of for a grace note."""
    return explain(drums, templates, thresholds, min_gain).events


def peak_score_histogram(drums: np.ndarray, tp: Template) -> list[int]:
    """How many matched-filter peaks fall in each NCC band 0.3-0.4 ... 0.9-1.0:
    a clean template shows a gap between its hits and everything else."""
    score, _ = sliding_ncc(drums, tp.audio[: samples(DETECT_MS[tp.band])], tp.band)
    _, props = find_peaks(score, height=0.3, distance=samples(MIN_GAP_MS[tp.band]))
    return np.histogram(props["peak_heights"], bins=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])[0].tolist()


def pitch_offset(midi: list[float] | np.ndarray) -> tuple[float, float]:
    """Common offset from equal temperament (semitones, -0.5..0.5) of pitches,
    averaged on the circle, and how much they agree on it (0..1)."""
    m = np.asarray(midi, dtype=float)
    if not len(m):
        return 0.0, 0.0
    z = np.mean(np.exp(2j * np.pi * (m - np.round(m))))
    return round(float(np.angle(z) / (2 * np.pi)), 3), round(float(abs(z)), 3)


def notes_808(bass: np.ndarray, tuning: float, fade_db: float = -35.0, own_tuning: bool = True,
              min_agreement: float = 0.5) -> list[dict]:
    """808 notes from the bass stem: onset, end (where it falls fade_db below
    its peak, or the next note), pitch rounded after removing a tuning offset,
    and the pitch where it ends (a slide when they differ).

    The offset is the 808's own when its notes agree on one (own_tuning, at
    least min_agreement): an 808 sample is often detuned on its own. On
    PIRAMIT the melodic stems sit at A440 (-1..+3 cents by spectral peaks and by
    pyin) while its 808 notes sit at +33 cents. Otherwise `tuning` (the
    track's) is used."""
    lo = band(bass, "808")
    on = onsets(lo, delta=0.05)
    w = samples(25)
    env = np.sqrt(np.convolve(lo ** 2, np.ones(w) / w, mode="same"))
    notes = []
    for k, s in enumerate(on):
        nxt = on[k + 1] if k + 1 < len(on) else len(lo)
        a, b = s + samples(60), min(s + samples(450), nxt - samples(20))
        if b - a < samples(60):
            continue
        f0, voiced, _ = librosa.pyin(lo[a:b], fmin=25, fmax=160, sr=SR, frame_length=4096, hop_length=512)
        f = f0[voiced] if np.any(voiced) else np.array([])
        if not len(f):
            continue
        midi = float(librosa.hz_to_midi(np.median(f)))
        tail = f[-3:] if len(f) >= 3 else f
        peak = float(env[s:s + samples(200)].max())
        seg = env[s:nxt]
        below = np.where(seg[samples(100):] < peak * 10 ** (fade_db / 20))[0]
        end = s + samples(100) + int(below[0]) if len(below) else nxt
        notes.append({"t": round(s / SR, 4), "end": round(min(end, nxt) / SR, 4),
                      "midi_float": round(midi, 2),
                      "slide": round(float(librosa.hz_to_midi(np.median(tail))) - midi, 1),
                      "peak": round(peak, 5)})
    offset, agreement = pitch_offset([n["midi_float"] for n in notes])
    use = offset if own_tuning and agreement >= min_agreement else tuning
    for n in notes:
        n["pitch"] = int(round(n["midi_float"] - use))
    return notes


def isolated(notes: list[dict], pitch: int | None = None, ring_s: float = 0.6) -> list[dict]:
    """Notes of one pitch (default: the most common) that ring at least ring_s
    before the next note: what the 808 sounds like on its own."""
    if not notes:
        return []
    pitch = pitch if pitch is not None else Counter(n["pitch"] for n in notes).most_common(1)[0][0]
    out = []
    for k, n in enumerate(notes):
        nxt = notes[k + 1]["t"] if k + 1 < len(notes) else float("inf")
        if n["pitch"] == pitch and nxt - n["t"] >= ring_s:
            out.append(n)
    return out
