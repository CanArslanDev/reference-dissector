"""Where each drum sound plays, and the 808's notes.

Drums, in two passes:

1. discover(): in each band (kick, snare, hat), the strong onsets that sit on
   the sixteenth grid are grouped per grid slot by waveform; groups of at
   least `min_hits` become templates (the averaged hits). Groups that are the
   same sound (in any slot or band) are merged.
2. detect(): each template slides over the whole stem (matched filter); its
   peaks above a threshold are its hits, with a level. Where two templates of
   one band claim the same moment, the better match keeps it.

The templates come from the song itself, so the second pass finds a sound
even where the first could not see it (under other hits, off the grid).
On PIRAMIT the snare templates separate cleanly: 28 hits at NCC >= 0.8 and
nothing between 0.3 and 0.8.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks

from .audio import PRE, SR, band, samples
from .grid import Grid
from .hits import Group, group, merge, onset_template, strength_peaks, template
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


def detect(drums: np.ndarray, templates: list[Template], thresholds: dict | None = None,
           same_hit_ms: float = 15, min_gain: float = 0.25) -> dict[str, list[dict]]:
    """Every hit of every template: {name: [{t, ncc, gain}]} (t in seconds)."""
    thresholds = {**THRESHOLD, **(thresholds or {})}
    raw: dict[str, list[dict]] = {}
    for tp in templates:
        score, gain = sliding_ncc(drums, tp.audio[: samples(DETECT_MS[tp.band])], tp.band)
        peaks, props = find_peaks(score, height=thresholds[tp.band], distance=samples(MIN_GAP_MS[tp.band]))
        found = [{"t": round((p + PRE) / SR, 4), "ncc": round(float(h), 3), "gain": round(float(gain[p]), 4)}
                 for p, h in zip(peaks, props["peak_heights"])]
        # NCC ignores level: the quiet ringing tail of a low hit matches its own
        # template too. Hits far below the template's usual level are dropped.
        if found:
            usual = float(np.median([e["gain"] for e in found]))
            found = [e for e in found if e["gain"] >= min_gain * usual]
        if tp.band == "low":  # and a ringing tail behind a louder hit of the same sound
            kept: list[dict] = []
            for e in found:
                if kept and e["t"] - kept[-1]["t"] < 0.25 and e["gain"] < 0.5 * kept[-1]["gain"]:
                    continue
                kept.append(e)
            found = kept
        raw[tp.name] = found
    # within a band, one moment belongs to the template that matches it best
    out = {k: [] for k in raw}
    by_band: dict[str, list[tuple[str, dict]]] = {}
    for tp in templates:
        by_band.setdefault(tp.band, []).extend((tp.name, e) for e in raw[tp.name])
    for events in by_band.values():
        events.sort(key=lambda ne: ne[1]["t"])
        i = 0
        while i < len(events):
            j = i
            while j + 1 < len(events) and events[j + 1][1]["t"] - events[i][1]["t"] <= same_hit_ms / 1000:
                j += 1
            name, e = max(events[i:j + 1], key=lambda ne: ne[1]["ncc"])
            out[name].append(e)
            i = j + 1
    return out


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
