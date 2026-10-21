"""Melodic notes and chords from the tonal stems, with Basic Pitch.

The model runs once per stem (basicpitch.transcribe, in its own environment)
and two presets are decoded from that run (basicpitch.PRESETS): `melodic` for
the notes that go to Live, `chord` (melodia on) kept for the pads. Per stem:

- pitch: Basic Pitch's semitone, on the stem moved onto A440 by its tuning
  (measured on the stem by the pipeline; the worker takes it out), and that
  tuning as cents; bends are relative to it.
- minimum frequency: just above the 808's usual range, so the 808 bleeding
  into other, piano and guitar is not transcribed (`floor_above_808`,
  FLOOR_STEMS); a vocal keeps its low notes.
- velocity: from the level in the note's own CQT bin (`note_levels`), not
  Basic Pitch's amplitude, which is the mean frame posterior, a confidence.
  Rank correlation with the true velocity on synthetic renders: plucks 0.75
  (amplitude 0.26), a lead 0.58 (0.38), pads 0.12 (0.06): overlapping chord
  tones defeat both. Velocity is linear in that level, 127 at the stem's
  90th percentile, as the drum hits' (live.velocities).
- beats: seconds x BPM / 60 from the start of the audio, like the drum notes
  in the plan; no quantization unless a snap tolerance is given (`to_beats`).
- loop voting where a section's notes repeat (`note_loops`, `vote`); a note
  is put back only where the stem sounds at its pitch.

Chords (`chords`): Basic Pitch's note posteriorgram of the melodic stems
summed into pitch classes per 8th note, matched against 14 chord qualities,
smoothed by Viterbi with a change penalty, and averaged across the repeats
of the loops that the same stems' notes show (`note_loops`). The 808 stays
out of those pitch classes (only pitches above its highest note count) but
names the bass: the 808 note sounding in a span gives the slash bass, and so
the inversion; without one, the lowest chord tone of the melodic stems does.
Chords with the same tones go to that bass, else to min7 and hdim7 over maj6
and min6 (LOW_BONUS, QUALITY_PRIOR).

On 36 synthetic renders (pads with 7ths, sus, dim and slash chords changing
on beat 3, beat 4 and the "and" of 2; full mixes; 4-bar loops played four
times), each one section as analyze gives a steady beat, with the truth's
grid and 808 notes and each render's measured tuning taken out, mir_eval
gives majmin 0.953, majmin_inv 0.804, sevenths 0.904, sevenths_inv 0.783
(root 0.837), against 0.732 / 0.512 / 0.201 / 0.017 for one major or minor
triad per bar from chroma. When the lower root index broke ties between
chords with the same tones (A:min7 read C:maj6), it was 0.877 / 0.738 /
0.824 / 0.714. The vote acted on 11 of the 12 loops (sevenths_inv there
0.894 -> 0.950) and on none of the 63 renders without repeats. A vote on
the cosine of the bars' pitch classes, the first version's, found a loop in
all 36 (same-key bars scored 0.37-0.61 whether they repeat or not) and gave
majmin 0.515 where the vote on notes gave 0.880 (both before the tie-break
and without the tuning step). Without the 808 as bass majmin_inv is 0.715;
per half bar 0.779, and sevenths 0.843.
"""

from __future__ import annotations

from collections import Counter

import librosa
import numpy as np
import soxr

from .audio import SR
from .basicpitch import MIDI_OFFSET, block_times
from .grid import Grid
from .live import velocities

MELODIC_STEMS = ("other", "piano", "guitar", "vocals")
CHORD_STEMS = ("other", "piano", "guitar")  # a rap vocal names no chord
# the stems whose lowest notes are cut at the 808's range (floor_above_808):
# the 808 bleeds into them, not into a vocal, whose own low notes it would
# cut (HAYIR DEMELIYDIM: 458 of 768 vocal notes under the first rule's floor)
FLOOR_STEMS = ("other", "piano", "guitar")

# Harte qualities (add9 is maj(9), 6 is maj6), intervals from the root
QUALITIES = {"maj": (0, 4, 7), "min": (0, 3, 7), "7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11),
             "min7": (0, 3, 7, 10), "sus2": (0, 2, 7), "sus4": (0, 5, 7), "dim": (0, 3, 6),
             "dim7": (0, 3, 6, 9), "hdim7": (0, 3, 6, 10), "aug": (0, 4, 8), "maj(9)": (0, 2, 4, 7),
             "maj6": (0, 4, 7, 9), "min6": (0, 3, 7, 9)}
PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
DEGREES = ("1", "b2", "2", "b3", "3", "4", "b5", "5", "b6", "6", "b7", "7")
STATES = [(r, q) for r in range(12) for q in QUALITIES]  # and N (no chord) after them
# Chords with the same tones (A:min7 = C:maj6, B:hdim7 = D:min6, A:sus2 =
# E:sus4, the dim7s) are told apart by the bass: the 808's note, or without
# one the lowest melodic tone (LOW_BONUS for the chord rooted there). Left
# tied, maj6 and min6 lose to min7 and hdim7 (QUALITY_PRIOR) instead of to
# the lower root index (C first). On the 36 synthetic chord renders, each
# transposed to all 12 keys so that no key's root order decides by luck:
# majmin 0.912 -> 0.955, sevenths_inv 0.749 -> 0.788, root 0.779 -> 0.835.
# Chosen on seeds 1-2 among values where the bass outweighs the prior; seed 3
# (mean of the four scores) 0.828 -> 0.865.
LOW_BONUS = 0.02
QUALITY_PRIOR = {"maj6": -0.01, "min6": -0.01}


def stem_tuning(analysis: dict, stem: str) -> float:
    """The stem's tuning in semitones from A440 (analysis.json), else the track's."""
    tuning = analysis["tuning"]
    return float(tuning.get("per_stem", {}).get(stem, tuning.get("consensus", 0.0)))


def floor_above_808(notes_808: list[dict], share: float = 0.95) -> tuple[float | None, int | None]:
    """The minimum frequency for the stems the 808 bleeds into: the semitone
    above the top of its usual range, and that top, the lowest pitch at or
    under which the 808 sounds `share` of its time (note lengths, t to end).
    (None, None) without an 808.

    The highest pitch heard twice was the first rule. HAYIR DEMELIYDIM has
    273 808 notes, 266 at or under MIDI 43 (96.9 % of the 808's time), but
    two at MIDI 51 put that floor at E3 (164.8 Hz); with share 0.95 it is
    G#2 (103.8 Hz), and guitar keeps 157 notes instead of 128 (159 with no
    floor). On the 63 synthetic renders both rules give the same floors
    (macro onset F1 0.705); with two octave-up slips of 0.2 s added to every
    808 line, the first rule's floor rose an octave and F1 fell to 0.700
    (pads with an 808 by up to 0.18), this one's held (0.706)."""
    if not notes_808:
        return None, None
    pitch = np.array([n["pitch"] for n in notes_808])
    length = np.array([max(n["end"] - n["t"], 0.0) for n in notes_808])
    length = length if length.sum() > 0 else np.ones(len(pitch))
    order = np.argsort(pitch, kind="stable")
    k = int(np.searchsorted(np.cumsum(length[order]) / length.sum(), share))
    top = int(pitch[order][min(k, len(pitch) - 1)])
    return round(float(librosa.midi_to_hz(top + 1)), 2), top


def level_meter(x: np.ndarray, tuning: float = 0.0, window_s: float = 0.06, sr: int = SR):
    """A function notes -> level (dB) of each in the stem x: the CQT magnitude
    in its own semitone bin (centred on the stem's tuning), the most over the
    first window_s after its start. NaN outside MIDI 24-107."""
    y = soxr.resample(np.asarray(x, dtype=np.float32), sr, 22050)
    C = np.abs(librosa.cqt(y, sr=22050, hop_length=128, fmin=librosa.midi_to_hz(24 + tuning), n_bins=84,
                           bins_per_octave=12))
    t = librosa.times_like(C, sr=22050, hop_length=128)

    def levels(notes: list[dict]) -> np.ndarray:
        out = np.full(len(notes), np.nan)
        for i, n in enumerate(notes):
            b = n["pitch"] - 24
            a, e = np.searchsorted(t, n["t"]), np.searchsorted(t, n["t"] + window_s)
            if 0 <= b < C.shape[0] and e > a:
                out[i] = 20 * np.log10(C[b, a:e].max() + 1e-9)
        return out
    return levels


def note_levels(x: np.ndarray, notes: list[dict], tuning: float = 0.0, window_s: float = 0.06,
                sr: int = SR) -> np.ndarray:
    """Level (dB) of each note (level_meter)."""
    return level_meter(x, tuning, window_s, sr)(notes)


def to_beats(t: float, grid: Grid, snap_ms: float | None = None, per_beat: int = 4) -> tuple[float, float]:
    """Beats from the start of the audio (as Grid.beats), and how far the time
    moved (ms): with snap_ms, to the nearest 1/per_beat of a beat on the grid
    when that is within snap_ms and not before the audio starts (a MIDI file
    cannot hold a negative time)."""
    moved = 0.0
    if snap_ms:
        step = grid.beat / per_beat
        line = grid.phase + round((t - grid.phase) / step) * step
        if line >= 0 and abs(line - t) * 1000 <= snap_ms:
            moved, t = (line - t) * 1000, line
    return float(grid.beats(t)), moved


# ── Loop voting ──────────────────────────────────────────────────────────────

def bar_notes(notes: list[dict], grid: Grid) -> dict[int, list[dict]]:
    """Notes by bar (0-based, Grid.bar_of: up to 50 ms early counts as the bar)."""
    out: dict[int, list[dict]] = {}
    for n, b in zip(notes, grid.bar_of([n["t"] for n in notes]) if notes else []):
        out.setdefault(int(b), []).append(n)
    return out


def _offset(n: dict, grid: Grid, bar: int) -> float:
    return n["t"] - (grid.phase + bar * grid.bar)


def note_similarity(a: list[dict], b: list[dict], bar_a: int, bar_b: int, grid: Grid, tol_s: float = 0.03) -> float:
    """F-measure of two bars' notes: same pitch, the same place in the bar
    within tol_s. Two empty bars are alike (1.0)."""
    if not a and not b:
        return 1.0
    free = [(n["pitch"], _offset(n, grid, bar_b)) for n in b]
    hits = 0
    for n in a:
        p, o = n["pitch"], _offset(n, grid, bar_a)
        j = next((j for j, (q, r) in enumerate(free) if q == p and abs(r - o) <= tol_s), None)
        if j is not None:
            hits += 1
            free.pop(j)
    return 2 * hits / (len(a) + len(b))


def loop_groups(sections: list[dict], similar, periods=(1, 2, 4, 8), min_similarity: float = 0.35,
                margin: float = 0.05) -> list[dict]:
    """Where bars repeat. Per section name, the period in bars whose bar pairs
    (b, b + period inside one occurrence) are most alike by similar(b1, b2)
    -> 0..1; the shortest period within `margin` of the best wins, a loop of 4
    bars being also one of 8. None reaching min_similarity: no loop there.

    The occurrences of a name are then put together only where they play the
    same music (`_occurrences`). structure.sections names a run by its drums
    and 808 alone, so a verse and a hook over the same drums share a name,
    and a section can return at another point of its loop. One group per set
    of occurrences that repeat: per loop position, the bars (0-based).

    min_similarity 0.35 sits between the renders that repeat and those that do
    not: 12 synthetic 4-bar loops played four times scored 0.40-0.82 (period 4
    found in all 12), 63 renders without repeats at most 0.27 (median 0.07)."""
    runs: dict[str, list[tuple[int, int]]] = {}
    for s in sections:
        runs.setdefault(s["name"], []).append((s["start_bar"] - 1, s["end_bar"] - 1))
    out = []
    for name, spans in runs.items():
        scored = []
        for p in periods:
            pairs = [(b, b + p) for a, e in spans for b in range(a, e - p + 1)]
            if len(pairs) >= 2:
                scored.append((float(np.mean([similar(x, y) for x, y in pairs])), p))
        if not scored or max(s for s, _ in scored) < min_similarity:
            continue
        best = max(s for s, _ in scored)
        period = min((p for s, p in scored if s >= best - margin))
        out += [{"section": name, "period_bars": period, **g}
                for g in _occurrences(spans, similar, period, min_similarity)]
    return out


def _occurrences(spans: list[tuple[int, int]], similar, period: int, min_similarity: float) -> list[dict]:
    """The occurrences (first, last bar) of one section name, grouped. An
    occurrence joins a group when its bars match the group's bars at the
    same loop position (mean similar() at least min_similarity), at the loop
    phase that matches best: it can start 0..period-1 bars into the loop (the
    phase that keeps the loop running through the song is tried first).
    Otherwise it starts a group of its own. A group votes when its pairs
    (bar and next repeat inside an occurrence, and the matches that joined
    it) average min_similarity; its loop positions count from its first
    occurrence's start."""
    groups: list[dict] = []
    for a, e in spans:
        own = [similar(b, b + period) for b in range(a, e - period + 1)]
        best = None  # (mean, group, shift, similarities)
        for g in groups:
            first = g["runs"][0][0]
            for shift in sorted(range(period), key=lambda s: s != (a - first) % period):
                found = [similar(x, b) for b in range(a, e + 1) for x in g["at"].get((b - a + shift) % period, [])]
                if found and (best is None or np.mean(found) > best[0]):
                    best = (float(np.mean(found)), g, shift, found)
        if best is not None and best[0] >= min_similarity:
            _, g, shift, found = best
        else:
            g, shift, found = {"runs": [], "at": {}, "pairs": []}, 0, []
            groups.append(g)
        g["runs"].append((a, e))
        g["pairs"] += own + found
        for b in range(a, e + 1):
            g["at"].setdefault((b - a + shift) % period, []).append(b)
    return [{"similarity": round(float(np.mean(g["pairs"])), 3), "runs": [[a + 1, e + 1] for a, e in g["runs"]],
             "positions": [sorted(g["at"][k]) for k in sorted(g["at"])]}
            for g in groups if g["pairs"] and np.mean(g["pairs"]) >= min_similarity
            and max(len(v) for v in g["at"].values()) >= 2]


def note_loops(notes: list[dict], grid: Grid, sections: list[dict]) -> list[dict]:
    """loop_groups with bars compared by their notes (note_similarity)."""
    by_bar = bar_notes(notes, grid)
    return loop_groups(sections, lambda a, b: note_similarity(by_bar.get(a, []), by_bar.get(b, []), a, b, grid))


def vote(notes: list[dict], grid: Grid, groups: list[dict], tol_s: float = 0.03,
         drop_below: float | None = None, level=None, gate_db: float = 12.0) -> tuple[list[dict], list[dict]]:
    """Across the repeats of each loop position, a note is the same pitch at
    the same place in the bar (within tol_s). Heard in more than half of the
    repeats, it is put back where it is missing (the median place, length and
    amplitude of the others), but with `level` (level_meter of the stem) only
    where the stem sounds at its pitch: at most gate_db under the median
    level of the notes it is copied from (else counted as too_quiet). Heard
    in fewer than half, it stays unless drop_below is given: then it goes
    when its onset posterior is under it. Returns the notes and what changed
    per group.

    Measured on synthetic 4-bar loops played four times (a different vocal
    under each repeat; seeds 1-2 chose, seed 3 checked), 12 with bar 16 a
    variation (its notes move pitch) and 12 with bar 16's pluck and lead
    silent (a drop-out). Putting notes back without the level check raised
    onset F1 0.533 -> 0.540 on the variations but filled the odd bar: of 105
    notes put into it, 93 were false (drop-outs: 113, 98 false). True
    restorations sat 0.6 dB under their cluster's level (median), false ones
    11.7 dB (drop-outs: 0.8 and 15.8 dB). With gate_db 12, F1 0.540 -> 0.544
    and precision 0.792 -> 0.807 on the variations, F1 0.533 -> 0.538 on the
    drop-outs (no vote: 0.533 and 0.528); gates of 6 to 15 dB scored within
    0.002 of each other. The odd bars still get 54 and 35 notes (11 and 12
    true): every false one sits where a pad holds its pitch class, which a
    level cannot tell from a new note.

    Dropping every note heard in fewer than half lost 0.032 (all 12 worse) and
    the variation (its recall 0.60 -> 0.04): most transcription errors repeat
    in every repeat, and true notes missed in some repeats went with the
    bleed. Dropping only weak ones (drop_below 0.7; true notes' onset
    posterior had a median of 0.80, false ones 0.64) lost 0.010 under ordinary
    bleed and gained 0.003 under loud vocals."""
    by_bar = bar_notes(notes, grid)
    heard_db = dict(zip(map(id, notes), level(notes))) if level is not None and notes else {}
    out = [n for b, ns in by_bar.items() for n in ns if not any(b in pos for g in groups for pos in g["positions"])]
    report = []
    for g in groups:
        row = {"section": g["section"], "runs": g.get("runs"), "period_bars": g["period_bars"],
               "similarity": g["similarity"], "repeats": max(len(p) for p in g["positions"]), "notes": 0,
               "restored": 0, "too_quiet": 0, "dropped": 0}
        for bars in g["positions"]:
            entries = sorted(((n["pitch"], _offset(n, grid, b), b, n) for b in bars for n in by_bar.get(b, [])),
                             key=lambda e: (e[0], e[1]))
            clusters: list[list[tuple]] = []  # one note per bar: a same-pitch roll stays apart
            for e in entries:
                c = clusters[-1] if clusters else None
                if c and c[0][0] == e[0] and e[1] - c[-1][1] <= tol_s and e[1] - c[0][1] <= 2 * tol_s \
                        and all(x[2] != e[2] for x in c):
                    c.append(e)
                else:
                    clusters.append([e])
            for c in clusters:
                heard = {e[2] for e in c}
                row["notes"] += len(c)
                if 2 * len(heard) > len(bars):
                    floor = np.median([heard_db[id(e[3])] for e in c]) - gate_db if heard_db else None
                    for b in bars:
                        if b not in heard:
                            new = _restored(c, b, grid)
                            if floor is None or level([new])[0] >= floor:
                                out.append(new)
                                row["restored"] += 1
                            else:
                                row["too_quiet"] += 1
                for e in c:
                    if 2 * len(heard) < len(bars) and drop_below is not None and e[3].get("onset", 0.0) < drop_below:
                        row["dropped"] += 1
                    else:
                        out.append(e[3])
        report.append(row)
    return sorted(out, key=lambda n: (n["t"], n["pitch"])), report


def _restored(cluster: list[tuple], bar: int, grid: Grid) -> dict:
    """The cluster's note put into `bar`; its place is the median offset of
    the others, which can be up to 50 ms before the bar line (Grid.bar_of),
    so a note in the first bar is kept at or after 0."""
    model = cluster[len(cluster) // 2][3]
    t = max(0.0, grid.phase + bar * grid.bar + float(np.median([e[1] for e in cluster])))
    length = float(np.median([e[3]["end"] - e[3]["t"] for e in cluster]))
    return {**model, "t": round(t, 4), "end": round(t + length, 4), "onset": 0.0, "bend": [],
            "amplitude": round(float(np.median([e[3]["amplitude"] for e in cluster])), 4), "voted": "restored"}


# ── Chords ───────────────────────────────────────────────────────────────────

def span_starts(grid: Grid, duration: float, step_beats: float) -> np.ndarray:
    """Span start times on the grid (its lines extended back to 0) up to duration."""
    step = step_beats * grid.beat
    first = grid.phase - np.ceil(grid.phase / step) * step
    return np.arange(first, duration, step)


def salience(note_pg: np.ndarray, starts: np.ndarray, step_s: float, lowest_midi: int = MIDI_OFFSET,
             floor: float = 0.15, time_scale: float = 1.0) -> np.ndarray:
    """(spans, 88): each span's mean note posterior per pitch (frames timed by
    block_times x time_scale, the worker's), below `floor` set to 0; pitches
    under lowest_midi are 0."""
    t = block_times(len(note_pg)) * time_scale
    edges = np.searchsorted(t, np.r_[starts, starts[-1] + step_s])
    S = np.zeros((len(starts), note_pg.shape[1]))
    for k in range(len(starts)):
        if edges[k + 1] > edges[k]:
            S[k] = note_pg[edges[k]:edges[k + 1]].mean(axis=0)
    S[S < floor] = 0.0
    S[:, :max(0, lowest_midi - MIDI_OFFSET)] = 0.0
    return S


def pitch_class_weights(S: np.ndarray) -> np.ndarray:
    """(spans, 12) from (spans, 88): column i is MIDI 21 + i."""
    return S @ np.eye(12)[(np.arange(S.shape[1]) + MIDI_OFFSET) % 12]


def bass_per_span(notes_808: list[dict], starts: np.ndarray, step_s: float, cover: float = 0.5) -> list[int | None]:
    """Pitch class of the 808 note that sounds through at least `cover` of each span."""
    out: list[int | None] = []
    for a in starts:
        best, pc = cover * step_s, None
        for n in notes_808:
            overlap = min(a + step_s, n["end"]) - max(a, n["t"])
            if overlap >= best:
                best, pc = overlap, n["pitch"] % 12
        out.append(pc)
    return out


def lowest_tone(S: np.ndarray, floor: float = 0.2) -> list[int | None]:
    """Pitch class of the lowest pitch reaching `floor` in each span of S (spans, 88)."""
    out: list[int | None] = []
    for row in S:
        idx = np.flatnonzero(row >= floor)
        out.append(int((idx[0] + MIDI_OFFSET) % 12) if len(idx) else None)
    return out


def chord_scores(W: np.ndarray, bass: list[int | None], min_total: float = 0.2,
                 low: list[int | None] | None = None) -> np.ndarray:
    """(spans, states + N). A chord scores the share of the span's pitch-class
    weight on its tones, less 0.3 x the share of its tones that are nearly
    absent (< 4 %) and 0.01 per tone over three, plus QUALITY_PRIOR; the 808
    adds 0.08 when it plays the root and takes 0.05 when it plays no chord
    tone; in a span without it, the lowest melodic tone (low) adds LOW_BONUS
    to the chords rooted on it. A span with less total weight than min_total
    is N (no chord)."""
    T = np.zeros((len(STATES), 12))
    for j, (r, q) in enumerate(STATES):
        T[j, [(r + i) % 12 for i in QUALITIES[q]]] = 1.0
    n = T.sum(axis=1)
    roots = np.array([r for r, _ in STATES])
    prior = np.array([QUALITY_PRIOR.get(q, 0.0) for _, q in STATES])
    total = W.sum(axis=1, keepdims=True)
    share = W / np.where(total > 0, total, 1.0)
    chord = share @ T.T - 0.3 * ((share < 0.04) @ T.T) / n - 0.01 * (n - 3) + prior
    for k, b in enumerate(bass):
        if b is not None:
            chord[k] += np.where(roots == b, 0.08, np.where(T[:, b] == 0, -0.05, 0.0))
        elif low is not None and low[k] is not None:
            chord[k] += np.where(roots == low[k], LOW_BONUS, 0.0)
    silent = total[:, 0] < min_total
    chord[silent] = -np.inf
    return np.c_[chord, np.where(silent, 0.0, -np.inf)]


def viterbi(scores: np.ndarray, penalty: float) -> np.ndarray:
    """The state path with the highest total score, a change costing `penalty`."""
    n, m = scores.shape
    dp = scores[0].copy()
    back = np.zeros((n, m), dtype=int)
    for k in range(1, n):
        j = int(np.argmax(dp))
        move = dp[j] - penalty
        stay = dp >= move
        back[k] = np.where(stay, np.arange(m), j)
        dp = scores[k] + np.where(stay, dp, move)
    path = [int(np.argmax(dp))]
    for k in range(n - 1, 0, -1):
        path.append(int(back[k, path[-1]]))
    return np.array(path[::-1])


def harte(root: int, quality: str, bass: int | None) -> str:
    """'A:min7/b7': root, quality, and the bass as a degree when it is not the root."""
    label = f"{PITCH_CLASSES[root]}:{quality}"
    return label + (f"/{DEGREES[(bass - root) % 12]}" if bass is not None and bass != root else "")


def vote_spans(W: np.ndarray, bass: list[int | None], starts: np.ndarray, grid: Grid,
               groups: list[dict]) -> tuple[np.ndarray, list[int | None]]:
    """Pitch-class weights averaged, and the bass taken by majority, over the
    repeats of each loop position (span by span within the bar)."""
    W, bass = W.copy(), list(bass)
    bar_of = grid.bar_of(starts + 1e-6)
    per_bar = max(1, int(round(grid.bar / (starts[1] - starts[0])))) if len(starts) > 1 else 1
    index = {(int(b), int(round((s - grid.phase - b * grid.bar) / grid.bar * per_bar))): k
             for k, (b, s) in enumerate(zip(bar_of, starts))}
    for g in groups:
        for bars in g["positions"]:
            for j in range(per_bar):
                ks = [index[(b, j)] for b in bars if (b, j) in index]
                if len(ks) < 2:
                    continue
                W[ks] = W[ks].mean(axis=0)
                heard = Counter(bass[k] for k in ks if bass[k] is not None)
                if heard:
                    pc = heard.most_common(1)[0][0]
                    for k in ks:
                        bass[k] = pc
    return W, bass


def chords(note_pgs: list[np.ndarray], grid: Grid, duration: float, notes_808: list[dict],
           lowest_midi: int = MIDI_OFFSET, step_beats: float = 0.5, penalty: float = 0.15,
           groups: list[dict] | None = None, tone_floor: float = 0.2,
           time_scales: list[float] | None = None) -> dict:
    """Chord segments from the note posteriorgrams of the melodic stems (summed;
    each stem's frames at its worker time_scale) and the 808 as bass; with
    loop groups (note_loops on the same stems' notes), weights are averaged
    across the repeats first.

    Only the melodic stems and pitches above lowest_midi (the 808's highest
    note + 1) count, so neither the bass stem nor the 808's own range enters
    the pitch classes. Zeroing the 808's harmonics as well (its note + 12, 19,
    24 in each span) cost majmin 0.882 -> 0.871 and sevenths_inv 0.717 ->
    0.706 on the 36 renders (+ 28 too: 0.832 / 0.671): the root an octave or
    two above the 808 is usually a chord tone."""
    starts = span_starts(grid, duration, step_beats)
    step = step_beats * grid.beat
    scales = time_scales or [1.0] * len(note_pgs)
    S = sum(salience(pg, starts, step, lowest_midi, time_scale=sc) for pg, sc in zip(note_pgs, scales))
    W = pitch_class_weights(S)
    bass = bass_per_span(notes_808, starts, step)
    groups = groups or []
    if groups:
        W, bass = vote_spans(W, bass, starts, grid, groups)
    path = viterbi(chord_scores(W, bass, low=lowest_tone(S, tone_floor)), penalty)
    segments: list[dict] = []
    for k, j in enumerate(path):
        if segments and segments[-1]["state"] == j and segments[-1]["bass_808"] == bass[k]:
            segments[-1]["spans"].append(k)
        else:
            segments.append({"state": int(j), "bass_808": bass[k], "spans": [k]})
    out = []
    for seg in segments:
        a, b = starts[seg["spans"][0]], starts[seg["spans"][-1]] + step
        if seg["state"] == len(STATES):
            label, bass_from = "N", None
        else:
            root, quality = STATES[seg["state"]]
            tones = {(root + i) % 12 for i in QUALITIES[quality]}
            low, bass_from = seg["bass_808"], "808"
            if low is None:
                mean = S[seg["spans"]].mean(axis=0)
                lows = [i + MIDI_OFFSET for i in np.flatnonzero(mean >= tone_floor) if (i + MIDI_OFFSET) % 12 in tones]
                low, bass_from = (lows[0] % 12, "melodic") if lows else (None, None)
            label = harte(root, quality, low)
        out.append({"start": round(float(max(a, 0.0)), 4), "end": round(float(min(b, duration)), 4),
                    "start_time": round(float(grid.beats(max(a, 0.0))), 4), "label": label, "bass_from": bass_from})
    merged: list[dict] = []
    for c in out:
        if merged and merged[-1]["label"] == c["label"]:
            merged[-1]["end"] = c["end"]
        else:
            merged.append(c)
    return {"step_beats": step_beats, "penalty": penalty, "lowest_midi": lowest_midi,
            "loops": [{k: v for k, v in g.items() if k != "positions"} for g in groups], "segments": merged}


# ── Per stem ─────────────────────────────────────────────────────────────────

def stem_notes(raw: list[dict], x: np.ndarray, grid: Grid, tuning: float = 0.0, sections: list[dict] | None = None,
               snap_ms: float | None = None, voting: bool = True,
               drop_below: float | None = None) -> tuple[list[dict], list[dict]]:
    """One stem's notes from the worker's melodic preset (start_s, end_s, midi,
    amplitude, onset, bend): voted across loop repeats, levels and velocities
    measured in the stem x after voting (restored notes get their own), then
    placed in beats. Returns the notes and the voting report."""
    notes = [{"t": n["start_s"], "end": n["end_s"], "pitch": n["midi"], "amplitude": n["amplitude"],
              "onset": n.get("onset", 0.0), "bend": n.get("bend", [])} for n in raw]
    if not notes:
        return [], []
    meter = level_meter(x, tuning)
    report: list[dict] = []
    if voting and sections:
        notes, report = vote(notes, grid, note_loops(notes, grid, sections), drop_below=drop_below, level=meter)
    levels = meter(notes)
    fill = float(np.nanmedian(levels)) if np.isfinite(levels).any() else -60.0
    levels = np.where(np.isfinite(levels), levels, fill)
    cents = int(round(100 * tuning))
    out = []
    for n, lv, v in zip(notes, levels, velocities(list(10 ** (levels / 20)))):
        start, moved = to_beats(n["t"], grid, snap_ms)
        end = float(grid.beats(n["end"]))
        row = {"t": n["t"], "end": n["end"], "pitch": n["pitch"], "cents": cents,
               "start_time": round(start, 4), "duration": round(max(0.05, end - start), 4),
               "velocity": v, "level_db": round(float(lv), 1), "amplitude": n["amplitude"],
               "onset": n["onset"]}
        if moved:
            row["snapped_ms"] = round(moved, 1)
        if any(n["bend"]):
            row["bend"] = n["bend"]  # per frame (11.6 ms), thirds of a semitone, corrected
        if n.get("voted"):
            row["voted"] = n["voted"]
        out.append(row)
    return out, report


def build(analysis: dict, transcribed: dict[str, dict], audio: dict[str, np.ndarray],
          note_pgs: dict[str, np.ndarray], snap_ms: float | None = None, voting: bool = True,
          drop_below: float | None = None, tunings: dict[str, float] | None = None) -> dict:
    """melodic.json from the worker's output per stem (transcribed: its JSON,
    notes per preset, on A440 when the worker took the stem's tuning out),
    the stems' audio and the posteriorgrams of the chord stems. tunings: per
    stem, the tuning the worker took out (default: analysis.json's)."""
    g = analysis["grid"]
    grid = Grid(g["bpm"], g["phase"], g.get("beats_per_bar", 4))
    sections = analysis.get("sections") or []
    _, top = floor_above_808(analysis["notes_808"])
    out: dict = {"grid": {"bpm": grid.bpm, "phase": grid.phase}, "stems": {}}
    for stem, t in transcribed.items():
        st = tunings[stem] if tunings and stem in tunings else stem_tuning(analysis, stem)
        notes, report = stem_notes(t["notes"]["melodic"], audio[stem], grid, st, sections, snap_ms, voting,
                                   drop_below)
        out["stems"][stem] = {"tuning_cents": int(round(100 * st)), "notes": notes, "voting": report,
                              "chord_preset": [{"t": n["start_s"], "end": n["end_s"], "pitch": n["midi"],
                                                "amplitude": n["amplitude"]} for n in t["notes"].get("chord", [])]}
    used = [s for s in CHORD_STEMS if s in note_pgs]
    if used:
        played = [{"t": n["start_s"], "end": n["end_s"], "pitch": n["midi"]}
                  for s in used if s in transcribed for n in transcribed[s]["notes"]["melodic"]]
        groups = note_loops(played, grid, sections) if voting and sections and played else []
        out["chords"] = chords([note_pgs[s] for s in used], grid, analysis["source"]["duration_s"],
                               analysis["notes_808"], lowest_midi=top + 1 if top is not None else MIDI_OFFSET,
                               groups=groups, time_scales=[transcribed.get(s, {}).get("time_scale", 1.0)
                                                           for s in used])
    return out

