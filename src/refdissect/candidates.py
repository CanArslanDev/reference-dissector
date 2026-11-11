"""Style candidates: short new beats drawn from a plan's statistics, to choose by ear.

The sketch step of a style beat (docs/11-style-beat-lessons.md): before a
whole beat is built, several 8-bar candidates are written, rendered and
played to the listener, who picks. Here they are drawn from the statistics
of a plan (pipeline.plan), not from its notes:

- tempo and meter, as the plan has them;
- swing (the analysis' grid.swing, or read from the plan's hats);
- each drum role's place in the bar (the share of bars with a hit on each
  16th) and how much a bar changes from the one two bars before;
- the hats' rolls: the share of beats holding one, their division and length;
- the 808: how often it starts with a kick and off one (per 16th), how often
  it repeats its pitch or plays the chord's root, its register, how long it
  holds, and how often it slides (and by how much);
- the chords (melodic.json): the vocabulary that covers most of the time, the
  changes between them, and how long one lasts.

Every candidate uses the plan's own pads, 808 one-shot and faders, so it
sounds like the rebuild; its notes are new. A candidate that would play the
reference's bars back (copy_score over max_copy) is drawn again: docs/09,
no new beat copies the reference's notes. Each candidate is drawn from its
own generator, seeded by (seed, number): the same seed gives the same
candidates, and candidate 3 is the same whether 3 or 10 are drawn.

The candidates are rendered with render.render_plan (drums and 808 as Live
plays them, after the plan's faders), and the chords as a plain additive pad
(a sketch, not a sound choice). The 808 and the chords duck under the
candidate's kicks by a gain curve fitted to the reference's own (kick_curve
on its stems): a stand-in for a Compressor keyed from the kick. Each
candidate's mix goes into one listening round (rounds.listening_round), and
its stems are measured against the reference's same bars (symmetric.measure
and compare): ducking and drum balance. Its stems are clean renders while the
reference's are separated, so differences there include the separation's
(docs/11); with separate, each candidate's mix goes through Demucs first.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .audio import SR

STEP = 0.25  # beats: a 16th
ROLES = ("kick", "snare", "hat", "perc")
ROLL_DIVISIONS = (6, 8, 12, 16)  # notes per beat a hat roll can play: 16th triplets, 32nds, 32nd triplets, 64ths
ROLL_IOI = 0.75 * STEP  # hats closer than this (beats) are a roll
SAME_BEAT = 0.02  # hats closer than this (beats) are one strike on two pads
VOCAB_SHARE, VOCAB_MAX = 0.8, 6  # the chords that cover this share of the time, at most this many
LEGATO_S = 0.01  # a sliding 808 note's predecessor is held this long over it (live.LEGATO_S)
GLIDE_MS = 60.0  # the 808's Glide Time when the plan has none
CHORD_LOW = 55  # the chords' lowest root when the plan has no melodic notes to place them by
# without reference stems: a kick-keyed duck down within 3 ms, back to 0 dB in a straight line (dB)
DUCK_DEFAULT = {"808": {"s": [0.0, 0.003, 0.2], "db": [0.0, -12.0, 0.0]},
                "chords": {"s": [0.0, 0.003, 0.27], "db": [0.0, -10.0, 0.0]}}
DUCK_FLOOR_DB = -24.0  # a reference curve deeper than this (an 808 that stops) ducks this deep
MAX_COPY = 0.75
TRIES = 50


# ── Reading the plan ──

def bar_beats(plan: dict) -> float:
    """A bar of the plan's meter in quarter beats."""
    return int(plan.get("beats_per_bar", 4)) * 4 / int(plan.get("beat_unit", 4))


def steps_per_bar(plan: dict) -> int:
    return int(round(bar_beats(plan) / STEP))


def origin(plan: dict) -> float:
    """Where the plan's bars start (beats): its first locator, which sits on
    the grid's first bar line, modulo a bar; 0 without locators."""
    loc = plan.get("locators") or []
    return float(loc[0][1]) % bar_beats(plan) if loc else 0.0


def n_bars(plan: dict) -> int:
    return int(np.ceil((float(plan["length_beats"]) - origin(plan)) / bar_beats(plan) - 1e-9))


def section_bars(plan: dict, section: str | None = None) -> list[int]:
    """The bars (from 0) the statistics come from: every bar, or those of
    every section named `section` (its locators: "A (1)" is section A)."""
    total = n_bars(plan)
    if not section:
        return list(range(total))
    bar, o = bar_beats(plan), origin(plan)
    loc = sorted(((str(n).split(" (")[0], float(b)) for n, b in plan.get("locators") or []), key=lambda x: x[1])
    out = []
    for k, (name, beat) in enumerate(loc):
        if name != section:
            continue
        end = loc[k + 1][1] if k + 1 < len(loc) else float(plan["length_beats"])
        out += range(int(round((beat - o) / bar)), int(round((end - o) / bar)))
    if not out:
        raise ValueError(f"No section named {section!r} (locators: {[n for n, _ in loc]})")
    return sorted(set(b for b in out if 0 <= b < total))


def _where(plan: dict, start_time: float) -> tuple[int, int]:
    """(bar, 16th in the bar) of a note, to the nearest 16th."""
    q = int(round((start_time - origin(plan)) / STEP))
    return q // steps_per_bar(plan), q % steps_per_bar(plan)


def drum_roles(plan: dict) -> dict[int, str]:
    return {int(k): (p.get("role") if p.get("role") in ROLES else "perc")
            for k, p in plan["tracks"]["drums"]["pads"].items()}


def chord_line(melodic: dict | None) -> list[dict]:
    """melodic.json's chord segments as {start, label} in beats, time order,
    one segment per change (no-chord segments kept as "N")."""
    segs = ((melodic or {}).get("chords") or {}).get("segments") or []
    out = []
    for s in sorted(segs, key=lambda s: s["start_time"]):
        if out and out[-1]["label"] == s["label"]:
            continue
        out.append({"start": float(s["start_time"]), "label": str(s["label"])})
    return out


def chord_at(line: list[dict], beat: float) -> str | None:
    label = None
    for s in line:
        if s["start"] > beat + 1e-6:
            break
        label = s["label"]
    return label


def chord_pitches(label: str) -> tuple[int, list[int]] | None:
    """(root pitch class, the chord's pitch classes over the root in
    semitones) of a mir_eval label; None for no chord."""
    import mir_eval
    try:
        root, bitmap, _ = mir_eval.chord.encode(label)
    except Exception:  # noqa: BLE001 - an unreadable label is no chord
        return None
    if root < 0:
        return None
    return int(root), [int(i) for i in np.flatnonzero(bitmap)]


# ── Statistics ──

def _change(sets: list[set], lag: int = 2) -> float:
    """How much a bar's hits differ from those `lag` bars before: the share
    of their union that only one plays, averaged."""
    rates = [len(a ^ b) / len(a | b) for a, b in zip(sets, sets[lag:]) if a | b]
    return round(float(np.mean(rates)), 3) if rates else 0.0


def style_stats(plan: dict, analysis: dict | None = None, melodic: dict | None = None,
                section: str | None = None) -> dict:
    """The statistics a candidate is drawn from (module docstring), over the
    bars of `section` (default: every bar)."""
    bars = section_bars(plan, section)
    in_stats = set(bars)
    spb, bar = steps_per_bar(plan), bar_beats(plan)
    roles = drum_roles(plan)
    hits: dict[str, dict[int, set]] = {r: defaultdict(set) for r in ROLES}
    pads: dict[str, Counter] = {r: Counter() for r in ROLES}
    pad_at: dict[str, dict[int, Counter]] = {r: defaultdict(Counter) for r in ROLES}
    vel: dict[str, list[int]] = {r: [] for r in ROLES}
    hat_times = []
    for n in plan["tracks"]["drums"]["notes"]:
        role = roles.get(int(n["pitch"]))
        b, s = _where(plan, n["start_time"])
        if role is None or b not in in_stats:
            continue
        hits[role][b].add(s)
        pads[role][int(n["pitch"])] += 1
        pad_at[role][s][int(n["pitch"])] += 1
        vel[role].append(int(n["velocity"]))
        if role == "hat":
            hat_times.append(float(n["start_time"]))
    nb = max(1, len(bars))
    drums = {}
    for r in ROLES:
        p = [sum(s in hits[r][b] for b in bars) / nb for s in range(spb)]
        drums[r] = {"p": [round(x, 3) for x in p], "change": _change([hits[r][b] for b in bars]),
                    "pads": {str(k): c for k, c in pads[r].most_common()},
                    "pad_at": {str(s): pad_at[r][s].most_common(1)[0][0] for s in pad_at[r]},
                    "velocities": sorted(vel[r])}
    # hat rolls: beats holding hats closer than ROLL_IOI, their division and length
    h = np.unique(np.round(hat_times, 4))
    if len(h):
        h = h[np.r_[True, np.diff(h) >= SAME_BEAT]]
    gaps = np.diff(h)
    rolled, divisions, lengths, run = set(), [], [], 0
    for k, g in enumerate(gaps):
        if g < ROLL_IOI:
            rolled.add(int(np.floor(h[k] + 1e-6)))
            divisions.append(min(ROLL_DIVISIONS, key=lambda d: abs(1 / d - g)))
            run += 1
        elif run:
            lengths.append(run * float(np.median(gaps[k - run:k])))
            run = 0
    if run:
        lengths.append(run * float(np.median(gaps[len(gaps) - run:])))
    beats = nb * bar
    rolls = {"rate": round(len(rolled) / beats, 3) if beats else 0.0,
             "divisions": dict(Counter(divisions)), "beats": [round(x, 3) for x in lengths]}
    in_roll = np.zeros(len(h), bool)  # rolls are not swing
    if len(h) > 1:
        in_roll[1:] |= gaps < ROLL_IOI
        in_roll[:-1] |= gaps < ROLL_IOI
    swing = _swing(plan, analysis, h[~in_roll].tolist())
    # the 808
    t808 = plan["tracks"].get("808") or {}
    notes = sorted(t808.get("notes") or [], key=lambda n: n["start_time"])
    line = chord_line(melodic)
    kick_at = hits["kick"]
    with_kick = [0, 0]  # 808 starts on a kick's 16th, kick 16ths
    off_kick = np.zeros(spb), np.zeros(spb)
    on808: dict[int, set] = defaultdict(set)
    pitches, held, slides, repeats, on_root, rooted = [], [], [], [0, 0], 0, 0
    for k, n in enumerate(notes):
        b, s = _where(plan, n["start_time"])
        if b not in in_stats:
            continue
        on808[b].add(s)
        pitches.append(int(n["pitch"]))
        if k + 1 < len(notes):
            gap = notes[k + 1]["start_time"] - n["start_time"]
            if gap > 0:
                held.append(min(1.0, n["duration"] / gap))
        if k > 0:
            prev = notes[k - 1]
            repeats[0] += int(n["pitch"]) == int(prev["pitch"])
            repeats[1] += 1
            if prev["start_time"] + prev["duration"] > n["start_time"] + 1e-6:
                slides.append(int(n["pitch"]) - int(prev["pitch"]))
        label = chord_at(line, n["start_time"]) if line else None
        cp = chord_pitches(label) if label else None
        if cp:
            rooted += 1
            on_root += int(n["pitch"]) % 12 == cp[0]
    for b in bars:
        for s in range(spb):
            if s in kick_at[b]:
                with_kick[1] += 1
                with_kick[0] += s in on808[b]
            else:
                off_kick[1][s] += 1
                off_kick[0][s] += s in on808[b]
    lo, med, hi = (np.percentile(pitches, [5, 50, 95]) if pitches else (24, 24, 36))
    glide = t808.get("glide_time_ms") or (float(np.median([s["ms"] for s in t808.get("slides", []) if s.get("ms")]))
                                          if any(s.get("ms") for s in t808.get("slides", [])) else GLIDE_MS)
    b808 = {"with_kick": round(with_kick[0] / with_kick[1], 3) if with_kick[1] else 0.0,
            "off_kick": [round(float(a / c), 3) if c else 0.0 for a, c in zip(*off_kick)],
            "change": _change([on808[b] for b in bars]),
            "pitches": dict(Counter(pitches)), "register": [int(round(lo)), int(round(med)), int(round(hi))],
            "repeat": round(repeats[0] / repeats[1], 3) if repeats[1] else 0.0,
            "root": round(on_root / rooted, 3) if rooted else None,
            "held": round(float(np.median(held)), 3) if held else 0.9,
            "slide_rate": round(len(slides) / max(1, len(pitches) - 1), 3),
            "slide_intervals": dict(Counter(slides)), "glide_time_ms": round(float(glide), 1),
            "velocities": sorted(int(n["velocity"]) for n in notes)}
    return {"tempo": plan["tempo"], "beats_per_bar": int(plan.get("beats_per_bar", 4)),
            "beat_unit": int(plan.get("beat_unit", 4)), "steps_per_bar": spb, "section": section,
            "bars": len(bars), "swing": swing, "drums": drums, "rolls": rolls, "808": b808,
            "chords": _chord_stats(line, bars, plan, melodic)}


def _swing(plan: dict, analysis: dict | None, hat_times: list[float]) -> dict:
    """ratio_16 / ratio_8 (0.5 straight): the analysis' grid.swing, else the
    median place of the plan's off-16th hats (rolls left out) in their 8th."""
    sw = ((analysis or {}).get("grid") or {}).get("swing") or {}
    if sw.get("ratio_16") is not None or sw.get("ratio_8") is not None:
        return {"ratio_16": sw.get("ratio_16"), "ratio_8": sw.get("ratio_8"), "from": "analysis"}
    v = (((np.array(hat_times) - origin(plan)) / 0.5) % 1.0) if hat_times else np.array([])
    off = v[(v >= 0.3) & (v <= 0.8)]
    return {"ratio_16": round(float(np.median(off)), 3) if len(off) >= 8 else None, "ratio_8": None,
            "from": "plan"}


def _chord_stats(line: list[dict], bars: list[int], plan: dict, melodic: dict | None) -> dict | None:
    """The vocabulary (the labels covering VOCAB_SHARE of the stats bars'
    time, at most VOCAB_MAX), the changes between them, and how long one
    lasts (in half bars); the register the melodic stems play in."""
    if not line:
        return None
    bar, o = bar_beats(plan), origin(plan)
    spans = [(o + b * bar, o + (b + 1) * bar) for b in bars]
    ends = [s["start"] for s in line[1:]] + [float(plan["length_beats"])]
    dur: Counter = Counter()
    seq, lengths = [], []
    for s, end in zip(line, ends):
        inside = sum(max(0.0, min(end, b1) - max(s["start"], b0)) for b0, b1 in spans)
        if inside <= 0 or chord_pitches(s["label"]) is None:
            continue
        dur[s["label"]] += inside
        seq.append(s["label"])
        lengths.append(end - s["start"])
    if not dur:
        return None
    total, vocab, covered = sum(dur.values()), [], 0.0
    for label, d in dur.most_common():
        if covered >= VOCAB_SHARE * total or len(vocab) >= VOCAB_MAX:
            break
        vocab.append(label)
        covered += d
    moves: Counter = Counter()
    kept = [x for x in seq if x in vocab]
    for a, b in zip(kept, kept[1:]):
        if a != b:
            moves[f"{a} > {b}"] += 1
    halves = Counter(int(min(4, max(1, round(d / (bar / 2))))) for d in lengths)
    low = CHORD_LOW
    mel = [n["pitch"] for st in ("other", "piano", "guitar") for n in
           ((melodic or {}).get("stems", {}).get(st) or {}).get("notes", [])]
    if mel:
        low = int(np.clip(np.percentile(mel, 25), 48, 66))
    return {"vocabulary": {k: round(dur[k] / total, 3) for k in vocab}, "moves": dict(moves),
            "half_bars": {str(k): v for k, v in sorted(halves.items())}, "low": low}


# ── Drawing a candidate ──

def _swung(step: int, swing: dict) -> float:
    """A 16th's place in its beat (beats) under the swing."""
    beat, k = divmod(step, 4)
    r16, r8 = swing.get("ratio_16"), swing.get("ratio_8")
    if r16 is not None:
        pos = (k // 2) * 0.5 + (r16 * 0.5 if k % 2 else 0.0)
    elif r8 is not None:
        pos = {0: 0.0, 1: r8 / 2, 2: r8, 3: r8 + (1 - r8) / 2}[k]
    else:
        pos = k * STEP
    return beat + pos


def _pattern(rng, p: list[float], bars: int, change: float, motif: int = 2) -> list[set]:
    """Hits per bar: the first `motif` bars drawn from p (each 16th's chance),
    each later bar a copy of the one `motif` before with each 16th drawn
    again at the rate `change`."""
    out = []
    for b in range(bars):
        if b < motif:
            out.append({s for s, x in enumerate(p) if rng.random() < x})
        else:
            prev = out[b - motif]
            out.append({s for s, x in enumerate(p)
                        if (rng.random() < x if rng.random() < change else s in prev)})
    return out


def _choice(rng, weights: dict):
    keys = list(weights)
    w = np.array([float(weights[k]) for k in keys])
    return keys[int(rng.choice(len(keys), p=w / w.sum()))]


def _velocity(rng, pool: list[int], default: int = 100) -> int:
    return int(pool[int(rng.integers(len(pool)))]) if pool else default


def _in_register(pc: int, near: int, lo: int, hi: int) -> int:
    """The pitch of pitch class pc nearest `near`, within lo-hi where one is."""
    options = [p for p in range(lo - 12, hi + 13) if p % 12 == pc]
    inside = [p for p in options if lo <= p <= hi] or options
    return min(inside, key=lambda p: abs(p - near))


def draw(stats: dict, seed: int, number: int, bars: int = 8) -> dict:
    """One candidate's notes (beats from 0): drums, 808 and chords, drawn
    from stats by a generator seeded with (seed, number). Returns
    {"drums", "808", "chords", "slides"} note lists."""
    return _draw(np.random.default_rng([seed, number]), stats, bars)


def _draw(rng, stats: dict, bars: int) -> dict:
    spb, swing = stats["steps_per_bar"], stats["swing"]
    bar = spb * STEP
    drums, kick_steps = [], []
    for role in ROLES:
        d = stats["drums"][role]
        if not d["pads"]:
            kick_steps = kick_steps or [set() for _ in range(bars)]
            continue
        pattern = _pattern(rng, d["p"], bars, d["change"])
        if role == "kick":
            kick_steps = pattern
        pad_at = {int(s): int(p) for s, p in d["pad_at"].items()}
        main = int(next(iter(d["pads"])))
        for b, steps in enumerate(pattern):
            for s in sorted(steps):
                drums.append({"pitch": pad_at.get(s, main), "start_time": round(b * bar + _swung(s, swing), 4),
                              "duration": STEP, "velocity": _velocity(rng, d["velocities"])})
    _rolls(rng, stats, bars, drums)
    chords = _chords(rng, stats, bars)
    notes808, slides = _bass(rng, stats, bars, kick_steps, chords)
    drums.sort(key=lambda n: (n["start_time"], n["pitch"]))
    return {"drums": drums, "808": notes808, "chords": chords, "slides": slides}


def _rolls(rng, stats: dict, bars: int, drums: list[dict]) -> None:
    """Hat rolls: each beat holds one at the reference's rate, from an 8th
    of it, at one of its divisions and lengths; the hats it covers go."""
    r, hat = stats["rolls"], stats["drums"]["hat"]
    if not r["rate"] or not r["divisions"] or not hat["pads"]:
        return
    pad = int(next(iter(hat["pads"])))
    beats = int(bars * stats["steps_per_bar"] * STEP)
    for beat in range(beats):
        if rng.random() >= r["rate"]:
            continue
        div = int(_choice(rng, r["divisions"]))
        length = float(r["beats"][int(rng.integers(len(r["beats"])))]) if r["beats"] else 0.5
        start = beat + 0.5 * int(rng.integers(2))
        end = min(float(beats), start + max(STEP, round(length / STEP) * STEP))
        drums[:] = [n for n in drums if not (n["pitch"] in map(int, hat["pads"]) and start <= n["start_time"] < end)]
        v0 = _velocity(rng, hat["velocities"])
        count = int(round((end - start) * div))
        for k in range(count):
            drums.append({"pitch": pad, "start_time": round(start + k / div, 4), "duration": round(1 / div, 4),
                          "velocity": int(np.clip(v0 * (0.7 + 0.3 * k / max(1, count - 1)), 1, 127))})


def _chords(rng, stats: dict, bars: int) -> list[dict]:
    """A progression over the bars: changes on half bars, each chord lasting
    a drawn number of half bars, the next one drawn from the reference's
    changes from it (its vocabulary when it has none); voiced from `low`."""
    c = stats.get("chords")
    if not c:
        return []
    half = stats["steps_per_bar"] * STEP / 2
    total = bars * 2
    out, at, label = [], 0, None
    while at < total:
        moves = {k.split(" > ")[1]: v for k, v in c["moves"].items() if k.split(" > ")[0] == label}
        label = _choice(rng, moves or {k: v for k, v in c["vocabulary"].items() if k != label}
                        or c["vocabulary"])
        length = min(total - at, int(_choice(rng, c["half_bars"])))
        root, tones = chord_pitches(label)
        base = _in_register(root, c["low"] + 5, c["low"], c["low"] + 11)
        for t in tones:
            out.append({"pitch": base + t, "start_time": round(at * half, 4), "duration": round(length * half, 4),
                        "velocity": 90, "label": label})
        at += length
    return out


def _bass(rng, stats: dict, bars: int, kick_steps: list[set], chords: list[dict]) -> tuple[list[dict], list]:
    """The 808: onsets drawn per 16th (with_kick on a kick's, off_kick
    elsewhere) as a two-bar motif; pitches repeated, on the chord's root or
    drawn from the reference's, in its register; held its share of the gap
    to the next; a slide at slide_rate, by one of its intervals."""
    s8 = stats["808"]
    if not s8["pitches"]:
        return [], []
    spb, swing = stats["steps_per_bar"], stats["swing"]
    bar = spb * STEP
    bps = stats["tempo"] / 60
    p_by_bar = [[s8["with_kick"] if s in kick_steps[b] else s8["off_kick"][s] for s in range(spb)]
                for b in range(bars)]
    onsets = []
    for b in range(bars):
        if b < 2:
            steps = {s for s in range(spb) if rng.random() < p_by_bar[b][s]}
        else:
            prev = {s for _, bb, s in onsets if bb == b - 2}
            steps = {s for s in range(spb)
                     if (rng.random() < p_by_bar[b][s] if rng.random() < s8["change"] else s in prev)}
        onsets += [(b * bar + _swung(s, swing), b, s) for s in sorted(steps)]
    lo, med, hi = s8["register"]
    roots = {n["start_time"]: chord_pitches(n["label"])[0] for n in chords}
    notes, slides, prev = [], [], None
    for t, _, _ in onsets:
        slide = prev is not None and s8["slide_intervals"] and rng.random() < s8["slide_rate"]
        if slide:
            step = int(_choice(rng, {int(k): v for k, v in s8["slide_intervals"].items() if int(k)} or {12: 1}))
            pitch = prev + step if lo - 12 <= prev + step <= hi + 12 else prev - step
        elif prev is not None and rng.random() < s8["repeat"]:
            pitch = prev
        elif roots and s8["root"] is not None and rng.random() < s8["root"]:
            at = max((k for k in roots if k <= t + 1e-6), default=min(roots))
            pitch = _in_register(roots[at], med, lo, hi)
        else:
            pitch = int(_choice(rng, {int(k): v for k, v in s8["pitches"].items()}))
        notes.append({"pitch": int(pitch), "start_time": round(t, 4), "velocity": _velocity(rng, s8["velocities"]),
                      "slide": bool(slide)})
        prev = int(pitch)
    end = bars * bar
    for k, n in enumerate(notes):
        nxt = notes[k + 1] if k + 1 < len(notes) else None
        gap = (nxt["start_time"] if nxt else end) - n["start_time"]
        if nxt and nxt["slide"]:
            n["duration"] = round(gap + LEGATO_S * bps, 4)
            slides.append({"start_time": nxt["start_time"], "from_midi": n["pitch"], "to_midi": nxt["pitch"]})
        else:
            n["duration"] = round(max(0.05, min(gap - 0.001, s8["held"] * gap)), 4)
    for n in notes:
        n.pop("slide")
    return notes, slides


# ── Not a copy (docs/09) ──

def bar_events(drums: list[dict], notes808: list[dict], chords: list[dict], roles: dict[int, str],
               spb: int, start: float = 0.0) -> dict[int, set]:
    """Per bar from `start` (beats): the kick and snare 16ths, the 808's
    (16th, pitch) and the chords' (16th, label) where they change."""
    out: dict[int, set] = defaultdict(set)

    def at(t):
        q = int(round((t - start) / STEP))
        return q // spb, q % spb

    for n in drums:
        role = roles.get(int(n["pitch"]))
        if role in ("kick", "snare"):
            b, s = at(n["start_time"])
            out[b].add((role, s))
    for n in notes808:
        b, s = at(n["start_time"])
        out[b].add(("808", s, int(n["pitch"])))
    for c in chords:
        b, s = at(c["start_time"])
        out[b].add(("chord", s, c["label"]))
    return out


def copy_score(candidate: dict[int, set], reference: dict[int, set]) -> dict:
    """Each candidate bar against the reference bar most like it (F1 of
    their events, bar_events): the mean and the highest. 1 is a bar played
    back as it is."""
    refs = [r for r in reference.values() if r]
    per = []
    for b in sorted(candidate):
        c = candidate[b]
        if not c:
            continue
        per.append(max((2 * len(c & r) / (len(c) + len(r)) for r in refs), default=0.0))
    return {"mean": round(float(np.mean(per)), 3) if per else 0.0, "max": round(max(per), 3) if per else 0.0}


def reference_events(plan: dict, melodic: dict | None, section: str | None = None) -> dict[int, set]:
    bars = set(section_bars(plan, section))
    chords = [{"start_time": s["start"], "label": s["label"]} for s in chord_line(melodic)]
    ev = bar_events(plan["tracks"]["drums"]["notes"], (plan["tracks"].get("808") or {}).get("notes") or [],
                    chords, drum_roles(plan), steps_per_bar(plan), origin(plan))
    return {b: e for b, e in ev.items() if b in bars}


def candidate(stats: dict, plan: dict, reference: dict[int, set], seed: int, number: int, bars: int = 8,
              max_copy: float = MAX_COPY, tries: int = TRIES) -> dict:
    """draw() until the candidate's copy_score mean is at most max_copy (each
    try from the same generator, so the result is the seed's), at most
    `tries` times (tries: how many were drawn; kept: which one is kept),
    the least like the reference when none is under."""
    rng = np.random.default_rng([seed, number])
    roles, best, k = drum_roles(plan), None, 0
    while k < tries:
        k += 1
        notes = _draw(rng, stats, bars)
        score = copy_score(bar_events(notes["drums"], notes["808"], notes["chords"], roles,
                                      stats["steps_per_bar"]), reference)
        if best is None or score["mean"] < best[1]["mean"]:
            best = (notes, score, k)
        if score["mean"] <= max_copy:
            break
    notes, score, kept = best
    return {"number": number, "seed": seed, "tries": k, "kept": kept, "copy": score,
            "under_max_copy": score["mean"] <= max_copy, "notes": notes}


def candidate_plan(plan: dict, stats: dict, notes: dict, bars: int, number: int) -> dict:
    """A plan of the candidate's notes with the reference plan's pads, 808
    track settings and faders (no reference stems), that render.render_plan,
    live.write_midi and live-build read like any plan."""
    t808 = {k: v for k, v in (plan["tracks"].get("808") or {}).items()
            if k not in ("notes", "slides", "voices", "glide_mode", "glide_time_ms")}
    t808["notes"] = notes["808"]
    if notes["slides"]:
        t808.update(voices=1, glide_mode="Glide", glide_time_ms=stats["808"]["glide_time_ms"],
                    slides=[{**s, "kind": "legato", "ms": stats["808"]["glide_time_ms"]} for s in notes["slides"]])
    used = {str(n["pitch"]) for n in notes["drums"]}
    out = {"tempo": plan["tempo"], "beats_per_bar": stats["beats_per_bar"], "beat_unit": stats["beat_unit"],
           "length_beats": float(bars * stats["steps_per_bar"] * STEP),
           "tracks": {"drums": {"pads": {k: v for k, v in plan["tracks"]["drums"]["pads"].items() if k in used},
                                "notes": notes["drums"]}, "808": t808},
           "stems": {}, "locators": [[f"candidate {number}", 0.0]], "levels": dict(plan.get("levels") or {})}
    if notes["chords"]:
        out["tracks"]["melodic"] = {"chords": {"instrument": "Drift", "cents": 0, "notes": [
            {k: n[k] for k in ("pitch", "start_time", "duration", "velocity")} for n in notes["chords"]]}}
    return out


def digest(notes: dict) -> str:
    return hashlib.sha256(json.dumps(notes, sort_keys=True).encode()).hexdigest()[:16]


# ── Sound ──

def render_chords(notes: list[dict], tempo: float, n: int, sr: int = SR) -> np.ndarray:
    """The chords as a plain pad (mono): per note six harmonics falling as
    1/h^1.5, two voices 6 cents apart, 20 ms in, 150 ms out. A sketch of
    the harmony, not a sound."""
    out = np.zeros(n)
    bps = tempo / 60
    for nt in notes:
        a = int(round(nt["start_time"] / bps * sr))
        length = int(round(nt["duration"] / bps * sr))
        rel = int(0.15 * sr)
        m = min(n - a, length + rel)
        if m <= 0:
            continue
        t = np.arange(m) / sr
        f0 = 440.0 * 2 ** ((nt["pitch"] - 69) / 12)
        x = np.zeros(m)
        for cents in (-3.0, 3.0):
            f = f0 * 2 ** (cents / 1200)
            for h in range(1, 7):
                if h * f < sr / 2:
                    x += np.sin(2 * np.pi * h * f * t) / h ** 1.5
        env = np.minimum(1.0, t / 0.02)
        env[length:] *= np.linspace(1.0, 0.0, m - length) if m > length else 1.0
        out[a:a + m] += 0.1 * nt["velocity"] / 127 * x * env
    return out


def duck_gain(n: int, kicks_s: list[float], curve: dict, sr: int = SR) -> np.ndarray:
    """A gain (linear, per sample) that follows `curve` ({"s": seconds after
    the kick, "db": gain}, straight lines in dB between its points, 0 dB
    past the last) from each kick, the lowest where kicks overlap: a
    Compressor keyed from the kick, as a curve."""
    s, g = np.asarray(curve["s"], float), np.minimum(0.0, np.asarray(curve["db"], float))
    shape = np.interp(np.arange(int(s[-1] * sr) + 1) / sr, s, g)
    db = np.zeros(n)
    for t in kicks_s:
        k = int(round(t * sr))
        if not 0 <= k < n:
            continue
        m = min(n - k, len(shape))
        db[k:k + m] = np.minimum(db[k:k + m], shape[:m])
    return 10 ** (db / 20)


def fit_ducking(stems: dict[str, np.ndarray], kick_gap: float = 0.2) -> dict:
    """{"808": curve, "chords": curve} for duck_gain: the reference's own
    kick curves (kick_curve on its bass and other stems, re their median
    level, around the drum stem's kicks), each point at most 0 dB and at
    least DUCK_FLOOR_DB, from the kick (0 dB) to the curve's last point
    (back to 0 dB there). A layer without a curve keeps DUCK_DEFAULT's."""
    from .groove import CURVE_POINTS, band_envelope, kick_curve
    from .symmetric import kick_times
    out = {k: dict(v, source="default") for k, v in DUCK_DEFAULT.items()}
    drums = stems.get("drums")
    if drums is None:
        return out
    kicks = kick_times(drums, kick_gap)
    for layer, (stem, band) in {"808": ("bass", (30, 150)), "chords": ("other", (150, 2000))}.items():
        if stem not in stems or len(kicks) < 4:
            continue
        try:
            c = kick_curve(band_envelope(stems[stem].mean(axis=1), *band), kicks, reference="median")
        except ValueError:
            continue
        db = [float(np.clip(c["curve_db"][f"{p:.2f}"], DUCK_FLOOR_DB, 0.0)) for p in CURVE_POINTS[1:-1]]
        out[layer] = {"s": [0.0, *CURVE_POINTS[1:]], "db": [0.0, *db, 0.0], "source": "reference"}
    return out


def kick_seconds(plan: dict) -> list[float]:
    roles, bps = drum_roles(plan), plan["tempo"] / 60
    return sorted(n["start_time"] / bps for n in plan["tracks"]["drums"]["notes"]
                  if roles.get(int(n["pitch"])) == "kick")


def render_candidate(cplan: dict, out_dir: Path, root: str | Path | None, ducking: dict | None,
                     chords_db: float = -6.0, name: str = "mix") -> dict[str, np.ndarray]:
    """render_plan's drums and 808, the chords (render_chords) chords_db LUFS
    over the drums (the reference's other stem's over its drums, where
    there are stems), the 808 and the chords ducked under
    the kicks (ducking None: not); writes chords.wav, 808-ducked.wav and
    the mix (name.wav), returns
    the stems as compare names them (drums, bass, other), stereo."""
    import soundfile as sf

    from .loudness import integrated_lufs
    from .render import render_plan
    m = render_plan(cplan, out_dir / "render", root=root,  # the bars' length, as the reference's span
                    n_samples=int(round(cplan["length_beats"] / (cplan["tempo"] / 60) * SR)))
    n = m["n_samples"]

    def load(name):
        if name not in m["tracks"] and name != "drums":
            return np.zeros(n)
        x, _ = sf.read(out_dir / "render" / (m["drums"] if name == "drums" else m["tracks"][name]))
        return np.pad(x, (0, max(0, n - len(x))))[:n]

    drums, b808 = load("drums"), load("808")
    chords = render_chords((cplan["tracks"].get("melodic") or {}).get("chords", {}).get("notes", []),
                           cplan["tempo"], n)
    if np.any(chords):
        target = integrated_lufs(drums, SR) + chords_db
        lv = integrated_lufs(chords, SR)
        if np.isfinite(lv) and np.isfinite(target):
            chords *= 10 ** ((target - lv) / 20)
    if ducking:
        kicks = kick_seconds(cplan)
        b808 = b808 * duck_gain(n, kicks, ducking["808"])
        chords = chords * duck_gain(n, kicks, ducking["chords"])
    stems = {"drums": drums, "bass": b808, "other": chords}
    mix = drums + b808 + chords
    sf.write(out_dir / "chords.wav", chords.astype(np.float32), SR, subtype="FLOAT")
    sf.write(out_dir / "808-ducked.wav", b808.astype(np.float32), SR, subtype="FLOAT")
    sf.write(out_dir / f"{name}.wav", mix.astype(np.float32), SR, subtype="FLOAT")
    return {k: np.stack([x, x], 1) for k, x in stems.items()}


# ── The whole step ──

def _load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def reference_span(plan: dict, start_bar: int, bars: int) -> tuple[float, float]:
    """Seconds of `bars` bars from bar start_bar (from 0) of the plan."""
    bps, bar = plan["tempo"] / 60, bar_beats(plan)
    a = (origin(plan) + start_bar * bar) / bps
    return a, a + bars * bar / bps


def summary(rows: dict) -> dict:
    """What a listener weighs a candidate by: the 808's and the chords' kick
    curves against the reference's (mean dB apart), each drum role's level
    against the reference's (dB, re the loudest), and how many measures
    differ by more than the reference's own halves do."""
    out = {"ducking_808_db": (rows.get("ducking.808.curve_distance_db") or {}).get("diff"),
           "ducking_chords_db": (rows.get("ducking.chords.curve_distance_db") or {}).get("diff"),
           "drum_roles_db": {k.split(".")[-1]: v["diff"] for k, v in rows.items() if k.startswith("drum_roles_db.")}}
    roles = [abs(v) for v in out["drum_roles_db"].values()]
    out["balance_db"] = round(float(np.mean(roles)), 2) if roles else None
    flagged = [v for v in rows.values() if "beyond_spread" in v]
    out["beyond_spread"] = f"{sum(v['beyond_spread'] for v in flagged)}/{len(flagged)}"
    return out


def run(work: str | Path, out_dir: str | Path, n: int = 4, seed: int = 1, bars: int = 8,
        section: str | None = None, reference_bar: int | None = None, stems: str | Path | None = None,
        question: str | None = None, max_copy: float = MAX_COPY, duck: bool = True, measure: bool = True,
        kick_gap: float = 0.2, separate_model: str | None = None, lufs: float = -14.0, log=print) -> dict:
    """n candidates from work's plan.json (with analysis.json's swing and
    melodic.json's chords where there): out/NN/ (plan.json, midi/, render/,
    chords.wav, 808-ducked.wav, candidate-NN.wav: the mix, compare.json),
    a listening round in out/round/
    and out/candidates.json with the statistics, each candidate's seed,
    tries, copy score, digest and compare summary. The reference stems
    (plan's stems folder or `stems`) give the round's reference bars, the
    ducking fit and the comparison; without them, neither."""
    from .live import write_midi
    work, out = Path(work), Path(out_dir)
    plan = _load(work / "plan.json")
    if plan is None:
        raise FileNotFoundError(f"No plan.json in {work} (run refdissect plan first)")
    analysis, melodic = _load(work / "analysis.json"), _load(work / "melodic.json")
    stats = style_stats(plan, analysis, melodic, section)
    reference = reference_events(plan, melodic, section)
    start = reference_bar if reference_bar is not None else section_bars(plan, section)[0]
    span = reference_span(plan, start, bars)
    stem_dir = Path(stems) if stems else (Path(next(iter(plan.get("stems", {}).values()))["path"]).parent
                                          if plan.get("stems") else None)
    ref = ref_full = None
    if stem_dir is not None and (stem_dir / "drums.wav").is_file():
        from .symmetric import load_stems
        ref = load_stems(stem_dir, span)
        if measure:
            sb = section_bars(plan, section)
            ref_full = load_stems(stem_dir, (reference_span(plan, sb[0], 1)[0],
                                             reference_span(plan, sb[-1], 1)[1]))
    out.mkdir(parents=True, exist_ok=True)
    ducking = (fit_ducking(ref, kick_gap) if ref is not None else fit_ducking({})) if duck else None
    chords_db = -6.0
    if ref is not None and "other" in ref and stats["chords"]:
        from .loudness import integrated_lufs
        chords_db = integrated_lufs(ref["other"], SR) - integrated_lufs(ref["drums"], SR)  # re the drums
    spread = ref_meas = None
    if measure and ref is not None:
        from .symmetric import baseline
        from .symmetric import measure as measure_stems
        log("measuring the reference")
        spread, ref_meas = baseline(ref_full, kick_gap), measure_stems(ref, kick_gap)
    if ref is not None:
        import soundfile as sf
        mix = sum(x for k, x in ref.items() if k != "vocals")
        sf.write(out / "reference.wav", mix.astype(np.float32), SR, subtype="FLOAT")
    rows = []
    for k in range(1, n + 1):
        c = candidate(stats, plan, reference, seed, k, bars, max_copy)
        cdir = out / f"{k:02d}"
        cplan = candidate_plan(plan, stats, c["notes"], bars, k)
        cdir.mkdir(parents=True, exist_ok=True)
        (cdir / "plan.json").write_text(json.dumps(cplan, indent=1), encoding="utf-8")
        write_midi(cplan, cdir / "midi")
        cstems = render_candidate(cplan, cdir, work, ducking, chords_db, f"candidate-{k:02d}")
        row = {k2: c[k2] for k2 in ("number", "seed", "tries", "kept", "copy", "under_max_copy")}
        row.update(digest=digest(c["notes"]), dir=str(cdir), mix=str(cdir / f"candidate-{k:02d}.wav"),
                   notes={"drums": len(c["notes"]["drums"]), "808": len(c["notes"]["808"]),
                          "slides": len(c["notes"]["slides"]),
                          "chords": list({x["start_time"]: x["label"] for x in c["notes"]["chords"]}.values())})
        if ref_meas is not None:
            from .symmetric import compare
            from .symmetric import measure as measure_stems
            if separate_model:
                from .separate import separate
                from .symmetric import load_stems
                separate(row["mix"], cdir / "stems", separate_model)
                cstems = load_stems(cdir / "stems")
            # the candidate as loud as the reference where their drums are: the levels then compare as balance
            from .loudness import integrated_lufs
            shift = integrated_lufs(ref["drums"], SR) - integrated_lufs(cstems["drums"], SR)
            g = 10 ** (shift / 20) if np.isfinite(shift) else 1.0
            crows = compare(ref_meas, measure_stems({k: x * g for k, x in cstems.items()}, kick_gap), spread)
            (cdir / "compare.json").write_text(json.dumps({"rows": crows, "reference_spread": spread}, indent=1),
                                               encoding="utf-8")
            row["compare"] = summary(crows)
        rows.append(row)
        log(_line(row))
    manifest = {"work": str(work), "seed": seed, "n": n, "bars": bars, "section": section,
                "reference_span_s": [round(x, 3) for x in span], "max_copy": max_copy,
                "ducking": ducking, "separated": separate_model, "stats": stats, "candidates": rows}
    if ref is not None:
        from .rounds import listening_round
        q = question or f"Which {bars}-bar sketches sound like the reference{f' {section}' if section else ''}?"
        manifest["round"] = str(out / "round")
        listening_round(out / "reference.wav", [r["mix"] for r in rows], out / "round", q, target_lufs=lufs)
    (out / "candidates.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    return manifest


def _line(row: dict) -> str:
    text = f"{row['number']:2d}: copy {row['copy']['mean']:.2f} (max {row['copy']['max']:.2f}, {row['tries']} tries)"
    if "compare" in row:
        c = row["compare"]

        def f(x):
            return "-" if x is None else f"{x:.1f}"
        text += (f"; ducking 808 {f(c['ducking_808_db'])} dB, chords {f(c['ducking_chords_db'])} dB; "
                 f"drum roles {f(c['balance_db'])} dB; beyond spread {c['beyond_spread']}")
    return text
