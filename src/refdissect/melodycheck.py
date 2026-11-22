"""The melody check: a stem's notes on a plain tone, to confirm by ear.

Before a sound is chosen for a layer its notes have to be right, and only
the ear says so (docs/13, section 6: the first study's sound tests ran on
notes nobody had confirmed, read from a stem of several sounds; read from
one layer alone, the notes were confirmed). For some bars of a stem, or of
one layer of it (`audio`), this writes:

- an A/B file: the stem, a beep, then the notes on a plain tone;
- an overlay: the stem in the left channel, the tone in the right;
- a MIDI file of the notes, to correct in a piano roll.

A corrected file (MIDI, or notes as JSON in beats) is read back the same way,
so the correction is heard against the stem too, and it is compared with the
reading: how many notes were kept, moved, repitched, added and removed.
The check confirms pitches and starts. A plain tone says little about how
long a note rings or how loud it is: those are taken from the layer.

Notes here are in beats (quarter notes) from the first bar asked for:
{"pitch": MIDI number, "start": beats, "length": beats}.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from .grid import Grid
from .listening import ab_file

PPQ = 480


def bar_span(grid: Grid, first: int, last: int) -> tuple[float, float]:
    """Seconds from the start of bar `first` to the end of bar `last` (bars count from 1)."""
    if first < 1 or last < first:
        raise ValueError(f"bars {first}-{last}: bars count from 1, and the last is not before the first")
    return grid.phase + (first - 1) * grid.bar, grid.phase + last * grid.bar


def notes_in_span(notes: list[dict], grid: Grid, t0: float, t1: float) -> list[dict]:
    """melodic.json's notes (t, end in seconds, pitch) between t0 and t1, in
    beats from t0. A note that starts before t0 and still sounds a sixteenth
    into the span is kept from the span's start, marked "tied"."""
    out = []
    for n in notes:
        if t0 <= n["t"] < t1:
            out.append({"pitch": int(n["pitch"]), "start": (n["t"] - t0) / grid.beat,
                        "length": (min(n["end"], t1) - n["t"]) / grid.beat})
        elif n["t"] < t0 and n["end"] >= t0 + grid.step:
            out.append({"pitch": int(n["pitch"]), "start": 0.0, "length": (min(n["end"], t1) - t0) / grid.beat,
                        "tied": True})
    return _sorted(out)


def _sorted(notes: list[dict]) -> list[dict]:
    return sorted(({**n, "start": round(n["start"], 4), "length": round(n["length"], 4)} for n in notes),
                  key=lambda n: (n["start"], n["pitch"]))


def tone(midi: float, seconds: float, sr: int) -> np.ndarray:
    """A plain tone: a sine with a quarter of its octave on top (so that low
    notes are heard on small speakers), 5 ms in, 20 ms out, no decay."""
    n = max(1, int(round(seconds * sr)))
    t = np.arange(n) / sr
    f = 440.0 * 2 ** ((midi - 69) / 12)
    y = np.sin(2 * np.pi * f * t) + 0.25 * np.sin(2 * np.pi * 2 * f * t)
    a, r = min(n, int(0.005 * sr)), min(n, int(0.02 * sr))
    if a:
        y[:a] *= np.linspace(0, 1, a, endpoint=False)
    if r:
        y[-r:] *= np.linspace(1, 0, r)
    return y


def render(notes: list[dict], bpm: float, seconds: float, sr: int, cents: float = 0.0) -> np.ndarray:
    """The notes on the plain tone, mono, `seconds` long; cents moves every
    note (the stem's tuning, so the tone does not beat against it)."""
    beat = 60.0 / bpm
    out = np.zeros(int(round(seconds * sr)))
    for n in notes:
        a = int(round(n["start"] * beat * sr))
        if a >= len(out):
            continue
        y = tone(n["pitch"] + cents / 100, max(n["length"] * beat, 0.03), sr)
        m = min(len(y), len(out) - a)
        if m < len(y):  # cut at the span's end: close it without a click
            r = min(m, int(0.005 * sr))
            y = y[:m].copy()
            y[m - r:] *= np.linspace(1, 0, r)
        out[a:a + m] += y[:m]
    return out


def write_midi(notes: list[dict], bpm: float, path: str | Path, ppq: int = PPQ,
               meter: tuple[int, int] = (4, 4)) -> Path:
    """The notes as a one-track MIDI file at the grid's tempo and meter, bar 1
    of the file being the first bar asked for. A note still sounding when
    its pitch starts again ends there: a MIDI file cannot tell two sounding
    notes of one pitch apart (their note-offs would pair the wrong way)."""
    import mido
    mid = mido.MidiFile(ticks_per_beat=ppq)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    track.append(mido.MetaMessage("time_signature", numerator=meter[0], denominator=meter[1], time=0))
    spans = sorted((int(n["pitch"]), int(round(n["start"] * ppq)),
                    int(round((n["start"] + n["length"]) * ppq))) for n in notes)
    events = []
    for k, (pitch, on, off) in enumerate(spans):
        off = max(on + 1, off)
        if k + 1 < len(spans) and spans[k + 1][0] == pitch and on < spans[k + 1][1]:
            off = min(off, spans[k + 1][1])
        events += [(on, 1, pitch), (off, 0, pitch)]
    now = 0
    for tick, kind, pitch in sorted(events):  # at one tick, note-offs first
        track.append(mido.Message("note_on" if kind else "note_off", note=int(pitch), velocity=100 if kind else 0,
                                  time=tick - now))
        now = tick
    path = Path(path)
    mid.save(str(path))
    return path


def read_midi(path: str | Path) -> list[dict]:
    """Every track's notes in beats from the file's start (the tempo in the
    file does not matter: ticks over ticks-per-beat)."""
    import mido
    mid = mido.MidiFile(str(path))
    notes = []
    for track in mid.tracks:
        tick, sounding = 0, {}
        for msg in track:
            tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                sounding.setdefault(msg.note, []).append(tick)
            elif msg.type in ("note_off", "note_on") and sounding.get(msg.note):
                on = sounding[msg.note].pop(0)
                notes.append({"pitch": int(msg.note), "start": on / mid.ticks_per_beat,
                              "length": (tick - on) / mid.ticks_per_beat})
    return _sorted(notes)


def read_notes(path: str | Path) -> list[dict]:
    """A corrected file: MIDI (.mid, .midi), or JSON: a list of notes, or
    {"notes": [...]}, each with pitch and start / length in beats (a Live
    clip's start_time / duration are read too)."""
    path = Path(path)
    if path.suffix.lower() in (".mid", ".midi"):
        return read_midi(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["notes"] if isinstance(data, dict) else data
    notes = []
    for n in rows:
        start = n["start"] if "start" in n else n["start_time"]
        length = n["length"] if "length" in n else n["duration"]
        notes.append({"pitch": int(n["pitch"]), "start": float(start), "length": float(length)})
    return _sorted(notes)


def diff(reading: list[dict], corrected: list[dict], tol: float = 0.125, moved_within: float = 1.0) -> dict:
    """What a correction changed. A note is kept when the corrected file has
    its pitch within `tol` beats (a 32nd) of its start; of the rest, one is
    repitched when a corrected note starts within tol at another pitch, and
    moved when its pitch starts again within `moved_within` beats; what is
    left was removed (reading) or added (corrected). Each note pairs once.
    precision and recall are the reading's against the corrected notes."""
    left_r, left_c = list(range(len(reading))), list(range(len(corrected)))

    def pair(ok) -> int:
        cands = sorted((abs(reading[i]["start"] - corrected[j]["start"]), i, j)
                       for i in left_r for j in left_c if ok(reading[i], corrected[j]))
        n = 0
        for _, i, j in cands:
            if i in left_r and j in left_c:
                left_r.remove(i)
                left_c.remove(j)
                n += 1
        return n

    kept = pair(lambda a, b: a["pitch"] == b["pitch"] and abs(a["start"] - b["start"]) <= tol)
    repitched = pair(lambda a, b: abs(a["start"] - b["start"]) <= tol)
    moved = pair(lambda a, b: a["pitch"] == b["pitch"] and abs(a["start"] - b["start"]) <= moved_within)
    return {"reading": len(reading), "corrected": len(corrected), "kept": kept, "repitched": repitched,
            "moved": moved, "removed": len(left_r), "added": len(left_c),
            "precision": round(kept / len(reading), 3) if reading else None,
            "recall": round(kept / len(corrected), 3) if corrected else None}


def _grid(analysis: dict) -> Grid:
    g = analysis["grid"]
    return Grid(bpm=g["bpm"], phase=g["phase"], beats_per_bar=g.get("beats_per_bar", 4),
                beat_unit=g.get("beat_unit", 4))


def check(work: str | Path, stem: str, bars: tuple[int, int], out_dir: str | Path | None = None,
          notes_file: str | Path | None = None, bars_per_file: int | None = None,
          target_lufs: float = -14.0, audio: str | Path | None = None) -> dict:
    """The melody check of bars `bars` (from 1, both included) of a stem of a
    work folder (analysis.json, melodic.json and the stems analysis.json
    names). Without notes_file the notes are melodic.json's; with it, the
    corrected file's (beats from the first bar), compared with the reading.
    `audio` is a file to check against instead of the stem, on the song's
    clock: one layer of the stem, cut by another separation of the same
    file. With `audio` there is no reading to compare with (melodic.json's
    notes are the stem's, not the layer's), and a layer melodic.json does
    not know needs notes_file. Writes into out_dir (WORK/melody-check by default): melody_AB.wav (one per
    `bars_per_file` bars when given), melody_overlay.wav, melody.mid and
    melody-check.json, which it returns."""
    work = Path(work)
    analysis = json.loads((work / "analysis.json").read_text(encoding="utf-8"))
    known = json.loads((work / "melodic.json").read_text(encoding="utf-8"))["stems"] \
        if (work / "melodic.json").is_file() else {}
    entry = known.get(stem)
    if entry is None and (audio is None or notes_file is None):
        raise ValueError(f"melodic.json has no stem {stem!r} (it has {', '.join(sorted(known)) or 'none'}): "
                         "a layer it does not know needs both audio and notes")
    grid = _grid(analysis)
    first, last = bars
    t0, t1 = bar_span(grid, first, last)
    path = Path(audio) if audio else Path(analysis["source"]["stems"]) / f"{stem}.wav"
    info = sf.info(str(path))
    sr = info.samplerate
    if t0 < 0 or t1 > info.frames / sr + grid.step:
        raise ValueError(f"bars {first}-{last} are {t0:.2f}-{t1:.2f} s, outside the stem's {info.frames / sr:.2f} s")
    ref, _ = sf.read(str(path), start=int(round(t0 * sr)), stop=int(round(t1 * sr)), always_2d=True)
    cents = float(entry.get("tuning_cents", 0)) if entry else 0.0
    reading = notes_in_span(entry["notes"], grid, t0, t1) if entry and not audio else None
    bar_beats = grid.bar / grid.beat
    if bars_per_file is not None and bars_per_file < 1:
        raise ValueError(f"bars_per_file is {bars_per_file}: one bar or more a file")
    if notes_file is None:
        notes = notes_in_span(entry["notes"], grid, t0, t1)
        source, changed = "melodic.json", None
    else:
        # a note that starts before the first bar is kept from the bar's start, if it still sounds there
        notes = _sorted({**n, "start": max(0.0, n["start"]), "length": n["length"] + min(0.0, n["start"])}
                        for n in read_notes(notes_file)
                        if n["start"] + n["length"] > 0 and n["start"] < (last - first + 1) * bar_beats)
        source, changed = str(notes_file), diff(reading, notes) if reading is not None else None
    out = Path(out_dir) if out_dir else work / "melody-check"
    out.mkdir(parents=True, exist_ok=True)
    plain = render(notes, grid.bpm, len(ref) / sr, sr, cents)
    files = []
    step = bars_per_file or (last - first + 1)
    for b in range(first, last + 1, step):
        e = min(b + step - 1, last)
        a_, b_ = int(round((b - first) * grid.bar * sr)), int(round((e - first + 1) * grid.bar * sr))
        name = "melody_AB.wav" if step >= last - first + 1 else f"melody_AB_bars{b}-{e}.wav"
        part = plain[a_:b_]
        lo, hi = (b - first) * bar_beats, (e - first + 1) * bar_beats
        row = {"bars": [b, e], "notes": sum(1 for n in notes if lo <= n["start"] < hi)}
        if np.any(part):
            row.update(ab_file(ref[a_:b_], part, sr, out / name, target_lufs=target_lufs))
        else:
            row["file"] = None  # no notes in these bars: nothing to play after the beep
        files.append(row)
    mono = ref.mean(1)
    overlay = np.stack([mono / (np.max(np.abs(mono)) or 1.0), plain / (np.max(np.abs(plain)) or 1.0)], 1) * 0.9
    sf.write(str(out / "melody_overlay.wav"), overlay.astype(np.float32), sr, subtype="PCM_16")
    write_midi(notes, grid.bpm, out / "melody.mid", meter=(grid.beats_per_bar, grid.beat_unit))
    per_bar = [sum(1 for n in notes if k * bar_beats <= n["start"] < (k + 1) * bar_beats)
               for k in range(last - first + 1)]
    manifest = {"stem": stem, "audio": str(path), "bars": [first, last], "span_s": [round(t0, 4), round(t1, 4)],
                "bpm": grid.bpm, "beats_per_bar": bar_beats, "tuning_cents": cents, "notes_from": source,
                "notes": notes, "notes_per_bar": per_bar, "ab": files,
                "overlay": str(out / "melody_overlay.wav"), "midi": str(out / "melody.mid"),
                "against_reading": changed,
                "how": "A/B: the stem, a beep, then the notes on a plain tone. Overlay: the stem left, the tone right. "
                       "Correct melody.mid in a piano roll and run again with --notes; only the ear confirms."}
    (out / "melody-check.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    return manifest
