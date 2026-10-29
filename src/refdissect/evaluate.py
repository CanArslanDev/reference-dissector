"""Scoring a rebuild against the answer, from events files (schema
refdissect.events/1): every drum hit, 808 note, melodic note and chord, the
grid, and the sounds.

A truth comes from a scenario. An estimate comes from an analysis
(estimate_from_analysis: what the pipeline heard) or from a plan
(estimate_from_plan: what Live will play). Both carry the same fields, so one
scorer reads both, and a step that loses a sound or a few milliseconds shows
up as the difference between the two.

Every time is where a sound attacks (audio.attack_time) on both sides, so no
offset is removed: the strict score. calibrate's rule (one constant offset
removed per sound) is reported beside it for continuity. That rule is why v1
never saw that its plans put the calibration beats' snares 21-24 ms early
(11 of 12 plans, from htdemucs and htdemucs_ft stems): it scores them F1
1.00, the strict F1 at 20 ms is 0.
"""

from __future__ import annotations

import hashlib
import json
import numbers
from pathlib import Path

import mir_eval
import numpy as np

from . import __version__
from . import scenarios as S
from .audio import ATTACK_BAND, PRE, SR, attack_time, band, read, rms_db, samples, transpose
from .calibrate import _f1
from .grid import BAR_SLACK
from .render import (
    RETRIGGER_S,
    as_played,
    chain_semitones,
    chains_808,
    fader_db,
    glide_s,
    late_808,
    pad_hits,
    pad_settings,
    pitch_808,
    read_at,
    resolve,
    settings_808,
    velocity_gain,
)

SCHEMA = "refdissect.events/1"
ROLES = ("kick", "snare", "clap", "rim", "hat", "open_hat", "perc", "tom", "crash", "ride", "808", "melodic", "vocal")
DRUM_ROLES = ROLES[:10]
WINDOWS = {"20": 0.02, "50": 0.05}  # onset tolerances (s): the strict one, and the usual MIREX one
OCTAVES = {"x1": 1.0, "x2": 2.0, "x1/2": 0.5, "x3/2": 1.5, "x2/3": 2 / 3}
CHORD_RULES = ("root", "majmin", "sevenths", "tetrads", "mirex")
NO_CHORD = "N"


# ── Events files ──

_FIELDS = {"drums": ("sound", "t", "t_attack"), "bass808": ("sound", "t", "t_attack", "end", "midi"),
           "notes": ("layer", "t", "end", "midi"), "chords": ("t", "end", "label")}
_NUMBERS = ("t", "t_attack", "end", "midi", "gain", "cents")


def _number(v) -> bool:
    return isinstance(v, numbers.Real) and not isinstance(v, bool) and bool(np.isfinite(v))


def _integer(v) -> bool:
    return isinstance(v, numbers.Integral) and not isinstance(v, bool)


def _event_problems(part: str, k: int, e: dict, sounds: dict | None) -> list[str]:
    where = f"{part}[{k}]"
    bad = [f"{where}: no {f}" for f in _FIELDS[part] if f not in e]
    bad += [f"{where}: {f} is not a number" for f in _NUMBERS if f in e and not _number(e[f])]
    if bad:
        return bad
    if "end" in e and e["end"] < e["t"]:
        bad.append(f"{where}: ends before it starts")
    if part == "drums" and e.get("role", "perc") not in DRUM_ROLES:
        bad.append(f"{where}: role {e.get('role')!r} is not a drum role")
    v = e.get("velocity")
    if v is not None and not (_integer(v) and 1 <= v <= 127):
        bad.append(f"{where}: velocity {v!r} is not 1..127")
    if part == "notes" and not _integer(e["midi"]):
        bad.append(f"{where}: midi is not an integer (cents carry the rest)")
    g = e.get("glide")
    if part == "bass808" and g is not None and not (all(_number(g.get(f)) for f in ("to_midi", "t", "ms"))
                                                    and g.get("shape") in ("exp", "lin")):
        bad.append(f"{where}: glide needs to_midi, t, ms and shape exp or lin")
    if part == "chords":
        try:
            mir_eval.chord.validate_chord_label(str(e["label"]))
        except mir_eval.chord.InvalidChordException as x:
            bad.append(f"{where}: {x}")
    if sounds is not None and part in ("drums", "bass808") and e["sound"] not in sounds:
        bad.append(f"{where}: sound {e['sound']!r} is not in sounds")
    return bad


def _time_problems(tm: dict) -> list[str]:
    bad = []
    if not (_number(tm.get("bpm")) and tm["bpm"] > 0):
        bad.append("time: bpm is not a positive number")
    bpb = tm.get("beats_per_bar", 4)
    if not _integer(bpb) or bpb < 1:
        bad.append("time: beats_per_bar is not a positive integer")
    if tm.get("beat_unit", 4) not in (4, 8):
        bad.append("time: beat_unit is not 4 or 8")
    for key in ("beats_s", "downbeats_s"):
        v = tm.get(key) or []
        if not all(_number(x) for x in v) or np.any(np.diff(v) < 0):
            bad.append(f"time: {key} is not ascending seconds")
    sw = tm.get("swing")
    if sw is not None and not (_number(sw.get("ratio")) and sw.get("level") in (8, 16)):
        bad.append("time: swing needs a ratio and level 8 or 16")
    return bad


def validate(ev: dict) -> dict:
    """ev itself when it follows the schema, else ValueError naming the
    problems. Only schema and kind are required (a scorer skips what a file
    does not carry); whatever is there must have the schema's types, and a
    truth's events must name sounds it lists."""
    bad = []
    if ev.get("schema") != SCHEMA:
        bad.append(f"schema {ev.get('schema')!r} is not {SCHEMA!r}")
    if ev.get("kind") not in ("truth", "estimate"):
        bad.append(f"kind {ev.get('kind')!r} is not 'truth' or 'estimate'")
    sounds = ev.get("sounds")
    for sid, s in (sounds or {}).items():
        if s.get("role") not in ROLES:
            bad.append(f"sounds[{sid!r}]: role {s.get('role')!r} is not one of {ROLES}")
    known = sounds if ev.get("kind") == "truth" and sounds is not None else None
    for part in _FIELDS:
        for k, e in enumerate(ev.get(part) or []):
            bad += _event_problems(part, k, e, known)
    if ev.get("time"):
        bad += _time_problems(ev["time"])
    if bad:
        more = f" (and {len(bad) - 10} more)" if len(bad) > 10 else ""
        raise ValueError(f"Not a {SCHEMA} file: " + "; ".join(bad[:10]) + more)
    return ev


def load_events(path: str | Path) -> dict:
    """An events file, validated. Its audio paths are relative to its folder."""
    return validate(json.loads(Path(path).read_text(encoding="utf-8")))


def save_events(ev: dict, path: str | Path) -> Path:
    """Write a validated events dict. Note-level: keep it local (docs/09)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(validate(ev), indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def _load(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).is_file() else ""


def _events(scenario: str, switches: dict, **parts) -> dict:
    return {"schema": SCHEMA, "kind": "estimate", "scenario": scenario, "seed": 0,
            "generator": {"module": __name__, "switches": switches, "refdissect": __version__}, **parts}


def _drum_role(role: str | None) -> str:
    """A matched or guessed role as an events role: anything else plays as
    perc, as make_plan does."""
    return role if role in DRUM_ROLES else "perc"


def _map_offsets(lines: np.ndarray, bpm: float, first: float, tempo_map) -> np.ndarray:
    """Where a tempo map (grid.drift: the grid's quarter-note lines from its
    first, moved where the beats drift) puts each line, less the line;
    interpolated between its beats and held beyond them."""
    if not tempo_map:
        return np.zeros(len(lines))
    m, q = np.asarray(tempo_map, dtype=float), 60 / bpm
    own = first + (np.round((m[0] - first) / q) + np.arange(len(m))) * q  # consecutive lines, as drift lists them
    return np.interp(lines, own, m - own)


def _grid_time(bpm: float, first: float, beats_per_bar: int, end: float, beat_unit: int = 4,
               swing: dict | None = None, tempo_map=None, flags=()) -> dict:
    """A grid as an events time block: beats of the meter's unit (bpm counts
    quarter notes, so an eighth in 6/8) on the lines through `first`, the
    first bar line, up to `end`, a bar line every beats_per_bar of them. The
    beats start at the earliest line at or after -BAR_SLACK: a pickup before
    the first bar line is beats too (as Grid.bar_of counts the bars). With a
    tempo map, every line moves by its offset there."""
    beat = 60 / bpm * 4 / beat_unit
    k = np.arange(np.ceil((-BAR_SLACK - first) / beat), max(np.ceil((end - first) / beat), 1))
    lines = first + k * beat
    lines = lines + _map_offsets(lines, bpm, first, tempo_map)
    down = lines[k % beats_per_bar == 0]
    return {"bpm": bpm, "beats_per_bar": beats_per_bar, "beat_unit": beat_unit,
            "downbeat_s": round(float(lines[k == 0][0]) if (k == 0).any() else first, 5),
            "beats_s": np.round(lines, 5).tolist(), "downbeats_s": np.round(down, 5).tolist(), "swing": swing,
            "tempo_map": tempo_map or None, "flags": list(flags)}


def _analysis_time(g: dict, end: float) -> dict:
    """The analysis grid (grid.estimate) as an events time block: its tempo,
    meter and first bar line (phase), its tempo map where the beats drift,
    the swing from the hats' attacks at the finer level it read (ratio_16 if
    it has one, else ratio_8: grid.swing leaves ratio_16 out when the 8ths
    swing), and "downbeat_unsure" when the bar line is a guess: no tracker
    (phase is fit_grid's first grid beat) or a vote it calls ambiguous."""
    sw = g.get("swing") or {}
    swing = ({"ratio": sw["ratio_16"], "level": 16} if sw.get("ratio_16") is not None
             else {"ratio": sw["ratio_8"], "level": 8} if sw.get("ratio_8") is not None else None)
    down = g.get("downbeat")
    flags = ["downbeat_unsure"] if not down or down.get("ambiguous") else []
    return _grid_time(g["bpm"], g["phase"], int(g.get("beats_per_bar", 4)), end, int(g.get("beat_unit", 4)), swing,
                      g.get("tempo_map"), flags)


def _with_kick(bass: list[dict], drums: list[dict], window: float = 0.02) -> None:
    """Mark the 808 notes that attack within `window` of a kick's attack."""
    kicks = np.array([e["t_attack"] for e in drums if e["role"] == "kick"])
    for n in bass:
        n["with_kick"] = bool(len(kicks) and np.min(np.abs(kicks - n["t_attack"])) <= window)


# ── Estimates ──

def _note(layer: str, t: float, end: float, pitch: float, velocity, flags=()) -> dict:
    """A melodic note of the events schema: a fractional pitch split into midi and cents."""
    midi = int(round(pitch))
    return {"layer": layer, "t": round(float(t), 5), "end": round(float(end), 5), "midi": midi,
            "cents": round((pitch - midi) * 100, 1), "velocity": int(velocity), "flags": list(flags)}


def _melodic(path: Path) -> dict:
    """Notes and chords from melodic.json (pipeline.melodic): each tonal
    stem's notes as a layer named after the stem, sounding at pitch +
    cents/100 (Basic Pitch's semitone on the stem moved onto A440, and the
    stem's tuning), restored ones flagged; the chord segments when it has
    them. {} without the file: the step did not run, so there is nothing to
    score (not nothing heard)."""
    if not path.is_file():
        return {}
    m = _load(path)
    notes = [_note(stem, n["t"], n["end"], n["pitch"] + n.get("cents", 0) / 100, n.get("velocity", 100),
                   ["restored"] if n.get("voted") == "restored" else [])
             for stem, info in m.get("stems", {}).items() for n in info["notes"]]
    out: dict = {"notes": sorted(notes, key=lambda n: (n["t"], n["layer"], n["midi"]))}
    if m.get("chords"):
        out["chords"] = [{"t": float(c["start"]), "end": float(c["end"]), "label": str(c["label"])}
                         for c in m["chords"]["segments"]]
    return out


def _plan_melodic(plan: dict, bps: float) -> dict:
    """The plan's melodic tracks, as notes at pitch + the track's cents; {}
    when it has none. They are built muted, with a placeholder instrument:
    what Live holds, not what it plays (the reference stem plays). A plan
    holds no chords."""
    tracks = plan["tracks"].get("melodic")
    if tracks is None:
        return {}
    return {"notes": sorted((_note(stem, n["start_time"] / bps, (n["start_time"] + n["duration"]) / bps,
                                   n["pitch"] + t.get("cents", 0) / 100, n["velocity"])
                             for stem, t in tracks.items() for n in t["notes"]),
                            key=lambda n: (n["t"], n["layer"], n["midi"]))}


def _lead(info: dict, audio: np.ndarray | None, role: str, semitones: int) -> float:
    """Seconds from a template's hit time t (PRE into its window) to where
    its sound starts: the template's attack_s (discover found it on all its
    hits), else attack_time of the template; for a pitched copy, the
    template transposed that far, as Simpler plays it."""
    if info.get("attack_s") is not None:
        return info["attack_s"] * 2 ** (-semitones / 12) - PRE / SR
    return attack_time(transpose(audio, semitones), role) - PRE / SR if audio is not None else 0.0


def _glide(g: dict | None) -> dict | None:
    """A notes_808 glide as the schema's (from_midi, to_midi, t, ms, shape)."""
    if not g:
        return None
    return {k: g[k] for k in ("from_midi", "to_midi", "t", "ms", "shape") if k in g}


def estimate_from_analysis(work: str | Path, scenario: str | None = None) -> dict:
    """What the analysis heard, as an estimate.

    Drums: every template's hits; sound id = the template's name, role = its
    match's (matches.json) or else its guess. detect() reports a hit at
    t = (peak + PRE) / SR, PRE (5 ms) after the start of the template's
    window, so the hit attacks at t - PRE/SR + the template's attack, unless
    the event carries t_attack. The attack is the template's attack_s, where
    discover() found the sound to start (on all its hits, in its role's
    bands), or else attack_time(template, role). Read by attack_time on the
    averaged template instead, the kicks of 32 suite beats came 15.7 ms late
    (their body), not 0.9 ms (drums F1@20 0.810 -> 0.717). In the role's
    band the attack is not on the tail of the kick or 808 before a snare, as
    in the full band (calibration snares: 21-24 ms early by the full band,
    -0.6 to +1.8 ms by the snare band, 12 analyses). 808: notes_808
    (bass808's: t_attack, end, midi_float the pitch heard, midi_settled,
    pitch the note written, glide; an analysis from before bass808 has no
    t_attack, and its t, an onset backtracked in the 808 band, is taken as
    the attack). Rhythm: the grid as
    grid.estimate left it (_analysis_time): tempo, meter, first bar line,
    swing and tempo map. Melodic notes and chords: melodic.json (_melodic);
    without it the estimate carries neither, and they are not scored.

    A hit of a pitched copy (its event carries semitones, layers.explain)
    is a sound of its own, <name><semitones:+d>, as a scenario's pitched
    copies are, attacking where the template transposed that far (as
    Simpler plays it) attacks. Hits found under another sound are flagged
    "layer", a flam's grace note "flam"."""
    work = Path(work)
    analysis = _load(work / "analysis.json")
    matches = _load(work / "matches.json") if (work / "matches.json").is_file() else {}
    sounds, drums = {}, []
    for name, info in analysis["templates"].items():
        role = _drum_role((matches.get(name) or {}).get("role") or info.get("role_guess"))
        path = work / "templates" / f"{name}.npy"
        audio = np.load(path) if path.is_file() else None
        for e in [{"semitones": 0}] + analysis["events"].get(name, []):
            st = int(e.get("semitones", 0))
            sid = f"{name}{st:+d}" if st else name
            if sid not in sounds:
                sounds[sid] = {"role": role, "path": str(path), "sha256": _sha256(path), "semitones": float(st),
                               "cents": 0.0, "attack_s": round(_lead(info, audio, role, st), 5)}
            if "t" not in e:
                continue
            flags = ["layer"] * (e.get("source") == "layer") + ["flam"] * ("flam_of" in e)
            drums.append({"sound": sid, "role": role, "t": e["t"],
                          "t_attack": round(max(0.0, e.get("t_attack", e["t"] + sounds[sid]["attack_s"])), 5),
                          "gain": e.get("gain", 1.0), "velocity": None, "layer_of": None, "flags": flags})
    bass = [{"sound": "808", "t": n["t"], "t_attack": n.get("t_attack", n["t"]), "end": n["end"],
             "midi": float(n.get("midi_float", n.get("pitch"))), "midi_settled": n.get("midi_settled"),
             "pitch": n.get("pitch"), "glide": _glide(n.get("glide")), "with_kick": False, "gain": n.get("peak", 1.0),
             "flags": list(n.get("flags", [])) + ([f"slide:{n['slide']:+.1f}"] if abs(n.get("slide", 0.0)) >= 1.5
                                                 else [])}
            for n in analysis.get("notes_808", [])]
    if bass:
        t808 = work / "templates" / "808.npy"
        sounds["808"] = {"role": "808", "path": str(t808), "sha256": _sha256(t808), "semitones": 0.0, "cents": 0.0,
                         "attack_s": 0.0}
    drums.sort(key=lambda e: e["t_attack"])
    _with_kick(bass, drums)
    duration = analysis["source"]["duration_s"]
    return _events(scenario or work.name, {"from": "analysis", "work": str(work)},
                   audio={"sr": SR, "n_samples": int(round(duration * SR))},
                   time=_analysis_time(analysis["grid"], duration),
                   sounds=sounds, drums=drums, bass808=bass, **_melodic(work / "melodic.json"))


def _plan_808(plan: dict, root, velocity_curve, sounds: dict, missing: list) -> list[dict]:
    t808 = plan["tracks"].get("808") or {}
    if not t808.get("notes") or "sample_path" not in t808:
        return []  # no one-shot: the build makes no 808 track, nothing plays
    s, path = settings_808(t808), resolve(t808["sample_path"], root)
    try:
        shot = read(path)[int(round(s["start_s"] * SR)):]
    except (OSError, RuntimeError):
        missing.append(str(t808["sample_path"]))
        return []
    f0, curve = S.tuning(shot)
    cents = s["cents"] if t808.get("cents") is not None else round(100 * (S.hz_to_midi(f0) - s["root_midi"]), 1)
    bps, gain = plan["tempo"] / 60, 10 ** ((fader_db(plan, "808") + s["volume_db"]) / 20)
    own = s["root_midi"] + cents / 100  # the one-shot's settled pitch, played at shift 0
    leads: dict[float, float] = {}
    lengths: dict[float, float] = {}

    def played_s(pitch: int) -> float:
        shift = pitch_808(s, pitch)[0]
        if shift not in lengths:
            lengths[shift] = len(as_played(shot, 0.0, shift)) / SR
        return lengths[shift]

    chains = chains_808(t808)
    late = late_808([c[0] for c in chains], bps, played_s) if not glide_s(t808) else np.zeros(len(chains), bool)
    bass = []
    for c, chain in enumerate(chains):
        first = chain[0]
        shift = pitch_808(s, first["pitch"])[0]
        if shift not in leads:
            leads[shift] = attack_time(as_played(shot, 0.0, shift), "808")
        semis = chain_semitones(chain, s, bps, glide_s(t808))
        pos = read_at(semis)  # where Simpler reads the one-shot, sample by sample
        heard = own + semis + S.drop(curve, pos / SR) / 100
        heard[pos >= len(shot)] = np.nan  # the one-shot has ended
        a0 = first["start_time"] / bps
        v = velocity_gain(first["velocity"], s["velocity_curve"] or velocity_curve, s["vol_vel"])
        for k, n in enumerate(chain):
            t = n["start_time"] / bps
            stop = chain[k + 1]["start_time"] / bps if k + 1 < len(chain) else t + n["duration"] / bps
            o = int(round((t - a0) * SR))
            ring = float(np.sum(np.isfinite(heard[o:])) - 1) / SR  # until the voice reads past the one-shot
            if ring <= 0:  # it had ended before this note started: nothing plays
                continue
            on = t + RETRIGGER_S * (late[c] and not k)  # where Live starts the sample, which sounds from there on
            dur = min(stop - on, ring)
            midi = S.sounding(lambda x, o=o, h=heard: h[np.clip(o + np.round(x * SR).astype(int), 0, len(h) - 1)],
                              dur)
            glide = None if not k else {"from_midi": round(own + semis[max(0, o - 1)], 4),
                                        "to_midi": round(own + pitch_808(s, n["pitch"])[0], 4), "t": round(t, 5),
                                        "ms": round(1000 * glide_s(t808), 1), "shape": "lin"}
            bass.append({"sound": "808", "t": round(on, 5), "t_attack": round(on + (leads[shift] if not k else 0.0), 5),
                         "end": round(stop, 5), "midi": round(float(midi), 4), "pitch": int(n["pitch"]),
                         "midi_settled": round(own + pitch_808(s, n["pitch"])[0], 4), "glide": glide,
                         "with_kick": False, "gain": round(gain * v, 5),
                         "flags": ["legato"] if k else ["retrigger"] if late[c] else []})
    at_root = as_played(shot, 0.0, pitch_808(s, s["root_midi"])[0])
    sounds["808"] = {"role": "808", "path": str(path), "sha256": _sha256(path), "semitones": s["transpose"],
                     "cents": cents, "attack_s": round(attack_time(at_root, "808"), 5), "f0_hz": round(f0, 3)}
    return bass


def estimate_from_plan(plan: dict, root: str | Path | None = None, scenario: str = "",
                       velocity_curve="live") -> dict:
    """What Live would play, as an estimate (render.py's assumptions).

    Drum note: t = start_time * 60 / tempo, RETRIGGER_S later where it
    comes while its pad's hit before still plays (render.pad_hits: Live's
    pads play one voice; flagged "retrigger"); t_attack = t +
    attack_time(the pad's sample as played: from frame 0, or from start_s,
    transposed by transpose + detune_cents/100 semitones) in the pad role's
    band; sound id = the pad's note, role = the pad's. 808 note (pitch: the MIDI note
    written): the one-shot transposed by (pitch - 60) + transpose +
    detune_cents/100 semitones, attack read in the 808 band, ending at the
    note end, RETRIGGER_S later where the voice of the note before at its
    pitch still sounds (render.late_808; flagged "retrigger"). It settles
    at root_midi + cents/100 + that shift
    (midi_settled; render.pitch_808: pitch plus the 808's tuning and
    heard_cents with tune_808's Transpose and Detune; a plan without cents
    gets the one-shot's own, scenarios.tuning), and sounds (midi) as a
    truth's note does: the one-shot's own pitch drop (its pitch curve, read
    at the note's rate) through scenarios.sounding over the note's length.
    With Glide on, a note that
    starts while the one before is held (render.chains_808) is a legato
    note: no attack (t_attack = t), the voice sliding to its pitch over the
    Glide Time with the one-shot read on, its glide in the schema's form.
    gain = track fader x volume_db x the velocity's level (render's law at
    the pad's or the 808's vol_vel; a chain's first note's).
    Notes on a pad without a readable sample, and 808 notes without a
    one-shot, play nothing and are left out (their files under "missing");
    so is a legato note that starts after the voice has read past the
    one-shot's end (a long chain at a high pitch: the voice is silent).
    Melodic notes: the plan's melodic tracks (_plan_melodic). Rhythm: Live's
    own grid, bar 1 at 0 s at the plan's tempo; the plan puts notes at their
    seconds, so the song's bars fall on it only when the song's first bar
    line is at 0 s."""
    bps = plan["tempo"] / 60
    track = plan["tracks"]["drums"]
    sounds, settings, shots, missing = {}, {}, {}, []
    for note, pad in track["pads"].items():
        key, s = str(note), pad_settings(pad)
        path, role = resolve(pad["path"], root), _drum_role(s["role"])
        try:
            x = as_played(read(path), s["start_s"], s["transpose"] + s["detune_cents"] / 100)
        except (OSError, RuntimeError):
            missing.append(str(pad["path"]))
            continue
        settings[key], shots[key] = s, x
        sounds[key] = {"role": role, "path": str(path), "sha256": _sha256(path), "semitones": s["transpose"],
                       "cents": s["detune_cents"], "attack_s": round(attack_time(x, role), 5)}
    drums = [{"sound": h["pad"], "role": sounds[h["pad"]]["role"], "t": round(h["t"], 5),
              "t_attack": round(h["t"] + sounds[h["pad"]]["attack_s"], 5), "gain": round(h["gain"], 5),
              "velocity": h["velocity"], "layer_of": None, "flags": ["retrigger"] * h["late"]}
             for h in pad_hits(plan, shots, settings, velocity_curve)]
    drums.sort(key=lambda e: e["t_attack"])
    bass = _plan_808(plan, root, velocity_curve, sounds, missing)
    _with_kick(bass, drums)
    length = plan.get("length_beats", 0.0) / bps
    return _events(scenario, {"from": "plan", "velocity_curve": str(velocity_curve), "missing": missing},
                   audio={"sr": SR, "n_samples": int(round(length * SR))},
                   time=_grid_time(plan["tempo"], 0.0, int(plan.get("beats_per_bar", 4)), length),
                   sounds=sounds, drums=drums, bass808=bass, **_plan_melodic(plan, bps))


# ── Drums ──

def _times(events: list[dict], key: str = "t_attack") -> np.ndarray:
    return np.sort(np.array([e[key] for e in events], dtype=float))


def _by_sound(events: list[dict]) -> dict[str, np.ndarray]:
    groups: dict[str, list[dict]] = {}
    for e in events:
        groups.setdefault(e["sound"], []).append(e)
    return {s: _times(groups[s]) for s in sorted(groups)}


def _pairs(ref: np.ndarray, est: np.ndarray, window: float) -> list[tuple[int, int]]:
    """A maximum one-to-one matching of times within window (mir_eval)."""
    return mir_eval.util.match_events(ref, est, window) if len(ref) and len(est) else []


def _prf(n: int, n_ref: int, n_est: int) -> tuple[float, float, float]:
    p = n / n_est if n_est else 0.0
    r = n / n_ref if n_ref else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


def _stats(d: np.ndarray) -> dict | None:
    if not len(d):
        return None
    return {"n": len(d), "mean": round(float(np.mean(d)), 2), "median": round(float(np.median(d)), 2),
            "p90_abs": round(float(np.percentile(np.abs(d), 90)), 2)}


def onset_scores(ref, est) -> dict:
    """F1, precision and recall of hit times at 20 and 50 ms with no offset
    removed, and the signed error (estimate - truth, ms) of the pairs matched
    within 50 ms: mean, median, and the 90th percentile of its size."""
    ref, est = np.sort(np.asarray(ref, dtype=float)), np.sort(np.asarray(est, dtype=float))
    out: dict = {"n_truth": len(ref), "n_est": len(est)}
    for name, w in WINDOWS.items():
        f, p, r = _prf(len(_pairs(ref, est, w)), len(ref), len(est))
        out |= {f"f1_{name}": round(f, 4), f"precision_{name}": round(p, 4), f"recall_{name}": round(r, 4)}
    out["error_ms"] = _stats(np.array([est[j] - ref[i] for i, j in _pairs(ref, est, WINDOWS["50"])]) * 1000)
    return out


def v1_scores(ref, groups: list) -> dict:
    """calibrate.blind's rule: each estimated sound's own constant offset
    removed (calibrate._f1; offsets_ms lists them), the sounds pooled, then
    F1 at 20 ms after one more offset."""
    ref = sorted(float(t) for t in ref)
    found: list[float] = []
    offsets = []
    for times in groups:
        offsets.append(_f1(list(times), ref)["offset_ms"])
        found += [float(t) - offsets[-1] / 1000 for t in times]
    return {**_f1(found, ref), "offsets_ms": offsets}


def moments(times: np.ndarray, window: float = WINDOWS["50"]) -> np.ndarray:
    """Sorted times with every time within `window` after the last one kept
    dropped: a sound's hits as distinct moments, a doubled hit once."""
    kept: list[float] = []
    for t in np.sort(np.asarray(times, dtype=float)):
        if not kept or t - kept[-1] > window:
            kept.append(float(t))
    return np.array(kept)


def assign(truth: dict[str, np.ndarray], est: dict[str, np.ndarray], share: float = 0.5,
           roles: tuple[dict, dict] = ({}, {})) -> tuple[dict, list]:
    """Which truth sound each estimated sound plays: the one it scores the
    best F1 at 50 ms on (ties: the more hits matched within 50 ms, then F1 at
    20 ms, then the same role, from roles = (truth roles, estimated roles)).
    Matched hits first would give a snare pad with one stray hit on a hat
    position to the hat, since every snare of these beats sits on a hat too
    (17 matches against 16). Several may play one truth sound. One whose
    distinct moments (moments(): a doubled hit counts once) match that sound
    less than `share` of the time is extra: listed with its hit count, scored
    nowhere. Counted over hits, a pad that plays every snare twice and one
    stray hit (16 of 33) was extra, while the same hits on two pads scored."""
    out, extra = {}, []
    for e, et in est.items():
        best, key = None, (0.0, 0, 0.0, False)
        for s, tt in truth.items():
            n = len(_pairs(tt, et, WINDOWS["50"]))
            k = (_prf(n, len(tt), len(et))[0], n, _prf(len(_pairs(tt, et, WINDOWS["20"])), len(tt), len(et))[0],
                 roles[0].get(s) is not None and roles[0].get(s) == roles[1].get(e))
            if k > key:
                best, key = s, k
        m = moments(et)
        on = len(_pairs(truth[best], m, WINDOWS["50"])) if best is not None else 0
        row = {"hits": len(et), "matched_50": key[1], "moments": len(m),
               "share": round(on / len(m), 3) if len(m) else 0.0}
        if best is not None and on >= share * len(m):
            out[e] = {"to": best, **row}
        else:
            extra.append({"sound": e, "closest": best, **row})
    return out, extra


def _roles(ev: dict, events: list[dict]) -> dict[str, str | None]:
    """Each sound's role: from sounds, else from its first event."""
    listed = {s: v.get("role") for s, v in (ev.get("sounds") or {}).items()}
    out: dict[str, str | None] = {}
    for e in events:
        out.setdefault(e["sound"], listed.get(e["sound"]) or e.get("role"))
    return out


def _drum_assignment(truth: dict, est: dict) -> tuple[dict, dict, dict, list]:
    t_events, e_events = truth.get("drums") or [], est.get("drums") or []
    t_times, e_times = _by_sound(t_events), _by_sound(e_events)
    return t_times, e_times, *assign(t_times, e_times, roles=(_roles(truth, t_events), _roles(est, e_events)))


def _same_file_cents(truth: dict, est: dict, assignment: dict) -> dict[str, dict[str, float]]:
    """Per truth sound, each estimated sound assigned to it that plays its
    file (the same sha256) and how many cents off its pitch (semitones +
    cents/100 on both sides, estimate minus truth). Another file has no
    pitch to compare."""
    t_sounds, e_sounds = truth.get("sounds") or {}, est.get("sounds") or {}
    out: dict[str, dict[str, float]] = {}
    for e, a in assignment.items():
        ts, es = t_sounds.get(a["to"]) or {}, e_sounds.get(e) or {}
        if ts.get("sha256") and ts["sha256"] == es.get("sha256"):
            pitch = [100 * v.get("semitones", 0.0) + v.get("cents", 0.0) for v in (es, ts)]
            out.setdefault(a["to"], {})[e] = round(float(pitch[0] - pitch[1]), 2)
    return out


def drum_scores(truth: dict, est: dict, files: bool = False) -> dict | None:
    """Per truth drum sound (a layer, layer_of, is a sound of its own): the
    onset scores of the estimated sounds assigned to it, their hits pooled (a
    hit played twice is a false positive, on one pad or on two), v1's rule
    beside them; then the assignment, the extra sounds, and the macro means
    over truth sounds. files: the estimate's sounds are the files a sampler
    plays (a plan's pads), so pitch scores the ones that play their truth
    sound's file (_same_file_cents): same_file, the share of truth sounds
    played by their own file, and |cents| over those pads, mean and max.
    The suite's detuned snare (-0.70 semitones) played with Transpose
    alone is 30 cents off, which no onset score sees."""
    events = truth.get("drums") or []
    if not events:
        return None
    t_times, e_times, assignment, extra = _drum_assignment(truth, est)
    layer_of, roles = {e["sound"]: e.get("layer_of") for e in events}, _roles(truth, events)
    cents = _same_file_cents(truth, est, assignment) if files else {}
    sounds = {}
    for s, ref in t_times.items():
        mine = [e for e, a in assignment.items() if a["to"] == s]
        pooled = np.concatenate([e_times[e] for e in mine]) if mine else np.zeros(0)
        sounds[s] = {"role": roles[s], "layer_of": layer_of[s], "estimated": mine,
                     **onset_scores(ref, pooled), "v1": v1_scores(ref, [e_times[e] for e in mine])}
        if files:
            sounds[s]["same_file_cents"] = cents.get(s)
    macro = {k: round(float(np.mean([v[k] for v in sounds.values()])), 4) for k in ("f1_20", "f1_50")}
    macro["v1_f1"] = round(float(np.mean([v["v1"]["f1"] for v in sounds.values()])), 4)
    out = {"sounds": sounds, "assignment": assignment, "extra": extra, "macro": macro}
    if files:
        off = [abs(c) for v in cents.values() for c in v.values()]
        out["pitch"] = {"same_file": round(len(cents) / len(sounds), 4), "pads": len(off),
                        "cents_abs_mean": round(float(np.mean(off)), 2) if off else None,
                        "cents_abs_max": round(float(np.max(off)), 2) if off else None}
    return out


# ── Notes: 808, melodic layers, chords ──

def _midi(n: dict) -> float:
    return float(n["midi"]) + float(n.get("cents", 0.0)) / 100


def _hz(notes: list[dict]) -> np.ndarray:
    return 440.0 * 2 ** ((np.array([_midi(n) for n in notes]) - 69) / 12)


def _intervals(notes: list[dict], onset: str) -> np.ndarray:
    on = np.maximum(np.array([n[onset] for n in notes], dtype=float), 0.0)
    return np.stack([on, np.maximum(np.array([n["end"] for n in notes], dtype=float), on + 1e-3)], axis=1)


def note_f1(ref: list[dict], est: list[dict], window: float = 0.05, onset: str = "t") -> float:
    """mir_eval's note F1: onset within window, pitch within 50 cents,
    offsets ignored. Pitches go in as Hz from midi + cents/100."""
    if not ref or not est:
        return 0.0
    return float(mir_eval.transcription.precision_recall_f1_overlap(
        _intervals(ref, onset), _hz(ref), _intervals(est, onset), _hz(est), onset_tolerance=window,
        pitch_tolerance=50.0, offset_ratio=None)[2])


def tuning_808(notes: list[dict]) -> float | None:
    """The tuning (semitones, -0.5 to 0.5) the notes' settled pitches share:
    the median of their fractional parts around their circular mean, legato
    notes aside (a slide may stop anywhere). A truth's notes share one
    exactly (the sample's own tuning, or a line's cents); an analysis' are
    the template's settled pitch plus each note's transposition."""
    s = np.array([n["midi_settled"] for n in notes
                  if n.get("midi_settled") is not None and "legato" not in (n.get("flags") or [])], dtype=float)
    if not len(s):
        return None
    frac = s - np.round(s)
    centre = float(np.angle(np.mean(np.exp(2j * np.pi * frac)))) / (2 * np.pi)
    return round(centre + float(np.median((frac - centre + 0.5) % 1.0 - 0.5)), 4)


def _settled(ref: list[dict], mine: list[dict], pairs) -> dict | None:
    """Over the notes matched within 50 ms: how far the settled pitch
    (midi_settled) is from the truth's (cents), the share within 50 cents,
    and the share whose written pitch (the MIDI note an estimate writes,
    its "pitch"; else, as for a truth, the settled pitch less the side's
    own tuning_808, rounded) is the truth's. The heard pitch (cents) cannot
    see a whole line written a semitone high on a tuning read 40 cents
    off: on bass_low-1 note F1@50 read 0.994 with every note written one
    semitone high. A plan's settled pitch carries its heard Detune (up to
    71 cents), which can take it past half a semitone from the note."""
    pairs = [(i, j) for i, j in pairs if ref[i].get("midi_settled") is not None
             and mine[j].get("midi_settled") is not None]
    if not pairs:
        return None
    tr, te = tuning_808(ref), tuning_808(mine)

    def written(n: dict, tuning: float) -> int:
        return int(n["pitch"]) if n.get("pitch") is not None else round(n["midi_settled"] - tuning)
    c = np.array([100 * (mine[j]["midi_settled"] - ref[i]["midi_settled"]) for i, j in pairs])
    written = [written(mine[j], te) == written(ref[i], tr) for i, j in pairs]
    return {"n": len(c), "median": round(float(np.median(c)), 1), "median_abs": round(float(np.median(np.abs(c))), 1),
            "within_50": round(float(np.mean(np.abs(c) <= 50)), 4), "written_equal": round(float(np.mean(written)), 4),
            "tuning_cents": {"truth": round(100 * tr, 1), "est": round(100 * te, 1)}}


def _legato(ref: list[dict], mine: list[dict], window: float = WINDOWS["50"]) -> dict:
    """The legato notes (flag "legato": no attack of their own, a slide from
    the note before): the estimate's against the truth's, matched one to one
    on where the slide starts (t, within window) with the pitch it slides to
    (midi_settled, else midi) within 50 cents. precision is None without an
    estimated legato note, recall without a truth one."""
    def legato(notes):
        return sorted((n for n in notes if "legato" in (n.get("flags") or [])), key=lambda n: n["t"])

    r, e = legato(ref), legato(mine)

    def to(n):
        return n["midi_settled"] if n.get("midi_settled") is not None else n["midi"]
    hit = sum(abs(to(e[j]) - to(r[i])) <= 0.5 for i, j in _pairs(_times(r, "t"), _times(e, "t"), window))
    return {"n_truth": len(r), "n_est": len(e), "matched": hit,
            "precision": round(hit / len(e), 4) if e else None, "recall": round(hit / len(r), 4) if r else None}


def bass_scores(truth: dict, est: dict) -> dict | None:
    """The 808 notes: onset scores on attack times (strict, and v1's rule),
    note F1 at 20 and 50 ms (pitch within 50 cents, offsets ignored), the
    pitch error (cents, estimate - truth) of the notes whose onsets match
    within 50 ms, the same notes' settled and written pitches (_settled),
    and the legato notes' precision and recall (_legato)."""
    ref = sorted(truth.get("bass808") or [], key=lambda n: n["t_attack"])
    if not ref:
        return None
    mine = sorted(est.get("bass808") or [], key=lambda n: n["t_attack"])
    rt, et = _times(ref), _times(mine)
    out = {**onset_scores(rt, et), "v1": v1_scores(rt, [et] if len(et) else [])}
    for name, w in WINDOWS.items():
        out[f"note_f1_{name}"] = round(note_f1(ref, mine, w, "t_attack"), 4)
    pairs = _pairs(rt, et, WINDOWS["50"])
    c = np.array([100 * (_midi(mine[j]) - _midi(ref[i])) for i, j in pairs])
    out["cents"] = None if not len(c) else {
        "n": len(c), "median": round(float(np.median(c)), 1), "median_abs": round(float(np.median(np.abs(c))), 1),
        "p90_abs": round(float(np.percentile(np.abs(c), 90)), 1)}
    out["settled"] = _settled(ref, mine, pairs)
    out["legato"] = _legato(ref, mine)
    return out


def _note_block(ref: list[dict], est: list[dict]) -> dict:
    onset = 0.0
    if ref and est:
        onset = mir_eval.transcription.onset_precision_recall_f1(_intervals(ref, "t"), _intervals(est, "t"),
                                                                 onset_tolerance=0.05)[2]
    return {"n_truth": len(ref), "n_est": len(est), "note_f1": round(note_f1(ref, est), 4),
            "onset_f1": round(float(onset), 4)}


def melodic_scores(truth: dict, est: dict) -> dict | None:
    """Note F1 (onset 50 ms, pitch 50 cents, offsets ignored) and onset-only
    F1, pooled and per truth layer. A truth layer is scored against the
    estimate's layer of the same name, else against the one that scores best
    on it (named in estimated_layer; none when no layer scores above 0).
    pooled: the truth's notes against those of the layers picked. A layer
    no truth layer picks is extra, listed with its note count and scored
    nowhere, as a drum sound assigned to no truth sound: the estimate's
    layers are the separated stems, and a scenario's vocal is a library
    sample whose notes its truth does not list."""
    ref = truth.get("notes") or []
    if not ref:
        return None
    named: dict[str, list[dict]] = {}
    for n in est.get("notes") or []:
        named.setdefault(n["layer"], []).append(n)
    layers = {}
    for layer in sorted({n["layer"] for n in ref}):
        r = [n for n in ref if n["layer"] == layer]
        pick = layer if layer in named else max(named, key=lambda k, r=r: note_f1(r, named[k]), default=None)
        if pick not in (None, layer) and note_f1(r, named[pick]) == 0:
            pick = None
        layers[layer] = {"estimated_layer": pick, **_note_block(r, named.get(pick, []))}
    picked = {v["estimated_layer"] for v in layers.values()}
    played = [n for k in sorted(picked - {None}) for n in named[k]]
    return {"pooled": _note_block(ref, played), "layers": layers,
            "extra": [{"layer": k, "notes": len(v)} for k, v in sorted(named.items()) if k not in picked]}


def _chord_track(chords: list[dict]) -> tuple[np.ndarray, list[str]]:
    """Intervals and labels in time order: a gap is no chord, an overlap ends
    where the next chord starts."""
    cs = sorted(chords, key=lambda c: c["t"])
    iv: list[list[float]] = []
    labels: list[str] = []
    for k, c in enumerate(cs):
        a, b = float(c["t"]), float(c["end"])
        if k + 1 < len(cs):
            b = min(b, float(cs[k + 1]["t"]))
        if iv and a > iv[-1][1] + 1e-6:
            iv.append([iv[-1][1], a])
            labels.append(NO_CHORD)
        if b > a:
            iv.append([a, b])
            labels.append(c["label"])
    return np.array(iv, dtype=float).reshape(-1, 2), labels


def chord_scores(truth: dict, est: dict) -> dict | None:
    """mir_eval's duration-weighted chord accuracy over the truth's span, per
    comparison rule (an estimate's chords outside the span are dropped, a
    span it leaves empty counts as no chord); None for a rule under which no
    truth chord can be compared."""
    ref = truth.get("chords") or []
    if not ref:
        return None
    ri, rl = _chord_track(ref)
    if not len(ri):
        return None
    lo, hi = float(ri[0, 0]), float(ri[-1, 1])
    ei, el = _chord_track(est.get("chords") or [])
    ei, el = mir_eval.util.adjust_intervals(ei, list(el), lo, hi, NO_CHORD, NO_CHORD)
    iv, rl, el = mir_eval.util.merge_labeled_intervals(ri, rl, ei, el)
    weights = mir_eval.util.intervals_to_durations(iv)
    out: dict = {"span_s": [round(lo, 3), round(hi, 3)]}
    for rule in CHORD_RULES:
        c = getattr(mir_eval.chord, rule)(rl, el)
        out[rule] = round(float(mir_eval.chord.weighted_accuracy(c, weights)), 4) if np.any(c >= 0) else None
    return out


# ── Rhythm ──

def octave(ratio: float, tolerance: float = 0.04) -> str:
    """The class of a tempo ratio (estimate / truth): x1, x2, x1/2, x3/2 or
    x2/3 within `tolerance` (4 %, as tempo accuracy is usually scored), else
    other."""
    return next((name for name, f in OCTAVES.items() if abs(ratio / f - 1) <= tolerance), "other")


def _first_downbeat(tm: dict) -> float | None:
    if tm.get("downbeat_s") is not None:
        return float(tm["downbeat_s"])
    return float(tm["downbeats_s"][0]) if tm.get("downbeats_s") else None


def _beat_s(tm: dict) -> float:
    """Seconds per beat. bpm counts quarter notes, as Live and the scenarios
    count it, also in 6/8, where a beat (beat_unit 8) is an eighth: a 6/8 bar
    at 136 BPM is 1.32 s, not the 2.65 s six beats of 60/bpm would make."""
    return 60 / tm["bpm"] * 4 / tm.get("beat_unit", 4)


def _f_beats(ref: list[float], est: list[float], span: tuple[float, float]) -> float:
    """mir_eval's beat F-measure (70 ms) over the truth's beat span: a truth
    lists beats where music plays, a grid runs to both ends of the file (on
    the calibration beats, 2 s of tail held 5 of 69 grid beats)."""
    e = np.array([t for t in est if span[0] <= t <= span[1]])
    return round(float(mir_eval.beat.f_measure(np.array(ref), e, 0.07)), 4) if len(ref) and len(e) else 0.0


def rhythm_scores(truth: dict, est: dict) -> dict | None:
    """Tempo (relative error and octave class), meter, the first downbeat's
    error modulo one bar (beats_per_bar of the truth's beats, _beat_s), with
    whether the estimate flagged its bar line as unsure, and the downbeat
    F-measure when the estimate has downbeats, the beat F-measure when both
    have beats, and swing when either has it."""
    tt, et = truth.get("time") or {}, est.get("time") or {}
    if not tt.get("bpm") or not et:
        return None
    out: dict = {}
    if et.get("bpm"):
        ratio = et["bpm"] / tt["bpm"]
        out["tempo"] = {"truth_bpm": tt["bpm"], "est_bpm": et["bpm"], "rel_error": round(ratio - 1, 6),
                        "octave": octave(ratio)}
    meters = [(m.get("beats_per_bar", 4), m.get("beat_unit", 4)) for m in (tt, et)]
    out["meter"] = {"truth": list(meters[0]), "est": list(meters[1]), "equal": meters[0] == meters[1]}
    beat = _beat_s(tt)
    first_t, first_e = _first_downbeat(tt), _first_downbeat(et)
    beats = tt.get("beats_s") or tt.get("downbeats_s") or []
    span = (beats[0] - beat / 2, beats[-1] + beat / 2) if beats else (0.0, 0.0)
    if first_t is not None and first_e is not None:
        bar = meters[0][0] * beat
        d = (first_e - first_t + bar / 2) % bar - bar / 2
        out["downbeat"] = {"first_error_ms": round(d * 1000, 2), "first_error_beats": round(d / beat, 3),
                           "unsure": "downbeat_unsure" in (et.get("flags") or [])}
        if tt.get("downbeats_s") and et.get("downbeats_s"):
            out["downbeat"]["f"] = _f_beats(tt["downbeats_s"], et["downbeats_s"], span)
    if tt.get("beats_s") and et.get("beats_s"):
        out["beat_f"] = _f_beats(tt["beats_s"], et["beats_s"], span)
    st, se = tt.get("swing"), et.get("swing")
    if st or se:
        both = bool(st and se)
        out["swing"] = {"truth": st, "est": se, "ratio_error": round(se["ratio"] - st["ratio"], 4) if both else None,
                        "level_equal": st["level"] == se["level"] if both else None}
    return out


# ── Levels, from rendered audio ──

def _energy(x: np.ndarray, t: float, n: int) -> float:
    a = max(0, int(round(t * SR)))
    return float(np.sum(np.square(x[a:a + n], dtype=np.float64)))


def _level(stem: np.ndarray, rebuild: np.ndarray, ref: np.ndarray, est: np.ndarray, n: int) -> dict:
    d = []
    for i, j in _pairs(ref, est, WINDOWS["50"]):
        a, b = _energy(rebuild, est[j], n), _energy(stem, ref[i], n)
        if a > 0 and b > 0:
            d.append(10 * np.log10(a / b))
    if not d:
        return {"n": 0}
    med = float(np.median(d))
    return {"n": len(d), "level_db": round(med, 2),
            "spread_p90_db": round(float(np.percentile(np.abs(np.array(d) - med), 90)), 2)}


def _kit(audio: dict, truth_dir, reference: dict) -> dict:
    """Per group with a reference stem: RMS (dB) of the reference and of the
    truth's group stem, and kit_db, their difference."""
    stems, gains = audio.get("stems") or {}, (audio.get("master") or {}).get("stems_gain") or {}
    out = {}
    for group, path in reference.items():
        if stems.get(group) and path and Path(path).is_file():
            ref_db = rms_db(read(path))
            truth_db = rms_db(read(resolve(stems[group], truth_dir)) / gains.get(group, 1.0))
            out[group] = {"reference_db": round(ref_db, 2), "truth_db": round(truth_db, 2),
                          "kit_db": round(ref_db - truth_db, 2)}
    return out


def level_scores(truth: dict, est: dict, render_dir: str | Path, truth_dir: str | Path | None = None,
                 render_gain: float = 1.0, ms: float = 50.0, reference: dict | None = None) -> dict:
    """Per truth sound, how loud the rebuild plays it: over its hits matched
    within 50 ms (strict times), the energy in the role's band over `ms` from
    each side's own attack, rebuild over truth stem; level_db is the median
    (dB), spread_p90_db the 90th percentile of each hit's distance from it.
    The rebuild is render_plan's tracks of the estimated sounds assigned to
    that truth sound (the 808 track for an 808), times render_gain (e.g. a
    master's small-signal gain, to compare in the truth stems' domain).
    balance_db is each level minus their median over sounds, which no
    overall gain changes. A truth sound without a stem or a rendered track is
    None. A stem written below unity is read back at its level, divided by
    audio.master.stems_gain (a scenario halves a stem that would clip in
    FLAC: three group stems of the 27 real beats, none of a sound).

    reference names the stems the plan's track faders were matched to, per
    group ({"drums": path, "bass": path}: the separated stems): a fader puts
    its track's RMS (at render_gain) on the reference's, so a perfect kit
    (the truth's sounds at the truth's relative gains) plays every sound of
    a group at its kit_db, RMS(reference) - RMS(truth group stem), and a
    sound's vs_kit_db (level_db - kit_db) is what is wrong within the kit.
    On the htdemucs baseline the drums' kit_db is -5.6 dB (median over 48
    beats; Demucs separates the clipped mix, the truth is before the
    clipper) and the bass's -1.0 dB: per-role level_db mostly showed that."""
    render_dir = Path(render_dir)
    tracks = _load(render_dir / "render.json")["tracks"]
    audio = truth.get("audio") or {}
    stems, gains = audio.get("stems") or {}, (audio.get("master") or {}).get("stems_gain") or {}
    t_drums, e_times, assignment, _ = _drum_assignment(truth, est)
    roles = _roles(truth, truth.get("drums") or [])
    jobs = [(s, roles[s], ref, [e for e, a in assignment.items() if a["to"] == s]) for s, ref in t_drums.items()]
    e_bass = _by_sound(est.get("bass808") or [])
    jobs += [(s, "808", ref, list(e_bass)) for s, ref in _by_sound(truth.get("bass808") or []).items()]
    e_times |= e_bass
    kit = _kit(audio, truth_dir, reference or {})
    out: dict = {}
    for s, role, ref, ids in jobs:
        ids = [e for e in ids if e in tracks]
        key = s if stems.get(s) else role
        if not stems.get(key) or not ids:
            out[s] = None
            continue
        b = ATTACK_BAND.get(role, "full")
        x_truth = band(read(resolve(stems[key], truth_dir)) / gains.get(key, 1.0), b)
        x_rebuild = band(sum(read(render_dir / tracks[e]) for e in ids) * render_gain, b)
        est_t = np.sort(np.concatenate([e_times[e] for e in ids]))
        out[s] = {"role": role, **_level(x_truth, x_rebuild, ref, est_t, samples(ms))}
        group = kit.get("bass" if role == "808" else "drums")
        if group and "level_db" in out[s]:
            out[s]["vs_kit_db"] = round(out[s]["level_db"] - group["kit_db"], 2)
    measured = [v["level_db"] for v in out.values() if v and "level_db" in v]
    for v in out.values():
        if v and "level_db" in v:
            v["balance_db"] = round(v["level_db"] - float(np.median(measured)), 2)
    return {"render_gain": render_gain, "window_ms": ms, "kit": kit, "sounds": out}


# ── Everything ──

def score(truth: dict | str | Path, estimate: dict | str | Path, render_dir: str | Path | None = None,
          truth_dir: str | Path | None = None, render_gain: float = 1.0, reference: dict | None = None) -> dict:
    """Every score above for one truth and one estimate (dicts, or paths of
    events files: a truth file's audio paths are read from its folder).
    levels needs render_dir, render_plan's output for the plan the estimate
    came from (and reference, the stems its faders were set against, for
    the kit levels). A plan's drum pads are also scored for pitch
    (drum_scores' files). Melodic notes and chords are scored only when the
    estimate carries them: without them the step did not run, and zeros
    would read as notes it missed. summary: the strict macro F1 at 20 ms over
    the truth's drum sounds and the 808 onsets, the end-to-end replica
    number, and the same macro by v1's rule."""
    if isinstance(truth, str | Path):
        truth_dir = truth_dir or Path(truth).parent
        truth = load_events(truth)
    if isinstance(estimate, str | Path):
        estimate = load_events(estimate)
    validate(truth)
    validate(estimate)
    source = ((estimate.get("generator") or {}).get("switches") or {}).get("from")
    out: dict = {"scenario": truth.get("scenario"), "estimate_from": source,
                 "drums": drum_scores(truth, estimate, files=source == "plan"), "808": bass_scores(truth, estimate),
                 "melodic": melodic_scores(truth, estimate) if "notes" in estimate else None,
                 "chords": chord_scores(truth, estimate) if "chords" in estimate else None,
                 "rhythm": rhythm_scores(truth, estimate)}
    if render_dir is not None:
        out["levels"] = level_scores(truth, estimate, render_dir, truth_dir, render_gain, reference=reference)
    parts = {s: (v["f1_20"], v["v1"]["f1"]) for s, v in ((out["drums"] or {}).get("sounds") or {}).items()}
    if out["808"]:
        parts["808 onsets"] = (out["808"]["f1_20"], out["808"]["v1"]["f1"])
    strict, v1 = (round(float(np.mean(f)), 4) for f in zip(*parts.values())) if parts else (None, None)
    out["summary"] = {"strict_f1_20": strict, "v1_f1": v1, "over": list(parts)}
    return out
