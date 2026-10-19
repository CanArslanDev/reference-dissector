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
from .audio import ATTACK_BAND, PRE, SR, attack_time, band, read, samples
from .calibrate import _f1
from .render import as_played, fader_db, pad_settings, pitch_808, resolve, settings_808, velocity_gain

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


def _grid_time(bpm: float, first: float, beats_per_bar: int, end: float) -> dict:
    beats = np.arange(first, max(end, first + 1e-9), 60 / bpm)
    return {"bpm": bpm, "beats_per_bar": beats_per_bar, "beat_unit": 4, "downbeat_s": round(first, 5),
            "beats_s": np.round(beats, 5).tolist(), "downbeats_s": np.round(beats[::beats_per_bar], 5).tolist(),
            "swing": None, "tempo_map": None}


def _with_kick(bass: list[dict], drums: list[dict], window: float = 0.02) -> None:
    """Mark the 808 notes that attack within `window` of a kick's attack."""
    kicks = np.array([e["t_attack"] for e in drums if e["role"] == "kick"])
    for n in bass:
        n["with_kick"] = bool(len(kicks) and np.min(np.abs(kicks - n["t_attack"])) <= window)


# ── Estimates ──

def _melodic(path: Path) -> tuple[list[dict], list[dict]]:
    """Notes and chords from melodic.json ({"notes": [...], "chords": [...]}
    with the events fields; a fractional midi is split into midi and cents)."""
    if not path.is_file():
        return [], []
    m = _load(path)
    notes = []
    for n in m.get("notes", []):
        pitch = float(n["midi"]) + float(n.get("cents", 0.0)) / 100
        midi = int(round(pitch))
        notes.append({"layer": str(n.get("layer", "melodic")), "t": float(n["t"]), "end": float(n["end"]),
                      "midi": midi, "cents": round((pitch - midi) * 100, 1), "velocity": int(n.get("velocity", 100)),
                      "flags": list(n.get("flags", []))})
    chords = [{"t": float(c["t"]), "end": float(c["end"]), "label": str(c["label"])} for c in m.get("chords", [])]
    return notes, chords


def estimate_from_analysis(work: str | Path, scenario: str | None = None) -> dict:
    """What the analysis heard, as an estimate.

    Drums: every template's hits; sound id = the template's name, role = its
    match's (matches.json) or else its guess. detect() reports a hit at
    t = (peak + PRE) / SR, PRE (5 ms) after the start of the template's
    window, so the hit attacks at t - PRE/SR + attack_time(template, role),
    unless the event carries t_attack. The template's attack is read in the
    role's band: where its full-band attack sits on the tail of the kick or
    808 before it, the band puts it back where the snare starts (calibration
    snares: 21-24 ms early by the full band, -0.6 to +1.8 ms by the snare
    band, 12 analyses). 808: notes_808 (t, end, midi_float; t is an onset
    backtracked in the 808 band, taken as the attack). Rhythm: the grid, its
    first beat taken as a bar line, as Grid does. Melodic notes and chords:
    melodic.json when it exists."""
    work = Path(work)
    analysis = _load(work / "analysis.json")
    matches = _load(work / "matches.json") if (work / "matches.json").is_file() else {}
    sounds, drums = {}, []
    for name, info in analysis["templates"].items():
        role = _drum_role((matches.get(name) or {}).get("role") or info.get("role_guess"))
        path = work / "templates" / f"{name}.npy"
        lead = attack_time(np.load(path), role) - PRE / SR if path.is_file() else 0.0
        sounds[name] = {"role": role, "path": str(path), "sha256": _sha256(path), "semitones": 0.0, "cents": 0.0,
                        "attack_s": round(lead, 5)}
        for e in analysis["events"].get(name, []):
            drums.append({"sound": name, "role": role, "t": e["t"],
                          "t_attack": round(max(0.0, e.get("t_attack", e["t"] + lead)), 5),
                          "gain": e.get("gain", 1.0), "velocity": None, "layer_of": None, "flags": []})
    bass = [{"sound": "808", "t": n["t"], "t_attack": n.get("t_attack", n["t"]), "end": n["end"],
             "midi": float(n.get("midi_float", n.get("pitch"))), "glide": None, "with_kick": False,
             "gain": n.get("peak", 1.0),
             "flags": [f"slide:{n['slide']:+.1f}"] if abs(n.get("slide", 0.0)) >= 1.5 else []}
            for n in analysis.get("notes_808", [])]
    if bass:
        t808 = work / "templates" / "808.npy"
        sounds["808"] = {"role": "808", "path": str(t808), "sha256": _sha256(t808), "semitones": 0.0, "cents": 0.0,
                         "attack_s": 0.0}
    drums.sort(key=lambda e: e["t_attack"])
    _with_kick(bass, drums)
    notes, chords = _melodic(work / "melodic.json")
    g, duration = analysis["grid"], analysis["source"]["duration_s"]
    return _events(scenario or work.name, {"from": "analysis", "work": str(work)},
                   audio={"sr": SR, "n_samples": int(round(duration * SR))},
                   time=_grid_time(g["bpm"], g["phase"], g.get("beats_per_bar", 4), duration),
                   sounds=sounds, drums=drums, bass808=bass, notes=notes, chords=chords)


def _plan_808(plan: dict, root, velocity_curve, sounds: dict, missing: list) -> list[dict]:
    t808 = plan["tracks"].get("808") or {}
    if not t808.get("notes") or "sample_path" not in t808:
        return []  # no one-shot: the build makes no 808 track, nothing plays
    s, path = settings_808(t808), resolve(t808["sample_path"], root)
    try:
        shot = read(path)
    except (OSError, RuntimeError):
        missing.append(str(t808["sample_path"]))
        return []
    bps, gain = plan["tempo"] / 60, 10 ** ((fader_db(plan, "808") + s["volume_db"]) / 20)
    leads: dict[float, float] = {}
    bass = []
    for n in t808["notes"]:
        shift, midi = pitch_808(s, n["pitch"])
        if shift not in leads:
            leads[shift] = attack_time(as_played(shot, s["start_s"], shift), "808")
        t = n["start_time"] / bps
        bass.append({"sound": "808", "t": round(t, 5), "t_attack": round(t + leads[shift], 5),
                     "end": round(t + n["duration"] / bps, 5), "midi": round(midi, 4), "glide": None,
                     "with_kick": False,
                     "gain": round(gain * velocity_gain(n["velocity"], s["velocity_curve"] or velocity_curve), 5),
                     "flags": []})
    own = pitch_808(s, s["root_midi"])[0]
    sounds["808"] = {"role": "808", "path": str(path), "sha256": _sha256(path), "semitones": s["transpose"],
                     "cents": s["cents"], "attack_s": round(attack_time(as_played(shot, s["start_s"], own), "808"), 5)}
    return bass


def estimate_from_plan(plan: dict, root: str | Path | None = None, scenario: str = "",
                       velocity_curve="linear") -> dict:
    """What Live would play, as an estimate (render.py's assumptions).

    Drum note: t = start_time * 60 / tempo; t_attack = t + attack_time(the
    pad's sample as played: from frame 0, or from start_s, transposed by
    transpose + detune_cents/100 semitones) in the pad role's band; sound id
    = the pad's note, role = the pad's. 808 note: the one-shot transposed by
    (pitch - 60) + transpose + detune_cents/100 semitones, attack read in the
    808 band, sounding at root_midi + cents/100 + that shift
    (render.pitch_808: pitch + cents/100 with the build's Transpose), ending
    at the note end. gain = track fader x volume_db x the velocity law.
    Notes on a pad without a readable sample, and 808 notes without a
    one-shot, play nothing and are left out (their files under "missing").
    Rhythm: Live's own grid, bar 1 at 0 s at the plan's tempo; the plan puts
    notes at their seconds, so the song's bars fall on it only when the
    song's first bar line is at 0 s."""
    bps = plan["tempo"] / 60
    track = plan["tracks"]["drums"]
    g_drums = 10 ** (fader_db(plan, "drums") / 20)
    sounds, settings, missing = {}, {}, []
    for note, pad in track["pads"].items():
        key, s = str(note), pad_settings(pad)
        path, role = resolve(pad["path"], root), _drum_role(s["role"])
        try:
            x = as_played(read(path), s["start_s"], s["transpose"] + s["detune_cents"] / 100)
        except (OSError, RuntimeError):
            missing.append(str(pad["path"]))
            continue
        settings[key] = s
        sounds[key] = {"role": role, "path": str(path), "sha256": _sha256(path), "semitones": s["transpose"],
                       "cents": s["detune_cents"], "attack_s": round(attack_time(x, role), 5)}
    drums = []
    for n in track["notes"]:
        key = str(n["pitch"])
        if key not in sounds:
            continue
        s, t = settings[key], n["start_time"] / bps
        curve = s["velocity_curve"] or velocity_curve
        gain = g_drums * 10 ** (s["volume_db"] / 20) * velocity_gain(n["velocity"], curve)
        drums.append({"sound": key, "role": sounds[key]["role"], "t": round(t, 5),
                      "t_attack": round(t + sounds[key]["attack_s"], 5), "gain": round(gain, 5),
                      "velocity": int(n["velocity"]), "layer_of": None, "flags": []})
    drums.sort(key=lambda e: e["t_attack"])
    bass = _plan_808(plan, root, velocity_curve, sounds, missing)
    _with_kick(bass, drums)
    length = plan.get("length_beats", 0.0) / bps
    return _events(scenario, {"from": "plan", "velocity_curve": str(velocity_curve), "missing": missing},
                   audio={"sr": SR, "n_samples": int(round(length * SR))},
                   time=_grid_time(plan["tempo"], 0.0, int(plan.get("beats_per_bar", 4)), length),
                   sounds=sounds, drums=drums, bass808=bass, notes=[], chords=[])


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


def assign(truth: dict[str, np.ndarray], est: dict[str, np.ndarray], share: float = 0.5,
           roles: tuple[dict, dict] = ({}, {})) -> tuple[dict, list]:
    """Which truth sound each estimated sound plays: the one most of its hits
    match within 50 ms (ties: the higher F1 at 50, then at 20 ms, then the same
    role, from roles = (truth roles, estimated roles)). Several may play one
    truth sound. One whose best match covers less than `share` of its hits is
    extra: listed with its hit count, scored nowhere."""
    out, extra = {}, []
    for e, et in est.items():
        best, key = None, (0, 0.0, 0.0, False)
        for s, tt in truth.items():
            n = len(_pairs(tt, et, WINDOWS["50"]))
            k = (n, _prf(n, len(tt), len(et))[0], _prf(len(_pairs(tt, et, WINDOWS["20"])), len(tt), len(et))[0],
                 roles[0].get(s) is not None and roles[0].get(s) == roles[1].get(e))
            if k > key:
                best, key = s, k
        row = {"hits": len(et), "matched_50": key[0], "share": round(key[0] / len(et), 3) if len(et) else 0.0}
        if best is not None and key[0] >= share * len(et):
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


def drum_scores(truth: dict, est: dict) -> dict | None:
    """Per truth drum sound (a layer, layer_of, is a sound of its own): the
    onset scores of the estimated sounds assigned to it, their hits pooled (a
    hit played twice is a false positive), v1's rule beside them; then the
    assignment, the extra sounds, and the macro means over truth sounds."""
    events = truth.get("drums") or []
    if not events:
        return None
    t_times, e_times, assignment, extra = _drum_assignment(truth, est)
    layer_of, roles = {e["sound"]: e.get("layer_of") for e in events}, _roles(truth, events)
    sounds = {}
    for s, ref in t_times.items():
        mine = [e for e, a in assignment.items() if a["to"] == s]
        pooled = np.concatenate([e_times[e] for e in mine]) if mine else np.zeros(0)
        sounds[s] = {"role": roles[s], "layer_of": layer_of[s], "estimated": mine,
                     **onset_scores(ref, pooled), "v1": v1_scores(ref, [e_times[e] for e in mine])}
    macro = {k: round(float(np.mean([v[k] for v in sounds.values()])), 4) for k in ("f1_20", "f1_50")}
    macro["v1_f1"] = round(float(np.mean([v["v1"]["f1"] for v in sounds.values()])), 4)
    return {"sounds": sounds, "assignment": assignment, "extra": extra, "macro": macro}


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


def bass_scores(truth: dict, est: dict) -> dict | None:
    """The 808 notes: onset scores on attack times (strict, and v1's rule),
    note F1 at 20 and 50 ms (pitch within 50 cents, offsets ignored), and the
    pitch error (cents, estimate - truth) of the notes whose onsets match
    within 50 ms."""
    ref = sorted(truth.get("bass808") or [], key=lambda n: n["t_attack"])
    if not ref:
        return None
    mine = sorted(est.get("bass808") or [], key=lambda n: n["t_attack"])
    rt, et = _times(ref), _times(mine)
    out = {**onset_scores(rt, et), "v1": v1_scores(rt, [et] if len(et) else [])}
    for name, w in WINDOWS.items():
        out[f"note_f1_{name}"] = round(note_f1(ref, mine, w, "t_attack"), 4)
    c = np.array([100 * (_midi(mine[j]) - _midi(ref[i])) for i, j in _pairs(rt, et, WINDOWS["50"])])
    out["cents"] = None if not len(c) else {
        "n": len(c), "median": round(float(np.median(c)), 1), "median_abs": round(float(np.median(np.abs(c))), 1),
        "p90_abs": round(float(np.percentile(np.abs(c), 90)), 1)}
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
    on it (named in estimated_layer)."""
    ref = truth.get("notes") or []
    if not ref:
        return None
    mine = est.get("notes") or []
    named: dict[str, list[dict]] = {}
    for n in mine:
        named.setdefault(n["layer"], []).append(n)
    layers = {}
    for layer in sorted({n["layer"] for n in ref}):
        r = [n for n in ref if n["layer"] == layer]
        pick = layer if layer in named else max(named, key=lambda k, r=r: note_f1(r, named[k]), default=None)
        layers[layer] = {"estimated_layer": pick, **_note_block(r, named.get(pick, []))}
    return {"pooled": _note_block(ref, mine), "layers": layers}


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


def _f_beats(ref: list[float], est: list[float], span: tuple[float, float]) -> float:
    """mir_eval's beat F-measure (70 ms) over the truth's beat span: a truth
    lists beats where music plays, a grid runs to both ends of the file (on
    the calibration beats, 2 s of tail held 5 of 69 grid beats)."""
    e = np.array([t for t in est if span[0] <= t <= span[1]])
    return round(float(mir_eval.beat.f_measure(np.array(ref), e, 0.07)), 4) if len(ref) and len(e) else 0.0


def rhythm_scores(truth: dict, est: dict) -> dict | None:
    """Tempo (relative error and octave class), meter, the first downbeat's
    error modulo one bar (a bar = beats_per_bar beats at the truth's tempo)
    and the downbeat F-measure when the estimate has downbeats, the beat
    F-measure when both have beats, and swing when either has it."""
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
    beat = 60 / tt["bpm"]
    first_t, first_e = _first_downbeat(tt), _first_downbeat(et)
    beats = tt.get("beats_s") or tt.get("downbeats_s") or []
    span = (beats[0] - beat / 2, beats[-1] + beat / 2) if beats else (0.0, 0.0)
    if first_t is not None and first_e is not None:
        bar = meters[0][0] * beat
        d = (first_e - first_t + bar / 2) % bar - bar / 2
        out["downbeat"] = {"first_error_ms": round(d * 1000, 2), "first_error_beats": round(d / beat, 3)}
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


def level_scores(truth: dict, est: dict, render_dir: str | Path, truth_dir: str | Path | None = None,
                 render_gain: float = 1.0, ms: float = 50.0) -> dict:
    """Per truth sound, how loud the rebuild plays it: over its hits matched
    within 50 ms (strict times), the energy in the role's band over `ms` from
    each side's own attack, rebuild over truth stem; level_db is the median
    (dB), spread_p90_db the 90th percentile of each hit's distance from it.
    The rebuild is render_plan's tracks of the estimated sounds assigned to
    that truth sound (the 808 track for an 808), times render_gain (e.g. a
    master's small-signal gain, to compare in the truth stems' domain).
    balance_db is each level minus their median over sounds, which no
    overall gain changes. A truth sound without a stem or a rendered track is
    None."""
    render_dir = Path(render_dir)
    tracks = _load(render_dir / "render.json")["tracks"]
    stems = (truth.get("audio") or {}).get("stems") or {}
    t_drums, e_times, assignment, _ = _drum_assignment(truth, est)
    roles = _roles(truth, truth.get("drums") or [])
    jobs = [(s, roles[s], ref, [e for e, a in assignment.items() if a["to"] == s]) for s, ref in t_drums.items()]
    e_bass = _by_sound(est.get("bass808") or [])
    jobs += [(s, "808", ref, list(e_bass)) for s, ref in _by_sound(truth.get("bass808") or []).items()]
    e_times |= e_bass
    out: dict = {}
    for s, role, ref, ids in jobs:
        ids = [e for e in ids if e in tracks]
        stem = stems.get(s) or stems.get(role)
        if not stem or not ids:
            out[s] = None
            continue
        b = ATTACK_BAND.get(role, "full")
        x_truth = band(read(resolve(stem, truth_dir)), b)
        x_rebuild = band(sum(read(render_dir / tracks[e]) for e in ids) * render_gain, b)
        est_t = np.sort(np.concatenate([e_times[e] for e in ids]))
        out[s] = {"role": role, **_level(x_truth, x_rebuild, ref, est_t, samples(ms))}
    measured = [v["level_db"] for v in out.values() if v and "level_db" in v]
    for v in out.values():
        if v and "level_db" in v:
            v["balance_db"] = round(v["level_db"] - float(np.median(measured)), 2)
    return {"render_gain": render_gain, "window_ms": ms, "sounds": out}


# ── Everything ──

def score(truth: dict | str | Path, estimate: dict | str | Path, render_dir: str | Path | None = None,
          truth_dir: str | Path | None = None, render_gain: float = 1.0) -> dict:
    """Every score above for one truth and one estimate (dicts, or paths of
    events files: a truth file's audio paths are read from its folder).
    levels needs render_dir, render_plan's output for the plan the estimate
    came from. summary: the strict macro F1 at 20 ms over the truth's drum
    sounds and the 808 onsets, the end-to-end replica number, and the same
    macro by v1's rule."""
    if isinstance(truth, str | Path):
        truth_dir = truth_dir or Path(truth).parent
        truth = load_events(truth)
    if isinstance(estimate, str | Path):
        estimate = load_events(estimate)
    validate(truth)
    validate(estimate)
    out: dict = {"scenario": truth.get("scenario"),
                 "estimate_from": ((estimate.get("generator") or {}).get("switches") or {}).get("from"),
                 "drums": drum_scores(truth, estimate), "808": bass_scores(truth, estimate),
                 "melodic": melodic_scores(truth, estimate), "chords": chord_scores(truth, estimate),
                 "rhythm": rhythm_scores(truth, estimate)}
    if render_dir is not None:
        out["levels"] = level_scores(truth, estimate, render_dir, truth_dir, render_gain)
    parts = {s: (v["f1_20"], v["v1"]["f1"]) for s, v in ((out["drums"] or {}).get("sounds") or {}).items()}
    if out["808"]:
        parts["808 onsets"] = (out["808"]["f1_20"], out["808"]["v1"]["f1"])
    strict, v1 = (round(float(np.mean(f)), 4) for f in zip(*parts.values())) if parts else (None, None)
    out["summary"] = {"strict_f1_20": strict, "v1_f1": v1, "over": list(parts)}
    return out
