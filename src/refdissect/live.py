"""From an analysis to Ableton Live: a plan (notes in beats, sounds, levels,
locators), MIDI files, and the build itself through ableton-live-mcp.

The plan is plain JSON, so it can be read, edited and rebuilt: Live's Trial
cannot save, and a rebuild takes about five seconds.

Live facts the build relies on (measured on Live 12.4, see
docs/07-building-in-live.md):
- Simpler loads samples at Volume -12 dB; the build sets 0 dB.
- Simpler plays a sample at its own pitch on C3 (MIDI 60). The 808 notes
  keep their real MIDI numbers (C1 = 24 here), so Transpose is about +36;
  Live 12.4.6 keeps whole semitones of it (11.99 plays as 11) and whole
  cents of Detune (+-50), so the one-shot's fraction goes to Detune
  (tune_808).
- With one voice (Simpler's voices, a property of the device) and Glide
  Mode "Glide", overlapping notes slide linearly in semitones over exactly
  the Glide Time without starting the sample again; notes 25 ms apart jump.
  The build sends Detune, Glide Mode and Glide Time as parameters and sets
  voices after build_tracks (the battery set each of them this way); a
  built 808 has not been captured in Live yet.
- A new file in the User Library shows up in Live's browser after a few
  seconds; samples load onto Drum Rack pads through the browser.
- Reference stems go in as Arrangement audio clips with warping off, so they
  play at their own speed and line up with notes placed at their seconds,
  once their start (moved to the first transient by Live's warp analysis)
  is put back at 0.
- Locators at fractional beats were refused ("Live did not create it");
  whole beats work.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mido
import numpy as np

from .audio import PRE, SR, attack_time, read, rms_db, transpose, trim_onset
from .render import glide_s, render_808

PADS = {"kick": [36, 35, 47, 45], "snare": [38, 40, 39, 37, 41], "hat": [42, 44, 46, 49, 51],
        "perc": [43, 48, 50, 52, 53]}
# a pitched copy of a sound (detect: events with semitones) plays on a pad of its own, transposed
PITCHED_PADS = list(range(54, 82))
STEMS_MUTED = {"drums", "bass", "vocals"}  # replaced by ours, or not part of the beat
DEFAULT_USER_LIBRARY = Path.home() / "Music" / "Ableton" / "User Library"
GENERATED = Path("Samples") / "AbletonMCP Generated"
# a placeholder for the melodic tracks until a sound is chosen per stem; they
# are built muted, so the reference stems still play as before
MELODIC_INSTRUMENT = "Drift"
LEGATO_S = 0.01  # a legato 808 note's predecessor is held this long past its start (overlapping notes glide)
GLIDE_LEAST = 3  # legato notes a plan needs to put its 808 on one voice with Glide (fewer play as notes of their own)


def no_overlaps(notes: list[dict], longest: float | None) -> list[dict]:
    """Each note at most `longest` beats (None: as given) and ending before the
    next note of its pitch."""
    by_pitch: dict[int, list[dict]] = {}
    for n in notes:
        by_pitch.setdefault(n["pitch"], []).append(n)
    out = []
    for pitch_notes in by_pitch.values():
        pitch_notes.sort(key=lambda n: n["start_time"])
        for a, b in zip(pitch_notes, pitch_notes[1:] + [None]):
            d = longest if longest else a["duration"]
            if b is not None:
                d = min(d, b["start_time"] - a["start_time"] - 0.005)
            out.append(dict(a, duration=round(max(0.05, d), 4)))
    return sorted(out, key=lambda n: (n["start_time"], n["pitch"]))


def pad_pitch(best: dict) -> dict:
    """A match's transposition (matches.json top entry) as Simpler plays it:
    transpose in whole semitones and detune_cents in whole cents (Live
    12.4.6 keeps whole cents of Detune, -50..+50). The speckle search finds
    noisy sounds to a quarter cent: the calibration's detuned snare at -0.70
    semitones is Transpose -1, Detune +30, where Transpose alone played it
    30 cents off. Matches written before match v2 carry whole semitones
    only, and get Detune 0."""
    st = float(best["semitones"])
    whole = int(best.get("transpose", round(st)))
    return {"transpose": whole, "detune_cents": int(round(best.get("detune_cents", 100 * (st - whole))))}


def velocities(gains: list[float], low: int = 35) -> list[int]:
    g = np.asarray(gains, dtype=float)
    if not len(g):
        return []
    ref = np.percentile(g, 90) or 1.0
    return [int(v) for v in np.clip(np.round(127 * g / ref), low, 127)]


def tune_808(own_midi: float, cents: float) -> tuple[int, int]:
    """Simpler's Transpose (whole semitones) and Detune (whole cents) for a
    sample whose sustain sits at own_midi (fractional MIDI), so that MIDI
    note m sounds at m + cents/100 (the reference 808's own tuning): Simpler
    plays the sample at own_midi + (m - 60) + Transpose + Detune/100. Live
    12.4.6 keeps whole semitones of Transpose (11.99 plays as 11) and whole
    cents of Detune (+-50; 11.5 plays as 11), so both are rounded here."""
    exact = 60 - own_midi + cents / 100
    transpose = int(np.floor(exact + 0.5))
    return transpose, int(np.floor(100 * (exact - transpose) + 0.5))


def heard_808(notes: list[dict], shot: np.ndarray, shift0: float) -> float | None:
    """How many cents the one-shot, tuned to settle where the 808 settled,
    sounds below the 808 as analysed: per note (legato notes aside), the
    drop the analysis heard (midi_float over midi_settled) less the
    one-shot's own drop heard at the note's pitch (shift0 + pitch semitones
    over its own) and length, both by the truth's rule (scenarios.sounding);
    the median over the notes at the line's most common written pitch, None
    without notes. The fitted one-shot falls as one exponential, faster
    than the songs' 808s after about 100 ms (v1-1: 42 cents over its
    settled pitch at 150 ms, the beat's 808 117), so tuned to settle where
    the 808 settled the plan's 808 sounded flat. How much depends on the
    rate a note plays the one-shot at, and Detune is one per Simpler: it is
    fitted where most notes are.

    Re-scored on the 808-after4 plans (plan level, mean over the 10 lane
    scenarios; htdemucs, then htdemucs_ft): the guard this replaces (a
    correction only where all notes agreed within 25 cents, interquartile
    range) gave note F1@50 0.849 and 0.826, median |cents| 11.1 and 36.2;
    it rejected bass_low-1's +70 cents (0.333 there; 0.949 with it). The
    median over all notes gave 0.881 and 0.829, but producer-2 on
    htdemucs_ft fell from 0.485 to 0.061 (its notes disagree by 97 cents);
    this rule 0.886 and 0.834, 8.4 and 31.4 cents (producer-2: 0.576)."""
    from .scenarios import drop, sounding, tuning
    curve = tuning(shot)[1]
    use = [n for n in notes if n.get("midi_settled") is not None and (n.get("glide") or {}).get("kind") != "legato"]
    if not use:
        return None
    pitches = [n["pitch"] for n in use]
    common = max(set(pitches), key=lambda p: (pitches.count(p), -p))
    d = [100 * (n["midi_float"] - n["midi_settled"] - sounding(
            lambda t, r=2 ** ((n["pitch"] + shift0) / 12): drop(curve, t * r) / 100, n["end"] - n["t"]))
         for n in use if n["pitch"] == common]
    return float(np.median(d))


def plan_808(analysis: dict, synth: dict | None, stems_dir: str | Path, beats_per_second: float) -> dict:
    """The 808 track: the notes at their written pitches (notes_808 pitch),
    lasting until they end (no floor: a quarter-beat floor over the next
    note at another pitch is a glide once Glide is on), and with a one-shot
    (fit_808's), its Transpose and Detune (tune_808: the one-shot's own
    settled pitch against the 808's tuning, analysis tuning "808"), and each
    note moved so that the one-shot's attack, as Simpler plays it at the
    note's pitch, lands on the note's t_attack (it lands t_attack - t after
    t otherwise: 0.3-13 ms, the drop's own attack against the template's).
    Detune also takes heard_808's correction where the notes agree on it
    (heard_cents; the long notes' tails then settle that much sharp).
    cents is the one-shot's own pitch over root_midi (metadata).

    A legato note (notes_808 glide kind "legato": no attack of its own)
    starts where its glide starts and the note before is held LEGATO_S past
    it: with one voice and Glide Mode "Glide", Live 12.4.6 slides
    overlapping notes linearly in semitones over exactly the Glide Time and
    does not start the sample again; notes 25 ms apart jump. The track then
    carries voices 1, glide_mode and glide_time_ms (the median of the
    glides', one per Simpler) for the build, where the line has GLIDE_LEAST
    legato notes or more: one or two play as notes of their own rather than
    turn the whole track to one voice. Every glide (legato, and the start
    and end glides no note plays yet) is listed under "slides". The
    durations are final here (one voice: a note ends 0.001 beat before the
    next starts, or holds over it to slide), as Glide would turn any other
    overlap into a slide."""
    notes = analysis["notes_808"]
    peaks = np.array([n["peak"] for n in notes]) if notes else np.array([1.0])
    med = float(np.median(peaks)) or 1.0
    track: dict = {}
    shot, shift0 = None, 0.0
    if synth:
        root = synth["root_midi"]
        own = 69 + 12 * math.log2(synth["pitch_hz"] / 440) if synth.get("pitch_hz") else float(root)
        # the 808's tuning as analysed, or the one-shot's where fit_808 found the analysis' reading unsettled
        tuning = 100 * (own - root if synth.get("pitch_from") == "fit"
                        else (analysis.get("tuning") or {}).get("808", own - root))
        semis, detune = tune_808(own, tuning)
        try:
            shot = read(synth["file"])
        except (OSError, RuntimeError):
            shot = None
        heard = None if shot is None else heard_808(notes, shot, semis + detune / 100 - 60)
        if heard is not None:  # sound as the notes were heard, not only settle where they settled
            semis, detune = tune_808(own, tuning + heard)
        # named after the song: the User Library keeps one file per name
        song = Path(stems_dir).resolve().parent.name
        track.update(sample=f"{song} {synth['file_name']}", sample_path=synth["file"], transpose=semis,
                     detune_cents=detune, root_midi=root, cents=round(100 * (own - root), 1),
                     tuning_cents=round(tuning, 1), heard_cents=None if heard is None else round(heard, 1))
        shift0 = semis + detune / 100 - 60
    leads: dict[float, float] = {}

    def lead(pitch: int) -> float:
        shift = round(pitch + shift0, 2)
        if shot is None:
            return 0.0
        if shift not in leads:
            leads[shift] = attack_time(transpose(shot, shift), "808")
        return leads[shift]

    out, slid, slides = [], [], []
    gliding = sum((n.get("glide") or {}).get("kind") == "legato" for n in notes[1:]) >= GLIDE_LEAST
    for n in notes:
        g = n.get("glide") or {}
        legato_note = gliding and g.get("kind") == "legato" and bool(out)
        start = n["t"] if legato_note or "t_attack" not in n or shot is None else n["t_attack"] - lead(n["pitch"])
        out.append({"pitch": n["pitch"], "start_time": round(max(0.0, start) * beats_per_second, 4),
                    "duration": round(max(0.05, (n["end"] - start) * beats_per_second), 4),
                    "velocity": int(np.clip(round(115 * n["peak"] / med), 70, 127))})
        slid.append(legato_note)
        if g:
            slides.append({"start_time": out[-1]["start_time"], "kind": g.get("kind", "legato"),
                           "from_midi": g.get("from_midi"), "to_midi": g.get("to_midi"), "ms": g.get("ms")})
    order = sorted(range(len(out)), key=lambda i: out[i]["start_time"])
    for a, b in zip(order, order[1:]):  # one voice: a note ends where the next starts, or holds over it to slide
        nxt = out[b]["start_time"]
        gap = nxt - out[a]["start_time"]
        if slid[b]:
            out[a]["duration"] = round(gap + LEGATO_S * beats_per_second, 4)
        else:
            out[a]["duration"] = round(max(min(0.01, gap / 2), min(out[a]["duration"], gap - 0.001)), 4)
    legato = [sl["ms"] for sl in slides if sl["kind"] == "legato" and sl["ms"]]
    if gliding and legato:
        track.update(voices=1, glide_mode="Glide", glide_time_ms=round(float(np.median(legato)), 1))
    if slides:
        track["slides"] = slides
    return {"notes": [out[i] for i in order], **track}


def make_plan(analysis: dict, matches: dict, synth: dict | None, stems_dir: str | Path,
              headroom_db: float = 4.0, min_hits: int = 12, min_clean: float = 0.5,
              min_ncc: float = 0.75, melodic: dict | None = None) -> dict:
    """Everything the build needs, decided here so it can be checked first.

    A detected sound gets a Drum Rack pad when it was detected often and
    cleanly and the library has something close (NCC >= min_ncc), or when
    it is a kick or snare (the backbone, played with the closest sound even
    if far). A sound found twice (same_as) shares the first one's pad. The
    pad plays the top match at its pitch: Transpose in whole semitones and
    Detune in whole cents (pad_pitch), as Live keeps them. A match's layers
    (matches.json) are not played yet. With melodic (melodic.json), each
    tonal stem's notes get a MIDI track."""
    bpm = analysis["grid"]["bpm"]
    beats_per_second = bpm / 60
    duration = analysis["source"]["duration_s"]
    length_beats = 4 * math.ceil(duration * beats_per_second / 4)
    pads, drum_notes, used = {}, [], {k: 0 for k in PADS}
    pad_of: dict[str, int] = {}
    # strongest identifications first; a template found twice shares its pad
    order = sorted(analysis["templates"], key=lambda k: (matches.get(k, {}).get("same_as") is not None,
                                                         -(matches.get(k, {}).get("top") or [{"ncc": 0}])[0]["ncc"]))
    for name in order:
        info, m = analysis["templates"][name], matches.get(name)
        if not m or info["detected_hits"] < min_hits or info.get("clean", 1.0) < min_clean:
            continue
        role = m["role"] if m["role"] in PADS else "perc"
        structural = role in ("kick", "snare") and info.get("clean", 1.0) >= 0.7 and info["detected_hits"] >= 16
        if m.get("same_as") in pad_of:
            note = pad_of[m["same_as"]]
        elif m["top"][0]["ncc"] >= min_ncc or structural:
            if used[role] >= len(PADS[role]):
                continue
            note = PADS[role][used[role]]
            used[role] += 1
            best = m["top"][0]
            pads[str(note)] = {"template": name, "path": best["path"], **pad_pitch(best),
                               "ncc": best["ncc"], "verdict": m["verdict"], "role": role}
        else:
            continue  # nothing in the library is close: left out rather than played wrong
        pad_of[name] = note
        hits = analysis["events"][name]
        notes = {0: note, **_pitched_pads(pads, note, {h.get("semitones", 0) for h in hits} - {0})}
        for e, v in zip(hits, velocities([h["gain"] for h in hits])):
            st = e.get("semitones", 0)
            if st not in notes:  # a copy without a free pad is left out, not played unpitched
                continue
            # a note goes where its hit attacks: t_attack, or the template's attack_s after its window starts
            t = e.get("t_attack", e["t"] - PRE / SR + info.get("attack_s", PRE / SR) * 2 ** (-st / 12))
            drum_notes.append({"pitch": notes[st], "start_time": round(t * beats_per_second, 4), "duration": 1.0,
                               "velocity": v})
    plan = {"tempo": bpm, "length_beats": float(length_beats),
            "tracks": {"drums": {"pads": pads, "notes": no_overlaps(drum_notes, 1.0)},
                       "808": plan_808(analysis, synth, stems_dir, beats_per_second)},
            "stems": {}, "locators": [], "levels": {}}
    stems_dir = Path(stems_dir)
    for stem in ("drums", "bass", "other", "piano", "guitar", "vocals"):
        path = stems_dir / f"{stem}.wav"
        if path.is_file():
            plan["stems"][stem] = {"path": str(path), "mute": stem in STEMS_MUTED}
    # sections count bars from the grid's first bar line, in its meter
    grid = analysis["grid"]
    first = max(0.0, grid.get("phase", 0.0)) * beats_per_second
    bar = grid.get("beats_per_bar", 4) * 4 / grid.get("beat_unit", 4)
    for s in analysis.get("sections", []):
        plan["locators"].append([f"{s['name']} ({s['start_bar']})", round(first + bar * (s["start_bar"] - 1), 3)])
    for stem, info in (melodic or {}).get("stems", {}).items():
        notes = [{k: n[k] for k in ("pitch", "start_time", "duration", "velocity")} for n in info["notes"]]
        plan["tracks"].setdefault("melodic", {})[stem] = {"instrument": MELODIC_INSTRUMENT,
                                                          "cents": info.get("tuning_cents", 0),
                                                          "notes": no_overlaps(notes, None)}
    plan["levels"] = estimate_levels(plan, analysis, headroom_db)
    return plan


def _pitched_pads(pads: dict, note: int, semitones) -> dict[int, int]:
    """A pad per pitched copy of the sound on pad `note`: the same sample,
    transposed `semitones` further, on the next free PITCHED_PADS note (or
    the pad that already plays it so). Returns {semitones: note}."""
    out = {}
    base = pads[str(note)]
    for st in sorted(semitones):
        transpose_to = int(base["transpose"] + st)
        same = next((int(k) for k, v in pads.items() if v["path"] == base["path"] and v["transpose"] == transpose_to),
                    None)
        free = same or next((k for k in PITCHED_PADS if str(k) not in pads), None)
        if free is None:
            break
        pads.setdefault(str(free), {**base, "transpose": transpose_to, "semitones": int(st)})
        out[st] = free
    return out


def _render(notes: list[dict], sample_for, n: int, bps: float, cut: bool) -> np.ndarray:
    out = np.zeros(n, np.float32)
    for e in notes:
        x = sample_for(e)
        if x is None:
            continue
        a = int(e["start_time"] / bps * SR)
        if cut:
            x = x[: int(e["duration"] / bps * SR)]
        b = min(n, a + len(x))
        if b > a:
            out[a:b] += x[: b - a] * (e["velocity"] / 127)
    return out


def estimate_levels(plan: dict, analysis: dict, headroom_db: float) -> dict:
    """Track faders (dB) that put our parts at the level of the stems they
    replace: our parts rendered offline at unity (Simpler at 0 dB, velocity as
    a linear level; the 808 as render_plan plays it, Transpose, Detune and
    Glide included, cut without a fade), compared by RMS over the whole
    track."""
    bps = plan["tempo"] / 60
    n = int(analysis["source"]["duration_s"] * SR)
    samples_by_pad = {}
    for note, pad in plan["tracks"]["drums"]["pads"].items():
        try:
            samples_by_pad[int(note)] = transpose(trim_onset(read(pad["path"])),
                                                  pad["transpose"] + pad.get("detune_cents", 0) / 100)
        except Exception:
            samples_by_pad[int(note)] = None
    drums = _render(plan["tracks"]["drums"]["notes"], lambda e: samples_by_pad.get(e["pitch"]), n, bps, False)
    out = {"headroom_db": headroom_db}
    stems = plan["stems"]
    if "drums" in stems:
        ref = rms_db(read(stems["drums"]["path"]))
        out["drums_track_db"] = round(ref - rms_db(drums) - headroom_db, 1)
    track808 = plan["tracks"]["808"]
    if "sample_path" in track808 and "bass" in stems:
        ours = render_808(track808, read(track808["sample_path"]), n, bps, 0, True, 1.0, "linear")
        out["808_track_db"] = round(rms_db(read(stems["bass"]["path"])) - rms_db(ours) - headroom_db, 1)
    out["stems_track_db"] = -headroom_db
    return out


def write_midi(plan: dict, out_dir: str | Path, ppq: int = 480) -> list[Path]:
    """drums.mid and 808.mid (and <stem>.mid per melodic track) at the plan's
    tempo (for import_midi_file)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    parts = {"drums": plan["tracks"]["drums"], "808": plan["tracks"]["808"], **plan["tracks"].get("melodic", {})}
    for key, part in parts.items():
        notes = part["notes"]
        mid = mido.MidiFile(ticks_per_beat=ppq)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(plan["tempo"]), time=0))
        events = []
        for nt in notes:
            on = int(round(nt["start_time"] * ppq))
            events.append((on, 1, mido.Message("note_on", note=nt["pitch"], velocity=nt["velocity"],
                                               channel=9 if key == "drums" else 0)))
            events.append((on + max(1, int(round(nt["duration"] * ppq))), 0,
                           mido.Message("note_off", note=nt["pitch"], velocity=0,
                                        channel=9 if key == "drums" else 0)))
        now = 0
        for tick, _order, msg in sorted(events, key=lambda e: (e[0], e[1])):
            track.append(msg.copy(time=tick - now))
            now = tick
        path = out_dir / f"{key}.mid"
        mid.save(str(path))
        written.append(path)
    return written


# ── The build (needs ableton-live-mcp: pip install "reference-dissector[live]") ──

def simpler_808(t808: dict) -> dict:
    """The 808 Simpler's parameters as build_tracks sets them: Volume 0 dB,
    Transpose and Detune (tune_808: whole semitones and whole cents, which
    Live 12.4.6 keeps as given), and with Glide on (voices 1) Glide Mode
    "Glide" and the Glide Time. These are what render_plan and
    estimate_from_plan assume; voices is a property of the device, not a
    parameter (build sets it)."""
    out = {"Volume": "0.0 dB", "Transpose": float(t808["transpose"]), "Detune": float(t808.get("detune_cents", 0))}
    if glide_s(t808):
        out |= {"Glide Mode": "Glide", "Glide Time": f"{float(t808['glide_time_ms']):.1f} ms"}
    return out


def build_spec(plan: dict) -> list[dict]:
    """The plan as ableton-live-mcp's build_tracks spec: the 808 on a Simpler
    (simpler_808; "voices" for build, which build_tracks does not read),
    the drums on a Drum Rack (one pad per sound), the melodic notes on MIDI
    tracks (muted), the reference stems as audio tracks; Simplers at 0 dB with
    warping off and each sound's transposition (a pad's Transpose and Detune),
    faders from the plan's levels."""
    levels = plan["levels"]
    t808 = plan["tracks"]["808"]
    spec = []
    if "sample" in t808:
        spec.append({"name": "808 (synth)",
                     "instrument": {"sample": t808["sample"], "warping": False},
                     "settings": simpler_808(t808), "voices": t808.get("voices"),
                     "volume_db": round(min(6.0, levels.get("808_track_db", -levels["headroom_db"])), 1),
                     "clips": [{"start": 0.0, "length": float(plan["length_beats"]), "name": "808",
                                "notes": t808["notes"]}] if t808["notes"] else []})
    drums = plan["tracks"]["drums"]
    spec.append({"name": "Drums (library)",
                 "instrument": {"drum_rack": {
                     str(note): {"sample": Path(pad["path"]).name, "warping": False,
                                 "settings": {"Volume": "0.0 dB", "Transpose": float(pad["transpose"]),
                                              "Detune": float(pad.get("detune_cents", 0))}}
                     for note, pad in drums["pads"].items()}},
                 "volume_db": round(min(6.0, levels.get("drums_track_db", -levels["headroom_db"])), 1),
                 "clips": [{"start": 0.0, "length": float(plan["length_beats"]), "name": "Drums",
                            "notes": drums["notes"]}] if drums["notes"] else []})
    for stem, t in plan["tracks"].get("melodic", {}).items():
        spec.append({"name": f"{stem} (notes)", "instrument": t["instrument"], "mute": True,
                     "volume_db": round(min(6.0, levels["stems_track_db"]), 1),
                     "clips": [{"start": 0.0, "length": float(plan["length_beats"]), "name": stem,
                                "notes": t["notes"]}] if t["notes"] else []})
    for stem, info in plan["stems"].items():
        spec.append({"name": f"REF {stem}", "kind": "audio", "mute": bool(info["mute"]),
                     "volume_db": round(min(6.0, levels["stems_track_db"]), 1),
                     "audio_clips": [{"file": info["path"], "start": 0.0, "warping": False}]})
    return spec


async def build(plan: dict, user_library: str | Path = DEFAULT_USER_LIBRARY, headroom: bool = True,
                log=print) -> dict:
    """Build the plan in the open Live set with ableton-live-mcp's build_tracks
    (one call for every track, device, setting, clip and note). Refuses unless
    every track in the set is empty; the set's empty tracks are left as they are."""
    import shutil

    try:
        from MCP_Server import extended_tools as et
        from MCP_Server.server import mcp
    except ImportError as e:
        raise RuntimeError("Building in Live needs ableton-live-mcp: "
                           "pip install 'reference-dissector[live]' (and Live with its Remote Script)") from e

    async def call(tool_name, **args):
        r = await mcp.call_tool(tool_name, args)
        text = "".join(getattr(c, "text", "") for c in r.content)
        if getattr(r, "is_error", False):
            raise RuntimeError(f"{tool_name}: {text}")
        return text

    overview = json.loads(await call("get_session_overview"))
    busy = [t["name"] for t in overview["tracks"]
            if t["devices"] or t["session_clips"] or t["arrangement_clip_count"]]
    if busy:
        raise RuntimeError(f"The set is not empty ({', '.join(busy)}); open a new set first")
    await call("set_song_properties", tempo=plan["tempo"], loop=False)
    t808 = plan["tracks"]["808"]
    if "sample_path" in t808:  # Live's browser lists it a few seconds after it lands in the User Library
        target = Path(user_library).expanduser() / GENERATED / t808["sample"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(t808["sample_path"], target)
    spec = build_spec(plan)
    built = json.loads(await call("build_tracks", tracks=spec))
    report: dict = {"tracks": built["tracks"]}
    problems = {r["track"]: r["problems"] for r in built["tracks"] if r["problems"]}
    if problems:
        report["problems"] = problems
    if plan["locators"]:
        report["locators"] = et.place_locators([(n, float(round(b))) for n, b in plan["locators"]])
    overview = json.loads(await call("get_session_overview"))
    first = next((t["index"] for t in overview["tracks"] if t["name"] == "808 (synth)"), None)
    voices = next((s.get("voices") for s in spec if s["name"] == "808 (synth)"), None)
    if voices and first is not None:  # one voice, so overlapping notes glide (the battery set it this way)
        report["808_voices"] = json.loads(await call("lom_set_property", path=f"live_set tracks {first} devices 0",
                                                     property="voices", value=int(voices)))
    await call("select_in_live", view="arrangement", track=first or 0)
    await call("transport", action="jump_to", time=0.0)
    log(json.dumps(report, ensure_ascii=False, default=str))
    return report
