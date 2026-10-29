"""From an analysis to Ableton Live: a plan (notes in beats, sounds, levels,
locators), MIDI files, and the build itself through ableton-live-mcp.

The plan is plain JSON, so it can be read, edited and rebuilt: Live's Trial
cannot save, and a rebuild takes about five seconds.

Live facts the build relies on (measured on Live 12.4, see
docs/07-building-in-live.md):
- Simpler loads samples at Volume -12 dB; the build sets the 808's to 0 dB
  and each pad's to its volume_db (pad_levels), sent the same way ("-6.5
  dB"); a pad's Volume under 0 dB has not been captured in Live yet.
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
  seconds; samples load onto Drum Rack pads through the browser. A file
  loaded through the browser keeps playing what Live loaded under its name
  after it is overwritten on disk (Live 12.4.6), so the 808's one-shot is
  named after its content (sample_808).
- Reference stems go in as Arrangement audio clips with warping off, so they
  play at their own speed and line up with notes placed at their seconds,
  once their start (moved to the first transient by Live's warp analysis)
  is put back at 0.
- Locators at fractional beats were refused ("Live did not create it");
  whole beats work.
- Velocity sets a Simpler's level, dB = 0.4 x "Vol < Vel" x (velocity - 70)
  - 0.096 over its Volume (Live 12.4.6), and "Vol < Vel" starts at 35 % or
  45 % by how the Simpler was made: the build sets it on every Simpler
  (render.VOL_VEL), and the plan's velocities follow the law (velocities,
  velocity_808). A build with it has not been captured in Live yet.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import mido
import numpy as np

from .audio import ATTACK_BAND, PRE, SR, attack_time, band, read, rms_db, samples, transpose, trim_onset
from .detect import RESIDUAL_BARS, SAME_MS, min_hits_for
from .grid import Grid
from .library import SILENT_DB
from .render import VOL_VEL, as_played, glide_s, pad_settings, render_808, velocity_gain

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
# a hit recurs where another hit of its sound sits at its place of the bar, within RECUR_MS, 1 to
# RECUR_BARS bars away (recurring)
RECUR_MS, RECUR_BARS = 6.0, 8
# a pad's level is read over LEVEL_MS from each hit's attack, as evaluate.level_scores reads a sound (pad_levels);
# a file with LOW_SHARE of its energy under 200 Hz is a kick's (the suite's kick files 0.91-0.99, its tom 0.79, its
# snares 0.53 at most), and on a kick's pad from KICK_SHARE on: a kick with a bright click (69 of the library's 683
# kick files hold under LOW_SHARE, 32 under 0.80 and 8 under KICK_SHARE)
LEVEL_MS, LOW_SHARE, KICK_SHARE = 50.0, 0.85, 0.6


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


def velocities(gains: list[float], low: int = 35, vol_vel: float | None = None) -> list[int]:
    """Velocities for hits of these gains: 127 at their 90th percentile and
    none under low/127 of it. Without vol_vel a velocity is a linear level
    (v/127; melodic's notes, for another instrument). With vol_vel,
    Simpler's "Vol < Vel", a hit plays its gain under Live 12.4.6's law
    (render.velocity_db): 0.4 x vol_vel dB a velocity step, so a hit at
    half the gain is velocity 84 at 0.35, where v/127 gave it 64 and Live
    plays 64 8.8 dB under 127."""
    g = np.asarray(gains, dtype=float)
    if not len(g):
        return []
    ref = np.percentile(g, 90) or 1.0
    if vol_vel is None:
        return [int(v) for v in np.clip(np.round(127 * g / ref), low, 127)]
    db = 20 * np.log10(np.clip(g / ref, low / 127, 1.0))
    return [int(v) for v in np.round(127 + db / (0.4 * vol_vel))]


def velocity_808(level: float, vol_vel: float = VOL_VEL) -> int:
    """An 808 note's velocity for its peak over the line's median peak:
    115 at the median and 0.4 x vol_vel dB a step from there, as Live
    12.4.6 plays them (render.velocity_db), within 70-127."""
    return int(np.clip(round(115 + 20 * np.log10(max(level, 1e-6)) / (0.4 * vol_vel)), 70, 127))


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


def one_shot_digest(path: str | Path) -> str:
    """The first 8 hex digits of a file's SHA-1."""
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:8]


def sample_808(song: str, synth: dict) -> str:
    """The one-shot's file name in the User Library: the song's name, the
    one-shot's, and its content's digest. Live 12.4.6 keeps playing a
    sample loaded through the browser under a name after the file is
    overwritten on disk (ableton-live-mcp's battery, 2026-10-19), so a
    one-shot fitted again gets a name of its own. Without a readable file,
    the names alone."""
    name = Path(synth["file_name"])
    try:
        return f"{song} {name.stem} {one_shot_digest(synth['file'])}{name.suffix}"
    except OSError:
        return f"{song} {name.name}"


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
        # named after the song and the one-shot's content: the User Library keeps one file per name
        song = Path(stems_dir).resolve().parent.name
        track.update(sample=sample_808(song, synth), sample_path=synth["file"], transpose=semis,
                     detune_cents=detune, vol_vel=VOL_VEL, root_midi=root, cents=round(100 * (own - root), 1),
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
                    "velocity": velocity_808(n["peak"] / med)})
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
              headroom_db: float = 4.0, min_hits: int | None = None, min_clean: float = 0.5,
              min_ncc: float = 0.75, melodic: dict | None = None, min_recur: float = 0.5) -> dict:
    """Everything the build needs, decided here so it can be checked first.

    A detected sound gets a Drum Rack pad on evidence that it is one:
    - it was detected min_hits times, by default as often as discover()
      asks of the rarest sound it keeps (min_hits_for at RESIDUAL_BARS: 3,
      or one per 16 bars);
    - one of discover()'s residual pass (found where no template of its band
      plays) recurs: at least min_recur of its hits have another at their
      place of the bar 1 to RECUR_BARS bars away (recurring, on the grid's
      bars as bar_clock reads them: its tempo map, its tempo change);
    - it is clean (sound_clean, over the templates merged into it: at least
      min_clean of its own matched-filter hits at NCC 0.8 or more);
    - the library has something close (NCC >= min_ncc), or it is the
      backbone: a kick or snare discover() found on the places of the beat
      (not in its residual pass), clean (0.7) and played throughout (one hit
      per 4 bars, min_hits_for), played with the closest sound even if far.
    Measured on the 48 suite beats (plan level, mean over scenarios,
    htdemucs; htdemucs_ft in brackets), against 12 hits and the backbone
    rule for any template: open hats F1@20 0.043 -> 0.980 [0.022 -> 0.993]
    (4-8 hits a beat), snares 0.836 -> 0.936 [0.816 -> 0.918] (tempo72's and
    tempo95's 8 hits: 0 -> 1.0), crash, tom and rim 0 -> 0.5 [0.5, 0.45, 0];
    sounds with a pad that play no truth sound (drums.extra) 0.71 -> 0.05
    [0.32 -> 0.02], false hits 22.0 -> 11.7 [10.1 -> 4.6] a beat. Those
    sounds are mostly residual templates on the transients of the suite's
    melodic loop, which plays at its own tempo (scenarios._bleed): 3 of the
    86 such templates recur (m2g), against 209 of the 214 whose moments are
    90 % or more a truth sound's. That is a suite figure: a loop on the
    beat's tempo recurs by construction, and with recurrence waived they
    are 0.37 [0.23] with 16.3 [7.1] false hits a beat (an estimate for such
    songs; by count alone 0.83 and 25.2). Held to recurrence, a grid-found
    sound depended on the tempo being exact and constant: plan drums F1@20
    0.866 -> 0.507 [0.897 -> 0.568] on a tempo rising 3 % over each beat,
    0.637 [0.662] with the bpm 0.2 % off (a forced --bpm), and every sound
    of a second tempo lost its pad; residual templates alone: 0.855
    [0.882], 0.853 [0.883], and with the drift kept as tempo map 0.867
    [0.897], the suite's own 0.866 [0.897] unchanged.
    A sound found twice (same_as) shares the first one's pad; the same file
    at another pitch plays on a pad of its own, transposed so far
    (_pitched_pads: discover() finds a pitched copy as a template too, and
    producer-2's hat 3 semitones down had played at the hat's pitch, its
    own pitched copies 300 cents off). A pad plays a strike once: a hit
    within SAME_MS of one that another template put on the pad is that
    strike found again (on 5 seed-2 suite beats a snare-band template of
    the kick-and-hat moments played the hat's file a second time on 6-15
    of them; false hits 11.7 -> 10.7 a beat, htdemucs). So does the
    backbone: it is planned first, most hits first, and another kick or
    snare template's hit within SAME_MS of one of its hits is that kick or
    snare seen again (a template left with none gets no pad). Ordered by
    NCC alone, swing66-2's snare-band template of 10 kick moments (NCC
    0.895, under the kick's 0.713) took pad 36 and played them a second
    time: kick F1@20 0.903 -> 0.846, swing66's 0.942 -> 0.914. With it,
    plan drums F1@20 0.866 -> 0.866 [0.897 -> 0.898], kick 0.859 -> 0.862
    [0.921 -> 0.925], false hits 10.7 -> 10.3 [4.3 -> 3.7] a beat, and no
    role of any scenario more than 0.005 under the base code's (the
    layers scenario's second kick, on every other kick, had no pad before
    either). The pad plays the
    top match at its pitch: Transpose in whole semitones and Detune in
    whole cents (pad_pitch), as Live keeps them. Velocities play each hit's
    gain under Live's law at the pad's "Vol < Vel" (velocities, VOL_VEL). A
    match's layers (matches.json) are not played yet. With melodic
    (melodic.json), each tonal stem's notes get a MIDI track."""
    bpm = analysis["grid"]["bpm"]
    beats_per_second = bpm / 60
    duration = analysis["source"]["duration_s"]
    length_beats = 4 * math.ceil(duration * beats_per_second / 4)
    pads, drum_notes, used = {}, [], {k: 0 for k in PADS}
    pad_of: dict[str, int] = {}
    leads: dict[tuple, float] = {}
    played: dict[int, list[float]] = {}  # where each pad's hits attack (s)
    g = analysis["grid"]
    grid = Grid(bpm, g.get("phase", 0.0), g.get("beats_per_bar", 4), g.get("beat_unit", 4))
    n = int(round(duration * SR))
    least = min_hits if min_hits is not None else min_hits_for(n, grid, RESIDUAL_BARS)
    backbone = min_hits_for(n, grid)  # one hit per 4 bars
    clean, clock = sound_clean(analysis), bar_clock(g)
    spine = {k: _backbone(v, matches.get(k), clean[k], backbone) for k, v in analysis["templates"].items()}
    # the backbone first, most hits first, then the strongest identifications; a template found twice shares
    # its pad
    order = sorted(analysis["templates"], key=lambda k: (
        matches.get(k, {}).get("same_as") is not None, not spine[k],
        -analysis["templates"][k]["detected_hits"] if spine[k]
        else -(matches.get(k, {}).get("top") or [{"ncc": 0}])[0]["ncc"]))
    struck: dict[str, list[float]] = {}  # where the backbone's kicks and snares attack (s)
    for name in order:
        info, m = analysis["templates"][name], matches.get(name)
        hits = analysis["events"][name]
        if (not m or info["detected_hits"] < least or clean[name] < min_clean
                or (info.get("found", "grid") != "grid"
                    and recurring([e.get("t_attack", e["t"]) for e in hits], clock) < min_recur)):
            continue
        role = m["role"] if m["role"] in PADS else "perc"
        seen = np.array(struck.get(role, []))
        if len(seen) and all(np.min(np.abs(seen - _attack_at(e, info))) <= SAME_MS / 1000 for e in hits):
            continue  # every hit is a kick (snare) the backbone plays: that sound seen again
        shift = 0  # semitones from the pad's pitch to the template's
        if m.get("same_as") in pad_of:
            note = pad_of[m["same_as"]]
            if m["top"][0]["path"] == pads[str(note)]["path"]:
                shift = pad_pitch(m["top"][0])["transpose"] - pads[str(note)]["transpose"]
        elif m["top"][0]["ncc"] >= min_ncc or spine[name]:
            if used[role] >= len(PADS[role]):
                continue
            note = PADS[role][used[role]]
            used[role] += 1
            best = m["top"][0]
            pads[str(note)] = {"template": name, "path": best["path"], **pad_pitch(best), "vol_vel": VOL_VEL,
                               "ncc": best["ncc"], "verdict": m["verdict"], "role": role}
        else:
            continue  # nothing in the library is close: left out rather than played wrong
        pad_of[name] = note
        notes = {0: note, **_pitched_pads(pads, note, {h.get("semitones", 0) + shift for h in hits} - {0})}
        before = {k: np.array(v) for k, v in played.items()}  # the other templates' hits on each pad
        for e, v in zip(hits, velocities([h["gain"] for h in hits], vol_vel=VOL_VEL)):
            st = e.get("semitones", 0)
            if st + shift not in notes:  # a copy without a free pad is left out, not played unpitched
                continue
            # the note starts that much before the hit's attack that the pad's sample attacks there (pad_lead)
            t = _attack_at(e, info)
            pad = notes[st + shift]
            if len(before.get(pad, [])) and np.min(np.abs(before[pad] - t)) <= SAME_MS / 1000:
                continue  # one moment of one sample: the pad plays it already, for a template of that sound
            if len(seen) and np.min(np.abs(seen - t)) <= SAME_MS / 1000:
                continue  # a kick (snare) the backbone plays already
            played.setdefault(pad, []).append(t)
            if spine[name]:
                struck.setdefault(role, []).append(t)
            drum_notes.append({"pitch": pad, "start_time": round(max(0.0, t - pad_lead(pads[str(pad)], leads))
                                                                   * beats_per_second, 4),
                               "duration": 1.0, "velocity": v})
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


def _backbone(info: dict, m: dict | None, clean: float, least: int) -> bool:
    """A kick or snare discover() found on the places of the beat (not in
    its residual pass), clean (0.7) and played throughout (least hits)."""
    return (bool(m) and m["role"] in ("kick", "snare") and info.get("found", "grid") == "grid" and clean >= 0.7
            and info["detected_hits"] >= least)


def _attack_at(e: dict, info: dict) -> float:
    """Where a hit attacks: t_attack, or its template's attack_s after its
    window starts (at the hit's semitones)."""
    return e.get("t_attack", e["t"] - PRE / SR + info.get("attack_s", PRE / SR) * 2 ** (-e.get("semitones", 0) / 12))


def recurring(t, bar, ms: float = RECUR_MS, bars: int = RECUR_BARS) -> float:
    """The share of hits (times in s) with another hit at the same place of
    the bar (within ms) 1 to `bars` bars away: drums are programmed in
    patterns. bar is a bar's length (s), or the song's bars as bar_clock
    gives them, where its tempo drifts or changes. Lags of 1 to 8 bars hold
    patterns of 3 bars (3/4 read in 4/4) and sounds that play every 8 bars
    (rare-1's rim), and a song's length does not raise the chance of a
    stray hit recurring. Only 1 to 4 bars away lost the rim, and the suite's
    snares 0.936 -> 0.918. A drum sound recurs within 4 ms where it does
    within 12: 209 of m2g's 214 templates whose moments are 90 % or more a
    truth sound's, and every PIRAMIT template that recurs. A loop at a
    nearby tempo drifts 8-12 ms a bar (producer-1 tiled six times: 72 hits
    of one recur within 10 ms, none within 8); within 12 ms, 7 of m2g's 86
    templates that play no truth sound recurred, against 3 within 6.

    That makes it a test of the tempo too: on a constant grid 0.2 % off the
    song's tempo (3.5 ms a bar at 136 BPM), the suite's drum sounds stopped
    recurring (plan drums F1@20 0.866 -> 0.637 with every template held to
    it), and on a tempo rising 3 % over the beat, 0.866 -> 0.507."""
    t = np.sort(np.asarray(t, dtype=float))
    if len(t) < 2:
        return 0.0
    s, b = bar if isinstance(bar, tuple) else (np.array([0.0, bar]), np.array([0.0, 1.0]))
    at = _continued(t, s, b)  # in bars
    tol = ms / 1000 * (_continued(t + 1e-3, s, b) - _continued(t - 1e-3, s, b)) / 2e-3
    near = np.zeros(len(t), bool)
    for lag in range(1, bars + 1):
        for target in (at + lag, at - lag):
            i = np.clip(np.searchsorted(at, target), 1, len(at) - 1)
            near |= np.minimum(np.abs(at[i] - target), np.abs(at[i - 1] - target)) <= tol
    return float(near.mean())


def bar_clock(grid: dict) -> tuple[np.ndarray, np.ndarray]:
    """Where an analysis grid's bars fall, as (seconds, bars), piecewise
    linear and continued past both ends: the grid's constant tempo, or the
    beats of its tempo_map where they drift from it (grid.drift), and
    outside the part the grid follows (tempo_change: span_s), the other
    tempo (ratio, the second's over the first's)."""
    g = Grid(grid["bpm"], grid.get("phase", 0.0), grid.get("beats_per_bar", 4), grid.get("beat_unit", 4))
    lines = np.asarray(grid.get("tempo_map") or [], dtype=float)
    if len(lines) >= 2:  # quarter-note lines
        s, b = lines, np.arange(len(lines)) * g.beat / g.bar
    else:
        s, b = np.array([g.phase, g.phase + g.bar]), np.array([0.0, 1.0])
    change = grid.get("tempo_change") or {}
    if change.get("span_s"):
        lo, hi = change["span_s"]
        other = change["ratio"] if change.get("grid_follows") == "first" else 1 / change["ratio"]
        edge = _continued(np.array([lo, hi]), s, b)
        inside = (s > lo) & (s < hi)
        s = np.r_[lo - g.bar, lo, s[inside], hi, hi + g.bar]
        b = np.r_[edge[0] - other, edge[0], b[inside], edge[1], edge[1] + other]
    return s, b


def _continued(x, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """np.interp, continued past both ends along the first and the last piece."""
    x = np.asarray(x, dtype=float)
    y = np.interp(x, xp, fp)
    lo, hi = x < xp[0], x > xp[-1]
    y[lo] = fp[0] + (x[lo] - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0])
    y[hi] = fp[-1] + (x[hi] - xp[-1]) * (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    return y


def sound_clean(analysis: dict) -> dict[str, float]:
    """Each template's clean share over its sound: itself and the templates
    layers.explain merged into it (their hits are its hits), the largest.
    A merged template keeps no hits, so its own share is read off its
    matched-filter peaks over the whole stem (peak_ncc_histogram: those at
    NCC 0.8 or more of those at 0.6 or more). On layers-2 the snare-band
    snare (clean 0.44, under min_clean) holds the low-band one merged into
    it (16 peaks at 0.9 or more, none between 0.4 and 0.9): taken by the
    name alone, no pad played the snare (plan snare F1@20 1.0 -> 0)."""
    merged = (analysis.get("layers") or {}).get("merged") or {}
    own = {}
    for name, info in analysis["templates"].items():
        peaks = info.get("peak_ncc_histogram_0.3_to_1.0")
        if info.get("detected_hits") or not peaks:
            own[name] = info.get("clean", 1.0)
        else:
            own[name] = sum(peaks[5:]) / sum(peaks[3:]) if sum(peaks[3:]) else 0.0
    out = dict(own)
    for name in merged:
        root, seen = name, {name}
        while root in merged and merged[root]["into"] not in seen:  # merged into one merged in turn
            root = merged[root]["into"]
            seen.add(root)
        if root in out:
            out[root] = max(out[root], own[name])
    return out


def pad_lead(pad: dict, cache: dict) -> float:
    """Seconds from a pad's note to its sample's attack as Simpler plays it
    (from start_s, at Transpose plus Detune; audio.attack_time in the pad
    role's band, as evaluate.estimate_from_plan times a pad), 0 for a
    sample that cannot be read. cache keeps one per sample and setting.

    Notes placed at the hits' t_attack played each hit that much late: on
    the 48 suite beats (htdemucs) the plan's timing medians were kick +2.06
    ms, snare +1.13, hat +0.56 (mean over scenarios), and 0.0-0.6 ms more
    than the analysis' on every sound. A note this much earlier attacks
    where the analysis heard the hit: 162 of the 164 sounds with a pad are
    timed within 0.3 ms of the analysis (the other two play fewer of its
    templates)."""
    s = pad_settings(pad)
    key = (pad["path"], s["start_s"], s["transpose"] + s["detune_cents"] / 100, s["role"])
    if key not in cache:
        try:
            cache[key] = attack_time(as_played(read(pad["path"]), s["start_s"], key[2]), s["role"])
        except (OSError, RuntimeError):
            cache[key] = 0.0
    return cache[key]


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


def _render(notes: list[dict], sample_for, gain_for, n: int, bps: float, cut: bool) -> np.ndarray:
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
            out[a:b] += x[: b - a] * gain_for(e)
    return out


def estimate_levels(plan: dict, analysis: dict, headroom_db: float) -> dict:
    """Track faders (dB) that put our parts at the level of the stems they
    replace, and each drum pad's volume_db. Each pad plays where the stems
    hold its sound (pad_levels), the loudest at 0 dB and the others under
    it; a pad without a level (its file cannot be read or peaks under
    SILENT_DB, or none of its hits can be read) stays at 0 dB. The faders
    compare our parts, rendered offline at unity (each pad at its
    volume_db, each velocity at its level under Live's law at the pad's
    vol_vel, as render_plan plays it; the 808 as render_plan plays it,
    Transpose, Detune and Glide included, cut without a fade), with the
    stems by RMS over the whole track: a perfect kit, the truth's sounds at
    their own balance, then plays every sound at the kit level
    evaluate.level_scores scores against. The drum fader stops where the
    loudest pad plays as loud as the stem holds its sound: the RMS match
    goes over that only where the stem holds what no pad plays (on 3
    [7] suite beats; tempo160-1, with no kick pad, sat 13.4 dB from a
    perfect kit under it and 5.1 dB under the stop).

    Without pad levels every pad played its file at 0 dB under one RMS
    fader. On the 48 suite beats (plan level, mean over scenarios,
    htdemucs; htdemucs_ft in brackets) the sounds sat a median 4.09 [3.70]
    dB from a perfect kit (levels.vs_kit_abs_db) and 3.03 [2.84] dB from
    its balance, hats +6.2 [+6.5] and open hats +11.6 [+10.5] dB over the
    kit: the files are louder than the songs play them. With pad levels,
    1.24 [1.31] and 0.57 [0.53] dB (1.61 [1.47] without the stop); per role
    (the median over a beat's sounds) kicks -1.00 [-0.10] dB from the kit,
    snares -0.27 [-0.89], hats -1.17 [-0.77], open hats +0.13 [-0.66].
    With every pad at the truth's own level for its sound and the RMS
    match alone, 1.37 [0.75] and 0.31 [0.04]: our files' tails and the
    sounds a plan misses or adds move the RMS match. With each pad
    at the drum stem's level and the fader at the loudest pad's (the
    stem's own level, no RMS match), 4.05 [1.54]: kicks -5.8 [-1.1] dB
    from the kit, snares +4.0 [+0.9], where htdemucs' drum stem holds them."""
    bps = plan["tempo"] / 60
    n = int(analysis["source"]["duration_s"] * SR)
    out = {"headroom_db": headroom_db}
    stems = plan["stems"]
    track808, held808, fader808 = plan["tracks"]["808"], None, None
    if "sample_path" in track808 and "bass" in stems:
        ours = render_808(track808, read(track808["sample_path"]), n, bps, 0, True, 1.0, "live")
        fader808 = round(rms_db(read(stems["bass"]["path"])) - rms_db(ours) - headroom_db, 1)
        held808 = ours * np.float32(10 ** ((fader808 + headroom_db) / 20))  # where the bass stem holds it
    if "drums" in stems:
        stem = read(stems["drums"]["path"])
        low = None
        if "bass" in stems:
            bass = read(stems["bass"]["path"])
            m = min(len(stem), len(bass))
            low = stem[:m] + bass[:m]
        at = pad_levels(plan, stem, low, held808)
        top = max(at.values(), default=0.0)
        for note, pad in plan["tracks"]["drums"]["pads"].items():
            pad["volume_db"] = round(at[note] - top, 1) + 0.0 if note in at else 0.0
        fader = round(rms_db(stem) - rms_db(_unity_drums(plan, n, bps)) - headroom_db, 1)
        out["drums_track_db"] = min(fader, round(top - headroom_db, 1)) if at else fader
    if fader808 is not None:
        out["808_track_db"] = fader808
    out["stems_track_db"] = -headroom_db
    return out


def _unity_drums(plan: dict, n: int, bps: float) -> np.ndarray:
    """The drum track at unity: each pad's file from its onset at its
    volume_db, each velocity at its level under Live's law at the pad's
    vol_vel."""
    samples_by_pad, gain = {}, {}
    for note, pad in plan["tracks"]["drums"]["pads"].items():
        s = pad_settings(pad)
        gain[int(note)] = (s["vol_vel"], 10 ** (s["volume_db"] / 20))
        try:
            samples_by_pad[int(note)] = transpose(trim_onset(read(pad["path"])),
                                                  pad["transpose"] + pad.get("detune_cents", 0) / 100)
        except Exception:
            samples_by_pad[int(note)] = None
    return _render(plan["tracks"]["drums"]["notes"], lambda e: samples_by_pad.get(e["pitch"]),
                   lambda e: velocity_gain(e["velocity"], "live", gain[e["pitch"]][0]) * gain[e["pitch"]][1],
                   n, bps, False)


def _energies(x: np.ndarray, at: np.ndarray, n: int) -> np.ndarray:
    """x's energy over n samples from each time in at (s)."""
    out = np.zeros(len(at))
    for i, t in enumerate(at):
        a = max(0, int(round(t * SR)))
        out[i] = np.sum(np.square(x[a:a + n], dtype=np.float64))
    return out


def low_share(x: np.ndarray) -> float:
    """The share of x's energy under 200 Hz (audio.BANDS "low")."""
    total = float(np.sum(np.square(x, dtype=np.float64)))
    return float(np.sum(np.square(band(x, "low"), dtype=np.float64))) / total if total > 0 else 0.0


def pad_levels(plan: dict, drums: np.ndarray, low: np.ndarray | None = None, under: np.ndarray | None = None,
               ms: float = LEVEL_MS) -> dict[str, float]:
    """Each pad's level (dB): how far over its own track at Volume 0 dB
    (its notes at their velocities, Live's law) the stems hold its sound.
    Per hit, the energy over `ms` from where the hit attacks, in the pad
    role's band (evaluate.level_scores reads a sound so), the stem's over
    the track's; the median over the sound's hits. A kick (a sound whose
    first pad's file has LOW_SHARE of its energy under 200 Hz, or
    KICK_SHARE on a kick's pad: a kick with a bright click) is read in the
    kick band on `low`, the drum and bass stems together, less `under`
    (the 808 track where the bass stem holds it) at each hit: a separation
    hands part of a kick to the bass stem, by a different amount in every
    song, and the 808 plays under most kicks. It is never read under the
    drum stem's own reading of it, the least the stems hold of it: where
    the bass stem holds the 808 quieter under a kick than `under` plays it
    (a sidechain ducks it, or the 808's fader is a dB high), taking `under`
    off leaves too little of the kick or none, and a kick left without a
    hit had no level. Every other sound is read on the drum stem. The pads
    of one sound (a template's pad and its pitched copies) share one level,
    their hits pooled. A file that peaks under SILENT_DB gets none.

    Measured per pad on the 48 suite beats against where the truth holds
    the sound it plays (median over pads; htdemucs, htdemucs_ft in
    brackets): a pad read on the drum stem sits -1.28 [-1.27] dB from it,
    p10 to p90 -1.95 to -0.15 [-1.94 to -0.08]. A kick read on the drum
    stem sat -10.01 [-3.59] dB from it (-13.74 to -2.55 [-5.27 to -2.02]):
    htdemucs moves much of a kick to the bass stem. On drums + bass the
    808 under it put it at +1.50 [+1.42]; less the plan's 808, -1.30
    [-0.75] (-2.62 to +0.17 [-2.71 to +0.44]). The floor moves the kicks of
    1 [4] suite beats, each toward the truth (v1orig-3: -1.79 -> -0.69
    [-1.93 -> -0.63] dB from a perfect kit); on PIRAMIT, with the plan's
    808 1 or 2 dB louder than its fader, it holds the kick at -16.4 dB, its
    drum stem's reading, where without it 32 and 6 of the kick's 93 hits
    were left and it read -23.2 and -20.3 dB. It does not hold
    a kick the separation splits under a ducked 808: 60 % of a kick in the
    bass stem, its 808 ducked 3 to 20 dB under it, read the kick 2.2 to 8.0
    dB low (synthetic; no suite beat ducks its 808, and PIRAMIT's bass stem
    holds the plan's 808 at -0.3 dB over a kick's first 50 ms, -1.8 dB at
    808 notes with no kick). No kick of the suite had three hits clear of
    every 808 note. Read only on its hits clear of other sounds, a pad sat
    -1.37 [-1.43] dB from the truth, and the kit no closer to a perfect
    one."""
    bps = plan["tempo"] / 60
    pads, notes = plan["tracks"]["drums"]["pads"], plan["tracks"]["drums"]["notes"]
    shots = {}
    for key, pad in pads.items():
        s = pad_settings(pad)
        try:
            x = as_played(read(pad["path"]), s["start_s"], s["transpose"] + s["detune_cents"] / 100)
        except (OSError, RuntimeError):
            continue
        if len(x) and float(np.max(np.abs(x))) >= 10 ** (SILENT_DB / 20):
            shots[key] = (x, s)
    sound = {k: p.get("template", k) for k, p in pads.items()}
    first: dict[str, str] = {}  # a sound is a kick by its first pad's file (its pitched copies come after it)
    for key in shots:
        first.setdefault(sound[key], key)
    kicks = {snd for snd, key in first.items() if low is not None
             and low_share(shots[key][0]) >= (KICK_SHARE if shots[key][1]["role"] == "kick" else LOW_SHARE)}
    w, banded = samples(ms), {}
    heard: dict[str, list[float]] = {}
    alone: dict[str, list[float]] = {}  # a kick on the drum stem alone: the least the stems hold of it
    for key, (x, s) in shots.items():
        kick = sound[key] in kicks
        b, name = ("kick", "low") if kick else (ATTACK_BAND.get(s["role"], "full"), "drums")
        for src, y in ((name, low if kick else drums), ("drums", drums)):
            if (src, b) not in banded:
                banded[src, b] = band(y, b)
        mine = [e for e in notes if str(e["pitch"]) == key]
        at = np.array([e["start_time"] / bps for e in mine]) + attack_time(x, s["role"])
        own = band(_render(mine, lambda e, x=x: x, lambda e, s=s: velocity_gain(e["velocity"], "live", s["vol_vel"]),
                           len(drums), bps, False), b)
        theirs = _energies(banded[name, b], at, w)
        if kick and under is not None:
            if ("under", b) not in banded:
                banded["under", b] = band(under, b)
            theirs = theirs - _energies(banded["under", b], at, w)
        ours = _energies(own, at, w)
        heard.setdefault(sound[key], []).extend(_db(theirs, ours))
        if kick:
            alone.setdefault(sound[key], []).extend(_db(_energies(banded["drums", b], at, w), ours))
    level = {snd: max(float(np.median(v)) for v in (heard[snd], alone.get(snd)) if v)
             for snd in heard if heard[snd] or alone.get(snd)}
    return {key: level[sound[key]] for key in shots if sound[key] in level}


def _db(theirs: np.ndarray, ours: np.ndarray) -> list[float]:
    """10 log10(theirs / ours) (dB) at each hit where both hold energy."""
    ok = (theirs > 0) & (ours > 0)
    return (10 * np.log10(theirs[ok] / ours[ok])).tolist()


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
    Live 12.4.6 keeps as given), "Vol < Vel" (vol_vel, the velocity law's
    amount, a raw 0..1), and with Glide on (voices 1) Glide Mode "Glide"
    and the Glide Time. These are what render_plan and estimate_from_plan
    assume; voices is a property of the device, not a parameter (build
    sets it)."""
    out = {"Volume": "0.0 dB", "Transpose": float(t808["transpose"]), "Detune": float(t808.get("detune_cents", 0)),
           "Vol < Vel": float(t808.get("vol_vel", VOL_VEL))}
    if glide_s(t808):
        out |= {"Glide Mode": "Glide", "Glide Time": f"{float(t808['glide_time_ms']):.1f} ms"}
    return out


def build_spec(plan: dict) -> list[dict]:
    """The plan as ableton-live-mcp's build_tracks spec: the 808 on a Simpler
    (simpler_808; "voices" for build, which build_tracks does not read),
    the drums on a Drum Rack (one pad per sound), the melodic notes on MIDI
    tracks (muted), the reference stems as audio tracks; Simplers with
    warping off, the 808's at 0 dB and each pad's at its volume_db
    (estimate_levels; never over 0 dB, 0 dB without one), each sound's
    transposition (a pad's Transpose and Detune), "Vol < Vel" at the plan's
    vol_vel (VOL_VEL), so velocities play the levels the plan assumes, and
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
                                 "settings": {"Volume": f"{pad_settings(pad)['volume_db'] + 0.0:.1f} dB",
                                              "Transpose": float(pad["transpose"]),
                                              "Detune": float(pad.get("detune_cents", 0)),
                                              "Vol < Vel": float(pad.get("vol_vel", VOL_VEL))}}
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
    every track in the set is empty; the set's empty tracks are left as they are.
    Refuses an 808 one-shot whose content is not the one the plan named
    (sample_808): fit-808 ran again after plan, and Live, having loaded
    that name before, could play the old one-shot; so would the plan's
    Transpose and Detune, made for the old one."""
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

    t808 = plan["tracks"]["808"]
    if "sample_path" in t808 and one_shot_digest(t808["sample_path"]) not in t808["sample"]:
        raise RuntimeError(f"{t808['sample_path']} changed after the plan named it {t808['sample']!r}: "
                           "run refdissect plan again")
    overview = json.loads(await call("get_session_overview"))
    busy = [t["name"] for t in overview["tracks"]
            if t["devices"] or t["session_clips"] or t["arrangement_clip_count"]]
    if busy:
        raise RuntimeError(f"The set is not empty ({', '.join(busy)}); open a new set first")
    await call("set_song_properties", tempo=plan["tempo"], loop=False)
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
