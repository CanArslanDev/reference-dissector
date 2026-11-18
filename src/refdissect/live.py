"""From an analysis to Ableton Live: a plan (notes in beats, sounds, levels,
locators), MIDI files, and the build_tracks spec that blueprint.py writes as
a blueprint for ableton-live-mcp's build_from_blueprint (live-build).

The plan is plain JSON, so it can be read, edited and rebuilt: Live's Trial
cannot save, and a rebuild takes about five seconds.

Live facts the build relies on (measured on Live 12.4, see
docs/07-building-in-live.md):
- Simpler loads samples at Volume -12 dB; the build sets the 808's to 0 dB
  and each pad's to its volume_db (pad_levels), sent the same way ("-6.5
  dB"). Live holds it a display step under the text ("-6.6 dB" holds
  -6.605 dB, "-14.9 dB" -14.95) and plays what it holds within 0.003 dB
  (the live check, 2026-10-29).
- Simpler plays a sample at its own pitch on C3 (MIDI 60). The 808 notes
  keep their real MIDI numbers (C1 = 24 here), so Transpose is about +36;
  Live 12.4.6 keeps whole semitones of it (11.99 plays as 11) and whole
  cents of Detune (+-50), so the one-shot's fraction goes to Detune
  (tune_808).
- With one voice (Simpler's voices, a property of the device) and Glide
  Mode "Glide", overlapping notes slide linearly in semitones over exactly
  the Glide Time without starting the sample again; notes 25 ms apart jump.
  The build sends Detune, Glide Mode and Glide Time as parameters and sets
  voices after build_tracks (the battery set each of them this way); in
  the live check they read back as sent, and a probe's slide played within
  1.04 cents (p90) of render.py's. No plan of the suite or PIRAMIT turns
  Glide on.
- Live's browser loads a sample by its file name: a file loaded through it
  keeps playing what Live loaded under its name after it is overwritten on
  disk (Live 12.4.6), and the browser holds other files of a pad's name.
  The 808's one-shot and every pad's file go into the User Library under a
  name of their content (sample_808, pad_sample); build_from_blueprint
  loads each copy by its path (Live 12.4+; through the browser by that
  name on older Live) and reads back the file each Simpler holds.
- Reference stems go in as Arrangement audio clips with warping off, so they
  play at their own speed and line up with notes placed at their seconds,
  once their start (moved to the first transient by Live's warp analysis)
  is put back at 0. plan --align-bar1 moves the notes and the clips alike
  (onto_live_bars).
- Locators at fractional beats were refused ("Live did not create it");
  whole beats work, but Live moves one off its grid (beat 9 went to 8 in
  4/4, to 12 after 3/4, in the live check): the build sets the plan's
  meter first, build_from_blueprint (ableton-live-mcp v2.4.1+) zooms the
  Arrangement in before it places them, and a locator Live still moves is
  deleted and reported (locators_moved).
- Velocity sets a Simpler's level, dB = 0.4 x "Vol < Vel" x (velocity - 70)
  - 0.096 over its Volume (Live 12.4.6), and "Vol < Vel" starts at 35 % or
  45 % by how the Simpler was made: the build sets it on every Simpler
  (render.VOL_VEL), and the plan's velocities follow the law (velocities,
  velocity_808). In the live check "Vol < Vel" read back 0.35 and the law
  held from velocity 35 to 127, on pads and on the 808.
- Snap, on by default, ends a sample 10-30 ms before its file does, and the
  808's Simpler loaded through the browser filters it (a 22 kHz lowpass
  that follows the note): the build turns both off (LIVE_OFF, simpler_808).
- A pad plays one voice: a hit that comes while its hit before still plays
  fades that one out over 1 ms and starts 1 ms late, and so does an 808
  note that comes while the voice of its pitch is still in its release
  (Live 12.4.6, the live check; render.RETRIGGER_S): such notes start 1 ms
  early in the plan (on_time, plan_808).
- The 808's Simpler ends a note in a release that falls 1.683 dB a ms and
  stops at -60 dB, 35.7 ms on (Ve Release 50 ms, Live's default, which the
  build sends; render.RELEASE_S).
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
from .render import (
    RELEASE_808_MS,
    RETRIGGER_S,
    VOL_VEL,
    as_played,
    chain_semitones,
    chains_808,
    choke_voices,
    glide_s,
    grouped,
    late_808,
    moved_beats,
    pad_hits,
    pad_settings,
    pad_shot,
    pad_track,
    pitch_808,
    read_at,
    render_808,
    settings_808,
    shift_notes,
    takes_over,
    voices_808,
)

PADS = {"kick": [36, 35, 47, 45], "snare": [38, 40, 39, 37, 41], "hat": [42, 44, 46, 49, 51],
        "perc": [43, 48, 50, 52, 53]}
# a pitched copy of a sound (detect: events with semitones) plays on a pad of its own, transposed
PITCHED_PADS = list(range(54, 82))
STEMS_MUTED = {"drums", "bass", "vocals"}  # replaced by ours, or not part of the beat
CLIENTS = 8  # the most MCP clients Live's Remote Script serves at once (AbletonMCP 2.1.0's MAX_CLIENTS)
# Simpler's Snap, on by default, stops a sample 10-30 ms before its file ends (ableton-live-mcp's battery,
# 2026-10-19: a kick 28.8 ms of its 163.5, a hat 9.9 of 71.4; with Snap off it plays to the last frame); in the
# live check (2026-10-29) a kick left its file 21.8 ms before the end (-51.9 dB under its peak) and played on
# with Snap off, and three pads of v1orig-3 and PIRAMIT ended 19.1 to 26.9 ms early. render.py plays every file
# to its last frame: the build sets Snap off on every Simpler
LIVE_OFF = {"Snap": "Off"}
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
FADER_DB = 6.0  # a Live track fader's top: build_spec and render.fader_db cap the plan's faders there
# a sound whose hits clear of every other sound's (by CLEAR_MS) number CLEAR_HITS or more, agree within CLEAR_IQR_DB
# (their interquartile range) and read more than CLEAR_GAP_DB under all its hits is read on those (pad_levels)
CLEAR_MS, CLEAR_HITS, CLEAR_IQR_DB, CLEAR_GAP_DB = 50.0, 3, 3.0, 6.0
# The hats choke their open hats (choked_hats) where the drum stem, from CHOKE_WINDOW_S[0] after the closed hat that
# follows an open hat to CHOKE_WINDOW_S[1] after it (or the next drum hit), holds more than CHOKE_DB under what the
# open hat's tail alone would ring there, on CHOKE_LEAST hits or more and on at least half the hits whose window is
# not masked (louder than that tail by over CHOKE_MASKED_DB: another sound plays there). They go in CHOKE_GROUP.
CHOKE_WINDOW_S, CHOKE_DB, CHOKE_LEAST, CHOKE_MASKED_DB, CHOKE_GROUP = (0.04, 0.15), -6.0, 2, 3.0, 1
# plan --align-bar1 (off by default) moves a plan whose grid's first bar line lies more than ALIGN_MS from 0 s
# onto Live's bars (onto_live_bars): Live's bar 1 starts at 0 s, so the plan's bars lie that far off Live's
ALIGN_MS = 20.0


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


def file_db(path: str | Path) -> float:
    """The scale (dB, 0 or under) that brings a file's peak, over its
    channels at its own rate (as Simpler plays it), to full scale: 0 for a
    file at or under it, or one that cannot be read. It is rounded down to
    the 0.1 dB step of a pad's Volume, so the scaled copy peaks at full
    scale or up to 0.1 dB under it (scale_files_to_full moves Volumes and
    the drum fader by whole steps).

    A float file can peak over full scale: 1 % of the library's 8,854
    entries peak over +6.4 dBFS, the loudest +19.4, and 7 of the 227 pads
    of the m5 plans (8 of 229 on htdemucs_ft) play one, at +1.3 to +7.3
    dBFS. The plan's levels read each file at its own level, so the drum
    fader made up for such a file (each of those 7 plans' faders -13.5 to
    -23.5 dB), and Live plays a float file over full scale without
    clipping it (v1orig-3's kick and snare at +5.1 and +1.6 dBFS summed to
    the Drums track within -123 dB in the live check), but every Simpler,
    pad and the Drum Rack ahead of the fader then carried peaks over full
    scale. The plan gives such a pad file_db, plays its file that much
    lower everywhere (render.pad_hits) and makes it up in the pad's Volume
    and the drum fader (scale_files_to_full), and the build copies it
    scaled (the blueprint's gain_db)."""
    try:
        import soundfile as sf

        x, _ = sf.read(str(path), dtype="float64", always_2d=True)
    except Exception:
        return 0.0
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    return -math.ceil(round(20 * math.log10(peak) * 10, 6)) / 10 + 0.0 if peak > 1.0 else 0.0


def pad_sample(path: str | Path, gain_db: float = 0.0) -> str:
    """A pad's file name in the User Library: the file's own name (its
    spaces run together) and its content's digest, as sample_808 names the
    808's one-shot; with gain_db (a pad's file_db, under 0) that gain after
    the digest ("-7.31dB"), for the copy scaled by it (the blueprint's
    gain_db). build_tracks (ableton-live-mcp 2.2.0) loads a pad
    through Live's browser by file name, and the browser holds other files
    of a pad's name: in the live check (Live 12.4.6, 2026-10-29) v1orig-3's
    open-hat pad played an 808 of the same name and PIRAMIT's closed-hat pad
    an FX hit, and a one-character name matched so many items that the file
    was not among the 20 the search keeps: build_tracks raised after 40 s
    with the set half built. 4 of the 466 pads of the suite's and PIRAMIT's
    97 plans named a file that other files of other content share. Without
    a readable file, the file's name."""
    p = Path(path)
    gain = f" {gain_db:.2f}dB" if gain_db < 0 else ""
    try:
        return f"{' '.join(p.stem.split())} {one_shot_digest(p)}{gain}{'.wav' if gain else p.suffix}"
    except OSError:
        return f"{p.stem}{gain}.wav" if gain else p.name


def heard_808(notes: list[dict], shot: np.ndarray, shift0: float, starts: list[float] | None = None) -> float | None:
    """How many cents the one-shot, tuned to settle where the 808 settled,
    sounds below the 808 as analysed: per note (legato notes aside), the
    drop the analysis heard (midi_float over midi_settled) less the
    one-shot's own drop heard at the note's pitch (shift0 + pitch semitones
    over its own) and length, both by the truth's rule (scenarios.sounding);
    the median over the notes at the line's most common written pitch, None
    without notes. Each note is read from where the plan starts it (starts,
    plan_808's; by default the note's onset) to its end, as
    estimate_from_plan plays it. A one-shot that falls as one exponential
    (synth808's fitted voice) had settled where the songs' 808s still hold
    100 cents or so over their settled pitch, so tuned to settle where the
    808 settled the plan's 808 sounded flat; how much depended on the rate
    and length of each note, and Detune is one per Simpler: it is fitted
    where most notes are. bass_low-1's notes under 120 ms, 120-250 and
    250-600 took 47.1, 123.4 and 82.6 cents, and the long ones played 42-52
    flat. fit_808's one-shot now follows the template's own periods, laid
    out as the template is (synth808.tracked_shot): those corrections are
    -0.1, -0.2 and -0.1 (htdemucs_ft: 3.8, -0.1, -0.2), the plan's notes
    sound 0.6, 0.7 and 0.8 cents from the analysis' by length (4.5, 1.3,
    1.1), and the line's correction lies within 8 cents of 0 on 95 of the
    suite's 96 beat runs (37 and 43 before; |median| 4.4 and 3.2 -> 1.15
    and 0.6 cents), the 96th a template whose settled pitch was read on its
    tail. Read from the onset while the one-shot started 7 ms later, they
    had been 0.1, -0.1, -0.1 where the plan sounded 37.7, 2.0 and 2.0 over
    the analysis.

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
    use = [(n, s) for n, s in zip(notes, starts if starts is not None else [n["t"] for n in notes])
           if n.get("midi_settled") is not None and (n.get("glide") or {}).get("kind") != "legato"]
    if not use:
        return None
    pitches = [n["pitch"] for n, _ in use]
    common = max(set(pitches), key=lambda p: (pitches.count(p), -p))
    d = [100 * (n["midi_float"] - n["midi_settled"] - sounding(
            lambda t, r=2 ** ((n["pitch"] + shift0) / 12): drop(curve, t * r) / 100, n["end"] - s))
         for n, s in use if n["pitch"] == common]
    return float(np.median(d))


def plan_808(analysis: dict, synth: dict | None, stems_dir: str | Path, beats_per_second: float) -> dict:
    """The 808 track: the notes at their written pitches (notes_808 pitch),
    lasting until they end (no floor: a quarter-beat floor over the next
    note at another pitch is a glide once Glide is on), and with a one-shot
    (fit_808's), its Transpose and Detune (tune_808: the one-shot's own
    settled pitch against the 808's tuning, analysis tuning "808"), and each
    note moved so that the one-shot's attack, as Simpler plays it at the
    note's pitch, lands on the note's t_attack (it lands t_attack - t after
    t otherwise: 0.3-13 ms, the drop's own attack against the template's;
    a one-shot laid out as its template, fit_808's, starts there at t).
    Detune also takes heard_808's correction, the notes read from those
    starts (heard_cents; the long notes' tails then settle that much
    sharp).
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
    overlap into a slide. A note that Live starts late, as it comes while
    the voice of the note before at its pitch still sounds in its release
    (render.late_808; with Glide on, only where render.GLIDE_TAKEOVER, and
    then the one voice's chain before), starts that much earlier, its end
    kept, as a pad's note does (on_time); the note before it then ends
    0.001 beat before it, as any."""
    notes = analysis["notes_808"]
    peaks = np.array([n["peak"] for n in notes]) if notes else np.array([1.0])
    med = float(np.median(peaks)) or 1.0
    track: dict = {}
    shot, shift0 = None, 0.0
    leads: dict[float, float] = {}

    def lead(pitch: int) -> float:
        shift = round(pitch + shift0, 2)
        if shot is None:
            return 0.0
        if shift not in leads:
            leads[shift] = attack_time(transpose(shot, shift), "808")
        return leads[shift]

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
        shift0 = semis + detune / 100 - 60
        # each note heard from where it starts below: its attack on t_attack
        starts = [n["t_attack"] - lead(n["pitch"]) if "t_attack" in n else n["t"] for n in notes]
        heard = None if shot is None else heard_808(notes, shot, shift0, starts)
        if heard is not None:  # sound as the notes were heard, not only settle where they settled
            semis, detune = tune_808(own, tuning + heard)
        # named after the song and the one-shot's content: the User Library keeps one file per name
        song = Path(stems_dir).resolve().parent.name
        track.update(sample=sample_808(song, synth), sample_path=synth["file"], transpose=semis,
                     detune_cents=detune, vol_vel=VOL_VEL, root_midi=root, cents=round(100 * (own - root), 1),
                     tuning_cents=round(tuning, 1), heard_cents=None if heard is None else round(heard, 1))
        shift0 = semis + detune / 100 - 60

    out, slid, glides = [], [], []
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
            glides.append((len(out) - 1, g))
    order = sorted(range(len(out)), key=lambda i: out[i]["start_time"])
    _one_voice(out, order, slid, beats_per_second)
    if shot is not None and takes_over(gliding):  # each note that Live starts late, that much earlier (on_time)
        lengths: dict[int, float] = {}
        heads = [k for k, i in enumerate(order) if not slid[i]]  # with Glide, each chain's voice starts there
        chains = [[out[i] for i in order[a:b]] for a, b in zip(heads, heads[1:] + [len(order)])]
        for nt, late in zip([c[0] for c in chains], late_808(voices_808(chains), beats_per_second, lambda p:
                            lengths.setdefault(p, len(transpose(shot, p + shift0)) / SR), gliding)):
            if late:  # its end stays
                step = min(nt["start_time"], round(RETRIGGER_S * beats_per_second, 4))
                nt["start_time"], nt["duration"] = round(nt["start_time"] - step, 4), round(nt["duration"] + step, 4)
        _one_voice(out, order, slid, beats_per_second)
    slides = [{"start_time": out[i]["start_time"], "kind": g.get("kind", "legato"), "from_midi": g.get("from_midi"),
               "to_midi": g.get("to_midi"), "ms": g.get("ms")} for i, g in glides]
    legato = [sl["ms"] for sl in slides if sl["kind"] == "legato" and sl["ms"]]
    if gliding and legato:
        track.update(voices=1, glide_mode="Glide", glide_time_ms=round(float(np.median(legato)), 1))
    if slides:
        track["slides"] = slides
    return {"notes": [out[i] for i in order], **track}


def _one_voice(out: list[dict], order: list[int], slid: list[bool], bps: float) -> None:
    """One voice: each note (in `order`) ends 0.001 beat before the next
    starts, or holds LEGATO_S over it where that one slides on (slid)."""
    for a, b in zip(order, order[1:]):
        gap = out[b]["start_time"] - out[a]["start_time"]
        if slid[b]:
            out[a]["duration"] = round(gap + LEGATO_S * bps, 4)
        else:
            out[a]["duration"] = round(max(min(0.01, gap / 2), min(out[a]["duration"], gap - 0.001)), 4)


def make_plan(analysis: dict, matches: dict, synth: dict | None, stems_dir: str | Path,
              headroom_db: float = 4.0, min_hits: int | None = None, min_clean: float = 0.5,
              min_ncc: float = 0.75, melodic: dict | None = None, min_recur: float = 0.5,
              chokes: bool = False, scale_files: bool = True, live_sr: int | None = None,
              align_bar1: bool = False) -> dict:
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
    0.866 -> 0.506 [0.897 -> 0.568] on a tempo rising 3 % over each beat,
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
    0.895, over the kick's 0.713) took pad 36 and played them a second
    time: kick F1@20 0.903 -> 0.846, swing66's 0.942 -> 0.914. With it,
    plan drums F1@20 0.866 -> 0.866 [0.897 -> 0.898], kick 0.859 -> 0.862
    [0.921 -> 0.925], false hits 10.7 -> 10.3 [4.3 -> 3.7] a beat, and no
    role of any scenario more than 0.005 under the base code's (the
    layers scenario's second kick, on every other kick, had no pad before
    either). The pad plays the
    top match at its pitch: Transpose in whole semitones and Detune in
    whole cents (pad_pitch), as Live keeps them. Velocities play each hit's
    gain under Live's law at the pad's "Vol < Vel" (velocities, VOL_VEL). A
    note that Live starts late, as it comes while its pad's hit before
    still plays, goes that much earlier (on_time). A
    match's layers (matches.json) are not played yet. With scale_files, a
    pad whose file peaks over full scale gets file_db, the scale that
    brings it to full scale: it plays that much lower in every estimate
    and render of the plan, its Volume and the drum fader make it up
    (scale_files_to_full), and the build plays a copy scaled so (the
    blueprint's gain_db). With melodic
    (melodic.json), each tonal stem's notes get a MIDI track. The plan
    carries the grid's meter (beats_per_bar, beat_unit), which the build
    sets as Live's time signature, and lasts whole bars of it. With chokes
    (off by default: on the suite it moved producer-1 alone, its open hat
    0.05 dB closer to a perfect kit and the levels' balance 0.5117 ->
    0.5120 dB, htdemucs; nothing on htdemucs_ft),
    the hat pads (an open hat's, by its match, and the other "hat" pads)
    go in one choke group (CHOKE_GROUP) where the drum stem shows the
    closed hats cutting the open hats (choked_hats; its reading under the
    drum track's "chokes"), as Live's choke group plays them. A pitched
    copy's pad stays out: in one group the pads of a roll's pitched copies
    cut each other, which the songs' rolls do not (on the suite, with them
    in: full's hats 0.11 dB further under a perfect kit, its balance 1.165
    -> 1.12 dB but htdemucs_ft's 1.41 -> 1.50).

    With live_sr (the audio device's rate, render.LIVE_SR measured; off by
    default) the plan carries it, and its levels (estimate_levels) and every
    render of it play each pad through Simpler's resampler at that rate
    (render.pad_shot), which takes highs off as Live 12.4.16 does. On the
    suite (tags hf-1, hf-1-ft against m5's plans rendered the same way,
    hf-0, hf-0-ft) the distance from a perfect kit went 1.267 -> 1.213 dB
    [1.311 -> 1.172] and hats -1.26 -> -0.94 dB [-0.93 -> -0.57], the
    balance 0.472 -> 0.506 dB [0.500 -> 0.487]. Within each stem group
    (evaluate.level_scores' balance_in_group_db; the same plans scored
    again, hf-0g -> hf-1g) it is worse on both models, a median over the
    48 beats of 0.190 -> 0.240 dB [0.195 -> 0.245]: off.

    With align_bar1 (off by default) the plan is then moved onto Live's
    bars (onto_live_bars), the levels read before, at the song's seconds."""
    bpm = analysis["grid"]["bpm"]
    beats_per_second = bpm / 60
    duration = analysis["source"]["duration_s"]
    g = analysis["grid"]
    meter = {"beats_per_bar": int(g.get("beats_per_bar", 4)), "beat_unit": int(g.get("beat_unit", 4))}
    bar = meter["beats_per_bar"] * 4 / meter["beat_unit"]  # a bar of the meter in the tempo's (quarter) beats
    length_beats = bar * math.ceil(duration * beats_per_second / bar)
    pads, drum_notes, used = {}, [], {k: 0 for k in PADS}
    pad_of: dict[str, int] = {}
    leads: dict[tuple, float] = {}
    played: dict[int, list[float]] = {}  # where each pad's hits attack (s)
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
    choke = None
    if chokes and (Path(stems_dir) / "drums.wav").is_file():
        opens = {k for k, p in pads.items()
                 if (matches.get(p["template"], {}).get("top") or [{}])[0].get("role") == "open_hat"}
        choke = choked_hats(pads, played, opens, read(Path(stems_dir) / "drums.wav"))
        if choke["choked"]:
            for p in pads.values():
                if p.get("role") == "hat" and "semitones" not in p:  # a pitched copy's pad stays out
                    p["choke_group"] = CHOKE_GROUP
    on_time(drum_notes, pads, beats_per_second, {})
    plan = {"tempo": bpm, **meter, "length_beats": float(length_beats),
            "tracks": {"drums": {"pads": pads, "notes": no_overlaps(drum_notes, 1.0),
                                 **({"chokes": choke} if choke else {})},
                       "808": plan_808(analysis, synth, stems_dir, beats_per_second)},
            "stems": {}, "locators": [], "levels": {}, **({"live_sr": int(live_sr)} if live_sr else {})}
    stems_dir = Path(stems_dir)
    for stem in ("drums", "bass", "other", "piano", "guitar", "vocals"):
        path = stems_dir / f"{stem}.wav"
        if path.is_file():
            plan["stems"][stem] = {"path": str(path), "mute": stem in STEMS_MUTED}
    # sections count bars from the grid's first bar line, in its meter
    first = max(0.0, g.get("phase", 0.0)) * beats_per_second
    for s in analysis.get("sections", []):
        plan["locators"].append([f"{s['name']} ({s['start_bar']})", round(first + bar * (s["start_bar"] - 1), 3)])
    for stem, info in (melodic or {}).get("stems", {}).items():
        notes = [{k: n[k] for k in ("pitch", "start_time", "duration", "velocity")} for n in info["notes"]]
        plan["tracks"].setdefault("melodic", {})[stem] = {"instrument": MELODIC_INSTRUMENT,
                                                          "cents": info.get("tuning_cents", 0),
                                                          "notes": no_overlaps(notes, None)}
    plan["levels"] = estimate_levels(plan, analysis, headroom_db)
    if scale_files:
        scale_files_to_full(plan)
    if align_bar1:
        onto_live_bars(plan, analysis)
    return plan


def bar1_move(phase: float, bar_s: float, within_ms: float = ALIGN_MS) -> float:
    """Seconds that move a grid's first bar line (phase, s from the audio's
    start) onto a bar line of Live's ruler, whose bar 1 starts at 0 s: a bar
    less the phase where the line lies after 0 s (onto bar 2: bar 1 holds
    the pickup), the phase's own size where it lies before (onto bar 1),
    and 0 within within_ms of 0 s. Never earlier, so every note keeps its
    place and the stems need no cut. grid.estimate puts the line from
    BAR_SLACK (50 ms) before the audio to under a bar into it."""
    if phase > within_ms / 1000:
        return (bar_s - phase) % bar_s
    if phase < -within_ms / 1000:
        return -phase
    return 0.0


def onto_live_bars(plan: dict, analysis: dict, within_ms: float = ALIGN_MS) -> None:
    """plan --align-bar1: make_plan's plan, at the song's seconds, moved
    later in place by bar1_move of the analysis' first bar line, in the
    plan's beats (4 decimals), so that the grid's bar lines fall on Live's,
    which count bar 1 from 0 s (a plan at the song's seconds has its bars
    that far off Live's: full's first bar line, 0.4 s in, 400 ms).
    - Every drum, 808 and melodic note and the 808's slides move by it, and
      so do the reference stems: build_spec starts their clips there
      (moved_beats). A note make_plan held at 0 s, its sample's lead-in
      starting before the audio, moves with the rest.
    - The plan ends on the first bar line of Live's at or after the moved
      song's end.
    - The locators are the sections' starts counted from the moved first
      bar line, so on Live's bar lines: from bar 2 where the line came
      after 0 s, from bar 1 where it came before (make_plan had put them
      there already, the line itself at 0 s; the notes and stems now meet
      them).
    Within within_ms of 0 s nothing moves. The plan records align_bar1:
    the first bar line (s), the move (s and beats) and the bar of Live's it
    lands on. render.unaligned takes the move back for whatever plays the
    plan against the reference (render_plan, evaluate.estimate_from_plan,
    candidates)."""
    bps, bar = plan["tempo"] / 60, bar_beats(plan)
    phase = float(analysis["grid"].get("phase", 0.0))
    moved = round(bar1_move(phase, bar / bps, within_ms) * bps, 4) + 0.0
    if moved:
        shift_notes(plan, moved)
        end = analysis["source"]["duration_s"] * bps + moved
        plan["length_beats"] = float(bar * math.ceil(end / bar - 1e-9))
        first = phase * bps + moved  # the grid's first bar line, on a bar line of Live's
        plan["locators"] = [[f"{s['name']} ({s['start_bar']})", round(first + bar * (s["start_bar"] - 1), 3) + 0.0]
                            for s in analysis.get("sections", [])]
    plan["align_bar1"] = {"first_bar_line_s": phase, "moved_s": round(moved / bps, 6), "moved_beats": moved,
                          "lands_on_bar": int(round((phase * bps + moved) / bar)) + 1}


def scale_files_to_full(plan: dict) -> None:
    """Gives each pad whose file peaks over full scale its file_db, and
    takes it back in the pad's Volume, as far as 0 dB; what is left the
    drum fader takes, every pad's Volume as much lower, as far as FADER_DB.
    Each pad's fader + Volume + file_db stays what estimate_levels set, so
    the plan sounds as before: the levels read the file at its own level.
    file_db is in the Volumes' 0.1 dB steps (file_db()), so are the moves.
    Read with the scaled file instead, the Volumes' rounding moved the
    sounds of 5 [6] plans by up to 0.1 dB either way (htdemucs
    [htdemucs_ft]), with 0.01 dB steps and with 0.1 dB steps alike, as a
    pad that becomes the loudest moves every other pad's rounding."""
    pads, levels = plan["tracks"]["drums"]["pads"], plan["levels"]
    gains = {k: file_db(p["path"]) for k, p in pads.items()}
    hot = {k: g for k, g in gains.items() if g < 0}
    if not hot:
        return
    fader = levels.get("drums_track_db")
    raised = {k: pad_settings(pads[k])["volume_db"] - g for k, g in hot.items()}
    over = round(max(0.0, *raised.values()), 1)
    if fader is not None:  # the fader takes what a pad's Volume cannot, as far as its top
        over = round(max(0.0, min(over, FADER_DB - fader)), 1)
        levels["drums_track_db"] = round(fader + over, 1) + 0.0
    else:
        over = 0.0
    for k, pad in pads.items():
        volume = round((raised[k] if k in hot else pad_settings(pad)["volume_db"]) - over, 1)
        pad["volume_db"] = min(0.0, volume) + 0.0
        if k in hot:
            pad["file_db"] = round(hot[k] + max(0.0, volume), 1) + 0.0


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
    it), and on a tempo rising 3 % over the beat, 0.866 -> 0.506."""
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
    the 48 suite beats (htdemucs, m2g-vel) the plan's timing medians were
    kick +2.06 ms, snare +1.13, hat +0.56 (mean over scenarios), and per
    sound with a pad 0.0-5.9 ms more than the analysis' (median 0.5 ms
    over 164 sounds; kicks 2.5). A note this much earlier attacks where
    the analysis heard the hit: 162 of those 164 sounds are timed within
    0.3 ms of the analysis (the other two play fewer of its templates),
    and 218 of the 219 with a pad at m3 (the other 0.76 ms)."""
    s = pad_settings(pad)
    key = (pad["path"], s["start_s"], s["transpose"] + s["detune_cents"] / 100, s["role"])
    if key not in cache:
        try:
            cache[key] = attack_time(as_played(read(pad["path"]), s["start_s"], key[2]), s["role"])
        except (OSError, RuntimeError):
            cache[key] = 0.0
    return cache[key]


def pad_voice_s(pad: dict, cache: dict) -> float:
    """How long Live plays a pad's hit (s): its file as Simpler plays it
    (from start_s, at Transpose plus Detune) to the last frame, Snap off
    (LIVE_OFF); 0 for a file that cannot be read. cache keeps one per
    sample and setting."""
    s = pad_settings(pad)
    key = (pad["path"], s["start_s"], s["transpose"] + s["detune_cents"] / 100)
    if key not in cache:
        try:
            cache[key] = len(as_played(read(pad["path"]), s["start_s"], key[2])) / SR
        except (OSError, RuntimeError):
            cache[key] = 0.0
    return cache[key]


def on_time(notes: list[dict], pads: dict, bps: float, cache: dict) -> None:
    """Each drum note that Live starts RETRIGGER_S late moved that much
    earlier, so that its sample still starts where the hit's did: a pad
    plays one voice, and a note that comes before the pad's hit before has
    played its file (pad_voice_s) cuts that one and starts late
    (render.pad_hits). Placed where its sample is to start, a hit is late
    where it comes within the file of the hit before; moved, it still is,
    and Live starts it on its place (render.late_starts gives the same
    hits on the moved notes). On a choke group's pads a hit before that
    another pad of the group has cut no longer sounds, and the hit after
    it starts on time (render.choke_voices)."""
    group_of = grouped({k: p.get("choke_group") for k, p in pads.items()})
    by_pad: dict[str, list[dict]] = {}
    for e in notes:
        by_pad.setdefault(str(e["pitch"]), []).append(e)
    for key, mine in by_pad.items():
        if key in group_of:
            continue
        mine.sort(key=lambda e: e["start_time"])
        start = np.array([e["start_time"] / bps for e in mine])
        for e in np.array(mine)[1:][start[1:] < start[:-1] + pad_voice_s(pads[key], cache)]:
            e["start_time"] = round(max(0.0, e["start_time"] - RETRIGGER_S * bps), 4)
    for group in sorted(set(group_of.values())):
        mine = sorted((e for key in by_pad if group_of.get(key) == group for e in by_pad[key]),
                      key=lambda e: e["start_time"])
        keys = [str(e["pitch"]) for e in mine]
        late, _ = choke_voices(np.array([e["start_time"] / bps for e in mine]),
                               [pad_voice_s(pads[k], cache) for k in keys], keys, group_of)
        for e in np.array(mine, dtype=object)[late]:
            e["start_time"] = round(max(0.0, e["start_time"] - RETRIGGER_S * bps), 4)


def _energy(x: np.ndarray, a: int, b: int) -> float:
    return float(np.sum(np.square(x[max(0, a):max(0, b)], dtype=np.float64)))


def choked_hats(pads: dict, played: dict, opens: set, drums: np.ndarray) -> dict:
    """Whether the closed hats choke the open hats, read off the drum stem
    (a producer puts them in one choke group; Live then cuts an open hat
    where the next closed hat starts, render.choke_voices). For each open
    hat followed by a closed hat inside its file (opens: the pads whose
    match is an open hat, the closed ones the other "hat" pads; played: each
    pad's attacks, s), the stem's energy in the hat band from
    CHOKE_WINDOW_S[0] after the closed hat to CHOKE_WINDOW_S[1] after it
    (or 2 ms before the next drum hit; 10 ms at least) against what the
    open hat's tail alone rings there (its file as played, at the level the
    stem holds its first 30 ms, or up to the closed hat). Other sounds only
    add energy: a window CHOKE_MASKED_DB or more over the tail is masked
    and left out, and one CHOKE_DB or more under it is a cut tail. Returns
    hits (windows read), cut and choked (CHOKE_LEAST cut tails or more and
    half the windows or more).

    On the suite's m5 analyses (htdemucs; htdemucs_ft in brackets), where
    the rolls, producer and full scenarios choke every open hat that a
    closed hat follows inside its file: windows cut, of those read,
    rolls-1 6 of 8 [none read], rolls-2 4 of 4 [4 of 4], full-1 6 of 7 [7
    of 7], full-2 3 of 4 [4 of 4], producer-1 5 of 10 [none read],
    producer-2 2 of 6 [2 of 6]: 5 [3] of the 6 choke beats choked. On the
    other 42 beats, which choke none, no window read cut [2 of melodic-1's
    5, 3 of rare-1's 13]: none choked."""
    closed = [k for k, p in pads.items() if p.get("role") == "hat" and k not in opens]
    out = {"hits": 0, "cut": 0, "choked": False}
    if not opens or not closed:
        return out
    stem = band(drums, "hat")
    shot, attack = {}, {}
    for k in list(opens) + closed:
        s = pad_settings(pads[k])
        try:
            x = as_played(read(pads[k]["path"]), s["start_s"], s["transpose"] + s["detune_cents"] / 100)
        except (OSError, RuntimeError):
            continue
        shot[k], attack[k] = band(x, "hat"), int(round(attack_time(x, "hat") * SR))
    hats = sorted(t for k in closed if k in shot for t in played.get(int(k), []))
    every = np.sort(np.array([t for v in played.values() for t in v] + [np.inf]))
    for o in (k for k in opens if k in shot):
        x, ax = shot[o], attack[o]
        for a0 in played.get(int(o), []):
            a1 = next((t for t in hats if t > a0 + 0.002), None)
            if a1 is None or ax + (a1 - a0) * SR >= len(x) - 1:
                continue  # no closed hat inside the open hat's file
            w0 = a1 + CHOKE_WINDOW_S[0]
            w1 = min(a1 + CHOKE_WINDOW_S[1], every[np.searchsorted(every, a1 + 0.005, "right")] - 0.002)
            if w1 - w0 < 0.01:
                continue
            head = int(min(a1 - a0, 0.03) * SR)
            mine = _energy(x, ax, ax + head)
            theirs = _energy(stem, int(round(a0 * SR)), int(round(a0 * SR)) + head)
            if mine <= 0 or theirs <= 0:
                continue
            tail = theirs / mine * _energy(x, ax + int(round((w0 - a0) * SR)), ax + int(round((w1 - a0) * SR)))
            heard = _energy(stem, int(round(w0 * SR)), int(round(w1 * SR)))
            if tail <= 0 or heard >= tail * 10 ** (CHOKE_MASKED_DB / 10):
                continue
            out["hits"] += 1
            out["cut"] += bool(heard <= tail * 10 ** (CHOKE_DB / 10))
    out["choked"] = out["cut"] >= CHOKE_LEAST and 2 * out["cut"] >= out["hits"]
    return out


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


def estimate_levels(plan: dict, analysis: dict, headroom_db: float) -> dict:
    """Track faders (dB) that put our parts at the level of the stems they
    replace; it sets each drum pad's volume_db in the plan. Each pad plays
    where the stems hold its sound (pad_levels), the loudest at 0 dB and
    the others under it; a pad without a level (its file cannot be read or
    peaks under SILENT_DB, or none of its hits can be read) stays at 0 dB.
    The faders compare our parts, rendered offline at unity (each pad at
    its volume_db, each velocity at its level under Live's law at the pad's
    vol_vel, as render_plan plays it, each pad one voice; the 808 as
    render_plan plays it, Transpose, Detune and Glide included, released at
    each note's end as Live releases it), with the stems they replace. The
    drum fader matches the drum track's RMS over the whole track to the
    drum stem's: a perfect kit, the truth's drums at their own balance,
    then plays every drum sound at the kit level evaluate.level_scores
    scores against (the RMS of the drum stem over the truth's drums). The
    808's fader puts the 808 where the bass stem holds it as
    evaluate.level_scores reads the 808: per note, in the 808 band over
    LEVEL_MS from its attack, the median over the notes (level_808); with
    no note to read (level_808 None) it falls back to the RMS match. A
    perfect 808 then plays at the stem's per-note ratio to the truth's 808,
    not at the bass group's kit level (the RMS of the bass stem over the
    truth's bass): read against the truth, the separated bass stem holds
    the 808 at the plan's note attacks 0.29 [1.53] dB under that ratio
    (mean over scenarios, htdemucs; htdemucs_ft in brackets), which is
    what the per-note fader leaves. Matched by RMS, the 808 sat 1.87
    [2.01] dB under a perfect kit (m5): our 808 track holds more of its
    energy away from the notes' attacks than the stem does (the per-note
    fader is 1.58 [0.53] dB over the RMS one). Per note (f808-3), 0.29
    [1.48] dB under it, closer on 19 [17] of 22 scenarios. The notes clear
    of the plan's kick hits (none within CLEAR_MS) do not read better,
    though a separation leaves part of each kick in the bass stem: the
    kit level carries that part too (the bass stem's RMS holds it), and
    read on those notes alone (at least CLEAR_HITS, else all), the 808 sat
    0.67 [2.13] dB under a perfect kit, and the two scenarios it overshoots
    moved +0.32 -> +0.27 and +0.81 -> +0.86 dB (f808-kc). A kick is still
    read under the 808 at the RMS-matched level (held808): under the
    per-note one (f808-1) the kicks read lower, the loudest pad with them,
    and the other pads rose over the kit, hats +1.65 and snares +1.61 dB,
    1.20 -> 1.94 [1.15 -> 1.30] dB from a perfect kit.
    The drum fader stops where the loudest pad plays as loud as the stem
    holds its sound, the headroom under it. The RMS match
    goes over that on 4 [5] suite beats: where the stem holds a kick no pad
    plays (layers-1, tempo160-1, v1orig-2 [bass_low-2]; tempo160-1 sits
    13.4 dB from a perfect kit under it and 5.1 dB under the stop), and on
    v1orig-3 [kick808-2, leadin-2, rolls-1, v1orig-3], whose stems hold no
    kick the plan misses (there it takes v1orig-3 0.3 [0.2] dB further from
    a perfect kit and the others 0.2 to 2.6 dB closer). Where the fader
    would go over FADER_DB, the top of Live's, the pads under 0 dB make up
    the rest, as far as 0 dB (no suite beat: its drum faders reach -13.5
    [-12.3] dB).

    Without pad levels every pad played its file at 0 dB under one RMS
    fader. On the 48 suite beats (plan level, mean over scenarios,
    htdemucs; htdemucs_ft in brackets) the sounds sat a median 4.09 [3.70]
    dB from a perfect kit (levels.vs_kit_abs_db) and 3.03 [2.84] dB from
    its balance, hats +6.2 [+6.5] and open hats +11.6 [+10.5] dB over the
    kit: the files are louder than the songs play them (m3). With pad
    levels and the 808 one-shot on its template's periods (m4), 1.25
    [1.23] and 0.55 [0.50] dB (1.62 [1.38] without the stop); per role
    (the median over a beat's sounds, its mean over the suite, signed and
    of its size) kicks -0.96 and 1.22 [-0.08 and 0.54] dB from the kit,
    snares -0.17 and 1.33 [-0.80 and 1.45], claps +0.60 and 1.02 [+1.58 and
    1.58], hats -1.10 and 2.39 [-0.69 and 1.47], open hats -0.09 and 1.16
    [-0.83 and 1.20]. With each pad on one voice and the 808 released as
    Live plays them (m5), 1.20 [1.15] and 0.51 [0.48] dB; signed per role,
    kicks -0.84 [-0.04], snares +0.04 [-0.75], claps +0.70 [+1.73], hats
    -0.90 [-0.57], open hats +0.15 [-0.72]. Measured with the levels alone,
    over m3's outputs
    and its one-shot under the kicks (lv-f3): pad levels 1.27 [1.31] and
    0.56 [0.54] dB; every pad at the truth's own level for its sound and
    the RMS match alone, 1.37 [0.75] and 0.31 [0.04]: our files' tails and
    the sounds a plan misses or adds move the RMS match; each pad at the
    drum stem's level and the fader at the loudest pad's (the stem's own
    level, no RMS match), 4.05 [1.54]: kicks -5.8 [-1.1] dB from the kit,
    snares +4.0 [+0.9], where htdemucs' drum stem holds them."""
    bps = plan["tempo"] / 60
    n = int(analysis["source"]["duration_s"] * SR)
    out = {"headroom_db": headroom_db}
    stems = plan["stems"]
    track808, held808, fader808 = plan["tracks"]["808"], None, None
    if "sample_path" in track808 and "bass" in stems:
        shot, bass = read(track808["sample_path"]), read(stems["bass"]["path"])
        ours = render_808(track808, shot, n, bps, None, True, 1.0, "live")
        by_rms = round(rms_db(bass) - rms_db(ours) - headroom_db, 1)
        per_note = level_808(track808, shot, ours, bass, bps)
        fader808 = by_rms if per_note is None else round(per_note - headroom_db, 1)
        held808 = ours * np.float32(10 ** ((by_rms + headroom_db) / 20))  # where the bass stem holds it under a kick
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
        fader = round(rms_db(stem) - rms_db(_unity_drums(plan, n)) - headroom_db, 1)
        fader = min(fader, round(top - headroom_db, 1)) if at else fader
        if fader > FADER_DB:  # what the fader cannot give, the pads under 0 dB make up as far as they can
            for pad in plan["tracks"]["drums"]["pads"].values():
                pad["volume_db"] = round(min(0.0, pad["volume_db"] + fader - FADER_DB), 1) + 0.0
            fader = FADER_DB
        out["drums_track_db"] = fader
    if fader808 is not None:
        out["808_track_db"] = fader808
    out["stems_track_db"] = -headroom_db
    return out


def attacks_808(track: dict, shot: np.ndarray, bps: float) -> np.ndarray:
    """Where each of the 808's notes attacks (s), as
    evaluate.estimate_from_plan times it for the level scores: a chain's
    first note at its start, RETRIGGER_S later where Live starts it late
    (late_808), plus the attack of the one-shot as played at its pitch, in
    the 808 band; a legato note (Glide) at its start. A legato note that
    starts after the voice has read past the one-shot's end plays nothing
    and is left out, as estimate_from_plan leaves it out (its render is
    silent there, so level_808 could not read it either)."""
    s = settings_808(track)
    chains, gliding = chains_808(track), glide_s(track) > 0
    room = len(shot) - int(round(s["start_s"] * SR))  # the one-shot's samples from its start
    leads: dict[float, tuple[float, float]] = {}  # per shift: the attack (s) and the length (s) as played

    def played(pitch: int) -> tuple[float, float]:
        shift = pitch_808(s, pitch)[0]
        if shift not in leads:
            x = as_played(shot, s["start_s"], shift)
            leads[shift] = (attack_time(x, "808"), len(x) / SR)
        return leads[shift]

    late = (late_808(voices_808(chains), bps, lambda p: played(p)[1], gliding) if takes_over(gliding)
            else np.zeros(len(chains), bool))
    out = []
    for c, chain in enumerate(chains):
        out.append(chain[0]["start_time"] / bps + RETRIGGER_S * late[c] + played(chain[0]["pitch"])[0])
        if len(chain) > 1:
            pos = read_at(chain_semitones(chain, s, bps, glide_s(track)))  # where the voice reads, sample by sample
            for n in chain[1:]:
                o = int(round((n["start_time"] - chain[0]["start_time"]) / bps * SR))
                if np.sum(pos[o:] < room) > 1:  # it has not read past the one-shot's end
                    out.append(n["start_time"] / bps)
    return np.array(out)


def level_808(track: dict, shot: np.ndarray, ours: np.ndarray, bass: np.ndarray, bps: float,
              ms: float = LEVEL_MS) -> float | None:
    """How far over `ours` (the 808 track at unity) the bass stem holds the
    808 (dB), read as evaluate.level_scores reads the 808: per note, the
    energy in the 808 band over `ms` from where the note attacks
    (attacks_808), the stem's over ours; the median over the notes. None
    where no note holds energy on both."""
    at, w = attacks_808(track, shot, bps), samples(ms)
    d = _db(_energies(band(bass, "808"), at, w), _energies(band(ours, "808"), at, w))
    return float(np.median(d)) if d else None


def _unity_drums(plan: dict, n: int) -> np.ndarray:
    """The drum track at unity: each pad's file from its onset at its
    volume_db, each velocity at its level under Live's law at the pad's
    vol_vel, each pad one voice (render.pad_hits)."""
    shots, settings = {}, {}
    for note, pad in plan["tracks"]["drums"]["pads"].items():
        try:
            semis = pad["transpose"] + pad.get("detune_cents", 0) / 100
            shots[str(note)] = (trim_onset(pad_shot(pad["path"], 0.0, semis, plan["live_sr"])) if plan.get("live_sr")
                                else transpose(trim_onset(read(pad["path"])), semis))
        except Exception:
            continue
        settings[str(note)] = pad_settings(pad)
    hits = pad_hits(plan, shots, settings, faders=False)
    out = np.zeros(n, np.float32)
    for key, x in shots.items():
        out += pad_track(x, [h for h in hits if h["pad"] == key], n)
    return out


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
    hit had no level. Every other sound is read on the drum stem; a hit
    another sound shares (one attacks within CLEAR_MS of it) reads that
    sound too, and where most of a sound's hits are shared with a louder
    one, their median is one of those: a sound whose CLEAR_HITS or more
    clear hits agree within CLEAR_IQR_DB and read more than CLEAR_GAP_DB
    under all its hits is read on them (_sound_level). The pads of one
    sound (a template's pad and its pitched copies) share one level, their
    hits pooled. A file that peaks under SILENT_DB gets none.

    Measured per pad on the 48 suite beats against where the truth holds
    the sound it plays (median over pads; htdemucs, htdemucs_ft in
    brackets; m4, the plan's 808 the one-shot on its template's periods):
    a pad read on the drum stem sits -1.27 [-1.28] dB from it, p10 to p90
    -1.95 to -0.15 [-1.94 to -0.08]. A kick read on the drum stem sat
    -10.01 [-3.59] dB from it (-13.74 to -2.55 [-5.27 to -2.02]): htdemucs
    moves much of a kick to the bass stem. On drums + bass the 808 under it
    put it at +1.50 [+1.42]; less the plan's 808, -1.36 [-0.86] (-2.68 to
    +0.24 [-2.61 to +0.44]). The floor moves the kicks of 1 [4] suite
    beats, each toward the truth (v1orig-3: -1.99 -> -0.69 [-2.03 -> -0.63]
    dB from a perfect kit); on PIRAMIT, with the plan's 808 1 or 2 dB
    louder than its fader, it holds the kick at -16.4 dB, its drum stem's
    reading, where without it 37 and 6 of the kick's 93 hits were left and
    it read -23.8 and -20.3 dB. It does not hold a kick the separation
    splits under a ducked 808: 60 % of a kick in the bass stem, its 808
    ducked 3 to 20 dB under it, read the kick 2.2 to 8.0 dB low (synthetic;
    no suite beat ducks its 808, and PIRAMIT's bass stem holds the plan's
    808 at -0.3 dB over a kick's first 50 ms, -1.7 dB at 808 notes with no
    kick). No kick of the suite had three hits clear of every 808 note.
    Read only on its hits clear of other sounds, a pad sat -1.37 [-1.43] dB
    from the truth, and the kit no closer to a perfect one. Of the suite's
    209 sounds with three clear hits or more, 200 read them no more than 3
    dB under all their hits (their interquartile range a median 0.15 dB,
    p95 2.8) and 9 more than 6 dB under, 5 of those within 3 dB of each
    other, now read on them: v1orig-2's extra open hat, 9 of whose 17 hits
    sat on the snare and the hat (+23.3 -> +4.0 dB from a perfect kit,
    against the truth's open hat; as the loudest pad it had put the hat's
    Volume at -36.3 dB and the drum fader at +5.6 dB), producer-2's open
    hat (+8.75 -> -2.95 [+8.24 -> -3.26]), v1orig-3's open hat, on a pad
    that also plays 12 of the snare's hits, and [tempo160-1's extra snare,
    21 of whose 37 hits play no snare: -13.0 -> -23.4]. The other 4 spread
    their clear hits 4.9 to 13.8 dB (three play another role's file) and
    keep all their hits."""
    bps = plan["tempo"] / 60
    pads, notes = plan["tracks"]["drums"]["pads"], plan["tracks"]["drums"]["notes"]
    shots = {}
    for key, pad in pads.items():
        s = pad_settings(pad)
        try:
            x = pad_shot(pad["path"], s["start_s"], s["transpose"] + s["detune_cents"] / 100, plan.get("live_sr"))
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
    # each pad's own track at Volume 0 dB and where its hits attack, its hits summed (mono=False): a stem holds
    # the tail of a sound's hit before under its next one (the suite's truth sums them), and read against the hits
    # cut as Live plays them, a pad took those tails for its own level (render_plan as Live plays: the suite's
    # levels.vs_kit_abs_db 1.20 -> 1.30 dB, htdemucs_ft 1.19 -> 1.27; balance_abs_db 0.49 -> 0.60, 0.50 -> 0.51)
    # A choke group's pads cut each other where Live cuts them (render.choke_voices), as the stem holds them
    group_of = grouped({key: s["choke_group"] for key, (x, s) in shots.items()})
    hits = {key: pad_hits(plan, {key: x}, {key: {**s, "volume_db": 0.0}}, faders=False, mono=False)
            for key, (x, s) in shots.items() if key not in group_of}
    if group_of:
        together = pad_hits(plan, {k: shots[k][0] for k in group_of},
                            {k: {**shots[k][1], "volume_db": 0.0} for k in group_of}, faders=False, mono=False)
        hits |= {k: [h for h in together if h["pad"] == k] for k in group_of}
    at = {key: np.array([e["start_time"] / bps for e in notes if str(e["pitch"]) == key]) + attack_time(x, s["role"])
          for key, (x, s) in shots.items()}
    heard: dict[str, list[float]] = {}
    alone: dict[str, list[float]] = {}  # a kick on the drum stem alone: the least the stems hold of it
    clear: dict[str, list[float]] = {}  # a sound that is no kick, at its hits clear of every other sound's
    for key, (x, s) in shots.items():
        kick = sound[key] in kicks
        b, name = ("kick", "low") if kick else (ATTACK_BAND.get(s["role"], "full"), "drums")
        for src, y in ((name, low if kick else drums), ("drums", drums)):
            if (src, b) not in banded:
                banded[src, b] = band(y, b)
        own = band(pad_track(x, hits[key], len(drums)), b)
        theirs = _energies(banded[name, b], at[key], w)
        if kick and under is not None:
            if ("under", b) not in banded:
                banded["under", b] = band(under, b)
            theirs = theirs - _energies(banded["under", b], at[key], w)
        ours = _energies(own, at[key], w)
        heard.setdefault(sound[key], []).extend(_db(theirs, ours))
        if kick:
            alone.setdefault(sound[key], []).extend(_db(_energies(banded["drums", b], at[key], w), ours))
        else:
            others = np.concatenate([at[k] for k in shots if sound[k] != sound[key]] + [np.full(1, np.inf)])
            apart = np.min(np.abs(at[key][:, None] - others[None, :]), axis=1) > CLEAR_MS / 1000
            clear.setdefault(sound[key], []).extend(_db(theirs[apart], ours[apart]))
    level = {snd: _sound_level(heard[snd], alone.get(snd, []), clear.get(snd, [])) for snd in heard}
    return {key: level[sound[key]] for key in shots if level.get(sound[key]) is not None}


def _db(theirs: np.ndarray, ours: np.ndarray) -> list[float]:
    """10 log10(theirs / ours) (dB) at each hit where both hold energy."""
    ok = (theirs > 0) & (ours > 0)
    return (10 * np.log10(theirs[ok] / ours[ok])).tolist()


def _sound_level(heard: list[float], alone: list[float], clear: list[float]) -> float | None:
    """A sound's level (pad_levels) from its hits' readings: their median,
    or where CLEAR_HITS or more of its hits clear of other sounds agree
    within CLEAR_IQR_DB (interquartile range) and read more than
    CLEAR_GAP_DB under it, theirs (its other hits read a louder sound
    too); a kick's at least its median on the drum stem alone."""
    if (len(clear) >= CLEAR_HITS and heard and np.subtract(*np.percentile(clear, [75, 25])) <= CLEAR_IQR_DB
            and np.median(heard) - np.median(clear) > CLEAR_GAP_DB):
        heard = clear
    got = [float(np.median(v)) for v in (heard, alone) if v]
    return max(got) if got else None


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


# ── The build's spec, which blueprint.py writes as a blueprint for build_from_blueprint ──

def simpler_808(t808: dict) -> dict:
    """The 808 Simpler's parameters as build_tracks sets them: Volume 0 dB,
    Transpose and Detune (tune_808: whole semitones and whole cents, which
    Live 12.4.6 keeps as given), "Vol < Vel" (vol_vel, the velocity law's
    amount, a raw 0..1), and with Glide on (voices 1) Glide Mode "Glide"
    and the Glide Time. These are what render_plan and estimate_from_plan
    assume; voices is a property of the device, not a parameter
    (build_from_blueprint sets it from the blueprint). Snap and the filter go off (LIVE_OFF): loaded through the
    browser, the 808's Simpler had its filter on, a 22.0 kHz lowpass whose
    cutoff follows the note ("Filt < Key" 100 %), and in the live check
    (Live 12.4.6, 2026-10-29) it delayed the 808's notes by 0.13 to 0.46 ms,
    the lower the note the more (0.46 ms at MIDI 12, 0.13 ms at 36), and
    took 0.14 dB off the 8th harmonic, 0.55 dB off the 16th and 2.2 dB off
    the 32nd, re the fundamental: swing66-1's three notes at MIDI 12, 19.6
    to 19.9 ms from the truth in the plan, crossed 20 ms in Live (808 F1@20
    0.944 -> 0.889 with Live's lags). render.py plays the one-shot
    unfiltered. Ve Release goes at Live's own default, RELEASE_808_MS,
    whose fall render.py plays (render._release), rather than at a value
    no capture has measured: on PIRAMIT, over 40 ms after the plan's notes
    that end in a rest, the bass stem holds a median -7.5 dB re their last
    20 ms, this release -7.9 dB, the 5 ms linear fade render.py used -10.7
    and the same law at 5 ms -18.3."""
    out = {"Volume": "0.0 dB", "Transpose": float(t808["transpose"]), "Detune": float(t808.get("detune_cents", 0)),
           "Vol < Vel": float(t808.get("vol_vel", VOL_VEL)), **LIVE_OFF, "F On": "Off",
           "Ve Release": f"{RELEASE_808_MS:.1f} ms"}
    if glide_s(t808):
        out |= {"Glide Mode": "Glide", "Glide Time": f"{float(t808['glide_time_ms']):.1f} ms"}
    return out


def bar_beats(plan: dict) -> float:
    """A bar of the plan's meter in its (quarter) beats: 4 in 4/4, 3 in 3/4 and 6/8."""
    return int(plan.get("beats_per_bar", 4)) * 4 / int(plan.get("beat_unit", 4))


def _moved(notes: list[dict], start: float, end: float) -> list[dict]:
    """The notes that start in [start, end), moved by -start beats, each
    ending by end."""
    return [{**n, "start_time": n["start_time"] - start, "duration": min(n["duration"], end - n["start_time"])}
            for n in notes if start <= n["start_time"] < end]


def _trimmed(src: str | Path, dst: Path, seconds: tuple[float, float]) -> None:
    """src's samples from seconds[0] to seconds[1] (fewer where it ends
    first) to dst, at its rate and sample format."""
    import soundfile as sf
    info = sf.info(str(src))
    a, b = (min(info.frames, int(round(t * info.samplerate))) for t in seconds)
    x, rate = sf.read(str(src), start=a, stop=b, dtype="float64", always_2d=True)  # every PCM and float exactly
    sf.write(str(dst), x, rate, subtype=info.subtype, format=info.format)


def excerpt(plan: dict, bars: tuple[int, int], stems_out: str | Path | None = None) -> dict:
    """Bars a to b of the plan (both in, counted from 1 as Live's ruler
    shows them: bar a starts at beat (a - 1) x bar_beats) as a plan of its
    own, for a short live check: Live has no export and records in real
    time (the two live checks so far took 1 to 1.5 h each), and 8 bars at
    140 BPM play in 14 s. Its drum, 808 and melodic notes, locators and
    clips start at 0:
    the notes that start in those bars move back by bar a's start and end by
    bar b's end (a note that starts before bar a is left out, also where it
    sounds into it, and an 808 note that slid from it starts on its own);
    length_beats is the excerpt's. The stems are cut at the
    same instants (bar a's start and bar b's end at the plan's tempo) into
    stems_out, which the plan's stems need, and the excerpt plays those. The
    808's one-shot, its settings, the pads and the levels are the plan's.
    excerpt records the bars, beats and seconds it took."""
    a, b = (int(v) for v in bars)
    bar = bar_beats(plan)
    last = math.ceil(float(plan["length_beats"]) / bar - 1e-9)
    if not 1 <= a <= b <= last:
        raise ValueError(f"bars {a}-{b} are not in the plan's 1-{last}")
    start, end = (a - 1) * bar, min(b * bar, float(plan["length_beats"]))
    seconds = (start * 60 / plan["tempo"], end * 60 / plan["tempo"])
    out = json.loads(json.dumps(plan))
    tracks = out["tracks"]
    for t in [tracks["drums"], tracks["808"], *tracks.get("melodic", {}).values()]:
        t["notes"] = _moved(t["notes"], start, end)
    out["length_beats"] = end - start
    out["locators"] = [[name, beat - start] for name, beat in plan["locators"] if start <= beat < end]
    if out["stems"]:
        if stems_out is None:
            raise ValueError("an excerpt of a plan with stems needs a folder for them, cut to its bars (stems_out)")
        Path(stems_out).mkdir(parents=True, exist_ok=True)
        for info in out["stems"].values():
            dst = Path(stems_out) / Path(info["path"]).name
            _trimmed(info["path"], dst, seconds)
            info["path"] = str(dst)
    out["excerpt"] = {"bars": [a, b], "beats": [start, end], "seconds": [round(v, 6) for v in seconds]}
    return out


def pad_volume(pad: dict) -> float:
    """A pad's Volume as the build sends it: its volume_db (at most 0 dB,
    render.pad_settings), a number in dB, which build_tracks sets as the
    parameter's value. Sent as display text ("-6.6 dB"), Live 12.4.6 held a
    display step under it (-6.605 dB; from -10 dB down 0.05 dB: "-14.9 dB"
    held -14.95) in the live check (2026-10-29); build_from_blueprint reads
    the value back and sends the text where the number did not hold
    (volumes_not_held)."""
    return float(pad_settings(pad)["volume_db"]) + 0.0


def choke_of(pad: dict) -> int:
    """A pad's choke group as Live takes it: 1 to 16, 0 for none."""
    return grouped({"pad": pad.get("choke_group")}).get("pad", 0)


def build_spec(plan: dict, bars: tuple[int, int] | None = None, stems_out: str | Path | None = None) -> list[dict]:
    """The plan as ableton-live-mcp's build_tracks spec: the 808 on a Simpler
    (simpler_808; "voices", which build_tracks does not read and the
    blueprint carries for build_from_blueprint), the drums on a Drum Rack
    (one pad per sound, its file by the name of its copy, pad_sample:
    scaled by its file_db where it has one), the melodic notes on MIDI
    tracks (muted), the reference stems as audio tracks (their clips at
    beat 0, or where a plan moved onto Live's bars moved them, moved_beats:
    the song's 0 s); Simplers with
    warping off, the 808's at 0 dB and each pad's at its volume_db
    (estimate_levels; never over 0 dB, 0 dB without one), each sound's
    transposition (a pad's Transpose and Detune), "Vol < Vel" at the plan's
    vol_vel (VOL_VEL), so velocities play the levels the plan assumes, Snap
    off (LIVE_OFF), and faders from the plan's levels. A pad's Volume goes
    as a number (pad_volume), and a pad of a choke group carries its group
    (choke_group: ableton-live-mcp's build_tracks reads it from 2.3.0).
    With bars (a, b), the spec of excerpt(plan, bars, stems_out): bars a to
    b, from 0."""
    if bars:
        plan = excerpt(plan, bars, stems_out)
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
                     str(note): {"sample": pad_sample(pad["path"], pad_settings(pad)["file_db"]), "warping": False,
                                 "settings": {"Volume": pad_volume(pad),
                                              "Transpose": float(pad["transpose"]),
                                              "Detune": float(pad.get("detune_cents", 0)),
                                              "Vol < Vel": float(pad.get("vol_vel", VOL_VEL)), **LIVE_OFF},
                                 **({"choke_group": choke_of(pad)} if choke_of(pad) else {})}
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
                     "audio_clips": [{"file": info["path"], "start": moved_beats(plan), "warping": False}]})
    return spec


def refused(error: Exception) -> str:
    """What a first call to Live that failed means. Live's Remote Script
    (AbletonMCP 2.1.0, MAX_CLIENTS) serves at most CLIENTS clients and
    closes the connection of one more: in the live check (2026-10-29) the
    MCP servers of other sessions held all 8, and live-build failed with
    "Connection reset by peer". Nothing answers when Live is closed or
    AbletonMCP is not one of its Control Surfaces."""
    text = str(error)
    if any(s in text for s in ("reset by peer", "closed the connection", "Errno 54", "Broken pipe")):
        return (f"Live refused the connection: its Remote Script serves at most {CLIENTS} clients, and other "
                "MCP servers (other sessions with ableton-live-mcp) are connected. Close the sessions you do not "
                f"need, or restart Live, and build again. ({text})")
    if any(s in text for s in ("Nothing answers", "Connection refused", "Could not connect")):
        return f"Live does not answer: is it open, with AbletonMCP as a Control Surface? ({text})"
    return text


def meter_of(plan: dict) -> str:
    """The plan's meter as Live's time signature ("3/4"); 4/4 for a plan
    from before plans carried one."""
    return f"{int(plan.get('beats_per_bar', 4))}/{int(plan.get('beat_unit', 4))}"
