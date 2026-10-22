"""The steps in order, each writing what it found to a work folder:

    analyze(stems)   -> analysis.json, bass808.json, templates/ (grid, tuning, drums, 808, sections)
    melodic()        -> melodic.json, basicpitch/             (notes per tonal stem, chords)
    match(library)   -> matches.json                          (library sound per template, verdicts)
    fit_808()        -> synth808.json, 808 one-shot .wav      (the synthesized 808)
    plan()           -> plan.json, midi/                      (what Live gets)

Everything under the work folder is derived from the song: keep it local.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
from scipy.signal import find_peaks

from . import bass808
from . import describe as describe_mod
from . import detect as det
from .audio import PRE, SR, band, read, samples, shift, transpose, write
from .audio import fit as fit_to
from .beats import track_stems
from .detect import (
    LENGTH_MS,
    Template,
    count_once,
    discover,
    explain,
    isolated,
    peak_score_histogram,
    pitch_offset,
)
from .grid import Grid, estimate, offsets, onset_grid, tuning
from .hits import Group, group, template
from .library import Library, null_test, pair_fit, per_hit_votes, verdict
from .structure import bar_activity, sections
from .synth808 import Params, compare, fit, laid_out, one_shot
from .xcorr import sliding_ncc

TONAL_STEMS = ("bass", "other", "piano", "vocals", "guitar")
# library roles grouped into the drum families a Drum Rack is laid out by
FAMILY = {"clap": "snare", "rim": "snare", "open_hat": "hat", "crash/ride": "hat", "808": "kick"}
DRUM_FAMILIES = ("kick", "snare", "hat", "perc")  # live.PADS: the families a pad is laid out by
# kick-band templates are not searched for layers: any sine-like file takes energy off a kick, so the pair
# search gave 3 of 24 research-beat kicks a layer, none of them true (and found no true one)
TONAL_LAYERS = "kick band: not searched (any sine-like file takes energy off a kick)"
SAME_SOUND = 0.5  # a layer candidate this correlated with the top match (aligned, in the band) is that sound
# a "layer" this much louder than the top match means the top match is not the template's sound (a kick
# template's snare-band match got a kick 15 dB louder as its layer); a clap on a snare sits -6..0 dB
LOUDER_DB = 10.0


def _save(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    return path


def _load(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def analyze(stems_dir: str | Path, work: str | Path, bpm_range=(100.0, 180.0), log=print, bpm: float | None = None,
            meter: str | None = None, tracker: bool = True) -> dict:
    """tracker=False (or a tracker that cannot run) fits the grid as before:
    fit_grid in bpm_range, first grid beat as phase."""
    stems_dir, work = Path(stems_dir), Path(work)
    drums, bass = read(stems_dir / "drums.wav"), read(stems_dir / "bass.wav")
    duration = len(drums) / SR
    log(f"stems: {duration:.1f} s")
    beats = track_stems(stems_dir, log) if tracker else None
    grid, fit_info = estimate(drums, beats, bpm_range, bpm, meter)
    what = "first beat" if fit_info["tracker"] is None else f"{grid.beats_per_bar}/{grid.beat_unit}, first bar line"
    log(f"grid: {grid.bpm} BPM, {what} at {grid.phase} s")
    if beats is not None and fit_info["tracker"] is None:
        log(f"  {fit_info['level']['why']}: the grid is fit_grid's")
    if fit_info["tracker"] is not None:
        level, swing = fit_info["level"], fit_info["swing"]
        log(f"  {level['level']} of the tracker's {level['tracker_bpm']} BPM ({level['why']}); phase "
            f"{fit_info['phase_correction_ms']:+.1f} ms onto the attacks; swing: 16ths {swing['ratio_16']}, "
            f"8ths {swing['ratio_8']}")
    if fit_info["tempo_map"]:
        log(f"  the beats drift up to {fit_info['drift_ms']} ms from a constant tempo: tempo_map kept "
            f"({len(fit_info['tempo_map'])} beats); the grid stays constant")
    change = fit_info["tempo_change"]
    if change:
        at = sorted((change["until"], change["from"]))
        log(f"  the tempo changes by x{change['ratio']} at {at[0]}-{at[1]} s: the grid follows the "
            f"{change['grid_follows']} part ({change['span_s'][0]}-{change['span_s'][1]} s); the rest is off it")
    down = fit_info["downbeat"]
    if down and down["ambiguous"]:
        why = down.get("why") or f"{down.get('share', 0):.0%} of the tracker's downbeats agree"
        log(f"  the first bar line is unsure ({why}); --meter sets the meter")
    for warning in fit_info["warnings"]:
        log(f"  warning: {warning}")
    # offsets() reads onset-strength peaks, which come a few ms after the attacks discover() works on
    peaks_grid = onset_grid(grid, fit_info)
    tunings = {s: tuning(read(stems_dir / f"{s}.wav")) for s in TONAL_STEMS
               if (stems_dir / f"{s}.wav").is_file()}
    consensus = round(float(np.median(list(tunings.values()))), 2) + 0.0 if tunings else 0.0  # no -0.0
    log(f"tuning (semitones from A440): {tunings} -> {consensus:+.2f}")
    templates = discover(drums, grid)
    found = explain(drums, templates)
    events = count_once(found.events, templates, drums)
    tdir = work / "templates"
    tdir.mkdir(parents=True, exist_ok=True)
    out_templates = {}
    for tp in templates:
        np.save(tdir / f"{tp.name}.npy", tp.audio)
        write(tdir / f"{tp.name}.wav", tp.audio)
        out_templates[tp.name] = {"role_guess": tp.role, "band": tp.band, "template_hits": tp.hits,
                                  "slots": tp.slots, "attack_s": tp.attack_s, "same_sound_as": tp.same_sound_as,
                                  "found": tp.found,
                                  "detected_hits": len(events[tp.name]),
                                  "peak_ncc_histogram_0.3_to_1.0": peak_score_histogram(drums, tp),
                                  "describe": describe_mod.describe(band(tp.audio[:samples(LENGTH_MS[tp.role])],
                                                                         tp.role)),
                                  "clean": _clean(events[tp.name]),
                                  "members": tp.group.members if tp.group else []}
        layered = sum(e.get("source") == "layer" for e in events[tp.name])
        pitched = sum(bool(e.get("semitones")) for e in events[tp.name])
        log(f"  {tp.name:8s} ({tp.band} band, looks like a {tp.role}) {tp.hits:3d} hits in the template, "
            f"{len(events[tp.name]):3d} detected ({layered} under other sounds, {pitched} pitched)")
    for name, into in found.info.get("merged", {}).items():
        log(f"  {name} plays what {into['into']} plays ({into['offset_ms']:+.1f} ms): its hits are {into['into']}'s")
    b808 = bass808.transcribe(bass, drums, log=log)
    _save(work / "bass808.json", b808)
    notes = bass808.project(b808)
    tpl = b808["template"]
    _, agreement_808 = pitch_offset([n["midi_settled"] for n in notes])
    tuning_808 = round(tpl["tuning_cents"] / 100, 4) if tpl else consensus
    if tpl:
        log(f"808: {len(notes)} notes, template {tpl['hz']:.2f} Hz = MIDI {tpl['midi']:.2f} ({tpl['method']}, "
            f"{tpl['notes_used']} notes): tuned {tpl['tuning_cents']:+.1f} cents")
    else:
        log("808: no note to learn a template from")
    n_bars = int(np.ceil((duration - grid.phase) / grid.bar))
    stems = {s: read(stems_dir / f"{s}.wav") for s in ("other", "piano", "vocals")
             if (stems_dir / f"{s}.wav").is_file()}
    rows = bar_activity(grid, n_bars, events, notes, stems)
    secs = sections(rows)
    log("sections: " + ", ".join(f"{s['name']} {s['start_bar']}-{s['end_bar']}" for s in secs))
    analysis = {"source": {"stems": str(stems_dir), "duration_s": round(duration, 3)},
                "grid": {**grid.to_dict(), **fit_info, "offsets_by_beat": offsets(drums, peaks_grid)},
                "tuning": {"per_stem": tunings, "consensus": consensus,
                           "a4_hz": round(440 * 2 ** (consensus / 12), 1),
                           "808": tuning_808, "808_agreement": agreement_808},
                "templates": out_templates, "events": events, "layers": found.info, "notes_808": notes,
                "bars": [{**r, "sounds": sorted(r["sounds"]), "stems": sorted(r["stems"])} for r in rows],
                "sections": secs}
    _save(work / "analysis.json", analysis)
    return analysis


def _clean(events: list[dict]) -> float:
    """Share of a template's own matched-filter hits at NCC 0.8 or more (a
    hit found under another sound is not its own match)."""
    own = [e["ncc"] >= 0.8 for e in events if e.get("source", "matched") == "matched"]
    return round(float(np.mean(own)), 3) if own else 0.0


def _template(work: Path, name: str) -> np.ndarray:
    return np.load(work / "templates" / f"{name}.npy")


def _entry(r: dict) -> dict:
    """A ranked library entry as matches.json lists it. semitones is the
    whole transposition (fractional from the speckle search); transpose and
    cents split it into Simpler's Transpose (whole semitones) and Detune
    (cents, -50..+50; detune_cents is the same number under Simpler's name)."""
    st = float(r["semitones"])
    whole = int(round(st))
    cents = round(100 * (st - whole), 2) + 0.0
    out = {"path": r["path"], "file": Path(r["path"]).name, "role": r["role"], "ncc": r["ncc"], "semitones": st,
           "transpose": whole, "cents": cents, "detune_cents": cents, "lag_ms": r["lag_ms"]}
    if "speckle" in r:
        out["speckle"] = r["speckle"]
    return out


def _search(library: Library, t: np.ndarray, b: str, tonal: bool, detune: float | None) -> tuple[list[dict], str]:
    """One template window ranked in one band (its best 40), and how. The
    snare and hat bands: by the speckle index at any pitch, with the
    whole-semitone search's best 20 tried at their speckle transposition
    too (Library.match_fine). The kick band, every band of a tonal template
    (one found in the low band: role guesses are no help, 51 of 71 hat
    templates of a 48-beat suite run were guessed kicks), and every band
    without a speckle index: whole semitones, as before (a fine search let
    other kicks at fractional transpositions beat the right one)."""
    if not tonal and b in library.speckles:
        return library.match_fine(t, b, semitones=range(-3, 4), union=20, top=40), "speckle"
    semis = range(-4, 5) if b == "kick" else range(-3, 4)
    best = None
    for base in [0.0] + ([detune] if detune else []):
        ranked = library.match(t, b, semitones=semis, base=base, top=40)
        if best is None or ranked[0]["ncc"] > best[0]["ncc"]:
            best = ranked
    return best, "semitones"


def _mix(stems: Path, n: int) -> np.ndarray | None:
    """The mix the stems were separated from (separation.json's audio), n samples long."""
    info = stems / "separation.json"
    if not info.is_file():
        return None
    audio = Path(_load(info).get("audio") or "")
    return fit_to(read(audio), n) if audio.is_file() else None


def _render_808(notes: list[dict], shot: np.ndarray, root: int, n: int) -> np.ndarray:
    """The 808 notes played with one one-shot: each transposed to its pitch,
    at its peak level, cut at its end."""
    out = np.zeros(n, np.float32)
    played: dict[int, np.ndarray] = {}
    for note in notes:
        k = int(note["pitch"]) - root
        if k not in played:
            played[k] = transpose(shot, k)
        a, b = int(round(note["t"] * SR)), min(n, int(round(note["end"] * SR)))
        x = played[k][: max(0, b - a)]
        out[a:a + len(x)] += float(note.get("peak", 1.0)) * x
    return out


def _explained(t: np.ndarray, atom: np.ndarray, max_lag_ms: float = 5.0) -> np.ndarray:
    """t without the 808 under it (atom: the rendered 808 notes averaged like
    t's hits), lined up by the largest positive correlation within
    +-max_lag_ms and scaled by least squares in the kick band, never added."""
    tb, ab = band(t, "kick").astype(np.float64), band(atom, "kick").astype(np.float64)
    lag = max(range(-samples(max_lag_ms), samples(max_lag_ms) + 1), key=lambda d: float(np.dot(tb, shift(ab, d))))
    ab = shift(ab, lag)
    g = max(0.0, float(np.dot(tb, ab) / (np.dot(ab, ab) or 1.0)))
    return (t - g * shift(atom, lag)).astype(np.float32)


def kick_sources(work: Path, analysis: dict, stems: Path, drums: np.ndarray) -> tuple[dict, np.ndarray | None, dict]:
    """What else a kick template can be read from: drums + bass and the mix
    the stems came from, each with the song's 808 taken out (analysis
    notes_808 played with fit-808's one-shot, synth808.json). Demucs puts
    part of a kick's body in the bass stem, and a kick under an 808 is lost
    in drums + bass unless the 808 is explained first. With no 808 notes
    the sources are taken as they are; with notes but no one-shot yet
    (fit-808 runs after match) only the drum stem is used.
    Returns ({source: signal}, the rendered 808 or None, what was used)."""
    notes = analysis.get("notes_808") or []
    r808, info = None, {"808_from": None}
    if notes:
        if not (work / "synth808.json").is_file() or not Path(_load(work / "synth808.json")["file"]).is_file():
            return {}, None, {"808_from": None, "why": "no 808 one-shot to take the 808 out with (run fit-808 first)"}
        synth = _load(work / "synth808.json")
        r808 = _render_808(notes, read(synth["file"]), int(synth["root_midi"]), len(drums))
        info = {"808_from": "synth808.json"}
    out = {}
    if (stems / "bass.wav").is_file():
        out["drums+bass" + ("-808" if notes else "")] = drums + fit_to(read(stems / "bass.wav"), len(drums))
    mix = _mix(stems, len(drums))
    if mix is not None:
        out["mix" + ("-808" if notes else "")] = mix
    return out, r808, info


def _heard(drums: np.ndarray, sound: np.ndarray, b: str) -> np.ndarray:
    """Where a sound (laid out like a template) plays in the drum stem, in
    seconds: a matched filter in band b at detect's threshold, its peaks
    MIN_GAP_MS apart, built from xcorr and scipy rather than from detect's
    own helpers, which are not an interface. detect() starts from the same
    filter (its candidates 8 ms apart) and then fits each moment with every
    template (layers.explain); a sound alone is found at the same moments."""
    score, _ = sliding_ncc(drums, sound, b)
    peaks, _ = find_peaks(score, height=det.THRESHOLD[b], distance=samples(det.MIN_GAP_MS[b]))
    return (peaks + PRE) / SR


def _elsewhere(library: Library, r: dict, b: str, drums: np.ndarray, hits: np.ndarray) -> tuple[int, int]:
    """At how many of a template's hits (attack times, s) a library sound is
    heard in the drum stem, and at how many other moments (_heard, with the
    sound's attack, over the whole stem)."""
    found = _heard(drums, library.candidate(r["i"], samples(det.DETECT_MS[b]), r["semitones"]), b)
    if not len(found):
        return 0, 0
    d = np.min(np.abs(found[:, None] - np.asarray(hits)[None, :]), axis=1) if len(hits) else np.full(len(found), 9.0)
    return int(np.sum(d <= 0.03)), int(np.sum(d > 0.05))


def _layer(library: Library, t: np.ndarray, b: str, ranked: list[dict], taken: list[int], min_db: float = 3.0,
           pool: int = 30, drums: np.ndarray | None = None, hits=()) -> tuple[list[dict], dict]:
    """The second sound of a layered noisy template (a clap on a snare), and
    the evidence. The template's top match is fitted together with each
    other candidate (the template's `pool` best, and the `pool` best for
    what the top match leaves), both gains positive (pair_fit). The partner
    that leaves the least is a layer when it takes the residual down by
    min_db or more and is not a sound another template was given (`taken`)
    or a copy of one: a plain snare's best partner is the hat that always
    plays with it. A partner more than LOUDER_DB louder than the top match
    is not its layer either. With the drum stem and the template's hit
    times, a partner heard at more moments away from those hits than at
    them (and at more than two) is a sound of its own, found or not by
    another template. gain_db is the layer's level against the top
    match's (both files peak-normalized), lag_ms how much later it starts.
    On 12 snares with a clap at a fixed offset and 12 plain ones (hits at
    the true times; the research chose the 3 dB rule on these 24) it finds
    both sounds in 10 and gives no plain snare a layer. A clap that moves
    0-8 ms from hit to hit (the suite's layers scenario) is smeared in the
    average: found in neither of 2 beats."""
    main, n = ranked[0], len(t)
    tb = band(t, b)
    atom, _ = library.atoms(t, b, [main])
    g = max(0.0, float(np.dot(tb, atom[0]) / (np.dot(atom[0], atom[0]) or 1.0)))
    rest = (tb - g * atom[0]).astype(np.float32)
    seen, partners = {main["i"]}, []
    for r in library.distinct(ranked, n, keep=pool) + library.distinct(_search(library, rest, b, False, None)[0], n,
                                                                        keep=pool):
        if r["i"] not in seen and not library.same(r["i"], main["i"], n):
            seen.add(r["i"])
            partners.append(r)
    atoms, lags = library.atoms(t, b, [main] + partners)
    fit2 = pair_fit(tb, atoms)
    # a partner that correlates with the top match is the same sound again (a copy at another pitch, a
    # near-copy), not a second one: its fit is ill-conditioned (an open hat's near-copy came out +23 dB)
    alike = atoms @ atoms[0] / (np.linalg.norm(atoms, axis=1) * np.linalg.norm(atoms[0]) + 1e-12) >= SAME_SOUND
    residual = np.where(alike, np.inf, fit2["residual_db"])
    k = int(np.argmin(residual)) if partners else 0
    if not partners or not np.isfinite(residual[k]):
        return [], {"accepted": False, "why": "no other sound fits with a positive gain"}
    r = partners[k - 1]
    better = round(fit2["single_db"] - float(fit2["residual_db"][k]), 2)
    layer = {**_entry(r), "gain_db": round(float(20 * np.log10(fit2["gains"][1][k] / fit2["gains"][0][k])), 2),
             "lag_ms": round(float(lags[k] - lags[0]) * 1000 / SR, 2)}
    evidence = {"single_db": round(fit2["single_db"], 2), "pair_db": round(float(fit2["residual_db"][k]), 2),
                "better_db": better, "partner": layer["file"]}
    if any(library.same(r["i"], i, n) for i in taken):
        return [], {**evidence, "accepted": False, "why": "the partner is another template's sound"}
    if layer["gain_db"] > LOUDER_DB:
        why = f"the partner is {layer['gain_db']:.0f} dB louder than the top match"
        return [], {**evidence, "accepted": False, "why": why}
    if drums is not None:
        at, away = _elsewhere(library, r, b, drums, hits)
        evidence["heard"] = {"at_hits": at, "elsewhere": away}
        if away > max(at, 2):
            return [], {**evidence, "accepted": False, "why": "the partner plays elsewhere in the song"}
    if better < min_db:
        return [], {**evidence, "accepted": False, "why": f"the partner takes off less than {min_db:g} dB"}
    return [layer], {**evidence, "accepted": True}


def _role(top: list[dict], named: Counter, guess: str, b: str) -> str:
    """A template's role, from its top five matches in band b: the top
    match's family when it is matched in the snare or hat band at NCC >=
    0.95 and that family is a drum one (DRUM_FAMILIES); else the majority
    of the five's names when the top match is close (>= 0.75) and three
    agree; else the energy guess. In a 20-beat suite run, 7 hat templates
    matched to their own hat at NCC 0.976-0.997 were called kick (5) or perc
    (2): the five names split, and the guess hears the kick under the hats.
    The kick band does not tell recordings apart (an open hat's top match
    there was an fx file at 0.957, a hat template's a kick at 0.952), and a
    family that is no drum would be played and scored as a perc."""
    family = FAMILY.get(top[0]["role"], top[0]["role"])
    if b in ("snare", "hat") and top[0]["ncc"] >= 0.95 and family in DRUM_FAMILIES:
        return family
    vote, votes_n = named.most_common(1)[0]
    return vote if top[0]["ncc"] >= 0.75 and votes_n >= 3 else guess


def _kick_source(library: Library, info: dict, ranked: list[dict], sources: dict, r808: np.ndarray | None,
                 claimed) -> tuple[dict, np.ndarray | None, list[dict] | None]:
    """A kick-band template read from each other source (kick_sources): the
    best match of each, and the source whose best match is closest wins,
    unless that match is a sound another template was given (claimed(path)).
    Returns (the evidence, the winning window and ranking, or None for the
    drum stem's)."""
    g = Group(members=[tuple(m) for m in info["members"]])
    tops, best = {"drums": ranked[0]["ncc"]}, (ranked[0]["ncc"], "drums", None, None)
    for src, x in sources.items():
        t = template(x, g, LENGTH_MS["kick"])[0]
        if r808 is not None:
            t = _explained(t, template(r808, g, LENGTH_MS["kick"])[0])
        found = library.match(t, "kick", semitones=range(-4, 5), top=40)
        tops[src] = found[0]["ncc"]
        if found[0]["ncc"] > best[0] and not claimed(found[0]["path"]):
            best = (found[0]["ncc"], src, t, found)
    return {"source": best[1], "top_ncc": tops}, best[2], best[3]


def match(work: str | Path, library: Library, stems_dir: str | Path | None = None, detune: float | None = None,
          kick_sources_on: bool = False, log=print) -> dict:
    """The library's sound for each template, with a verdict.

    Every template is matched in all three bands (_search): in the snare and
    hat bands at any pitch through the speckle index (when the library has
    one), in the kick band, and for templates found in the low band in every
    band, by whole semitones. A template often holds two sounds that always
    play together (a hat on every kick): each band shows one of them.
    Identities are handed out from the strongest match down; a sample
    already given to another template goes to this one too only if it
    matches it just as well (>= 0.9: the same sound found twice, `same_as`),
    otherwise this template takes its best match in another band. With
    kick_sources_on, a template given the kick band is then also read from
    drums + bass and from the mix with the 808 taken out (kick_sources), and
    the source whose best match is closest wins (`source`). It is off by
    default: it needs fit-808's one-shot, which a full run makes after
    match, and with fit-808's synth one-shot it found 13 of 20 kicks at the
    true times against 12 from the drum stem, while in a 20-beat suite run
    both of its switches put an 808 file first, one of them pushing the
    true kick out of the top five. Last, a template matched in the snare or
    hat band gets its second sound, if it is layered (_layer).
    """
    work = Path(work)
    analysis = _load(work / "analysis.json")
    templates = analysis["templates"]
    stems = Path(stems_dir or analysis["source"]["stems"])
    drums = read(stems / "drums.wav")
    if library.speckle_info.get("ignored"):
        log(f"speckle index ignored: {library.speckle_info['ignored']}; noisy sounds by whole semitones")
    by_band: dict[str, dict[str, list[dict]]] = {}
    ranked_by: dict[tuple[str, str], list[dict]] = {}
    windows: dict[tuple[str, str], np.ndarray] = {}
    method: dict[tuple[str, str], str] = {}
    for name, info in templates.items():
        full = _template(work, name)
        by_band[name] = {}
        for role in ("kick", "snare", "hat"):
            t = windows[name, role] = full[: samples(LENGTH_MS[role])]
            ranked_by[name, role], method[name, role] = _search(library, t, role, info["band"] == "low", detune)
            by_band[name][role] = library.distinct(ranked_by[name, role], len(t))
    claims: dict[str, tuple[str, float]] = {}
    chosen: dict[str, str] = {}
    same_as: dict[str, str] = {}
    pairs = sorted(((r[0]["ncc"], name, b) for name, bands in by_band.items() for b, r in bands.items()),
                   reverse=True)
    for score, name, b in pairs:
        if name in chosen or name in same_as:
            continue
        path = by_band[name][b][0]["path"]
        if path not in claims:
            chosen[name] = b
            claims[path] = (name, score)
        elif score >= 0.9 and claims[path][1] >= 0.9:
            same_as[name] = claims[path][0]
    band_of = {name: chosen.get(name) or (same_as.get(name) and chosen.get(same_as[name]))
               or {"low": "kick"}.get(info["band"], info["band"]) for name, info in templates.items()}
    kicks = [n for n in templates if kick_sources_on and band_of[n] == "kick" and n not in same_as
             and templates[n].get("members")]
    sources, r808, source_info = kick_sources(work, analysis, stems, drums) if kicks else ({}, None, {})
    kick_source: dict[str, dict] = {}
    for name in kicks:
        kick_source[name], t, ranked = _kick_source(
            library, templates[name], ranked_by[name, "kick"], sources, r808,
            lambda path, me=name: path in claims and claims[path][0] != me)
        kick_source[name] |= source_info
        if ranked is not None:
            windows[name, "kick"], ranked_by[name, "kick"] = t, ranked
            by_band[name]["kick"] = library.distinct(ranked, len(t))
    out = {}
    for name, info in templates.items():
        b = band_of[name]
        t = windows[name, b]
        top = by_band[name][b][:5]
        named = Counter(FAMILY.get(r["role"], r["role"]) for r in top)
        role = _role(top, named, info["role_guess"], b)
        margin = round(top[0]["ncc"] - top[1]["ncc"], 3) if len(top) > 1 else top[0]["ncc"]
        candidates = {r["path"]: library.candidate(r["i"], len(t), r["semitones"]) for r in top[:3]}
        members = info.get("members") or []
        votes = per_hit_votes(drums, members, candidates, b) if members else None
        wins_share = (votes["wins"][top[0]["path"]] / max(1, sum(votes["wins"].values()))) if votes else None
        out[name] = {"role": role, "matched_in_band": b, "same_as": same_as.get(name),
                     "search": method[name, b], "library_roles_top5": dict(named),
                     "top": [_entry(r) for r in top],
                     "other_bands": {ob: {"file": Path(r[0]["path"]).name, "ncc": r[0]["ncc"],
                                          "semitones": r[0]["semitones"], "search": method[name, ob]}
                                     for ob, r in by_band[name].items() if ob != b},
                     "margin": margin, "per_hit": votes,
                     "null_test_db": {Path(p).name: null_test(t, c, b) for p, c in candidates.items()},
                     "verdict": verdict(role, top[0]["ncc"], margin, wins_share)}
        if name in kick_source:
            out[name] |= {"source": kick_source[name]["source"], "kick_source": kick_source[name]}
        if library.speckle_info.get("version"):
            out[name]["speckle_index"] = library.speckle_info["version"]
        if name in same_as:
            out[name]["verdict"] = f"same sound as {same_as[name]}"
    for name in templates:
        b, m = band_of[name], out[name]
        if name in same_as or b == "kick":
            why = f"same sound as {same_as[name]}" if name in same_as else TONAL_LAYERS
            m["layers"], m["layer_search"] = [], {"accepted": False, "why": why}
        else:
            taken = [by_band[o][band_of[o]][0]["i"] for o in templates if o != name]
            hits = np.array([s - lag for s, lag, *_ in templates[name].get("members") or []]) / SR
            m["layers"], m["layer_search"] = _layer(library, windows[name, b], b, ranked_by[name, b], taken,
                                                    drums=drums, hits=hits)
        others = ", ".join(f"{ob}: {v['file'][:22]} {v['ncc']:.2f}" for ob, v in m["other_bands"].items())
        layer = f" + {m['layers'][0]['file'][:24]} {m['layers'][0]['gain_db']:+.1f} dB" if m["layers"] else ""
        where = f"{b}, {m['source']}" if m.get("source") else b
        log(f"  {name:9s} -> {m['role']:6s} {m['top'][0]['file'][:34]:34s} st {m['top'][0]['semitones']:+.2f} "
            f"ncc {m['top'][0]['ncc']:.3f} margin {m['margin']:+.3f} [{where}] ({m['verdict']}){layer} | {others}")
    _save(work / "matches.json", out)
    return out


def _fit_notes(work: Path, analysis: dict) -> tuple[list[int], int, float | None, str | None, float]:
    """The onsets (samples) of the notes the 808 is fitted on, their written
    pitch, the settled pitch measured on them (Hz) and how, and the longest
    window (ms) they ring alone: bass808's template notes, or (an analysis
    without bass808.json) the isolated notes of the commonest pitch at the
    analysis' 808 tuning."""
    b808 = _load(work / "bass808.json") if (work / "bass808.json").is_file() else None
    tpl = (b808 or {}).get("template")
    if tpl:
        return ([int(round(t * SR)) for t in tpl["onsets_s"]], int(tpl["root_midi"]), float(tpl["hz"]),
                tpl["method"], float(min(600, int(1000 * tpl["length_s"]))))
    notes = isolated(analysis["notes_808"])
    if not notes:
        return [], 0, None, None, 600.0
    root = notes[0]["pitch"]
    return [int(n["t"] * SR) for n in notes], root, None, None, 600.0


def fit_808(work: str | Path, stems_dir: str | Path | None = None, library_808: Library | None = None,
            log=print, **fit_options) -> dict:
    """Fit the 808 on bass808's template notes (drums + bass, so the attack
    is in) and write a one-shot at the 808's settled pitch. fit_options go
    to synth808.fit (maxiter, popsize: a smaller search for tests).

    The pitch is the template's settled pitch as bass808 measured it (the
    QIFFT of its notes' sustain, 300-600 ms: on PIRAMIT C1 +11.7 cents,
    where the pYIN window read +33; or the asymptote of its fitted drop when
    no note rings that long), or the fit's f_end where the two agree within
    2 cents, or where the template only had its unsettled tail to read
    ("qifft_tail": the fit's drop extrapolates). synth808.json keeps both,
    the one used (pitch_hz, pitch_from) and its cents over root_midi."""
    work = Path(work)
    analysis = _load(work / "analysis.json")
    stems_dir = Path(stems_dir or analysis["source"]["stems"])
    mix = read(stems_dir / "drums.wav") + read(stems_dir / "bass.wav")
    starts, root, hz, method, ms = _fit_notes(work, analysis)
    if not starts:
        raise ValueError("No 808 note rings long enough on its own to fit")
    g = group(band(mix, "808"), starts=starts, threshold=-1.0, cluster_ms=min(200, ms), min_db=-90)[0]
    ref, used = template(mix, g, ms)
    np.save(work / "templates" / "808.npy", ref)
    # the template's own pitch, or the 808's tuning (analyses written before bass808: the track's)
    f1 = hz or float(librosa.midi_to_hz(root + analysis["tuning"].get("808", analysis["tuning"]["consensus"])))
    log(f"808: {used} notes at MIDI {root}, {f1:.2f} Hz ({method or 'tuning'}); fitting (1-3 min)")
    params, info = fit(ref, f1, **fit_options)
    apart = 1200 * np.log2(params.f_end_hz / f1)
    if abs(apart) <= 2 or method == "qifft_tail":
        pitch, source = params.f_end_hz, "agree" if abs(apart) <= 2 else "fit"
    else:
        pitch, source = f1, method or "tuning"
    shot = one_shot(params, pitch)
    name = "808 (synth).wav"
    write(work / name, shot, peak=None)
    result = {"params": params.to_dict(), "glide_semitones": round(params.glide_semitones, 2),
              "root_midi": root, "pitch_hz": round(pitch, 4), "pitch_from": source,
              "cents": round(1200 * np.log2(pitch / 440) + 6900 - 100 * root, 1),
              "measured_hz": round(f1, 4), "measured_by": method or "tuning", "fit_vs_measured_cents": round(apart, 2),
              "fit_notes": used, "fit": info, "file": str(work / name), "file_name": name,
              "compare": {"synth": compare(ref, laid_out(params, len(ref)), f1)}}
    if library_808 is not None:
        ranked = library_808.match(ref, "808", semitones=range(-12, 13), top=10, prefilter=60)
        best = ranked[0]
        cand = library_808.candidate(best["i"], len(ref), best["semitones"])
        result["compare"]["best_library"] = {**compare(ref, cand, f1), "file": Path(best["path"]).name,
                                             "semitones": best["semitones"],
                                             "margin": round(best["ncc"] - ranked[1]["ncc"], 3)}
    log(f"808: one-shot at {pitch:.3f} Hz ({source}; fit f_end {params.f_end_hz:.3f} Hz, {apart:+.1f} cents "
        f"from the measured {f1:.3f})")
    log(json.dumps(result["compare"]))
    _save(work / "synth808.json", result)
    return result


def melodic(work: str | Path, stems_dir: str | Path | None = None, min_hz: float | None = None,
            snap_ms: float | None = None, voting: bool = True, drop_below: float | None = None, log=print) -> dict:
    """Notes per tonal stem and chords (melodic.py): Basic Pitch runs once per
    stem in its own environment, on the stem moved onto A440 by its tuning;
    its notes and posteriorgrams go to basicpitch/. The minimum frequency is
    just above the 808's usual range for the stems it bleeds into
    (melodic.FLOOR_STEMS), or min_hz (as heard) for every stem when given.

    The tuning is measured here on each stem (grid.tuning, as analyze does):
    an analysis.json written before the stable-peak estimator has other
    values (HAYIR DEMELIYDIM's: +21 to +35 cents where the stems measure -2
    to -5), and taking those out moved in-tune stems off A440: the other
    stem kept 165 of its 520 notes."""
    from . import melodic as mel
    from .basicpitch import PRESETS, transcribe
    work = Path(work)
    analysis = _load(work / "analysis.json")
    stems_dir = Path(stems_dir or analysis["source"]["stems"])
    stems = [s for s in mel.MELODIC_STEMS if (stems_dir / f"{s}.wav").is_file()]
    if not stems:
        raise ValueError(f"No tonal stem ({', '.join(mel.MELODIC_STEMS)}) in {stems_dir}")
    audio = {s: read(stems_dir / f"{s}.wav") for s in stems}
    tunings = {s: round(float(tuning(audio[s])), 3) for s in stems}
    stale = [f"{s} {100 * mel.stem_tuning(analysis, s):+.0f}" for s in stems
             if abs(tunings[s] - mel.stem_tuning(analysis, s)) > 0.05]
    if stale:
        log(f"analysis.json has other tunings ({', '.join(stale)} cents; an older analyze?): measured on the stems")
    floor_hz, top = (min_hz, None) if min_hz else mel.floor_above_808(analysis["notes_808"])
    files = {}
    for s in stems:
        # the 808's floor is a semitone (on A440 already); a given min_hz is as heard
        hz = min_hz * 2 ** (-tunings[s] / 12) if min_hz else (floor_hz if s in mel.FLOOR_STEMS else None)
        files[s] = {"tuning": tunings[s], "min_hz": round(hz, 2) if hz else None}
    floored = [s for s in stems if files[s]["min_hz"]]
    log(f"melodic: {', '.join(stems)}; tuning taken out: "
        + ", ".join(f"{s} {100 * files[s]['tuning']:+.0f} cents" for s in stems) + "; minimum frequency "
        + (f"{floor_hz} Hz for {', '.join(floored)}" if floored else "none")
        + (f" (above the 808's MIDI {top})" if top is not None and floored else ""))
    run = transcribe([stems_dir / f"{s}.wav" for s in stems], work / "basicpitch", PRESETS, files)
    transcribed = {s: _load(Path(run["stems"][s]["notes"])) for s in stems}
    pgs = {}
    for s in stems:
        if s in mel.CHORD_STEMS:
            with np.load(run["stems"][s]["posteriorgram"]) as z:
                pgs[s] = z["note"].astype(np.float32)
    out = mel.build(analysis, transcribed, audio, pgs, snap_ms, voting, drop_below, tunings)
    floor_from = "808" if top is not None else ("parameter" if min_hz else None)
    out["source"] = {"stems": str(stems_dir), "model": run["model"], "backend": run["backend"],
                     "python": run["python"], "settings": {"presets": PRESETS, "files": files},
                     "min_hz_from": floor_from}
    for s, info in out["stems"].items():
        changed = ", ".join(f"{r['section']}: period {r['period_bars']} bars, {r['restored']} restored "
                            f"({r['too_quiet']} too quiet), {r['dropped']} dropped"
                            for r in info["voting"]) or "no loop"
        log(f"  {s:7s} {len(info['notes']):4d} notes ({changed})")
    if "chords" in out:
        log(f"  chords: {len(out['chords']['segments'])} segments")
    _save(work / "melodic.json", out)
    return out


def plan(work: str | Path, stems_dir: str | Path | None = None, log=print) -> dict:
    from .live import make_plan, write_midi
    work = Path(work)
    analysis = _load(work / "analysis.json")
    matches = _load(work / "matches.json") if (work / "matches.json").is_file() else {}
    synth = _load(work / "synth808.json") if (work / "synth808.json").is_file() else None
    melodic_notes = _load(work / "melodic.json") if (work / "melodic.json").is_file() else None
    p = make_plan(analysis, matches, synth, stems_dir or analysis["source"]["stems"], melodic=melodic_notes)
    _save(work / "plan.json", p)
    written = write_midi(p, work / "midi")
    tonal = ", ".join(f"{len(t['notes'])} {s}" for s, t in p["tracks"].get("melodic", {}).items())
    log(f"plan: {len(p['tracks']['drums']['notes'])} drum notes on {len(p['tracks']['drums']['pads'])} pads, "
        f"{len(p['tracks']['808']['notes'])} 808 notes" + (f", notes: {tonal}" if tonal else "")
        + f"; MIDI: {', '.join(str(w) for w in written)}")
    return p


__all__ = ["analyze", "melodic", "match", "fit_808", "plan", "Grid", "Template", "Params", "LENGTH_MS"]
