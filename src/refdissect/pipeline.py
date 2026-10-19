"""The steps in order, each writing what it found to a work folder:

    analyze(stems)   -> analysis.json, templates/*.npy|wav    (grid, tuning, drums, 808, sections)
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

from . import describe as describe_mod
from .audio import SR, band, read, samples, write
from .detect import LENGTH_MS, Template, detect, discover, isolated, notes_808, peak_score_histogram, pitch_offset
from .grid import Grid, fit_grid, offsets, tuning
from .hits import group, template
from .library import Library, null_test, per_hit_votes, verdict
from .structure import bar_activity, sections
from .synth808 import Params, compare, fit, laid_out, one_shot

TONAL_STEMS = ("bass", "other", "piano", "vocals", "guitar")
# library roles grouped into the drum families a Drum Rack is laid out by
FAMILY = {"clap": "snare", "rim": "snare", "open_hat": "hat", "crash/ride": "hat", "808": "kick"}


def _save(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    return path


def _load(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def analyze(stems_dir: str | Path, work: str | Path, bpm_range=(100.0, 180.0), log=print) -> dict:
    stems_dir, work = Path(stems_dir), Path(work)
    drums, bass = read(stems_dir / "drums.wav"), read(stems_dir / "bass.wav")
    duration = len(drums) / SR
    log(f"stems: {duration:.1f} s")
    grid, fit_info = fit_grid(drums, bpm_range)
    log(f"grid: {grid.bpm} BPM, first beat at {grid.phase} s")
    tunings = {s: tuning(read(stems_dir / f"{s}.wav")) for s in TONAL_STEMS
               if (stems_dir / f"{s}.wav").is_file()}
    consensus = round(float(np.median(list(tunings.values()))), 2) + 0.0 if tunings else 0.0  # no -0.0
    log(f"tuning (semitones from A440): {tunings} -> {consensus:+.2f}")
    templates = discover(drums, grid)
    events = detect(drums, templates)
    tdir = work / "templates"
    tdir.mkdir(parents=True, exist_ok=True)
    out_templates = {}
    for tp in templates:
        np.save(tdir / f"{tp.name}.npy", tp.audio)
        write(tdir / f"{tp.name}.wav", tp.audio)
        out_templates[tp.name] = {"role_guess": tp.role, "band": tp.band, "template_hits": tp.hits,
                                  "slots": tp.slots,
                                  "detected_hits": len(events[tp.name]),
                                  "peak_ncc_histogram_0.3_to_1.0": peak_score_histogram(drums, tp),
                                  "describe": describe_mod.describe(band(tp.audio[:samples(LENGTH_MS[tp.role])],
                                                                         tp.role)),
                                  "clean": round(float(np.mean([e["ncc"] >= 0.8 for e in events[tp.name]]))
                                                 if events[tp.name] else 0.0, 3),
                                  "members": tp.group.members if tp.group else []}
        log(f"  {tp.name:8s} ({tp.band} band, looks like a {tp.role}) {tp.hits:3d} hits in the template, "
            f"{len(events[tp.name]):3d} detected")
    notes = notes_808(bass, consensus)
    offset_808, agreement_808 = pitch_offset([n["midi_float"] for n in notes])
    tuning_808 = offset_808 if agreement_808 >= 0.5 else consensus
    log(f"808: {len(notes)} notes, tuned {100 * offset_808:+.0f} cents (agreement {agreement_808:.2f})")
    n_bars = int(np.ceil((duration - grid.phase) / grid.bar))
    stems = {s: read(stems_dir / f"{s}.wav") for s in ("other", "piano", "vocals")
             if (stems_dir / f"{s}.wav").is_file()}
    rows = bar_activity(grid, n_bars, events, notes, stems)
    secs = sections(rows)
    log("sections: " + ", ".join(f"{s['name']} {s['start_bar']}-{s['end_bar']}" for s in secs))
    analysis = {"source": {"stems": str(stems_dir), "duration_s": round(duration, 3)},
                "grid": {**grid.to_dict(), **fit_info, "offsets_by_beat": offsets(drums, grid)},
                "tuning": {"per_stem": tunings, "consensus": consensus,
                           "a4_hz": round(440 * 2 ** (consensus / 12), 1),
                           "808": tuning_808, "808_agreement": agreement_808},
                "templates": out_templates, "events": events, "notes_808": notes,
                "bars": [{**r, "sounds": sorted(r["sounds"]), "stems": sorted(r["stems"])} for r in rows],
                "sections": secs}
    _save(work / "analysis.json", analysis)
    return analysis


def _template(work: Path, name: str) -> np.ndarray:
    return np.load(work / "templates" / f"{name}.npy")


def match(work: str | Path, library: Library, stems_dir: str | Path | None = None, detune: float | None = None,
          log=print) -> dict:
    """The library's sound for each template, with a verdict.

    Every template is matched in all three bands. A template often holds two
    sounds that always play together (a hat on every kick): each band shows
    one of them. Identities are handed out from the strongest match down; a
    sample already given to another template goes to this one too only if it
    matches it just as well (>= 0.9: the same sound found twice, `same_as`),
    otherwise this template takes its best match in another band.
    """
    work = Path(work)
    analysis = _load(work / "analysis.json")
    drums = read(Path(stems_dir or analysis["source"]["stems"]) / "drums.wav")
    by_band: dict[str, dict[str, list[dict]]] = {}
    for name in analysis["templates"]:
        full = _template(work, name)
        by_band[name] = {}
        for role in ("kick", "snare", "hat"):
            t = full[: samples(LENGTH_MS[role])]
            semis = range(-4, 5) if role == "kick" else range(-3, 4)
            best = None
            for base in [0.0] + ([detune] if detune else []):
                ranked = library.distinct(library.match(t, role, semitones=semis, base=base, top=40), len(t))
                if best is None or ranked[0]["ncc"] > best[0]["ncc"]:
                    best = ranked
            by_band[name][role] = best
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
    out = {}
    for name, info in analysis["templates"].items():
        b = chosen.get(name) or (same_as.get(name) and chosen.get(same_as[name])) \
            or {"low": "kick"}.get(info["band"], info["band"])
        full = _template(work, name)
        t = full[: samples(LENGTH_MS[b])]
        top = by_band[name][b][:5]
        named = Counter(FAMILY.get(r["role"], r["role"]) for r in top)
        vote, votes_n = named.most_common(1)[0]
        role = vote if top[0]["ncc"] >= 0.75 and votes_n >= 3 else info["role_guess"]
        margin = round(top[0]["ncc"] - top[1]["ncc"], 3) if len(top) > 1 else top[0]["ncc"]
        candidates = {r["path"]: library.candidate(r["i"], len(t), r["semitones"]) for r in top[:3]}
        members = info.get("members") or []
        votes = per_hit_votes(drums, members, candidates, b) if members else None
        wins_share = (votes["wins"][top[0]["path"]] / max(1, sum(votes["wins"].values()))) if votes else None
        out[name] = {"role": role, "matched_in_band": b, "same_as": same_as.get(name),
                     "library_roles_top5": dict(named),
                     "top": [{"path": r["path"], "file": Path(r["path"]).name, "role": r["role"], "ncc": r["ncc"],
                              "semitones": r["semitones"], "lag_ms": r["lag_ms"]} for r in top],
                     "other_bands": {ob: {"file": Path(r[0]["path"]).name, "ncc": r[0]["ncc"],
                                          "semitones": r[0]["semitones"]}
                                     for ob, r in by_band[name].items() if ob != b},
                     "margin": margin, "per_hit": votes,
                     "null_test_db": {Path(p).name: null_test(t, c, b) for p, c in candidates.items()},
                     "verdict": verdict(role, top[0]["ncc"], margin, wins_share)}
        if name in same_as:
            out[name]["verdict"] = f"same sound as {same_as[name]}"
        others = ", ".join(f"{ob}: {v['file'][:22]} {v['ncc']:.2f}" for ob, v in out[name]["other_bands"].items())
        log(f"  {name:9s} -> {role:6s} {Path(top[0]['path']).name[:34]:34s} st {top[0]['semitones']:+.0f} "
            f"ncc {top[0]['ncc']:.3f} margin {margin:+.3f} [{b}] ({out[name]['verdict']}) | {others}")
    _save(work / "matches.json", out)
    return out


def fit_808(work: str | Path, stems_dir: str | Path | None = None, library_808: Library | None = None,
            log=print) -> dict:
    """Fit the 808 on its isolated root notes (drums + bass, so the attack is
    in) and write a one-shot at the 808's own tuning."""
    work = Path(work)
    analysis = _load(work / "analysis.json")
    stems_dir = Path(stems_dir or analysis["source"]["stems"])
    mix = read(stems_dir / "drums.wav") + read(stems_dir / "bass.wav")
    notes = isolated(analysis["notes_808"])
    if not notes:
        raise ValueError("No 808 note rings long enough on its own to fit")
    root = notes[0]["pitch"]
    g = group(band(mix, "808"), starts=[int(n["t"] * SR) for n in notes], threshold=-1.0, cluster_ms=200,
              min_db=-90)[0]
    ref, used = template(mix, g, 600)
    np.save(work / "templates" / "808.npy", ref)
    # the 808's own tuning (analyses written before it was measured: the track's)
    f1 = float(librosa.midi_to_hz(root + analysis["tuning"].get("808", analysis["tuning"]["consensus"])))
    log(f"808: {used} isolated notes at MIDI {root}, {f1:.2f} Hz; fitting (1-3 min)")
    params, info = fit(ref, f1)
    shot = one_shot(params, f1)
    name = "808 (synth).wav"
    write(work / name, shot, peak=None)
    result = {"params": params.to_dict(), "glide_semitones": round(params.glide_semitones, 2),
              "root_midi": root, "pitch_hz": round(f1, 2), "isolated_notes": used, "fit": info,
              "file": str(work / name), "file_name": name,
              "compare": {"synth": compare(ref, laid_out(params, len(ref)), f1)}}
    if library_808 is not None:
        ranked = library_808.match(ref, "808", semitones=range(-12, 13), top=10, prefilter=60)
        best = ranked[0]
        cand = library_808.candidate(best["i"], len(ref), best["semitones"])
        result["compare"]["best_library"] = {**compare(ref, cand, f1), "file": Path(best["path"]).name,
                                             "semitones": best["semitones"],
                                             "margin": round(best["ncc"] - ranked[1]["ncc"], 3)}
    log(json.dumps(result["compare"]))
    _save(work / "synth808.json", result)
    return result


def melodic(work: str | Path, stems_dir: str | Path | None = None, min_hz: float | None = None,
            snap_ms: float | None = None, voting: bool = True, drop_below: float | None = None, log=print) -> dict:
    """Notes per tonal stem and chords (melodic.py): Basic Pitch runs once per
    stem in its own environment, its notes and posteriorgrams go to
    basicpitch/. The minimum frequency is just above the 808's highest note
    unless min_hz is given."""
    from . import melodic as mel
    from .basicpitch import PRESETS, transcribe
    work = Path(work)
    analysis = _load(work / "analysis.json")
    stems_dir = Path(stems_dir or analysis["source"]["stems"])
    stems = [s for s in mel.MELODIC_STEMS if (stems_dir / f"{s}.wav").is_file()]
    if not stems:
        raise ValueError(f"No tonal stem ({', '.join(mel.MELODIC_STEMS)}) in {stems_dir}")
    floor_hz, top = (min_hz, None) if min_hz else mel.floor_above_808(analysis["notes_808"])
    log(f"melodic: {', '.join(stems)}; minimum frequency {floor_hz or 'none'}"
        + (f" Hz (above the 808's MIDI {top})" if top is not None else ""))
    settings = {name: {**preset, "min_hz": floor_hz} for name, preset in PRESETS.items()}
    run = transcribe([stems_dir / f"{s}.wav" for s in stems], work / "basicpitch", settings)
    transcribed = {s: _load(Path(run["stems"][s]["notes"])) for s in stems}
    pgs = {}
    for s in stems:
        if s in mel.CHORD_STEMS:
            with np.load(run["stems"][s]["posteriorgram"]) as z:
                pgs[s] = z["note"].astype(np.float32)
    out = mel.build(analysis, transcribed, {s: read(stems_dir / f"{s}.wav") for s in stems}, pgs, snap_ms, voting,
                    drop_below)
    floor_from = "808" if top is not None else ("parameter" if min_hz else None)
    out["source"] = {"stems": str(stems_dir), "model": run["model"], "backend": run["backend"],
                     "python": run["python"], "settings": settings, "min_hz_from": floor_from}
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
