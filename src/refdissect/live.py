"""From an analysis to Ableton Live: a plan (notes in beats, sounds, levels,
locators), MIDI files, and the build itself through ableton-live-mcp.

The plan is plain JSON, so it can be read, edited and rebuilt: Live's Trial
cannot save, and a rebuild takes about five seconds.

Live facts the build relies on (measured on Live 12.4, see
docs/07-building-in-live.md):
- Simpler loads samples at Volume -12 dB; the build sets 0 dB.
- Simpler plays a sample at its own pitch on C3 (MIDI 60). The 808 notes
  keep their real MIDI numbers (C1 = 24 here), so Transpose is +36.
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
import os
from pathlib import Path

import mido
import numpy as np

from .audio import SR, read, rms_db, transpose, trim_onset

PADS = {"kick": [36, 35, 47, 45], "snare": [38, 40, 39, 37, 41], "hat": [42, 44, 46, 49, 51],
        "perc": [43, 48, 50, 52, 53]}
STEMS_MUTED = {"drums", "bass", "vocals"}  # replaced by ours, or not part of the beat
DEFAULT_USER_LIBRARY = Path.home() / "Music" / "Ableton" / "User Library"
GENERATED = Path("Samples") / "AbletonMCP Generated"


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


def velocities(gains: list[float], low: int = 35) -> list[int]:
    g = np.asarray(gains, dtype=float)
    if not len(g):
        return []
    ref = np.percentile(g, 90) or 1.0
    return [int(v) for v in np.clip(np.round(127 * g / ref), low, 127)]


def make_plan(analysis: dict, matches: dict, synth: dict | None, stems_dir: str | Path,
              headroom_db: float = 4.0, min_hits: int = 12, min_clean: float = 0.5,
              min_ncc: float = 0.75) -> dict:
    """Everything the build needs, decided here so it can be checked first.

    A detected sound gets a Drum Rack pad when it was detected often and
    cleanly and the library has something close (NCC >= min_ncc), or when
    it is a kick or snare (the backbone, played with the closest sound even
    if far). A sound found twice (same_as) shares the first one's pad."""
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
            pads[str(note)] = {"template": name, "path": best["path"], "transpose": int(round(best["semitones"])),
                               "ncc": best["ncc"], "verdict": m["verdict"], "role": role}
        else:
            continue  # nothing in the library is close: left out rather than played wrong
        pad_of[name] = note
        hits = analysis["events"][name]
        for e, v in zip(hits, velocities([h["gain"] for h in hits])):
            drum_notes.append({"pitch": note, "start_time": round(e["t"] * beats_per_second, 4),
                               "duration": 1.0, "velocity": v})
    notes808 = analysis["notes_808"]
    peaks = np.array([n["peak"] for n in notes808]) if notes808 else np.array([1.0])
    med = float(np.median(peaks)) or 1.0
    bass_notes = [{"pitch": n["pitch"], "start_time": round(n["t"] * beats_per_second, 4),
                   "duration": round(max(0.25, (n["end"] - n["t"]) * beats_per_second), 4),
                   "velocity": int(np.clip(round(115 * n["peak"] / med), 70, 127))} for n in notes808]
    plan = {"tempo": bpm, "length_beats": float(length_beats),
            "tracks": {"drums": {"pads": pads, "notes": no_overlaps(drum_notes, 1.0)},
                       "808": {"notes": no_overlaps(bass_notes, None)}},
            "stems": {}, "locators": [], "levels": {}}
    if synth:
        root = synth["root_midi"]
        # named after the song: the User Library keeps one file per name
        song = Path(stems_dir).resolve().parent.name
        plan["tracks"]["808"].update(sample=f"{song} {synth['file_name']}", sample_path=synth["file"],
                                     transpose=int(60 - root), root_midi=root)
    stems_dir = Path(stems_dir)
    for stem in ("drums", "bass", "other", "piano", "guitar", "vocals"):
        path = stems_dir / f"{stem}.wav"
        if path.is_file():
            plan["stems"][stem] = {"path": str(path), "mute": stem in STEMS_MUTED}
    for s in analysis.get("sections", []):
        plan["locators"].append([f"{s['name']} ({s['start_bar']})", 4.0 * (s["start_bar"] - 1)])
    plan["levels"] = estimate_levels(plan, analysis, headroom_db)
    return plan


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
    a linear level), compared by RMS over the whole track."""
    bps = plan["tempo"] / 60
    n = int(analysis["source"]["duration_s"] * SR)
    samples_by_pad = {}
    for note, pad in plan["tracks"]["drums"]["pads"].items():
        try:
            samples_by_pad[int(note)] = transpose(trim_onset(read(pad["path"])), pad["transpose"])
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
        shot = read(track808["sample_path"])
        cache: dict[int, np.ndarray] = {}

        def sample_808(e):
            if e["pitch"] not in cache:
                cache[e["pitch"]] = transpose(shot, e["pitch"] - track808["root_midi"])
            return cache[e["pitch"]]
        ours = _render(track808["notes"], sample_808, n, bps, True)
        out["808_track_db"] = round(rms_db(read(stems["bass"]["path"])) - rms_db(ours) - headroom_db, 1)
    out["stems_track_db"] = -headroom_db
    return out


def write_midi(plan: dict, out_dir: str | Path, ppq: int = 480) -> list[Path]:
    """drums.mid and 808.mid at the plan's tempo (for import_midi_file)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for key in ("drums", "808"):
        notes = plan["tracks"][key]["notes"]
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

async def build(plan: dict, user_library: str | Path = DEFAULT_USER_LIBRARY, headroom: bool = True,
                log=print) -> dict:
    """Build the plan in the open Live set, through ableton-live-mcp's tools.
    Refuses unless every track in the set is empty (nothing gets overwritten);
    empty tracks are reused, the rest created."""
    import asyncio
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

    def lom(command, **params):
        return et._get_connection().send_command(command, params)

    async def browser_id(name, wait=40):
        for _ in range(wait):
            found = json.loads(await call("browser_search", query=os.path.splitext(name)[0], category="all",
                                          max_results=10))
            exact = [r for r in found["results"] if r["name"] == name]
            if exact:
                return exact[0]["id"]
            await asyncio.sleep(1.0)
        raise RuntimeError(f"Live's browser does not list {name}")

    def set_simpler(path, semitones):
        names = [p.get("name") for p in lom("lom_get", path=f"{path} parameters", properties=["name"])["items"]]
        result = {}
        if "Volume" in names:
            i = names.index("Volume")
            lom("param_set_display", path=f"{path} parameters {i}", value="0.0 dB")
            result["Volume"] = lom("lom_get", path=f"{path} parameters {i}", properties=["$display"])["$display"]
        if "Transpose" in names:
            i = names.index("Transpose")
            lom("lom_set", path=f"{path} parameters {i}", property="value", value=float(semitones))
            result["Transpose"] = lom("lom_get", path=f"{path} parameters {i}", properties=["$display"])["$display"]
        if lom("lom_get", path=f"{path} sample", properties=["warping"]).get("warping"):
            lom("lom_set", path=f"{path} sample", property="warping", value=False)
        return result

    overview = json.loads(await call("get_session_overview"))
    busy = [t["name"] for t in overview["tracks"]
            if t["devices"] or t["session_clips"] or t["arrangement_clip_count"]]
    if busy:
        raise RuntimeError(f"The set is not empty ({', '.join(busy)}); open a new set first")
    free = {"midi": [t["index"] for t in overview["tracks"] if t["type"] == "midi"],
            "audio": [t["index"] for t in overview["tracks"] if t["type"] == "audio"]}
    count = len(overview["tracks"])
    wanted = [("808", "midi", "808 (synth)"), ("drums", "midi", "Drums (library)")]
    wanted += [(f"stem:{s}", "audio", f"REF {s}") for s in plan["stems"]]
    index: dict[str, int] = {}
    for key, kind, name in wanted:
        if free[kind]:
            index[key] = free[kind].pop(0)
            await call("manage_track", action="rename", track=index[key], name=name)
        else:
            await call("manage_track", action=f"create_{kind}", name=name)
            index[key] = count
            count += 1
    await call("set_song_properties", tempo=plan["tempo"], loop=False)
    report: dict = {"tracks": {k: v for k, v in index.items()}}
    # 808: the one-shot through the User Library and the browser
    t808 = plan["tracks"]["808"]
    if "sample_path" in t808:
        target = Path(user_library).expanduser() / GENERATED / t808["sample"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(t808["sample_path"], target)
        await call("browser_load", track=index["808"], item=await browser_id(t808["sample"]))
        report["808"] = set_simpler(f"live_set tracks {index['808']} devices 0", t808["transpose"])
    # drums: a Drum Rack, one pad per sound
    await call("add_device", track=index["drums"], device="Drum Rack")
    report["pads"] = {}
    for note, pad in plan["tracks"]["drums"]["pads"].items():
        await call("browser_load", track=index["drums"], item=await browser_id(Path(pad["path"]).name),
                   drum_pad=int(note))
        report["pads"][note] = set_simpler(f"live_set tracks {index['drums']} devices 0 drum_pads {note} "
                                           f"chains 0 devices 0", pad["transpose"])
    for key, name in (("808", "808"), ("drums", "Drums")):
        notes = plan["tracks"][key]["notes"]
        if not notes:
            continue
        await call("manage_clip", action="create_arrangement_clip", track=index[key], start_time=0.0,
                   length=plan["length_beats"], name=name)
        for a in range(0, len(notes), 200):
            await call("edit_clip_notes", track=index[key], mode="add", notes=notes[a:a + 200],
                       arrangement_clip_index=0)
    report["stems"] = {}
    for stem, info in plan["stems"].items():
        tp = f"live_set tracks {index['stem:' + stem]}"
        clip = f"{tp} arrangement_clips 0"
        lom("lom_call", path=tp, method="create_audio_clip", args=[info["path"], 0.0])
        lom("lom_set", path=clip, property="warping", value=False)
        # Live's warp analysis moves the clip's start to the first transient
        # (21-30 ms on these stems, 3.6 s on one with a quiet intro) and the
        # start stays there with warping off: back to the file's start, or the
        # stem plays early against the notes. loop_start carries start_marker.
        lom("lom_set", path=clip, property="loop_start", value=0.0)
        if lom("lom_get", path=clip, properties=["start_marker"]).get("start_marker"):
            lom("lom_set", path=clip, property="start_marker", value=0.0)
        report["stems"][stem] = lom("lom_get", path=clip, properties=["start_marker", "end_time"])
    levels = plan["levels"]
    faders = {index["808"]: levels.get("808_track_db", -levels["headroom_db"]),
              index["drums"]: levels.get("drums_track_db", -levels["headroom_db"])}
    for stem in plan["stems"]:
        faders[index["stem:" + stem]] = levels["stems_track_db"]
    for track, db in faders.items():
        mute = any(index.get("stem:" + s) == track and info["mute"] for s, info in plan["stems"].items())
        await call("set_track_mixer", track=track, volume_db=round(min(6.0, db), 1), mute=mute)
    report["faders"] = faders
    if plan["locators"]:
        report["locators"] = et.place_locators([(n, float(round(b))) for n, b in plan["locators"]])
    await call("select_in_live", view="arrangement", track=index["808"])
    await call("transport", action="jump_to", time=0.0)
    log(json.dumps(report, ensure_ascii=False, default=str))
    return report
