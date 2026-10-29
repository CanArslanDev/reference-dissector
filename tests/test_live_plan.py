import re
from pathlib import Path

import mido
import numpy as np
import pytest
import soundfile as sf

from refdissect.audio import PRE, SR
from refdissect.live import make_plan, no_overlaps, recurring, velocities, write_midi


def test_no_overlaps_and_velocities():
    notes = [{"pitch": 38, "start_time": 0.0, "duration": 1.0, "velocity": 100},
             {"pitch": 38, "start_time": 0.5, "duration": 1.0, "velocity": 100},
             {"pitch": 42, "start_time": 0.25, "duration": 1.0, "velocity": 90}]
    out = no_overlaps(notes, 1.0)
    first = next(n for n in out if n["pitch"] == 38 and n["start_time"] == 0.0)
    assert first["duration"] < 0.5
    assert velocities([0.5, 1.0, 1.0]) == [64, 127, 127]
    # under Live 12.4.6's law at "Vol < Vel" 35 %: half the gain is 6.02 dB down, 43 steps of 0.14 dB; the
    # floor (35/127 of the 90th percentile) is 11.2 dB down
    assert velocities([0.5, 1.0, 1.0], vol_vel=0.35) == [84, 127, 127]
    assert velocities([0.01, 1.0], vol_vel=0.35) == [47, 127]


def _fixture(tmp_path):
    snare = tmp_path / "snare.wav"
    sf.write(snare, np.random.default_rng(0).standard_normal(SR // 4) * 0.5, SR)
    for stem in ("drums", "bass", "other"):
        sf.write(tmp_path / f"{stem}.wav", np.random.default_rng(1).standard_normal(SR * 4) * 0.1, SR)
    shot = tmp_path / "808 (synth).wav"
    sf.write(shot, np.sin(2 * np.pi * 33.2 * np.arange(SR) / SR) * 0.9, SR)
    analysis = {
        "grid": {"bpm": 120.0}, "source": {"duration_s": 4.0},
        "templates": {"sound_1": {"detected_hits": 20, "clean": 1.0},
                      "sound_2": {"detected_hits": 20, "clean": 1.0}},
        "events": {"sound_1": [{"t": 0.5 * k, "gain": 1.0} for k in range(8)],
                   "sound_2": [{"t": 0.25 + 0.5 * k, "gain": 0.5} for k in range(8)]},
        "notes_808": [{"t": 0.0, "end": 1.0, "pitch": 24, "peak": 1.0}],
        "sections": [{"name": "A", "start_bar": 1, "end_bar": 2}]}
    top = [{"path": str(snare), "ncc": 0.97, "semitones": -1.0}, {"path": str(snare), "ncc": 0.5, "semitones": 0.0}]
    matches = {"sound_1": {"role": "snare", "top": top, "verdict": "same sample", "same_as": None},
               "sound_2": {"role": "snare", "top": top, "verdict": "same sound as sound_1", "same_as": "sound_1"}}
    synth = {"root_midi": 24, "file_name": shot.name, "file": str(shot)}
    return analysis, matches, synth


def test_plan(tmp_path):
    analysis, matches, synth = _fixture(tmp_path)
    plan = make_plan(analysis, matches, synth, tmp_path)
    pads = plan["tracks"]["drums"]["pads"]
    assert list(pads) == ["38"] and pads["38"]["transpose"] == -1
    notes = plan["tracks"]["drums"]["notes"]
    assert len(notes) == 16 and {n["pitch"] for n in notes} == {38}  # same_as shares the pad
    assert notes[1]["start_time"] == 0.5  # 0.25 s at 120 BPM
    assert plan["tracks"]["808"]["transpose"] == 36  # MIDI 24 plays the sample's own pitch
    assert plan["stems"]["drums"]["mute"] and not plan["stems"]["other"]["mute"]
    assert plan["locators"] == [["A (1)", 0.0]]
    assert "drums_track_db" in plan["levels"] and "808_track_db" in plan["levels"]


def test_drum_notes_go_where_their_hits_attack(tmp_path):
    """A template whose hits start 19 ms before its window says (attack_s
    under PRE: detect.starts_before) has its notes there: at t_attack, or
    at t plus that lead when a hit carries only t."""
    analysis, matches, synth = _fixture(tmp_path)
    analysis["templates"]["sound_1"]["attack_s"] = PRE / SR - 0.019
    analysis["events"]["sound_1"] = [{"t": 0.1 + 0.5 * k, "gain": 1.0} for k in range(8)]
    first = min(n["start_time"] for n in make_plan(analysis, matches, synth, tmp_path)["tracks"]["drums"]["notes"])
    assert first == pytest.approx((0.1 - 0.019) * 2, abs=1e-4)  # beats at 120 BPM
    analysis["events"]["sound_1"][0]["t_attack"] = 0.09
    first = min(n["start_time"] for n in make_plan(analysis, matches, synth, tmp_path)["tracks"]["drums"]["notes"])
    assert first == pytest.approx(0.18, abs=1e-4)


def test_a_pads_sample_attacks_where_its_hit_did(tmp_path):
    """A note starts as long before its hit's t_attack as the pad's sample
    takes to attack as Simpler plays it (pad_lead), so the plan, timed as
    Live plays it (evaluate.estimate_from_plan), attacks where the analysis
    heard each hit; a sample with 6 ms in front of its attack played every
    hit that much late."""
    from refdissect import evaluate
    analysis, matches, synth = _fixture(tmp_path)
    late = tmp_path / "late.wav"
    sf.write(late, np.r_[np.zeros(int(0.006 * SR)), np.random.default_rng(2).standard_normal(SR // 4) * 0.5], SR)
    matches["sound_1"]["top"] = [{"path": str(late), "ncc": 0.97, "semitones": -1.0}]
    del matches["sound_2"]
    hits = [0.1 + 0.5 * k for k in range(8)]
    analysis["events"]["sound_1"] = [{"t": t, "t_attack": t, "gain": 1.0} for t in hits]
    est = evaluate.estimate_from_plan(make_plan(analysis, matches, synth, tmp_path), root=tmp_path)
    played = sorted(e["t_attack"] for e in est["drums"])
    assert np.max(np.abs(np.array(played) - hits)) < 1e-4


@pytest.mark.parametrize("named", ["sound_1", "sound_3"])
def test_a_sound_is_as_clean_as_the_templates_merged_into_it(tmp_path, named):
    """layers.explain merged a clean template into a less clean one of the
    same sound (layers-2: the low-band snare, 16 matched-filter peaks at NCC
    0.9 or more, into the snare-band one, clean 0.44): the sound gets its
    pad whichever of the two carries the hits."""
    analysis, matches, synth = _fixture(tmp_path)
    other = "sound_3" if named == "sound_1" else "sound_1"
    analysis["templates"] = {named: {"detected_hits": 16, "clean": 0.44},
                             other: {"detected_hits": 0, "clean": 0.0,
                                     "peak_ncc_histogram_0.3_to_1.0": [5, 1, 0, 0, 0, 0, 16]}}
    analysis["events"] = {named: [{"t": 0.5 * k, "gain": 1.0} for k in range(8)], other: []}
    analysis["layers"] = {"merged": {other: {"into": named, "offset_ms": -0.25}}}
    matches = {named: matches["sound_1"], other: {**matches["sound_1"], "verdict": f"same sound as {named}",
                                                  "same_as": named}}
    plan = make_plan(analysis, matches, synth, tmp_path)
    assert [p["template"] for p in plan["tracks"]["drums"]["pads"].values()] == [named]
    assert len(plan["tracks"]["drums"]["notes"]) == 8


def _bars(tmp_path, times, info, match):
    """A 16-bar beat (120 BPM, a bar is 2 s) with one template, its hits
    at `times`, and its match."""
    analysis, matches, synth = _fixture(tmp_path)
    analysis["source"]["duration_s"] = 32.0
    analysis["grid"].update(phase=0.0, beats_per_bar=4, beat_unit=4)
    analysis["templates"] = {"sound_1": {"detected_hits": len(times), "clean": 1.0, **info}}
    analysis["events"] = {"sound_1": [{"t": t, "t_attack": t, "gain": 1.0} for t in times]}
    top = [{**matches["sound_1"]["top"][0], "ncc": match.get("ncc", 0.97)}]
    return analysis, {"sound_1": {**matches["sound_1"], "top": top, **match}}, synth


def test_a_sound_heard_once_every_other_bar_gets_a_pad(tmp_path):
    """An open hat on beat 4 of every other bar: 8 hits in 16 bars, under
    the 12 a pad needed (the suite's open hats: plan F1@20 0.043, against
    0.979 in the analysis), but as many as discover() asks of its rarest
    sounds, and every one of them recurs."""
    hits = [4.0 * k + 1.75 for k in range(8)]
    plan = make_plan(*_bars(tmp_path, hits, {"found": "grid"}, {"role": "hat", "verdict": "same sample"}), tmp_path)
    assert [p["template"] for p in plan["tracks"]["drums"]["pads"].values()] == ["sound_1"]
    assert len(plan["tracks"]["drums"]["notes"]) == 8


def test_hits_that_never_recur_are_no_drum_sound(tmp_path):
    """A melodic loop at its own tempo leaves transients in the drum stem
    that discover()'s residual pass groups into a clean template, close to
    a library kick. Its hits sit at a new place of the bar every time; the
    same hits on one place of the bar keep it."""
    loop = [1.37 * k + 0.11 for k in range(16)]
    info, match = {"found": "residual"}, {"role": "kick", "verdict": "same family", "ncc": 0.9}
    assert recurring(loop, 2.0) < 0.2
    assert not make_plan(*_bars(tmp_path, loop, info, match), tmp_path)["tracks"]["drums"]["pads"]
    steady = [2.0 * k + 0.5 for k in range(16)]
    assert recurring(steady, 2.0) == 1.0
    assert make_plan(*_bars(tmp_path, steady, info, match), tmp_path)["tracks"]["drums"]["pads"]


def _ramp(tmp_path, times, info, match, bars: int = 32, rise: float = 0.03):
    """A beat whose tempo rises `rise` over its `bars` bars from 120 BPM (its
    last bar 59 ms shorter than its first) on the constant grid through its
    beats, as fit_grid keeps it, with one template whose hits are the
    quarter-note lines at `times` (indices); and those lines."""
    lines = np.r_[0.0, np.cumsum(0.5 * (1 - rise * np.arange(4 * bars) / (4 * bars)))]
    analysis, matches, synth = _bars(tmp_path, lines[times].tolist(), info, match)
    analysis["source"]["duration_s"] = float(lines[-1])
    analysis["grid"]["bpm"] = 60 / float(np.polyfit(np.arange(len(lines)), lines, 1)[0])
    return (analysis, matches, synth), lines


def test_a_sound_found_on_the_beat_needs_no_recurrence(tmp_path):
    """Recurrence is measured on the grid's constant tempo, so it tests the
    tempo too: on a tempo rising 3 % the kick on every bar line stops
    recurring (lag-1 bars up to 30 ms off), and with every template held
    to it the suite's drums fell 0.866 -> 0.507 (plan F1@20). A sound
    discover() found on the places of the beat needs none."""
    (analysis, matches, synth), lines = _ramp(tmp_path, slice(0, -1, 4), {"found": "grid"},
                                              {"role": "kick", "verdict": "same sample", "ncc": 0.97})
    assert recurring(lines[:-1:4], 240 / analysis["grid"]["bpm"]) < 0.3
    plan = make_plan(analysis, matches, synth, tmp_path)
    assert [p["template"] for p in plan["tracks"]["drums"]["pads"].values()] == ["sound_1"]
    assert len(plan["tracks"]["drums"]["notes"]) == 32


def test_a_residual_sound_recurs_on_the_grids_tempo_map(tmp_path):
    """Where the beats drift from the constant grid, fit_grid keeps them as
    tempo_map; a residual template's hits recur on it (bar_clock): on the
    suite with every time warped by a 3 % rise, plan drums F1@20 0.855
    without the map, 0.867 with it (0.866 unwarped)."""
    (analysis, matches, synth), lines = _ramp(tmp_path, slice(3, -1, 8), {"found": "residual"},  # beat 4, odd bars
                                              {"role": "hat", "verdict": "same sample", "ncc": 0.97})
    assert not make_plan(analysis, matches, synth, tmp_path)["tracks"]["drums"]["pads"]
    analysis["grid"]["tempo_map"] = lines.tolist()
    assert len(make_plan(analysis, matches, synth, tmp_path)["tracks"]["drums"]["notes"]) == 16


def test_a_residual_sound_recurs_at_its_own_tempo_after_a_tempo_change(tmp_path):
    """Two tempi: the grid follows the longer part (grid.tempo_change), and
    a sound of the other part recurs at that part's tempo. Two suite beats
    joined (v1-2 then tempo95-2, B's templates all residual): B's sounds
    plan F1@20 0 -> 0.982 (mean)."""
    hits = [16.0 + 0.4 + 240 / 142 * k for k in range(8)]  # 8 bars at 142 BPM after 8 at 120
    analysis, matches, synth = _bars(tmp_path, hits, {"found": "residual"},
                                     {"role": "snare", "verdict": "same sample", "ncc": 0.97})
    analysis["source"]["duration_s"] = 16.0 + 8 * 240 / 142
    assert recurring(hits, 2.0) == 0.0
    assert not make_plan(analysis, matches, synth, tmp_path)["tracks"]["drums"]["pads"]
    analysis["grid"]["tempo_change"] = {"until": 16.0, "from": 16.0, "ratio": round(142 / 120, 3),
                                        "grid_follows": "first", "span_s": [0.0, 16.0]}
    assert len(make_plan(analysis, matches, synth, tmp_path)["tracks"]["drums"]["notes"]) == 8


def test_a_residual_template_is_no_backbone(tmp_path):
    """A kick or snare far from anything in the library plays as the
    backbone only when discover() found it on the places of the beat; one
    of its residual pass (found where no template of its band plays) needs
    a close match, as any other sound does."""
    hits = [2.0 * k + 0.5 for k in range(16)]
    match = {"role": "kick", "verdict": "no close match", "ncc": 0.7}
    assert make_plan(*_bars(tmp_path, hits, {"found": "grid"}, match), tmp_path)["tracks"]["drums"]["pads"]
    assert not make_plan(*_bars(tmp_path, hits, {"found": "residual"}, match), tmp_path)["tracks"]["drums"]["pads"]
    close = {**match, "ncc": 0.9, "verdict": "same family"}
    assert make_plan(*_bars(tmp_path, hits, {"found": "residual"}, close), tmp_path)["tracks"]["drums"]["pads"]


def test_a_pad_plays_each_moment_once(tmp_path):
    """Two templates of one sound (same_as: the same file, found in two
    bands) share a pad; where both found the same strike, the pad plays it
    once. On 5 of the suite's seed-2 beats a snare-band template of the
    kick-and-hat moments played the hat's file a second time on 6-15 of
    them."""
    analysis, matches, synth = _fixture(tmp_path)
    analysis["events"]["sound_2"] += [{"t": 0.5 * k + 0.002, "gain": 0.5} for k in (0, 2, 4, 6)]
    notes = make_plan(analysis, matches, synth, tmp_path)["tracks"]["drums"]["notes"]
    assert len(notes) == 16 and {n["pitch"] for n in notes} == {38}


def test_the_backbone_plays_its_kicks_once(tmp_path):
    """swing66-2: a snare-band template of 10 of the kick's moments, closer
    to a library kick (NCC 0.895) than the kick itself (0.713, played as
    the backbone), took pad 36 and played those kicks a second time. The
    backbone is planned first and plays each kick once; a template of only
    its kicks gets no pad."""
    analysis, matches, synth = _bars(tmp_path, [1.0 * k + 0.25 for k in range(32)], {"found": "grid"},
                                     {"role": "kick", "verdict": "no close match", "ncc": 0.713})
    analysis["templates"]["sound_2"] = {"detected_hits": 10, "clean": 0.8, "found": "grid"}
    analysis["events"]["sound_2"] = [{"t": t + 0.001, "t_attack": t + 0.001, "gain": 1.0}
                                     for t in [2.0 * k + 0.25 for k in range(10)]]
    matches["sound_2"] = {**matches["sound_1"], "top": [{**matches["sound_1"]["top"][0], "ncc": 0.895}]}
    plan = make_plan(analysis, matches, synth, tmp_path)
    assert [p["template"] for p in plan["tracks"]["drums"]["pads"].values()] == ["sound_1"]
    assert len(plan["tracks"]["drums"]["notes"]) == 32


def test_pitched_copies_get_transposed_pads(tmp_path):
    """Hits of a pitched copy (detect: semitones) play on pads of their own,
    the sound's sample transposed that much further, the plain ones on the
    sound's pad; and the plan scores them as sounds of their own."""
    from refdissect import evaluate
    analysis, matches, synth = _fixture(tmp_path)
    analysis["events"]["sound_1"] += [{"t": 0.2 + 0.5 * k, "gain": 1.0, "semitones": 1 + k % 2} for k in range(6)]
    analysis["events"]["sound_2"].append({"t": 3.9, "gain": 0.5, "semitones": 1})  # its twin's copy: the same pad
    plan = make_plan(analysis, matches, synth, tmp_path)
    pads = plan["tracks"]["drums"]["pads"]
    assert pads["38"]["transpose"] == -1 and (pads["54"]["transpose"], pads["55"]["transpose"]) == (0, 1)
    assert len(pads) == 3
    notes = plan["tracks"]["drums"]["notes"]
    assert sum(n["pitch"] == 54 for n in notes) == 4 and sum(n["pitch"] == 55 for n in notes) == 3
    est = evaluate.estimate_from_plan(plan, root=tmp_path)
    assert {s: v["semitones"] for s, v in est["sounds"].items() if s != "808"} == {"38": -1, "54": 0, "55": 1}


def test_the_same_file_at_another_pitch_plays_on_a_pad_of_its_own(tmp_path):
    """A template found twice (same_as) as the same file 3 semitones lower
    (producer-2's hat at -3, found as a template of its own) plays on a pad
    of that file transposed so far, and its pitched copies from there; at
    the twin's pitch it shared the twin's pad and played 3 semitones high."""
    analysis, matches, synth = _fixture(tmp_path)
    matches["sound_2"]["top"] = [{**matches["sound_1"]["top"][0], "semitones": -4.0}]
    analysis["events"]["sound_2"].append({"t": 3.9, "gain": 0.5, "semitones": 2})
    plan = make_plan(analysis, matches, synth, tmp_path)
    pads = plan["tracks"]["drums"]["pads"]
    assert pads["38"]["transpose"] == -1 and (pads["54"]["transpose"], pads["55"]["transpose"]) == (-4, -2)
    notes = plan["tracks"]["drums"]["notes"]
    assert [sum(n["pitch"] == p for n in notes) for p in (38, 54, 55)] == [8, 8, 1]


def test_locators_on_the_bar_lines(tmp_path):
    """Bar 1 starts at the grid's phase (here 1 s = 2 beats), bars of 3 beats in 3/4."""
    analysis, matches, synth = _fixture(tmp_path)
    analysis["grid"].update(phase=1.0, beats_per_bar=3, beat_unit=4)
    analysis["sections"] = [{"name": "A", "start_bar": 1, "end_bar": 2}, {"name": "B", "start_bar": 3, "end_bar": 4}]
    plan = make_plan(analysis, matches, synth, tmp_path)
    assert plan["locators"] == [["A (1)", 2.0], ["B (3)", 8.0]]


def test_midi_files(tmp_path):
    analysis, matches, synth = _fixture(tmp_path)
    plan = make_plan(analysis, matches, synth, tmp_path)
    written = write_midi(plan, tmp_path / "midi")
    mid = mido.MidiFile(str(written[0]))
    ons = [m for m in mid.tracks[0] if m.type == "note_on"]
    assert len(ons) == 16
    tempo = next(m.tempo for m in mid.tracks[0] if m.type == "set_tempo")
    assert round(mido.tempo2bpm(tempo)) == 120


def test_build_spec_for_build_tracks(tmp_path):
    from refdissect.live import build_spec
    analysis, matches, synth = _fixture(tmp_path)
    plan = make_plan(analysis, matches, synth, tmp_path)
    spec = {t["name"]: t for t in build_spec(plan)}
    t808, drums = spec["808 (synth)"], spec["Drums (library)"]
    # the one-shot is named after the song and its content, so plans for different songs do not collide
    assert t808["instrument"] == {"sample": plan["tracks"]["808"]["sample"], "warping": False}
    assert re.fullmatch(r".+ 808 \(synth\) [0-9a-f]{8}\.wav", t808["instrument"]["sample"])
    assert t808["settings"]["Transpose"] == 36.0 and t808["clips"][0]["notes"]
    pad = drums["instrument"]["drum_rack"]["38"]
    assert re.fullmatch(r"snare [0-9a-f]{8}\.wav", pad["sample"]) and pad["settings"]["Transpose"] == -1.0
    assert len(drums["clips"][0]["notes"]) == 16
    ref = spec["REF drums"]
    assert ref["kind"] == "audio" and ref["mute"] and ref["audio_clips"][0]["warping"] is False
    assert not spec["REF other"]["mute"]


def test_build_spec_sends_the_808s_detune_and_glide(tmp_path):
    """The 808 Simpler gets the plan's Transpose and Detune and, with Glide
    on, Glide Mode and Glide Time: what render_plan and estimate_from_plan
    assume it plays. voices is a device property build_tracks cannot set:
    it rides along for build."""
    from refdissect.live import build_spec
    analysis, matches, synth = _fixture(tmp_path)
    plan = make_plan(analysis, matches, synth, tmp_path)
    plan["tracks"]["808"]["detune_cents"] = -12
    t808 = {t["name"]: t for t in build_spec(plan)}["808 (synth)"]
    assert t808["settings"] == {"Volume": "0.0 dB", "Transpose": 36.0, "Detune": -12.0, "Vol < Vel": 0.35}
    assert t808["voices"] is None
    plan["tracks"]["808"].update(voices=1, glide_mode="Glide", glide_time_ms=60.7)
    t808 = {t["name"]: t for t in build_spec(plan)}["808 (synth)"]
    assert (t808["settings"]["Glide Mode"], t808["settings"]["Glide Time"], t808["voices"]) == ("Glide", "60.7 ms", 1)


def _fake_live(monkeypatch, library=None, held=None, answers=None) -> list:
    """ableton-live-mcp faked for build: an empty set, build_tracks building
    every track it is sent, each Simpler holding the User Library copy its
    spec names (held: another file, by pad note or "808"); answers: a tool's
    own answer, answers[name](args, calls), which may raise as the server
    does. Returns the list of (tool, arguments) calls."""
    import json
    import sys
    import types

    calls = []

    def holds(path: str) -> str | None:
        spec = {t["name"]: t for t in next(a for n, a in calls if n == "build_tracks")["tracks"]}
        pad = re.search(r"drum_pads (\d+)", path)
        key = pad.group(1) if pad else "808"
        sample = (spec["Drums (library)"]["instrument"]["drum_rack"][key] if pad
                  else spec["808 (synth)"]["instrument"])["sample"]
        copy = Path(library) / "Samples" / "AbletonMCP Generated" / sample if library else None
        return (held or {}).get(key) or (str(copy) if copy else None)

    async def call_tool(name, args):
        calls.append((name, args))
        if name in (answers or {}):
            text = answers[name](args, calls)
        elif name == "get_session_overview":
            built = any(c[0] == "build_tracks" for c in calls)
            tracks = [{"name": "Drums (library)", "index": 0}, {"name": "808 (synth)", "index": 1}] if built else []
            text = json.dumps({"tracks": tracks})
        elif name == "build_tracks":
            text = json.dumps({"tracks": [{"track": t["name"], "problems": []} for t in args["tracks"]]})
        elif name == "lom_get_object":
            text = json.dumps({"file_path": holds(args["path"])})
        else:
            text = "{}"
        return types.SimpleNamespace(content=[types.SimpleNamespace(text=text)], is_error=False)

    server = types.ModuleType("MCP_Server.server")
    server.mcp = types.SimpleNamespace(call_tool=call_tool)
    tools = types.ModuleType("MCP_Server.extended_tools")
    tools.place_locators = lambda cues: [{"name": n} for n, _ in cues]
    package = types.ModuleType("MCP_Server")
    package.extended_tools, package.server = tools, server
    for name, module in (("MCP_Server", package), ("MCP_Server.server", server),
                         ("MCP_Server.extended_tools", tools)):
        monkeypatch.setitem(sys.modules, name, module)
    return calls


def test_build_gives_the_808_simpler_one_voice(tmp_path, monkeypatch):
    """build (ableton-live-mcp faked) sends the spec to build_tracks, then
    sets the 808 Simpler's voices to 1 by its LOM property."""
    import asyncio

    from refdissect.live import build
    analysis, matches, synth = _fixture(tmp_path)
    plan = make_plan(analysis, matches, synth, tmp_path)
    plan["tracks"]["808"].update(detune_cents=7, voices=1, glide_mode="Glide", glide_time_ms=80.0)
    calls = _fake_live(monkeypatch)
    report = asyncio.run(build(plan, user_library=tmp_path / "lib", log=lambda s: None))
    spec = next(a for n, a in calls if n == "build_tracks")["tracks"]
    assert next(t for t in spec if t["name"] == "808 (synth)")["settings"]["Detune"] == 7.0
    assert ("lom_set_property", {"path": "live_set tracks 1 devices 0", "property": "voices", "value": 1}) in calls
    assert "808_voices" in report and (tmp_path / "lib" / "Samples" / "AbletonMCP Generated").is_dir()


def test_the_808_one_shot_is_named_by_its_content(tmp_path, monkeypatch):
    """Live keeps playing a sample loaded through the browser under its name
    after the file changes on disk, so the one-shot's name carries its
    content's digest: fitted again, it gets another name. build copies it
    into the User Library under that name, and refuses a one-shot that
    changed after the plan named it."""
    import asyncio

    from refdissect.live import build, one_shot_digest
    analysis, matches, synth = _fixture(tmp_path)
    plan = make_plan(analysis, matches, synth, tmp_path)
    first = plan["tracks"]["808"]["sample"]
    assert first.endswith(f" 808 (synth) {one_shot_digest(synth['file'])}.wav")
    _fake_live(monkeypatch)
    asyncio.run(build(plan, user_library=tmp_path / "lib", log=lambda s: None))
    copied = tmp_path / "lib" / "Samples" / "AbletonMCP Generated" / first
    assert copied.read_bytes() == Path(synth["file"]).read_bytes()
    sf.write(synth["file"], np.sin(2 * np.pi * 36.7 * np.arange(SR) / SR) * 0.9, SR)  # fit-808 ran again
    assert make_plan(analysis, matches, synth, tmp_path)["tracks"]["808"]["sample"] != first
    with pytest.raises(RuntimeError, match="plan again"):
        asyncio.run(build(plan, user_library=tmp_path / "lib", log=lambda s: None))


def test_each_pad_loads_its_own_file_by_its_content(tmp_path, monkeypatch):
    """build_tracks loads a pad through Live's browser by file name, and the
    browser holds other files of a pad's name (the live check: an 808 on
    v1orig-3's open-hat pad): each pad's file goes into the User Library
    under its name and its content's digest, the spec loads that copy, and
    the build reads back the file every Simpler holds and names a pad that
    holds another."""
    import asyncio

    from refdissect.live import build, build_spec, one_shot_digest
    analysis, matches, synth = _fixture(tmp_path)
    plan = make_plan(analysis, matches, synth, tmp_path)
    snare = plan["tracks"]["drums"]["pads"]["38"]["path"]
    pad = {t["name"]: t for t in build_spec(plan)}["Drums (library)"]["instrument"]["drum_rack"]["38"]
    assert pad["sample"] == f"snare {one_shot_digest(snare)}.wav"
    lib = tmp_path / "lib"
    _fake_live(monkeypatch, lib)
    report = asyncio.run(build(plan, user_library=lib, log=lambda s: None))
    assert (lib / "Samples" / "AbletonMCP Generated" / pad["sample"]).read_bytes() == Path(snare).read_bytes()
    assert report["samples_checked"] == 2 and "wrong_samples" not in report  # the pad and the 808
    other = tmp_path / "kit" / "snare.wav"  # another file of the pad's name, as the browser found one
    other.parent.mkdir()
    sf.write(other, np.random.default_rng(5).standard_normal(SR // 8) * 0.5, SR)
    _fake_live(monkeypatch, lib, held={"38": str(other)})
    logged = []
    report = asyncio.run(build(plan, user_library=lib, log=logged.append))
    assert list(report["wrong_samples"]) == ["38"] and report["wrong_samples"]["38"]["live"] == str(other)
    assert any(line.startswith(f"pad 38 holds {other}") for line in logged)


def test_a_build_that_stops_says_where_and_what_it_left(tmp_path, monkeypatch):
    """In the live check build_tracks raised at a pad whose file Live's
    browser did not list, after the 808 track and an empty Drum Rack were
    made, and only the error was said: the build names the pad, its file
    and the tracks it left. A connection Live refuses says that its Remote
    Script serves at most 8 clients."""
    import asyncio
    import json

    from refdissect.live import build
    analysis, matches, synth = _fixture(tmp_path)
    plan = make_plan(analysis, matches, synth, tmp_path)

    def stop(args, calls):
        name = next(t for t in args["tracks"] if t["name"] == "Drums (library)")["instrument"]["drum_rack"]["38"]
        raise RuntimeError(f"Live's browser does not list {name['sample']!r}")

    def overview(args, calls):
        made = [{"name": "808 (synth)", "index": 4, "devices": ["Simpler"]},
                {"name": "5-MIDI", "index": 5, "devices": ["Drum Rack"]}]
        before = [{"name": f"{k}-MIDI", "index": k - 1, "devices": [], "session_clips": [], "arrangement_clip_count": 0}
                  for k in range(1, 5)]  # the set's own empty tracks stay
        return json.dumps({"tracks": before + (made if any(c[0] == "build_tracks" for c in calls) else [])})

    logged = []
    _fake_live(monkeypatch, tmp_path / "lib", answers={"build_tracks": stop, "get_session_overview": overview})
    with pytest.raises(RuntimeError) as e:
        asyncio.run(build(plan, user_library=tmp_path / "lib", log=logged.append))
    said = str(e.value)
    assert "pad 38" in said and plan["tracks"]["drums"]["pads"]["38"]["path"] in said
    assert "'808 (synth)' (Simpler), '5-MIDI' (Drum Rack)" in said and "1-MIDI" not in said
    assert json.loads(logged[0])["tracks_made"][1]["devices"] == ["Drum Rack"]

    def reset(args, calls):  # the 2.2.0 server's words, with 8 other clients connected
        raise RuntimeError("Could not connect to Ableton Live. Is Live open, with AbletonMCP chosen as a Control "
                           "Surface? ([Errno 54] Connection reset by peer)")

    _fake_live(monkeypatch, answers={"get_session_overview": reset})
    with pytest.raises(RuntimeError, match="serves at most 8 clients"):
        asyncio.run(build(plan, user_library=tmp_path / "lib", log=lambda s: None))


def test_the_plan_plays_its_hits_gains_under_lives_velocity_law(tmp_path):
    """Every Simpler is sent "Vol < Vel" 0.35, and the velocities follow
    Live 12.4.6's law there: hits at 1, 0.5 and 0.3 of the 90th
    percentile's gain play 0, -6.02 and -10.46 dB from it (within half a
    velocity step, 0.07 dB). The faders are worked out under the same law,
    so the render puts each track on its stem's RMS less the headroom (a
    drum stem of the pad's own hits at those gains)."""
    from refdissect.audio import rms_db, transpose
    from refdissect.live import build_spec
    from refdissect.render import render_plan
    analysis, matches, synth = _fixture(tmp_path)
    gains = [1.0, 0.5, 0.3, 1.0, 1.0, 0.5, 0.3, 1.0]
    analysis["events"] = {"sound_1": [{"t": 0.5 * k, "gain": g} for k, g in enumerate(gains)], "sound_2": []}
    played = transpose(sf.read(tmp_path / "snare.wav", dtype="float32")[0], -1)
    stem = np.zeros(4 * SR, np.float32)
    for k, g in enumerate(gains):
        stem[int(0.5 * k * SR):int(0.5 * k * SR) + len(played)] += 0.2 * g * played
    sf.write(tmp_path / "drums.wav", stem, SR, subtype="FLOAT")
    plan = make_plan(analysis, matches, synth, tmp_path)
    spec = {t["name"]: t for t in build_spec(plan)}
    assert spec["808 (synth)"]["settings"]["Vol < Vel"] == 0.35
    assert {p["settings"]["Vol < Vel"] for p in spec["Drums (library)"]["instrument"]["drum_rack"].values()} == {0.35}
    m = render_plan(plan, tmp_path / "r", n_samples=4 * SR)
    drums = sf.read(tmp_path / "r" / m["drums"], dtype="float32")[0]
    peaks = [np.abs(drums[int(0.5 * k * SR):int((0.5 * k + 0.25) * SR)]).max() for k in range(len(gains))]
    assert 20 * np.log10(np.array(peaks) / max(peaks)) == pytest.approx(20 * np.log10(gains), abs=0.07)
    stem = sf.read(tmp_path / "drums.wav", dtype="float32")[0]
    assert rms_db(drums) == pytest.approx(rms_db(stem) - plan["levels"]["headroom_db"], abs=0.06)
    bass = sf.read(tmp_path / "bass.wav", dtype="float32")[0]
    t808 = sf.read(tmp_path / "r" / m["tracks"]["808"], dtype="float32")[0]
    assert rms_db(t808) == pytest.approx(rms_db(bass) - plan["levels"]["headroom_db"], abs=0.1)


def _kit(tmp_path, kick_in_bass: float = 0.6, under: float = 0.0, copy_gain: float | None = None,
         extra: dict | None = None, hat_only: bool = False, duck_db: float = 0.0, click: float = 0.0,
         stem_gain: float = 1.0, sounding: list | None = None):
    """A kick (120 Hz) and a hat file at one peak level, and stems that hold
    the hat 20 dB under the kick: of the kick's amplitude (0.5) the drum
    stem holds 1 - kick_in_bass and the bass stem the rest, as htdemucs
    hands part of a kick to the bass stem. With `under`, the bass stem also
    holds the plan's own 808, at that gain, under every kick, duck_db
    quieter over the kick's first 100 ms (a sidechain); with copy_gain, the
    last hat is 3 semitones up (a pitched copy), held at that gain. extra:
    more sounds, {name: (role, file, events)}; hat_only: the analysis found
    the hat alone (the stems still hold the kick); click: a click of noise
    over the kick's body, at that gain (a bright kick); stem_gain: the
    stems that much louder; sounding: more audio in the drum stem,
    [(samples, time (s), gain)]."""
    from refdissect.audio import attack_time, transpose
    from refdissect.render import render_808
    t = np.arange(int(0.3 * SR)) / SR
    kick = 0.9 * np.sin(2 * np.pi * 120 * t) * np.exp(-t / 0.08)
    if click:
        kick = kick + click * np.random.default_rng(5).standard_normal(len(t)) * np.exp(-t / 0.004)
        kick *= 0.9 / np.max(np.abs(kick))
    hat = 0.9 * np.random.default_rng(2).standard_normal(len(t)) * np.exp(-t / 0.03)
    sf.write(tmp_path / "kick.wav", kick, SR, subtype="FLOAT")
    sf.write(tmp_path / "hat.wav", hat, SR, subtype="FLOAT")
    drums, bass = np.zeros(4 * SR), np.zeros(4 * SR)
    for k in range(4):
        drums[k * SR:k * SR + len(t)] += 0.5 * (1 - kick_in_bass) * kick
        bass[k * SR:k * SR + len(t)] += 0.5 * kick_in_bass * kick
        x = 0.05 * hat if copy_gain is None or k < 3 else copy_gain * transpose(hat.astype(np.float32), 3)
        drums[k * SR + SR // 2:k * SR + SR // 2 + len(x)] += x
    for x, at, gain in sounding or []:
        a = int(at * SR)
        drums[a:a + len(x)] += gain * x[:len(drums) - a]
    a_kick, a_hat = attack_time(kick, "kick"), attack_time(hat, "hat")
    events = {"sound_1": [{"t": float(k), "t_attack": k + a_kick, "gain": 1.0} for k in range(4)],
              "sound_2": [{"t": k + 0.5, "t_attack": k + 0.5 + a_hat, "gain": 1.0} for k in range(4)]}
    if copy_gain is not None:
        events["sound_2"][3] |= {"semitones": 3, "t_attack": 3.5 + attack_time(transpose(hat, 3), "hat")}
    file = {"sound_1": ("kick", "kick.wav"), "sound_2": ("hat", "hat.wav")}
    for name, (role, path, hits) in (extra or {}).items():
        events[name], file[name] = hits, (role, path)
    if hat_only:
        del events["sound_1"]
    analysis = {"grid": {"bpm": 120.0}, "source": {"duration_s": 4.0}, "events": events,
                "notes_808": [{"t": float(k), "end": k + 0.45, "pitch": 24, "peak": 1.0} for k in range(4)] if under
                else [], "templates": {k: {"detected_hits": 20, "clean": 1.0} for k in events}}
    matches = {k: {"role": file[k][0], "verdict": "same sample", "same_as": None,
                   "top": [{"path": str(tmp_path / file[k][1]), "ncc": 0.97, "semitones": 0.0}]} for k in events}
    synth = None
    if under:
        shot = tmp_path / "808 (synth).wav"
        sf.write(shot, np.sin(2 * np.pi * 33.2 * np.arange(SR) / SR) * 0.9, SR)
        synth = {"root_midi": 24, "file_name": shot.name, "file": str(shot)}
        sf.write(tmp_path / "bass.wav", bass, SR, subtype="FLOAT")
        t808 = make_plan(analysis, matches, synth, tmp_path, min_hits=1)["tracks"]["808"]
        duck = np.ones(4 * SR)
        for k in range(4):
            duck[k * SR:k * SR + SR // 10] = 10 ** (-duck_db / 20)
        bass = bass + under * duck * render_808(t808, sf.read(shot, dtype="float32")[0], 4 * SR, 2.0, 0, True, 1.0,
                                                "live")
    sf.write(tmp_path / "drums.wav", stem_gain * drums, SR, subtype="FLOAT")
    sf.write(tmp_path / "bass.wav", stem_gain * bass, SR, subtype="FLOAT")
    return make_plan(analysis, matches, synth, tmp_path, min_hits=1)


def _volume(spec_pad: dict) -> float:
    return float(spec_pad["settings"]["Volume"].removesuffix(" dB"))


def test_each_pad_plays_where_the_stems_hold_its_sound(tmp_path):
    """Two files at one peak level, the stems holding the hat 20 dB under
    the kick and 60 % of the kick in the bass stem: the hat's pad plays 20
    dB under the kick's, and the build sends each pad's Volume. Every pad
    at its file's level, the suite's open hats sat 11.6 dB over a perfect
    kit (htdemucs); read on the drum stem alone, this kick would have been
    8 dB short (the hat 12 dB under it)."""
    from refdissect.live import build_spec
    plan = _kit(tmp_path)
    pads = plan["tracks"]["drums"]["pads"]
    assert pads["36"]["volume_db"] == 0.0 and pads["42"]["volume_db"] == pytest.approx(-20.0, abs=0.2)
    rack = {t["name"]: t for t in build_spec(plan)}["Drums (library)"]["instrument"]["drum_rack"]
    assert rack["36"]["settings"]["Volume"] == "0.0 dB"
    assert _volume(rack["42"]) == pads["42"]["volume_db"] and max(_volume(p) for p in rack.values()) <= 0.0


def test_no_pad_plays_over_0_db(tmp_path):
    """A pad's volume_db over 0 dB (a plan edited by hand) is built and
    played at 0 dB: the build keeps every Simpler at or under 0 dB, and the
    renders and estimates of the plan play what it builds."""
    from refdissect.live import build_spec
    from refdissect.render import render_plan
    plan = _kit(tmp_path, kick_in_bass=0.0)
    m = render_plan(plan, tmp_path / "a", n_samples=4 * SR)
    plan["tracks"]["drums"]["pads"]["36"]["volume_db"] = 3.0
    rack = {t["name"]: t for t in build_spec(plan)}["Drums (library)"]["instrument"]["drum_rack"]
    assert rack["36"]["settings"]["Volume"] == "0.0 dB"
    m3 = render_plan(plan, tmp_path / "b", n_samples=4 * SR)
    kick, kick3 = (sf.read(tmp_path / d / x["tracks"]["36"], dtype="float32")[0] for d, x in (("a", m), ("b", m3)))
    assert np.array_equal(kick, kick3)


def test_the_pads_make_up_what_the_drum_fader_cannot(tmp_path):
    """Stems 28 dB louder than the other kits': the drum fader would need
    +10.1 dB (-17.9 dB in those), and Live's stops at +6 dB, where every
    pad played 4.1 dB under its level. The pads under 0 dB make it up: the
    hat plays where it should (+6 - 15.9 = +10.1 - 20 dB), the kick, at
    0 dB already, 4.1 dB under."""
    plan = _kit(tmp_path, kick_in_bass=0.0, stem_gain=25.0)
    pads = plan["tracks"]["drums"]["pads"]
    assert plan["levels"]["drums_track_db"] == 6.0
    assert pads["36"]["volume_db"] == 0.0 and pads["42"]["volume_db"] == pytest.approx(-15.9, abs=0.2)


def test_a_perfect_kit_plays_at_the_stems_level(tmp_path):
    """Stems made of the pads' own files at their own balance: the render
    holds each pad's hits and the whole drum track the headroom under the
    drum stem, the perfect kit evaluate.level_scores scores against."""
    from refdissect.audio import band, rms_db
    from refdissect.render import render_plan
    plan = _kit(tmp_path, kick_in_bass=0.0)
    m = render_plan(plan, tmp_path / "r", n_samples=4 * SR)
    drums = sf.read(tmp_path / "r" / m["drums"], dtype="float32")[0]
    stem = sf.read(tmp_path / "drums.wav", dtype="float32")[0]
    head = plan["levels"]["headroom_db"]
    assert rms_db(drums) == pytest.approx(rms_db(stem) - head, abs=0.1)
    hats = [slice(k * SR + SR // 2, k * SR + SR // 2 + SR // 20) for k in range(4)]
    ours, theirs = (sum(float(np.sum(np.square(band(x, "hat")[h], dtype=np.float64))) for h in hats)
                    for x in (drums, stem))
    assert 10 * np.log10(ours / theirs) == pytest.approx(-head, abs=0.2)


def test_the_drum_fader_stops_where_the_loudest_pad_meets_its_stem(tmp_path):
    """A stem that holds a sound no pad plays (here the kick: the analysis
    found the hat alone) has more RMS than the plan can match: matched by
    RMS, the hat played 17 dB over the stem, 21 dB over the headroom under
    it. The fader stops where the loudest pad plays as loud as the stem
    holds its sound, the headroom under it (tempo160-1, with no kick pad:
    13.4 -> 5.1 dB from a perfect kit)."""
    from refdissect.audio import band
    from refdissect.render import render_plan
    plan = _kit(tmp_path, kick_in_bass=0.0, hat_only=True)
    assert list(plan["tracks"]["drums"]["pads"]) == ["42"]
    m = render_plan(plan, tmp_path / "r", n_samples=4 * SR)
    drums = sf.read(tmp_path / "r" / m["drums"], dtype="float32")[0]
    stem = sf.read(tmp_path / "drums.wav", dtype="float32")[0]
    hats = [slice(k * SR + SR // 2, k * SR + SR // 2 + SR // 20) for k in range(4)]
    ours, theirs = (sum(float(np.sum(np.square(band(x, "hat")[h], dtype=np.float64))) for h in hats)
                    for x in (drums, stem))
    assert 10 * np.log10(ours / theirs) == pytest.approx(-plan["levels"]["headroom_db"], abs=0.2)


def test_the_808_under_a_kick_is_not_the_kicks(tmp_path):
    """The bass stem holds the plan's own 808 under every kick (here 1.9 dB
    under the kick over the kick's first 50 ms, in the kick band; the
    suite's 808s sit 1.3 dB over its kicks there, median): read on drums +
    bass, the kick takes its own energy, not the 808's too, which put it
    2.0 dB high here (and the hat that much lower)."""
    from refdissect.live import pad_levels
    from refdissect.render import render_808
    plan = _kit(tmp_path, kick_in_bass=0.0, under=0.15)
    assert plan["tracks"]["drums"]["pads"]["42"]["volume_db"] == pytest.approx(-20.0, abs=0.3)
    drums, bass = (sf.read(tmp_path / f"{s}.wav", dtype="float32")[0] for s in ("drums", "bass"))
    held = 0.15 * render_808(plan["tracks"]["808"], sf.read(tmp_path / "808 (synth).wav", dtype="float32")[0],
                             4 * SR, 2.0, 0, True, 1.0, "live")
    alone, under = pad_levels(plan, drums, drums + bass), pad_levels(plan, drums, drums + bass, held)
    assert alone["36"] - under["36"] > 1.5 and under["36"] - under["42"] == pytest.approx(20.0, abs=0.3)


def test_a_kick_under_a_ducked_808_is_read_on_the_drum_stem(tmp_path):
    """The bass stem holds the 808 louder than the kick but 6 dB quieter
    over each kick's first 100 ms (a sidechain), the plan's 808 track does
    not: taken off drums + bass, it leaves nothing of the kick, which went
    without a level, both pads at 0 dB under a fader set by the hat. The
    drum stem alone holds the least of a kick there is: the kick is read
    there, and the hat 20 dB under it."""
    plan = _kit(tmp_path, kick_in_bass=0.0, under=0.6, duck_db=6.0)
    pads = plan["tracks"]["drums"]["pads"]
    assert pads["36"]["volume_db"] == 0.0 and pads["42"]["volume_db"] == pytest.approx(-20.0, abs=0.3)


def test_a_bright_kick_on_a_kick_pad_is_read_as_a_kick(tmp_path):
    """A kick file with a click over its body holds 77 % of its energy
    under 200 Hz, under LOW_SHARE (as 69 of the 683 kick files of the
    library measured do): read as any other sound, on the drum stem alone,
    which holds 40 % of it here, it played 8 dB short, the hat 12 dB under
    it. On a kick's pad it is a kick from KICK_SHARE on."""
    from refdissect.live import KICK_SHARE, LOW_SHARE, low_share
    plan = _kit(tmp_path, click=1.5)
    assert KICK_SHARE < low_share(sf.read(tmp_path / "kick.wav", dtype="float32")[0]) < LOW_SHARE
    pads = plan["tracks"]["drums"]["pads"]
    assert pads["36"]["volume_db"] == 0.0 and pads["42"]["volume_db"] == pytest.approx(-20.0, abs=0.3)


def test_a_sound_heard_mostly_with_a_louder_one_is_read_where_it_is_alone(tmp_path):
    """An open hat at the hat's level, alone on 3 of its hits and found on
    4 more where a snare plays 20 dB louder: over all its hits the median
    is one of the snare's, and the open hat read 18 dB over where the
    stems hold it, 1.6 dB under the kick (v1orig-2, htdemucs: an extra open
    hat, 9 of whose 17 hits sat on the snare and the hat, read 22.8 dB over
    its other 8; it played 23.3 dB over a perfect kit, and as the loudest
    pad it put the hat's Volume at -36.3 dB and the drum fader at +5.6
    dB). Its hits clear of every other sound agree and read more than
    CLEAR_GAP_DB under: it is read on those, with the hat."""
    from refdissect.audio import attack_time
    t = np.arange(int(0.3 * SR)) / SR
    noise = np.random.default_rng(7).standard_normal((2, len(t)))
    ohat, snare = 0.9 * noise[0] * np.exp(-t / 0.1), 0.9 * noise[1] * np.exp(-t / 0.05)
    sf.write(tmp_path / "ohat.wav", ohat, SR, subtype="FLOAT")
    sf.write(tmp_path / "snare.wav", snare, SR, subtype="FLOAT")
    a_ohat, a_snare = attack_time(ohat, "hat"), attack_time(snare, "snare")
    alone, shared = [k + 0.25 for k in range(3)], [k + 0.75 for k in range(4)]
    extra = {"sound_3": ("hat", "ohat.wav", [{"t": t0, "t_attack": t0 + a_ohat, "gain": 1.0} for t0 in alone + shared]),
             "sound_4": ("snare", "snare.wav", [{"t": t0, "t_attack": t0 + a_snare, "gain": 1.0} for t0 in shared])}
    held = [(ohat, t0, 0.05) for t0 in alone] + [(snare, t0, 0.5) for t0 in shared]
    plan = _kit(tmp_path, extra=extra, sounding=held)
    pads = plan["tracks"]["drums"]["pads"]
    ohat_pad = next(k for k, p in pads.items() if p["path"].endswith("ohat.wav"))
    assert pads[ohat_pad]["volume_db"] == pytest.approx(pads["42"]["volume_db"], abs=0.3)
    assert pads["42"]["volume_db"] == pytest.approx(-20.0, abs=0.3)


def test_a_sounds_pitched_copy_shares_its_level_and_a_silent_file_has_none(tmp_path):
    """A copy's few hits sit where the stem may hold anything (here its one
    hit, 3 semitones up, 12 dB over the hat's others); it takes its sound's
    level, read over all of the sound's hits (rolls-1, htdemucs_ft: a copy
    read alone sat 22.9 dB over a perfect kit). A file that peaks under
    -60 dBFS gets no level and keeps 0 dB."""
    sf.write(tmp_path / "fx.wav", np.full(SR // 4, 1e-22), SR, subtype="FLOAT")
    extra = {"sound_3": ("perc", "fx.wav", [{"t": k + 0.75, "t_attack": k + 0.75, "gain": 1.0} for k in range(4)])}
    plan = _kit(tmp_path, copy_gain=0.2, extra=extra)
    pads = plan["tracks"]["drums"]["pads"]
    copy = next(k for k, p in pads.items() if p.get("semitones") == 3)
    assert pads[copy]["volume_db"] == pads["42"]["volume_db"] == pytest.approx(-20.0, abs=0.2)
    silent = next(k for k, p in pads.items() if p["path"].endswith("fx.wav"))
    assert pads[silent]["volume_db"] == 0.0 and pads["36"]["volume_db"] == 0.0


def test_a_match_between_semitones_is_played_with_detune(tmp_path):
    """A snare found 0.70 semitones down (match v2: transpose -1, +30 cents)
    gets Transpose -1 and Detune +30 on its pad and in the build, and the
    plan plays it at -0.70; a match from before match v2 (whole semitones
    only) plays with Detune 0."""
    from refdissect import evaluate
    from refdissect.live import build_spec
    analysis, matches, synth = _fixture(tmp_path)
    for m in matches.values():
        m["top"][0] = {**m["top"][0], "semitones": -0.7, "transpose": -1, "cents": 30.0, "detune_cents": 30.0}
    plan = make_plan(analysis, matches, synth, tmp_path)
    pad = plan["tracks"]["drums"]["pads"]["38"]
    assert (pad["transpose"], pad["detune_cents"]) == (-1, 30)
    rack = {t["name"]: t for t in build_spec(plan)}["Drums (library)"]["instrument"]["drum_rack"]
    assert (rack["38"]["settings"]["Transpose"], rack["38"]["settings"]["Detune"]) == (-1.0, 30.0)
    est = evaluate.estimate_from_plan(plan, root=tmp_path)
    assert est["sounds"]["38"]["semitones"] + est["sounds"]["38"]["cents"] / 100 == pytest.approx(-0.7)
    old = make_plan(*_fixture(tmp_path), tmp_path)["tracks"]["drums"]["pads"]["38"]
    assert (old["transpose"], old["detune_cents"]) == (-1, 0)


def test_melodic_tracks_in_the_plan(tmp_path):
    """melodic.json adds one muted MIDI track per stem; the drums and the 808
    are planned exactly as without it."""
    from refdissect.live import MELODIC_INSTRUMENT, build_spec
    analysis, matches, synth = _fixture(tmp_path)
    melodic = {"stems": {"other": {"tuning_cents": 12, "notes": [
        {"pitch": 64, "start_time": 1.0, "duration": 2.0, "velocity": 90, "t": 0.5, "level_db": -20.0},
        {"pitch": 64, "start_time": 2.0, "duration": 1.0, "velocity": 80, "t": 1.0, "level_db": -21.0}]}}}
    plain = make_plan(analysis, matches, synth, tmp_path)
    plan = make_plan(analysis, matches, synth, tmp_path, melodic=melodic)
    assert "melodic" not in plain["tracks"]
    assert {k: plan["tracks"][k] for k in ("drums", "808")} == {k: plain["tracks"][k] for k in ("drums", "808")}
    track = plan["tracks"]["melodic"]["other"]
    assert track["instrument"] == MELODIC_INSTRUMENT and track["cents"] == 12
    assert [n["duration"] for n in track["notes"]] == [0.995, 1.0]  # ends before the next note of its pitch
    spec = {t["name"]: t for t in build_spec(plan)}
    assert spec["other (notes)"]["mute"] is True and spec["other (notes)"]["instrument"] == MELODIC_INSTRUMENT
    assert spec["other (notes)"]["clips"][0]["notes"] == track["notes"]
    written = write_midi(plan, tmp_path / "midi")
    assert [w.name for w in written] == ["drums.mid", "808.mid", "other.mid"]
    ons = [m for m in mido.MidiFile(str(written[2])).tracks[0] if m.type == "note_on"]
    assert [m.note for m in ons] == [64, 64] and ons[0].channel == 0
