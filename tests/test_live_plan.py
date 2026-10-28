import re
from pathlib import Path

import mido
import numpy as np
import pytest
import soundfile as sf

from refdissect.audio import PRE, SR
from refdissect.live import make_plan, no_overlaps, velocities, write_midi


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
    assert pad["sample"] == "snare.wav" and pad["settings"]["Transpose"] == -1.0
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


def _fake_live(monkeypatch) -> list:
    """ableton-live-mcp faked for build: an empty set, build_tracks building
    every track it is sent; returns the list of (tool, arguments) calls."""
    import json
    import sys
    import types

    calls = []

    async def call_tool(name, args):
        calls.append((name, args))
        if name == "get_session_overview":
            built = any(c[0] == "build_tracks" for c in calls)
            tracks = [{"name": "Drums (library)", "index": 0}, {"name": "808 (synth)", "index": 1}] if built else []
            text = json.dumps({"tracks": tracks})
        elif name == "build_tracks":
            text = json.dumps({"tracks": [{"track": t["name"], "problems": []} for t in args["tracks"]]})
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


def test_the_plan_plays_its_hits_gains_under_lives_velocity_law(tmp_path):
    """Every Simpler is sent "Vol < Vel" 0.35, and the velocities follow
    Live 12.4.6's law there: hits at 1, 0.5 and 0.3 of the 90th
    percentile's gain play 0, -6.02 and -10.46 dB from it (within half a
    velocity step, 0.07 dB). The faders are worked out under the same law,
    so the render puts each track on its stem's RMS less the headroom."""
    from refdissect.audio import rms_db
    from refdissect.live import build_spec
    from refdissect.render import render_plan
    analysis, matches, synth = _fixture(tmp_path)
    gains = [1.0, 0.5, 0.3, 1.0, 1.0, 0.5, 0.3, 1.0]
    analysis["events"] = {"sound_1": [{"t": 0.5 * k, "gain": g} for k, g in enumerate(gains)], "sound_2": []}
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
