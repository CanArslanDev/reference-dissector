"""live-build through ableton-live-mcp's build_from_blueprint, against its
simulator (the real Remote Script over fake Live, MCP_Server/simulator.py and
fake_live.py): a synthetic plan built into an empty set, and the set read
back. The set expected here is the one the build before build_from_blueprint
(live-build --direct, gone on 2026-11-16) made of the same plan in the same
simulator, which the two paths built alike until then: but for what loading
a file by path leaves to Live (names, a pad's voices, and its F On and Fade
In, which the blueprint sets), and for the excerpt's empty clip and stem
names. Needs the [live] extra (ableton-live-mcp with blueprints); skipped
without it, and failed instead where REFDISSECT_REQUIRE_LIVE is set (CI). No
Live, no Demucs, no sample library: the plan of test_blueprint.plan_for."""

import asyncio
import json
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

if os.environ.get("REFDISSECT_REQUIRE_LIVE"):  # CI installs the [live] extra: a missing one fails, not skips
    import MCP_Server.blueprint  # noqa: F401
    import MCP_Server.simulator  # noqa: F401
pytest.importorskip("MCP_Server.blueprint", reason="ableton-live-mcp 2.4.0 or later (the [live] extra) not installed")
simulator_module = pytest.importorskip("MCP_Server.simulator")

from MCP_Server.blueprint import GENERATED  # noqa: E402
from test_blueprint import SR, _wav, plan_for  # noqa: E402

from refdissect import blueprint as B  # noqa: E402
from refdissect.cli import main  # noqa: E402
from refdissect.live import onto_live_bars, pad_sample, scale_files_to_full  # noqa: E402

# A new Live set: two empty MIDI tracks, no clips, no locators (as ableton-live-mcp's tests make one)
EMPTY_SET = ([{"type": "lom_call", "params": {"path": "live_set", "method": "delete_track", "args": [i]}}
              for i in (2, 1, 0)]
             + [{"type": "lom_call", "params": {"path": "live_set", "method": "create_midi_track",
                                                "args": [-1]}}] * 2)


@pytest.fixture
def live(tmp_path, monkeypatch):
    """A factory of simulated Lives, each an empty set served on a port of its
    own, reached by ableton-live-mcp's server in-process (MCP_Server.server's
    mcp, every tool of the full profile registered), its User Library in
    tmp_path."""
    from mcp.server.mcpserver import MCPServer
    from MCP_Server import als, extended_tools
    from MCP_Server import server as live_server

    library = tmp_path / "User Library"
    library.mkdir()
    monkeypatch.setenv("ABLETON_MCP_PROFILE", "full")
    monkeypatch.delenv("ABLETON_PORT", raising=False)
    monkeypatch.setattr(als, "_user_libraries", lambda: [library])
    mcp = MCPServer("refdissect-test")
    extended_tools.register(mcp, live_server.get_ableton_connection)
    monkeypatch.setattr(extended_tools, "_get_connection", live_server.get_ableton_connection)
    monkeypatch.setattr(live_server, "mcp", mcp)
    running = []

    def disconnect():
        if live_server._ableton_connection is not None:
            live_server._ableton_connection.disconnect()
            live_server._ableton_connection = None

    def start():
        disconnect()
        for sim in running:
            sim.stop()
        session = tmp_path / f"session-{len(running)}.json"
        monkeypatch.setenv("ABLETON_MCP_SESSION_FILE", str(session))
        sim = simulator_module.Simulator(simulator_module.free_port(), session)
        sim.run_setup(EMPTY_SET)
        sim.start()
        running.append(sim)
        return sim

    start.library = library
    yield start
    disconnect()
    for sim in running:
        sim.stop()


def a_plan(tmp_path: Path, meter: tuple[int, int] = (4, 4)) -> tuple[dict, Path]:
    """plan_for's plan with what the build sets beyond it: a pad in a choke
    group, a file over full scale (copied scaled), an 808 on one voice with
    Glide, stems of 8 s (the plan's 16 beats at 140 BPM: 6.9 s); written to
    <tmp>/Song/work/plan.json."""
    plan = plan_for(tmp_path, meter=meter)
    for k, stem in enumerate(plan["stems"].values()):
        _wav(Path(stem["path"]), np.random.default_rng(10 + k).standard_normal(8 * SR) * 0.1)
    pads = plan["tracks"]["drums"]["pads"]
    pads["42"]["choke_group"] = 1
    x = np.random.default_rng(3).standard_normal(SR // 8)
    _wav(Path(pads["38"]["path"]), x / np.max(np.abs(x)) * 10 ** (4.23 / 20), subtype="FLOAT")  # +4.23 dBFS
    scale_files_to_full(plan)
    plan["tracks"]["808"].update(voices=1, glide_mode="Glide", glide_time_ms=45.0)
    work = tmp_path / "Song" / "work"
    work.mkdir()
    (work / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    return plan, work


def simplers(sim) -> dict:
    """Each Simpler of the set, by "808 (synth)" or pad note: its playback
    mode, the file it holds, its voices (the 808's), and its chain's choke
    group (a pad's). A pad's voices differ by how it was loaded, in Live
    12.4.16 as in the fake from ableton-live-mcp 2.4.2: 3 through the
    browser, 6 by path; a One-Shot pad plays one hit at a time either way."""
    out = {}
    for track in sim.song.tracks:
        for device in track.devices:
            if device.class_name == "OriginalSimpler":
                out[track.name] = {"mode": device.playback_mode, "file": device.sample.file_path,
                                   "voices": device.voices}
            for pad in getattr(device, "drum_pads", []) or []:
                for chain in pad.chains:
                    s = chain.devices[0]
                    out[str(pad.note)] = {"mode": s.playback_mode, "file": s.sample.file_path,
                                          "choke_group": chain.choke_group}
    return out


def played(state: dict) -> list[dict]:
    """The set's tracks as describe_song gives them, without what depends on
    how a file was loaded rather than what plays: a Simpler's and a pad
    chain's name (Live names them after the file through the browser;
    "Simpler" and no name by path), a pad's voices (simplers) and the rack's
    pad names (its chains')."""
    tracks = json.loads(json.dumps(state["tracks"]))
    for t in tracks:
        for d in t["devices"]:
            if d["class_name"] == "OriginalSimpler":
                d.pop("name")
            for chain in (d.get("chains") or {}).values():
                for c in chain:
                    c.pop("name")
                    for s in c["devices"]:
                        s.pop("name")
                        s.pop("voices")
            d.pop("pads", None)
    return tracks


# What a new set's track holds, and a built track where the plan sets nothing else
NEW_TRACK = {"arm": False, "clip_slots": {}, "color": "#000000", "color_index": 1, "pan": {"display": "C", "value": 0},
             "sends": [{"display": "-inf dB", "value": 0}] * 2, "solo": False}
STEM_FADER = {"display": "-4.0 dB", "value": 0.6732}


def track(name: str, volume: dict | None = None, mute: bool = False, devices=(), clips=(), audio: bool = False):
    return {**NEW_TRACK, "name": name, "type": "audio" if audio else "midi", **({"input": "Ext. In"} if audio else {}),
            "volume": volume or {"display": "0.0 dB", "value": 0.85}, "mute": mute, "devices": list(devices),
            **({"arrangement_clips": list(clips)} if clips else {})}


def clip(name: str, length: float, notes: list | None = None, file: str | None = None) -> dict:
    return {"name": name, "start_time": 0, "end_time": length, "length": length, "loop_start": 0, "loop_end": length,
            "looping": True, "muted": False, "start_marker": 0, **({"notes": notes} if notes is not None else {}),
            **({"file_path": file} if file else {})}


def simpler(**params) -> dict:
    """A Simpler's parameters in the set: the plan's (live.build_spec), the
    blueprint's F On off and Fade In 0 on a pad (PAD_OFF), and the fake's
    own for the rest."""
    return {"Detune": "0 ct", "Device On": "1.00", "F On": "Off", "Fade In": "0.0 ms", "Fade Out": "0.1 ms",
            "Glide Mode": "Off", "Glide Time": "50.0 ms", "S Start": "0 %", "Snap": "Off", "Transpose": "0 st",
            "Trigger Mode": "Gate", "Ve Release": "50.0 ms", "Vol < Vel": "35 %", **params}


def the_set(plan: dict, clips: dict) -> list[dict]:
    """a_plan's set as live-build --direct built it, with the clips given
    by track name: the 808's Simpler on its one-shot at 0 dB, +36 st and
    -12 ct, with Glide; three pads at their Volumes (the snare's raised by
    its file's 4.3 dB scale), the hat at -8 ct in choke group 1; Drift
    muted; the stems muted but for other; the faders from the plan's
    levels."""
    pads = plan["tracks"]["drums"]["pads"]

    def pad(note: str, volume: str, detune: str = "0 ct", choke: int = 0, gain: float = 0.0) -> list[dict]:
        return [{"choke_group": choke, "devices": [{"class_name": "OriginalSimpler", "reversed": False,
                                                    "sample": pad_sample(pads[note]["path"], gain),
                                                    "parameters": simpler(Volume=volume, Detune=detune)}]}]

    return [track("1-MIDI"), track("2-MIDI"),
            track("808 (synth)", {"display": "-13.9 dB", "value": 0.3808}, clips=clips["808 (synth)"], devices=[
                {"class_name": "OriginalSimpler", "reversed": False, "sample": plan["tracks"]["808"]["sample"],
                 "voices": 1, "parameters": simpler(Detune="-12 ct", **{"Fade In": "0.1 ms"}, Transpose="36 st",
                                                    Volume="-0.0 dB", **{"Glide Mode": "Glide",
                                                                         "Glide Time": "45.0 ms"})}]),
            track("Drums (library)", {"display": "-12.6 dB", "value": 0.4104}, clips=clips["Drums (library)"],
                  devices=[{"class_name": "DrumGroupDevice", "name": "Drum Rack", "parameters": {"Device On": "1.00"},
                            "chains": {"36": pad("36", "-6.6 dB"), "38": pad("38", "-2.5 dB", gain=-4.3),
                                       "42": pad("42", "-7.2 dB", "-8 ct", choke=1)}}]),
            track("other (notes)", STEM_FADER, mute=True, clips=clips["other (notes)"], devices=[
                {"class_name": "Drift", "name": "Drift",
                 "parameters": {"Device On": "1.00", "Filter Freq": "4.72 kHz", "Filter Res": "0.20",
                                "Volume": "-12.0 dB"}}]),
            *(track(f"REF {s}", STEM_FADER, mute=s != "other", audio=True, clips=clips[f"REF {s}"])
              for s in ("drums", "bass", "other", "vocals"))]


def test_live_build_builds_the_plan(live, tmp_path):
    """The plan into an empty set by live-build (blueprint.json,
    build_from_blueprint, files by path): every track, device, parameter,
    note, Volume, choke group, fader, mute and audio clip, the tempo,
    signature, locators, loop and selected track as live-build --direct
    built them; every Simpler holding its copy in the User Library, the
    pads One-Shot and the 808 Classic on one voice; and the report in
    live-build's keys, with nothing wrong."""
    plan, work = a_plan(tmp_path)
    sim = live()
    logged = []
    report = asyncio.run(B.build(work, log=logged.append))
    state = sim.state()
    assert {k: state[k] for k in ("tempo", "signature", "loop", "locators", "selected_track")} == {
        "tempo": 140, "signature": "4/4", "loop": {"length": 16, "on": False, "start": 0},
        "locators": [{"name": "A (1)", "time": 0}, {"name": "B (3)", "time": 8}], "selected_track": "808 (synth)"}
    stems = {s: plan["stems"][s]["path"] for s in plan["stems"]}
    assert played(state) == the_set(plan, {
        "808 (synth)": [clip("808", 16, [[24, 0, 1.5, 115], [27, 4, 1, 100]])],
        "Drums (library)": [clip("Drums", 16, [[36, 0, 0.25, 100], [42, 0.5, 0.25, 100], [38, 1, 0.25, 100],
                                               [36, 4, 0.25, 100], [38, 5, 0.25, 100]])],
        "other (notes)": [clip("other", 16, [[60, 0, 2, 90]])],
        **{f"REF {s}": [clip(s, 8, file=path)] for s, path in stems.items()}})
    lib = live.library / GENERATED
    pads = plan["tracks"]["drums"]["pads"]
    t808 = plan["tracks"]["808"]
    assert simplers(sim) == {
        "808 (synth)": {"mode": 0, "file": str(lib / t808["sample"]), "voices": 1},
        "36": {"mode": 1, "file": str(lib / pad_sample(pads["36"]["path"])), "choke_group": 0},
        "38": {"mode": 1, "file": str(lib / pad_sample(pads["38"]["path"], -4.3)), "choke_group": 0},
        "42": {"mode": 1, "file": str(lib / pad_sample(pads["42"]["path"])), "choke_group": 1}}
    # the copies: the one-shot and two pads as they are, the snare scaled to full scale, a float WAV
    for name, src in ((t808["sample"], t808["sample_path"]), (pad_sample(pads["36"]["path"]), pads["36"]["path"]),
                      (pad_sample(pads["42"]["path"]), pads["42"]["path"])):
        assert (lib / name).read_bytes() == Path(src).read_bytes(), name
    scaled = lib / pad_sample(pads["38"]["path"], -4.3)
    x = sf.read(scaled, dtype="float64")[0]
    assert sf.info(scaled).subtype == "FLOAT" and 0.988 < np.max(np.abs(x)) <= 1.0
    assert np.allclose(x, sf.read(pads["38"]["path"], dtype="float64")[0] * 10 ** (-4.3 / 20), atol=1e-6)
    # the report, in live-build's keys
    assert {k: v for k, v in report.items() if k != "build_from_blueprint"} == {
        "tracks": [{"track": "808 (synth)", "devices": ["Simpler"], "notes": 2},
                   {"track": "Drums (library)", "devices": ["Drum Rack"], "notes": 5},
                   {"track": "other (notes)", "devices": ["Drift"], "notes": 1},
                   *({"track": f"REF {s}", "devices": [], "notes": 0, "audio_clips": 1} for s in stems)],
        "locators": 2, "choke_groups": {"42": {"plan": 1, "live": 1, "same": True}}, "volumes_held": 3,
        "samples_checked": 4, "808_voices": {"want": 1, "live": 1, "same": True},
        "tempo": {"want": 140.0, "live": 140.0, "same": True},
        "time_signature": {"want": "4/4", "live": "4/4", "same": True},
        "files": {"folder": str(lib), "copied": 3, "scaled": 1}, "ok": True}
    assert len(logged) == 2 and logged[0].startswith("blueprint: 7 tracks, 3 pads (1 scaled), an 808 one-shot, ")
    assert json.loads(logged[-1])["ok"] is True
    assert (work / "blueprint.json").is_file()


def test_an_excerpt(live, tmp_path):
    """--bars 2-3: the notes, locators and devices as live-build --direct
    built them, the stems cut at the same instants (bar 2's start and bar 3's
    end at the plan's tempo). The stems are named by their content in
    excerpt-2-3/ beside the blueprint, and a MIDI clip without notes stays
    where a track plays none in those bars (--direct made none)."""
    plan, work = a_plan(tmp_path)
    sim = live()
    logged = []
    report = asyncio.run(B.build(work, bars=(2, 3), log=logged.append))
    state = sim.state()
    assert {k: state[k] for k in ("tempo", "signature", "loop", "locators", "selected_track")} == {
        "tempo": 140, "signature": "4/4", "loop": {"length": 16, "on": False, "start": 0},
        "locators": [{"name": "B (3)", "time": 4}], "selected_track": "808 (synth)"}
    tracks = played(state)
    a, b = (round(beat * 60 / 140 * SR) for beat in (4, 12))
    for t in tracks:  # each stem cut sample for sample, named by its content
        for c in t.get("arrangement_clips", []):
            if "file_path" in c:
                stem = t["name"].removeprefix("REF ")
                cut = Path(c.pop("file_path"))
                assert cut.parent == work / "excerpt-2-3" and c.pop("name") == cut.stem
                assert cut.name.startswith(f"{stem} ") and len(cut.stem) == len(stem) + 9
                assert np.array_equal(sf.read(cut, dtype="float64")[0],
                                      sf.read(plan["stems"][stem]["path"], dtype="float64")[0][a:b])
    want = the_set(plan, {"808 (synth)": [clip("808", 8, [[27, 0, 1, 100]])],
                          "Drums (library)": [clip("Drums", 8, [[36, 0, 0.25, 100], [38, 1, 0.25, 100]])],
                          "other (notes)": [clip("other", 8, [])],  # its note is in bar 1
                          **{f"REF {s}": [clip(s, 8)] for s in plan["stems"]}})
    for t in want:
        for c in t.get("arrangement_clips", []):
            if t["name"].startswith("REF "):
                c.pop("name")
    assert tracks == want
    assert report["excerpt"] == {"bars": [2, 3], "beats": [4.0, 12.0],
                                 "seconds": [round(4 * 60 / 140, 6), round(12 * 60 / 140, 6)]}
    assert any(line.startswith("excerpt: bars 2-3, ") for line in logged)
    assert report["ok"] is True and report["samples_checked"] == 4 and "wrong_samples" not in report


def test_a_3_4_plan_is_built_in_3_4(live, tmp_path):
    """Live's time signature is the plan's meter, set before the locators,
    which land on the plan's bar lines in it."""
    plan, work = a_plan(tmp_path, meter=(3, 4))
    sim = live()
    report = asyncio.run(B.build(work, log=lambda _: None))
    state = sim.state()
    assert state["signature"] == "3/4" and report["time_signature"] == {"want": "3/4", "live": "3/4", "same": True}
    assert state["locators"] == [{"name": "A (1)", "time": 0}, {"name": "B (3)", "time": 6}]
    assert report["ok"] is True and "locators_moved" not in report


def test_stems_out_takes_the_excerpts_stems(live, tmp_path):
    """--bars with --stems-out: the excerpt is made in-process by
    ableton-live-mcp's blueprint.excerpt, its stems cut into that folder."""
    plan, work = a_plan(tmp_path)
    live()
    report = asyncio.run(B.build(work, bars=(1, 2), stems_out=tmp_path / "cuts", log=lambda _: None))
    assert report["ok"] is True and report["excerpt"]["bars"] == [1, 2]
    assert sorted(p.name.split(" ")[0] for p in (tmp_path / "cuts").iterdir()) == ["bass", "drums", "other",
                                                                                   "vocals"]


def test_a_relative_stems_out(live, tmp_path, monkeypatch):
    """--stems-out as given on the command line: the excerpt passed as an
    object names its clips by absolute paths, or none could be resolved."""
    plan, work = a_plan(tmp_path)
    live()
    from MCP_Server import blueprint as bp
    made = []
    excerpt = bp.excerpt
    monkeypatch.setattr(bp, "excerpt", lambda *a, **k: made.append(excerpt(*a, **k)) or made[-1])
    monkeypatch.chdir(tmp_path)
    report = asyncio.run(B.build(work, bars=(1, 2), stems_out="cuts", log=lambda _: None))
    assert report["ok"] is True
    files = [c["file"] for t in made[0]["tracks"] for c in t.get("audio_clips") or []]
    assert files and all(Path(f).is_absolute() for f in files)
    assert sorted(p.name.split(" ")[0] for p in (tmp_path / "cuts").iterdir()) == ["bass", "drums", "other",
                                                                                   "vocals"]


def test_the_command_and_a_set_that_is_not_empty(live, tmp_path, capsys):
    """refdissect live-build WORK builds the plan (exit 0); built again into
    the same set, build_from_blueprint refuses it, and live-build ends in
    that RuntimeError."""
    plan, work = a_plan(tmp_path)
    live()
    assert main(["live-build", str(work)]) == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["ok"] is True
    with pytest.raises(RuntimeError, match="The set is not empty"):
        main(["live-build", str(work)])


def test_a_plan_moved_onto_lives_bars(live, tmp_path):
    """A plan made with --align-bar1 (live.onto_live_bars): a_plan's first
    bar line, 0.053 beats in (its locators), lies 22.7 ms after 0 s, so it
    moves 3.947 beats later onto bar 2. Built by build_from_blueprint, the
    stems' clips start there, the notes and locators moved alike; bars 2-3
    of it start on the first bar line, the stems cut from it."""
    plan, work = a_plan(tmp_path)
    bps = plan["tempo"] / 60
    analysis = {"grid": {"phase": 0.053 / bps}, "source": {"duration_s": 16 / bps},
                "sections": [{"name": "A", "start_bar": 1}, {"name": "B", "start_bar": 3}]}
    onto_live_bars(plan, analysis)
    assert plan["align_bar1"]["moved_beats"] == 3.947 and plan["length_beats"] == 20.0
    (work / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    sim = live()
    report = asyncio.run(B.build(work, log=lambda _: None))
    assert report["ok"] is True and "locators_moved" not in report
    state = sim.state()
    assert state["locators"] == [{"name": "A (1)", "time": 4}, {"name": "B (3)", "time": 12}]
    refs = [c for t in state["tracks"] if t["name"].startswith("REF ") for c in t["arrangement_clips"]]
    assert len(refs) == 4 and {c["start_time"] for c in refs} == {3.947}
    drums = next(t for t in state["tracks"] if t["name"] == "Drums (library)")["arrangement_clips"][0]
    assert sorted(n[1] for n in drums["notes"]) == pytest.approx([3.947, 4.447, 4.947, 7.947, 8.947])
    sim = live()
    report = asyncio.run(B.build(work, bars=(2, 3), log=lambda _: None))
    assert report["ok"] is True and report["excerpt"]["beats"] == [4.0, 12.0]
    state = sim.state()
    assert state["locators"] == [{"name": "A (1)", "time": 0}]  # B, on bar 4, is past the excerpt
    refs = [c for t in state["tracks"] if t["name"].startswith("REF ") for c in t["arrangement_clips"]]
    assert {c["start_time"] for c in refs} == {0.0}
    source = sf.read(next(iter(plan["stems"].values()))["path"], dtype="float64")[0]
    cut = sf.read(next(c["file_path"] for c in refs if Path(c["file_path"]).name.startswith("drums")),
                  dtype="float64")[0]
    a = int(round((4.0 - 3.947) / bps * SR))
    assert np.array_equal(cut, source[a:a + int(round(8.0 / bps * SR))])


def test_the_server_is_imported_with_the_full_profile():
    """live-build sets ABLETON_MCP_PROFILE=full before it imports
    MCP_Server.server, which registers its tools once, on import: in a fresh
    process with no profile set, build_from_blueprint is there."""
    import subprocess
    import sys

    code = ("import asyncio, os\n"
            "os.environ.pop('ABLETON_MCP_PROFILE', None)\n"
            "from refdissect.blueprint import _server\n"
            "tools = asyncio.run(_server().list_tools())\n"
            "print([t.name for t in getattr(tools, 'tools', tools) if t.name == 'build_from_blueprint'])\n")
    env = {k: v for k, v in os.environ.items() if k != "ABLETON_MCP_PROFILE"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, env=env)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == "['build_from_blueprint']"
