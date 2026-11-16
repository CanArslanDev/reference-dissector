"""live-build through ableton-live-mcp's build_from_blueprint, against its
simulator (the real Remote Script over fake Live, MCP_Server/simulator.py and
fake_live.py): a synthetic plan built by the new path and by live-build
--direct, each into an empty set of its own, and the two sets compared. Needs
the [live] extra (ableton-live-mcp with blueprints); skipped without it, and
failed instead where REFDISSECT_REQUIRE_LIVE is set (CI). No
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

from test_blueprint import SR, _wav, plan_for  # noqa: E402

from refdissect import blueprint as B  # noqa: E402
from refdissect import live as L  # noqa: E402
from refdissect.cli import main  # noqa: E402
from refdissect.live import GENERATED, scale_files_to_full  # noqa: E402

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
    tmp_path. The fake browser lists the copies live-build --direct loads by
    name, and a file it drops on a pad loads One-Shot from the User Library,
    as Live's browser does (the fake's own browser holds a few fixed items)."""
    from mcp.server.mcpserver import MCPServer
    from MCP_Server import als, extended_tools
    from MCP_Server import server as live_server
    from MCP_Server.fake_live import FakeBrowserItem, FakeSample

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

    def start(names: list[str]):
        disconnect()
        for sim in running:
            sim.stop()
        session = tmp_path / f"session-{len(running)}.json"
        monkeypatch.setenv("ABLETON_MCP_SESSION_FILE", str(session))
        sim = simulator_module.Simulator(simulator_module.free_port(), session)
        sim.run_setup(EMPTY_SET)
        browser = sim.browser
        browser.samples.children.extend(FakeBrowserItem(n, source="User Library") for n in names)
        dropped = browser.load_item

        def load_item(item):
            dropped(item)
            track = browser.song.view.selected_track
            rack = next((d for d in track.devices if d.class_name == "DrumGroupDevice"), None)
            pad = rack.view.selected_drum_pad if rack is not None else None
            simpler = pad.chains[0].devices[0] if pad is not None else track.devices[-1]
            simpler.sample = FakeSample(str(library / GENERATED / item.name))
            if pad is not None:
                simpler.playback_mode = 1  # One-Shot

        object.__setattr__(browser, "load_item", load_item)
        sim.start()
        running.append(sim)
        return sim

    start.library = library
    yield start
    disconnect()
    for sim in running:
        sim.stop()


def a_plan(tmp_path: Path) -> tuple[dict, Path]:
    """plan_for's plan with what the build sets beyond it: a pad in a choke
    group, a file over full scale (copied scaled), an 808 on one voice with
    Glide, stems of 8 s (the plan's 16 beats at 140 BPM: 6.9 s); written to
    <tmp>/Song/work/plan.json."""
    plan = plan_for(tmp_path)
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


def copies(plan: dict) -> list[str]:
    """The names live-build --direct copies the files under, for the fake browser."""
    t808 = plan["tracks"]["808"]
    return [t808["sample"]] + [L.pad_sample(p["path"], p.get("file_db") or 0.0)
                               for p in plan["tracks"]["drums"]["pads"].values()]


def simplers(sim) -> dict:
    """Each Simpler of the set, by "808" or pad note: its playback mode, the
    file it holds, its voices (the 808's), and its chain's choke group (a
    pad's). A pad's voices differ by how it was loaded, in Live 12.4.16 as in
    the fake from ableton-live-mcp 2.4.2: 3 through the browser, 6 by path;
    a One-Shot pad plays one hit at a time either way."""
    out = {}
    for track in sim.song.tracks:
        for device in track.devices:
            if device.class_name == "OriginalSimpler":
                out[track.name] = {"mode": device.playback_mode, "file": device.sample.file_path,
                                   "voices": device.voices}
            for pad in getattr(device, "drum_pads", []) or []:
                for chain in pad.chains:  # a pad's voices: below
                    s = chain.devices[0]
                    out[str(pad.note)] = {"mode": s.playback_mode, "file": s.sample.file_path,
                                          "choke_group": chain.choke_group}
    return out


def comparable(state: dict, *, skip_empty_clips: bool = False) -> dict:
    """The set as describe_song gives it, without what differs by how a file
    was loaded rather than what plays: a Simpler's and a pad chain's name
    (Live names them after the file through the browser; "Simpler" and no
    name by path), its voices, the rack's pad names (its chains'), each pad's "F On"
    (the blueprint turns it off, PAD_OFF; the browser leaves Live's
    default), and the undo history and browser log."""
    out = {k: state[k] for k in ("tempo", "signature", "loop", "locators", "selected_track")}
    tracks = []
    for t in state["tracks"]:
        t = json.loads(json.dumps(t))
        for d in t["devices"]:
            if d["class_name"] == "OriginalSimpler":
                d.pop("name")
            for chain in (d.get("chains") or {}).values():
                for c in chain:
                    c.pop("name")
                    for s in c["devices"]:
                        s.pop("name")
                        s.pop("voices", None)  # 3 through the browser, 6 by path (simplers)
                        s["parameters"].pop("F On")
                        s["parameters"].pop("Fade In", None)
            d.pop("pads", None)
        if skip_empty_clips and "arrangement_clips" in t:
            t["arrangement_clips"] = [c for c in t["arrangement_clips"] if c.get("notes", True)]
            if not t["arrangement_clips"]:
                del t["arrangement_clips"]
        for c in t.get("arrangement_clips", []):
            if "file_path" in c:  # compared by content
                c["file_path"] = sf.read(c["file_path"])[0].tobytes()
                c.pop("name")
        tracks.append(t)
    out["tracks"] = tracks
    return out


@pytest.fixture
def both(live, tmp_path):
    """Build a_plan by both paths; (plan, work, {path: (state, simplers, report, log)})."""

    def run(bars=None, stems_out=None):
        plan, work = a_plan(tmp_path)
        out = {}
        sim = live(copies(plan))
        logged = []
        report = asyncio.run(L.build(plan, user_library=live.library, log=logged.append, bars=bars,
                                     stems_out=stems_out or (work / "direct-excerpt" if bars else None)))
        out["direct"] = (sim.state(), simplers(sim), report, logged)
        sim = live([])
        logged = []
        report = asyncio.run(B.build(work, bars=bars, log=logged.append))
        out["blueprint"] = (sim.state(), simplers(sim), report, logged)
        return plan, work, out

    return run


def test_the_new_path_builds_the_same_set_as_direct(both):
    """The same plan into two empty sets, by live-build (blueprint.json,
    build_from_blueprint, files by path) and by live-build --direct
    (build_tracks, every file through the browser by name): the same tracks,
    devices, parameters, notes, Volumes, choke groups, mixer, audio clips,
    tempo, signature, locators, loop and selected track; every Simpler
    holding the same copy in the User Library, the pads One-Shot and the 808
    Classic on one voice; and the report in --direct's keys, with nothing
    wrong in either."""
    plan, work, out = both()
    (s_direct, x_direct, r_direct, _), (s_new, x_new, r_new, logged) = out["direct"], out["blueprint"]
    assert comparable(s_new) == comparable(s_direct)
    assert x_new == x_direct
    assert x_new["808 (synth)"] == {"mode": 0, "file": str(live_generated(work) / plan["tracks"]["808"]["sample"]),
                                    "voices": 1}
    assert all(x_new[n]["mode"] == 1 for n in plan["tracks"]["drums"]["pads"])
    assert x_new["42"]["choke_group"] == 1 and x_new["38"]["file"].endswith(" -4.30dB.wav")
    assert all(Path(s["file"]).parent == live_generated(work) for s in x_new.values())
    # the pads' filter: off by the blueprint (PAD_OFF)
    rack = next(t for t in s_new["tracks"] if t["name"] == "Drums (library)")["devices"][0]
    assert {c[0]["devices"][0]["parameters"]["F On"] for c in rack["chains"].values()} == {"Off"}
    # and their Fade In at 0, where a Simpler made by path starts at 0.1 ms in Live 12.4.16
    assert {c[0]["devices"][0]["parameters"]["Fade In"] for c in rack["chains"].values()} == {"0.0 ms"}
    # the report, in --direct's keys
    for key in ("samples_checked", "volumes_held", "choke_groups"):
        assert r_new[key] == r_direct[key], key
    assert r_new["samples_checked"] == 4 and r_new["volumes_held"] == 3
    assert r_new["choke_groups"] == {"42": {"plan": 1, "live": 1, "same": True}}
    for key in ("wrong_samples", "volumes_not_held", "locators_moved", "problems"):
        assert key not in r_new and key not in r_direct, key
    assert r_new["808_voices"] == {"want": 1, "live": 1, "same": True}
    assert r_new["time_signature"]["same"] and r_new["ok"] is True
    assert r_new["tempo"]["same"] is True  # 2.4.0 checks the tempo too
    assert [t["track"] for t in r_new["tracks"]] == [t["track"] for t in r_direct["tracks"]]
    assert json.loads(logged[-1])["ok"] is True and logged[0].startswith("blueprint: ")
    assert (work / "blueprint.json").is_file()


def live_generated(work: Path) -> Path:
    return work.parent.parent / "User Library" / GENERATED


def test_an_excerpt_by_the_new_path_and_by_direct(both):
    """--bars 2-3 by both paths: the same notes, locators and stems' audio,
    the stems cut at the same instants. What differs: the new path keeps a
    MIDI clip without notes where a track plays none in those bars (--direct
    makes none), and names the cut stems by their content in excerpt-2-3/
    beside the blueprint (--direct: as the stems, in --stems-out)."""
    plan, work, out = both(bars=(2, 3))
    (s_direct, _, r_direct, _), (s_new, _, r_new, logged) = out["direct"], out["blueprint"]
    assert comparable(s_new, skip_empty_clips=True) == comparable(s_direct)
    melodic = next(t for t in s_new["tracks"] if t["name"] == "other (notes)")
    assert [c["notes"] for c in melodic["arrangement_clips"]] == [[]]  # its note is in bar 1
    assert "arrangement_clips" not in next(t for t in s_direct["tracks"] if t["name"] == "other (notes)")
    stems = [c["file_path"] for t in s_new["tracks"] for c in t.get("arrangement_clips", []) if "file_path" in c]
    assert {Path(p).parent for p in stems} == {work / "excerpt-2-3"}
    assert r_new["excerpt"] == {"bars": [2, 3], "beats": [4.0, 12.0],
                                "seconds": [round(4 * 60 / 140, 6), round(12 * 60 / 140, 6)]}
    assert any(line.startswith("excerpt: bars 2-3, ") for line in logged)
    assert r_new["samples_checked"] == r_direct["samples_checked"] == 4 and "wrong_samples" not in r_new
    assert s_new["locators"] == s_direct["locators"] == [{"name": "B (3)", "time": 4}]


def test_stems_out_takes_the_excerpts_stems(live, tmp_path):
    """--bars with --stems-out: the excerpt is made in-process by
    ableton-live-mcp's blueprint.excerpt, its stems cut into that folder."""
    plan, work = a_plan(tmp_path)
    live([])
    report = asyncio.run(B.build(work, bars=(1, 2), stems_out=tmp_path / "cuts", log=lambda _: None))
    assert report["ok"] is True and report["excerpt"]["bars"] == [1, 2]
    assert sorted(p.name.split(" ")[0] for p in (tmp_path / "cuts").iterdir()) == ["bass", "drums", "other",
                                                                                   "vocals"]


def test_a_relative_stems_out(live, tmp_path, monkeypatch):
    """--stems-out as given on the command line: the excerpt passed as an
    object names its clips by absolute paths, or none could be resolved."""
    plan, work = a_plan(tmp_path)
    live([])
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
    """refdissect live-build WORK runs the new path (exit 0); built again into
    the same set, build_from_blueprint refuses it, and live-build ends in
    that RuntimeError, as --direct did."""
    plan, work = a_plan(tmp_path)
    live([])
    assert main(["live-build", str(work)]) == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["ok"] is True
    with pytest.raises(RuntimeError, match="The set is not empty"):
        main(["live-build", str(work)])


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
