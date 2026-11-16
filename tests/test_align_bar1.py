"""plan --align-bar1 (live.onto_live_bars): a plan moved later onto Live's
bars where the grid's first bar line lies more than 20 ms from 0 s, and with
the option off the same plan.json, MIDI files and blueprint as before it.
Synthetic analyses at 120 BPM in 4/4 (a bar of 2 s, 4 beats) over 4 s of
noise stems; no Live, no Demucs, no sample library."""

import hashlib
import json
import os
import re
from pathlib import Path

import mido
import numpy as np
import pytest
import soundfile as sf

from refdissect.audio import SR
from refdissect.cli import main

BPM = 120.0
BAR_S = 2.0  # 4/4 at 120 BPM


def _work(tmp_path: Path, phase: float) -> Path:
    """A work folder as analyze, melodic and match leave one, its grid's
    first bar line at phase (s): a snare on beats 2 and 4 from it, hats
    every 8th from 0.05 s (in the pickup too, where the line comes after 0
    s), two 808 notes, the second with a slide, two melodic notes, sections
    A (bar 1) and B (bar 2), and noise stems of 4 s in <tmp>/Song/stems."""
    rng = np.random.default_rng(7)
    stems = tmp_path / "Song" / "stems"
    stems.mkdir(parents=True)
    for stem in ("drums", "bass", "other"):
        sf.write(stems / f"{stem}.wav", rng.standard_normal(4 * SR) * 0.1, SR)
    lib = tmp_path / "library"
    lib.mkdir()
    for name in ("snare", "hat"):
        sf.write(lib / f"{name}.wav", rng.standard_normal(SR // 8) * 0.5, SR)
    snares = [round(phase + 0.5 + k, 4) for k in range(4) if 0 <= phase + 0.5 + k < 4]
    hats = [round(0.05 + 0.25 * k, 4) for k in range(15)]
    analysis = {
        "source": {"stems": str(stems), "duration_s": 4.0},
        "grid": {"bpm": BPM, "phase": phase, "beats_per_bar": 4, "beat_unit": 4},
        "templates": {"sound_1": {"detected_hits": len(snares), "clean": 1.0},
                      "sound_2": {"detected_hits": len(hats), "clean": 1.0}},
        "events": {"sound_1": [{"t": t, "gain": 1.0} for t in snares],
                   "sound_2": [{"t": t, "gain": 0.5 + 0.03 * k} for k, t in enumerate(hats)]},
        "notes_808": [{"t": 0.45, "end": 0.95, "pitch": 24, "peak": 1.0},
                      {"t": 1.45, "end": 2.3, "pitch": 26, "peak": 0.8,
                       "glide": {"kind": "legato", "from_midi": 24, "to_midi": 26, "ms": 60.0}}],
        "sections": [{"name": "A", "start_bar": 1, "end_bar": 1}, {"name": "B", "start_bar": 2, "end_bar": 2}]}
    matches = {"sound_1": {"role": "snare", "verdict": "same sample", "same_as": None,
                           "top": [{"path": str(lib / "snare.wav"), "ncc": 0.97, "semitones": 0.0}]},
               "sound_2": {"role": "hat", "verdict": "same sample", "same_as": None,
                           "top": [{"path": str(lib / "hat.wav"), "ncc": 0.96, "semitones": 0.0}]}}
    melodic = {"stems": {"other": {"tuning_cents": 0, "notes": [
        {"pitch": 60, "start_time": 1.0, "duration": 1.0, "velocity": 90},
        {"pitch": 64, "start_time": 3.5, "duration": 2.0, "velocity": 80}]}}}
    work = tmp_path / "Song" / "work"
    work.mkdir()
    for name, obj in (("analysis", analysis), ("matches", matches), ("melodic", melodic)):
        (work / f"{name}.json").write_text(json.dumps(obj), encoding="utf-8")
    return work


def _inputs(work: Path) -> tuple[tuple, dict]:
    """make_plan's arguments as pipeline.plan reads them from the work folder:
    (analysis, matches, synth, stems_dir) and the melodic keyword."""
    analysis, matches, melodic = (json.loads((work / f"{n}.json").read_text(encoding="utf-8"))
                                  for n in ("analysis", "matches", "melodic"))
    return (analysis, matches, None, analysis["source"]["stems"]), {"melodic": melodic}


def _digests(tmp_path: Path, work: Path) -> dict[str, str]:
    """What `refdissect plan` and `refdissect blueprint` wrote into work, as
    SHA-256 starts of the files' bytes with the tmp folder's path, the
    blueprint's made_by (the code's version and commit) and the plan's SHA-1
    in its meta (of the plan's bytes, path included) taken out."""
    roots = sorted({str(tmp_path), str(tmp_path.resolve())}, key=len, reverse=True)

    def bare(data: bytes) -> bytes:
        text = data.decode("utf-8")
        for root in roots:
            text = text.replace(root, "TMP")
        text = re.sub(r'"made_by": "[^"]*"', '"made_by": "-"', text)
        return re.sub(r'"plan_sha1": "[0-9a-f]+"', '"plan_sha1": "-"', text).encode("utf-8")

    files = {"plan.json": bare((work / "plan.json").read_bytes()),
             "blueprint.json": bare((work / "blueprint.json").read_bytes())}
    files |= {f"midi/{p.name}": p.read_bytes() for p in sorted((work / "midi").glob("*.mid"))}
    return {name: hashlib.sha256(data).hexdigest()[:16] for name, data in files.items()}


# What the code before the option (ffb045c) wrote for _work(tmp, 0.4): `refdissect plan WORK`, then
# `refdissect blueprint WORK`, read by _digests
BEFORE = {"plan.json": "ffddace6989002c1", "blueprint.json": "9bd46482751cec46", "midi/808.mid": "5111faf6fc0a77af",
          "midi/drums.mid": "7c0baaf0c1d1da49", "midi/other.mid": "ae07b43fff6f182b"}


def test_with_the_option_off_plan_writes_what_it_wrote_before(tmp_path):
    """The option is off by default: plan.json, the MIDI files and the
    blueprint of a plan whose first bar line lies 0.4 s in are, byte for
    byte, what the code before the option wrote (BEFORE), and `--align-bar1`
    writes others."""
    work = _work(tmp_path, 0.4)
    assert main(["plan", str(work)]) == 0
    assert main(["blueprint", str(work)]) == 0
    plan = json.loads((work / "plan.json").read_text(encoding="utf-8"))
    assert "align_bar1" not in plan
    assert _digests(tmp_path, work) == BEFORE
    assert main(["plan", str(work), "--align-bar1"]) == 0
    assert main(["blueprint", str(work)]) == 0
    moved = _digests(tmp_path, work)
    assert all(moved[name] != BEFORE[name] for name in BEFORE)


def _plans(tmp_path: Path, phase: float, **kwargs) -> tuple[dict, dict]:
    """(the plan without the option, the plan with it) for _work(tmp_path, phase)."""
    from refdissect.live import make_plan
    args, melodic = _inputs(_work(tmp_path, phase))
    return make_plan(*args, **melodic, **kwargs), make_plan(*args, **melodic, **kwargs, align_bar1=True)


def _times(plan: dict) -> dict[str, list]:
    """Every timed thing of a plan: each track's notes (start, then the rest
    as is), the 808's slides, the locators."""
    tracks = plan["tracks"]
    parts = {"drums": tracks["drums"]["notes"], "808": tracks["808"]["notes"],
             "slides": tracks["808"].get("slides") or [], **tracks.get("melodic", {})}
    out = {k: [(n["start_time"], {f: v for f, v in n.items() if f != "start_time"}) for n in
               (v["notes"] if isinstance(v, dict) else v)] for k, v in parts.items()}
    out["locators"] = [tuple(loc) for loc in plan["locators"]]
    return out


def _moved_by(plain: dict, moved: dict, beats: float, locators: float) -> None:
    """moved is plain with every note and slide `beats` later, its locators
    `locators` later, and the rest the same (pads, levels, stems, the 808's
    one-shot settings)."""
    a, b = _times(plain), _times(moved)
    assert {k: len(v) for k, v in a.items()} == {k: len(v) for k, v in b.items()}
    for key in a:
        if key == "locators":
            assert [(n, t + locators) for n, t in a[key]] == [(n, pytest.approx(t, abs=1e-9)) for n, t in b[key]]
            continue
        assert [(t + beats, rest) for t, rest in a[key]] == [(pytest.approx(t, abs=1e-9), rest) for t, rest in b[key]]
    for key in ("tempo", "beats_per_bar", "beat_unit", "stems", "levels"):
        assert moved[key] == plain[key], key
    assert moved["tracks"]["drums"]["pads"] == plain["tracks"]["drums"]["pads"]
    assert {k: v for k, v in moved["tracks"]["808"].items() if k not in ("notes", "slides")} == \
        {k: v for k, v in plain["tracks"]["808"].items() if k not in ("notes", "slides")}


def _stem_starts(plan: dict) -> set[float]:
    from refdissect.live import build_spec
    return {c["start"] for t in build_spec(plan) for c in t.get("audio_clips") or []}


def test_a_first_bar_line_after_0_s_lands_on_bar_2(tmp_path):
    """The first bar line 0.4 s in: everything moves a bar less that, 1.6 s
    (3.2 beats) later. The line lands on Live's bar 2 (beat 4), bar 1 holds
    the pickup (the hats from 0.05 s), the locators sit on Live's bar lines,
    the stems' clips start at beat 3.2, and the plan ends on the first bar
    line after the moved song's end (8 + 3.2 beats: 12)."""
    from refdissect.live import write_midi
    plain, moved = _plans(tmp_path, 0.4)
    assert plain["length_beats"] == 8.0 and plain["locators"] == [["A (1)", 0.8], ["B (2)", 4.8]]
    _moved_by(plain, moved, 3.2, 3.2)
    assert moved["locators"] == [["A (1)", 4.0], ["B (2)", 8.0]]
    assert moved["length_beats"] == 12.0
    assert moved["align_bar1"] == {"first_bar_line_s": 0.4, "moved_s": 1.6, "moved_beats": 3.2, "lands_on_bar": 2}
    assert _stem_starts(plain) == {0.0} and _stem_starts(moved) == {3.2}
    assert min(n["start_time"] for n in moved["tracks"]["drums"]["notes"]) == pytest.approx(3.3)  # the pickup
    # the MIDI files follow the notes: 3.2 beats at 480 PPQ, 1536 ticks later
    ticks = {}
    for name, plan in (("plain", plain), ("moved", moved)):
        for path in write_midi(plan, tmp_path / name):
            ons, now = [], 0
            for msg in mido.MidiFile(str(path)).tracks[0]:
                now += msg.time
                if msg.type == "note_on":
                    ons.append(now)
            ticks.setdefault(path.name, []).append(ons)
    assert set(ticks) == {"drums.mid", "808.mid", "other.mid"}
    for plain_ons, moved_ons in ticks.values():
        assert [t + 1536 for t in plain_ons] == moved_ons


def test_a_first_bar_line_before_0_s_lands_on_bar_1(tmp_path):
    """The first bar line 30 ms before the audio: everything moves 30 ms
    (0.06 beats) later, onto bar 1. The locators stay where they were: the
    plan puts a first bar line before 0 s at 0 s, on Live's bar lines
    already, 30 ms after the notes' bars; the notes and stems now meet
    them."""
    plain, moved = _plans(tmp_path, -0.03)
    assert plain["locators"] == [["A (1)", 0.0], ["B (2)", 4.0]]
    _moved_by(plain, moved, 0.06, 0.0)
    assert moved["length_beats"] == 12.0  # 8.06 beats end in bar 3
    assert moved["align_bar1"] == {"first_bar_line_s": -0.03, "moved_s": 0.03, "moved_beats": 0.06,
                                   "lands_on_bar": 1}
    assert _stem_starts(moved) == {0.06}


@pytest.mark.parametrize("phase", [0.015, 0.02, -0.02, 0.0])
def test_a_first_bar_line_within_20_ms_of_0_s_moves_nothing(tmp_path, phase):
    """Within 20 ms of 0 s (PIRAMIT's first bar line lies 0.5 ms before it)
    nothing moves: the plan is the one without the option, but for the
    record that it was asked."""
    plain, moved = _plans(tmp_path, phase)
    record = moved.pop("align_bar1")
    assert moved == plain
    assert record == {"first_bar_line_s": phase, "moved_s": 0.0, "moved_beats": 0.0, "lands_on_bar": 1}
    assert _stem_starts(moved) == {0.0}


def test_bar1_move_never_moves_earlier():
    from refdissect.live import bar1_move
    assert bar1_move(0.4, BAR_S) == pytest.approx(1.6)
    assert bar1_move(1.95, BAR_S) == pytest.approx(0.05)
    assert bar1_move(-0.03, BAR_S) == pytest.approx(0.03)
    assert bar1_move(-0.05, BAR_S) == pytest.approx(0.05)
    assert [bar1_move(p, BAR_S) for p in (0.02, -0.02, 0.0)] == [0.0, 0.0, 0.0]
    assert bar1_move(0.03, 1.5, within_ms=50) == 0.0  # the tolerance is a parameter


def test_the_blueprint_and_its_bars_count_on_lives_ruler(tmp_path):
    """The moved plan's blueprint holds the format as it is (the schema copy
    in tests/fixtures): its stems' clips start at 3.2 beats, its locators
    on Live's bar lines, its clips last the plan. Bars count on Live's
    ruler from 1, so bar 1 is the pickup and bars 2-3 start on the first
    bar line: ableton-live-mcp's blueprint.excerpt cuts the stems there
    (0.4 s in) and places them at 0, and bar 1's from 0 s to the line,
    placed at 3.2."""
    from test_blueprint import assert_valid

    from refdissect.blueprint import make_blueprint
    plain, moved = _plans(tmp_path, 0.4)
    bp = make_blueprint(moved, "Song", made="refdissect test", log=lambda _: None)
    assert_valid(bp)
    assert bp["length_beats"] == 12.0
    assert bp["locators"] == [{"name": "A (1)", "beat": 4.0}, {"name": "B (2)", "beat": 8.0}]
    refs = {t["name"]: t["audio_clips"] for t in bp["tracks"] if t.get("kind") == "audio"}
    assert set(refs) == {"REF drums", "REF bass", "REF other"}
    assert all(clips == [{"file": clips[0]["file"], "start": 3.2, "warping": False}] for clips in refs.values())
    assert {c["length"] for t in bp["tracks"] for c in t.get("clips") or []} == {12.0}
    plain_bp = make_blueprint(plain, "Song", made="refdissect test", log=lambda _: None)
    assert {c["start"] for t in plain_bp["tracks"] for c in t.get("audio_clips") or []} == {0.0}

    if os.environ.get("REFDISSECT_REQUIRE_LIVE"):  # CI: fail, not skip
        import MCP_Server.blueprint  # noqa: F401
    mcp_bp = pytest.importorskip("MCP_Server.blueprint",
                                 reason="ableton-live-mcp 2.4.0 or later (the [live] extra) not installed")
    assert mcp_bp.validate(bp) == []
    drums = sf.read(json.loads(json.dumps(moved))["stems"]["drums"]["path"], dtype="float64")[0]
    first = mcp_bp.excerpt(bp, (2, 3), None, tmp_path / "bars-2-3")
    assert first["meta"]["excerpt"] == {"bars": [2, 3], "beats": [4.0, 12.0], "seconds": [2.0, 6.0]}
    assert first["locators"] == [{"name": "A (1)", "beat": 0.0}, {"name": "B (2)", "beat": 4.0}]
    clip = next(t for t in first["tracks"] if t["name"] == "REF drums")["audio_clips"][0]
    assert clip["start"] == 0.0
    assert np.array_equal(sf.read(clip["file"], dtype="float64")[0], drums[int(0.4 * SR):])  # to the file's end
    notes = next(t for t in first["tracks"] if t["name"] == "Drums (library)")["clips"][0]["notes"]
    snare = next(n for n in notes if n["pitch"] == 38)
    assert snare["start_time"] == pytest.approx(1.0, abs=0.01)  # the line's beat 2, 0.5 s after it
    pickup = mcp_bp.excerpt(bp, (1, 1), None, tmp_path / "bar-1")
    clip = next(t for t in pickup["tracks"] if t["name"] == "REF drums")["audio_clips"][0]
    assert clip["start"] == 3.2
    assert np.array_equal(sf.read(clip["file"], dtype="float64")[0], drums[:int(0.4 * SR)])


def test_what_plays_the_plan_against_the_reference_reads_it_at_the_songs_seconds(tmp_path):
    """render.unaligned takes the move back: the moved plan's notes are the
    plain plan's, its render (render_plan) the same audio over the plain
    one's span, and its estimate (evaluate.estimate_from_plan, the suite's
    plan level) the same hits and notes. Live's grid moves back with it:
    its bar lines fall where the analysis put the song's, so the plan's
    first downbeat error against a truth whose first bar line is 0.4 s in
    is 0 ms, where the plain plan's is -400 ms."""
    from refdissect.evaluate import estimate_from_plan, rhythm_scores
    from refdissect.render import render_plan, unaligned
    plain, moved = _plans(tmp_path, 0.4)
    back = unaligned(moved)
    assert "align_bar1" not in back and "align_bar1" in moved  # a copy
    assert _times(back) | {"locators": None} == _times(plain) | {"locators": None}
    assert back["locators"] == [["A (1)", 0.8], ["B (2)", 4.8]]
    assert back["length_beats"] == pytest.approx(8.8)
    assert unaligned(plain) is plain
    a, b = render_plan(plain, tmp_path / "r-plain"), render_plan(moved, tmp_path / "r-moved")
    assert a["tracks"] == b["tracks"] and b["n_samples"] >= a["n_samples"]
    x, y = (sf.read(tmp_path / r / "drums.wav", dtype="float64")[0] for r in ("r-plain", "r-moved"))
    assert np.array_equal(y[:len(x)], x) and not np.any(y[len(x):])
    ea, eb = estimate_from_plan(plain), estimate_from_plan(moved)
    for key in ("drums", "bass808", "notes"):
        assert eb.get(key) == ea.get(key), key
    assert ea["time"]["downbeats_s"][0] == 0.0 and eb["time"]["downbeats_s"][0] == pytest.approx(0.4)
    truth = {"time": {"bpm": BPM, "beats_per_bar": 4, "beat_unit": 4, "downbeat_s": 0.4,
                      "beats_s": [0.4 + 0.5 * k for k in range(7)]}}
    assert rhythm_scores(truth, ea)["downbeat"]["first_error_ms"] == pytest.approx(-400.0)
    assert rhythm_scores(truth, eb)["downbeat"]["first_error_ms"] == pytest.approx(0.0, abs=0.01)
    # before 0 s: the notes come back as they were, the locators on the song's own bar lines
    plain, moved = _plans(tmp_path / "before", -0.03)
    back = unaligned(moved)
    assert _times(back) | {"locators": None} == _times(plain) | {"locators": None}
    assert back["locators"] == [["A (1)", 0.0], ["B (2)", 3.94]]


def test_the_command_and_suite_run_pass_it_on(tmp_path, monkeypatch, capsys):
    """plan --align-bar1 reaches pipeline.plan, and suite run hands it to
    every beat's plan step (cli._plan_kwargs); without it neither sees it.
    pipeline.plan says what moved."""
    from refdissect import pipeline, suite
    from refdissect.cli import _plan_kwargs

    passed = []
    monkeypatch.setattr(pipeline, "plan", lambda work, **k: passed.append(k))
    main(["plan", str(tmp_path), "--align-bar1"])
    main(["plan", str(tmp_path)])
    assert [k.get("align_bar1") for k in passed] == [True, False]
    options = []

    def run(*args, **kwargs):
        options.append(args[8])
        return {"beats": 0, "failed": [], "overall": {"analysis": {}, "plan": {}}}

    monkeypatch.setattr(suite, "run", run)
    base = ["suite", "run", str(tmp_path), "--library", "lib.npz", "--tag", "t"]
    main(base + ["--align-bar1"])
    main(base)
    assert options[0]["plan"] == {"align_bar1": True} and "plan" not in options[1]
    assert _plan_kwargs(type("A", (), {"chokes": True, "live_sr": None, "align_bar1": True})()) == \
        {"plan": {"chokes": True, "align_bar1": True}}
    monkeypatch.undo()
    logged = []
    for phase in (0.4, 0.01):
        work = _work(tmp_path / str(phase), phase)
        pipeline.plan(work, log=logged.append, align_bar1=True)
    assert logged[1] == ("  the first bar line, at 0.4 s, moved 1.6 s (3.2 beats) later, onto Live's bar 2: "
                         "notes, stems, locators and the end alike")
    assert logged[3] == "  the first bar line, at 0.01 s, lies within 20 ms of Live's bar 1: nothing moved"
