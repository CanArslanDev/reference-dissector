"""The suite commands on a tiny synthetic suite: a fake inventory, 8-bar
beats (4 bars find no drum templates), a stub separator (the truth's group
stems handed back as Demucs' stems) and a stub Basic Pitch (the truth's
notes), so no sample library, Demucs or model download is needed; and the
export of the means, which must name no file."""

import functools
import json
import runpy
import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import butter, sosfilt

from refdissect import basicpitch, pipeline, suite, synth808
from refdissect.audio import SR, read
from refdissect.basicpitch import block_times
from refdissect.cli import main
from refdissect.library import Library, inventory
from refdissect.synth808 import Params, voice

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="the scenario beats need ffmpeg")
QUICK_FIT = {"maxiter": 2, "popsize": 4}  # the 808 fit's search, cut from 150 x 20 (about 50 s a beat) for tests


def _burst(seed, ms, lo, hi, decay_ms, tone_hz=0.0):
    rng = np.random.default_rng(seed)
    n = int(ms * SR / 1000)
    t = np.arange(n) / SR
    x = sosfilt(butter(4, [lo, hi], btype="band", fs=SR, output="sos"), rng.standard_normal(n))
    if tone_hz:
        x = x * 0.3 + np.sin(2 * np.pi * tone_hz * t) * np.max(np.abs(x))
    x *= np.exp(-t / (decay_ms / 1000))
    return (x / np.max(np.abs(x))).astype(np.float32)


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    """A fake inventory (the roles a v1 kit draws) and its two indexes."""
    folder = tmp_path_factory.mktemp("library")
    t = np.arange(9 * SR) / SR
    files = {"kicks/kick a.wav": _burst(1, 300, 30, 2000, 120, tone_hz=55.0),
             "snares/snare a.wav": _burst(2, 250, 200, 9000, 60),
             "claps/clap a.wav": _burst(7, 200, 600, 6000, 40),
             "hats/hat a.wav": _burst(3, 80, 6000, 16000, 15),
             "open hats/open hat a.wav": _burst(4, 600, 5000, 16000, 200),
             "808s/808 a.wav": voice(Params(41.2, 46.0, 15, 1.2, 2, 1.5, 0.05, 0.3), 1.6),
             "808s/808 b.wav": voice(Params(36.7, 44.0, 20, 1.0, 2, 1.2, 0.05, 0.3), 1.4),
             "loops/keys loop.wav": 0.3 * np.sin(2 * np.pi * 220 * t)[: int(8.5 * SR)],
             "vocals/vocal take.wav": 0.3 * np.sin(2 * np.pi * 180 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))}
    for rel, x in files.items():
        (folder / rel).parent.mkdir(parents=True, exist_ok=True)
        sf.write(folder / rel, x, SR, subtype="PCM_24")
    rows = inventory([folder])
    out = tmp_path_factory.mktemp("work")
    (out / "inventory.json").write_text(json.dumps(rows))
    Library.build(rows).save(out / "library.npz")
    Library.build(rows, seconds=1.0, max_duration=6.0, roles={"808"}).save(out / "library_808.npz")
    return out


def stub_separator(calls):
    """separate_many as a perfect separator: each beat's truth groups as its stems."""
    def separate_many(jobs, model, device=None, seed=0, shifts=1, log=None):
        out = []
        for mix, stems in jobs:
            calls.append(stems.parent.name)
            stems.mkdir(parents=True, exist_ok=True)
            for group in ("drums", "bass", "other", "vocals"):
                x = read(mix.parent / "stems_truth" / f"{group}.flac")
                sf.write(stems / f"{group}.wav", np.stack([x, x], 1), SR, subtype="PCM_24")
            info = {"stems": {}, "model": model, "seed": seed, "shifts": shifts, "device": "stub",
                    "versions": {}, "audio": str(mix), "audio_sha256": suite._sha256(mix), "seconds": 0.0}
            (stems / "separation.json").write_text(json.dumps(info))
            out.append(info)
            if log:
                log(info)
        return out
    return separate_many


def truth_basic_pitch(audio, out_dir, presets, files=None):
    """basicpitch.transcribe as a perfect transcriber of the other stem (the
    beat's truth notes, and a posteriorgram where they sound) that also hears
    two notes in the vocals, which the truth has none for."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"stems": {}, "model": "basic-pitch 0.4.0", "backend": "FAKE", "python": "3.11",
              "settings": {"presets": presets, "files": files or {}}}
    for path in map(Path, audio):
        truth = json.loads((path.parents[1] / "events.json").read_text())["notes"] if path.stem == "other" else [
            {"t": 1.0, "end": 1.5, "midi": 64}, {"t": 2.0, "end": 2.5, "midi": 67}]
        notes = [{"start_s": n["t"], "end_s": n["end"], "start_frame": 0, "end_frame": 1, "midi": n["midi"],
                  "amplitude": 0.5, "onset": 0.9, "bend": [0]} for n in truth]
        t = block_times(int(sf.info(path).duration * 86.2) + 1)
        pg = np.zeros((len(t), 88), np.float16)
        for n in truth:
            pg[(t >= n["t"]) & (t < n["end"]), n["midi"] - 21] = 0.9
        (out_dir / f"{path.stem}.json").write_text(json.dumps({"stem": path.stem, "notes": {"melodic": notes,
                                                                                          "chord": notes}}))
        np.savez_compressed(out_dir / f"{path.stem}.npz", note=pg)
        result["stems"][path.stem] = {"notes": str(out_dir / f"{path.stem}.json"),
                                      "posteriorgram": str(out_dir / f"{path.stem}.npz")}
    return result


@needs_ffmpeg
def test_build_run_and_report(library, tmp_path, monkeypatch, capsys):
    root = tmp_path / "suite"
    args = ["suite", "build", str(library / "inventory.json"), "--out", str(root), "--scenarios",
            "v1,meter68,producer", "--seeds", "1", "--bars", "8", "--midi-dir", str(tmp_path / "no midi")]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "producer: no single-role hat pattern" in out and "2 beats" in out
    built = (root / "v1-1" / "events.json").stat().st_mtime_ns
    assert main(args) == 0 and (root / "v1-1" / "events.json").stat().st_mtime_ns == built  # kept, not rebuilt
    capsys.readouterr()
    assert [b.name for b in suite.beat_dirs(root)] == ["v1-1", "meter68-1"]
    assert [b.name for b in suite.beat_dirs(root, ["meter68"])] == ["meter68-1"]

    calls = []
    monkeypatch.setattr(suite, "separate_many", stub_separator(calls))
    monkeypatch.setattr(pipeline, "fit", functools.partial(synth808.fit, **QUICK_FIT))
    monkeypatch.setattr(basicpitch, "transcribe", truth_basic_pitch)
    lib = ["--library", str(library / "library.npz"), "--library-808", str(library / "library_808.npz")]
    # no beat tracker here (it would download beat_this): the grid is fit_grid's
    assert main(["suite", "run", str(root), *lib, "--tag", "a", "--jobs", "1", "--render", "--no-tracker"]) == 0
    assert sorted(calls) == ["meter68-1", "v1-1"]
    assert "2 beats scored under a (0 failed)" in capsys.readouterr().out
    rec = json.loads((root / "v1-1" / "runs" / "a" / "scores.json").read_text())
    assert rec["separation"]["seed"] == 0 and rec["separation"]["shifts"] == 1
    assert rec["options"] == {"analyze": {"bpm_range": [100.0, 180.0], "bpm": None, "meter": None, "tracker": False},
                              "match": {"detune": None, "kick_sources_on": False}}
    assert rec["analysis"]["estimate_from"] == "analysis" and rec["plan"]["estimate_from"] == "plan"
    assert rec["plan"]["levels"]["sounds"] and not (root / "v1-1" / "runs" / "a" / "render").exists()
    for step in ("analyze", "melodic", "match", "fit_808", "plan", "score_analysis", "render", "score_plan", "total"):
        assert rec["seconds"][step] >= 0, step
    assert set(rec["steps"].values()) == {"run"} and rec["reuse"] is None
    assert rec["plan_size"]["melodic_notes"] == {"other": 0, "vocals": 2}  # v1 plays no melodic notes
    # the stub's stems are the truth's groups, so a perfect kit would sit at the truth's level
    assert {g: v["kit_db"] for g, v in rec["plan"]["levels"]["kit"].items()} == {"drums": 0.0, "bass": 0.0}
    agg = json.loads((root / "scores-a.json").read_text())
    assert list(agg["scenarios"]) == ["v1", "meter68"] and agg["beats"] == 2 and agg["failed"] == []
    assert agg["scenarios"]["v1"]["plan"]["levels.kit.drums_db"] == 0.0
    assert len(agg["options"]) == 1 and len(agg["refdissect"]) == 1 and agg["refdissect"][0]["version"]
    for part in ("analysis", "plan"):
        assert 0 <= agg["scenarios"]["v1"][part]["strict_f1_20"] <= 1
    assert agg["scenarios"]["v1"]["plan"]["levels.balance_abs_db"] >= 0

    # the stems are kept per model, and two processes give the numbers one gives (the options travel
    # to them; the patched fit and Basic Pitch do not: melodic is left out, and these beats have no notes)
    options = {"analyze": {"bpm_range": (100.0, 180.0), "tracker": False}, "match": {"detune": None},
               "fit_808": QUICK_FIT}
    b = suite.run(root, library / "library.npz", library / "library_808.npz", "b", jobs=2, options=options,
                  steps=["analyze", "match", "fit-808", "plan", "score"])
    assert json.loads((root / "v1-1" / "runs" / "b" / "scores.json").read_text())["steps"]["melodic"] == "skipped"
    assert len(calls) == 2 and b["last_run"] == {"beats": 2, "jobs": 2, "wall_s": b["last_run"]["wall_s"]}
    for name in ("v1", "meter68"):
        assert b["scenarios"][name]["analysis"] == agg["scenarios"][name]["analysis"]
        assert b["scenarios"][name]["plan"]["strict_f1_20"] == agg["scenarios"][name]["plan"]["strict_f1_20"]
    assert "levels.balance_abs_db" not in b["scenarios"]["v1"]["plan"]  # no render, no levels

    assert main(["suite", "report", str(root), "--tags", "a,b", "--out", str(tmp_path / "r.md")]) == 0
    text = (tmp_path / "r.md").read_text()
    assert "| scenario | a | b | Δ |" in text and "| v1 |" in text and "| **overall** |" in text
    assert "+0.000" in text  # the same numbers under both tags


def test_steps_need_their_inputs_or_a_reuse():
    assert suite.check_steps() == suite.STEPS
    assert suite.check_steps(["score", "analyze", "plan"]) == ("analyze", "plan", "score")  # run in their order
    with pytest.raises(ValueError, match="need the outputs of plan"):
        suite.check_steps(["analyze", "score"])
    with pytest.raises(ValueError, match="need the outputs of analyze"):
        suite.check_steps(["melodic"])
    with pytest.raises(ValueError, match="Unknown steps: fit_808"):
        suite.check_steps(["fit_808"])
    assert suite.check_steps(["score"], reuse="base") == ("score",)


@needs_ffmpeg
def test_melodic_scores_and_steps_reused(library, tmp_path, monkeypatch):
    """The melodic scenario with a perfect Basic Pitch: notes and chords
    scored at analysis level, the plan's notes at plan level, the vocals'
    notes extra. A run that reuses everything but plan and score scores the
    same; fit-808 found no 808 that rings alone here, an answer the reuse
    carries. A reused run must have every output, which is checked before
    anything runs; in place, the kept outputs stay."""
    root = tmp_path / "suite"
    main(["suite", "build", str(library / "inventory.json"), "--out", str(root), "--scenarios", "melodic",
          "--seeds", "1", "--bars", "8"])
    monkeypatch.setattr(suite, "separate_many", stub_separator([]))
    monkeypatch.setattr(basicpitch, "transcribe", truth_basic_pitch)
    lib, lib808 = library / "library.npz", library / "library_808.npz"
    options = {"analyze": {"tracker": False}, "fit_808": QUICK_FIT}
    suite.run(root, lib, lib808, "full", jobs=1, options=options)
    beat = root / "melodic-1"
    rec = json.loads((beat / "runs" / "full" / "scores.json").read_text())
    m = rec["analysis"]["melodic"]
    assert {v["estimated_layer"] for v in m["layers"].values()} == {"other"} and m["extra"][0]["layer"] == "vocals"
    assert m["pooled"]["note_f1"] > 0.9 and rec["plan"]["melodic"]["pooled"]["note_f1"] > 0.9
    assert 0 <= rec["analysis"]["chords"]["majmin"] <= 1 and rec["plan"]["chords"] is None
    assert "fit_808" in rec["problems"] and not rec["plan_size"]["808_one_shot"]

    re = suite.run(root, lib, lib808, "re", jobs=1, options=options, steps=["plan", "score"], reuse="full")
    work = beat / "runs" / "re"
    again = json.loads((work / "scores.json").read_text())
    assert again["steps"] == {"analyze": "reused", "melodic": "reused", "match": "reused", "fit-808": "reused",
                              "plan": "run", "score": "run"} and again["reuse"] == "full"
    assert "analyze" not in again["seconds"] and "fit_808" not in again["seconds"]
    assert again["analysis"] == rec["analysis"] and again["plan"] == rec["plan"]
    assert again["problems"] == rec["problems"] and re["reuse"] == ["full"]
    assert "melodic, match, fit-808 reused from full" in suite.report(root, ["full", "re"])

    # in place: the kept outputs stay, the record is rewritten
    kept = (work / "analysis.json").stat().st_mtime_ns
    suite.run(root, lib, lib808, "re", jobs=1, steps=["score"], reuse="re")
    assert (work / "analysis.json").stat().st_mtime_ns == kept
    assert json.loads((work / "scores.json").read_text())["analysis"] == rec["analysis"]
    assert (work / "log.txt").read_text().count("steps: ") == 2

    # what is missing is named before anything runs; another model's run is refused
    (work / "melodic.json").unlink()
    with pytest.raises(ValueError, match=r"melodic-1: melodic.json \(melodic\)"):
        suite.run(root, lib, lib808, "x", jobs=1, steps=["score"], reuse="re")
    assert not (beat / "runs" / "x").exists()
    assert "analysed stems-htdemucs" in suite.reuse_problems(beat, "full", "htdemucs_ft", ("score",))[0]
    (beat / "runs" / "full" / "scores.json").write_text(json.dumps({"failed": "RuntimeError: no grid"}))
    assert suite.reuse_problems(beat, "full", "htdemucs", ("score",)) == ["runs/full failed (RuntimeError: no grid)"]


def test_copied_outputs_point_into_their_new_folder(tmp_path):
    """The one-shot fit-808 writes is named by path in synth808.json and
    plan.json: copied, both name the copy; a folder whose name starts like
    the source's is left alone, and analyze's templates go without 808.npy."""
    src, dst = tmp_path / "runs" / "base", tmp_path / "runs" / "new"
    (src / "templates").mkdir(parents=True)
    dst.mkdir()
    for name in ("sound_1.npy", "808.npy"):
        (src / "templates" / name).write_bytes(b"x")
    (src / suite.ONE_SHOT).write_bytes(b"x")
    other = str(tmp_path / "runs" / "base-2" / suite.ONE_SHOT)
    (src / "synth808.json").write_text(json.dumps({"file": str(src / suite.ONE_SHOT), "other": other}))
    (src / "plan.json").write_text(json.dumps({"tracks": {"808": {"sample_path": str(src / suite.ONE_SHOT)}}}))
    for rel in ("templates", "synth808.json", "plan.json"):
        suite._copy(src, dst, rel)
    synth = json.loads((dst / "synth808.json").read_text())
    assert synth == {"file": str(dst / suite.ONE_SHOT), "other": other}
    assert json.loads((dst / "plan.json").read_text())["tracks"]["808"]["sample_path"] == synth["file"]
    assert sorted(p.name for p in (dst / "templates").iterdir()) == ["sound_1.npy"]


@needs_ffmpeg
def test_a_failed_beat_is_recorded_and_resume_skips_done_beats(library, tmp_path, monkeypatch):
    root = tmp_path / "suite"
    main(["suite", "build", str(library / "inventory.json"), "--out", str(root), "--scenarios", "v1",
          "--seeds", "1,2", "--bars", "4"])
    monkeypatch.setattr(suite, "separate_many", stub_separator([]))

    def broken(stems, work, **kw):
        raise RuntimeError("no grid")
    monkeypatch.setattr(suite.pipeline, "analyze", broken)
    s = suite.run(root, library / "library.npz", tag="x", jobs=1)
    assert s["beats"] == 0 and s["failed"] == ["v1-1", "v1-2"]
    assert "no grid" in json.loads((root / "v1-1" / "runs" / "x" / "scores.json").read_text())["failed"]
    s = suite.run(root, library / "library.npz", tag="x", jobs=1, resume=True)
    assert s["last_run"]["beats"] == 0  # both have a scores.json, failed or not


def test_the_export_keeps_means_per_scenario_only(tmp_path):
    ns = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "export_results.py"), run_name="export")
    ns["suite_scores"].__globals__["ROOT"] = tmp_path
    agg = {"schema": "refdissect.suite/1", "tag": "t", "model": "htdemucs", "options": [{}], "render": [True],
           "refdissect": [{"version": "0.1.0"}], "beats": 2, "last_run": {"beats": 2},
           "separation": {"model": "htdemucs", "seed": 0, "shifts": 1, "devices": ["mps"], "versions": {},
                          "audio": "/Users/someone/beat/mix_mp3.wav"},
           "failed": ["v1-3"], "overall": {"scenarios": 1, "analysis": {"strict_f1_20": 0.5}},
           "scenarios": {"v1": {"beats": 2, "seeds": [1, 2], "analysis": {"strict_f1_20": 0.5}, "plan": {},
                                "seconds": {}, "records": [{"beat": "v1-1", "path": "/Users/someone/x.wav"}]}}}
    (tmp_path / "scores-t.json").write_text(json.dumps(agg))
    out = ns["suite_scores"](tmp_path, "t")
    text = (tmp_path / "results" / "calibration" / "suite-t.json").read_text()
    assert out["failed"] == 1 and "/Users/" not in text and ".wav" not in text and "v1-3" not in text
    assert json.loads(text)["scenarios"]["v1"]["analysis"] == {"strict_f1_20": 0.5}
    # an option a later step adds may hold a path or a file name: only numbers, booleans, null and short
    # identifiers go out
    for bad in ("/Users/someone/model.npz", "/Volumes/Samples/Some Kit/", "~/Music/Reference Dissect/work",
                "/private/tmp/x", "C:\\Users\\me", "some kit snare 1", "kick.m4a", "set.als", "rack.adg", "take.ogg"):
        agg["options"] = [{"analyze": {"templates": bad}}]
        (tmp_path / "scores-t.json").write_text(json.dumps(agg))
        with pytest.raises(ValueError, match="short identifier|local file"):
            ns["suite_scores"](tmp_path, "t")
    agg["options"] = [{"analyze": {"bpm_range": [100.0, 180.0], "model": "htdemucs_ft"}, "match": {"detune": None}}]
    agg["steps"] = [dict.fromkeys(suite.STEPS, "run") | {"fit-808": "reused", "melodic": "no_tonal_stem"}]
    agg["reuse"] = ["base-htdemucs"]
    (tmp_path / "scores-t.json").write_text(json.dumps(agg))
    out = ns["suite_scores"](tmp_path, "t")
    assert (out["options"], out["steps"], out["reuse"]) == (agg["options"], agg["steps"], agg["reuse"])
    for bad in ("/Volumes/Samples/kick", "~/Music/x", "/Applications/Some App.app/x", "a.m4a"):
        assert ns["LOCAL"].search(json.dumps({"x": bad})), bad
