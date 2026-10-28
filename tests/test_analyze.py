"""analyze() with the beat tracker off, failing and mocked. Without it the
grid, the hits and everything after them are what fit_grid gave before."""

import json
import subprocess

import numpy as np
import pytest
import soundfile as sf
from conftest import drum_beat, switch, tracked

from refdissect import beats, pipeline
from refdissect.audio import SR, read
from refdissect.detect import detect, discover
from refdissect.grid import fit_grid


@pytest.fixture(scope="module")
def stems(kit, tmp_path_factory):
    truth = drum_beat(kit, 137, 12)
    d = tmp_path_factory.mktemp("stems")
    sf.write(d / "drums.wav", truth["audio"], SR, subtype="FLOAT")
    sf.write(d / "bass.wav", np.zeros_like(truth["audio"]), SR, subtype="FLOAT")
    drums = read(d / "drums.wav")
    old, info = fit_grid(drums, (100.0, 180.0))
    return {"dir": d, "truth": truth, "old": old, "old_info": info, "old_events": detect(drums, discover(drums, old))}


def assert_as_before(a, stems):
    old = stems["old"]
    assert (a["grid"]["bpm"], a["grid"]["phase"], a["grid"]["beats_per_bar"]) == (old.bpm, old.phase, 4)
    assert a["grid"]["score"] == stems["old_info"]["score"] and a["grid"]["tracker"] is None
    assert a["events"] == stems["old_events"]


def test_without_the_tracker_analyze_is_as_before(stems, tmp_path, monkeypatch):
    monkeypatch.setattr(beats, "track", lambda *a, **kw: pytest.fail("the tracker ran"))
    assert_as_before(pipeline.analyze(stems["dir"], tmp_path, log=lambda *_: None, tracker=False), stems)


def test_a_failing_tracker_falls_back_to_before(stems, tmp_path, monkeypatch):
    monkeypatch.setattr(beats.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 1, "", "RuntimeError: no network"))
    logs = []
    assert_as_before(pipeline.analyze(stems["dir"], tmp_path, log=logs.append), stems)
    assert any("unavailable" in line for line in logs)


def test_tracker_beats_without_a_line_fall_back_to_before(stems, tmp_path, monkeypatch):
    monkeypatch.setattr(beats, "track", lambda paths, **kw: {"beats": [1.0, 2.0], "downbeats": [1.0]})
    logs = []
    assert_as_before(pipeline.analyze(stems["dir"], tmp_path, log=logs.append), stems)
    assert "  no line through the tracker's 2 beats: the grid is fit_grid's" in logs


def test_an_unsure_bar_line_is_logged(stems, tmp_path, monkeypatch):
    tracker = tracked(stems["truth"])
    tracker["downbeats"] = [d + 0.44 * (i % 3 == 1) for i, d in enumerate(tracker["downbeats"])]
    monkeypatch.setattr(beats, "track", lambda paths, **kw: tracker)
    logs = []
    a = pipeline.analyze(stems["dir"], tmp_path, log=logs.append)
    assert a["grid"]["downbeat"]["ambiguous"]
    assert any(line.startswith("  the first bar line is unsure (the tracker's bars disagree") for line in logs)


def test_a_tempo_change_is_logged(kit, tmp_path, monkeypatch):
    truth = switch(kit, 136, 160, 6)
    d = tmp_path / "stems"
    d.mkdir()
    sf.write(d / "drums.wav", truth["audio"], SR, subtype="FLOAT")
    sf.write(d / "bass.wav", np.zeros_like(truth["audio"]), SR, subtype="FLOAT")
    monkeypatch.setattr(beats, "track", lambda paths, **kw: tracked(truth))
    logs = []
    a = pipeline.analyze(d, tmp_path / "work", log=logs.append)
    assert a["grid"]["tempo_change"]["grid_follows"] == "first" and abs(a["grid"]["bpm"] - 136) < 0.05
    assert any(line.startswith("  the tempo changes by x1.17") and "follows the first part" in line for line in logs)


def test_with_the_tracker_phase_is_the_first_bar_line(stems, tmp_path, monkeypatch):
    truth = stems["truth"]
    monkeypatch.setattr(beats, "track", lambda paths, **kw: tracked(truth))
    a = pipeline.analyze(stems["dir"], tmp_path, log=lambda *_: None)
    grid = json.loads((tmp_path / "analysis.json").read_text(encoding="utf-8"))["grid"]
    assert abs(grid["phase"] - truth["downbeats"][0]) < 0.002
    assert (grid["beats_per_bar"], grid["beat_unit"]) == (4, 4) and grid["phase_correction_ms"] < -4
    assert grid["tracker"]["name"] == "beat_this" and len(grid["alternatives"]) == 5
    assert grid["drift_ms"] < 10 and grid["tempo_map"] is None
    assert set(grid["swing"]) == {"ratio_16", "ratio_8", "evidence"}
    # discover() groups onsets by their place in the beat, so a phase on the attacks finds the same hits
    assert {k: len(v) for k, v in a["events"].items()} == {k: len(v) for k, v in stems["old_events"].items()}


def test_the_808_reader_is_read_back_while_its_stems_and_code_are_the_same(tmp_path, monkeypatch):
    """pipeline._transcribe keeps bass808.transcribe's result by the two
    stems' samples and the reader's code, and reads it back with its log
    lines, through JSON either way; other stems, or no cache, read again."""
    from refdissect import pipeline

    calls = []

    def reader(bass, drums, log=None):
        calls.append(1)
        log("  808 template from drums onsets: 9 notes")
        return {"template": {"hz": np.float64(32.7), "root_midi": 24}, "notes": [{"t": np.float32(0.5), "pitch": 24}]}
    monkeypatch.setattr(pipeline.bass808, "transcribe", reader)
    bass, drums = np.ones(100, np.float32), np.zeros(100, np.float32)
    first, again = [], []
    out = pipeline._transcribe(bass, drums, tmp_path, first.append)
    assert pipeline._transcribe(bass, drums, tmp_path, again.append) == out and again == first and len(calls) == 1
    assert out["notes"][0]["t"] == 0.5 and isinstance(out["template"]["hz"], float)
    pipeline._transcribe(bass * 2, drums, tmp_path, lambda m: None)
    pipeline._transcribe(bass, drums, None, lambda m: None)
    assert len(calls) == 3
