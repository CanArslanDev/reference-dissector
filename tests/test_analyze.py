"""analyze() with the beat tracker off, failing and mocked. Without it the
grid, the hits and everything after them are what fit_grid gave before."""

import json
import subprocess

import numpy as np
import pytest
import soundfile as sf
from conftest import drum_beat, tracked

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
    # discover() still gets the grid on its onset-strength time base: the same hits
    assert {k: len(v) for k, v in a["events"].items()} == {k: len(v) for k, v in stems["old_events"].items()}
