"""The beat_this worker: its command, its output and its failures, mocked
(no torch, no download). REFDISSECT_SLOW=1 also runs the real worker."""

import importlib.util
import json
import os
import subprocess

import numpy as np
import pytest
import soundfile as sf
from conftest import drum_beat

from refdissect import beats
from refdissect.audio import SR


def test_command_runs_the_pinned_tracker_in_its_own_environment(tmp_path):
    cmd = beats.command([tmp_path / "drums.wav", tmp_path / "bass.wav"], device="cpu")
    assert cmd[1:6] == ["run", "--isolated", "--no-project", "--python", "3.12"]
    assert "beat-this==1.1.0" in cmd and str(beats.WORKER) in cmd
    assert cmd[cmd.index(str(beats.WORKER)) + 1:] == [str(tmp_path / "drums.wav"), str(tmp_path / "bass.wav"),
                                                       "--checkpoint", "final0", "--device", "cpu"]


def test_track_reads_the_last_line(monkeypatch):
    out = {"beats": [0.5, 1.0], "downbeats": [0.5], "name": "beat_this", "version": "1.1.0"}
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 0, "Downloading...\n" + json.dumps(out) + "\n", ""))
    assert beats.track(["a.wav"]) == out


def test_the_stems_are_summed_and_a_failure_is_none(tmp_path, monkeypatch):
    for name in ("vocals", "drums", "bass", "mix"):
        (tmp_path / f"{name}.wav").write_bytes(b"")
    seen, logs = [], []

    def failing(cmd, **kw):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 1, "", "Traceback ...\nRuntimeError: no network")
    monkeypatch.setattr(subprocess, "run", failing)
    assert beats.track_stems(tmp_path, logs.append) is None
    summed = [c for c in seen[0] if c.endswith(".wav")]
    assert summed == [str(tmp_path / f"{s}.wav") for s in ("drums", "bass", "vocals")]
    assert "unavailable" in logs[0] and "no network" in logs[0]
    monkeypatch.delenv("UV", raising=False)
    monkeypatch.setattr(beats.shutil, "which", lambda name: None)
    assert beats.track_stems(tmp_path, logs.append) is None and "uv is needed" in logs[-1]


def test_worker_sums_files_of_different_lengths(tmp_path):
    spec = importlib.util.spec_from_file_location("beats_worker", beats.WORKER)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    sf.write(tmp_path / "a.wav", np.ones((100, 2), np.float32) * 0.25, SR)
    sf.write(tmp_path / "b.wav", np.ones(50, np.float32) * 0.5, SR)
    total, rate = worker.mono_sum([str(tmp_path / "a.wav"), str(tmp_path / "b.wav")])
    assert rate == SR and len(total) == 100
    assert np.allclose(total[:50], 0.75) and np.allclose(total[50:], 0.25)


@pytest.mark.skipif(os.environ.get("REFDISSECT_SLOW") != "1", reason="runs beat_this (REFDISSECT_SLOW=1)")
def test_real_tracker_on_a_synthetic_beat(kit, tmp_path):
    """20 s at 137 BPM: the worker's beats and downbeats give the grid back."""
    from refdissect.grid import estimate

    truth = drum_beat(kit, 137, 11)
    sf.write(tmp_path / "beat.wav", truth["audio"], SR)
    out = beats.track([tmp_path / "beat.wav"])
    assert out["version"] == "1.1.0" and len(out["beats"]) >= 30 and out["downbeats"]
    grid, info = estimate(truth["audio"], out, (100.0, 180.0))
    assert abs(grid.bpm - 137) < 0.05
    assert abs(grid.phase - truth["downbeats"][0]) < 0.003
