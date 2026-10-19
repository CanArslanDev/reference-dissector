"""The Basic Pitch worker and its runner, without the model (Basic Pitch's
note-creation steps faked); the real worker runs only with REFDISSECT_SLOW=1."""

import json
import os
import sys
import types

import numpy as np
import pytest
import soundfile as sf

from refdissect import basicpitch
from refdissect.audio import SR
from refdissect.basicpitch import block_times

WORKER_DIR = os.path.join(os.path.dirname(basicpitch.__file__), "workers")


def worker():
    sys.path.insert(0, WORKER_DIR)
    try:
        import basicpitch_worker
    finally:
        sys.path.remove(WORKER_DIR)
    return basicpitch_worker


def test_worker_times_and_lengths():
    w = worker()
    assert np.array_equal(w.block_times(1000), block_times(1000))
    assert block_times(143)[142] == pytest.approx(36164 / 22050)  # the second block starts one hop in
    assert w.min_note_frames(35.0) == 3 and w.min_note_frames(127.7) == 11  # predict()'s conversion


def test_block_times_against_basic_pitch_mapping():
    """Same mean slope as frame_times (no drift between them), but frame_times
    is up to ~18 ms off inside a song (8.5 ms steps every 142 frames, not
    10.3 ms every 172)."""
    n = 20000
    d = basicpitch.frame_times(n) - block_times(n)
    assert abs(d[-1]) < 0.012
    assert 0.010 < d.max() - d.min() < 0.020


def test_worker_decode_corrects_bends_and_uses_copies(monkeypatch):
    """decode() with Basic Pitch's two note-creation steps faked: bends come
    back one bin lower, times from the block mapping, the minimum frequency
    floored at 27.5 Hz, and the caller's posteriorgram untouched."""
    seen = {}

    def notes(frames, onsets, **kw):
        seen.update(kw)
        frames[:] = 0  # constrain_frequency zeroes bins in place
        return [(150, 160, 60, 0.5)]

    nc = types.SimpleNamespace(output_to_notes_polyphonic=notes,
                               get_pitch_bends=lambda contour, events: [(*e, [1] * 9 + [2]) for e in events])
    monkeypatch.setitem(sys.modules, "basic_pitch", types.SimpleNamespace(note_creation=nc))
    monkeypatch.setitem(sys.modules, "basic_pitch.note_creation", nc)
    w = worker()
    out = {"note": np.ones((200, 88), np.float32), "onset": np.full((200, 88), 0.7, np.float32),
           "contour": np.zeros((200, 264), np.float32)}
    got = w.decode(out, {"onset": 0.6, "frame": 0.3, "min_ms": 35.0, "melodia": False, "min_hz": 20.0})
    assert out["note"].min() == 1.0
    assert seen["min_freq"] == 27.5 and seen["min_note_len"] == 3 and seen["melodia_trick"] is False
    (n,) = got
    assert n["bend"] == [0] * 9 + [1] and n["midi"] == 60 and n["onset"] == pytest.approx(0.7)
    assert n["start_s"] == pytest.approx(round(float(block_times(151)[150]), 4))


def test_transcribe_runs_the_worker_in_its_own_environment(monkeypatch, tmp_path):
    cmd = basicpitch.command([tmp_path / "other.wav"], tmp_path / "bp", {"melodic": {"onset": 0.6}},
                             {"other": {"min_hz": 50.0}})
    assert cmd[cmd.index("--python") + 1] == "3.11" and "basic-pitch==0.4.0" in cmd and "setuptools<81" in cmd
    settings = json.loads(cmd[-2])
    assert cmd[-1].endswith("other.wav") and settings == {"presets": {"melodic": {"onset": 0.6}},
                                                          "files": {"other": {"min_hz": 50.0}}}
    w = worker()  # a file's own settings go on top of every preset
    assert w.presets_for(settings, "other") == {"melodic": {"onset": 0.6, "min_hz": 50.0}}
    assert w.presets_for(settings, "vocals") == {"melodic": {"onset": 0.6}}

    class Done:
        returncode, stdout, stderr = 0, 'Core ML says hello\n{"stems": {"other": {}}}\n', ""
    monkeypatch.setattr(basicpitch.subprocess, "run", lambda *a, **k: Done())
    assert basicpitch.transcribe([tmp_path / "other.wav"], tmp_path / "bp", {}) == {"stems": {"other": {}}}


@pytest.mark.skipif(not os.environ.get("REFDISSECT_SLOW"), reason="runs Basic Pitch (REFDISSECT_SLOW=1)")
def test_real_worker_on_tones(tmp_path):
    """Four plucked tones: every one found at its pitch within 30 ms."""
    melody = [(0.5, 60), (1.0, 64), (1.5, 67), (2.0, 72)]
    x = np.zeros(int(3.5 * SR), np.float32)
    t = np.arange(int(0.4 * SR)) / SR
    for start, midi in melody:
        f = 440 * 2 ** ((midi - 69) / 12)
        y = 0.4 * sum(np.sin(2 * np.pi * k * f * t) / k for k in range(1, 5)) * np.exp(-t / 0.25)
        x[int(start * SR):int(start * SR) + len(y)] += (y * np.minimum(1, t / 0.002)).astype(np.float32)
    sf.write(tmp_path / "other.wav", x, SR)
    settings = {"melodic": {**basicpitch.PRESETS["melodic"], "min_hz": None}}
    out = basicpitch.transcribe([tmp_path / "other.wav"], tmp_path / "bp", settings)
    notes = json.loads((tmp_path / "bp" / "other.json").read_text())["notes"]["melodic"]
    assert out["stems"]["other"]["notes_per_preset"]["melodic"] == len(notes)
    for start, midi in melody:
        assert any(n["midi"] == midi and abs(n["start_s"] - start) < 0.03 for n in notes), (start, midi)
    with np.load(tmp_path / "bp" / "other.npz") as z:
        assert z["note"].shape[1] == 88
