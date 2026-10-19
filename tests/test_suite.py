"""The suite commands on a tiny synthetic suite: a fake inventory, 4-bar
beats and a stub separator (the truth's group stems handed back as Demucs'
stems), so no sample library, Demucs or model download is needed."""

import functools
import json
import shutil

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import butter, sosfilt

from refdissect import pipeline, suite, synth808
from refdissect.audio import SR, read
from refdissect.cli import main
from refdissect.library import Library, inventory
from refdissect.synth808 import Params, voice

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="the scenario beats need ffmpeg")
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
    lib = ["--library", str(library / "library.npz"), "--library-808", str(library / "library_808.npz")]
    assert main(["suite", "run", str(root), *lib, "--tag", "a", "--jobs", "1", "--render"]) == 0
    assert sorted(calls) == ["meter68-1", "v1-1"]
    assert "2 beats scored under a (0 failed)" in capsys.readouterr().out
    rec = json.loads((root / "v1-1" / "runs" / "a" / "scores.json").read_text())
    assert rec["separation"]["seed"] == 0 and rec["separation"]["shifts"] == 1
    assert rec["options"] == {"analyze": {"bpm_range": [100.0, 180.0]}, "match": {"detune": None}}
    assert rec["analysis"]["estimate_from"] == "analysis" and rec["plan"]["estimate_from"] == "plan"
    assert rec["plan"]["levels"]["sounds"] and not (root / "v1-1" / "runs" / "a" / "render").exists()
    for step in ("analyze", "match", "fit_808", "plan", "score_analysis", "render", "score_plan", "total"):
        assert rec["seconds"][step] >= 0, step
    agg = json.loads((root / "scores-a.json").read_text())
    assert list(agg["scenarios"]) == ["v1", "meter68"] and agg["beats"] == 2 and agg["failed"] == []
    assert len(agg["options"]) == 1 and len(agg["refdissect"]) == 1 and agg["refdissect"][0]["version"]
    for part in ("analysis", "plan"):
        assert 0 <= agg["scenarios"]["v1"][part]["strict_f1_20"] <= 1
    assert agg["scenarios"]["v1"]["plan"]["levels.balance_abs_db"] >= 0

    # the stems are kept per model, and two processes give the numbers one gives (the options travel
    # to them; the patched fit does not)
    options = {"analyze": {"bpm_range": (100.0, 180.0)}, "match": {"detune": None}, "fit_808": QUICK_FIT}
    b = suite.run(root, library / "library.npz", library / "library_808.npz", "b", jobs=2, options=options)
    assert len(calls) == 2 and b["last_run"] == {"beats": 2, "jobs": 2, "wall_s": b["last_run"]["wall_s"]}
    for name in ("v1", "meter68"):
        assert b["scenarios"][name]["analysis"] == agg["scenarios"][name]["analysis"]
        assert b["scenarios"][name]["plan"]["strict_f1_20"] == agg["scenarios"][name]["plan"]["strict_f1_20"]
    assert "levels.balance_abs_db" not in b["scenarios"]["v1"]["plan"]  # no render, no levels

    assert main(["suite", "report", str(root), "--tags", "a,b", "--out", str(tmp_path / "r.md")]) == 0
    text = (tmp_path / "r.md").read_text()
    assert "| scenario | a | b | Δ |" in text and "| v1 |" in text and "| **overall** |" in text
    assert "+0.000" in text  # the same numbers under both tags


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
