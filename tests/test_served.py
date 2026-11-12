"""served: beat_this's and Basic Pitch's workers kept running between calls,
with a fake worker (no torch, no model); REFDISSECT_SLOW=1 also checks the
real ones answer as a call does."""

import importlib.util
import io
import json
import os
import sys

import numpy as np
import pytest
import soundfile as sf

from refdissect import basicpitch, beats, served
from refdissect.audio import SR

# a worker: with --serve, one answer per line (after a line of noise), its pid in it; "fail" is
# answered with an error, "die" ends it
FAKE = """
import json, os, sys
MARK = "#refdissect-served# "
assert sys.argv[1:] == ["--serve"], sys.argv
for line in sys.stdin:
    argv = json.loads(line)["argv"]
    print("a library says hello", flush=True)
    if argv == ["die"]:
        print("dying", file=sys.stderr, flush=True)
        sys.exit(3)
    answer = {"error": "ValueError: bad"} if argv == ["fail"] else {"result": {"argv": argv, "pid": os.getpid()}}
    sys.stdout.write(MARK + json.dumps(answer) + "\\n")
    sys.stdout.flush()
"""


@pytest.fixture
def fake(tmp_path):
    path = tmp_path / "fake_worker.py"
    path.write_text(FAKE)
    yield [sys.executable, str(path)]
    served.keep(False)


def test_a_kept_worker_answers_every_call_and_is_let_go_when_it_fails(fake):
    first = served.ask(fake, {"argv": ["a.wav"]})
    assert first["argv"] == ["a.wav"] and served.ask(fake, {"argv": ["b.wav"]})["pid"] == first["pid"]
    with pytest.raises(RuntimeError, match="ValueError: bad"):
        served.ask(fake, {"argv": ["fail"]})
    again = served.ask(fake, {"argv": ["a.wav"]})
    assert again["pid"] != first["pid"]
    with pytest.raises(RuntimeError, match="(?s)stopped \\(3\\).*dying"):
        served.ask(fake, {"argv": ["die"]})
    assert served.ask(fake, {"argv": ["c.wav"]})["pid"] not in (first["pid"], again["pid"])
    served.close()
    assert not served._running


def test_keep_off_stops_the_workers(fake):
    served.keep(True)
    served.ask(fake, {"argv": ["a.wav"]})
    proc = next(iter(served._running.values())).proc
    served.keep(False)
    assert not served.kept() and not served._running and proc.wait(timeout=10) == 0


def test_the_trackers_call_a_kept_worker_with_their_own_arguments(tmp_path, monkeypatch):
    """With keep on, beats.track and basicpitch.transcribe ask the worker
    started by their command up to the script (with --serve) for the rest
    of it; off, they run the command (the tests of beats and basicpitch)."""
    asked = []
    monkeypatch.setattr(served, "ask", lambda command, request: asked.append((command, request)) or {"ok": 1})
    monkeypatch.setattr(served, "_keep", True)
    assert beats.track([tmp_path / "a.wav"], device="cpu") == {"ok": 1}
    cmd = beats.command([tmp_path / "a.wav"], device="cpu")
    assert asked[-1] == (cmd[:cmd.index(str(beats.WORKER)) + 1], {"argv": [str(tmp_path / "a.wav"), "--checkpoint",
                                                                            "final0", "--device", "cpu"]})
    assert basicpitch.transcribe([tmp_path / "o.wav"], tmp_path / "bp", {"melodic": {}}) == {"ok": 1}
    cmd = basicpitch.command([tmp_path / "o.wav"], tmp_path / "bp", {"melodic": {}})
    assert asked[-1][0] == cmd[:cmd.index(str(basicpitch.WORKER)) + 1] and asked[-1][0][-2] == "-I"
    assert asked[-1][1] == {"argv": cmd[len(asked[-1][0]):]} and asked[-1][1]["argv"][0] == str(tmp_path / "bp")


def _load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("worker, call", [(beats.WORKER, "track"), (basicpitch.WORKER, "run")])
def test_a_worker_serves_what_a_call_prints(worker, call, monkeypatch):
    """A worker's --serve loop: each request's argv through the call's own
    function, answered after MARK (served's); a failure is an error answer
    and the loop goes on."""
    w = _load(worker)
    assert w.MARK == served.MARK
    monkeypatch.setattr(w, call, lambda argv, model: {"argv": argv} if argv != ["x"] else 1 / 0)
    released = []
    if hasattr(w, "release"):  # beat_this's gives torch's GPU cache back after every call
        monkeypatch.setattr(w, "release", lambda: released.append(1))
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"argv": ["a"]}\n{"argv": ["x"]}\n{"argv": ["b", "c"]}\n'))
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    assert w.serve({}) == 0
    lines = out.getvalue().splitlines()
    assert all(line.startswith(served.MARK) for line in lines), lines
    answers = [json.loads(line[len(served.MARK):]) for line in lines]
    assert answers == [{"result": {"argv": ["a"]}}, {"error": "ZeroDivisionError: division by zero"},
                       {"result": {"argv": ["b", "c"]}}]
    assert released == ([1, 1, 1] if call == "track" else [])


@pytest.mark.skipif(os.environ.get("REFDISSECT_SLOW") != "1",
                    reason="runs beat_this and Basic Pitch (REFDISSECT_SLOW=1)")
def test_the_real_workers_kept_answer_as_a_call_does(kit, tmp_path):
    """Twice from a kept worker, the same beats and the same Basic Pitch
    files as from a call of each (their seconds aside)."""
    from conftest import drum_beat

    truth = drum_beat(kit, 137, 11)
    sf.write(tmp_path / "beat.wav", truth["audio"], SR)
    t = np.arange(int(3 * SR)) / SR
    sf.write(tmp_path / "other.wav", (0.4 * np.sin(2 * np.pi * 220 * t) * (t % 0.5 < 0.3)).astype(np.float32), SR)
    settings = {"melodic": {**basicpitch.PRESETS["melodic"], "min_hz": None}}

    def strip(out):
        return {k: v for k, v in out.items() if k != "seconds"}
    once = strip(beats.track([tmp_path / "beat.wav"]))
    basicpitch.transcribe([tmp_path / "other.wav"], tmp_path / "once", settings)
    served.keep(True)
    try:
        for k in range(2):
            assert strip(beats.track([tmp_path / "beat.wav"])) == once
            basicpitch.transcribe([tmp_path / "other.wav"], tmp_path / f"kept{k}", settings)
            kept, first = tmp_path / f"kept{k}", tmp_path / "once"
            assert (kept / "other.json").read_bytes() == (first / "other.json").read_bytes()
            with np.load(kept / "other.npz") as a, np.load(first / "other.npz") as b:  # zip dates aside
                assert set(a) == set(b) and all(a[n].tobytes() == b[n].tobytes() for n in a)
    finally:
        served.keep(False)


def test_a_suite_run_in_one_job_keeps_the_workers_while_it_runs(monkeypatch, tmp_path):
    """jobs=1: the beats run with served.keep(keep_workers), and the setting
    is put back after (the workers stopped); a pool's jobs get it through
    their initializer."""
    from refdissect import suite
    seen = []
    monkeypatch.setattr(suite, "beat_dirs", lambda *a: [tmp_path / "a", tmp_path / "b"])
    monkeypatch.setattr(suite, "separate_beats", lambda *a: 0)
    monkeypatch.setattr(suite, "_ready_stores", lambda *a: None)
    monkeypatch.setattr(suite, "_run_job", lambda args: seen.append(served.kept()) or {})
    monkeypatch.setattr(suite, "_done", lambda record, log: None)
    monkeypatch.setattr(suite, "aggregate", lambda root, tag: {})
    for keep in (True, False):
        seen.clear()
        suite.run(tmp_path, "lib.npz", tag="t", jobs=1, keep_workers=keep, log=lambda m: None)
        assert seen == [keep, keep] and not served.kept()
