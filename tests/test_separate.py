"""The separator's command line and how its output is read, with a stand-in
worker: no Demucs, torch or model download."""

import json
import sys

import pytest

from refdissect import separate as sep


def test_separator_command_and_its_output(tmp_path, monkeypatch):
    cmd = sep.command([("a.wav", "out")], "htdemucs_ft", None, 3, 2)
    assert cmd[-7:] == [str(sep.WORKER), "htdemucs_ft", "auto", "3", "2", "a.wav", "out"]
    lines = ["Downloading", json.dumps({"stems": {}, "seed": 3}), "mps failed (x); using the CPU",
             json.dumps({"stems": {}, "seed": 3})]
    script = "import sys\nfor line in sys.argv[1:]: print(line)"
    monkeypatch.setattr(sep, "command", lambda *a, **k: [sys.executable, "-c", script, *lines])
    seen = []
    out = sep.separate_many([("a", "b"), ("c", "d")], log=seen.append)
    assert [o["seed"] for o in out] == [3, 3] and len(seen) == 2
    fails = [sys.executable, "-c", "import sys; print('boom'); sys.exit(1)"]
    monkeypatch.setattr(sep, "command", lambda *a, **k: fails)
    with pytest.raises(RuntimeError, match="boom"):
        sep.separate("a.wav", tmp_path)
